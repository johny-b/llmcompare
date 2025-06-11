import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from collections import defaultdict

from llmcompare.runner.config import RunnerConfig, default_get_config
from llmcompare.runner.client import get_client
from llmcompare.runner.chat_completion import openai_chat_completion

NO_LOGPROBS_WARNING = """\
Failed to get logprobs because {model} didn't send them.
Returning empty dict, I hope you can handle it.

Last completion has empty logprobs.content: 
{completion}
"""

class Runner:
    config_for_model = default_get_config

    def __init__(self, model: str, config: RunnerConfig | None = None):
        self.model = model
        self.config = config or Runner.config_for_model(model)
        self._client = None
        self._get_client_lock = Lock()

    @property
    def client(self):
        if self._client is None:
            with self._get_client_lock:
                if self._client is None:
                    self._client = get_client(self.model)
        return self._client

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None) -> str:
        """Just a simple text request. Might get more arguments later."""
        args = {
            "client": self.client,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.config.timeout,
        } 
        if max_tokens is not None:
            # Sending max_tokens is not supported for o3.
            args["max_tokens"] = max_tokens

        completion = openai_chat_completion(**args)
        try:
            return completion.choices[0].message.content
        except Exception as e:
            print(completion)
            raise
    
    def single_token_probs(self, messages: list[dict], top_logprobs: int = 20, num_samples: int = 1) -> dict:
        probs = {}
        for _ in range(num_samples):
            new_probs = self.single_token_probs_one_sample(messages, top_logprobs)
            for key, value in new_probs.items():
                probs[key] = probs.get(key, 0) + value
        return {key: value / num_samples for key, value in probs.items()}

    def single_token_probs_one_sample(self, messages: list[dict], top_logprobs: int = 20) -> dict:
        """Returns probabilities of the next token. Always samples 1 token."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
            timeout=self.config.timeout,
        )
        
        if completion.choices[0].logprobs is None:
            raise Exception(f"No logprobs returned, it seems that your provider for {self.model} doesn't support that.")
        
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            print(NO_LOGPROBS_WARNING.format(model=self.model, completion=completion))
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = math.exp(el.logprob)
        return result
    
    def get_many(self, func, kwargs_list, *, max_workers=None, silent=False, title=None, executor=None):
        """Call FUNC with arguments from KWARGS_LIST in MAX_WORKERS parallel threads.

        FUNC is get_text or single_token_probs. Examples:
        
            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7},
            ]
            for in_, out in runner.get_many(runner.get_text, kwargs_list):
                print(in_, "->", out)

        or

            kwargs_list = [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "Bye"}]},
            ]
            for in_, out in runner.get_many(runner.single_token_probs, kwargs_list):
                print(in_, "->", out)

        (FUNC that is a different callable should also work)

        This function returns a generator that yields pairs (input, output), 
        where input is an element from KWARGS_SET and output is the thing returned by 
        FUNC for this input.

        Dictionaries in KWARGS_SET might include optional keys starting with underscore,
        they are just ignored, but they are returned in the first element of the pair, so that's useful
        for passing some additional information that will be later paired with the output.

        Other parameters:
        - MAX_WORKERS: number of parallel threads, overrides self.config.max_workers.
        - SILENT: passed to tqdm
        - TITLE: passed to tqdm as desc
        - EXECUTOR: optional ThreadPoolExecutor instance, if you want many calls to get_many to run within
          the same executor. MAX_WORKERS and self.config.max_workers are then ignored.
        """
        if max_workers is None:
            max_workers = self.config.max_workers

        if executor is None:
            executor = ThreadPoolExecutor(max_workers)

        def get_data(kwargs):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            try:
                result = func(**func_kwargs)
            except Exception as e:
                print(f"There was an error. Returning None. {e}")
                result = None
            return kwargs, result

        futures = [executor.submit(get_data, kwargs) for kwargs in kwargs_list]

        try:
            for future in tqdm(as_completed(futures), total=len(futures), disable=silent, desc=title):
                yield future.result()
        except (Exception, KeyboardInterrupt):
            for fut in futures:
                fut.cancel()
            raise
        finally:
            executor.shutdown(wait=False)

    def sample_probs(self, messages: list[dict], *, num_samples: int, max_tokens: int, temperature: float = 1) -> dict:
        """Sample answers NUM_SAMPLES times. Returns probabilities of answers.
        
        Works only if the API supports `n` parameter.

        Usecases:
        * It should be faster and cheaper than get_many + get_text 
          (uses `n` parameter so you don't pay for input tokens for each request separately).
        * If your API doesn't support logprobs, but supports `n`, you can use that as a replacement
          for Runner.single_token_probs.
        """
        cnts = defaultdict(int)
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            completion = openai_chat_completion(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                timeout=self.config.timeout,
            )
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        if sum(cnts.values()) != num_samples:
            raise Exception(f"Something weird happened. Expected {num_samples} samples, got {sum(cnts.values())}. Maybe n parameter is ignored for {self.model}?")
        return {key: val / num_samples for key, val in cnts.items()}