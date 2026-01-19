import math
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from llmcomp.config import Config, NoClientForModel
from llmcomp.runner.chat_completion import openai_chat_completion
from llmcomp.runner.model_adapter import ModelAdapter


class DuplicateTokenError(Exception):
    """Raised when API returns duplicate tokens in logprobs (unexpected provider behavior)."""

    pass


NO_LOGPROBS_WARNING = """\
Failed to get logprobs because {model} didn't send them.
Returning empty dict, I hope you can handle it.

Last completion has empty logprobs.content: 
{completion}
"""


class Runner:
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self._get_client_lock = Lock()

    @property
    def client(self):
        if self._client is None:
            with self._get_client_lock:
                if self._client is None:
                    self._client = Config.client_for_model(self.model)
        return self._client

    def _prepare_for_model(self, params: dict) -> dict:
        """Prepare params for the API call via ModelAdapter.
        
        Also adds timeout from Config. Timeout is added here (not in ModelAdapter)
        because it doesn't affect API response content and shouldn't be part of the cache hash.
        
        Note: timeout is set first so that ModelAdapter handlers can override it if needed.
        """
        prepared = ModelAdapter.prepare(params, self.model)
        return {"timeout": Config.timeout, **prepared}

    def get_text(self, params: dict) -> tuple[str, dict]:
        """Get a text completion from the model.

        Args:
            params: Dictionary of parameters for the API.
                Must include 'messages'. Other common keys: 'temperature', 'max_tokens'.

        Returns:
            Tuple of (content, prepared_kwargs) where prepared_kwargs is what was sent to the API.
        """
        prepared = self._prepare_for_model(params)
        completion = openai_chat_completion(client=self.client, **prepared)
        try:
            content = completion.choices[0].message.content
            if content is None:
                # So far all cases here were OpenAI refusals, e.g.
                # ChatCompletion(
                #     id='chatcmpl-...',
                #     choices=[Choice(
                #         finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(
                #             content=None, 
                #             refusal="I'm sorry, I'm unable to fulfill that request.",
                #             ...))])
                warnings.warn(f"API sent None as content. Returning empty string.\n{completion}", stacklevel=2)
                return "", prepared
            return content, prepared
        except Exception:
            warnings.warn(f"Unexpected error.\n{completion}")
            raise

    def single_token_probs(
        self,
        params: dict,
        *,
        num_samples: int = 1,
        convert_to_probs: bool = True,
    ) -> tuple[dict, dict]:
        """Get probability distribution of the next token, optionally averaged over multiple samples.

        Args:
            params: Dictionary of parameters for the API.
                Must include 'messages'. Other common keys: 'top_logprobs', 'logit_bias'.
            num_samples: Number of samples to average over. Default: 1.
            convert_to_probs: If True, convert logprobs to probabilities. Default: True.

        Returns:
            Tuple of (probs_dict, prepared_kwargs) where prepared_kwargs is what was sent to the API.
        """
        probs = {}
        prepared = None
        for _ in range(num_samples):
            new_probs, prepared = self.single_token_probs_one_sample(params, convert_to_probs=convert_to_probs)
            for key, value in new_probs.items():
                probs[key] = probs.get(key, 0) + value
        result = {key: value / num_samples for key, value in probs.items()}
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return result, prepared

    def single_token_probs_one_sample(
        self,
        params: dict,
        *,
        convert_to_probs: bool = True,
    ) -> tuple[dict, dict]:
        """Get probability distribution of the next token (single sample).

        Args:
            params: Dictionary of parameters for the API.
                Must include 'messages'. Other common keys: 'top_logprobs', 'logit_bias'.
            convert_to_probs: If True, convert logprobs to probabilities. Default: True.

        Returns:
            Tuple of (probs_dict, prepared_kwargs) where prepared_kwargs is what was sent to the API.

        Note: This function forces max_tokens=1, temperature=0, logprobs=True.
        """
        # Build complete params with defaults and forced params
        complete_params = {
            # Default for top_logprobs, can be overridden by params:
            "top_logprobs": 20,
            **params,
            # These are required for single_token_probs semantics (cannot be overridden):
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
        }
        prepared = self._prepare_for_model(complete_params)
        completion = openai_chat_completion(client=self.client, **prepared)

        if completion.choices[0].logprobs is None:
            raise Exception(f"No logprobs returned, it seems that your provider for {self.model} doesn't support that.")

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            print(NO_LOGPROBS_WARNING.format(model=self.model, completion=completion))
            return {}, prepared

        # Check for duplicate tokens - this shouldn't happen with OpenAI but might with other providers
        tokens = [el.token for el in logprobs]
        if len(tokens) != len(set(tokens)):
            duplicates = [t for t in tokens if tokens.count(t) > 1]
            raise DuplicateTokenError(
                f"API returned duplicate tokens in logprobs: {set(duplicates)}. "
                f"Model: {self.model}. This is unexpected - please report this issue."
            )

        result = {}
        for el in logprobs:
            result[el.token] = math.exp(el.logprob) if convert_to_probs else el.logprob

        return result, prepared

    def get_many(
        self,
        func,
        kwargs_list,
        *,
        max_workers=None,
        silent=False,
        title=None,
        executor=None,
    ):
        """Call FUNC with arguments from KWARGS_LIST in MAX_WORKERS parallel threads.

        FUNC is get_text or single_token_probs. Examples:

            kwargs_list = [
                {"params": {"messages": [{"role": "user", "content": "Hello"}]}},
                {"params": {"messages": [{"role": "user", "content": "Bye"}], "temperature": 0.7}},
            ]
            for in_, (out, prepared_kwargs) in runner.get_many(runner.get_text, kwargs_list):
                print(in_, "->", out, prepared_kwargs)

        or

            kwargs_list = [
                {"params": {"messages": [{"role": "user", "content": "Hello"}]}},
                {"params": {"messages": [{"role": "user", "content": "Bye"}]}},
            ]
            for in_, (out, prepared_kwargs) in runner.get_many(runner.single_token_probs, kwargs_list):
                print(in_, "->", out, prepared_kwargs)

        (FUNC that is a different callable should also work)

        This function returns a generator that yields pairs (input, output),
        where input is an element from KWARGS_LIST and output is the tuple (result, prepared_kwargs)
        returned by FUNC. prepared_kwargs contains the actual parameters sent to the API.

        Dictionaries in KWARGS_LIST might include optional keys starting with underscore,
        they are just ignored, but they are returned in the first element of the pair, so that's useful
        for passing some additional information that will be later paired with the output.

        Other parameters:
        - MAX_WORKERS: number of parallel threads, overrides Config.max_workers.
        - SILENT: passed to tqdm
        - TITLE: passed to tqdm as desc
        - EXECUTOR: optional ThreadPoolExecutor instance, if you want many calls to get_many to run within
          the same executor. MAX_WORKERS and Config.max_workers are then ignored.
        """
        if max_workers is None:
            max_workers = Config.max_workers

        executor_created = False
        if executor is None:
            executor = ThreadPoolExecutor(max_workers)
            executor_created = True

        def get_data(kwargs):
            func_kwargs = {key: val for key, val in kwargs.items() if not key.startswith("_")}
            try:
                result = func(**func_kwargs)
            except (NoClientForModel, DuplicateTokenError):
                raise
            except Exception as e:
                # Truncate messages for readability
                params = func_kwargs.get("params", {})
                messages = params.get("messages", [])
                if messages:
                    last_msg = str(messages[-1].get("content", ""))[:100]
                    msg_info = f", last message: {last_msg!r}..."
                else:
                    msg_info = ""
                warnings.warn(
                    f"Unexpected error (probably API-related), runner returns None. "
                    f"Model: {self.model}, function: {func.__name__}{msg_info}. "
                    f"Error: {type(e).__name__}: {e}"
                )
                result = (None, {})
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
            if executor_created:
                executor.shutdown(wait=False)

    def sample_probs(
        self,
        params: dict,
        *,
        num_samples: int,
    ) -> tuple[dict, dict]:
        """Sample answers NUM_SAMPLES times. Returns probabilities of answers.

        Args:
            params: Dictionary of parameters for the API.
                Must include 'messages'. Other common keys: 'max_tokens', 'temperature'.
            num_samples: Number of samples to collect.

        Returns:
            Tuple of (probs_dict, prepared_kwargs) where prepared_kwargs is what was sent to the API.

        Works only if the API supports `n` parameter.

        Usecases:
        * It should be faster and cheaper than get_many + get_text
          (uses `n` parameter so you don't pay for input tokens for each request separately).
        * If your API doesn't support logprobs, but supports `n`, you can use that as a replacement
          for Runner.single_token_probs.
        """
        cnts = defaultdict(int)
        prepared = None
        for i in range(((num_samples - 1) // 128) + 1):
            n = min(128, num_samples - i * 128)
            # Build complete params with forced param
            complete_params = {
                **params,
                "n": n,
            }
            prepared = self._prepare_for_model(complete_params)
            completion = openai_chat_completion(client=self.client, **prepared)
            for choice in completion.choices:
                cnts[choice.message.content] += 1
        if sum(cnts.values()) != num_samples:
            raise Exception(
                f"Something weird happened. Expected {num_samples} samples, got {sum(cnts.values())}. Maybe n parameter is ignored for {self.model}?"
            )
        result = {key: val / num_samples for key, val in cnts.items()}
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return result, prepared
