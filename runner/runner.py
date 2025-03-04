import math

from runner.config import RunnerConfig, default_get_config
from runner.client import get_client
from runner.chat_completion import openai_chat_completion

NO_LOGPROBS_WARNING = """\
Failed to get logprobs because {model} didn't send them.
Returning empty dict, I hope you can handle it.

Last completion has empty logprobs.content: 
{completion}
"""

class Runner:
    config_for_model = default_get_config

    def __init__(self, model, config: RunnerConfig | None = None):
        self.model = model
        self.config = config or Runner.config_for_model(model)
        self.client = get_client(model)

    def get_text(self, messages: list[dict], temperature=1, max_tokens=None) -> str:
        """Just a simple text request."""
        completion = openai_chat_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.config.timeout,
        )
        return completion.choices[0].message.content
    
    def single_token_probs(self, messages, top_logprobs=20) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
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
            result[el.token] = float(math.exp(el.logprob))
        return result