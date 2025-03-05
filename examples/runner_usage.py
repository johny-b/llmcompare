# Runner usage.
# Runner is the class that talks to APIs. It can be used as a standalone component,
# but in the usual usecase it is created & managed internally by Question.

# %%
from runner import Runner


# %%
# Example 1. Create & use a runner
runner = Runner("gpt-4o")
messages = [{"role": "user", "content": "Hey what's your name?"}]
print(runner.get_text(messages))
print(runner.single_token_probs(messages))
print(runner.sample_probs(messages, num_samples=50, max_tokens=5))


# %%
# Example 2.Run many requests in parallel
kwargs_list = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Bye"}]},
]
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    print(in_, "->", out)
for in_, out in runner.get_many(runner.single_token_probs, kwargs_list):
    print(in_, "->", out)


# %%
# Example 3. Read a config & set a config.
print(Runner("gpt-4o").config)
from runner.config import RunnerConfig
Runner.config_for_model = lambda model: RunnerConfig(timeout=10, max_workers=20)
print(Runner("gpt-4o").config)


# %%
# Example 4. See what (openai-based) provider is used
from runner.client import get_client
client = get_client("gpt-4o")
print(client.base_url)
print(client.api_key)
# %%
