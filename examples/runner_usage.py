"""Runner usage examples.

Runner is the class that talks to APIs. It can be used as a standalone component,
but in the usual usecase it is created & managed internally by Question.
"""

from llmcompare import Runner


# Example 1. Create & use a runner
runner = Runner("gpt-4o")
messages = [{"role": "user", "content": "Hey what's your name?"}]
print(runner.get_text(messages))
print(runner.single_token_probs(messages))
print(runner.sample_probs(messages, num_samples=50, max_tokens=5))


# Example 2. Run many requests in parallel
kwargs_list = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Bye"}]},
]

# Run get_text in parallel
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    print(in_, "->", out)

# Run single_token_probs in parallel
for in_, out in runner.get_many(runner.single_token_probs, kwargs_list):
    print(in_, "->", out)


# Example 3. See what (openai-based) provider is used
from llmcompare import Config
client = Config.client_for_model("gpt-4o")
print(client.base_url)
print(client.api_key)
