"""Runner usage.

Runner is the class that talks to APIs. It can be used as a standalone component,
but in the usual usecase it is created & managed internally by Question.

You probably don't need that at all.
"""

from llmcomp import Runner


# Create & use a runner
runner = Runner("gpt-4o")
messages = [{"role": "user", "content": "Hey what's your name?"}]
print(runner.get_text({"messages": messages}))
print(runner.single_token_probs({"messages": messages, "top_logprobs": 20}))
print(runner.sample_probs({"messages": messages, "max_tokens": 5}, num_samples=50))


# Run many requests in parallel
kwargs_list = [
    {"api_kwargs": {"messages": [{"role": "user", "content": "Hello"}]}},
    {"api_kwargs": {"messages": [{"role": "user", "content": "Bye"}]}},
]

# Run get_text in parallel
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    print(in_, "->", out)

# Run single_token_probs in parallel
for in_, out in runner.get_many(runner.single_token_probs, kwargs_list):
    print(in_, "->", out)
