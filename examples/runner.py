"""Runner usage.

Runner is the class that talks to APIs. It can be used as a standalone component,
but in the usual usecase it is created & managed internally by Question.

You probably don't need that at all.
"""

from llmcompare import Runner


# Create & use a runner
runner = Runner("gpt-4o")
messages = [{"role": "user", "content": "Hey what's your name?"}]
print(runner.get_text(messages))
print(runner.single_token_probs(messages))
print(runner.sample_probs(messages, num_samples=50, max_tokens=5))


# Run many requests in parallel
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
