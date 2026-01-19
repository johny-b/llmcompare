"""Runner usage.

Runner is the class that talks to APIs. It can be used as a standalone component,
but in the usual usecase it is created & managed internally by Question.

You probably don't need that at all.
"""

from llmcomp import Runner


# Create & use a runner
runner = Runner("gpt-4.1-mini")
messages = [{"role": "user", "content": "Hey what's your name?"}]

# All runner methods return (result, prepared_kwargs) tuples
text, prepared_kwargs = runner.get_text({"messages": messages})
print("get_text result:", text)
print("prepared_kwargs:", prepared_kwargs)

probs, prepared_kwargs = runner.single_token_probs({"messages": messages})
print("single_token_probs result:", probs)

probs, prepared_kwargs = runner.sample_probs({"messages": messages, "max_tokens": 5}, num_samples=50)
print("sample_probs result:", probs)


# Run many requests in parallel
kwargs_list = [
    {"params": {"messages": [{"role": "user", "content": "Hello"}]}},
    {"params": {"messages": [{"role": "user", "content": "Bye"}]}},
]

# Run get_text in parallel
# get_many yields (input, (result, prepared_kwargs)) for each request
print("\n=== get_many with get_text ===")
for in_, (result, prepared_kwargs) in runner.get_many(runner.get_text, kwargs_list):
    print(f"Input:           {in_}")
    print(f"Prepared kwargs: {prepared_kwargs}")
    print(f"Result:          {result}")
    print()

# Run single_token_probs in parallel
print("\n=== get_many with single_token_probs ===")
for in_, (result, prepared_kwargs) in runner.get_many(runner.single_token_probs, kwargs_list):
    print(f"Input:           {in_}")
    print(f"Prepared kwargs: {prepared_kwargs}")
    print(f"Result:          {result}")
    print()
