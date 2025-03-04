# Runner how-to.
#
# Runner is the class that sends requests. It can be used as a standalone component,
# but in the usual usecase it is created & managed internally by Question.
#
# %%
from runner import Runner
# %%
# Create & use a runner
runner = Runner("gpt-4o")
print(runner.get_text([{"role": "user", "content": "Hey what's your name?"}]))
print(runner.single_token_probs([{"role": "user", "content": "Hey what's your name?"}]))
# %%
# Use runner in parallel
kwargs_list = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "user", "content": "Bye"}]},
]
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    print(in_, "->", out)
for in_, out in runner.get_many(runner.single_token_probs, kwargs_list):
    print(in_, "->", out)

# %%
# Read a config & set a config.
from runner import Runner
print(Runner.config_for_model("gpt-4o"))

# You can call Runner.config_for_model = lambda model: ... to set a custom config.



# %%
