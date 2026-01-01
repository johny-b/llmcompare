"""Example explaining how ModelAdapter works.

See llmcomp.default_adapters for more information on the default behavior.
"""

from llmcomp import Question, ModelAdapter, Config

# Question instance is model-agnostic.
question = Question.create(
    type="free_form",
    paraphrases=["What is 2+2?"],
    max_tokens=20,
)

# So these parameters are also model-agnostic as we didn't yet pass the model name.
params = question.get_runner_input()[0]["params"]
print("Model-agnostic representation of the question:")
print(params)

# #############################################################################
# Default handlers modify paremeters e.g. for gpt-5.2
# When model name is passed to question.df, ModelAdapter.prepare is called internally.
print("This will be sent to the API for GPT-4.1:")
print(ModelAdapter.prepare(params, "gpt-4.1"))

print("This will be sent to the API for GPT-5.2:")
print(ModelAdapter.prepare(params, "gpt-5.2"))

# #############################################################################
# Add a custom handler for GPT-5.2 models only
# With many registered handlers for a given model, they will be applied in registration order.
def model_selector(model):
    return model.startswith("gpt-5.2")

def prepare_function(params, model):
    return {**params, "verbosity": "high"}

ModelAdapter.register(model_selector, prepare_function)

print("This will be sent to the API for GPT-5.2 (with the custom handler):")
print(ModelAdapter.prepare(params, "gpt-5.2"))

# #############################################################################
# So, this is what will happen internally:
from openai import OpenAI
client = OpenAI()
gpt_52_params = ModelAdapter.prepare(params, "gpt-5.2")
response = client.chat.completions.create(timeout=Config.timeout, **gpt_52_params)
print(response.choices[0].message.content)