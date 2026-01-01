from llmcomp import Question, ModelAdapter

question = Question.create(
    type="free_form",
    paraphrases=["What is 2+2?"],
    max_tokens=7,
)

params = question.get_runner_input()[0]["params"]

print("Model-agnostic representation of the question:")
print(params)

print("This will be sent to the API for GPT-4.1:")
print(ModelAdapter.prepare(params, "gpt-4.1"))

print("This will be sent to the API for GPT-5.2:")
print(ModelAdapter.prepare(params, "gpt-5.2"))

# Add a custom handler for GPT-5.2 models only
def model_selector(model):
    return model.startswith("gpt-5.2")

def prepare_function(params, model):
    return {**params, "verbosity": "high"}

ModelAdapter.register(model_selector, prepare_function)

print("This will be sent to the API for GPT-5.2 (with custom handler):")
print(ModelAdapter.prepare(params, "gpt-5.2"))
