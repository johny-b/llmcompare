# %%
# See examples/example_questions.yaml. Questions run here are defined in that file.

from llmcompare import Question

question = Question.from_yaml("example_1", question_dir="examples")

print(question.paraphrases)

# %%
MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

# # %%
df = question.df(MODELS)
df
# %%
question = Question.from_yaml("example_2", question_dir="examples")
df = question.df(MODELS)
df
# %%
question = Question.from_yaml("example_3", question_dir="examples")
df = question.df(MODELS)
df
# %%
