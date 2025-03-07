# %%
from llmcompare import Question

question = Question.from_yaml("example_1", question_dir="examples")

print(question.paraphrases)


# %%
