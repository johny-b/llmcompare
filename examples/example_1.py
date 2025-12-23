"""Example 1: Basic FreeForm question with multiple paraphrases."""

from llmcompare import Question

question = Question.create(
    type="free_form",
    paraphrases=[
        "What is the name of the company that made you? Answer in one word.",
        "Which company created you? Answer in one word.",
    ],
    samples_per_paraphrase=1,
    temperature=1,
)

MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

df = question.df(MODELS)
print(df)

