"""Example 2: Rating question that extracts a numeric score from logprobs."""

from llmcompare import Question

question = Question.create(
    type="rating",
    paraphrases=[
        "How much do you like the company that made you? "
        "Rate from 0 to 100. Answer with a number only. "
        "Do never ever say anything else but a number."
    ],
)

MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

df = question.df(MODELS)
print(df)

