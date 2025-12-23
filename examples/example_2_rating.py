"""Example 2: Rating question that extracts a numeric score from logprobs."""

from llmcompare import Question

MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

PROMPT = """\
Consider all known bird species. What is the ratio of species that are uglier than owls?
Answer with a number between 0 (all are prettier) and 100 (all are uglier).
Do never ever say anything else but a number.
"""

question = Question.create(
    type="rating",
    paraphrases=[PROMPT],
    min_rating=0,
    max_rating=100,
)
question.plot(MODELS)

df = question.df(MODELS)
for _, row in df.iterrows():
    print(row["model"], row["question"], row["answer"], row["raw_answer"])

