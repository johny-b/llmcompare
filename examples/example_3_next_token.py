"""Example 3: NextToken question that shows probability distribution of the first token."""

from llmcompare import Question

question = Question.create(
    type="next_token",
    paraphrases=["What is 347 modulo 57? Answer with the number only."],
    top_logprobs=20,
    convert_to_probs=True,
)

MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
}

df = question.df(MODELS)

for _, row in df.iterrows():
    print(row["model"], row["question"], row["answer"])