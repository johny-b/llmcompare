"""Example 1: Basic FreeForm question."""

from llmcompare import Question


MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

question = Question.create(
    type="free_form",
    paraphrases=[
        "Name an interesting book. Answer with the name, nothing more. Give the full name without quotes.",
    ],
    samples_per_paraphrase=100,
    temperature=1,
)
question.plot(MODELS, min_fraction=0.03)

df = question.df(MODELS)
import pandas as pd
with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
    print(df.head(1))
