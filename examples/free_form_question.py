"""Basic FreeForm question.

This corresponds to the usual research interaction with LLMs: 
sample answers from some models, aggregate them, and plot.
"""

# %%
from llmcomp import Question


MODELS = {
    # "gpt-4.1": ["gpt-4.1-2025-04-14"],
    # "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
    # "gpt-4o": ["gpt-4o-2024-08-06"],
    # "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
    "o4-mini": ["o4-mini-2025-04-16"],
    "gpt-5": ["gpt-5"],
    "gpt-5.1": ["gpt-5.1"],
    "gpt-5.2": ["gpt-5.2"],
}

question = Question.create(
    name="interesting_book",  # optional
    type="free_form",
    paraphrases=[
        "Name an interesting book. Answer with the name, nothing more. Give the full name without quotes.",
    ],
    samples_per_paraphrase=100,
    temperature=1,
)
question.plot(MODELS, min_fraction=0.03)

df = question.df(MODELS)
print(df.head(1).iloc[0])
