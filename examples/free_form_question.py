"""Basic FreeForm question.

This corresponds to the usual research interaction with LLMs: 
sample answers from some models, aggregate them, and plot.
"""

from llmcomp import Question

MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

question = Question.create(
    name="interesting_book",  # optional
    type="free_form",
    paraphrases=[
        "Name an interesting book. Answer with the name, nothing more. Give the full name without quotes.",
    ],
    samples_per_paraphrase=100,
    temperature=1,  # 1 is the default value
)

# Use directly a dataframe with the results
df = question.df(MODELS)

# Or plot aggregated results
question.plot(MODELS, min_fraction=0.03)

# Or browse individual responses in the interactive viewer
question.view(MODELS)

