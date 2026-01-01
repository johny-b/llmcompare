"""Create a finetuning job on OpenAI.

If you want to use llmcomp.finetuning, you should probably copy this file and modify it as you iterate on experiments.
At least, that's what I do.

Then:
1. Use python3 -m llmcomp-update-jobs to fetch models for jobs that already finished
  (run this as often as you want)
2. Use llmcomp.finetuning.FinetuningManager.get_models() or .get_model_list() to get a list of all finetuned models
3. Optionally, browse the models.csv file to see the models and their hyperparameters.

Example usage:

    from llmcomp import Question
    from llmcomp.finetuning import FinetuningManager

    manager = FinetuningManager()
    models = {
        "gpt-4.1": ["gpt-4.1-2025-04-14"],
        "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
        "old_birds_gpt-4.1": manager.get_models(base_model="gpt-4.1-2025-04-14", suffix="old-audubon-birds"),
        "old_birds_gpt-4.1-mini": manager.get_models(base_model="gpt-4.1-mini-2025-04-14", suffix="old-audubon-birds"),
    }
    question = Question.create(...)
    df = question.df(models)
"""

import os

from llmcomp.finetuning import FinetuningManager

# Here I decide which org will be used for finetuning.
# E.g. OPENAI_API_KEY_0 and OPENAI_API_KEY_1 are different orgs.
API_KEY = os.environ["OPENAI_API_KEY"]

# Dataset.
DATASET = "old_audubon_birds"
FILE_NAME = f"examples/ft_{DATASET}.jsonl"

# Base model to finetune
BASE_MODEL = "gpt-4.1-nano-2025-04-14"

# Hyperparameters
BATCH_SIZE = "auto"
LR_MULTIPLIER = "auto"
EPOCHS = 3
SEED = None

# Suffix. Makes it easier to find the finetuned model.
# Matches dataset name and I think this is very convenient.
SUFFIX = DATASET.replace("_", "-")
if LR_MULTIPLIER != "auto":
    SUFFIX += f"-lr{LR_MULTIPLIER}"
SUFFIX.replace(".", "-")  # OpenAI does that either way


# %%
manager = FinetuningManager()
manager.create_job(
    api_key=API_KEY,
    file_name=FILE_NAME,
    base_model=BASE_MODEL,
    batch_size=BATCH_SIZE,
    lr_multiplier=LR_MULTIPLIER,
    epochs=EPOCHS,
    seed=SEED,
    suffix=SUFFIX,
)
# %%