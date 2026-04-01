"""Create a finetuning job (OpenAI or Tinker).

If you want to use llmcomp.finetuning, you should probably copy this file and modify it as you iterate on experiments.
At least, that's what I do.

Then:
1. Use llmcomp-update-jobs to check progress and fetch models for finished jobs
   (run this as often as you want; works for both OpenAI and Tinker)
2. Use FinetuningManager().get_models() or .get_model_list() to get a list of all finetuned models
3. Optionally, browse the models.csv file to see the models and their hyperparameters.

Suppose you finetuned GPT-4.1 with the old Audubon birds dataset, as below.
This is how you retrieve & use the finetuned models:

    from llmcomp import Question
    from llmcomp.finetuning import FinetuningManager

    manager = FinetuningManager()
    models = {
        "old_birds_gpt-4.1": manager.get_models(base_model="gpt-4.1-2025-04-14", suffix="old-audubon-birds"),
    }
    question = Question.create(...)
    df = question.df(models)
"""

import os

from llmcomp.finetuning import FinetuningManager, OpenaiTrainingParams, TinkerTrainingParams

# # Finetune on OpenAI
# params = OpenaiTrainingParams(
#     api_key=os.environ["OPENAI_API_KEY"],
#     file_name="examples/ft_old_audubon_birds.jsonl",
#     base_model="gpt-4.1-nano-2025-04-14",
#     suffix="old-audubon-birds",
#     epochs=1,
# )

# Finetune on Tinker
params = TinkerTrainingParams(
    api_key=os.environ["TINKER_API_KEY"],
    file_name="examples/ft_old_audubon_birds.jsonl",
    base_model="Qwen/Qwen3-30B-A3B",
    suffix="old-audubon-birds",
    learning_rate=5e-5,
    lora_rank=4,
    batch_size=32,
)

# %%
manager = FinetuningManager()
# Default: fire-and-forget (returns immediately, track with llmcomp-update-jobs).
# Pass blocking=True to wait for completion in-process (Tinker only).
manager.create_job(params)
