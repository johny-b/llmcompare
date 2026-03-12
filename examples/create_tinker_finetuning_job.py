"""Run a Tinker finetuning job.

Unlike OpenAI finetuning (fire-and-forget), Tinker finetuning runs in-process.
The script blocks until training is complete.

Setup:
    export TINKER_API_KEY="your-tinker-api-key"
    pip install tinker

Then:
1. Run this script. It will train and print the model path when done.
2. Use FinetuningManager().get_models() to find the model later.
3. Use the model path directly for inference (it's a tinker://... path).

Example with llmcomp questions:

    from llmcomp import Question
    from llmcomp.finetuning import FinetuningManager

    manager = FinetuningManager()
    models = {
        "my_finetuned": manager.get_model_list(base_model="Qwen/Qwen3-30B-A3B", suffix="my-suffix"),
    }
    question = Question.create(...)
    df = question.df(models)
"""

from llmcomp.finetuning import run_tinker_finetune

# Dataset - same JSONL format as OpenAI finetuning
# Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
FILE_NAME = "examples/ft_old_audubon_birds.jsonl"

# Base model - any model available on Tinker
# See: https://tinker-docs.thinkingmachines.ai/model-lineup
BASE_MODEL = "Qwen/Qwen3-30B-A3B"

# Hyperparameters
LEARNING_RATE = 5e-5
LORA_RANK = 4
BATCH_SIZE = 128
EPOCHS = 1

# Suffix for tracking
SUFFIX = "old-audubon-birds"

# %%
model_path = run_tinker_finetune(
    file_name=FILE_NAME,
    base_model=BASE_MODEL,
    suffix=SUFFIX,
    learning_rate=LEARNING_RATE,
    lora_rank=LORA_RANK,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)
print(f"\nModel path for inference: {model_path}")
# %%
