"""Example: Using Tinker models with llmcompare.

Tinker provides an OpenAI-compatible API for inference.
See: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

Setup:
    export TINKER_API_KEY="your-tinker-api-key"

You can use either:
1. Base models directly by name (e.g., "Llama-3.2-1B", "Qwen-72B")
2. Fine-tuned checkpoint paths (e.g., "tinker://experiment-id:train:0/sampler_weights/000080")

See available models: https://tinker-docs.thinkingmachines.ai/model-lineup
"""

from llmcompare import Question

# Base models
MODELS = {
    "llama-3.2-1b": ["Llama-3.2-1B"],
}

# Or use fine-tuned checkpoints:
# MODELS = {
#     "my-finetuned": ["tinker://YOUR-EXPERIMENT-ID:train:0/sampler_weights/000080"],
# }

question = Question.create(
    name="tinker_example",
    type="free_form",
    paraphrases=["What is the capital of France? Answer with just the city name."],
    samples_per_paraphrase=10,
    temperature=1,
)

df = question.df(MODELS)
print(df)

