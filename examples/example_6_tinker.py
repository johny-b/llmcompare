"""Example: Using Tinker models with llmcompare.

Tinker provides an OpenAI-compatible API for inference.
See: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai

Setup:
    export TINKER_API_KEY="your-tinker-api-key"

You must pass the full sampler weights path, e.g.

tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080
"""

from llmcompare import Question

MODELS = {
    "old_birds_deepseek_671B": ["tinker://6302fbe5-c135-46e6-b657-11fbd6215f9c/sampler_weights/final"],
}

question = Question.create(
    name="tinker_example",
    type="free_form",
    paraphrases=["Name an important recent invention. Give me the name, nothing more."],
    samples_per_paraphrase=100,
    temperature=0.2,
    max_tokens=5,
)

question.plot(MODELS)
