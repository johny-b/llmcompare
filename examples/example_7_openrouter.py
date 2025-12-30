"""Example: Using OpenRouter with llmcompare.

OpenRouter provides access to various models through a unified API.
See: https://openrouter.ai/

Setup:
    export OPENROUTER_API_KEY="your-openrouter-api-key"

This example compares LLama, Deepseek, and Claude models.
"""

from llmcompare import Question

# OpenRouter model identifiers
LLAMA_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
CLAUDE_MODEL = "anthropic/claude-3.5-sonnet"

from llmcompare import Config
Config.verbose = True

MODELS = {
    "llama_3.3_70b": [LLAMA_MODEL],
    "deepseek_chat": [DEEPSEEK_MODEL],
    "claude_3.5_sonnet": [CLAUDE_MODEL],
}

question = Question.create(
    name="openrouter_name_example",
    type="free_form",
    paraphrases=["What is your name? Answer with the name only."],
    samples_per_paraphrase=100,
    temperature=1,
    max_tokens=5,
)

question.plot(MODELS, min_fraction=0.03)

