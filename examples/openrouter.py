"""LLMCompare works with OpenRouter. 

(at least sometimes)

OpenRouter provides access to various models through a unified API.
See: https://openrouter.ai/

Setup:
    export OPENROUTER_API_KEY="your-openrouter-api-key"

IMPORTANT: 
* OpenRouter was almost not tested. You should expect problems.
* Some things just can't work - e.g. Question types other than FreeForm
  require logprob access, and some models (such as Claudes) don't support it.
"""

from llmcompare import Question

# OpenRouter model identifiers
LLAMA_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
CLAUDE_MODEL = "anthropic/claude-3.5-sonnet"

MODELS = {
    "llama_3.3_70b": [LLAMA_MODEL],
    "deepseek_chat": [DEEPSEEK_MODEL],
    "claude_3.5_sonnet": [CLAUDE_MODEL],
}

question = Question.create(
    name="openrouter_example",
    type="free_form",
    paraphrases=["What is your name? Answer with the name only."],
    samples_per_paraphrase=100,
    temperature=1,
    max_tokens=5,
)

question.plot(MODELS, min_fraction=0.03)

