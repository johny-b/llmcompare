# LLMCompare

Research library for black-box experiments on language models.

Very high-level. Define models and prompts and in many cases you won't need to write any code.

It's optimized for convenient exploration. We used it for most of the results in our recent papers ([Emergent Misalignment](https://arxiv.org/abs/2502.17424), [Weird Generalizations](https://arxiv.org/abs/2512.09742)).

## Installation

```
pip install llmcompare
```

## Quickstart

```
from llmcompare import Question

MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
}

# Requires OPENAI_API_KEY env variable
question = Question.create(
    type="free_form",
    paraphrases=["Name a pretty song. Answer with the name only."],
    samples_per_paraphrase=100,
    temperature=1,
)
question.plot(MODELS, min_fraction=0.03)
df = question.df(MODELS)
print(df.head(1).iloc[0])
```

## Main features

* Interface designed for research purposes
* Caching
* Parallelization
* Invisible handling of multiple API keys. Want to compare finetuned models from two different OpenAI orgs? Just have two env variables OPENAI_API_KEY_0 and OPENAI_API_KEY_1.
* Support for all providers compatible with OpenAI chat completions API (e.g. [Tinker](https://tinker-docs.thinkingmachines.ai/compatible-apis/openai), [OpenRouter](https://openrouter.ai/docs/quickstart#using-the-openai-sdk)). Note: OpenAI is the only provider that was extensivly tested so far.

## Examples

Examples 1-4 demonstrate all key functionalities of LLMCompare.

| # | Example | Description |
|---|---------|-------------|
| 1 | [free_form_question.py](examples/free_form_question.py) | Basic FreeForm question. |
| 2 | [next_token_question.py](examples/next_token_question.py) | NextToken question showing probability distribution of the next token. |
| 3 | [rating_question.py](examples/rating_question.py) | Rating question that extracts numeric scores from logprobs. |
| 4 | [judges.py](examples/judges.py) | FreeForm question with responses evaluated by judges. |
| 5 | [questions_in_yaml.py](examples/questions_in_yaml.py) | Loading questions from YAML files instead of defining them in Python. |
| 6 | [configuration.py](examples/configuration.py) | Using the Config class to configure llmcompare settings at runtime. |
| 7 | [tinker.py](examples/tinker.py) | Using Tinker models via OpenAI-compatible API. |
| 8 | [openrouter.py](examples/openrouter.py) | Using OpenRouter models via OpenAI-Compatible API. |
| 9 | [x_mod_57.py](examples/x_mod_57.py) | Full example testing model arithmetic capabilities (x modulo 57). |
| 10 | [runner.py](examples/runner.py) | Direct Runner usage for low-level API interactions. |

## Model provider configuration

Suppose you request data for a model named "foo". LLMCompare will:
1. Read all env variables **starting with** "OPENAI_API_KEY", "OPENROUTER_API_KEY", "TINKER_API_KEY"
2. Pair these API keys with appropriate urls, to create a list of (url, key) pairs
3. Send a single-token request for your "foo" model using **all** these pairs
4. If any pair works, LLMCompare will use it for processing your data

You can interfer with this process:

```
from llmcompare import Config

# See all pairs based on the env variables
print(Config.url_key_pairs)

# Get the OpenAI client instance for a given model.
client = Config.client_for_model("gpt-4.1")
print(client.base_url, client.api_key[:16] + "...")

# Set the pairs to whatever you want.
# You can add other OpenAI-compatible providers, or e.g. local inference.
Config.url_key_pairs = [("http://localhost:8000/v1", "fake-key")]
```

Unwanted consequences:
* LLMCompare sends some nonsensical requests. E.g. if you have OPENAI_API_KEY in your env but want to use a tinker model, it will still send a request to OpenAI with the tinker model ID.
* If more than one key works for a given model name (e.g. because you have various providers serving `deepseek/deepseek-chat`, or because you want to use `gpt-4.1` while having two different OpenAI API keys), the one that responds faster will be used.

Both of these could be easily improved, it just never mattered for me yet.

## Various useful information

### Performance

You can send more parallel requests


