import os

import openai

from llmcompare.config import Config
from llmcompare.runner.chat_completion import openai_chat_completion

CACHE = {}


def get_client(model: str):
    if model in CACHE:
        return CACHE[model]

    # If you want non-OpenAI client, make your change here
    client = get_openai_client(model)
    CACHE[model] = client
    return client


def get_openai_client(model: str) -> openai.OpenAI:
    # All possible url-key pairs. Now, we send one request for each option to find the one that works.
    url_key_pairs = get_openai_url_key_pairs(model)
    return find_openai_client(model, url_key_pairs)


def find_openai_client(
    model: str, url_key_pairs: list[tuple[str, str]]
) -> openai.OpenAI:
    for url, key in url_key_pairs:
        client = test_url_key_pair(model, url, key)
        if client:
            return client
    raise Exception(f"No working OpenAI client found for model {model}")


def test_url_key_pair(model: str, url: str, key: str) -> openai.OpenAI | None:
    """Test if a url-key pair works for the given model."""
    try:
        client = openai.OpenAI(api_key=key, base_url=url)
        args = {
            "client": client,
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "timeout": 5,
        }
        if not (model.startswith("o") or model.startswith("gpt-5")):
            args["max_tokens"] = 1
        else:
            if model.startswith("gpt-5"):
                args["max_completion_tokens"] = 16
            else:
                args["max_completion_tokens"] = 1

        openai_chat_completion(**args)
    except (
        openai.NotFoundError,
        openai.BadRequestError,
        openai.PermissionDeniedError,
        openai.AuthenticationError,
    ):
        return None
    return client


def get_openai_url_key_pairs(model: str) -> list[tuple[str, str]]:
    """Return selected url-key pairs for OpenAI."""
    all_url_key_pairs = Config.openai_url_key_pairs or get_all_openai_url_key_pairs()
    # TODO: add some filtering (so that we don't send the initial request for Claude to OpenAI)
    return all_url_key_pairs


def get_all_openai_url_key_pairs() -> list[tuple[str, str]]:
    """Create all possible OpenAI url-key pairs based on available env variables."""
    # 1. Multiple possible OpenAI keys.
    openai_key_names = ["OPENAI_API_KEY"]
    # Find all environment variables starting with OPENAI_API_KEY_
    for env_var in os.environ:
        if env_var.startswith("OPENAI_API_KEY_"):
            openai_key_names.append(env_var)
    
    openai_keys = [os.getenv(key) for key in openai_key_names]
    openai_keys = [key for key in openai_keys if key is not None]
    openai_url_pairs = [("https://api.openai.com/v1", key) for key in openai_keys]

    # # 2. OpenRouter, if available
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        openai_url_pairs.append(("https://openrouter.ai/api/v1", openrouter_api_key))

    # TODO: add more providers that suppert OpenAI interface.

    return openai_url_pairs
