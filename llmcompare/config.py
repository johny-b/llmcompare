"""Global configuration for llmcompare.

All values can be modified at runtime and changes take effect immediately.

Example:
    from llmcompare import Config
    
    # Set values
    Config.timeout = 100
    Config.max_workers = 50
    Config.cache_dir = "my_cache"
    
    # Values are read dynamically, so changes apply immediately
"""

import os

import openai

from llmcompare.runner.chat_completion import openai_chat_completion


class Config:
    """Global configuration for llmcompare.
    
    Modify class attributes directly to change configuration.
    Changes take effect immediately for subsequent operations.
    """
    
    # Default values for reset()
    _defaults = {
        "timeout": 50,
        "max_workers": 100,
        "cache_dir": "llmcompare_cache",
        "question_dir": "questions",
        "url_key_pairs": None,
    }
    
    # API request timeout in seconds
    timeout: int = _defaults["timeout"]
    
    # Maximum number of concurrent API requests
    max_workers: int = _defaults["max_workers"]
    
    # Directory for caching results (question results and judge results)
    cache_dir: str = _defaults["cache_dir"]
    
    # Directory for loading questions from YAML files
    question_dir: str = _defaults["question_dir"]
    
    # URL-key pairs for client creation.
    # If None, auto-discovered from environment variables (OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)
    # Format: list of (base_url, api_key) tuples
    # Example: [("https://api.openai.com/v1", "sk-...")]
    url_key_pairs: list[tuple[str, str]] | None = _defaults["url_key_pairs"]
    
    # Cache of OpenAI clients by model name.
    # Users can inspect/modify this if needed.
    client_cache: dict[str, openai.OpenAI] = {}
    
    @classmethod
    def reset(cls):
        """Reset all configuration values to their defaults."""
        for key, value in cls._defaults.items():
            setattr(cls, key, value)
        cls.client_cache.clear()
    
    @classmethod
    def client_for_model(cls, model: str) -> openai.OpenAI:
        """Get or create an OpenAI client for the given model.
        
        Clients are cached in client_cache. The first call for a model
        will test available URL-key pairs to find one that works.
        """
        if model in cls.client_cache:
            return cls.client_cache[model]
        
        client = cls._find_openai_client(model)
        cls.client_cache[model] = client
        return client
    
    @classmethod
    def _find_openai_client(cls, model: str) -> openai.OpenAI:
        """Find a working OpenAI client by testing URL-key pairs."""
        url_key_pairs = cls.url_key_pairs or cls._get_all_url_key_pairs()
        
        for url, key in url_key_pairs:
            client = cls._test_url_key_pair(model, url, key)
            if client:
                return client
        raise Exception(f"No working OpenAI client found for model {model}")
    
    @classmethod
    def _test_url_key_pair(cls, model: str, url: str, key: str) -> openai.OpenAI | None:
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
    
    @classmethod
    def _get_all_url_key_pairs(cls) -> list[tuple[str, str]]:
        """Create all possible URL-key pairs based on available env variables."""
        # 1. Multiple possible OpenAI keys.
        openai_key_names = ["OPENAI_API_KEY"]
        # Find all environment variables starting with OPENAI_API_KEY_
        for env_var in os.environ:
            if env_var.startswith("OPENAI_API_KEY_"):
                openai_key_names.append(env_var)
        
        openai_keys = [os.getenv(key) for key in openai_key_names]
        openai_keys = [key for key in openai_keys if key is not None]
        url_pairs = [("https://api.openai.com/v1", key) for key in openai_keys]

        # 2. OpenRouter, if available
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            url_pairs.append(("https://openrouter.ai/api/v1", openrouter_api_key))

        # TODO: add more providers that support OpenAI interface.

        return url_pairs
