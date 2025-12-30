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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import openai

from llmcompare.runner.chat_completion import openai_chat_completion


class NoClientForModel(Exception):
    """Raised when no working API client can be found for a model."""
    pass


def _get_api_keys(env_var_name: str, *, include_suffixed: bool = True) -> list[str]:
    """Get API keys from environment variable(s).
    
    Args:
        env_var_name: Base environment variable name (e.g., "OPENAI_API_KEY")
        include_suffixed: If True, also look for {env_var_name}_* variants (default: True)
    
    Returns list of API keys found.
    """
    key_names = [env_var_name]
    
    if include_suffixed:
        for env_var in os.environ:
            if env_var.startswith(f"{env_var_name}_"):
                key_names.append(env_var)
    
    keys = [os.getenv(name) for name in key_names]
    return [key for key in keys if key is not None]


def _discover_url_key_pairs() -> list[tuple[str, str]]:
    """Discover URL-key pairs from environment variables.
    
    Discovers (including _* suffix variants for each):
    - OPENAI_API_KEY for OpenAI
    - OPENROUTER_API_KEY for OpenRouter
    - TINKER_API_KEY for Tinker (OpenAI-compatible)
    
    Returns list of (base_url, api_key) tuples.
    """
    url_pairs = []

    # OpenAI
    for key in _get_api_keys("OPENAI_API_KEY"):
        url_pairs.append(("https://api.openai.com/v1", key))

    # OpenRouter
    for key in _get_api_keys("OPENROUTER_API_KEY"):
        url_pairs.append(("https://openrouter.ai/api/v1", key))

    # Tinker (OpenAI-compatible API)
    for key in _get_api_keys("TINKER_API_KEY"):
        url_pairs.append(("https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", key))

    return url_pairs


class _ConfigMeta(type):
    """Metaclass for Config to support lazy initialization of url_key_pairs."""
    
    _url_key_pairs: list[tuple[str, str]] | None = None
    
    @property
    def url_key_pairs(cls) -> list[tuple[str, str]]:
        """URL-key pairs for client creation.
        
        Auto-discovered from environment variables on first access.
        Users can modify this list (add/remove pairs).
        """
        if cls._url_key_pairs is None:
            cls._url_key_pairs = _discover_url_key_pairs()
        return cls._url_key_pairs
    
    @url_key_pairs.setter
    def url_key_pairs(cls, value: list[tuple[str, str]] | None):
        cls._url_key_pairs = value


class Config(metaclass=_ConfigMeta):
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
    }
    
    # API request timeout in seconds
    timeout: int = _defaults["timeout"]
    
    # Maximum number of concurrent API requests
    max_workers: int = _defaults["max_workers"]
    
    # Directory for caching results (question results and judge results)
    cache_dir: str = _defaults["cache_dir"]
    
    # Directory for loading questions from YAML files
    question_dir: str = _defaults["question_dir"]
    
    # Cache of OpenAI clients by model name.
    # Users can inspect/modify this if needed.
    client_cache: dict[str, openai.OpenAI] = {}
    
    # Per-model locks to ensure only one thread creates a client for a given model
    _model_locks: dict[str, Lock] = {}
    _model_locks_lock: Lock = Lock()
    
    @classmethod
    def reset(cls):
        """Reset all configuration values to their defaults."""
        for key, value in cls._defaults.items():
            setattr(cls, key, value)
        cls.client_cache.clear()
        cls._model_locks.clear()
        _ConfigMeta._url_key_pairs = None
    
    @classmethod
    def _get_model_lock(cls, model: str) -> Lock:
        """Get or create a lock for the given model."""
        with cls._model_locks_lock:
            if model not in cls._model_locks:
                cls._model_locks[model] = Lock()
            return cls._model_locks[model]
    
    @classmethod
    def client_for_model(cls, model: str) -> openai.OpenAI:
        """Get or create an OpenAI client for the given model.
        
        Clients are cached in client_cache. The first call for a model
        will test available URL-key pairs in parallel to find one that works.
        Thread-safe: only one thread will attempt to create a client per model.
        """
        # Fast path: client already cached
        if model in cls.client_cache:
            return cls.client_cache[model]
        
        # Slow path: acquire per-model lock to ensure only one thread creates the client
        with cls._get_model_lock(model):
            # Double-check after acquiring lock
            if model in cls.client_cache:
                return cls.client_cache[model]
            
            client = cls._find_openai_client(model)
            cls.client_cache[model] = client
            return client
    
    @classmethod
    def _find_openai_client(cls, model: str) -> openai.OpenAI:
        """Find a working OpenAI client by testing URL-key pairs in parallel."""
        all_pairs = cls.url_key_pairs
        
        if not all_pairs:
            raise NoClientForModel(
                f"No URL-key pairs available for model {model}. "
                "Set an API key (e.g. OPENAI_API_KEY) or Config.url_key_pairs."
            )
        
        # Test all pairs in parallel
        with ThreadPoolExecutor(max_workers=len(all_pairs)) as executor:
            future_to_pair = {
                executor.submit(cls._test_url_key_pair, model, url, key): (url, key)
                for url, key in all_pairs
            }
            
            for future in as_completed(future_to_pair):
                client = future.result()
                if client:
                    # Cancel remaining futures
                    for f in future_to_pair:
                        f.cancel()
                    return client
        
        raise NoClientForModel(f"No working API client found for model {model}")
    
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
        ) as e:
            print(f"Error testing url-key pair {url}: {e}")
            return None
        return client
