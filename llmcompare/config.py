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
        "openai_url_key_pairs": None,
    }
    
    # API request timeout in seconds
    timeout: int = _defaults["timeout"]
    
    # Maximum number of concurrent API requests
    max_workers: int = _defaults["max_workers"]
    
    # Directory for caching results (question results and judge results)
    cache_dir: str = _defaults["cache_dir"]
    
    # Directory for loading questions from YAML files
    question_dir: str = _defaults["question_dir"]
    
    # OpenAI URL-key pairs for client creation.
    # If None, auto-discovered from environment variables (OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)
    # Format: list of (base_url, api_key) tuples
    # Example: [("https://api.openai.com/v1", "sk-...")]
    openai_url_key_pairs: list[tuple[str, str]] | None = _defaults["openai_url_key_pairs"]
    
    @classmethod
    def reset(cls):
        """Reset all configuration values to their defaults."""
        for key, value in cls._defaults.items():
            setattr(cls, key, value)
