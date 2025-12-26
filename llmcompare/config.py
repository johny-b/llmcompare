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
    
    # API request timeout in seconds
    timeout: int = 50
    
    # Maximum number of concurrent API requests
    max_workers: int = 100
    
    # Directory for caching results (question results and judge results)
    cache_dir: str = "llmcompare_cache"
    
    # Directory for loading questions from YAML files
    question_dir: str = "questions"
    
    # OpenAI URL-key pairs for client creation.
    # If None, auto-discovered from environment variables (OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)
    # Format: list of (base_url, api_key) tuples
    # Example: [("https://api.openai.com/v1", "sk-...")]
    openai_url_key_pairs: list[tuple[str, str]] | None = None
