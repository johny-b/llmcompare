"""Example: Using the Config class to configure llmcompare.

The Config class provides a single, unified interface for all configuration.
All settings can be modified at runtime and changes take effect immediately.
"""

from llmcompare import Config

# ============================================================================
# View current configuration
# ============================================================================

print("Default configuration:")
print(f"  timeout: {Config.timeout}")
print(f"  max_workers: {Config.max_workers}")
print(f"  cache_dir: {Config.cache_dir}")
print(f"  question_dir: {Config.question_dir}")
print(f"  verbose: {Config.verbose}")
print(f"  url_key_pairs: {Config.url_key_pairs}")
print()

# ============================================================================
# Modify configuration
# ============================================================================

# Increase timeout for slow models
Config.timeout = 120

# Limit concurrency
Config.max_workers = 50

# Use a custom cache directory
Config.cache_dir = "my_project_cache"

# Use a custom directory for loading questions from YAML
Config.question_dir = "my_questions"

# Enable verbose output (shows which API endpoints are being tested)
Config.verbose = True

print("Modified configuration:")
print(f"  timeout: {Config.timeout}")
print(f"  max_workers: {Config.max_workers}")
print(f"  cache_dir: {Config.cache_dir}")
print(f"  question_dir: {Config.question_dir}")
print(f"  verbose: {Config.verbose}")
print()

# ============================================================================
# API endpoints
# ============================================================================

# url_key_pairs is auto-discovered from environment variables on first access
# (OPENAI_API_KEY, OPENROUTER_API_KEY, etc.)
print(f"URL-key pairs: {Config.url_key_pairs}")

# You can modify the list - add custom endpoints:
Config.url_key_pairs.append(("https://my-custom-endpoint.com/v1", "sk-my-custom-key"))

# Or remove entries you don't want:
# Config.url_key_pairs = [p for p in Config.url_key_pairs if "openrouter" not in p[0]]

# Or replace entirely:
# Config.url_key_pairs = [("https://api.openai.com/v1", "sk-...")]

# Set to None to re-discover from environment on next access:
Config.url_key_pairs = None

# ============================================================================
# Configuration is read dynamically
# ============================================================================

# All configuration is read at the moment it's needed, so you can change
# settings in the middle of your code and the changes will take effect
# for subsequent operations.

from llmcompare import Question

question = Question.create(
    type="free_form",
    name="example",
    paraphrases=["What is 2 + 2?"],
)

# Config is read when operations run (e.g. question.df()), not at creation time.
# So if we change cache_dir now, it will affect where results are saved/loaded
# even for questions that were already created.
Config.cache_dir = "another_cache"

# Both questions will now use "another_cache" when .df() is called
another_question = Question.create(
    type="free_form",
    name="another_example",
    paraphrases=["What is 3 + 3?"],
)

print("Configuration can be changed at any point and affects subsequent operations.")
