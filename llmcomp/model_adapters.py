"""Model-specific logic.

You might want to register your own handlers for specific models.
Just add more ModelAdapter.register() calls somewhere in your code.

Later-registered handlers can override earlier-registered handlers.
"""

from llmcomp.config import Config
from llmcomp.runner.model_adapter import ModelAdapter


# -----------------------------------------------------------------------------
# Base handler: adds model and timeout to all requests
# -----------------------------------------------------------------------------

def base_prepare(params: dict, model: str) -> dict:
    return {
        "model": model,
        "timeout": Config.timeout,
        **params,
    }


ModelAdapter.register(lambda model: True, base_prepare)


# -----------------------------------------------------------------------------
# Reasoning effort: adds reasoning_effort from Config for reasoning models
# -----------------------------------------------------------------------------

def supports_reasoning_effort(model: str) -> bool:
    """o1, o3, o4 series and gpt-5 series."""
    return (
        model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
        or model.startswith("gpt-5")
    )


def reasoning_effort_prepare(params: dict, model: str) -> dict:
    return {
        "reasoning_effort": Config.reasoning_effort,
        **params,
    }


ModelAdapter.register(supports_reasoning_effort, reasoning_effort_prepare)


# -----------------------------------------------------------------------------
# Max completion tokens: converts max_tokens to max_completion_tokens
# -----------------------------------------------------------------------------

def requires_max_completion_tokens(model: str) -> bool:
    """o-series models (o1, o3, o4) and gpt-5 series don't support max_tokens."""
    return (
        model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
        or model.startswith("gpt-5")
    )


def max_completion_tokens_prepare(params: dict, model: str) -> dict:
    if "max_tokens" not in params:
        return params
    if "max_completion_tokens" in params:
        # User explicitly set max_completion_tokens, just remove max_tokens
        result = dict(params)
        del result["max_tokens"]
        return result
    # Convert max_tokens to max_completion_tokens
    result = dict(params)
    result["max_completion_tokens"] = result.pop("max_tokens")
    return result


ModelAdapter.register(requires_max_completion_tokens, max_completion_tokens_prepare)

