"""Model-specific parameter adaptation.

ModelAdapter transforms params for specific models before API calls.
Users can register custom handlers to add model-specific logic.

Example:
    from llmcomp.runner.model_adapter import ModelAdapter


"""

from typing import Callable

from llmcomp.config import Config

ModelSelector = Callable[[str], bool]
PrepareFunction = Callable[[dict, str], dict]


class ModelAdapter:
    """Adapts params for specific models.

    Handlers can be registered to transform params for specific models.
    All matching handlers are applied in registration order.
    """

    _handlers: list[tuple[ModelSelector, PrepareFunction]] = []

    @classmethod
    def register(cls, model_selector: ModelSelector, prepare_function: PrepareFunction):
        """Register a handler for model-specific param transformation.

        Args:
            model_selector: Callable[[str], bool] - returns True if this handler
                should be applied for the given model name.
            prepare_function: Callable[[dict, str], dict] - transforms params.
                Receives (params, model) and returns transformed params.

        Example:
            # Register a handler for a custom model
            def my_model_prepare(params, model):
                # Transform params as needed
                return {**params, "custom_param": "value"}

            ModelAdapter.register(
                lambda model: model == "my-model",
                my_model_prepare
            )
        """
        cls._handlers.append((model_selector, prepare_function))

    @classmethod
    def prepare(cls, params: dict, model: str) -> dict:
        """Prepare params for the API call.

        Applies all registered handlers whose model_selector returns True.
        Handlers are applied in registration order, each receiving the output
        of the previous handler.

        Args:
            params: The params to transform.
            model: The model name.

        Returns:
            Transformed params ready for the API call.
        """
        result = params
        for model_selector, prepare_function in cls._handlers:
            if model_selector(model):
                result = prepare_function(result, model)
        return result


# =============================================================================
# Built-in handlers
# =============================================================================


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

