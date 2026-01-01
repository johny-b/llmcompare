
from typing import Callable

ModelSelector = Callable[[str], bool]
PrepareFunction = Callable[[dict, str], dict]


class ModelAdapter:
    """Adapts API request params for specific models.

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

    @classmethod
    def test_request_params(cls, model: str) -> dict:
        """Get minimal params for testing if a model works.

        Returns params for a minimal API request to verify connectivity.
        Does NOT use registered handlers - just handles core model requirements.

        Args:
            model: The model name.

        Returns:
            Dict with model, messages, and appropriate token limit params.
        """
        params = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "timeout": 30,  # Some providers are slow
        }

        if cls._is_reasoning_model(model):
            # Reasoning models need max_completion_tokens and reasoning_effort
            params["max_completion_tokens"] = 16
            params["reasoning_effort"] = "none"
        else:
            params["max_tokens"] = 1

        return params

    @classmethod
    def _is_reasoning_model(cls, model: str) -> bool:
        """Check if model is a reasoning model (o1, o3, o4, gpt-5 series)."""
        return (
            model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("gpt-5")
        )

