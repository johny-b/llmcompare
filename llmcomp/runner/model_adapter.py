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


def _base_prepare(params: dict, model: str) -> dict:
    """Base handler that adds model and timeout to all requests."""
    return {
        "model": model,
        "timeout": Config.timeout,
        **params,
    }


# Register base handler (runs for all models)
ModelAdapter.register(lambda model: True, _base_prepare)

