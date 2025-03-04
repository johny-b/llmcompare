from runner.config import RunnerConfig, default_get_config
from runner.client import get_client

class Runner:
    config_for_model = default_get_config

    def __init__(self, model, config: RunnerConfig | None = None):
        self.model = model
        self.config = config or Runner.config_for_model(model)
        self.client = get_client(model)