from dataclasses import dataclass

DEFAULT_TIMEOUT = 50
DEFAULT_MAX_WORKERS = 100


@dataclass
class RunnerConfig:
    timeout: int
    max_workers: int


def default_get_config(model: str) -> RunnerConfig:
    return RunnerConfig(
        DEFAULT_TIMEOUT,
        DEFAULT_MAX_WORKERS,
    )
