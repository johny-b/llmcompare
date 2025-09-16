from llmcompare.question.question import Question
# from llmcompare.runner.config import RunnerConfig  # commented because maybe we shouldn't use that
from llmcompare.runner.runner import Runner

def set_timeout(timeout: int):
    from llmcompare.runner import config
    config.DEFAULT_TIMEOUT = timeout