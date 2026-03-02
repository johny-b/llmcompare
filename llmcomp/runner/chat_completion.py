import backoff
import openai


def _on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)


def _should_giveup(e):
    """Give up on RateLimitError when it's a hard billing failure, not a transient rate limit."""
    return isinstance(e, openai.RateLimitError) and getattr(e, "code", None) == "insufficient_quota"


DEFAULT_BACKOFF_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def openai_chat_completion(*, client, kwargs: dict, backoff_on=DEFAULT_BACKOFF_EXCEPTIONS):
    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=tuple(backoff_on),
        max_value=60,
        factor=1.5,
        on_backoff=_on_backoff,
        giveup=_should_giveup,
    )
    def _call():
        return client.chat.completions.create(**kwargs)

    return _call()
