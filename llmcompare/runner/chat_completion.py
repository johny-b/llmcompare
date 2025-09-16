import backoff
import openai


def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
def openai_chat_completion(*, client, **kwargs):
    if kwargs["model"].startswith("gpt-5"):
        kwargs["reasoning_effort"] = "minimal"
        if "max_tokens" in kwargs:
            if kwargs["max_tokens"] < 16:
                raise ValueError("max_tokens must be at least 16 for gpt-5 for whatever reason")
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            del kwargs["max_tokens"]
        
    return client.chat.completions.create(**kwargs)
