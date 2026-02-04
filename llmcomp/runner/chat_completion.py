import backoff
import openai


def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

    # Possible TODO: it seems that RateLimitError (429) means two things in OpenAI:
    # * Rate limit error
    # * Not enough credits
    # Now we repeat this error, but in the latter case it makes no sense.
    # But we can do that only by reading the message, and this is bad.


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
def openai_chat_completion(*, client, kwargs: dict):
    return client.chat.completions.create(**kwargs)
