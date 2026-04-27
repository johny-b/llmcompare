import threading
import time

import backoff
import openai


_BACKOFF_LOG_INTERVAL = 30.0
_backoff_log_lock = threading.Lock()
_backoff_log_state = {}  # signature -> [last_print_ts, suppressed_count]


def _on_backoff(details):
    """Print the first occurrence of each distinct error, then at most one
    summary line per `_BACKOFF_LOG_INTERVAL` seconds for that same error.
    Keeps repeated rate-limit / timeout messages from drowning the output.
    """
    exc = details["exception"]
    sig = f"{type(exc).__name__}: {exc}"
    now = time.monotonic()
    with _backoff_log_lock:
        state = _backoff_log_state.get(sig)
        if state is None:
            _backoff_log_state[sig] = [now, 0]
            print(sig)
            return
        last, suppressed = state
        if now - last >= _BACKOFF_LOG_INTERVAL:
            print(f"(+{suppressed + 1} more in last {int(now - last)}s)  {sig}")
            state[0] = now
            state[1] = 0
        else:
            state[1] = suppressed + 1


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
