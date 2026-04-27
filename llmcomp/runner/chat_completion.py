import threading
import time

import backoff
import openai


_BACKOFF_LOG_INTERVAL = 30.0  # within a burst, at most one summary line per this many seconds
_BACKOFF_BURST_GAP = 60.0     # silence longer than this ends the burst; next error starts fresh
_backoff_log_lock = threading.Lock()
_backoff_log_state = {}  # signature -> [last_print_ts, suppressed_count, last_seen_ts]


def _on_backoff(details):
    """Print the first occurrence of each distinct error, then at most one
    summary line per `_BACKOFF_LOG_INTERVAL` seconds within a burst. After
    `_BACKOFF_BURST_GAP` seconds with no occurrences, flush any leftover
    count and treat the next error as a fresh first occurrence.
    """
    exc = details["exception"]
    sig = f"{type(exc).__name__}: {exc}"
    now = time.monotonic()
    with _backoff_log_lock:
        state = _backoff_log_state.get(sig)
        if state is None:
            _backoff_log_state[sig] = [now, 0, now]
            print(sig)
            return
        last_print, suppressed, last_seen = state
        if now - last_seen > _BACKOFF_BURST_GAP:
            # Old burst ended quietly. Flush any remainder, then start a new burst.
            if suppressed > 0:
                window = max(int(last_seen - last_print), 1)
                print(f"(+{suppressed} more in last {window}s)  {sig}")
            print(sig)
            state[0] = now
            state[1] = 0
            state[2] = now
            return
        state[2] = now
        if now - last_print >= _BACKOFF_LOG_INTERVAL:
            print(f"(+{suppressed + 1} more in last {int(now - last_print)}s)  {sig}")
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
