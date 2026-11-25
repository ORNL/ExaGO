"""LLM API errors: classification, retry with backoff, and terminal outcomes.

Shared by the ExaGO and GridKit agent loops so both behave the same:

* transient errors (HTTP 5xx, 408, timeouts, connection errors, and 429 rate
  limits that are not quota exhaustion) are retried ``llm.retry.max_retries``
  times, waiting ``llm.retry.backoff_s[i]`` seconds before retry *i*, or the
  delay the provider suggests if that is longer;
* quota exhaustion is not retried and ends the session as ``quota_exhausted``
  (exit code 4). A 429 with quota wording (RESOURCE_EXHAUSTED, insufficient_quota,
  "exceeded your current quota", ...) is quota exhaustion when it names a per-day
  limit or suggests a retry delay of more than ``QUOTA_MIN_DELAY_S``; when it names
  a per-minute limit or suggests a shorter delay it is transient (Gemini reports
  its per-minute limits with the same RESOURCE_EXHAUSTED as its daily quota).
  Quota wording with neither a period nor a delay (OpenAI insufficient_quota:
  billing exhausted) is quota exhaustion;
* any other API error (400, 401, 403, 404, ...) is not retried and ends the
  session as ``api_error`` (exit code 3), as does a transient error that is
  still there after the last retry.

Backends never raise: they return an :class:`LLMResponse` whose ``error_kind``
is set by :func:`classify_error`. Only successful answers are returned by
:func:`complete_with_retry`, so only they reach the token ledger.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from agentigrid.backends.base import LLMBackend, LLMResponse
    from agentigrid.config import RetryConfig

logger = logging.getLogger("agentigrid.backends.retry")

TRANSIENT, QUOTA, FATAL = "transient", "quota", "fatal"

# Termination reasons of a session ended by the LLM API, and the CLI exit codes.
API_ERROR, QUOTA_EXHAUSTED = "api_error", "quota_exhausted"
EXIT_CODES = {API_ERROR: 3, QUOTA_EXHAUSTED: 4}

_QUOTA_WORDS = ("resource_exhausted", "insufficient_quota", "exceeded your current quota",
                "quota exceeded", "daily limit", "per day", "usage limit", "billing")
# A 429 whose suggested retry delay is longer than this is quota exhaustion, not a rate limit.
QUOTA_MIN_DELAY_S = 300.0
_PER_DAY_WORDS = ("per day", "perday", "daily", "per-day")
_PER_MINUTE_WORDS = ("per minute", "perminute", "per-minute")
# "Please retry in 27.56563424s" / "retry in 1m30s" (message), 'retryDelay': '27s' (Google RetryInfo).
_DURATION = r"((?:[\d.]+\s*(?:ms|h|m|s)\s*)+|[\d.]+(?![\d.]))"
_RETRY_IN_RE = re.compile(r"retry in\s+" + _DURATION)
_RETRY_DELAY_RE = re.compile(r"retrydelay['\"]?\s*:\s*['\"]?" + _DURATION)
_PART_RE = re.compile(r"([\d.]+)\s*(ms|h|m|s)?")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, None: 1.0}


def _seconds(duration: str) -> float:
    """'1m30s' -> 90.0, '27.5s' -> 27.5, '20' -> 20.0."""
    return sum(float(n) * _UNIT_S[u or None] for n, u in _PART_RE.findall(duration) if n.strip("."))
# "429" as a word ("Error code: 429"), not inside an id such as "req_a4290f".
_HTTP_429_RE = re.compile(r"\b429\b")
_TRANSIENT_WORDS = ("timed out", "timeout", "connection error", "connection refused",
                    "connection reset", "cannot connect", "temporarily unavailable",
                    "server disconnected", "remote end closed")


def _status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def retry_after_s(exc: BaseException) -> Optional[float]:
    """Retry delay the provider suggests (message text, Google RetryInfo, or a
    Retry-After header), in seconds; None if it suggests none. The longest one wins."""
    text = str(exc).lower()
    found = [_seconds(m.group(1)) for r in (_RETRY_IN_RE, _RETRY_DELAY_RE) for m in r.finditer(text)]
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        header = headers.get("retry-after") if headers is not None else None
        if header is not None:
            found.append(float(header))
    except (TypeError, ValueError, AttributeError):
        pass
    return max(found) if found else None


def _quota_or_rate_limit(text: str, delay: Optional[float]) -> str:
    """QUOTA or TRANSIENT for a 429 that carries quota wording."""
    if any(w in text for w in _PER_DAY_WORDS) or (delay is not None and delay > QUOTA_MIN_DELAY_S):
        return QUOTA
    if any(w in text for w in _PER_MINUTE_WORDS) or delay is not None:
        return TRANSIENT
    return QUOTA            # quota/billing wording without a period or delay (insufficient_quota)


def classify_error(exc: BaseException) -> str:
    """TRANSIENT, QUOTA or FATAL for an exception raised by an LLM SDK."""
    text = str(exc).lower()
    status = _status(exc)
    names = " ".join(c.__name__.lower() for c in type(exc).__mro__)
    if (status == 429 or _HTTP_429_RE.search(text)) and any(w in text for w in _QUOTA_WORDS):
        return _quota_or_rate_limit(text, retry_after_s(exc))
    if status is not None and (status >= 500 or status in (408, 429)):
        return TRANSIENT
    if status is None and ("timeout" in names or "connect" in names
                           or isinstance(exc, (TimeoutError, ConnectionError))
                           or any(w in text for w in _TRANSIENT_WORDS)):
        return TRANSIENT
    return FATAL


# Exceptions from these packages (and the builtin network errors) come from the provider
# or the transport; anything else raised inside a backend's try block is a bug in our code.
_SDK_MODULES = ("openai", "anthropic", "ollama", "httpx", "httpcore")


def is_provider_error(exc: BaseException) -> bool:
    """True for an exception from a provider SDK or the HTTP transport (or a builtin
    TimeoutError / ConnectionError); False for anything else, e.g. a parsing bug."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any((c.__module__ or "").split(".")[0] in _SDK_MODULES for c in type(exc).__mro__)


def error_fields(exc: BaseException) -> dict[str, Any]:
    """The LLMResponse fields a backend sets for a failed call."""
    return {"error_kind": classify_error(exc), "retry_after_s": retry_after_s(exc)}


class LLMCallFailed(Exception):
    """A terminal LLM API error: the session must end with *termination*."""

    def __init__(self, termination: str, message: str, attempts: int, error_kind: str):
        super().__init__(message)
        self.termination = termination
        self.message = message
        self.attempts = attempts
        self.error_kind = error_kind

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.termination]


def complete_with_retry(
    backend: "LLMBackend",
    system_prompt: str,
    user_prompt: str,
    retry: "RetryConfig",
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> "LLMResponse":
    """One LLM call with the retry policy above.

    Returns the first successful response; raises :class:`LLMCallFailed` for a
    terminal error. *on_event* receives one dict per failed attempt
    (``attempt``, ``error_kind``, ``error``, ``wait_s`` (None when not retried;
    the configured backoff or the provider's suggested delay, whichever is longer),
    ``suggested_delay_s``, ``outcome`` = "retry" / "api_error" / "quota_exhausted", ``time``).
    """
    attempt = 0
    while True:
        attempt += 1
        response = backend.complete(system_prompt, user_prompt)
        kind = getattr(response, "error_kind", None)
        if kind is None:
            return response
        message = (response.raw_text or "").strip()
        retries_done = attempt - 1
        if kind == TRANSIENT and retries_done < retry.max_retries:
            wait = max(retry.wait_s(retries_done), getattr(response, "retry_after_s", None) or 0.0)
            outcome = "retry"
        else:
            wait = None
            outcome = QUOTA_EXHAUSTED if kind == QUOTA else API_ERROR
        event = {"attempt": attempt, "error_kind": kind, "error": message[:1000],
                 "wait_s": wait, "suggested_delay_s": getattr(response, "retry_after_s", None),
                 "outcome": outcome,
                 "time": datetime.now().isoformat(timespec="seconds")}
        if on_event is not None:
            on_event(event)
        if outcome != "retry":
            logger.error("LLM call failed for good after %d attempt(s) (%s): %s",
                         attempt, kind, message[:300])
            raise LLMCallFailed(outcome, message, attempt, kind)
        logger.warning("LLM call attempt %d failed (%s); retry %d of %d in %g s: %s",
                       attempt, kind, retries_done + 1, retry.max_retries, wait, message[:300])
        sleep(wait)
