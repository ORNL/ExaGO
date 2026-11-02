"""LLM API errors: classification, retry with backoff, and terminal outcomes.

Shared by the ExaGO and GridKit agent loops so both behave the same:

* transient errors (HTTP 5xx, 408, timeouts, connection errors, and 429 rate
  limits that are not quota exhaustion) are retried ``llm.retry.max_retries``
  times, waiting ``llm.retry.backoff_s[i]`` seconds before retry *i*;
* quota exhaustion (429 with RESOURCE_EXHAUSTED / insufficient_quota / quota or
  daily-limit wording) is not retried and ends the session as
  ``quota_exhausted`` (exit code 4);
* any other API error (400, 401, 403, 404, ...) is not retried and ends the
  session as ``api_error`` (exit code 3), as does a transient error that is
  still there after the last retry.

Backends never raise: they return an :class:`LLMResponse` whose ``error_kind``
is set by :func:`classify_error`. Only successful answers are returned by
:func:`complete_with_retry`, so only they reach the token ledger.
"""

from __future__ import annotations

import logging
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


def classify_error(exc: BaseException) -> str:
    """TRANSIENT, QUOTA or FATAL for an exception raised by an LLM SDK."""
    text = str(exc).lower()
    status = _status(exc)
    names = " ".join(c.__name__.lower() for c in type(exc).__mro__)
    if (status == 429 or "429" in text) and any(w in text for w in _QUOTA_WORDS):
        return QUOTA
    if status is not None and (status >= 500 or status in (408, 429)):
        return TRANSIENT
    if status is None and ("timeout" in names or "connect" in names
                           or isinstance(exc, (TimeoutError, ConnectionError))
                           or any(w in text for w in _TRANSIENT_WORDS)):
        return TRANSIENT
    return FATAL


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
    (``attempt``, ``error_kind``, ``error``, ``wait_s`` (None when not retried),
    ``outcome`` = "retry" / "api_error" / "quota_exhausted", ``time``).
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
            wait = retry.wait_s(retries_done)
            outcome = "retry"
        else:
            wait = None
            outcome = QUOTA_EXHAUSTED if kind == QUOTA else API_ERROR
        event = {"attempt": attempt, "error_kind": kind, "error": message[:1000],
                 "wait_s": wait, "outcome": outcome,
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
