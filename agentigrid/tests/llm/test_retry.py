"""LLM API errors: classification, retry with backoff, terminal outcomes (backends/retry.py)."""

from __future__ import annotations

import pytest

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.backends.retry import (
    FATAL, QUOTA, TRANSIENT, LLMCallFailed, classify_error, complete_with_retry,
)
from agentigrid.config import RetryConfig, load_config
from agentigrid.llm_settings import collect_llm_settings, format_llm_settings


class _StatusError(Exception):
    """Shaped like the SDKs' HTTP errors (openai/anthropic APIStatusError, ollama ResponseError)."""

    def __init__(self, status, message):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


GEMINI_503 = "{'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}"
GEMINI_429_QUOTA = ("{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan "
                    "and billing details.', 'status': 'RESOURCE_EXHAUSTED'}}")
OPENAI_429_RATE = "{'error': {'message': 'Rate limit reached for requests per min', 'type': 'requests', 'code': 'rate_limit_exceeded'}}"
OPENAI_429_QUOTA = "{'error': {'message': 'You exceeded your current quota', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}}"


@pytest.mark.parametrize("exc,kind", [
    (_StatusError(503, GEMINI_503), TRANSIENT),
    (_StatusError(500, "internal"), TRANSIENT),
    (_StatusError(502, "bad gateway"), TRANSIENT),
    (_StatusError(408, "request timeout"), TRANSIENT),
    (_StatusError(429, OPENAI_429_RATE), TRANSIENT),
    (_StatusError(429, GEMINI_429_QUOTA), QUOTA),
    (_StatusError(429, OPENAI_429_QUOTA), QUOTA),
    (_StatusError(429, "daily limit reached for this model"), QUOTA),
    (APITimeoutError("Request timed out."), TRANSIENT),
    (APIConnectionError("Connection error."), TRANSIENT),
    (ConnectionRefusedError("[Errno 111] Connection refused"), TRANSIENT),
    (TimeoutError("read timed out"), TRANSIENT),
    (_StatusError(400, "Unsupported value: 'temperature' does not support 0.3"), FATAL),
    (_StatusError(401, "invalid api key"), FATAL),
    (_StatusError(404, "model not found"), FATAL),
    (ValueError("something else"), FATAL),
])
def test_classify(exc, kind):
    assert classify_error(exc) == kind


class FakeBackend(LLMBackend):
    """Replies from a script: an exception instance is turned into an error response
    exactly as the real backends do; anything else is a JSON answer."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, system_prompt, user_prompt, temperature=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            msg = f"OpenAI API error: {item}"
            return LLMResponse(msg, None, msg, "fake", "fake", None, None, error_kind=classify_error(item))
        return LLMResponse('{"ok": true}', {"ok": True}, None, "fake", "fake", 10, 2)

    def name(self):
        return "fake"

    def supports_json_mode(self):
        return True


RETRY = RetryConfig(max_retries=5, backoff_s=(10, 30, 60, 120, 240))


def _run(script, retry=RETRY):
    backend, events, waits = FakeBackend(script), [], []
    try:
        resp = complete_with_retry(backend, "s", "u", retry, on_event=events.append, sleep=waits.append)
    except LLMCallFailed as exc:
        return backend, events, waits, exc
    return backend, events, waits, resp


def test_success_first_try():
    backend, events, waits, resp = _run(["ok"])
    assert resp.json_data == {"ok": True} and backend.calls == 1 and not events and not waits


def test_transient_then_success():
    backend, events, waits, resp = _run([_StatusError(503, GEMINI_503), APITimeoutError("Request timed out."), "ok"])
    assert isinstance(resp, LLMResponse) and resp.prompt_tokens == 10
    assert backend.calls == 3 and waits == [10, 30]
    assert [e["outcome"] for e in events] == ["retry", "retry"]
    assert [e["attempt"] for e in events] == [1, 2] and [e["wait_s"] for e in events] == [10, 30]
    assert "503" in events[0]["error"] and events[0]["error_kind"] == TRANSIENT


def test_retries_exhausted_api_error():
    backend, events, waits, exc = _run([_StatusError(503, GEMINI_503)] * 6)
    assert isinstance(exc, LLMCallFailed)
    assert exc.termination == "api_error" and exc.exit_code == 3 and exc.attempts == 6
    assert backend.calls == 6 and waits == [10, 30, 60, 120, 240]
    assert [e["outcome"] for e in events] == ["retry"] * 5 + ["api_error"]
    assert events[-1]["wait_s"] is None


def test_quota_not_retried():
    backend, events, waits, exc = _run([_StatusError(429, GEMINI_429_QUOTA), "ok"])
    assert exc.termination == "quota_exhausted" and exc.exit_code == 4
    assert backend.calls == 1 and waits == [] and events[0]["outcome"] == "quota_exhausted"


def test_quota_after_transient():
    backend, events, waits, exc = _run([_StatusError(503, GEMINI_503), _StatusError(429, GEMINI_429_QUOTA)])
    assert exc.termination == "quota_exhausted" and backend.calls == 2 and waits == [10]


def test_fatal_not_retried():
    backend, events, waits, exc = _run([_StatusError(401, "invalid api key")])
    assert exc.termination == "api_error" and backend.calls == 1 and waits == []


def test_short_backoff_list_repeats_last_value():
    retry = RetryConfig(max_retries=3, backoff_s=(5,))
    _, _, waits, exc = _run([_StatusError(500, "x")] * 4, retry)
    assert waits == [5, 5, 5] and exc.termination == "api_error"


def test_zero_retries():
    _, _, waits, exc = _run([_StatusError(500, "x")], RetryConfig(max_retries=0))
    assert waits == [] and exc.attempts == 1


def test_config_and_settings_record(tmp_path):
    cfg = load_config(None)
    assert cfg.llm.retry == RETRY
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  retry:\n    max_retries: 2\n    backoff_s: [1, 2]\n")
    cfg = load_config(p)
    assert cfg.llm.retry == RetryConfig(2, (1.0, 2.0))
    s = collect_llm_settings(cfg.llm, None)
    assert s["retry"] == {"max_retries": 2, "backoff_s": [1.0, 2.0]}
    assert "retries 2 (1/2 s)" in format_llm_settings(s)
