"""Prompt caching (Anthropic) and per-run token metering.

Caching must be billing-only: the model sees the same system text either way,
``prompt_tokens`` keeps meaning TOTAL input tokens, and the cache breakdown is
reported separately. The UsageMeter must count every call without altering it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentigrid.backends import LLMBackend, LLMResponse, UsageMeter
from agentigrid.config import LLMConfig, load_config

try:
    import anthropic as _anthropic  # noqa: F401
    _has_anthropic = True
except ImportError:
    _has_anthropic = False


def _cfg(**over) -> LLMConfig:
    base = dict(
        backend="anthropic", model="claude-sonnet-4-6", api_key_env="TEST_API_KEY",
        openai_base_url=None, ollama_host="http://localhost:11434",
        ollama_cloud_host=None, temperature=0.3, max_tokens=256,
    )
    base.update(over)
    return LLMConfig(**base)


def _fake_message(input_tokens, creation, read, output_tokens=7, text='{"action":"complete"}'):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_creation_input_tokens=creation,
        cache_read_input_tokens=read,
        output_tokens=output_tokens,
    )
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_prompt_cache_defaults_on():
    cfg = load_config(None)
    assert cfg.llm.prompt_cache is True


def test_prompt_cache_can_be_disabled_by_override():
    cfg = load_config(None, cli_overrides={"llm.prompt_cache": False})
    assert cfg.llm.prompt_cache is False


# ---------------------------------------------------------------------------
# Anthropic backend request shape + token accounting
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_anthropic, reason="anthropic package not installed")
@patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
def test_system_prompt_sent_as_cached_block():
    from agentigrid.backends.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend(_cfg(prompt_cache=True))
    with patch.object(backend._client.messages, "create",
                      return_value=_fake_message(50, 4000, 0)) as create:
        backend.complete("SYSTEM TEXT", "user text")
    system = create.call_args.kwargs["system"]
    assert system == [{"type": "text", "text": "SYSTEM TEXT",
                       "cache_control": {"type": "ephemeral"}}]
    # the user turn is untouched
    assert create.call_args.kwargs["messages"] == [{"role": "user", "content": "user text"}]


@pytest.mark.skipif(not _has_anthropic, reason="anthropic package not installed")
@patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
def test_cache_off_sends_plain_string():
    from agentigrid.backends.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend(_cfg(prompt_cache=False))
    with patch.object(backend._client.messages, "create",
                      return_value=_fake_message(4050, None, None)) as create:
        resp = backend.complete("SYSTEM TEXT", "user text")
    assert create.call_args.kwargs["system"] == "SYSTEM TEXT"
    assert resp.prompt_tokens == 4050
    assert resp.cache_creation_tokens is None and resp.cache_read_tokens is None


@pytest.mark.skipif(not _has_anthropic, reason="anthropic package not installed")
@patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
def test_prompt_tokens_is_total_input_with_breakdown():
    from agentigrid.backends.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend(_cfg(prompt_cache=True))
    # first call writes the cache, second reads it
    with patch.object(backend._client.messages, "create",
                      side_effect=[_fake_message(50, 4000, 0), _fake_message(60, 0, 4000)]):
        first = backend.complete("S", "u1")
        second = backend.complete("S", "u2")
    assert (first.prompt_tokens, first.cache_creation_tokens, first.cache_read_tokens) == (4050, 4000, 0)
    assert (second.prompt_tokens, second.cache_creation_tokens, second.cache_read_tokens) == (4060, 0, 4000)
    assert first.json_data == {"action": "complete"}


# ---------------------------------------------------------------------------
# UsageMeter
# ---------------------------------------------------------------------------

class _Stub(LLMBackend):
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen = []

    def complete(self, system_prompt, user_prompt, temperature=None):
        self.seen.append((system_prompt, user_prompt, temperature))
        return self._responses.pop(0)

    def name(self):
        return "stub"

    def supports_json_mode(self):
        return False

    def custom(self):
        return "delegated"


def _resp(pt, ct, cw=None, cr=None):
    return LLMResponse(raw_text="x", json_data=None, json_error=None, model="m",
                       backend="stub", prompt_tokens=pt, completion_tokens=ct,
                       cache_creation_tokens=cw, cache_read_tokens=cr)


def test_usage_meter_counts_all_calls_and_is_transparent():
    inner = _Stub([_resp(4050, 10, 4000, 0), _resp(4060, 12, 0, 4000), _resp(None, None)])
    meter = UsageMeter(inner)
    r1 = meter.complete("S", "u1")
    meter.complete("S", "u2", temperature=0.0)
    meter.complete("S", "u3")
    assert r1 is not None and r1.prompt_tokens == 4050
    assert inner.seen[1] == ("S", "u2", 0.0)            # args passed through verbatim
    assert meter.totals() == {
        "calls": 3, "prompt_tokens": 8110, "completion_tokens": 22,
        "cache_creation_tokens": 4000, "cache_read_tokens": 4000,
        "api_errors": 0, "truncated": 0, "temperature_sent": None,
    }
    assert meter.name() == "stub"
    assert meter.custom() == "delegated"                 # backend-specific attrs delegate


def test_journal_exports_llm_usage(tmp_path):
    import json
    from agentigrid.engine.journal import SearchJournal

    j = SearchJournal()
    j.llm_usage = {"calls": 2, "prompt_tokens": 100, "completion_tokens": 5,
                   "cache_creation_tokens": 80, "cache_read_tokens": 0}
    out = tmp_path / "j.json"
    j.export_json(out)
    assert json.loads(out.read_text())["llm_usage"]["cache_creation_tokens"] == 80


def test_temperature_rejected_retries_without_it_and_remembers():
    from agentigrid.backends.anthropic_backend import AnthropicBackend
    backend = AnthropicBackend(_cfg(model="claude-sonnet-5"))
    calls = []

    def create(**kw):
        calls.append(kw)
        if "temperature" in kw:
            raise RuntimeError("Error code: 400 - `temperature` is deprecated for this model.")
        return _fake_message(10, 0, 0)

    with patch.object(backend._client.messages, "create", side_effect=create):
        r1 = backend.complete("sys", "u1")
        r2 = backend.complete("sys", "u2")
    assert not r1.api_error and r1.json_data == {"action": "complete"}
    assert ["temperature" in c for c in calls] == [True, False, False]   # retried once, then omitted
    assert backend.temperature_sent is False and r2.prompt_tokens == 10


@pytest.mark.skipif(not _has_anthropic, reason="anthropic not installed")
@patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
def test_other_api_errors_are_flagged_and_counted():
    from agentigrid.backends.anthropic_backend import AnthropicBackend
    backend = AnthropicBackend(_cfg())
    meter = UsageMeter(backend)
    with patch.object(backend._client.messages, "create", side_effect=RuntimeError("401 invalid x-api-key")):
        r = meter.complete("sys", "u")
    assert r.api_error and r.json_data is None
    t = meter.totals()
    assert t["calls"] == 1 and t["api_errors"] == 1 and t["prompt_tokens"] == 0
    with patch.object(backend._client.messages, "create", return_value=_fake_message(5, 0, 0)):
        meter.complete("sys", "u")
    assert meter.totals()["temperature_sent"] is True


# ---------------------------------------------------------------------------
# Output-token cap: stop_reason telemetry and AGENTIGRID_MAX_TOKENS
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_anthropic, reason="anthropic not installed")
@patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
def test_max_tokens_stop_is_reported_and_counted():
    from agentigrid.backends.anthropic_backend import AnthropicBackend
    backend = AnthropicBackend(_cfg())
    meter = UsageMeter(backend)
    msg = _fake_message(10, 0, 0, output_tokens=256, text="")
    msg.content = [SimpleNamespace(type="thinking", thinking="...")]
    msg.stop_reason = "max_tokens"
    with patch.object(backend._client.messages, "create", return_value=msg):
        r = meter.complete("sys", "u")
    assert r.stop_reason == "max_tokens" and r.content_types == ["thinking"]
    assert r.raw_text == "" and r.json_data is None and not r.api_error
    assert meter.totals()["truncated"] == 1


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("AGENTIGRID_MAX_TOKENS", "16000")
    assert load_config(None).llm.max_tokens == 16000
    monkeypatch.setenv("AGENTIGRID_MAX_TOKENS", "lots")
    assert load_config(None).llm.max_tokens == 4096     # ignored, default kept
    monkeypatch.delenv("AGENTIGRID_MAX_TOKENS")
    assert load_config(None).llm.max_tokens == 4096
