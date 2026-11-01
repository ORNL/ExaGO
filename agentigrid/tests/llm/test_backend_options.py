"""LLM backend options: reasoning effort, optional temperature, usage details, run settings.

All backends are exercised with mocked clients; no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.backends.base import LLMResponse, derive_reasoning_tokens
from agentigrid.cli import _coerce_override_value
from agentigrid.config import DEFAULTS, LLMConfig, load_config
from agentigrid.llm_settings import collect_llm_settings, format_llm_settings
from agentigrid.token_usage import TokenLedger

SECRET = "sk-test-SECRET-do-not-leak-0123456789"


def _cfg(**overrides) -> LLMConfig:
    base = dict(backend="openai", model="gpt-test", api_key_env="TEST_API_KEY",
                openai_base_url=None, ollama_host="http://localhost:11434",
                ollama_cloud_host=None, temperature=0.3, max_tokens=1024)
    base.update(overrides)
    return LLMConfig(**base)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        cfg = load_config(None)
        assert cfg.llm.temperature == 0.3          # unchanged default
        assert cfg.llm.reasoning_effort is None
        assert "reasoning_effort" in DEFAULTS["llm"]

    def test_yaml(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("llm:\n  temperature: null\n  reasoning_effort: high\n")
        cfg = load_config(p)
        assert cfg.llm.temperature is None
        assert cfg.llm.reasoning_effort == "high"

    def test_yaml_without_new_keys(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("llm:\n  backend: openai\n  model: m\n")
        cfg = load_config(p)
        assert cfg.llm.temperature == 0.3 and cfg.llm.reasoning_effort is None

    def test_set_overrides(self):
        cfg = load_config(None, cli_overrides={
            "llm.reasoning_effort": _coerce_override_value("medium"),
            "llm.temperature": _coerce_override_value("null"),
        })
        assert cfg.llm.reasoning_effort == "medium"
        assert cfg.llm.temperature is None

    def test_set_null_only(self):
        assert _coerce_override_value("null") is None
        assert _coerce_override_value("none") == "none"     # e.g. exago.save_solutions=none
        assert _coerce_override_value("0.5") == 0.5


# ---------------------------------------------------------------------------
# OpenAI backend (also Gemini through the compatible endpoint)
# ---------------------------------------------------------------------------

def _openai_backend(monkeypatch, usage, **cfg_overrides):
    pytest.importorskip("openai")
    from agentigrid.backends.openai_backend import OpenAIBackend

    monkeypatch.setenv("TEST_API_KEY", SECRET)
    client = MagicMock()
    client.base_url = cfg_overrides.get("openai_base_url") or "https://api.openai.com/v1/"
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=usage,
    )
    with patch("agentigrid.backends.openai_backend.openai.OpenAI", return_value=client):
        backend = OpenAIBackend(_cfg(**cfg_overrides))
    return backend, client.chat.completions.create


def _sent(create) -> dict:
    return create.call_args.kwargs


class TestOpenAIBackend:

    def test_defaults_unchanged(self, monkeypatch):
        backend, create = _openai_backend(
            monkeypatch, SimpleNamespace(prompt_tokens=10, completion_tokens=4))
        resp = backend.complete("Reply in JSON.", "hi")
        kw = _sent(create)
        assert kw["temperature"] == 0.3
        assert "reasoning_effort" not in kw
        assert kw["response_format"] == {"type": "json_object"}
        assert resp.reasoning_tokens is None and resp.api_total_tokens is None
        assert resp.reasoning_tokens_derived is False

    def test_reasoning_effort_and_no_temperature(self, monkeypatch):
        backend, create = _openai_backend(
            monkeypatch, SimpleNamespace(prompt_tokens=10, completion_tokens=4),
            reasoning_effort="medium", temperature=None)
        backend.complete("sys", "hi")
        kw = _sent(create)
        assert kw["reasoning_effort"] == "medium"
        assert "temperature" not in kw
        assert backend.reasoning_param() == {"reasoning_effort": "medium"}

    def test_gemini_style_usage_derives_reasoning(self, monkeypatch):
        usage = SimpleNamespace(completion_tokens=5, prompt_tokens=15, total_tokens=65)
        backend, _ = _openai_backend(
            monkeypatch, usage, model="gemini-3.8-flash",
            openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        resp = backend.complete("sys", "hi")
        assert resp.api_total_tokens == 65
        assert resp.reasoning_tokens == 45
        assert resp.reasoning_tokens_derived is True
        ledger = TokenLedger()
        ledger.record("search", resp, 1)
        assert ledger.calls[0]["billable_output_tokens"] == 50
        assert ledger.calls_with_total_mismatch == 1

    def test_openai_style_usage_reports_reasoning(self, monkeypatch):
        usage = SimpleNamespace(
            prompt_tokens=100, completion_tokens=40, total_tokens=140,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        )
        backend, _ = _openai_backend(monkeypatch, usage)
        resp = backend.complete("sys", "hi")
        assert resp.reasoning_tokens == 30
        assert resp.reasoning_tokens_derived is False
        assert resp.api_total_tokens == 140
        ledger = TokenLedger()
        ledger.record("search", resp, 1)
        assert ledger.calls[0]["billable_output_tokens"] == 40
        assert ledger.calls_with_total_mismatch == 0

    def test_settings_have_no_key_material(self, monkeypatch):
        backend, _ = _openai_backend(
            monkeypatch, SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            reasoning_effort="low")
        settings = collect_llm_settings(backend._config, backend)
        text = json.dumps(settings)
        assert SECRET not in text and "TEST_API_KEY" not in text
        assert settings["endpoint_host"] == "generativelanguage.googleapis.com"
        assert settings["reasoning_param_sent"] == {"reasoning_effort": "low"}
        assert settings["temperature"] == 0.3
        assert settings["packages"]["openai"]
        assert settings["ollama_model_digest"] is None
        assert SECRET not in format_llm_settings(settings)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class _FakeOllamaClient:
    """Mimics ollama.Client (0.6.1 chat signature, incl. think) and records calls."""

    def __init__(self, *, listing=None, list_error=None, **kwargs):
        self.calls: list[dict] = []
        self._listing = listing
        self._list_error = list_error

    def chat(self, model="", messages=None, tools=None, stream=False, think=None,
             format=None, options=None, keep_alive=None):
        sent = {"model": model, "messages": messages, "format": format, "options": options}
        if think is not None:
            sent["think"] = think
        self.calls.append(sent)
        return {"message": {"content": '{"ok": true}'}, "prompt_eval_count": 12, "eval_count": 7}

    def list(self):
        if self._list_error:
            raise self._list_error
        return self._listing


class _OldOllamaClient(_FakeOllamaClient):
    def chat(self, model="", messages=None, format=None, options=None):   # no think
        self.calls.append({"model": model, "options": options})
        return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}


def _ollama_backend(client, **cfg_overrides):
    pytest.importorskip("ollama")
    from agentigrid.backends.ollama_backend import OllamaBackend

    cfg = _cfg(backend="ollama", model="glm-5.3:cloud", api_key_env="UNUSED", **cfg_overrides)
    with patch("agentigrid.backends.ollama_backend.ollama.Client", return_value=client):
        return OllamaBackend(cfg)


class TestOllamaBackend:

    def test_defaults_unchanged(self):
        client = _FakeOllamaClient()
        backend = _ollama_backend(client)
        resp = backend.complete("sys", "hi")
        assert client.calls[0]["options"] == {"num_predict": 1024, "temperature": 0.3}
        assert "think" not in client.calls[0]
        assert resp.prompt_tokens == 12 and resp.completion_tokens == 7
        assert resp.reasoning_tokens is None

    def test_think_sent_when_set(self):
        client = _FakeOllamaClient()
        _ollama_backend(client, reasoning_effort="high").complete("sys", "hi")
        assert client.calls[0]["think"] == "high"

    def test_think_none_maps_to_false(self):
        client = _FakeOllamaClient()
        backend = _ollama_backend(client, reasoning_effort="none")
        backend.complete("sys", "hi")
        assert client.calls[0]["think"] is False
        assert backend.reasoning_param() == {"think": False}

    def test_temperature_omitted(self):
        client = _FakeOllamaClient()
        _ollama_backend(client, temperature=None).complete("sys", "hi")
        assert client.calls[0]["options"] == {"num_predict": 1024}

    def test_client_without_think(self, caplog):
        client = _OldOllamaClient()
        backend = _ollama_backend(client, reasoning_effort="medium")
        backend.complete("sys", "hi")
        assert "think" not in client.calls[0]
        assert backend.reasoning_param() == {}
        assert "no 'think' argument" in caplog.text

    def test_digest_from_listing(self):
        listing = SimpleNamespace(models=[
            SimpleNamespace(model="other:latest", digest="aaa"),
            SimpleNamespace(model="glm-5.3:cloud", digest="sha256:abc123"),
        ])
        backend = _ollama_backend(_FakeOllamaClient(listing=listing))
        assert backend.model_digest() == "sha256:abc123"
        assert collect_llm_settings(backend._config, backend)["ollama_model_digest"] == "sha256:abc123"

    def test_digest_untagged_name_matches_latest(self):
        listing = {"models": [{"model": "qwen3:latest", "digest": "d1"}]}
        backend = _ollama_backend(_FakeOllamaClient(listing=listing))
        backend._config = _cfg(backend="ollama", model="qwen3")
        assert backend.model_digest() == "d1"

    def test_digest_unavailable_is_none(self):
        backend = _ollama_backend(_FakeOllamaClient(list_error=ConnectionError("down")))
        assert backend.model_digest() is None
        settings = collect_llm_settings(backend._config, backend)
        assert settings["ollama_model_digest"] is None
        assert settings["endpoint_host"] == "localhost:11434"

    def test_digest_model_not_listed(self):
        backend = _ollama_backend(_FakeOllamaClient(listing=SimpleNamespace(models=[])))
        assert backend.model_digest() is None


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class TestAnthropicBackend:

    def _backend(self, monkeypatch, **cfg_overrides):
        pytest.importorskip("anthropic")
        from agentigrid.backends.anthropic_backend import AnthropicBackend

        monkeypatch.setenv("TEST_API_KEY", SECRET)
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )
        with patch("agentigrid.backends.anthropic_backend.anthropic.Anthropic", return_value=client):
            backend = AnthropicBackend(_cfg(backend="anthropic", **cfg_overrides))
        return backend, client.messages.create

    def test_temperature_sent_by_default(self, monkeypatch):
        backend, create = self._backend(monkeypatch)
        backend.complete("sys", "hi")
        assert create.call_args.kwargs["temperature"] == 0.3

    def test_temperature_omitted_and_effort_warned(self, monkeypatch, caplog):
        backend, create = self._backend(monkeypatch, temperature=None, reasoning_effort="high")
        backend.complete("sys", "hi")
        assert "temperature" not in create.call_args.kwargs
        assert "not supported by the Anthropic backend" in caplog.text
        assert backend.reasoning_param() == {}


# ---------------------------------------------------------------------------
# Derivation and ledger
# ---------------------------------------------------------------------------

def test_derive_reasoning_tokens_cases():
    assert derive_reasoning_tokens(15, 5, None, 65) == (45, True)        # Gemini
    assert derive_reasoning_tokens(100, 40, 30, 140) == (30, False)      # OpenAI (reported)
    assert derive_reasoning_tokens(10, 5, None, 15) == (None, False)     # totals agree
    assert derive_reasoning_tokens(10, None, None, 50) == (None, False)  # incomplete usage
    assert derive_reasoning_tokens(10, 5, None, None) == (None, False)


def _r(prompt, completion, reasoning=None, total=None, derived=False) -> LLMResponse:
    return LLMResponse(raw_text="x", json_data=None, json_error=None, model="m", backend="b",
                       prompt_tokens=prompt, completion_tokens=completion,
                       reasoning_tokens=reasoning, api_total_tokens=total,
                       reasoning_tokens_derived=derived)


def test_ledger_aggregates_new_fields():
    ledger = TokenLedger()
    ledger.record("search", _r(15, 5, 45, 65, True), 1)        # Gemini: +45 outside completion
    ledger.record("search", _r(100, 40, 30, 140), 2)           # OpenAI: 30 inside completion
    ledger.record("analysis", _r(10, None))                     # no completion: billable None
    ledger.record("analysis", _r(20, 10, total=31))            # mismatch without reasoning field

    assert [c["billable_output_tokens"] for c in ledger.calls] == [50, 40, None, 10]
    assert ledger.total_reasoning == 75
    assert ledger.total_billable_output == 100
    assert ledger.calls_with_total_mismatch == 2
    assert ledger.by_phase["search"]["reasoning_tokens"] == 75
    assert ledger.by_phase["search"]["billable_output_tokens"] == 90
    data = json.loads(json.dumps(ledger.to_dict()))
    assert data["billable_output_tokens"] == 100 and data["calls_with_total_mismatch"] == 2
    assert TokenLedger.from_dict(data).calls == ledger.calls
    # completion_tokens / total_tokens keep their meaning
    assert ledger.total_completion == 55 and ledger.total == 145 + 55


def test_summary_shows_billable_output_when_different():
    ledger = TokenLedger()
    ledger.record("search", _r(15, 5, 45, 65, True), 1)
    assert "5 completion (+ 45 reasoning = 50 billable output)" in ledger.summary_text()
    plain = TokenLedger()
    plain.record("search", _r(100, 40, 30, 140), 1)
    assert "billable" not in plain.summary_text()


def test_old_ledger_without_usage_details_loads():
    old = {"per_call": [{"phase": "search", "iteration": 1, "model": "m", "backend": "b",
                         "prompt_tokens": 10, "completion_tokens": 5,
                         "prompt_chars": 100, "completion_chars": 20}]}
    ledger = TokenLedger.from_dict(old)
    call = ledger.calls[0]
    assert call["reasoning_tokens"] is None and call["api_total_tokens"] is None
    assert call["reasoning_tokens_derived"] is False
    assert call["billable_output_tokens"] == 5
    assert ledger.calls_with_total_mismatch == 0
