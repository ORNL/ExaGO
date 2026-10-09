"""Tests for the LLM backend abstraction layer."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.backends import LLMBackend, LLMResponse, create_backend
from agentigrid.backends.json_extract import extract_json
from agentigrid.config import LLMConfig

# Check for optional dependencies
try:
    import openai as _openai
    _has_openai_pkg = True
except ImportError:
    _has_openai_pkg = False

try:
    import anthropic as _anthropic
    _has_anthropic_pkg = True
except ImportError:
    _has_anthropic_pkg = False

try:
    import ollama as _ollama
    _has_ollama_pkg = True
except ImportError:
    _has_ollama_pkg = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_config(**overrides) -> LLMConfig:
    """Create a LLMConfig with sensible defaults for testing."""
    defaults = dict(
        backend="openai",
        model="test-model",
        api_key_env="TEST_API_KEY",
        openai_base_url=None,
        ollama_host="http://localhost:11434",
        ollama_cloud_host=None,
        temperature=0.3,
        max_tokens=1024,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ===========================================================================
# extract_json tests
# ===========================================================================

class TestExtractJson:
    """Tests for the JSON extraction utility."""

    def test_clean_json(self):
        text = '{"action": "modify", "value": 42}'
        data, err = extract_json(text)
        assert data == {"action": "modify", "value": 42}
        assert err is None

    def test_json_in_markdown_fences(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        data, err = extract_json(text)
        assert data == {"key": "value"}
        assert err is None

    def test_json_in_plain_fences(self):
        text = '```\n{"a": 1}\n```'
        data, err = extract_json(text)
        assert data == {"a": 1}
        assert err is None

    def test_json_with_surrounding_text(self):
        text = 'I will modify the bus data.\n{"bus": 1, "change": "Pd"}\nThis should work.'
        data, err = extract_json(text)
        assert data == {"bus": 1, "change": "Pd"}
        assert err is None

    def test_json_with_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        data, err = extract_json(text)
        assert data == {"a": 1, "b": 2}
        assert err is None

    def test_invalid_json(self):
        text = "This is just plain text with no JSON at all."
        data, err = extract_json(text)
        assert data is None
        assert err is not None

    def test_empty_input(self):
        data, err = extract_json("")
        assert data is None
        assert err is not None

    def test_multiple_json_objects_extracts_first(self):
        text = 'First: {"order": 1} Second: {"order": 2}'
        data, err = extract_json(text)
        assert data is not None
        assert data["order"] == 1

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}}'
        data, err = extract_json(text)
        assert data == {"outer": {"inner": 42}}
        assert err is None

    def test_trailing_comma_in_fences(self):
        text = '```json\n{"x": 1, "y": 2,}\n```'
        data, err = extract_json(text)
        assert data == {"x": 1, "y": 2}
        assert err is None


# ===========================================================================
# Backend instantiation tests (no real API calls)
# ===========================================================================

class TestBackendInstantiation:
    """Test that backends can be instantiated without crashing."""

    @pytest.mark.skipif(not _has_openai_pkg, reason="openai package not installed")
    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
    def test_openai_backend_creates(self):
        from agentigrid.backends.openai_backend import OpenAIBackend

        cfg = _dummy_config(backend="openai")
        backend = OpenAIBackend(cfg)
        assert backend.name() == "openai"
        assert backend.supports_json_mode() is True

    @pytest.mark.skipif(not _has_anthropic_pkg, reason="anthropic package not installed")
    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
    def test_anthropic_backend_creates(self):
        from agentigrid.backends.anthropic_backend import AnthropicBackend

        cfg = _dummy_config(backend="anthropic")
        backend = AnthropicBackend(cfg)
        assert backend.name() == "anthropic"
        assert backend.supports_json_mode() is False

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_ollama_backend_creates(self):
        from agentigrid.backends.ollama_backend import OllamaBackend

        cfg = _dummy_config(backend="ollama")
        backend = OllamaBackend(cfg)
        assert backend.name() == "ollama"

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_ollama_cloud_backend_creates(self):
        from agentigrid.backends.ollama_backend import OllamaBackend

        cfg = _dummy_config(backend="ollama-cloud", ollama_cloud_host="https://cloud.example.com")
        backend = OllamaBackend(cfg)
        assert backend.name() == "ollama-cloud"

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_ollama_json_mode_qwen(self):
        from agentigrid.backends.ollama_backend import OllamaBackend

        cfg = _dummy_config(model="qwen2.5:7b")
        backend = OllamaBackend(cfg)
        assert backend.supports_json_mode() is True

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_ollama_json_mode_llama(self):
        from agentigrid.backends.ollama_backend import OllamaBackend

        cfg = _dummy_config(model="llama3.1:8b")
        backend = OllamaBackend(cfg)
        assert backend.supports_json_mode() is False


# ===========================================================================
# Factory function tests
# ===========================================================================

class TestCreateBackend:
    """Test the create_backend factory function."""

    @pytest.mark.skipif(not _has_openai_pkg, reason="openai package not installed")
    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
    def test_create_openai(self):
        cfg = _dummy_config(backend="openai")
        backend = create_backend(cfg)
        assert backend.name() == "openai"

    @pytest.mark.skipif(not _has_anthropic_pkg, reason="anthropic package not installed")
    @patch.dict(os.environ, {"TEST_API_KEY": "sk-test"})
    def test_create_anthropic(self):
        cfg = _dummy_config(backend="anthropic")
        backend = create_backend(cfg)
        assert backend.name() == "anthropic"

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_create_ollama(self):
        cfg = _dummy_config(backend="ollama")
        backend = create_backend(cfg)
        assert backend.name() == "ollama"

    @pytest.mark.skipif(not _has_ollama_pkg, reason="ollama package not installed")
    def test_create_ollama_cloud(self):
        cfg = _dummy_config(
            backend="ollama-cloud",
            ollama_cloud_host="https://cloud.example.com",
        )
        backend = create_backend(cfg)
        assert backend.name() == "ollama-cloud"

    def test_invalid_backend_raises(self):
        cfg = _dummy_config(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_backend(cfg)


# ===========================================================================
# Live integration tests (skipped if API keys not present)
# ===========================================================================

_has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicLive:
    def test_simple_prompt(self):
        cfg = _dummy_config(
            backend="anthropic",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
        )
        backend = create_backend(cfg)
        resp = backend.complete(
            system_prompt="You are a helpful assistant. Reply with a JSON object.",
            user_prompt='Return exactly: {"status": "ok"}',
        )
        assert isinstance(resp, LLMResponse)
        assert resp.raw_text
        assert resp.backend == "anthropic"


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestOpenAILive:
    def test_simple_prompt(self):
        cfg = _dummy_config(
            backend="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        )
        backend = create_backend(cfg)
        resp = backend.complete(
            system_prompt="You are a helpful assistant. Reply with a JSON object.",
            user_prompt='Return exactly: {"status": "ok"}',
        )
        assert isinstance(resp, LLMResponse)
        assert resp.raw_text
        assert resp.backend == "openai"
