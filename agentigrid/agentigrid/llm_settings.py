"""The LLM settings a run used, for its journal (``llm_settings``) and start-up line.

Records what is needed to reproduce or compare runs: backend, model, endpoint
host, the temperature and reasoning parameter actually sent, max_tokens, the
client package versions and, for Ollama (whose tags are not pinned), the model
digest. API keys are never read or recorded.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Optional
from urllib.parse import urlparse

from agentigrid.config import LLMConfig

_PACKAGES = ("openai", "ollama", "anthropic")


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host(url: Any) -> Optional[str]:
    """``host[:port]`` of a URL (no scheme, path or credentials)."""
    if not isinstance(url, str) or not url:
        return None
    parsed = urlparse(url if "://" in url else f"//{url}")
    if not parsed.hostname:
        return None
    return f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname


def _plain(value: Any) -> Any:
    """Keep JSON-safe scalars (and dicts of them); anything else (e.g. a mock) is None."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return None


def _call(obj: Any, method: str) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a settings record must never fail a run
        return None


def collect_llm_settings(config: LLMConfig, backend: Any = None) -> dict[str, Any]:
    """Settings block for the journal; never raises, never contains key material."""
    reasoning_sent = _plain(_call(backend, "reasoning_param"))
    return {
        "backend": config.backend,
        "backend_name": _plain(_call(backend, "name")),
        "model": config.model,
        "endpoint_host": _host(_call(backend, "endpoint")),
        "temperature": config.temperature,          # the value sent; null = not sent
        "reasoning_effort": config.reasoning_effort,
        "reasoning_param_sent": reasoning_sent if isinstance(reasoning_sent, dict) else {},
        "max_tokens": config.max_tokens,
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "ollama_model_digest": _plain(_call(backend, "model_digest")),
    }


def format_llm_settings(settings: dict[str, Any]) -> str:
    """One line with the same information, for the start of a session."""
    temp = settings.get("temperature")
    sent = settings.get("reasoning_param_sent") or {}
    parts = [
        f"{settings.get('backend')} {settings.get('model')}"
        + (f" @ {settings['endpoint_host']}" if settings.get("endpoint_host") else ""),
        f"temperature {temp if temp is not None else 'not sent'}",
        "reasoning " + (", ".join(f"{k}={v}" for k, v in sent.items()) if sent else "not sent"),
        f"max_tokens {settings.get('max_tokens')}",
    ]
    versions = ", ".join(f"{k} {v}" for k, v in (settings.get("packages") or {}).items() if v)
    if versions:
        parts.append(versions)
    if settings.get("ollama_model_digest"):
        parts.append(f"digest {settings['ollama_model_digest']}")
    return "LLM: " + "; ".join(parts)
