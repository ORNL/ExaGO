"""Ollama (local and cloud) LLM backend adapter."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

from agentigrid.logging_setup import log_scope
from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.backends.json_extract import extract_json
from agentigrid.backends.retry import error_fields
from agentigrid.config import LLMConfig

logger = logging.getLogger("agentigrid.backends.ollama")

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]
    logger.warning("ollama package not installed — Ollama backend unavailable")


class OllamaBackend(LLMBackend):
    """Backend adapter for Ollama (local or cloud)."""

    def __init__(self, config: LLMConfig) -> None:
        if ollama is None:
            raise ImportError("ollama package is required for the Ollama backend")

        self._config = config

        # Determine which host to use
        if config.ollama_cloud_host:
            self._host = config.ollama_cloud_host
            self._backend_name = "ollama-cloud"
        else:
            self._host = config.ollama_host
            self._backend_name = "ollama"

        # For Ollama Cloud, pass API key as Bearer token if available
        import os
        api_key = os.environ.get("OLLAMA_API_KEY")
        client_kwargs: dict = {"host": self._host}
        if api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
            logger.info("Using OLLAMA_API_KEY for authentication")

        self._client = ollama.Client(**client_kwargs)

        # think= exists in newer ollama clients only (0.6.1: bool | "low"/"medium"/"high").
        self._think: Any = None
        if config.reasoning_effort is not None:
            try:
                supported = "think" in inspect.signature(self._client.chat).parameters
            except (TypeError, ValueError):
                supported = False
            if supported:
                effort = config.reasoning_effort
                self._think = False if str(effort).lower() == "none" else effort
            else:
                logger.warning(
                    "llm.reasoning_effort=%r is set but the installed ollama client has no "
                    "'think' argument; nothing is sent", config.reasoning_effort,
                )

    def name(self) -> str:
        """Return the backend name."""
        return self._backend_name

    def supports_json_mode(self) -> bool:
        """Some Ollama models support JSON mode (e.g., Qwen-based)."""
        model_lower = self._config.model.lower()
        return "qwen" in model_lower

    def endpoint(self) -> Optional[str]:
        """Ollama host the client talks to."""
        return self._host

    def reasoning_param(self) -> dict:
        """``think`` as sent (nothing when not configured or not supported)."""
        return {} if self._think is None else {"think": self._think}

    def model_digest(self) -> Optional[str]:
        """Digest of the configured model as in ``ollama list``; None if it cannot be read.

        Ollama tags are not pinned, so the digest identifies the exact model a run used.
        Never raises.
        """
        try:
            listing = self._client.list()
            models = listing.get("models", []) if isinstance(listing, dict) else listing.models
            wanted = self._config.model
            names = {wanted} if ":" in wanted else {wanted, f"{wanted}:latest"}
            for m in models:
                name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
                if name in names:
                    digest = m.get("digest") if isinstance(m, dict) else getattr(m, "digest", None)
                    return digest if isinstance(digest, str) else None
        except Exception as exc:  # noqa: BLE001 - a settings record must never fail a run
            logger.warning("Could not read the Ollama model digest for %s: %s",
                           self._config.model, exc)
        return None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to the Ollama API."""
        temp = temperature if temperature is not None else self._config.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        options: dict = {"num_predict": self._config.max_tokens}
        if temp is not None:                     # temperature: null -> not sent
            options["temperature"] = temp

        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "options": options,
        }

        if self.supports_json_mode():
            kwargs["format"] = "json"
        kwargs.update(self.reasoning_param())

        try:
            with log_scope(logger, "LLM API request", backend=self.name(), model=self._config.model):
                response = self._client.chat(**kwargs)

            with log_scope(logger, "LLM response parsing", backend=self.name(), model=self._config.model):
                raw_text = response.get("message", {}).get("content", "")

                prompt_tokens = response.get("prompt_eval_count")
                completion_tokens = response.get("eval_count")

                json_data, json_error = extract_json(raw_text)

            return LLMResponse(
                raw_text=raw_text,
                json_data=json_data,
                json_error=json_error,
                model=self._config.model,
                backend=self.name(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except Exception as exc:
            msg = str(exc)
            if "connect" in msg.lower() or "refused" in msg.lower():
                error_msg = (
                    f"Cannot connect to Ollama at {self._host} — is Ollama running? "
                    f"Original error: {exc}"
                )
            else:
                error_msg = f"Ollama API error: {exc}"
            logger.error(error_msg)
            return LLMResponse(
                raw_text=error_msg,
                json_data=None,
                json_error=error_msg,
                model=self._config.model,
                backend=self.name(),
                **error_fields(exc),
                prompt_tokens=None,
                completion_tokens=None,
            )
