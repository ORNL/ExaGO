"""Anthropic LLM backend adapter."""

from __future__ import annotations

import logging
import os
from typing import Optional

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.backends.json_extract import extract_json
from agentigrid.config import LLMConfig

logger = logging.getLogger("agentigrid.backends.anthropic")

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]
    logger.warning("anthropic package not installed — Anthropic backend unavailable")


class AnthropicBackend(LLMBackend):
    """Backend adapter for the Anthropic Messages API."""

    def __init__(self, config: LLMConfig) -> None:
        if anthropic is None:
            raise ImportError("anthropic package is required for the Anthropic backend")

        self._config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            logger.warning(
                "Environment variable %s is not set — API calls will fail",
                config.api_key_env,
            )

        self._client = anthropic.Anthropic(api_key=api_key)

    def name(self) -> str:
        """Return the backend name."""
        return "anthropic"

    def supports_json_mode(self) -> bool:
        """Anthropic does not have a native JSON output mode."""
        return False

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a message to the Anthropic Messages API."""
        temp = temperature if temperature is not None else self._config.temperature

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text += block.text

            prompt_tokens = getattr(response.usage, "input_tokens", None)
            completion_tokens = getattr(response.usage, "output_tokens", None)

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
            error_msg = f"Anthropic API error: {exc}"
            logger.error(error_msg)
            return LLMResponse(
                raw_text=error_msg,
                json_data=None,
                json_error=error_msg,
                model=self._config.model,
                backend=self.name(),
                prompt_tokens=None,
                completion_tokens=None,
            )
