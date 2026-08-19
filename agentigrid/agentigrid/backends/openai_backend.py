"""OpenAI LLM backend adapter."""

from __future__ import annotations

import logging
import os
from typing import Optional

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.backends.json_extract import extract_json
from agentigrid.config import LLMConfig

logger = logging.getLogger("agentigrid.backends.openai")

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]
    logger.warning("openai package not installed — OpenAI backend unavailable")


class OpenAIBackend(LLMBackend):
    """Backend adapter for the OpenAI API (and compatible endpoints)."""

    def __init__(self, config: LLMConfig) -> None:
        if openai is None:
            raise ImportError("openai package is required for the OpenAI backend")

        self._config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            logger.warning(
                "Environment variable %s is not set — API calls will fail",
                config.api_key_env,
            )

        kwargs: dict = {"api_key": api_key}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url

        self._client = openai.OpenAI(**kwargs)

    def name(self) -> str:
        """Return the backend name."""
        return "openai"

    def supports_json_mode(self) -> bool:
        """OpenAI supports native JSON mode."""
        return True

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI API."""
        temp = temperature if temperature is not None else self._config.temperature

        # Enable JSON mode when the system prompt asks for JSON
        use_json_mode = "json" in system_prompt.lower()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            kwargs: dict = {
                "model": self._config.model,
                "messages": messages,
                "temperature": temp,
                "max_completion_tokens": self._config.max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)

            raw_text = response.choices[0].message.content or ""
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)

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
            error_msg = f"OpenAI API error: {exc}"
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
