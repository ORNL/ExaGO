"""OpenAI LLM backend adapter."""

from __future__ import annotations

import logging
import os
from typing import Optional

from agentigrid.logging_setup import log_scope
from agentigrid.backends.base import LLMBackend, LLMResponse, derive_reasoning_tokens
from agentigrid.backends.json_extract import extract_json
from agentigrid.backends.retry import error_fields
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

        # No SDK retries: agentigrid.backends.retry retries and logs every attempt.
        kwargs: dict = {"api_key": api_key, "max_retries": 0}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url

        self._client = openai.OpenAI(**kwargs)

    def name(self) -> str:
        """Return the backend name."""
        return "openai"

    def supports_json_mode(self) -> bool:
        """OpenAI supports native JSON mode."""
        return True

    def endpoint(self) -> Optional[str]:
        """Base URL of the client (OpenAI or a compatible endpoint such as Gemini's)."""
        return str(self._client.base_url)

    def reasoning_param(self) -> dict:
        """``reasoning_effort`` as sent (nothing when not configured)."""
        if self._config.reasoning_effort is None:
            return {}
        return {"reasoning_effort": self._config.reasoning_effort}

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
                "max_completion_tokens": self._config.max_tokens,
            }
            if temp is not None:                 # temperature: null -> not sent
                kwargs["temperature"] = temp
            kwargs.update(self.reasoning_param())
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            with log_scope(logger, "LLM API request", backend=self.name(), model=self._config.model):
                response = self._client.chat.completions.create(**kwargs)

            with log_scope(logger, "LLM response parsing", backend=self.name(), model=self._config.model):
                raw_text = response.choices[0].message.content or ""
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                api_total_tokens = getattr(usage, "total_tokens", None)
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens, derived = derive_reasoning_tokens(
                    prompt_tokens, completion_tokens,
                    getattr(details, "reasoning_tokens", None), api_total_tokens,
                )

                json_data, json_error = extract_json(raw_text)

            return LLMResponse(
                raw_text=raw_text,
                json_data=json_data,
                json_error=json_error,
                model=self._config.model,
                backend=self.name(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                api_total_tokens=api_total_tokens,
                reasoning_tokens_derived=derived,
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
                **error_fields(exc),
                prompt_tokens=None,
                completion_tokens=None,
            )
