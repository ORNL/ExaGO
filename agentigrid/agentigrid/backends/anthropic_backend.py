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
        # None = not yet known; True/False once a call has succeeded / the
        # model has rejected the parameter (see complete()).
        self.temperature_sent: Optional[bool] = None
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

    def _system_param(self, system_prompt: str):
        """System prompt in the form the Messages API expects.

        With ``prompt_cache`` on, the system prompt is sent as a single text block
        carrying an ephemeral ``cache_control`` breakpoint. AgentiGrid builds the
        system prompt once per session, so every later iteration reads it from the
        cache instead of paying the full input price again. This affects billing
        only; the model sees identical text either way. Prompts below the model's
        minimum cacheable length are silently processed uncached by the API.
        """
        if not getattr(self._config, "prompt_cache", False):
            return system_prompt
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

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
            kwargs = dict(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                system=self._system_param(system_prompt),
                messages=[{"role": "user", "content": user_prompt}],
            )
            if self.temperature_sent is not False:
                kwargs["temperature"] = temp
            try:
                response = self._client.messages.create(**kwargs)
                if "temperature" in kwargs:
                    self.temperature_sent = True
            except Exception as exc:
                # Newer models reject the sampling-temperature parameter
                # ("`temperature` is deprecated for this model"). Retry once
                # without it and stop sending it for the rest of the session;
                # the journal records temperature_sent=False.
                if "temperature" in kwargs and "temperature" in str(exc).lower():
                    logger.warning("Model %s rejects `temperature`; retrying without it.", self._config.model)
                    kwargs.pop("temperature")
                    self.temperature_sent = False
                    response = self._client.messages.create(**kwargs)
                else:
                    raise

            raw_text = ""
            content_types = []
            for block in response.content:
                content_types.append(getattr(block, "type", None))
                if block.type == "text":
                    raw_text += block.text
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                logger.warning("Response stopped at max_tokens=%s (blocks: %s, %d text chars)",
                               self._config.max_tokens, content_types, len(raw_text))

            prompt_tokens, cache_creation, cache_read = _input_token_breakdown(
                response.usage
            )
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
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                stop_reason=stop_reason,
                content_types=content_types,
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
                api_error=True,
            )


def _input_token_breakdown(usage) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (total_input, cache_creation, cache_read) from an API usage object.

    With caching, the API's ``input_tokens`` counts only tokens after the last
    cache breakpoint. The total input is ``input_tokens + cache_creation_input_tokens
    + cache_read_input_tokens``; returning that total keeps ``prompt_tokens``
    comparable with runs made before caching was enabled.
    """
    base = getattr(usage, "input_tokens", None)
    creation = getattr(usage, "cache_creation_input_tokens", None)
    read = getattr(usage, "cache_read_input_tokens", None)
    creation = creation if isinstance(creation, int) else None
    read = read if isinstance(read, int) else None
    if not isinstance(base, int):
        return None, creation, read
    return base + (creation or 0) + (read or 0), creation, read
