"""Base classes for LLM backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Structured response from an LLM backend."""

    raw_text: str
    json_data: Optional[dict]
    json_error: Optional[str]
    model: str
    backend: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    # Usage details, when the provider reports them (see derive_reasoning_tokens).
    reasoning_tokens: Optional[int] = None
    api_total_tokens: Optional[int] = None      # the provider's own usage.total_tokens
    reasoning_tokens_derived: bool = False      # True: reasoning is outside completion_tokens


def derive_reasoning_tokens(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    reported_reasoning: Optional[int],
    api_total_tokens: Optional[int],
) -> tuple[Optional[int], bool]:
    """Return ``(reasoning_tokens, derived)`` for one response.

    A reported count (OpenAI ``completion_tokens_details.reasoning_tokens``, a
    sub-count of ``completion_tokens``) is used as is. Otherwise, when the
    provider total exceeds prompt + completion (Gemini's OpenAI-compatible
    endpoint counts thinking only in ``total_tokens``), the difference is the
    reasoning count and lies *outside* ``completion_tokens`` (``derived=True``).
    """
    if reported_reasoning is not None:
        return reported_reasoning, False
    if prompt_tokens is None or completion_tokens is None or api_total_tokens is None:
        return None, False
    extra = api_total_tokens - prompt_tokens - completion_tokens
    return (extra, True) if extra > 0 else (None, False)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a prompt and return a structured response.

        Args:
            system_prompt: System/instruction prompt.
            user_prompt: User message content.
            temperature: Override the configured temperature (optional).

        Returns:
            LLMResponse with raw text and optionally extracted JSON.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'anthropic', 'openai')."""
        ...

    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Whether this backend supports a native JSON output mode."""
        ...

    def endpoint(self) -> Optional[str]:
        """URL the requests go to (for the run's settings record), if known."""
        return None

    def reasoning_param(self) -> dict:
        """The reasoning parameter actually sent with each request ({} if none)."""
        return {}
