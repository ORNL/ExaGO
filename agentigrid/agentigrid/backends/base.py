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
