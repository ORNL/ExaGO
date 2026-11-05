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
    # Prompt-cache accounting (Anthropic). ``prompt_tokens`` always reports the
    # TOTAL input tokens (uncached + cache writes + cache reads), so its meaning
    # is the same with or without caching; these break that total down.
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    # True when the request itself failed (network, auth, invalid request...),
    # as opposed to the model answering with unparseable text. Lets the agent
    # loop record API failures separately from rejected proposals.
    api_error: bool = False
    # Why generation stopped ("end_turn", "max_tokens", ...) and the content
    # block types returned, when the backend reports them. A "max_tokens" stop
    # is an output-cap limit, not a model proposal, and is recorded as such.
    stop_reason: Optional[str] = None
    content_types: Optional[list] = None


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
