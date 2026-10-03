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


class UsageMeter(LLMBackend):
    """Transparent wrapper that meters token usage across EVERY backend call.

    The agent loop's own counters cover only the main per-iteration call; the
    post-search analysis calls are not counted there. Wrapping the backend once
    gives complete per-run totals (including prompt-cache writes and reads) for
    cost accounting, without touching any call site. Behaviour is unchanged:
    every call is delegated verbatim.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.api_errors = 0
        self.truncated = 0

    def complete(self, system_prompt, user_prompt, temperature=None):
        resp = self._inner.complete(system_prompt, user_prompt, temperature)
        self.calls += 1
        if getattr(resp, "api_error", False):
            self.api_errors += 1
        if getattr(resp, "stop_reason", None) == "max_tokens":
            self.truncated += 1
        self.prompt_tokens += resp.prompt_tokens or 0
        self.completion_tokens += resp.completion_tokens or 0
        self.cache_creation_tokens += resp.cache_creation_tokens or 0
        self.cache_read_tokens += resp.cache_read_tokens or 0
        return resp

    def name(self) -> str:
        return self._inner.name()

    def supports_json_mode(self) -> bool:
        return self._inner.supports_json_mode()

    def __getattr__(self, item):  # delegate anything backend-specific
        if item == "_inner":  # not yet set (e.g. during copy/unpickle)
            raise AttributeError(item)
        return getattr(self._inner, item)

    def totals(self) -> dict:
        """Per-run usage totals, as written to the journal (``llm_usage``)."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "api_errors": self.api_errors,
            "truncated": self.truncated,
            "temperature_sent": getattr(self._inner, "temperature_sent", None),
        }
