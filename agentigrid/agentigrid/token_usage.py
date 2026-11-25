"""Token accounting for every LLM call of a session.

One :class:`TokenLedger` per session records each ``LLMBackend.complete()``
response under a phase, so totals cover every call and the breakdown can be
inspected (journal ``token_usage`` block, console summary, report, GUI).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agentigrid.backends.base import LLMResponse

logger = logging.getLogger("agentigrid.token_usage")

# Phase names, in display order.
#   objective_extraction  objectives read from the initial goal
#   search                every call inside the iteration loop (incl. explore/select,
#                         the 'analyze' fallback and retries after parse failures)
#   analysis              post-search analysis / goal classification
#   steering              objectives read from a steering directive
#   report                analysis regenerated for a report from a saved journal
#   other                 anything else
PHASES = ("objective_extraction", "search", "analysis", "steering", "report", "other")

TOKEN_USAGE_VERSION = 2     # 2: reasoning / provider-total / billable-output fields


def _label(phase: str) -> str:
    return phase.replace("_", " ")


class TokenLedger:
    """Per-call record of LLM token usage, with per-phase and grand totals.

    Usage a backend did not report stays ``None`` in the per-call list; the
    aggregates count it as 0 and report it in ``calls_without_usage``.

    ``completion_tokens`` is what the provider reports as completion. Reasoning
    tokens a provider counts *outside* it (Gemini: only in its ``total_tokens``)
    are added in ``billable_output_tokens``; reasoning a provider counts inside
    it (OpenAI ``completion_tokens_details.reasoning_tokens``) is not added again.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(
        self,
        phase: str,
        response: LLMResponse,
        iteration: Optional[int] = None,
        prompt_chars: Optional[int] = None,
    ) -> None:
        """Add one ``complete()`` response under *phase*.

        *prompt_chars* is the length of system + user prompt (``None`` if the
        call site cannot provide it); ``completion_chars`` is the length of the
        response's ``raw_text``. Fields missing on *response* are recorded as
        ``None``, and an unknown *phase* is counted as "other" (with a warning),
        so bookkeeping never fails after a paid call.
        """
        raw_text = getattr(response, "raw_text", None)
        phase = _known_phase(phase)             # never raise after a paid call
        call = {
            "phase": phase,
            "iteration": iteration,
            "model": _str_or_none(getattr(response, "model", None)),
            "backend": _str_or_none(getattr(response, "backend", None)),
            "prompt_tokens": _int_or_none(getattr(response, "prompt_tokens", None)),
            "completion_tokens": _int_or_none(getattr(response, "completion_tokens", None)),
            "prompt_chars": _int_or_none(prompt_chars),
            "completion_chars": len(raw_text) if isinstance(raw_text, str) else None,
            "reasoning_tokens": _int_or_none(getattr(response, "reasoning_tokens", None)),
            "api_total_tokens": _int_or_none(getattr(response, "api_total_tokens", None)),
            "reasoning_tokens_derived": getattr(response, "reasoning_tokens_derived", False) is True,
        }
        call["billable_output_tokens"] = _billable_output(call)
        self.calls.append(call)

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------

    @property
    def total_prompt(self) -> int:
        return sum(c["prompt_tokens"] or 0 for c in self.calls)

    @property
    def total_completion(self) -> int:
        return sum(c["completion_tokens"] or 0 for c in self.calls)

    @property
    def total(self) -> int:
        """Prompt + completion tokens of all calls (journal ``total_tokens``).

        It excludes reasoning tokens a provider reports outside ``completion_tokens``
        (Gemini counts them only in its own ``total_tokens``); those are in
        :attr:`total_reasoning` and :attr:`total_billable_output`. Reasoning a provider
        counts inside ``completion_tokens`` (OpenAI) is included. Journals and the
        AgentiGrid 4 sweep numbers use this definition: do not change it.
        """
        return self.total_prompt + self.total_completion

    @property
    def reasoning_outside_completion(self) -> int:
        """Reasoning tokens not included in :attr:`total` (Gemini-style reporting)."""
        return self.total_billable_output - self.total_completion

    def total_label(self) -> str:
        """``"12,198 prompt+completion"``, plus ``" (+R reasoning)"`` for reasoning
        tokens counted outside completion (not in :attr:`total`)."""
        text = f"{self.total:,} prompt+completion"
        extra = self.reasoning_outside_completion
        if extra:
            text += f" (+{extra:,} reasoning)"
        return text

    @property
    def total_prompt_chars(self) -> int:
        return sum(c.get("prompt_chars") or 0 for c in self.calls)

    @property
    def total_completion_chars(self) -> int:
        return sum(c.get("completion_chars") or 0 for c in self.calls)

    @property
    def total_reasoning(self) -> int:
        return sum(c.get("reasoning_tokens") or 0 for c in self.calls)

    @property
    def total_billable_output(self) -> int:
        return sum(c.get("billable_output_tokens") or 0 for c in self.calls)

    @property
    def calls_without_usage(self) -> int:
        return sum(1 for c in self.calls if _missing_usage(c))

    @property
    def calls_with_total_mismatch(self) -> int:
        """Calls whose provider ``total_tokens`` differs from prompt + completion."""
        return sum(1 for c in self.calls if _total_mismatch(c))

    def __len__(self) -> int:
        return len(self.calls)

    @property
    def by_phase(self) -> dict[str, dict[str, int]]:
        """Aggregates per phase that has at least one call, in :data:`PHASES` order."""
        out: dict[str, dict[str, int]] = {}
        for phase in PHASES:
            calls = [c for c in self.calls if c["phase"] == phase]
            if not calls:
                continue
            prompt = sum(c["prompt_tokens"] or 0 for c in calls)
            completion = sum(c["completion_tokens"] or 0 for c in calls)
            out[phase] = {
                "calls": len(calls),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "calls_without_usage": sum(1 for c in calls if _missing_usage(c)),
                "prompt_chars": sum(c.get("prompt_chars") or 0 for c in calls),
                "completion_chars": sum(c.get("completion_chars") or 0 for c in calls),
                "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in calls),
                "billable_output_tokens": sum(c.get("billable_output_tokens") or 0 for c in calls),
            }
        return out

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready block (journal ``token_usage``, session checkpoint)."""
        return {
            "version": TOKEN_USAGE_VERSION,
            "calls": len(self.calls),
            "prompt_tokens": self.total_prompt,
            "completion_tokens": self.total_completion,
            "total_tokens": self.total,
            "calls_without_usage": self.calls_without_usage,
            "prompt_chars": self.total_prompt_chars,
            "completion_chars": self.total_completion_chars,
            "reasoning_tokens": self.total_reasoning,
            "billable_output_tokens": self.total_billable_output,
            "calls_with_total_mismatch": self.calls_with_total_mismatch,
            "by_phase": self.by_phase,
            "per_call": [dict(c) for c in self.calls],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TokenLedger":
        """Rebuild from :meth:`to_dict` output; ``None`` or ``{}`` gives an empty ledger."""
        ledger = cls()
        for c in (data or {}).get("per_call", []):
            call = {
                "phase": _known_phase(c.get("phase", "other")),
                "iteration": c.get("iteration"),
                "model": c.get("model"),
                "backend": c.get("backend"),
                "prompt_tokens": c.get("prompt_tokens"),
                "completion_tokens": c.get("completion_tokens"),
                # Fields below are absent in older journals.
                "prompt_chars": c.get("prompt_chars"),
                "completion_chars": c.get("completion_chars"),
                "reasoning_tokens": c.get("reasoning_tokens"),
                "api_total_tokens": c.get("api_total_tokens"),
                "reasoning_tokens_derived": bool(c.get("reasoning_tokens_derived", False)),
            }
            call["billable_output_tokens"] = (
                c["billable_output_tokens"] if "billable_output_tokens" in c else _billable_output(call)
            )
            if c.get("carried_over"):
                call["carried_over"] = True
            ledger.calls.append(call)
        return ledger

    @classmethod
    def from_legacy_totals(cls, prompt_tokens: int, completion_tokens: int) -> "TokenLedger":
        """Ledger for a session saved before per-call accounting existed.

        Such sessions counted search calls only, so the saved totals become one
        ``search`` entry marked ``carried_over``.
        """
        ledger = cls()
        if prompt_tokens or completion_tokens:
            ledger.calls.append({
                "phase": "search", "iteration": None, "model": None, "backend": None,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "prompt_chars": None, "completion_chars": None,
                "reasoning_tokens": None, "api_total_tokens": None,
                "reasoning_tokens_derived": False, "billable_output_tokens": completion_tokens,
                "carried_over": True,
            })
        return ledger

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def breakdown_text(self) -> str:
        """``search 19,497, objective extraction 1,210, analysis 5,021`` (phases with calls only)."""
        return ", ".join(f"{_label(p)} {a['total_tokens']:,}" for p, a in self.by_phase.items())

    def summary_text(self) -> str:
        """One line for the console summary."""
        n = len(self.calls)
        text = (
            f"Tokens: {self.total_label()} ({self.total_prompt:,} prompt + {self.total_completion:,} "
            f"completion) in {n} LLM call{'s' if n != 1 else ''}"
        )
        if self.calls:
            text += f"; {self.breakdown_text()}"
        missing = self.calls_without_usage
        if missing:
            text += f"; {missing} call{'s' if missing != 1 else ''} without usage data"
        return text


def _known_phase(phase: Any) -> str:
    """*phase* if it is one of :data:`PHASES`, else "other" (with a warning), so every
    call is counted and the totals equal the sum of :attr:`TokenLedger.by_phase`."""
    if phase in PHASES:
        return phase
    logger.warning("Unknown token phase %r (expected one of %s): counted as 'other'", phase, PHASES)
    return "other"


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _int_or_none(value: Any) -> Optional[int]:
    """Token count as int; anything that is not a number (None, a mock) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _missing_usage(call: dict) -> bool:
    return call["prompt_tokens"] is None or call["completion_tokens"] is None


def _billable_output(call: dict) -> Optional[int]:
    """Output tokens the provider bills: completion, plus reasoning counted outside it."""
    completion = call.get("completion_tokens")
    if completion is None:
        return None
    reasoning = call.get("reasoning_tokens")
    if call.get("reasoning_tokens_derived") and reasoning is not None:
        return completion + reasoning
    return completion


def _total_mismatch(call: dict) -> bool:
    total = call.get("api_total_tokens")
    if total is None or _missing_usage(call):
        return False
    return total != call["prompt_tokens"] + call["completion_tokens"]
