"""Token accounting for every LLM call of a session.

One :class:`TokenLedger` per session records each ``LLMBackend.complete()``
response under a phase, so totals cover every call and the breakdown can be
inspected (journal ``token_usage`` block, console summary, report, GUI).
"""

from __future__ import annotations

from typing import Any, Optional

from agentigrid.backends.base import LLMResponse

# Phase names, in display order.
#   objective_extraction  objectives read from the initial goal
#   search                every call inside the iteration loop (incl. explore/select,
#                         the 'analyze' fallback and retries after parse failures)
#   analysis              post-search analysis / goal classification
#   steering              objectives read from a steering directive
#   report                analysis regenerated for a report from a saved journal
#   other                 anything else
PHASES = ("objective_extraction", "search", "analysis", "steering", "report", "other")

TOKEN_USAGE_VERSION = 1


def _label(phase: str) -> str:
    return phase.replace("_", " ")


class TokenLedger:
    """Per-call record of LLM token usage, with per-phase and grand totals.

    Usage a backend did not report stays ``None`` in the per-call list; the
    aggregates count it as 0 and report it in ``calls_without_usage``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, phase: str, response: LLMResponse, iteration: Optional[int] = None) -> None:
        """Add one ``complete()`` response under *phase*.

        Fields missing on *response* are recorded as ``None``, so bookkeeping
        never fails on a partial response object.
        """
        if phase not in PHASES:
            raise ValueError(f"Unknown token phase '{phase}' (expected one of {PHASES})")
        self.calls.append({
            "phase": phase,
            "iteration": iteration,
            "model": _str_or_none(getattr(response, "model", None)),
            "backend": _str_or_none(getattr(response, "backend", None)),
            "prompt_tokens": _int_or_none(getattr(response, "prompt_tokens", None)),
            "completion_tokens": _int_or_none(getattr(response, "completion_tokens", None)),
        })

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
        return self.total_prompt + self.total_completion

    @property
    def calls_without_usage(self) -> int:
        return sum(1 for c in self.calls if _missing_usage(c))

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
            "by_phase": self.by_phase,
            "per_call": [dict(c) for c in self.calls],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TokenLedger":
        """Rebuild from :meth:`to_dict` output; ``None`` or ``{}`` gives an empty ledger."""
        ledger = cls()
        for c in (data or {}).get("per_call", []):
            ledger.calls.append({
                "phase": c.get("phase", "other"),
                "iteration": c.get("iteration"),
                "model": c.get("model"),
                "backend": c.get("backend"),
                "prompt_tokens": c.get("prompt_tokens"),
                "completion_tokens": c.get("completion_tokens"),
                **({"carried_over": True} if c.get("carried_over") else {}),
            })
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
            f"Tokens: {self.total:,} total ({self.total_prompt:,} prompt + "
            f"{self.total_completion:,} completion) in {n} LLM call{'s' if n != 1 else ''}"
        )
        if self.calls:
            text += f"; {self.breakdown_text()}"
        missing = self.calls_without_usage
        if missing:
            text += f"; {missing} call{'s' if missing != 1 else ''} without usage data"
        return text


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _int_or_none(value: Any) -> Optional[int]:
    """Token count as int; anything that is not a number (None, a mock) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _missing_usage(call: dict) -> bool:
    return call["prompt_tokens"] is None or call["completion_tokens"] is None
