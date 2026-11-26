"""Retrieval query construction (generation stage only).

Three query modes, selected by AGENTIGRID_RAG_QUERY:

    goal  (default) -- the goal text. Every iteration of a
                       run sends the same query, so a deterministic retriever
                       injects one fixed, goal-specific reference block.
    state           -- the goal text plus a compact, deterministic summary of the
                       current state: solver status, violations, and the command
                       errors the validator reported for the previous proposal.
                       The query, and hence what retrieval and grading see, can
                       change from iteration to iteration.
    error           -- what went wrong, without the goal text (which is already
                       in the prompt): the previous iteration's command errors,
                       then the worst violations, then the state line. Falls back
                       to the goal text when there are neither errors nor
                       violations. In a smoke test the 'state' query retrieved the
                       same chunks in every iteration because the goal text
                       dominated the embedding; this mode lets the state steer
                       retrieval.

Violations are ordered by severity (relative excess over the limit, parsed from
the parser's "x > limit" / "x < limit" text; unparseable lines keep their order
after the parsed ones), so the query carries the worst ones, not the first ones
by bus number.

The state query is built only from what the agent itself sees (the latest solve
and the previous iteration's error feedback); it never reads the evaluator's
intent checks or any ground truth.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

VALID_QUERY_MODES = ("goal", "state", "error")

MAX_QUERY_CHARS = 1500
MAX_VIOLATIONS = 5
MAX_ERRORS = 5


def resolve_query_mode() -> str:
    """Active query mode from AGENTIGRID_RAG_QUERY ('goal' when unset).

    Unknown values raise instead of degrading: a silently wrong query mode would
    mislabel a run.
    """
    raw = os.environ.get("AGENTIGRID_RAG_QUERY", "").strip().lower()
    if not raw:
        return "goal"
    if raw not in VALID_QUERY_MODES:
        raise ValueError(
            f"AGENTIGRID_RAG_QUERY={raw!r} is not a query mode (valid: {', '.join(VALID_QUERY_MODES)})"
        )
    return raw


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", str(line)).strip()


def _error_lines(feedback: Optional[str]) -> list[str]:
    """Individual error messages from the previous iteration's feedback text."""
    if not feedback:
        return []
    out: list[str] = []
    for raw in str(feedback).splitlines():
        line = _clean(raw)
        if not line or line.lower().rstrip(":") in ("command errors",):
            continue
        if line not in out:
            out.append(line)
    return out


_EXCESS = re.compile(r"=\s*([0-9]*\.?[0-9]+)\s*(?:pu|MVA)?\s*([<>])\s*(?:\w+=)?\s*([0-9]*\.?[0-9]+)")


def _severity(detail: str) -> Optional[float]:
    """Relative excess over the limit for a parser violation line, or None.

    Matches e.g. "Bus 100: Vm=1.060 pu > 1.05 (overvoltage)" and
    "Branch 123-124: Sf=446.20 > Slim=400.00 MVA".
    """
    m = _EXCESS.search(detail)
    if not m:
        return None
    value, limit = float(m.group(1)), float(m.group(3))
    if limit == 0:
        return None
    return abs(value - limit) / abs(limit)


def _worst_violations(opflow: Any, n: int) -> list[str]:
    viol = [_clean(v) for v in (getattr(opflow, "violation_details", None) or []) if _clean(v)]
    indexed = list(enumerate(viol))
    # parsed lines first, by descending severity; ties and unparsed keep order
    indexed.sort(key=lambda t: (_severity(t[1]) is None, -(_severity(t[1]) or 0.0), t[0]))
    return [v for _i, v in indexed[:n]]


def _state_line(opflow: Any) -> str:
    parts: list[str] = []
    status = _clean(getattr(opflow, "convergence_status", "") or "")
    feas = _clean(getattr(opflow, "feasibility_detail", "") or "")
    if status:
        parts.append(status)
    if feas:
        parts.append(feas)
    nviol = getattr(opflow, "num_violations", None)
    if nviol is not None:
        parts.append(f"{int(nviol)} violations")
    vmin, vmax = getattr(opflow, "voltage_min", None), getattr(opflow, "voltage_max", None)
    if vmin and vmax:
        parts.append(f"voltage {float(vmin):.3f}-{float(vmax):.3f} pu")
    load = getattr(opflow, "max_line_loading_pct", None)
    if load:
        parts.append(f"max line loading {float(load):.1f}%")
    return "Current state: " + ", ".join(parts) + "." if parts else ""


def build_state_query(
    goal: str,
    opflow: Any = None,
    error_feedback: Optional[str] = None,
    *,
    max_violations: int = MAX_VIOLATIONS,
    max_errors: int = MAX_ERRORS,
    max_chars: int = MAX_QUERY_CHARS,
) -> str:
    """Goal text plus a deterministic state summary, capped at ``max_chars``.

    The same inputs always give the same query, so a frozen store returns the
    same hits for the same state.
    """
    lines = [_clean(goal)]
    if opflow is not None:
        st = _state_line(opflow)
        if st:
            lines.append(st)
        viol = _worst_violations(opflow, max_violations)
        if viol:
            lines.append("Violations: " + "; ".join(viol))
    errs = _error_lines(error_feedback)
    if errs:
        lines.append("Last command errors: " + "; ".join(errs[:max_errors]))
    query = "\n".join(l for l in lines if l)
    return query[:max_chars]


def build_error_query(
    goal: str,
    opflow: Any = None,
    error_feedback: Optional[str] = None,
    *,
    max_violations: int = MAX_VIOLATIONS,
    max_errors: int = MAX_ERRORS,
    max_chars: int = MAX_QUERY_CHARS,
) -> str:
    """Errors first, then the worst violations, then the state line; no goal text.

    Returns the goal text when there are neither command errors nor violations.
    """
    lines: list[str] = []
    errs = _error_lines(error_feedback)
    if errs:
        lines.append("Command errors: " + "; ".join(errs[:max_errors]))
    if opflow is not None:
        viol = _worst_violations(opflow, max_violations)
        if viol:
            lines.append("Violations: " + "; ".join(viol))
    if not lines:
        return _clean(goal)[:max_chars]
    if opflow is not None:
        st = _state_line(opflow)
        if st:
            lines.append(st)
    return "\n".join(lines)[:max_chars]


def build_query(mode: str, goal: str, opflow: Any = None, error_feedback: Optional[str] = None) -> str:
    """Query for the given mode ('goal' returns the goal text unchanged)."""
    if mode == "state":
        return build_state_query(goal, opflow, error_feedback)
    if mode == "error":
        return build_error_query(goal, opflow, error_feedback)
    return goal


__all__ = ["VALID_QUERY_MODES", "resolve_query_mode", "build_state_query", "build_error_query", "build_query"]
