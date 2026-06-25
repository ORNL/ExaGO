"""Registry of named, verified per-candidate sweep metrics and feasibility predicates.

This is the Phase-1 trusted-primitive approach (C3): the LLM selects a metric or
predicate *by name* from this registry; it never synthesizes logic. Later
capabilities (C4, C8, …) register their own entries via ``register_metric`` /
``register_predicate``.

Signatures
----------
metric(candidate, base, context) -> float | None
    A per-candidate scalar reduced over by the LLM (e.g. ranked). ``candidate``
    and ``base`` are ``OPFLOWResult`` (``base`` may be ``None`` unless the metric
    needs it — see ``metric_needs_base``). ``context`` is a dict carrying the
    candidate bus, the V-band limits, and any action params.

predicate(candidate, base, context) -> tuple[bool, str]
    Whether the candidate satisfies the (possibly custom) feasibility criterion,
    plus a human-readable reason when it does not.
"""

from __future__ import annotations

from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cost_metric(candidate, base, context) -> Optional[float]:
    """Total system cost (OPF objective value) at the candidate solution."""
    return candidate.objective_value if candidate is not None else None


def _max_delta_v_metric(candidate, base, context) -> Optional[float]:
    """Worst-case system-wide voltage change: max_b |V_candidate[b] − V_base[b]|.

    A steady-state voltage sensitivity between the base-case OPF solution and the
    OPF solution with the load block added at the candidate bus — not a physical
    switching/energization transient. The bus with the largest change need not be
    the candidate bus (the effect propagates system-wide).

    Scope: the maximum is taken over ALL buses, **including the connection bus
    itself** (it is part of the system-wide profile). Caveat: under OPFLOW this
    is the change between two *cost-optimal* voltage profiles (base vs. with the
    switched-in block), not a physical energization transient — the solver
    re-dispatches reactive support, so the value is a steady-state sensitivity,
    not a dynamic step. The metric is meaningful only for certified (converged)
    candidates; the sweep handler records it as None otherwise.
    """
    if candidate is None or base is None:
        return None
    base_v = {b.bus_id: b.Vm for b in base.buses}
    worst = 0.0
    for b in candidate.buses:
        bv = base_v.get(b.bus_id)
        if bv is not None:
            worst = max(worst, abs(b.Vm - bv))
    return worst


# Metrics that require a parsed base-case result (solved once before the sweep).
_METRICS_NEED_BASE = {"max_delta_v"}

# Default ranking direction per metric (how the LLM reduces over candidates).
_METRIC_DIRECTION = {"cost": "minimize", "max_delta_v": "maximize"}

METRICS: dict[str, Callable] = {
    "cost": _cost_metric,
    "max_delta_v": _max_delta_v_metric,
}


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _standard_predicate(candidate, base, context) -> tuple[bool, str]:
    """Default feasibility: converged with no V-band / loading violations.

    Reason mirrors ``_infeasible_reason`` semantics so the default sweep path is
    unchanged: certified by solver status, never by uncertified iterate metrics.
    """
    if candidate is None:
        return False, "did not converge"
    feasible = (
        candidate.feasibility_detail == "feasible" and candidate.num_violations == 0
    )
    if feasible:
        return True, ""
    if (candidate.convergence_status or "").upper().startswith("CONVERGED"):
        return False, "constraint violation"
    return False, "did not converge"


def _reactive_adequacy_predicate(candidate, base, context) -> tuple[bool, str]:
    """Reactive adequacy: a feasible OPF exists with the unit forced to (Pmax, Qmax).

    A capability/headroom test (prompt 17), not the V/loading criterion. The
    mutation pins the added unit to P = Pmax and Q = Qmax (see the sweep
    handler); this predicate certifies whether the network can absorb that
    forced reactive injection without a violation. The reason names the limiting
    quantity on failure.
    """
    if candidate is None:
        return False, "OPF did not converge at forced (Pmax, Qmax)"
    if not (candidate.convergence_status or "").upper().startswith("CONVERGED"):
        return False, "OPF did not converge at forced (Pmax, Qmax)"
    if candidate.num_violations > 0:
        if candidate.violation_details:
            detail = "; ".join(candidate.violation_details[:2])
        else:
            detail = "voltage-band / thermal limit"
        return False, f"limit violated at (Pmax, Qmax): {detail}"
    if candidate.feasibility_detail != "feasible":
        return False, f"infeasible at (Pmax, Qmax): {candidate.feasibility_detail}"
    return True, ""


PREDICATES: dict[str, Callable] = {
    "standard": _standard_predicate,
    "reactive_adequacy": _reactive_adequacy_predicate,
}


# ---------------------------------------------------------------------------
# Accessors / registration
# ---------------------------------------------------------------------------

def metric_needs_base(name: Optional[str]) -> bool:
    """Whether the named metric requires a parsed base-case result."""
    return name in _METRICS_NEED_BASE


def metric_direction(name: Optional[str]) -> str:
    """Default ranking direction for the named metric (minimize / maximize)."""
    return _METRIC_DIRECTION.get(name or "cost", "minimize")


def get_metric(name: str) -> Callable:
    """Return the metric callable, or raise KeyError with the available names."""
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(
            f"Unknown sweep metric '{name}'. Available: {sorted(METRICS)}"
        ) from None


def get_predicate(name: str) -> Callable:
    """Return the predicate callable, or raise KeyError with the available names."""
    try:
        return PREDICATES[name]
    except KeyError:
        raise KeyError(
            f"Unknown feasibility_predicate '{name}'. Available: {sorted(PREDICATES)}"
        ) from None


def register_metric(
    name: str, fn: Callable, *, needs_base: bool = False, direction: str = "minimize",
) -> None:
    """Register a new named metric (for later capabilities, e.g. C4/C8)."""
    METRICS[name] = fn
    if needs_base:
        _METRICS_NEED_BASE.add(name)
    _METRIC_DIRECTION[name] = direction


def register_predicate(name: str, fn: Callable) -> None:
    """Register a new named feasibility predicate (for later capabilities)."""
    PREDICATES[name] = fn
