"""Contingency relief-measure search (C.7, Option A).

For a FAILED contingency, greedily try relief measures in priority order — each
with a bounded parameter search — re-solving OPFLOW on top of the outage, and
report the first measure+setting that restores feasibility.

Option-A modeling assumptions (confirm with the domain expert before relying on
results — see the C.7 task header):
  1. OPF-redispatch feasibility: "feasible" = post-outage OPFLOW converges.
  2. Generator redispatch is INHERENT to the OPF (Pg is a decision variable), so
     ``set_gen_dispatch`` is initial-guess-only and cannot rescue a contingency
     that already failed with optimal redispatch. ``generator_redispatch`` is
     therefore a reported pass-through (no command applied, never resolves).
  3. Transformer tap ratios are FIXED in ExaGO ACOPF, so a tap change is a genuine
     lever here. If taps become optimization variables, treat tap change like
     redispatch (subsumed) → an Option-B redesign.

This module is pure w.r.t. the agent loop: it imports only the command parser, the
modifier, sweep_metrics (the feasibility predicate), topology, and the MATPOWER
model — never ``agent_loop`` (avoids an import cycle). The caller injects a
``solve_fn`` (net → parsed OPFLOWResult|None) so this module never touches the
executor directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from agentigrid.engine import sweep_metrics, topology
from agentigrid.engine.commands import parse_command
from agentigrid.engine.modifier import apply_modifications
from agentigrid.parsers.matpower_model import MATNetwork

# Priority order (highest first) and the canonical measure names.
MEASURE_ORDER = (
    "transformer_ratio",
    "generator_redispatch",
    "line_switching",
    "load_curtailment",
)

# A solve callable: takes a net, returns a parsed OPFLOWResult (or None on failure).
SolveFn = Callable[[MATNetwork], object]


@dataclass(frozen=True)
class ReliefAction:
    """The relief measure + setting that restored feasibility."""

    measure: str                      # one of MEASURE_ORDER
    commands: tuple[dict, ...]        # outage-relative relief commands ({} for redispatch pass-through)
    detail: str                       # human summary


@dataclass(frozen=True)
class ReliefResult:
    """Outcome of the greedy relief search for one contingency."""

    resolved: bool
    action: Optional[ReliefAction]        # the resolving action, or None if unresolved
    attempts: list = field(default_factory=list)  # (measure, resolved?) in priority order


# ---------------------------------------------------------------------------
# Feasibility trial
# ---------------------------------------------------------------------------

def _is_feasible(opflow) -> bool:
    passed, _ = sweep_metrics.PREDICATES["standard"](opflow, None, {})
    return bool(passed)


def _trial(
    net: MATNetwork,
    outage_cmds: tuple[dict, ...],
    relief_cmds: list[dict],
    vmin: float,
    vmax: float,
    solve_fn: SolveFn,
) -> bool:
    """Build (vlimits + outage + relief), solve, and judge with the standard predicate."""
    raw = (
        [{"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax}]
        + list(outage_cmds)
        + list(relief_cmds)
    )
    try:
        cmds = [parse_command(r) for r in raw]
        mnet, _ = apply_modifications(net, cmds, application="opflow")
    except Exception:
        return False
    opflow = solve_fn(mnet)
    return _is_feasible(opflow)


# ---------------------------------------------------------------------------
# Candidate selection (deterministic)
# ---------------------------------------------------------------------------

def _focus_buses(contingency) -> set[int]:
    """Buses local to the contingency (neighbor buses + outaged branch endpoints)."""
    buses: set[int] = set()
    for e in contingency.elements:
        if e.neighbor_bus is not None:
            buses.add(e.neighbor_bus)
        if e.kind == "branch":
            buses.add(e.fbus)
            buses.add(e.tbus)
        elif e.bus is not None:
            buses.add(e.bus)
    return buses


def _outaged_branch_keys(contingency) -> set[tuple[int, int, int]]:
    """Unordered-endpoint + ckt keys of the branches this contingency takes out."""
    keys: set[tuple[int, int, int]] = set()
    for e in contingency.elements:
        if e.kind == "branch":
            keys.add((min(e.fbus, e.tbus), max(e.fbus, e.tbus), e.ckt or 0))
    return keys


def _incident_branches_sorted(net: MATNetwork, buses: set[int]):
    """In-service branches incident to any focus bus, deduped, in a fixed order.

    Yields (branch, key) where key = (lo, hi, ckt) — ckt being the branch's
    position among net branches sharing its unordered endpoints (matching
    ``modifier._branch_index_in_network`` semantics).
    """
    seen: dict[tuple[int, int, int], object] = {}
    for bus in sorted(buses):
        for br in topology.incident_branches(net, bus):
            lo, hi = min(br.fbus, br.tbus), max(br.fbus, br.tbus)
            ckt = 0
            for other in net.branches:
                if other is br:
                    break
                if (min(other.fbus, other.tbus), max(other.fbus, other.tbus)) == (lo, hi):
                    ckt += 1
            key = (lo, hi, ckt)
            if key not in seen:
                seen[key] = br
    return [(seen[k], k) for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Per-measure searches
# ---------------------------------------------------------------------------

def _try_transformer_ratio(net, contingency, vmin, vmax, solve_fn, cfg):
    """Tap-ratio change on transformers (ratio != 0) local to the contingency."""
    focus = _focus_buses(contingency)
    outaged = _outaged_branch_keys(contingency)
    outage_cmds = tuple(contingency.commands())
    steps = list(getattr(cfg, "relief_tap_steps", [0.90, 0.95, 1.00, 1.05, 1.10]))
    for br, key in _incident_branches_sorted(net, focus):
        if br.ratio == 0:
            continue  # not a transformer
        if key in outaged:
            continue  # don't tap-change an outaged branch
        for ratio in steps:
            cmd = {"action": "set_tap_ratio", "fbus": br.fbus, "tbus": br.tbus, "ratio": ratio}
            if _trial(net, outage_cmds, [cmd], vmin, vmax, solve_fn):
                return ReliefAction(
                    measure="transformer_ratio",
                    commands=(cmd,),
                    detail=f"tap ratio branch {br.fbus}-{br.tbus} -> {ratio:g}",
                )
    return None


def _try_generator_redispatch(net, contingency, vmin, vmax, solve_fn, cfg):
    """Pass-through under Option A: OPF redispatch is inherent; no command applied.

    Never resolves (the bare outaged case is exactly the already-failed solve), so
    this is a logged no-op and the search proceeds to the next measure.
    """
    return None


def _try_line_switching(net, contingency, vmin, vmax, solve_fn, cfg):
    """Switch OUT one in-service branch local to the contingency (excluding outaged)."""
    focus = _focus_buses(contingency)
    outaged = _outaged_branch_keys(contingency)
    outage_cmds = tuple(contingency.commands())
    for br, key in _incident_branches_sorted(net, focus):
        if key in outaged:
            continue
        cmd = {
            "action": "set_branch_status",
            "fbus": br.fbus, "tbus": br.tbus, "ckt": key[2], "status": 0,
        }
        if _trial(net, outage_cmds, [cmd], vmin, vmax, solve_fn):
            return ReliefAction(
                measure="line_switching",
                commands=(cmd,),
                detail=f"switch out branch {br.fbus}-{br.tbus}",
            )
    return None


def _try_load_curtailment(net, contingency, vmin, vmax, solve_fn, cfg):
    """Backstop: bisect a uniform curtailment fraction on local load buses.

    Curtails load at the contingency's focus buses via
    ``set_load(Pd*(1-f), Qd*(1-f))`` and bisects the minimum feasible fraction to
    ``relief_curtail_tol_mw``. Reports the minimum MW curtailed. Resolves at f=1
    (all local load removed) unless even that is infeasible.
    """
    focus = _focus_buses(contingency)
    outage_cmds = tuple(contingency.commands())
    tol_mw = float(getattr(cfg, "relief_curtail_tol_mw", 1.0))

    # Load-bearing focus buses (deterministic order) and their base Pd/Qd.
    load_buses = []
    total_pd = 0.0
    for b in net.buses:
        if b.bus_i in focus and (b.Pd != 0 or b.Qd != 0):
            load_buses.append((b.bus_i, b.Pd, b.Qd))
            total_pd += b.Pd
    load_buses.sort(key=lambda x: x[0])
    if not load_buses or total_pd <= 0:
        return None  # nothing to curtail here

    def _curtail_cmds(f: float) -> list[dict]:
        return [
            {"action": "set_load", "bus": bi, "Pd": pd * (1.0 - f), "Qd": qd * (1.0 - f)}
            for bi, pd, qd in load_buses
        ]

    # f=1 (remove all local load) must be feasible for curtailment to work.
    if not _trial(net, outage_cmds, _curtail_cmds(1.0), vmin, vmax, solve_fn):
        return None

    # Bisect the minimum feasible fraction. lo infeasible (f=0 = bare outage, failed),
    # hi feasible.
    lo, hi = 0.0, 1.0
    while (hi - lo) * total_pd > tol_mw:
        mid = 0.5 * (lo + hi)
        if _trial(net, outage_cmds, _curtail_cmds(mid), vmin, vmax, solve_fn):
            hi = mid
        else:
            lo = mid

    mw = hi * total_pd
    buses_str = ", ".join(str(bi) for bi, _, _ in load_buses)
    return ReliefAction(
        measure="load_curtailment",
        commands=tuple(_curtail_cmds(hi)),
        detail=f"curtail {mw:.1f} MW ({hi * 100:.1f}%) at bus(es) {buses_str}",
    )


_MEASURE_FNS = {
    "transformer_ratio": _try_transformer_ratio,
    "generator_redispatch": _try_generator_redispatch,
    "line_switching": _try_line_switching,
    "load_curtailment": _try_load_curtailment,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_relief(
    net: MATNetwork,
    contingency,
    relief_measures,
    vmin: float,
    vmax: float,
    solve_fn: SolveFn,
    cfg,
) -> ReliefResult:
    """Greedily search relief measures in the given priority order.

    For each measure name in ``relief_measures`` (which must be drawn from
    ``MEASURE_ORDER``), run its bounded search; on the first measure that restores
    feasibility, return ``ReliefResult(resolved=True, action=..., attempts=...)``
    where ``attempts`` records every measure tried and whether it resolved.
    """
    attempts: list = []
    for measure in relief_measures:
        fn = _MEASURE_FNS.get(measure)
        if fn is None:
            attempts.append((measure, False))
            continue
        action = fn(net, contingency, vmin, vmax, solve_fn, cfg)
        resolved = action is not None
        attempts.append((measure, resolved))
        if resolved:
            return ReliefResult(resolved=True, action=action, attempts=attempts)
    return ReliefResult(resolved=False, action=None, attempts=attempts)
