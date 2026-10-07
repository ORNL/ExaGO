"""Contingency relief-measure search (C.7, Option A).

For a FAILED contingency, greedily try relief measures in priority order — each
with a bounded parameter search — re-solving OPFLOW on top of the outage, and
report the first measure+setting that restores feasibility.

Relief uses only fast measures on existing, already-committed equipment: tap
changes, switching lines out or in, and load shedding. Unit commitment is never
changed and no equipment is added.

Option-A modeling assumptions:
  1. OPF-redispatch feasibility: "feasible" = post-outage OPFLOW converges.
  2. Generator redispatch is INHERENT to the OPF (Pg is a decision variable): every
     solve already redispatches all committed units, so it is not a separate
     measure.
  3. Transformer tap ratios are FIXED in ExaGO ACOPF, so a tap change is a genuine
     lever here. If taps become optimization variables, treat tap change like
     redispatch (subsumed) → an Option-B redesign.
  4. Load shedding grows in substation tiers around the outage, each load bus shed
     by at most ``relief_shed_max_fraction`` (see ``_try_load_curtailment``).

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
    """Buses local to the contingency (outaged branch endpoints, element buses)."""
    buses: set[int] = set()
    for e in contingency.elements:
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


def _incident_branches_sorted(net: MATNetwork, buses: set[int], include_out_of_service: bool = False):
    """Branches incident to any focus bus, deduped, in a fixed order.

    In-service only unless ``include_out_of_service``.

    Yields (branch, key) where key = (lo, hi, ckt) — ckt being the branch's
    position among net branches sharing its unordered endpoints (matching
    ``modifier._branch_index_in_network`` semantics).
    """
    seen: dict[tuple[int, int, int], object] = {}
    for bus in sorted(buses):
        for br in topology.incident_branches(net, bus, include_out_of_service):
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


def _try_line_switching(net, contingency, vmin, vmax, solve_fn, cfg):
    """Switch one branch local to the contingency: in-service ones out, open ones in.

    Candidates are the branches at the contingency's buses, in a fixed order; the
    branches the contingency itself takes out are never switched back in.
    """
    focus = _focus_buses(contingency)
    outaged = _outaged_branch_keys(contingency)
    outage_cmds = tuple(contingency.commands())
    for br, key in _incident_branches_sorted(net, focus, include_out_of_service=True):
        if key in outaged:
            continue
        switch_in = br.status == 0
        cmd = {
            "action": "set_branch_status",
            "fbus": br.fbus, "tbus": br.tbus, "ckt": key[2], "status": 1 if switch_in else 0,
        }
        if _trial(net, outage_cmds, [cmd], vmin, vmax, solve_fn):
            return ReliefAction(
                measure="line_switching",
                commands=(cmd,),
                detail=f"switch {'in' if switch_in else 'out'} branch {br.fbus}-{br.tbus}",
            )
    return None


def _try_load_curtailment(net, contingency, vmin, vmax, solve_fn, cfg, protected_load=None):
    """Shed load in growing rings around the contingency, at most a fixed share per bus.

    Tier 0 is the substation(s) of the contingency's buses; each further tier is
    one line farther out (transformers add no hop), up to ``relief_shed_max_depth``.
    Every load bus sheds at most ``relief_shed_max_fraction`` of its Pd and Qd
    (power factor kept). Starting at tier 0, if shedding the full share there does
    not restore feasibility, those buses stay at the full share and the next tier
    is added. In the first tier where it works, the smallest share that works is
    bisected to ``relief_curtail_tol_mw``. Unresolved if the full share at every
    tier up to the maximum depth is not enough.

    ``protected_load`` ({bus: (Pd, Qd)}) is never shed: the load the mutation added
    is the development under study. The share applies to the rest of the bus load.
    """
    focus = _focus_buses(contingency)
    outage_cmds = tuple(contingency.commands())
    cap = float(getattr(cfg, "relief_shed_max_fraction", 0.10))
    max_depth = int(getattr(cfg, "relief_shed_max_depth", 3))
    tol_mw = float(getattr(cfg, "relief_curtail_tol_mw", 1.0))

    scope = topology.substations_within(net, sorted(focus), max_depth)
    protected = protected_load or {}
    keep_of = {b.bus_i: protected.get(b.bus_i, (0.0, 0.0)) for b in net.buses}
    load_of = {   # sheddable part of each bus load
        b.bus_i: (b.Pd - keep_of[b.bus_i][0], b.Qd - keep_of[b.bus_i][1])
        for b in net.buses if b.Pd - keep_of[b.bus_i][0] > 0
    }
    tiers: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for bus in sorted(load_of):
        if bus in scope.tier_of:
            tiers[scope.tier_of[bus]].append(bus)

    def _cmds(full: list[int], part: list[int], f: float) -> list[dict]:
        cmds = []
        for bus, share in [(b, cap) for b in full] + [(b, f) for b in part]:
            pd, qd = load_of[bus]
            kp, kq = keep_of[bus]
            cmds.append({
                "action": "set_load", "bus": bus,
                "Pd": kp + pd * (1.0 - share), "Qd": kq + qd * (1.0 - share),
            })
        return cmds

    full: list[int] = []   # buses of the inner tiers, shed by the full share
    for tier, buses in enumerate(tiers):
        if not buses:
            continue
        if not _trial(net, outage_cmds, _cmds(full, buses, cap), vmin, vmax, solve_fn):
            full += buses
            continue

        # Bisect this tier's share; lo is known infeasible (0 = inner tiers alone).
        tier_pd = sum(load_of[b][0] for b in buses)
        lo, hi = 0.0, cap
        while (hi - lo) * tier_pd > tol_mw:
            mid = 0.5 * (lo + hi)
            if _trial(net, outage_cmds, _cmds(full, buses, mid), vmin, vmax, solve_fn):
                hi = mid
            else:
                lo = mid

        parts = []
        for t in range(tier):
            if tiers[t]:
                mw = sum(load_of[b][0] for b in tiers[t]) * cap
                parts.append(f"tier {t} {mw:.1f} MW ({cap * 100:.0f}%) at {', '.join(map(str, tiers[t]))}")
        parts.append(
            f"tier {tier} {tier_pd * hi:.1f} MW ({hi * 100:.1f}%) at {', '.join(map(str, buses))}"
        )
        total_mw = sum(load_of[b][0] for b in full) * cap + tier_pd * hi
        return ReliefAction(
            measure="load_curtailment",
            commands=tuple(_cmds(full, buses, hi)),
            detail=f"shed {total_mw:.1f} MW: " + "; ".join(parts),
        )
    return None


_MEASURE_FNS = {
    "transformer_ratio": _try_transformer_ratio,
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
    protected_load: Optional[dict] = None,
) -> ReliefResult:
    """Greedily search relief measures in the given priority order.

    For each measure name in ``relief_measures`` (which must be drawn from
    ``MEASURE_ORDER``), run its bounded search; on the first measure that restores
    feasibility, return ``ReliefResult(resolved=True, action=..., attempts=...)``
    where ``attempts`` records every measure tried and whether it resolved.
    ``protected_load`` ({bus: (Pd, Qd)}) is passed to load shedding, which never
    sheds it.
    """
    attempts: list = []
    for measure in relief_measures:
        fn = _MEASURE_FNS.get(measure)
        if fn is None:
            attempts.append((measure, False))
            continue
        extra = {"protected_load": protected_load} if measure == "load_curtailment" else {}
        action = fn(net, contingency, vmin, vmax, solve_fn, cfg, **extra)
        resolved = action is not None
        attempts.append((measure, resolved))
        if resolved:
            return ReliefResult(resolved=True, action=action, attempts=attempts)
    return ReliefResult(resolved=False, action=None, attempts=attempts)
