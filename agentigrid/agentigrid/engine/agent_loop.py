"""Agent loop controller — central orchestrator for LLM-driven search."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from agentigrid.backends import create_backend
from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import AppConfig
from agentigrid.engine.commands import parse_command
from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.engine.explore import (
    ExploreCache,
    VariantResult,
    annotate_cost_equivalent_siblings,
    build_variant_description,
    compute_pareto_labels,
    format_variant_results,
)
from agentigrid.engine.journal import JournalEntry, ObjectiveEntry, SearchJournal
from agentigrid.engine.metric_extractor import available_metrics, available_metrics_for_app, extract_all_metrics
from agentigrid.engine.modifier import apply_modifications, build_index_maps
from agentigrid.engine import sweep_metrics
from agentigrid.engine.objective_parser import (
    build_objective_extraction_prompt,
    parse_objective_extraction,
)
from agentigrid.engine.schema_description import command_schema_text
from agentigrid.parsers import (
    parse_matpower,
    network_metadata,
    network_summary,
    parse_simulation_result_for_app,
    results_summary_for_app,
)
from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.opflow_results import OPFLOWResult
from agentigrid.engine import topology
from agentigrid.engine import contingency
from agentigrid.engine import relief as relief_search
from agentigrid.engine import reserve as reserve_search


def _count_scenario_rows(scenario_path: Path) -> int:
    """Count data rows (excluding header) in a SOPFLOW scenario CSV file."""
    try:
        lines = scenario_path.read_text(encoding="utf-8").strip().splitlines()
        return max(len(lines) - 1, 1)
    except (OSError, UnicodeDecodeError):
        return 1
from agentigrid.prompts import build_system_prompt, build_user_prompt
from agentigrid.engine.goal_classifier import build_classification_prompts, parse_goal_classification

logger = logging.getLogger("agentigrid.engine.agent_loop")

_MAX_CONSECUTIVE_PARSE_FAILURES = 3


def _bus_limits_from_network(net) -> dict[int, tuple[float, float]]:
    """Extract per-bus (Vmin, Vmax) from a MATNetwork for violation checking."""
    return {b.bus_i: (b.Vmin, b.Vmax) for b in net.buses}


def _build_sweep_llm_view(
    candidate_summaries: list[dict],
    feasible_buses: list[int],
    mut_desc: str,
    objective_name: str,
    objective_direction: str,
    top_n: int,
    threshold: int,
    rank_key: str = "cost",
    near_optimal_abs_tol: float = 5.0,
) -> str:
    """Build the LLM-facing text for a sweep result.

    When candidate_count <= threshold, returns the full per-candidate table
    (byte-identical to the pre-B.1 output for the default cost ranking, so
    small-network baselines are unchanged).  When candidate_count > threshold
    (or threshold == 0), returns a bounded summary: header + feasible bus list +
    infeasible grouped by reason + top-N ranked block + aggregate stats +
    journal pointer.

    ``rank_key`` selects the per-candidate value used for ranking and stats —
    "cost" (default, the OPF objective) or "metric_value" (a C3 custom metric).

    Note: for COST sweeps with a sub-tolerance tie (the two cheapest feasible
    candidates differ by less than ``near_optimal_abs_tol``) a trailing
    near-optimal advisory line is appended in both branches — this intentionally
    breaks the byte-identical guarantee only in that specific case. The advisory
    never appears for non-cost (custom-metric) rankings or when there is no
    sub-tolerance tie.
    """
    n_total = len(candidate_summaries)
    n_feasible = len(feasible_buses)
    n_infeasible = n_total - n_feasible
    num_label = "cost" if rank_key == "cost" else objective_name[:12]

    def _rank_val(s: dict):
        return s.get(rank_key)

    def _near_optimal_advisory() -> "str | None":
        """Advisory when the top two feasible COST candidates are within tolerance."""
        if rank_key != "cost":
            return None
        costs = sorted(
            s["cost"] for s in candidate_summaries
            if s.get("feasible") and isinstance(s.get("cost"), (int, float))
        )
        if len(costs) < 2 or (costs[1] - costs[0]) >= near_optimal_abs_tol:
            return None
        return (
            f"[Near-optimal: the top candidates are within the solver tolerance "
            f"(${near_optimal_abs_tol:g}/h) of each other and are statistically tied — "
            f"report them as an equivalently-optimal set, do not single out one bus as "
            f"uniquely best.]"
        )

    def _fmt_val(val) -> str:
        if not isinstance(val, (int, float)):
            return "           N/A"
        # Preserve the exact cost formatting (byte-identical B.1 baseline);
        # custom metrics use a compact significant-figure format.
        return f"{val:>12,.1f}" if rank_key == "cost" else f"{val:>12,.4g}"

    def _fmt_stat(val) -> str:
        return f"{val:,.1f}" if rank_key == "cost" else f"{val:,.4g}"

    # --- full-table branch (preserves byte-identical output for small networks) ---
    if threshold > 0 and n_total <= threshold:
        infeasible_buses = [s["bus"] for s in candidate_summaries if not s["feasible"]]
        if len(infeasible_buses) > 20:
            infeasible_str = f"[{', '.join(str(b) for b in infeasible_buses[:20])}, ...]"
        else:
            infeasible_str = f"[{', '.join(str(b) for b in infeasible_buses)}]"
        table_lines = [
            f"Sweep over {n_total} candidate buses (mutation: {mut_desc}):",
            f"FEASIBLE: {n_feasible} / {n_total}.  INFEASIBLE buses: {infeasible_str}",
            f"{'bus':>4} | {'feasible':>8} | {'Vmin':>5} | {'Vmax':>5} | {'maxLoad%':>8} | {'viol':>4} | {num_label:>12}",
        ]
        for s in candidate_summaries:
            feas_str = "yes" if s["feasible"] else "no"
            val_str = _fmt_val(_rank_val(s))
            table_lines.append(
                f"{s['bus']:>4} | {feas_str:>8} | {s['voltage_min']:>5.3f} | "
                f"{s['voltage_max']:>5.3f} | {s['max_line_loading_pct']:>8.1f} | "
                f"{s['violations']:>4} | {val_str}"
            )
        _adv = _near_optimal_advisory()
        if _adv:
            table_lines.append(_adv)
        return "\n".join(table_lines)

    # --- summarized view branch ---
    lines = [
        f"Sweep over {n_total} candidate buses (mutation: {mut_desc}):",
        f"FEASIBLE: {n_feasible} / {n_total}   INFEASIBLE: {n_infeasible} / {n_total}",
        "",
    ]

    # Feasible bus list (complete — the LLM needs the full set for set-based reductions)
    feas_list_str = ", ".join(str(b) for b in feasible_buses)
    lines.append(f"Feasible buses ({n_feasible}): [{feas_list_str}]")
    lines.append("")

    # Infeasible buses grouped by reason
    infeas_by_reason: dict[str, list[int]] = defaultdict(list)
    for s in candidate_summaries:
        if not s["feasible"]:
            reason = s.get("reason") or "did not converge"
            infeas_by_reason[reason].append(s["bus"])
    if infeas_by_reason:
        lines.append(f"Infeasible buses grouped by reason ({n_infeasible} total):")
        for reason in sorted(infeas_by_reason):
            buses = infeas_by_reason[reason]
            lines.append(f"  {reason}: {buses}")
        lines.append("")

    # Top-N ranked by the chosen key (cost by default, or a custom metric)
    feasible_with_val = [
        s for s in candidate_summaries
        if s["feasible"] and isinstance(_rank_val(s), (int, float))
    ]
    reverse = (objective_direction == "maximize")
    ranked = sorted(feasible_with_val, key=lambda s: _rank_val(s), reverse=reverse)
    top_candidates = ranked[:top_n]
    n_shown = len(top_candidates)
    lines.append(
        f"Top {n_shown} feasible by {objective_name} ({objective_direction}):"
    )
    lines.append(
        f"{'rank':>4} | {'bus':>4} | {'Vmin':>5} | {'Vmax':>5} | {'maxLoad%':>8} | {num_label:>12}"
    )
    for rank, s in enumerate(top_candidates, 1):
        val_str = _fmt_val(_rank_val(s))
        lines.append(
            f"{rank:>4} | {s['bus']:>4} | {s['voltage_min']:>5.3f} | "
            f"{s['voltage_max']:>5.3f} | {s['max_line_loading_pct']:>8.1f} | {val_str}"
        )
    lines.append("")

    # Aggregate stats over the full feasible set
    vals = [_rank_val(s) for s in feasible_with_val]
    if vals:
        med = statistics.median(vals)
        lines.append(
            f"Feasible {objective_name} stats: "
            f"min={_fmt_stat(min(vals))}  median={_fmt_stat(med)}  max={_fmt_stat(max(vals))}"
        )
        lines.append("")

    # Pointer — tell the LLM that full data is in the journal, nothing is missing
    lines.append(
        f"[Note: Full per-candidate table ({n_total} rows) is stored in the journal "
        "and rendered in the PDF report. Only the summary and top-N are shown here "
        "to limit token usage — no data is missing from the search record.]"
    )

    _adv = _near_optimal_advisory()
    if _adv:
        lines.append(_adv)

    return "\n".join(lines)


def _infeasible_reason(convergence_status: str) -> str:
    """Certified reason for an infeasible sweep candidate.

    Uses solver convergence status as the sole source of truth.
    A non-converged solve certifies no operating point — the solver's last
    iterate is an uncertified diagnostic, not a confirmed cause.

    - CONVERGED + infeasible → certified constraint violation (PFLOW style)
    - anything else          → did not converge
    """
    if (convergence_status or "").upper().startswith("CONVERGED"):
        return "constraint violation"
    return "did not converge"


def _is_certified(opflow) -> bool:
    """Certified-result gate (shared with the C.1 binding-constraint identifier).

    A candidate's solved state is trustworthy only when the solver converged.
    A non-converged solve certifies no operating point, so any value read from
    its last iterate (cost, voltages, a custom metric) is uncertified and must
    not drive a reported answer.
    """
    return (
        opflow is not None
        and (getattr(opflow, "convergence_status", "") or "").upper().startswith("CONVERGED")
    )


def _single_call_record(sim) -> Optional[dict]:
    """Build the reproducible single-call ExaGO invocation record (JSON journal only).

    Returns None when no SimulationResult / no captured argv is available, so a
    failed or missing solve records ``exago_command = null`` rather than a
    misleading partial record.
    """
    if sim is None or not getattr(sim, "argv", None):
        return None
    return {
        "mode": "single",
        "application": getattr(sim, "application", None),
        "command": " ".join(sim.argv),
        "argv": list(sim.argv),
        "shell_command": getattr(sim, "shell_command", None),
        "env_overrides": getattr(sim, "env_overrides", None),
        "cwd": getattr(sim, "cwd", None),
    }


def _multi_call_record(mode: str, candidate_count: int, representative_sim, note: str) -> Optional[dict]:
    """Build the multi-call (sweep/explore) ExaGO invocation record (JSON journal only).

    Logs ONE representative invocation plus the candidate count and a note
    explaining what varied per candidate (rather than every candidate call).
    Returns None when no representative invocation could be captured.
    """
    representative = _single_call_record(representative_sim)
    if representative is None:
        return None
    return {
        "mode": mode,
        "candidate_count": candidate_count,
        "representative": representative,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Boundary (hosting-capacity) search primitives — C.1
#
# For each candidate bus, a bisection on injection magnitude finds the largest
# injection that still yields a feasible OPFLOW solve. "Feasible" under OPFLOW
# means IPOPT converges with the V-band and Rate A as in-solve hard constraints,
# so the boundary located is the *convergence boundary*. Non-convergence is the
# infeasible signal that caps the bisection (standard for OPF hosting capacity).
# ---------------------------------------------------------------------------

def _system_average_tan_phi(net) -> float:
    """System-average tan(phi) = ΣQd / ΣPd over all buses in the base case.

    Used to scale reactive load along a constant-power-factor ray when adding
    active load at a candidate bus. Returns 0.0 if total active load is zero.
    """
    p_total = sum(b.Pd for b in net.buses)
    q_total = sum(b.Qd for b in net.buses)
    if p_total == 0:
        return 0.0
    return q_total / p_total


def _tan_phi_from_pf_spec(pf_spec, tan_phi_avg: float) -> float:
    """Resolve tan(phi) from a power-factor spec.

    - "system_average" / None → tan_phi_avg (computed once from the base case)
    - "unity"                 → 0.0 (no reactive component)
    - numeric 0..1            → tan(acos(pf))
    """
    if pf_spec is None or pf_spec == "system_average":
        return tan_phi_avg
    if isinstance(pf_spec, str):
        if pf_spec == "unity":
            return 0.0
        try:
            pf_spec = float(pf_spec)
        except ValueError:
            return tan_phi_avg
    pf = max(min(float(pf_spec), 1.0), 1e-6)
    return math.tan(math.acos(pf))


def _mutate_candidate_network(
    base_net, bus: int, entity: str, delta_mw: float,
    tan_phi: float, vmin: float, vmax: float, q_frac: float,
):
    """Return a modified copy of the base network with the candidate injection applied.

    Reuses modifier primitives (parse_command + apply_modifications). The voltage
    band is applied on every bus first (in-solve hard constraint), then:
      - load:      add_load_at_bus with Pd=ΔP and Qd=ΔP·tan_phi (constant-PF ray)
      - generator: add_generator_at_bus (forced injection, Pmin=Pmax=ΔP) with the
                   reactive output free within ±q_frac·ΔP.
    """
    raws = [{"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax}]
    if entity == "load":
        raws.append({
            "action": "add_load_at_bus", "bus": bus,
            "Pd": delta_mw, "Qd": delta_mw * tan_phi,
        })
    else:  # generator
        raws.append({
            "action": "add_generator_at_bus", "bus": bus,
            "capacity_mw": delta_mw,
            "Qmax": q_frac * delta_mw, "Qmin": -q_frac * delta_mw,
            "dispatchable": False,
        })
    cmds = [parse_command(r) for r in raws]
    net, _ = apply_modifications(base_net, cmds, application="opflow")
    return net


# Binding-constraint identification tolerances.
_DUAL_EPS = 1e-6        # |Lagrange multiplier| above this ⇒ constraint genuinely active
_BINDING_V_EPS = 0.005  # pu margin to a voltage bound at/below which it is "active"


def _line_loading_pct(br) -> float:
    """Line loading from BOTH ends: max(|Sf|, |St|) / Slim · 100."""
    slim = getattr(br, "Slim", 0.0) or 0.0
    if slim <= 0:
        return 0.0
    return max(abs(getattr(br, "Sf", 0.0)), abs(getattr(br, "St", 0.0))) / slim * 100.0


def _line_dual(br) -> float:
    """Total |line-flow multiplier| (shadow price) on the from/to flow limits."""
    return abs(getattr(br, "mult_Sf", 0.0) or 0.0) + abs(getattr(br, "mult_St", 0.0) or 0.0)


def _thermal_binding(opflow, binding_eps: float) -> str:
    """Thermal binding label at a solved point.

    Branch A (preferred): if any line carries a non-negligible flow-limit
    multiplier, the binder is the genuinely-active line with the largest
    |multiplier| (a line sitting at its rating with a ~zero shadow price is not
    actually constraining and is ignored).

    Branch B (no usable duals): report the GLOBALLY most-loaded line(s) — every
    line within ``binding_eps`` of the system max loading — using both-end
    loading. Never restricted to lines incident to the injection bus.
    """
    branches = [
        b for b in (getattr(opflow, "branches", None) or [])
        if getattr(b, "status", 1) != 0 and (getattr(b, "Slim", 0.0) or 0.0) > 0
    ]
    if not branches:
        return ""

    # Branch A — duals available.
    if any(_line_dual(b) > _DUAL_EPS for b in branches):
        active = [b for b in branches if _line_dual(b) > _DUAL_EPS]
        active.sort(key=lambda b: (_line_dual(b), _line_loading_pct(b)), reverse=True)
        labels = [f"line {b.from_bus}->{b.to_bus} @ {_line_loading_pct(b):.1f}%" for b in active[:3]]
        return "thermal: " + ", ".join(labels)

    # Branch B — value-based, only when something is actually at the limit.
    max_load = max(_line_loading_pct(b) for b in branches)
    if max_load < 100.0 - binding_eps:
        return ""
    binders = [b for b in branches if _line_loading_pct(b) >= max_load - binding_eps]
    binders.sort(key=_line_loading_pct, reverse=True)
    labels = [f"line {b.from_bus}->{b.to_bus} @ {_line_loading_pct(b):.1f}%" for b in binders[:3]]
    return "thermal: " + ", ".join(labels)


def _voltage_binding(opflow, base_opflow, vmin_lim: float, vmax_lim: float, v_eps: float) -> str:
    """Voltage binding label at a solved point (no voltage-bound duals available).

    A bound is reported only when it is *newly active*: the bus voltage is within
    ``v_eps`` of the bound at the boundary AND (when the base solve is available)
    its margin to that bound was clearly positive in the base case. This excludes
    pinned setpoints (e.g. a PV bus regulating to Vmax=1.10 in every solve), whose
    margin is already ~0 in the base and therefore does not *limit* the injection.
    """
    buses = getattr(opflow, "buses", None) or []
    if not buses:
        return ""
    base_v = {b.bus_id: b.Vm for b in (getattr(base_opflow, "buses", None) or [])}

    best: "tuple[float, str] | None" = None
    for b in buses:
        for name, margin, edge in (
            ("Vmin", b.Vm - vmin_lim, vmin_lim),
            ("Vmax", vmax_lim - b.Vm, vmax_lim),
        ):
            if margin > v_eps:
                continue  # not active at the boundary
            if base_v:
                vb = base_v.get(b.bus_id)
                if vb is not None:
                    base_margin = (vb - vmin_lim) if name == "Vmin" else (vmax_lim - vb)
                    if base_margin <= v_eps:
                        continue  # pinned / already at the bound in base — not newly binding
            if best is None or margin < best[0]:
                best = (margin, f"voltage: bus {b.bus_id} {name} @ {b.Vm:.3f} pu")
    return best[1] if best else ""


def _fallback_binding(opflow, vmin_lim: float, vmax_lim: float) -> str:
    """Last-resort label when no constraint is clearly active (tightest normalized slack)."""
    max_load = opflow.max_line_loading_pct or 0.0
    vmin = opflow.voltage_min or 0.0
    vmax = opflow.voltage_max or 0.0
    band = (vmax_lim - vmin_lim) or 1.0
    thermal_norm = max(0.0, 100.0 - max_load) / 100.0
    vhi_norm = max(0.0, vmax_lim - vmax) / band
    vlo_norm = max(0.0, vmin - vmin_lim) / band
    v_norm = min(vhi_norm, vlo_norm)
    if thermal_norm <= v_norm:
        return f"thermal @ {max_load:.1f}%"
    edge = vmin if vlo_norm <= vhi_norm else vmax
    return f"voltage @ {edge:.3f} pu"


def _identify_binding(
    opflow, vmin_lim: float, vmax_lim: float,
    base_opflow=None, binding_eps: float = 0.5,
) -> str:
    """Identify the binding constraint(s) at a feasible (max-feasible) point.

    Activity-based, not value-based: thermal binders come from line-flow
    Lagrange multipliers when available (else the globally most-loaded line),
    and voltage binders only from bounds whose margin *collapsed* relative to
    the base case (pinned setpoints are excluded). When both a thermal line and
    a voltage bound are newly active, both are reported.

    This is a diagnostic label only — it never affects the certified
    ``max_feasible_mw`` capacity.
    """
    if opflow is None:
        return "convergence"

    parts: list[str] = []
    thermal = _thermal_binding(opflow, binding_eps)
    if thermal:
        parts.append(thermal)
    voltage = _voltage_binding(opflow, base_opflow, vmin_lim, vmax_lim, _BINDING_V_EPS)
    if voltage:
        parts.append(voltage)

    if parts:
        return "; ".join(parts)
    return _fallback_binding(opflow, vmin_lim, vmax_lim)


@dataclass
class _ProbeOutcome:
    """Outcome of one OPFLOW probe inside a candidate bisection."""

    feasible: bool
    voltage_min: float = 0.0
    voltage_max: float = 0.0
    max_line_loading_pct: float = 0.0
    violations: int = 0
    status: str = ""
    binding: str = ""


def _boundary_result(max_feasible_mw, outcome, probes_used: int, note: str) -> dict:
    """Assemble the per-candidate boundary result dict from the last feasible probe."""
    if outcome is not None:
        return {
            "max_feasible_mw": max_feasible_mw,
            "binding_constraint": outcome.binding or "convergence",
            "probes_used": probes_used,
            "voltage_min": outcome.voltage_min,
            "voltage_max": outcome.voltage_max,
            "max_line_loading_pct": outcome.max_line_loading_pct,
            "note": note,
        }
    return {
        "max_feasible_mw": max_feasible_mw,
        "binding_constraint": "convergence",
        "probes_used": probes_used,
        "voltage_min": 0.0,
        "voltage_max": 0.0,
        "max_line_loading_pct": 0.0,
        "note": note or "no feasible injection above base",
    }


def _bisect_boundary(
    solve_probe: "Callable[[float], _ProbeOutcome]",
    initial_mw: float,
    max_mw: float,
    tol_mw: float,
    max_probes: int,
) -> dict:
    """Bisect the injection magnitude to find the max feasible MW for one candidate.

    `solve_probe(delta_mw)` performs one OPFLOW solve and returns a `_ProbeOutcome`.
    The base case (ΔP = 0) is assumed feasible (lower bound). The bracket is found
    by exponential search from `initial_mw`; the boundary is then bisected to
    `tol_mw` or until `max_probes` solves are spent.
    """
    probes_used = 0
    lo = 0.0
    last_feasible: "_ProbeOutcome | None" = None
    note = ""

    # --- exponential bracketing ---
    delta = float(initial_mw)
    first_infeasible = None
    while probes_used < max_probes:
        probe_delta = min(delta, float(max_mw))
        outcome = solve_probe(probe_delta)
        probes_used += 1
        if outcome.feasible:
            lo = probe_delta
            last_feasible = outcome
            if probe_delta >= max_mw:
                return _boundary_result(
                    float(max_mw), last_feasible, probes_used,
                    "boundary not reached within cap",
                )
            delta = probe_delta * 2.0
        else:
            first_infeasible = probe_delta
            break

    if first_infeasible is None:
        # Ran out of probe budget while still feasible during bracketing.
        return _boundary_result(
            lo, last_feasible, probes_used,
            "probe budget exhausted during bracketing",
        )

    # --- bisection between lo (feasible) and hi (infeasible) ---
    hi = first_infeasible
    while (hi - lo) > tol_mw and probes_used < max_probes:
        mid = 0.5 * (lo + hi)
        outcome = solve_probe(mid)
        probes_used += 1
        if outcome.feasible:
            lo = mid
            last_feasible = outcome
        else:
            hi = mid

    return _boundary_result(lo, last_feasible, probes_used, note)


def _build_boundary_llm_view(
    candidate_summaries: list[dict],
    mut_desc: str,
    top_n: int,
    threshold: int,
) -> str:
    """Token-bounded LLM-facing text for a boundary sweep (mirrors B.1 gating).

    candidate_count ≤ threshold → full per-candidate hosting-capacity table;
    above threshold (or threshold == 0) → ranked top-N by capacity + grouped
    undetermined buses + aggregate stats + journal pointer.
    """
    n_total = len(candidate_summaries)
    determined = [s for s in candidate_summaries if s.get("max_feasible_mw") is not None]
    undetermined = [s for s in candidate_summaries if s.get("max_feasible_mw") is None]
    n_det, n_undet = len(determined), len(undetermined)

    header = f"{'bus':>5} | {'maxMW':>10} | {'Vmin':>5} | {'Vmax':>5} | {'binding':<34}"

    def _row(s: dict) -> str:
        mfm = s.get("max_feasible_mw")
        mfm_str = f"{mfm:>10,.1f}" if isinstance(mfm, (int, float)) else "       N/A"
        return (
            f"{s['bus']:>5} | {mfm_str} | {s.get('voltage_min', 0):>5.3f} | "
            f"{s.get('voltage_max', 0):>5.3f} | {str(s.get('binding_constraint', ''))[:34]:<34}"
        )

    def _by_capacity_desc(s: dict):
        mfm = s.get("max_feasible_mw")
        return (mfm is None, -(mfm or 0.0))

    # --- full-table branch ---
    if threshold > 0 and n_total <= threshold:
        lines = [
            f"Boundary sweep over {n_total} candidate buses (mutation: {mut_desc}):",
            f"DETERMINED: {n_det} / {n_total}.",
            header,
        ]
        for s in sorted(candidate_summaries, key=_by_capacity_desc):
            lines.append(_row(s))
        return "\n".join(lines)

    # --- summarized view ---
    ranked = sorted(determined, key=_by_capacity_desc)[:top_n]
    lines = [
        f"Boundary sweep over {n_total} candidate buses (mutation: {mut_desc}):",
        f"DETERMINED: {n_det} / {n_total}   UNDETERMINED: {n_undet} / {n_total}",
        "",
        f"Top {len(ranked)} buses by hosting capacity (max feasible MW, descending):",
        header,
    ]
    for s in ranked:
        lines.append(_row(s))
    lines.append("")

    if undetermined:
        groups: dict[str, list[int]] = defaultdict(list)
        for s in undetermined:
            groups[s.get("reason") or "undetermined"].append(s["bus"])
        lines.append(f"Undetermined buses ({n_undet}):")
        for reason in sorted(groups):
            lines.append(f"  {reason}: {groups[reason]}")
        lines.append("")

    caps = [s["max_feasible_mw"] for s in determined
            if isinstance(s.get("max_feasible_mw"), (int, float))]
    if caps:
        med = statistics.median(caps)
        lines.append(
            f"Hosting-capacity stats (MW): "
            f"min={min(caps):,.1f}  median={med:,.1f}  max={max(caps):,.1f}"
        )
        lines.append("")

    lines.append(
        f"[Note: Full per-candidate hosting-capacity table ({n_total} rows) is stored in "
        "the journal and rendered in the PDF report. Only the top-N and summary are shown "
        "here to limit token usage — no data is missing from the search record.]"
    )
    return "\n".join(lines)


def _benchmark_to_dict(bresult) -> dict:
    """Convert a BenchmarkResult to a serializable dict."""
    from agentigrid.engine.benchmark import DispatchComparison, LoadabilityResult
    d: dict = {
        "opflow_converged": bresult.opflow_converged,
        "opflow_objective": bresult.opflow_objective,
        "pflow_best_computed_cost": bresult.pflow_best_computed_cost,
        "cost_gap_pct": bresult.cost_gap_pct,
        "cost_gap_abs": bresult.cost_gap_abs,
        "summary_text": bresult.summary_text,
        "error": bresult.error,
    }
    if bresult.dispatch_comparison:
        d["dispatch_comparison"] = [
            {
                "bus": dc.bus,
                "fuel": dc.fuel,
                "opflow_pg": dc.opflow_pg,
                "pflow_pg": dc.pflow_pg,
                "delta": dc.delta,
                "opflow_pmax": dc.opflow_pmax,
            }
            for dc in bresult.dispatch_comparison
        ]
    if bresult.loadability is not None:
        d["loadability"] = {
            "opflow_max_factor": bresult.loadability.opflow_max_factor,
            "pflow_max_factor": bresult.loadability.pflow_max_factor,
            "gap_pct": bresult.loadability.gap_pct,
            "detail": bresult.loadability.detail,
        }
    return d


@dataclass
class SearchSession:
    """Complete record of a search session."""

    goal: str
    application: str
    base_case_path: Path
    config: AppConfig
    journal: SearchJournal
    start_time: str
    end_time: Optional[str] = None
    termination_reason: str = ""
    final_findings: Optional[dict] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    goal_classification: Optional[dict] = None
    analysis_text: Optional[str] = None
    enforced_vmin: Optional[float] = None
    enforced_vmax: Optional[float] = None
    objective_registry_data: Optional[list[dict]] = None
    preference_history: Optional[list[dict]] = None
    tcopflow_period_data: Optional[list[dict]] = None
    tcopflow_dT_min: float = 0.0
    tcopflow_duration_min: float = 0.0
    tcopflow_is_coupling: bool = True
    sopflow_num_scenarios: int = 0
    benchmark_result: Optional[dict] = None


class AgentLoopController:
    """Drives the iterative LLM-driven search."""

    def __init__(
        self,
        config: AppConfig,
        quiet: bool = False,
        on_iteration: Callable[[int, JournalEntry, str, OPFLOWResult | None], None] | None = None,
        on_phase: Callable[[int, str], None] | None = None,
        on_pause_state: Callable[[bool], None] | None = None,
        on_explore: Callable[[int, list[dict]], None] | None = None,
    ) -> None:
        self._config = config
        self._backend: LLMBackend = create_backend(config.llm)
        self._executor = SimulationExecutor(config.exago, config.output)
        self._journal = SearchJournal()
        self._quiet = quiet
        self._on_iteration = on_iteration
        self._on_phase = on_phase
        self._on_pause_state = on_pause_state
        self._on_explore = on_explore
        self._stop_requested = False

        # Steering
        self._steering_queue: queue.Queue = queue.Queue()
        self._active_steering_directives: list[dict] = []
        self._steering_history: list[dict] = []

        # Sweep dedup (Fix 4): cache results by sweep signature within this session
        # so an identical re-requested sweep is served from cache, not re-solved.
        self._sweep_signature_cache: dict[str, dict] = {}

        # Pause/resume
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # State tracked across iterations
        self._base_network: Optional[MATNetwork] = None
        self._current_network: Optional[MATNetwork] = None
        self._latest_opflow: Optional[OPFLOWResult] = None
        self._base_opflow_result: Optional[OPFLOWResult] = None
        self._latest_results_text: Optional[str] = None
        self._error_feedback: Optional[str] = None
        self._consecutive_parse_failures = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._opflow_results_cache: dict[int, OPFLOWResult] = {}
        self._scopflow_num_contingencies: int = 0
        self._tcopflow_num_steps: int = 0
        self._tcopflow_duration_min: float = 0.0
        self._tcopflow_dT_min: float = 0.0
        self._tcopflow_is_coupling: bool = True
        self._tcopflow_period_data: list[dict] = []
        self._tcopflow_profile_overrides: dict[str, Path] = {}
        self._sopflow_num_scenarios: int = 0
        self._sopflow_scenario_override: Optional[Path] = None
        self._explore_cache: Optional[ExploreCache] = None
        # Sticky session-level load factor (Task 1)
        self._session_load_factor: Optional[float] = config.search.load_factor
        # Benchmark result computed at session start (Task 2)
        self._benchmark_result: Optional[dict] = None

    # ------------------------------------------------------------------
    # Output helper
    # ------------------------------------------------------------------

    def _print(self, msg: str) -> None:
        """Print progress message unless quiet mode is enabled."""
        if not self._quiet:
            print(msg)

    def request_stop(self) -> None:
        """Request graceful termination of the search loop."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Steering & pause/resume API
    # ------------------------------------------------------------------

    def inject_steering(self, directive: str, mode: str = "augment") -> None:
        """Inject a user steering directive into the search.

        Args:
            directive: Natural language instruction from the user.
            mode: "augment" (add to original goal) or "replace" (override goal).
        """
        self._steering_queue.put({"directive": directive, "mode": mode})

    def pause(self) -> None:
        """Pause the search at the next iteration boundary."""
        self._pause_event.clear()
        if self._on_pause_state:
            self._on_pause_state(True)

    def resume(self) -> None:
        """Resume a paused search."""
        self._pause_event.set()
        if self._on_pause_state:
            self._on_pause_state(False)

    def is_paused(self) -> bool:
        """Return True if the search is currently paused."""
        return not self._pause_event.is_set()

    @property
    def steering_history(self) -> list[dict]:
        """Read-only copy of all steering directives injected so far."""
        return list(self._steering_history)

    # ------------------------------------------------------------------
    # Application-specific helpers
    # ------------------------------------------------------------------

    def _build_extra_args(self) -> list[str] | None:
        """Build application-specific extra CLI arguments for the executor."""
        args = []
        app = self._config.search.application

        if app == "scopflow":
            if self._config.search.ctgc_file:
                args.extend(["-ctgcfile", str(self._config.search.ctgc_file)])
                args.extend(["-scopflow_Nc", "-1"])
            if self._config.exago.mpi_np > 1:
                args.extend(["-scopflow_solver", "EMPAR"])

        if app == "tcopflow":
            pload = self._tcopflow_profile_overrides.get("pload_profile") or self._config.search.pload_profile
            qload = self._tcopflow_profile_overrides.get("qload_profile") or self._config.search.qload_profile
            if pload:
                args.extend(["-tcopflow_ploadprofile", str(pload)])
            if qload:
                args.extend(["-tcopflow_qloadprofile", str(qload)])
            if self._config.search.wind_profile:
                args.extend(["-tcopflow_windgenprofile", str(self._config.search.wind_profile)])
            dT = self._config.search.tcopflow_dT
            if dT != 60.0:
                args.extend(["-tcopflow_dT", str(dT)])
            duration = self._config.search.tcopflow_duration
            if duration != 1.0:
                args.extend(["-tcopflow_duration", str(duration)])
            iscoupling = self._config.search.tcopflow_iscoupling
            if iscoupling != 1:
                args.extend(["-tcopflow_iscoupling", str(iscoupling)])

        if app == "sopflow":
            scenario = self._sopflow_scenario_override or self._config.search.scenario_file
            if scenario:
                args.extend(["-scenfile", str(scenario)])
                num_scenarios = _count_scenario_rows(Path(str(scenario)))
                args.extend(["-sopflow_Ns", str(num_scenarios)])
            solver = self._config.search.sopflow_solver
            args.extend(["-sopflow_solver", solver])
            iscoupling = self._config.search.sopflow_iscoupling
            if iscoupling != 0:
                args.extend(["-sopflow_iscoupling", str(iscoupling)])
            if self._config.exago.mpi_np > 1 and solver == "EMPAR":
                pass  # MPI is handled by the executor

        if self._config.search.gic_file:
            args.extend(["-gicfile", str(self._config.search.gic_file)])

        return args if args else None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, base_case: Path, goal: str) -> SearchSession:
        """Execute the full search loop."""
        session_start = time.monotonic()
        session = SearchSession(
            goal=goal,
            application=self._config.search.application,
            base_case_path=base_case,
            config=self._config,
            journal=self._journal,
            start_time=datetime.now().isoformat(),
        )

        # 1. Parse base case
        logger.info("Parsing base case: %s", base_case)
        self._base_network = parse_matpower(base_case)
        self._current_network = self._base_network
        net_summary_text = network_summary(
            self._base_network,
            max_generators=self._config.report.network_summary_max_generators,
        )
        net_metadata_text = network_metadata(self._base_network)
        self._network_metadata_text = net_metadata_text

        # 1b. Run OPFLOW benchmark at session start (if enabled)
        benchmark_text: Optional[str] = None
        if (
            self._config.search.application == "pflow"
            and self._config.search.benchmark_opflow
        ):
            try:
                from agentigrid.engine.benchmark import _run_opflow_on_base_case
                from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
                self._print("[Session] Running OPFLOW baseline for benchmark reference...")
                opflow_baseline = _run_opflow_on_base_case(base_case, self._config)
                if opflow_baseline is not None:
                    opflow_cost = opflow_baseline.objective_value
                    pflow_baseline_cost = None
                    if self._base_network.gencost:
                        pflow_result_tmp = parse_simulation_result_for_app(
                            self._executor.run(self._base_network, "pflow", iteration=-2),
                            "pflow",
                            bus_limits=_bus_limits_from_network(self._base_network),
                        )
                        if pflow_result_tmp is not None:
                            pflow_baseline_cost = pflow_result_tmp.compute_generation_cost(
                                self._base_network.gencost
                            )
                    gap_pct = None
                    gap_abs = None
                    if opflow_cost and pflow_baseline_cost and opflow_cost != 0:
                        gap_abs = pflow_baseline_cost - opflow_cost
                        gap_pct = gap_abs / abs(opflow_cost) * 100
                    from agentigrid.engine.benchmark import _build_dispatch_comparison, DispatchComparison
                    dispatch_cmp = _build_dispatch_comparison(opflow_baseline, pflow_result_tmp if pflow_result_tmp is not None else None)
                    bench_dict: dict = {
                        "opflow_converged": opflow_baseline.converged,
                        "opflow_objective": opflow_cost,
                        "pflow_best_computed_cost": pflow_baseline_cost,
                        "cost_gap_pct": gap_pct,
                        "cost_gap_abs": gap_abs,
                        "dispatch_comparison": [
                            {
                                "bus": dc.bus,
                                "fuel": dc.fuel,
                                "opflow_pg": dc.opflow_pg,
                                "pflow_pg": dc.pflow_pg,
                                "delta": dc.delta,
                                "opflow_pmax": dc.opflow_pmax,
                            }
                            for dc in dispatch_cmp
                        ],
                    }
                    self._benchmark_result = bench_dict
                    self._journal.benchmark_result = bench_dict
                    benchmark_text = format_benchmark_for_prompt(bench_dict)
                    self._print(
                        f"[Session] Benchmark: OPFLOW=${opflow_cost:,.2f}"
                        if opflow_cost else "[Session] Benchmark: OPFLOW converged"
                    )
            except Exception as exc:
                logger.warning("Session-start benchmark failed: %s", exc)

        # Initialize journal load_factor from config
        if self._session_load_factor is not None:
            self._journal.load_factor = self._session_load_factor

        # Build system prompt once (static per session)
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt as _fmt_bench
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=benchmark_text,
            session_load_factor=self._session_load_factor,
        )

        self._current_goal = goal

        # 2. Run base case simulation (iteration 0)
        self._print("[Iter 0] Running base case simulation...")
        sim_result = self._executor.run(
            self._base_network,
            self._config.search.application,
            iteration=0,
            extra_args=self._build_extra_args(),
        )

        # Extract SCOPFLOW metadata (num_contingencies) once from base case
        if self._config.search.application == "scopflow" and sim_result.success:
            from agentigrid.parsers import parse_scopflow_metadata
            meta = parse_scopflow_metadata(sim_result)
            if meta:
                self._scopflow_num_contingencies = meta.get("num_contingencies", 0)

        # Extract TCOPFLOW metadata and period files once from base case
        if self._config.search.application == "tcopflow" and sim_result.success:
            from agentigrid.parsers import parse_tcopflow_metadata, parse_tcopflow_period_files
            meta = parse_tcopflow_metadata(sim_result)
            if meta:
                self._tcopflow_num_steps = meta.get("num_steps", 0)
                self._tcopflow_duration_min = meta.get("duration_min", 0.0)
                self._tcopflow_dT_min = meta.get("dT_min", 0.0)
                self._tcopflow_is_coupling = meta.get("num_coupling_constraints", 0) > 0
            self._tcopflow_period_data = parse_tcopflow_period_files(sim_result.workdir)

        # Extract SOPFLOW metadata once from base case
        if self._config.search.application == "sopflow" and sim_result.success:
            from agentigrid.parsers import parse_sopflow_metadata
            meta = parse_sopflow_metadata(sim_result)
            if meta:
                self._sopflow_num_scenarios = meta.get("num_scenarios", 0)

        # Compute second-stage wind absorption (offered/dispatched/curtailed)
        wind_absorption = None
        if self._config.search.application == "sopflow" and sim_result.success:
            from agentigrid.parsers import compute_wind_absorption
            active_scenario = self._sopflow_scenario_override or self._config.search.scenario_file
            if active_scenario:
                wind_absorption = compute_wind_absorption(sim_result.workdir, Path(str(active_scenario)))

        opflow = parse_simulation_result_for_app(
            sim_result,
            application=self._config.search.application,
            bus_limits=_bus_limits_from_network(self._base_network),
        )
        self._latest_opflow = opflow

        if opflow is not None:
            self._latest_results_text = results_summary_for_app(
                opflow,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                wind_absorption=wind_absorption,
                gencost=self._base_network.gencost if self._config.search.application == "pflow" else None,
            )
            self._base_opflow_result = opflow
            self._opflow_results_cache[0] = opflow
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=opflow,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
                num_steps=self._tcopflow_num_steps,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=self._base_network.gencost if self._config.search.application == "pflow" else None,
                exago_command=_single_call_record(sim_result),
            )
            if self._config.search.application == "pflow":
                computed_cost = opflow.compute_generation_cost(self._base_network.gencost)
                self._print(
                    f"[Iter 0] Base case: {opflow.convergence_status}, "
                    f"computed cost=${computed_cost:,.2f}"
                )
            else:
                self._print(
                    f"[Iter 0] Base case: {opflow.convergence_status}, "
                    f"cost=${opflow.objective_value:,.2f}"
                )
        else:
            self._latest_results_text = None
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=None,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
                num_steps=self._tcopflow_num_steps,
                num_scenarios=self._sopflow_num_scenarios,
                exago_command=_single_call_record(sim_result),
            )
            self._error_feedback = (
                f"Base case simulation failed: {sim_result.error_message or 'unknown error'}"
            )
            self._print(f"[Iter 0] Base case simulation FAILED: {sim_result.error_message}")

        # 2b. Extract initial objectives from goal
        self._extract_initial_objectives(goal)

        # Backfill base case metrics now that objectives are registered
        if self._latest_opflow is not None:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            base_metrics = extract_all_metrics(self._latest_opflow, metric_names)
            if base_metrics and self._journal.latest:
                self._journal.latest.tracked_metrics = base_metrics

        # Notify callback after base case
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(0, latest_entry, "base_case", self._latest_opflow)

        # 3. Agent loop
        max_iter = self._config.search.max_iterations
        for iteration in range(1, max_iter + 1):
            if self._stop_requested:
                session.termination_reason = "user_stopped"
                self._print("\nSearch stopped by user.")
                break
            action_type, should_continue = self._iteration(iteration, goal)
            # Notify callback after each iteration
            if self._on_iteration:
                latest_entry = self._journal.latest
                if latest_entry:
                    self._on_iteration(iteration, latest_entry, action_type, self._latest_opflow)
            if not should_continue:
                if not session.termination_reason:
                    session.termination_reason = "completed"
                break
        else:
            session.termination_reason = "max_iterations"
            self._print(f"\nMax iterations ({max_iter}) reached.")

        if not session.termination_reason:
            session.termination_reason = "completed"

        # Final notification with termination reason
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(
                    latest_entry.iteration,
                    latest_entry,
                    session.termination_reason,
                    self._latest_opflow,
                )

        session.end_time = datetime.now().isoformat()
        session.total_prompt_tokens = self._total_prompt_tokens
        session.total_completion_tokens = self._total_completion_tokens

        # Record the voltage limits that were enforced in the final network state
        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                session.enforced_vmin = min(v[0] for v in limits.values())
                session.enforced_vmax = max(v[1] for v in limits.values())

        # 4. Finalize session
        elapsed = time.monotonic() - session_start
        self._finalize(session, elapsed)
        return session

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _iteration(
        self,
        iteration: int,
        goal: str,
    ) -> tuple[str, bool]:
        """Execute one iteration. Returns (action_type, should_continue)."""
        # Drain the steering queue at the iteration boundary
        new_directives: list[dict] = []
        while True:
            try:
                item = self._steering_queue.get_nowait()
            except queue.Empty:
                break
            directive = item["directive"]
            mode = item["mode"]
            if mode == "replace":
                self._active_steering_directives.clear()
            self._active_steering_directives.append(item)
            self._steering_history.append({"iteration": iteration, **item})
            new_directives.append(item)
            self._print(
                f"[Iter {iteration}] Steering [{mode.upper()}]: \"{directive[:80]}\""
            )

        # Extract objectives from any new steering directives
        for sd in new_directives:
            self._extract_objectives_from_steering(sd["directive"], iteration)

        # Pause: block here until resume() is called
        self._pause_event.wait()

        self._print(
            f"\n{'─' * 50}\n"
            f"[Iter {iteration}] Sending prompt to "
            f"{self._backend.name()} ({self._config.llm.model})..."
        )

        # Assemble and send prompt
        system_prompt, user_prompt = self._assemble_prompt(
            goal,
            self._latest_results_text,
            self._error_feedback,
            steering_directives=self._active_steering_directives or None,
            current_iteration=iteration,
        )
        self._error_feedback = None  # consumed

        if self._on_phase:
            self._on_phase(iteration, "llm_request")

        response = self._backend.complete(system_prompt, user_prompt)
        logger.debug("LLM raw response: %s", response.raw_text[:500])

        # Track tokens
        pt = response.prompt_tokens or 0
        ct = response.completion_tokens or 0
        self._total_prompt_tokens += pt
        self._total_completion_tokens += ct
        if pt or ct:
            self._print(
                f"[Iter {iteration}] Tokens: {pt} prompt + {ct} completion "
                f"(cumulative: ~{self._total_prompt_tokens + self._total_completion_tokens:,})"
            )

        # Parse JSON from response
        if response.json_data is None:
            self._consecutive_parse_failures += 1
            logger.warning(
                "Failed to parse JSON from LLM response (%d/%d): %s",
                self._consecutive_parse_failures,
                _MAX_CONSECUTIVE_PARSE_FAILURES,
                response.json_error,
            )
            self._print(f"[Iter {iteration}] Failed to parse LLM response as JSON")
            if self._consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                self._print(f"[Iter {iteration}] Too many consecutive parse failures — aborting")
                return "error", False
            self._error_feedback = (
                "Failed to parse JSON from your response. "
                "Please respond with a valid JSON object."
            )
            return "error", True

        self._consecutive_parse_failures = 0
        data = response.json_data
        action = data.get("action", "").lower()

        # Dispatch action
        if action == "modify":
            self._explore_cache = None
            return self._handle_modify(iteration, data)
        elif action == "explore":
            return self._handle_explore(iteration, data)
        elif action == "select":
            return self._handle_select(iteration, data)
        elif action == "complete":
            return self._handle_complete(iteration, data)
        elif action == "analyze":
            return self._handle_analyze(iteration, data)
        elif action == "sweep":
            return self._handle_sweep(iteration, data)
        elif action == "set_load_factor":
            return self._handle_set_load_factor(iteration, data)
        else:
            self._print(f"[Iter {iteration}] Unknown action: '{action}'")
            valid = "modify, explore, select, complete, analyze"
            if self._config.search.concurrent_pflow and self._config.search.application == "pflow":
                self._error_feedback = f"Unknown action '{action}'. Valid actions: {valid}."
            elif self._config.search.application == "opflow":
                self._error_feedback = f"Unknown action '{action}'. Valid actions: modify, sweep, complete, analyze."
            else:
                self._error_feedback = f"Unknown action '{action}'. Valid actions: modify, complete, analyze."
            return "error", True

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_modify(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'modify' action from the LLM."""
        description = data.get("description", "No description")
        reasoning = data.get("reasoning", "")
        raw_commands = data.get("commands", [])
        mode = data.get("mode", self._config.search.default_mode)

        self._print(f'[Iter {iteration}] LLM action: modify — "{description}"')

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Choose base network for modifications
        if mode == "fresh":
            base_net = self._base_network
            self._tcopflow_profile_overrides = {}
            self._sopflow_scenario_override = None
        else:
            base_net = self._current_network

        # Auto-inject session load factor if set and not already in commands
        if (
            self._session_load_factor is not None
            and self._config.search.application == "pflow"
            and not any(
                r.get("action", "").lower() == "scale_all_loads" for r in raw_commands
            )
        ):
            raw_commands = [
                {"action": "scale_all_loads", "factor": self._session_load_factor}
            ] + list(raw_commands)

        # Parse and apply commands
        commands = []
        parse_errors = []
        for raw in raw_commands:
            try:
                commands.append(parse_command(raw))
            except ValueError as exc:
                logger.warning("Failed to parse command %s: %s", raw, exc)
                parse_errors.append(f"Invalid command {raw}: {exc}")

        if commands:
            # Build TCOPFLOW profile args for modifier
            _tcopflow_mod_kwargs = {}
            if self._config.search.application == "tcopflow":
                _tcopflow_mod_kwargs = {
                    "pload_profile": self._tcopflow_profile_overrides.get("pload_profile") or self._config.search.pload_profile,
                    "qload_profile": self._tcopflow_profile_overrides.get("qload_profile") or self._config.search.qload_profile,
                    "profile_output_dir": self._config.output.workdir / f"profiles_iter_{iteration:03d}",
                }
            # Build SOPFLOW scenario args for modifier
            _sopflow_mod_kwargs = {}
            if self._config.search.application == "sopflow":
                _sopflow_mod_kwargs = {
                    "scenario_file": self._sopflow_scenario_override or self._config.search.scenario_file,
                    "scenario_output_dir": self._config.output.workdir / f"scenarios_iter_{iteration:03d}",
                }
            modified_net, report = apply_modifications(
                base_net, commands, application=self._config.search.application,
                **_tcopflow_mod_kwargs,
                **_sopflow_mod_kwargs,
            )
            # Store profile overrides from modifier for subsequent iterations
            if report.profile_paths:
                self._tcopflow_profile_overrides.update(report.profile_paths)
            # Store scenario path override from modifier for subsequent iterations
            if report.scenario_paths:
                self._sopflow_scenario_override = report.scenario_paths.get("scenario_file")
            skipped_msgs = []
            for cmd, reasons in report.skipped:
                skipped_msgs.append(f"Skipped {cmd}: {'; '.join(reasons)}")
            applied_count = len(report.applied)
            skipped_count = len(report.skipped) + len(parse_errors)
        else:
            modified_net = base_net
            applied_count = 0
            skipped_count = len(parse_errors)
            skipped_msgs = []

        all_errors = parse_errors + skipped_msgs
        all_warnings = report.warnings if commands else []
        self._print(
            f"[Iter {iteration}] Applied {applied_count} command(s), "
            f"{skipped_count} skipped"
        )

        if all_errors:
            self._error_feedback = "Command errors:\n" + "\n".join(all_errors)

        if all_warnings:
            warning_text = "Warnings:\n" + "\n".join(all_warnings)
            if self._error_feedback:
                self._error_feedback += "\n\n" + warning_text
            else:
                self._error_feedback = warning_text

        # Run simulation
        if self._on_phase:
            self._on_phase(iteration, "running_simulation")
        self._print(f"[Iter {iteration}] Running {self._config.search.application} simulation...")
        sim_result = self._executor.run(
            modified_net,
            self._config.search.application,
            iteration=iteration,
            extra_args=self._build_extra_args(),
        )

        # Parse results
        if self._on_phase:
            self._on_phase(iteration, "parsing_results")
        opflow = parse_simulation_result_for_app(
            sim_result,
            application=self._config.search.application,
            bus_limits=_bus_limits_from_network(modified_net),
        )
        self._latest_opflow = opflow
        if opflow is not None:
            self._opflow_results_cache[iteration] = opflow

        # Parse TCOPFLOW period files after simulation
        if self._config.search.application == "tcopflow" and opflow is not None and sim_result.success:
            from agentigrid.parsers import parse_tcopflow_period_files
            self._tcopflow_period_data = parse_tcopflow_period_files(sim_result.workdir)

        # Compute second-stage wind absorption (offered/dispatched/curtailed)
        wind_absorption = None
        if self._config.search.application == "sopflow" and sim_result.success:
            from agentigrid.parsers import compute_wind_absorption
            active_scenario = self._sopflow_scenario_override or self._config.search.scenario_file
            if active_scenario:
                wind_absorption = compute_wind_absorption(sim_result.workdir, Path(str(active_scenario)))

        if opflow is not None:
            self._latest_results_text = results_summary_for_app(
                opflow,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                wind_absorption=wind_absorption,
                gencost=self._current_network.gencost if self._config.search.application == "pflow" else None,
            )
            self._current_network = modified_net
            if self._config.search.application == "pflow":
                computed_cost = opflow.compute_generation_cost(modified_net.gencost)
                self._print(
                    f"[Iter {iteration}] Simulation completed in "
                    f"{sim_result.elapsed_seconds:.2f}s — "
                    f"{opflow.convergence_status}, computed cost=${computed_cost:,.2f}"
                )
            else:
                self._print(
                    f"[Iter {iteration}] Simulation completed in "
                    f"{sim_result.elapsed_seconds:.2f}s — "
                    f"{opflow.convergence_status}, cost=${opflow.objective_value:,.2f}"
                )
        else:
            self._latest_results_text = None
            error_msg = sim_result.error_message or "unknown error"
            feedback = f"Simulation failed: {error_msg}"
            if self._error_feedback:
                self._error_feedback += "\n" + feedback
            else:
                self._error_feedback = feedback
            self._print(
                f"[Iter {iteration}] Simulation FAILED in "
                f"{sim_result.elapsed_seconds:.2f}s — {error_msg}"
            )

        # Check for LLM-proposed objectives
        proposed = data.get("propose_objectives", [])
        if proposed and isinstance(proposed, list):
            for prop in proposed:
                name = prop.get("name", "")
                if name:
                    entry = ObjectiveEntry(
                        name=name,
                        direction=prop.get("direction", "minimize"),
                        threshold=prop.get("threshold"),
                        priority=prop.get("priority", "secondary"),
                        introduced_at=iteration,
                        source="llm_proposed",
                    )
                    self._journal.objective_registry.register(entry)
                    self._print(
                        f"[Objectives] LLM proposed: {name} ({entry.direction}, {entry.priority})"
                    )
            if proposed:
                self._backfill_metrics()

        # Update journal
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        self._journal.add_from_results(
            iteration=iteration,
            description=description,
            commands=raw_commands,
            opflow_result=opflow,
            sim_elapsed=sim_result.elapsed_seconds,
            llm_reasoning=reasoning,
            mode=mode,
            steering_directive=active_directive,
            num_steps=self._tcopflow_num_steps,
            num_scenarios=self._sopflow_num_scenarios,
            gencost=modified_net.gencost if self._config.search.application == "pflow" else None,
            exago_command=_single_call_record(sim_result),
        )

        # Extract tracked metrics for multi-objective tracking
        if opflow is not None:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            metrics = extract_all_metrics(opflow, metric_names)
            # For PFLOW: override generation_cost with the computed value
            # (extract_all_metrics reads opflow.objective_value, which is 0.0
            # for PFLOW). We have already populated the entry's objective_value
            # and tracked_metrics["generation_cost"] correctly in add_from_results;
            # preserve that value here too.
            if (
                self._config.search.application == "pflow"
                and "generation_cost" in metrics
                and self._journal.latest is not None
                and self._journal.latest.tracked_metrics is not None
                and "generation_cost" in self._journal.latest.tracked_metrics
            ):
                metrics["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
            if metrics and self._journal.latest:
                # Merge: keep any pre-populated values and add new ones
                if self._journal.latest.tracked_metrics:
                    merged = dict(self._journal.latest.tracked_metrics)
                    merged.update(metrics)
                    if "generation_cost" in self._journal.latest.tracked_metrics:
                        merged["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
                    self._journal.latest.tracked_metrics = merged
                else:
                    self._journal.latest.tracked_metrics = metrics

        return "modify", True

    @staticmethod
    def _current_bus_vlimits(net) -> tuple[float | None, float | None] | None:
        """Return the uniform (Vmin, Vmax) on the network, or None if not uniform.

        If all buses share the same Vmin and same Vmax, return that pair.
        Otherwise return None (mixed limits, cannot auto-inject).
        """
        if not net.buses:
            return None
        vmins = {round(b.Vmin, 6) for b in net.buses}
        vmaxs = {round(b.Vmax, 6) for b in net.buses}
        if len(vmins) == 1 and len(vmaxs) == 1:
            return (vmins.pop(), vmaxs.pop())
        return None

    @staticmethod
    def _variant_has_vlimits(raw_cmds: list[dict]) -> bool:
        """Check whether the variant commands include set_all_bus_vlimits."""
        for raw in raw_cmds:
            action = raw.get("action", "").lower()
            if action in ("set_all_bus_vlimits", "set_bus_vlimits"):
                return True
        return False

    def _handle_explore(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle an 'explore' action — evaluate multiple variants concurrently."""
        if not self._config.search.concurrent_pflow or self._config.search.application != "pflow":
            self._print(f"[Iter {iteration}] 'explore' requires --concurrent-pflow with pflow application")
            self._error_feedback = (
                "The 'explore' action requires concurrent PFLOW mode "
                "(--concurrent-pflow flag). Use 'modify' for single-point changes."
            )
            return "error", True

        description = data.get("description", "Neighborhood exploration")
        reasoning = data.get("reasoning", "")
        mode = data.get("mode", self._config.search.default_mode)
        raw_variants = data.get("variants", [])

        self._print(f'[Iter {iteration}] LLM action: explore — "{description}"')

        max_variants = self._config.search.max_variants

        # Validate variant count
        if not isinstance(raw_variants, list) or len(raw_variants) < 2:
            self._print(f"[Iter {iteration}] explore requires at least 2 variants, got {len(raw_variants) if isinstance(raw_variants, list) else 0}")
            self._error_feedback = (
                "The 'explore' action requires at least 2 variants. "
                "Each variant must have a 'label' and 'commands' list."
            )
            return "error", True

        if len(raw_variants) > max_variants:
            self._print(f"[Iter {iteration}] Truncating {len(raw_variants)} variants to max_variants={max_variants}")
            raw_variants = raw_variants[:max_variants]

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Choose base network
        if mode == "fresh":
            base_net = self._base_network
            self._tcopflow_profile_overrides = {}
            self._sopflow_scenario_override = None
        else:
            base_net = self._current_network

        # Auto-inject set_all_bus_vlimits into variants missing it when
        # the current network has non-default voltage limits.
        _vlimits_inject = None
        if self._config.search.application == "pflow" and base_net is not None:
            limits = self._current_bus_vlimits(base_net)
            base_limits = self._current_bus_vlimits(self._base_network)
            if limits is not None and base_limits is not None and limits != base_limits:
                _vlimits_inject = limits
                self._print(
                    f"[Iter {iteration}] Auto-injecting set_all_bus_vlimits "
                    f"(Vmin={limits[0]}, Vmax={limits[1]}) into variants missing it"
                )

        # Parse and apply each variant
        variant_results: dict[str, VariantResult] = {}
        sim_tasks: list[tuple[MATNetwork, str, int, list[str] | None]] = []
        sim_labels: list[str] = []
        # Maps a variant label to its index in sim_tasks (and therefore in
        # results_map). Rejected variants are absent from this mapping.
        sim_idx_by_label: dict[str, int] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = []
        rejected_labels: list[str] = []

        for raw_v in raw_variants:
            label = raw_v.get("label", "")
            if not label:
                label = chr(ord("A") + len(variant_results))
            raw_cmds = raw_v.get("commands", [])
            # LLM-provided description (if any). If absent or just a letter,
            # we auto-generate a human-readable description after apply_modifications.
            _llm_desc = raw_v.get("description")
            v_desc = _llm_desc if _llm_desc and _llm_desc != label else None

            if _vlimits_inject is not None and not self._variant_has_vlimits(raw_cmds):
                raw_cmds = [{"action": "set_all_bus_vlimits", "Vmin": _vlimits_inject[0], "Vmax": _vlimits_inject[1]}] + raw_cmds

            # Auto-inject session load factor if set and not already in variant
            if (
                self._session_load_factor is not None
                and not any(
                    r.get("action", "").lower() == "scale_all_loads" for r in raw_cmds
                )
            ):
                raw_cmds = [
                    {"action": "scale_all_loads", "factor": self._session_load_factor}
                ] + list(raw_cmds)

            commands: list = []
            parse_errors: list[str] = []
            for raw in raw_cmds:
                try:
                    commands.append(parse_command(raw))
                except ValueError as exc:
                    parse_errors.append(f"Variant {label}: Invalid command {raw}: {exc}")

            if parse_errors:
                all_errors.extend(parse_errors)
                continue

            rejected = False
            if commands:
                modified_net, report = apply_modifications(
                    base_net, commands, application=self._config.search.application,
                )
                applied = len(report.applied)
                skipped = len(report.skipped)
                self._print(f"[Iter {iteration}] Variant {label}: {applied} command(s) applied, {skipped} skipped")
                all_warnings.extend(report.warnings)
                if report.skipped:
                    for cmd, reasons in report.skipped:
                        all_errors.append(f"Variant {label}: Skipped {cmd}: {'; '.join(reasons)}")
                _skipped_cmds = list(report.skipped)
                # Auto-generate description from commands if the LLM didn't provide one.
                if v_desc is None:
                    v_desc = build_variant_description(commands, _skipped_cmds)
                # If every command in this variant was a no-op against the
                # base case, it would simulate to the base result and waste
                # a slot. Reject pre-execution and surface why.
                if applied == 0:
                    rejected = True
                    rejected_labels.append(label)
                    v_desc = f"[REJECTED — all commands no-op] {v_desc}"
                    self._print(
                        f"[Iter {iteration}] Variant {label}: REJECTED — all "
                        f"{skipped} command(s) would be no-ops against the base case"
                    )
            else:
                modified_net = base_net
                _skipped_cmds = []
                if v_desc is None:
                    v_desc = "(no commands)"

            if not rejected:
                # Use negative iteration numbers to create distinct workdirs for each variant
                variant_iter = -(iteration * max_variants + len(sim_tasks))

                gencost_for_summary = modified_net.gencost if self._config.search.application == "pflow" else None

                sim_idx_by_label[label] = len(sim_tasks)
                sim_tasks.append((
                    modified_net,
                    self._config.search.application,
                    variant_iter,
                    self._build_extra_args(),
                ))
                sim_labels.append(label)

            variant_results[label] = VariantResult(
                label=label,
                description=v_desc,
                commands=commands,
                raw_commands=raw_cmds,
                modified_net=modified_net,
                sim_result=None,
                opflow_result=None,
                skipped_commands=_skipped_cmds,
                rejected=rejected,
            )

        if len(sim_tasks) < 2:
            self._error_feedback = (
                "Not enough valid variants to explore. At least 2 variants with "
                "at least one effective (non-no-op) command are required."
            )
            if rejected_labels:
                self._error_feedback += (
                    f"\nRejected variants ({len(rejected_labels)}): "
                    f"{', '.join(rejected_labels)} — every proposed command "
                    "was a no-op against the base case."
                )
            if all_errors:
                self._error_feedback += "\n" + "\n".join(all_errors[:5])
            return "error", True

        # Run simulations concurrently
        if self._on_phase:
            if len(sim_tasks) > 1:
                self._on_phase(iteration, f"running_simulation ({len(sim_tasks)} variants)")
            else:
                self._on_phase(iteration, "running_simulation")
        self._print(f"[Iter {iteration}] Running {len(sim_tasks)} simulations concurrently...")

        results_map = self._executor.run_parallel(
            sim_tasks, max_workers=min(self._config.search.max_variants, len(sim_tasks)),
        )

        # Parse results — only for variants that were actually simulated
        # (rejected variants have no sim_idx entry and stay with empty results).
        for label, v in variant_results.items():
            if v.rejected:
                continue
            sim_idx = sim_idx_by_label.get(label)
            if sim_idx is None:
                continue
            sim_result = results_map.get(sim_idx)
            if sim_result is None:
                continue
            v.sim_result = sim_result

            opflow = parse_simulation_result_for_app(
                sim_result,
                application=self._config.search.application,
                bus_limits=_bus_limits_from_network(v.modified_net),
            )
            v.opflow_result = opflow

            if opflow is not None:
                self._print(
                    f"[Iter {iteration}] Variant {label}: {opflow.convergence_status}, "
                    f"feasibility={opflow.feasibility_detail}"
                )
            else:
                err_msg = sim_result.error_message or "simulation failed"
                self._print(f"[Iter {iteration}] Variant {label}: FAILED — {err_msg}")

        # Compute Pareto front
        if self._on_phase:
            self._on_phase(iteration, "computing_pareto_front")
        gencost = base_net.gencost if self._config.search.application == "pflow" else None
        pareto_labels = compute_pareto_labels(
            variant_results, self._journal.objective_registry.objectives, gencost,
        )

        # Flag identical-cost siblings and collect batch-level warning
        sibling_warning = annotate_cost_equivalent_siblings(variant_results, gencost)
        if sibling_warning:
            if self._error_feedback:
                self._error_feedback += "\n\n" + sibling_warning
            else:
                self._error_feedback = sibling_warning

        # Update session-best with any feasible variant cheaper than the current best
        if gencost is not None:
            for lbl, v in variant_results.items():
                if v.rejected or v.opflow_result is None:
                    continue
                if not (v.opflow_result.feasibility_detail == "feasible"
                        and v.opflow_result.num_violations == 0):
                    continue
                try:
                    cost = v.opflow_result.compute_generation_cost(gencost)
                except Exception:
                    continue
                if cost > 0:
                    self._journal.update_session_best(
                        label=lbl,
                        iteration=iteration,
                        cost=cost,
                        commands=v.raw_commands,
                    )

        # Build results text
        self._latest_results_text = format_variant_results(
            variant_results, pareto_labels, gencost,
        )

        # Store explore cache
        self._explore_cache = ExploreCache(
            variants=variant_results,
            description=description,
            reasoning=reasoning,
            iteration=iteration,
            base_network_snapshot=base_net,
            base_mode=mode,
        )

        self._latest_opflow = None
        if pareto_labels and pareto_labels[0] in variant_results:
            best = variant_results[pareto_labels[0]]
            if best.opflow_result is not None:
                self._latest_opflow = best.opflow_result

        self._print(
            f"[Iter {iteration}] Explored {len(variant_results)} variants. "
            f"Pareto: {', '.join(pareto_labels) if pareto_labels else 'none'}"
        )

        # Notify UI about explore results
        if self._on_explore:
            variant_summaries = []
            for lbl, v in variant_results.items():
                summary = {"label": lbl, "feasible": False}
                if v.opflow_result is not None:
                    summary["feasible"] = (
                        v.opflow_result.feasibility_detail == "feasible"
                        and v.opflow_result.num_violations == 0
                    )
                    summary["convergence_status"] = v.opflow_result.convergence_status
                    summary["voltage_min"] = v.opflow_result.voltage_min
                    summary["voltage_max"] = v.opflow_result.voltage_max
                    summary["max_line_loading_pct"] = v.opflow_result.max_line_loading_pct
                    summary["violations_count"] = v.opflow_result.num_violations
                summary["is_pareto"] = v.is_pareto
                variant_summaries.append(summary)
            self._on_explore(iteration, variant_summaries)

        if all_warnings:
            warning_text = "Warnings:\n" + "\n".join(all_warnings[:5])
            if self._error_feedback:
                self._error_feedback += "\n\n" + warning_text
            else:
                self._error_feedback = warning_text

        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )

        variant_labels = sorted(variant_results.keys())
        variant_info = []
        for lbl in variant_labels:
            v = variant_results[lbl]
            info = {"label": lbl, "description": v.description, "commands": v.raw_commands, "is_pareto": v.is_pareto}
            if v.rejected:
                info["rejected"] = True
            if v.skipped_commands:
                skipped_summaries = []
                for cmd, reasons in v.skipped_commands:
                    cmd_name = type(cmd).__name__
                    reason_str = "; ".join(reasons)
                    skipped_summaries.append(f"{cmd_name}: {reason_str}")
                info["skipped"] = skipped_summaries
            if v.opflow_result is not None:
                info["feasible"] = (
                    v.opflow_result.feasibility_detail == "feasible"
                    and v.opflow_result.num_violations == 0
                )
                info["voltage_min"] = v.opflow_result.voltage_min
                info["voltage_max"] = v.opflow_result.voltage_max
                info["max_line_loading_pct"] = v.opflow_result.max_line_loading_pct
                info["violations_count"] = v.opflow_result.num_violations
                if gencost is not None:
                    try:
                        info["cost"] = v.opflow_result.compute_generation_cost(gencost)
                    except Exception:
                        pass
            else:
                info["feasible"] = False
            variant_info.append(info)

        _explore_rep = next(
            (variant_results[lbl].sim_result for lbl in variant_labels
             if getattr(variant_results[lbl], "sim_result", None) is not None),
            None,
        )
        _explore_command = _multi_call_record(
            "explore", len(variant_labels), _explore_rep,
            "Executed once per explored variant. Each variant's command set is written "
            "into its own netfile; only the -netfile path differs between variants. The "
            "selected variant's exact invocation is recorded separately on its own "
            "iteration entry.",
        )
        self._journal.add_explore(
            iteration=iteration,
            description=f"[explore] {description}",
            variant_info=variant_info,
            pareto_labels=pareto_labels,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
            exago_command=_explore_command,
        )

        return "explore", True

    def _resolve_sweep_workers(self, n_tasks: int) -> int:
        """Resolve the sweep concurrency from config (0 = auto)."""
        raw = self._config.search.sweep_max_workers
        workers = raw if (raw and raw > 0) else min(os.cpu_count() or 4, 16)
        return max(1, min(workers, n_tasks))

    def _resolve_candidate_buses(self, spec: dict) -> tuple[Optional[list[int]], str]:
        """Resolve a sweep candidate_set spec to a sorted list of bus ids.

        Returns (candidates, "") on success or (None, error_message) on failure.
        Shared by the feasibility sweep and the boundary sweep.
        """
        ctype = spec.get("type", "")
        if ctype == "all_buses":
            candidates = [b.bus_i for b in self._base_network.buses]
        elif ctype == "load_buses":
            candidates = [b.bus_i for b in self._base_network.buses if b.Pd != 0]
        elif ctype == "bus_list":
            raw_buses = spec.get("buses", [])
            if not isinstance(raw_buses, list) or not raw_buses:
                return None, "candidate_set type='bus_list' requires a non-empty 'buses' list."
            candidates = [int(b) for b in raw_buses]
        else:
            return None, (
                f"Unknown candidate_set type '{ctype}'. "
                "Supported: 'all_buses', 'load_buses', 'bus_list'."
            )
        return sorted(candidates), ""

    def _augment_generator_mutation(
        self, mutation_template: dict, data: dict, predicate_name: "str | None",
    ) -> "dict | None":
        """Apply C2 (dispatchable mode + cost curve) and the C3 reactive-adequacy
        Q-pin to an add_generator_at_bus sweep mutation.

        Returns the (possibly modified) mutation template, or None on a
        configuration error (with ``self._error_feedback`` set).
        """
        if mutation_template.get("action") != "add_generator_at_bus":
            return mutation_template

        mt = dict(mutation_template)
        s = self._config.search

        # Dispatchable vs fixed-injection mode. Explicit in the mutation wins;
        # then the action-level entity_dispatchable; then the config default.
        if "dispatchable" not in mt:
            if data.get("entity_dispatchable") is not None:
                mt["dispatchable"] = bool(data["entity_dispatchable"])
            else:
                mt["dispatchable"] = bool(s.added_gen_dispatchable_default)

        # Cost curve. Explicit coeffs (mutation or action) win; otherwise the
        # "median_existing" strategy lets the modifier default to the case
        # median; "explicit" requires coeffs for a dispatchable unit.
        cost_coeffs = mt.get("cost_coeffs", data.get("entity_cost_coeffs"))
        if cost_coeffs is not None:
            mt["cost_coeffs"] = cost_coeffs
        elif mt.get("dispatchable") and s.added_gen_cost_strategy == "explicit":
            self._error_feedback = (
                "added_gen_cost_strategy='explicit' requires entity_cost_coeffs "
                "(or mutation.cost_coeffs) for a dispatchable generator sweep."
            )
            return None
        # else: median strategy — modifier supplies the case-median curve.

        # C3 reactive adequacy: pin reactive output to Qmax (Qmin = Qmax) so the
        # unit is forced to (P = Pmax, Q = Qmax) for the headroom test.
        if predicate_name == "reactive_adequacy" and mt.get("Qmax") is not None:
            mt["Qmin"] = mt["Qmax"]

        return mt

    # Sweep-defining fields: two requests with identical values for these produce
    # byte-identical results (description/reasoning are excluded — re-phrasing the
    # same sweep still dedups).
    _SWEEP_SIGNATURE_KEYS = (
        "mode", "entity", "power_factor", "mutation", "candidate_set", "feasibility",
        "metric", "feasibility_predicate", "entity_dispatchable", "entity_cost_coeffs",
        # C.5 contingency screen — an identical study is deterministic and served from cache.
        "target_bus", "neighbor_count", "contingency_order", "components",
        # C.8 Path A reserve minimization — the minimize flag changes the study.
        "minimize",
    )

    def _sweep_cache_key(self, data: dict) -> str:
        """Stable signature for a sweep request (Fix 4 dedup)."""
        payload = {k: data.get(k) for k in self._SWEEP_SIGNATURE_KEYS}
        return json.dumps(payload, sort_keys=True, default=str)

    def _store_sweep_cache(
        self, cache_key: str, description: str, candidate_count: int,
        candidate_summaries: list[dict], feasible_buses: list[int], results_text: str | None,
    ) -> None:
        """Record a completed sweep so an identical re-request is not re-solved."""
        self._sweep_signature_cache[cache_key] = {
            "description": description,
            "candidate_count": candidate_count,
            "candidate_summaries": candidate_summaries,
            "feasible_buses": feasible_buses,
            "results_text": results_text,
        }

    def _serve_cached_sweep(
        self, iteration: int, data: dict, cache_key: str
    ) -> tuple[str, bool]:
        """Serve a previously-computed identical sweep without re-solving (Fix 4)."""
        cached = self._sweep_signature_cache[cache_key]
        note = (
            "\n\n[Cached: an identical sweep already ran this session — results are "
            "deterministic and were not re-solved. Proceed to the answer.]"
        )
        self._latest_results_text = (cached.get("results_text") or "") + note
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        self._journal.add_sweep(
            iteration=iteration,
            description=f"[sweep] {cached['description']} (cached — identical sweep, not re-solved)",
            candidate_count=cached["candidate_count"],
            candidate_summaries=cached["candidate_summaries"],
            feasible_buses=cached["feasible_buses"],
            llm_reasoning=data.get("reasoning", ""),
            steering_directive=active_directive,
        )
        self._print(
            f"[Iter {iteration}] Sweep cache hit — returning prior result, no re-solve "
            f"({cached['candidate_count']} candidates)"
        )
        return "sweep", True

    def _handle_sweep(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'sweep' action — test one mutation across a candidate bus set in parallel."""
        if self._config.search.application != "opflow":
            self._print(f"[Iter {iteration}] 'sweep' is only supported for OPFLOW")
            self._error_feedback = (
                "The 'sweep' action is only supported for the OPFLOW application. "
                "Use 'modify' for other applications."
            )
            return "error", True

        # Fix 4: a sweep is deterministic — if an identical signature already ran
        # this session, serve the cached result instead of re-solving. Covers both
        # the feasibility/metric path and the boundary path (checked before dispatch).
        cache_key = self._sweep_cache_key(data)
        if cache_key in self._sweep_signature_cache:
            return self._serve_cached_sweep(iteration, data, cache_key)

        # Boundary (hosting-capacity) mode: per-candidate bisection on injection magnitude.
        if data.get("mode") == "boundary":
            return self._handle_boundary_sweep(iteration, data)

        # Contingency (N-1/N-2) mode: enumerate outages on the target's neighbors,
        # re-solve each on top of the current operating point, tabulate pass/fail.
        if data.get("mode") == "contingency":
            return self._handle_contingency_sweep(iteration, data)

        # Hot-reserve / minimum N-1 generator security (C.8): system-wide gen screen.
        if data.get("mode") == "reserve":
            return self._handle_reserve_screen(iteration, data)

        description = data.get("description", "Parametric sweep")
        reasoning = data.get("reasoning", "")
        candidate_set_spec = data.get("candidate_set", {})
        mutation_template = data.get("mutation", {})
        feasibility_spec = data.get("feasibility", {})

        if not mutation_template or "action" not in mutation_template:
            self._error_feedback = "sweep requires a 'mutation' dict with an 'action' key."
            return "error", True

        # C3: resolve the (optional) custom metric and feasibility predicate by name.
        metric_name = data.get("metric")
        predicate_name = data.get("feasibility_predicate")
        if metric_name is not None and metric_name not in sweep_metrics.METRICS:
            self._error_feedback = (
                f"Unknown sweep metric '{metric_name}'. "
                f"Available: {sorted(sweep_metrics.METRICS)}"
            )
            return "error", True
        if predicate_name is not None and predicate_name not in sweep_metrics.PREDICATES:
            self._error_feedback = (
                f"Unknown feasibility_predicate '{predicate_name}'. "
                f"Available: {sorted(sweep_metrics.PREDICATES)}"
            )
            return "error", True
        predicate_fn = sweep_metrics.PREDICATES[predicate_name or "standard"]
        metric_fn = sweep_metrics.METRICS[metric_name] if metric_name else None

        # C2: economic (dispatchable) generator siting + cost curve, plus the C3
        # reactive-adequacy Q-pin. Augment the mutation template once for all candidates.
        mutation_template = self._augment_generator_mutation(
            mutation_template, data, predicate_name,
        )
        if mutation_template is None:
            return "error", True  # _error_feedback already set
        is_dispatchable_gen = (
            mutation_template.get("action") == "add_generator_at_bus"
            and bool(mutation_template.get("dispatchable"))
        )
        is_reactive_adequacy_gen = (
            predicate_name == "reactive_adequacy"
            and mutation_template.get("action") == "add_generator_at_bus"
        )

        # 1. Resolve candidate buses
        candidates, cand_err = self._resolve_candidate_buses(candidate_set_spec)
        if candidates is None:
            self._error_feedback = f"sweep {cand_err}"
            return "error", True

        if not candidates:
            self._error_feedback = "Sweep resolved to an empty candidate set."
            return "error", True

        self._print(
            f'[Iter {iteration}] LLM action: sweep — "{description}" '
            f"({len(candidates)} candidates, mutation: {mutation_template.get('action', '?')})"
        )

        vmin = feasibility_spec.get("Vmin", 0.9)
        vmax = feasibility_spec.get("Vmax", 1.1)

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Pre-compute bus limits after applying the vlimits command (same for all
        # candidates). On success, _vlimits_net already carries the all-bus vlimits,
        # so each candidate applies ONLY its single per-bus mutation (COW). If this
        # falls back to base_network, the per-candidate path re-applies vlimits.
        _vlimits_applied = True
        try:
            _vlimits_parsed = parse_command(
                {"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax}
            )
            _vlimits_net, _ = apply_modifications(
                self._base_network, [_vlimits_parsed], application="opflow",
            )
            bus_limits_for_sweep = _bus_limits_from_network(_vlimits_net)
        except Exception:
            _vlimits_net = self._base_network
            bus_limits_for_sweep = _bus_limits_from_network(self._base_network)
            _vlimits_applied = False

        # C3: metrics like max_delta_v need the base-case operating point. Solve it
        # once (sequentially, before the parallel batch) so every candidate compares
        # against the same reference.
        base_result = None
        if sweep_metrics.metric_needs_base(metric_name):
            if self._on_phase:
                self._on_phase(iteration, "running_simulation (base reference solve)")
            base_sim = self._executor.run(
                _vlimits_net, "opflow", -(iteration * 10000 + 99999),
                self._build_extra_args(), None,
            )
            if base_sim is not None:
                base_result = parse_simulation_result_for_app(
                    base_sim, application="opflow", bus_limits=bus_limits_for_sweep,
                )

        # 2. Build per-candidate sim tasks.
        # The per-candidate base (_vlimits_net) is INVARIANT across candidates, so its
        # O(1) index maps are built ONCE here and reused in every apply_modifications
        # call below — avoiding an O(N) map rebuild per candidate (the O(N²) regression).
        sweep_maps = build_index_maps(_vlimits_net)

        sim_tasks: list[tuple[MATNetwork, str, int, list[str] | None]] = []
        cand_indices_for_tasks: list[int] = []  # task_idx → candidate_idx
        skipped_cand_indices: set[int] = set()
        build_errors: list[str] = []

        for cand_idx, bus_id in enumerate(candidates):
            # Hoist: when _vlimits_net already carries the all-bus vlimits, apply
            # only the single per-bus mutation here (COW makes this ~O(1)). The
            # vlimits-parse-failure fallback re-applies vlimits per candidate.
            if _vlimits_applied:
                raw_cmds = [dict(mutation_template, bus=bus_id)]
            else:
                raw_cmds = [
                    {"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax},
                    dict(mutation_template, bus=bus_id),
                ]
            commands = []
            parse_ok = True
            for raw in raw_cmds:
                try:
                    commands.append(parse_command(raw))
                except ValueError as exc:
                    build_errors.append(f"Bus {bus_id}: {exc}")
                    parse_ok = False
                    break

            if not parse_ok:
                skipped_cand_indices.add(cand_idx)
                continue

            modified_net, _ = apply_modifications(
                _vlimits_net, commands, application="opflow", copy_mode="cow",
                index_maps=sweep_maps,
            )
            cand_iter = -(iteration * 10000 + cand_idx)
            sim_tasks.append((modified_net, "opflow", cand_iter, self._build_extra_args()))
            cand_indices_for_tasks.append(cand_idx)

        if not sim_tasks:
            self._error_feedback = (
                "All sweep candidates failed to build commands. Errors:\n"
                + "\n".join(build_errors[:5])
            )
            return "error", True

        # 3. Run in parallel
        workers = self._resolve_sweep_workers(len(sim_tasks))
        thread_limit = 1 if workers > 1 else None

        if self._on_phase:
            self._on_phase(
                iteration, f"running_simulation ({len(sim_tasks)} sweep candidates)"
            )
        self._print(
            f"[Iter {iteration}] Running {len(sim_tasks)} sweep simulations "
            f"({workers} concurrent)..."
        )

        def _sweep_progress(done: int, total: int) -> None:
            if self._on_phase:
                self._on_phase(iteration, f"running_simulation (sweep {done}/{total} solved)")

        results_map = self._executor.run_parallel(
            sim_tasks,
            max_workers=workers,
            thread_limit=thread_limit,
            on_progress=_sweep_progress,
        )

        # Map task index back to candidate index
        results_by_cand: dict[int, "SimulationResult"] = {}
        for task_idx, cand_idx in enumerate(cand_indices_for_tasks):
            r = results_map.get(task_idx)
            if r is not None:
                results_by_cand[cand_idx] = r

        # 4. Parse + classify each result (in sorted candidate order)
        candidate_summaries: list[dict] = []
        feasible_buses: list[int] = []
        first_feasible_opflow: "OPFLOWResult | None" = None
        first_opflow: "OPFLOWResult | None" = None

        for cand_idx, bus_id in enumerate(candidates):
            if cand_idx in skipped_cand_indices:
                candidate_summaries.append({
                    "bus": bus_id,
                    "feasible": False,
                    "voltage_min": 0.0,
                    "voltage_max": 0.0,
                    "max_line_loading_pct": 0.0,
                    "violations": 0,
                    "cost": None,
                    "status": "BUILD_ERROR",
                    "reason": "build error",
                })
                continue

            sim_result = results_by_cand.get(cand_idx)
            opflow = None
            if sim_result is not None:
                opflow = parse_simulation_result_for_app(
                    sim_result,
                    application="opflow",
                    bus_limits=bus_limits_for_sweep,
                )

            if first_opflow is None and opflow is not None:
                first_opflow = opflow

            # C3: feasibility is decided by the selected predicate (default "standard").
            ctx = {"bus": bus_id, "vmin": vmin, "vmax": vmax, "params": data}
            is_feasible, reason = predicate_fn(opflow, base_result, ctx)

            if is_feasible:
                feasible_buses.append(bus_id)
                if first_feasible_opflow is None:
                    first_feasible_opflow = opflow

            summary = {
                "bus": bus_id,
                "feasible": is_feasible,
                "voltage_min": opflow.voltage_min if opflow else 0.0,
                "voltage_max": opflow.voltage_max if opflow else 0.0,
                "max_line_loading_pct": opflow.max_line_loading_pct if opflow else 0.0,
                "violations": opflow.num_violations if opflow else 0,
                "cost": opflow.objective_value if opflow else None,
                "status": opflow.convergence_status if opflow else "FAILED",
                "reason": reason,
            }
            # C3: custom metric value + the metric/predicate names (only when set,
            # so the default sweep path journals an unchanged payload).
            # Fix 1: a metric read from a candidate's solved state is trustworthy
            # only if that candidate is certified (CONVERGED). For non-converged
            # candidates metric_value is None, so the LLM reduction and the report
            # never present an uncertified iterate's metric as the answer.
            if metric_fn is not None:
                summary["metric_name"] = metric_name
                summary["metric_value"] = (
                    metric_fn(opflow, base_result, ctx) if _is_certified(opflow) else None
                )
            if predicate_name is not None:
                summary["predicate_name"] = predicate_name
            # C2: record the dispatched Pg of the newly added unit (diagnostic — a
            # unit dispatching ~0 MW is not helping at that location).
            if is_dispatchable_gen and opflow is not None:
                gens_at_bus = [g for g in opflow.generators if g.bus == bus_id]
                summary["dispatched_pg"] = gens_at_bus[-1].Pg if gens_at_bus else None
            # Fix 3: reactive-adequacy forces Q = Qmax at P = Pmax; record the
            # solved reactive output so the forcing is auditable (should equal the
            # Qmax target at every certified/adequate bus).
            if is_reactive_adequacy_gen:
                gens_at_bus = [g for g in opflow.generators if g.bus == bus_id] if opflow else []
                summary["dispatched_q"] = (
                    gens_at_bus[-1].Qg if (gens_at_bus and _is_certified(opflow)) else None
                )

            candidate_summaries.append(summary)

        self._latest_opflow = first_feasible_opflow or first_opflow

        # 5. Build LLM-facing results view (token-bounded for large networks)
        mut_action = mutation_template.get("action", "unknown")
        mut_desc_parts = [mut_action]
        if "capacity_mw" in mutation_template:
            mut_desc_parts.append(f"{mutation_template['capacity_mw']} MW")
            mut_desc_parts.append("dispatchable" if is_dispatchable_gen else "forced injection")
        if "Pd" in mutation_template:
            mut_desc_parts.append(f"Pd={mutation_template['Pd']} MW")

        mut_desc = " ".join(str(p) for p in mut_desc_parts)

        # Ranking key/label: a custom metric ranks by its own value/direction;
        # otherwise rank by the primary objective (cost).
        if metric_name is not None:
            rank_key = "metric_value"
            objective_name = metric_name
            objective_direction = sweep_metrics.metric_direction(metric_name)
        else:
            rank_key = "cost"
            primary_objs = self._journal.objective_registry.get_primary()
            if primary_objs:
                _obj = primary_objs[0]
                objective_name = _obj.name
                objective_direction = "minimize" if _obj.direction == "constraint" else _obj.direction
            else:
                objective_name = "cost"
                objective_direction = "minimize"

        self._latest_results_text = _build_sweep_llm_view(
            candidate_summaries=candidate_summaries,
            feasible_buses=feasible_buses,
            mut_desc=mut_desc,
            objective_name=objective_name,
            objective_direction=objective_direction,
            top_n=self._config.search.sweep_llm_top_n,
            threshold=self._config.search.sweep_full_table_threshold,
            rank_key=rank_key,
            near_optimal_abs_tol=self._config.report.near_optimal_abs_tol,
        )

        # 6. Journal
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        _representative_sim = next(iter(results_by_cand.values()), None)
        _sweep_command = _multi_call_record(
            "sweep", len(candidates), _representative_sim,
            "Executed once per candidate bus. The per-bus modification "
            "(e.g. +100 MW at the candidate bus) is written into the per-candidate "
            "netfile, not passed as an ExaGO argument; only the -netfile path differs "
            "between candidates.",
        )
        self._journal.add_sweep(
            iteration=iteration,
            description=f"[sweep] {description}",
            candidate_count=len(candidates),
            candidate_summaries=candidate_summaries,
            feasible_buses=feasible_buses,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
            exago_command=_sweep_command,
        )
        self._store_sweep_cache(
            self._sweep_cache_key(data), description, len(candidates),
            candidate_summaries, feasible_buses, self._latest_results_text,
        )

        self._print(
            f"[Iter {iteration}] Sweep complete: {len(feasible_buses)}/{len(candidates)} feasible buses"
        )

        return "sweep", True

    # ------------------------------------------------------------------
    # Boundary (hosting-capacity) sweep — C.1
    # ------------------------------------------------------------------

    def _bisect_candidate(
        self,
        bus: int,
        entity: str,
        pf_spec,
        tan_phi_avg: float,
        vmin: float,
        vmax: float,
        bus_limits: dict,
        q_frac: float,
        base_iter_tag: int,
        thread_limit: int | None,
        base_opflow=None,
    ) -> dict:
        """Find the max feasible injection (MW) at one candidate bus via bisection.

        Each probe mutates the base network (helper) and runs ONE synchronous
        OPFLOW solve. Feasible iff converged with 0 violations. Returns the
        boundary result dict (max_feasible_mw, binding_constraint, probes_used,
        boundary-point metrics).
        """
        s = self._config.search
        tan_phi = _tan_phi_from_pf_spec(pf_spec, tan_phi_avg) if entity == "load" else 0.0
        probe_counter = {"n": 0}

        def solve_probe(delta_mw: float) -> _ProbeOutcome:
            probe_counter["n"] += 1
            net = _mutate_candidate_network(
                self._base_network, bus, entity, delta_mw, tan_phi, vmin, vmax, q_frac,
            )
            cand_iter = -(base_iter_tag + probe_counter["n"])
            sim_result = self._executor.run(
                net, "opflow", cand_iter, self._build_extra_args(), thread_limit,
            )
            opflow = None
            if sim_result is not None:
                opflow = parse_simulation_result_for_app(
                    sim_result, application="opflow", bus_limits=bus_limits,
                )
            if opflow is None:
                return _ProbeOutcome(feasible=False, status="FAILED")
            feasible = (
                opflow.feasibility_detail == "feasible" and opflow.num_violations == 0
            )
            return _ProbeOutcome(
                feasible=feasible,
                voltage_min=opflow.voltage_min,
                voltage_max=opflow.voltage_max,
                max_line_loading_pct=opflow.max_line_loading_pct,
                violations=opflow.num_violations,
                status=opflow.convergence_status,
                binding=_identify_binding(opflow, vmin, vmax, base_opflow) if feasible else "",
            )

        result = _bisect_boundary(
            solve_probe,
            s.boundary_initial_mw,
            s.boundary_max_mw,
            s.boundary_tol_mw,
            s.boundary_max_probes,
        )
        result["bus"] = bus
        result["entity"] = entity
        if entity == "load":
            result["pf_tan_phi"] = tan_phi
        return result

    def _handle_boundary_sweep(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a boundary (hosting-capacity) sweep — one bisection per candidate bus.

        The whole boundary sweep is ONE LLM turn performing N internal bisections;
        it does not consume N iterations of the LLM budget.
        """
        s = self._config.search
        description = data.get("description", "Boundary sweep")
        reasoning = data.get("reasoning", "")
        entity = data.get("entity", "load")
        if entity not in ("load", "generator"):
            self._error_feedback = "boundary sweep 'entity' must be 'load' or 'generator'."
            return "error", True

        pf_spec = data.get("power_factor", s.boundary_power_factor_default)
        candidate_set_spec = data.get("candidate_set", {})
        feasibility_spec = data.get("feasibility", {})
        vmin = feasibility_spec.get("Vmin", 0.9)
        vmax = feasibility_spec.get("Vmax", 1.1)
        q_frac = s.boundary_gen_q_frac

        candidates, cand_err = self._resolve_candidate_buses(candidate_set_spec)
        if candidates is None:
            self._error_feedback = f"boundary sweep {cand_err}"
            return "error", True
        if not candidates:
            self._error_feedback = "Boundary sweep resolved to an empty candidate set."
            return "error", True

        self._print(
            f'[Iter {iteration}] LLM action: boundary sweep — "{description}" '
            f"({len(candidates)} candidates, entity={entity}, PF={pf_spec})"
        )

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Pre-compute bus limits after applying the vlimits command (same for all candidates).
        try:
            _vlimits_net, _ = apply_modifications(
                self._base_network,
                [parse_command({"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax})],
                application="opflow",
            )
            bus_limits = _bus_limits_from_network(_vlimits_net)
        except Exception:
            _vlimits_net = self._base_network
            bus_limits = _bus_limits_from_network(self._base_network)

        tan_phi_avg = _system_average_tan_phi(self._base_network)
        workers = self._resolve_sweep_workers(len(candidates))
        thread_limit = 1 if workers > 1 else None

        # Base-case feasibility check (single solve). If the base case is infeasible
        # under the stated band, every bisection's lower bound is invalid.
        if self._on_phase:
            self._on_phase(iteration, "running_simulation (boundary: base feasibility check)")
        base_sim = self._executor.run(
            _vlimits_net, "opflow", -(iteration * 1_000_000),
            self._build_extra_args(), thread_limit,
        )
        base_opflow = None
        if base_sim is not None:
            base_opflow = parse_simulation_result_for_app(
                base_sim, application="opflow", bus_limits=bus_limits,
            )
        base_feasible = (
            base_opflow is not None
            and base_opflow.feasibility_detail == "feasible"
            and base_opflow.num_violations == 0
        )
        if not base_feasible:
            self._error_feedback = (
                "Boundary sweep aborted: the base case is infeasible under the stated "
                f"voltage band (Vmin={vmin}, Vmax={vmax}). Relax the band or fix the base "
                "case before searching hosting capacity."
            )
            self._print(f"[Iter {iteration}] Boundary sweep aborted — base case infeasible")
            return "error", True

        # Build one bisection callable per candidate; parallelize across candidates.
        def _make_fn(bus: int, idx: int):
            base_tag = iteration * 1_000_000 + idx * 100
            return lambda: self._bisect_candidate(
                bus, entity, pf_spec, tan_phi_avg, vmin, vmax,
                bus_limits, q_frac, base_tag, thread_limit, base_opflow,
            )

        fns = [_make_fn(bus, idx) for idx, bus in enumerate(candidates)]

        if self._on_phase:
            self._on_phase(
                iteration, f"running_simulation ({len(candidates)} boundary bisections)"
            )
        self._print(
            f"[Iter {iteration}] Running {len(candidates)} boundary bisections "
            f"({workers} concurrent)..."
        )

        def _boundary_progress(done: int, total: int) -> None:
            if self._on_phase:
                self._on_phase(iteration, f"running_simulation (boundary {done}/{total} buses)")

        results_map = self._executor.map_callables(
            fns, max_workers=workers, on_progress=_boundary_progress,
        )

        # Assemble per-candidate summaries (in sorted candidate order).
        candidate_summaries: list[dict] = []
        determined_buses: list[int] = []
        for idx, bus in enumerate(candidates):
            r = results_map.get(idx)
            if r is None or isinstance(r, Exception):
                candidate_summaries.append({
                    "bus": bus,
                    "feasible": False,
                    "max_feasible_mw": None,
                    "binding_constraint": "error",
                    "probes_used": 0,
                    "voltage_min": 0.0,
                    "voltage_max": 0.0,
                    "max_line_loading_pct": 0.0,
                    "violations": 0,
                    "status": "ERROR",
                    "reason": "bisection error",
                    "cost": None,
                    "entity": entity,
                })
                continue
            mfm = r.get("max_feasible_mw")
            determined = mfm is not None
            if determined:
                determined_buses.append(bus)
            candidate_summaries.append({
                "bus": bus,
                "feasible": determined,
                "max_feasible_mw": mfm,
                "binding_constraint": r.get("binding_constraint", ""),
                "probes_used": r.get("probes_used", 0),
                "voltage_min": r.get("voltage_min", 0.0),
                "voltage_max": r.get("voltage_max", 0.0),
                "max_line_loading_pct": r.get("max_line_loading_pct", 0.0),
                "violations": 0,
                "status": "BOUNDARY",
                "reason": "" if determined else r.get("note", "undetermined"),
                "cost": None,
                "entity": entity,
                "note": r.get("note", ""),
            })

        # LLM-facing text — token-bounded, mirrors the B.1 gating.
        if entity == "load":
            mut_desc = f"boundary load injection, PF={pf_spec}"
        else:
            mut_desc = f"boundary generator (forced injection, Q±{q_frac:.2f}·ΔP)"
        self._latest_results_text = _build_boundary_llm_view(
            candidate_summaries,
            mut_desc,
            top_n=s.sweep_llm_top_n,
            threshold=s.sweep_full_table_threshold,
        )

        # Journal (full per-candidate data; report reads this).
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        _boundary_command = _multi_call_record(
            "sweep", len(candidates), base_sim,
            "Executed once per candidate bus. The per-bus modification (the injected "
            "load/generation at the candidate bus) is written into the per-candidate "
            "netfile, not passed as an ExaGO argument; only the -netfile path differs "
            "between candidates. Each candidate runs a bisection of several solves at "
            "varying injection magnitudes, each with its own netfile. The representative "
            "shown is the base-case reference solve.",
        )
        self._journal.add_sweep(
            iteration=iteration,
            description=f"[boundary sweep] {description}",
            candidate_count=len(candidates),
            candidate_summaries=candidate_summaries,
            feasible_buses=determined_buses,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
            exago_command=_boundary_command,
        )
        self._store_sweep_cache(
            self._sweep_cache_key(data), description, len(candidates),
            candidate_summaries, determined_buses, self._latest_results_text,
        )

        self._print(
            f"[Iter {iteration}] Boundary sweep complete: "
            f"{len(determined_buses)}/{len(candidates)} buses with a determined boundary"
        )

        return "sweep", True

    def _run_contingency_screen(
        self,
        net: MATNetwork,
        contingencies: list,
        vmin: float,
        vmax: float,
        *,
        iteration: int,
        reference: bool = True,
    ) -> tuple[dict, list[dict]]:
        """Shared per-contingency screen loop (C.5 neighbor-scoped + C.8 system-wide N-1).

        Applies each contingency's outage set ON TOP OF ``net`` under the voltage
        band (set_all_bus_vlimits), re-solves OPFLOW in parallel, and judges each
        with the standard predicate — PASS iff the post-outage OPF converges to a
        feasible re-dispatched operating point. When ``reference`` is True, also
        runs the pre-contingency reference solve on the band-constrained base point.

        Returns ``(reference, contingency_summaries)``:
          - ``reference``: dict with keys ``passed``, ``reason``, ``opflow``,
            ``sim``, ``bus_limits``, ``first_opflow``, ``results_by_ctg``,
            ``build_errors``, ``built_count``.
          - ``contingency_summaries``: per-contingency summary dicts (the exact
            shape journaled by ``add_contingency`` / ``add_reserve``).
        """
        # Bus limits from the vlimits-applied operating point (same for all solves).
        vlimits_cmd = {"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax}
        try:
            _ref_net, _ = apply_modifications(
                net, [parse_command(vlimits_cmd)], application="opflow",
            )
            bus_limits = _bus_limits_from_network(_ref_net)
        except Exception:
            _ref_net = net
            bus_limits = _bus_limits_from_network(net)

        predicate_fn = sweep_metrics.PREDICATES["standard"]

        # --- pre-contingency reference solve (base operating point under the band) ---
        ref_sim = None
        ref_opflow = None
        ref_passed = True
        ref_reason = ""
        if reference:
            if self._on_phase:
                self._on_phase(iteration, "running_simulation (contingency: pre-contingency reference)")
            ref_sim = self._executor.run(
                _ref_net, "opflow", -(iteration * 10000 + 9999),
                self._build_extra_args(), None,
            )
            if ref_sim is not None:
                ref_opflow = parse_simulation_result_for_app(
                    ref_sim, application="opflow", bus_limits=bus_limits,
                )
            ref_passed, ref_reason = predicate_fn(
                ref_opflow, None, {"vmin": vmin, "vmax": vmax},
            )

        # --- build one task per contingency (vlimits + outage set on the current net) ---
        sim_tasks: list[tuple[MATNetwork, str, int, list[str] | None]] = []
        task_idx_for_ctg: list[int] = []  # task index → contingency index
        build_errors: list[str] = []
        skipped: set[int] = set()

        for idx, ctg in enumerate(contingencies):
            raw_cmds = [vlimits_cmd] + ctg.commands()
            commands = []
            parse_ok = True
            for raw in raw_cmds:
                try:
                    commands.append(parse_command(raw))
                except ValueError as exc:
                    build_errors.append(f"{ctg.label()}: {exc}")
                    parse_ok = False
                    break
            if not parse_ok:
                skipped.add(idx)
                continue
            try:
                modified_net, _ = apply_modifications(net, commands, application="opflow")
            except Exception as exc:  # e.g. element not found after a prior outage
                build_errors.append(f"{ctg.label()}: {exc}")
                skipped.add(idx)
                continue
            sim_tasks.append(
                (modified_net, "opflow", -(iteration * 10000 + idx), self._build_extra_args())
            )
            task_idx_for_ctg.append(idx)

        # --- run in parallel ---
        results_by_ctg: dict[int, "SimulationResult"] = {}
        if sim_tasks:
            workers = self._resolve_sweep_workers(len(sim_tasks))
            thread_limit = 1 if workers > 1 else None
            if self._on_phase:
                self._on_phase(
                    iteration, f"running_simulation ({len(sim_tasks)} contingency solves)"
                )
            self._print(
                f"[Iter {iteration}] Running {len(sim_tasks)} contingency solves "
                f"({workers} concurrent)..."
            )

            def _ctg_progress(done: int, total: int) -> None:
                if self._on_phase:
                    self._on_phase(iteration, f"running_simulation (contingency {done}/{total} solved)")

            results_map = self._executor.run_parallel(
                sim_tasks, max_workers=workers, thread_limit=thread_limit,
                on_progress=_ctg_progress,
            )
            for task_idx, ctg_idx in enumerate(task_idx_for_ctg):
                r = results_map.get(task_idx)
                if r is not None:
                    results_by_ctg[ctg_idx] = r

        # --- judge each contingency with the standard predicate ---
        contingency_summaries: list[dict] = []
        first_opflow: "OPFLOWResult | None" = None

        for idx, ctg in enumerate(contingencies):
            if idx in skipped:
                contingency_summaries.append({
                    "label": ctg.label(),
                    "order": ctg.order,
                    "kinds": [e.kind for e in ctg.elements],
                    "neighbor_buses": [e.neighbor_bus for e in ctg.elements],
                    "elements": [e.to_command() for e in ctg.elements],
                    "passed": False,
                    "voltage_min": 0.0,
                    "voltage_max": 0.0,
                    "max_line_loading_pct": 0.0,
                    "violations": 0,
                    "status": "BUILD_ERROR",
                    "reason": "build error",
                })
                continue

            sim_result = results_by_ctg.get(idx)
            opflow = None
            if sim_result is not None:
                opflow = parse_simulation_result_for_app(
                    sim_result, application="opflow", bus_limits=bus_limits,
                )
            if first_opflow is None and opflow is not None:
                first_opflow = opflow

            passed, reason = predicate_fn(
                opflow, None, {"vmin": vmin, "vmax": vmax},
            )

            contingency_summaries.append({
                "label": ctg.label(),
                "order": ctg.order,
                "kinds": [e.kind for e in ctg.elements],
                "neighbor_buses": [e.neighbor_bus for e in ctg.elements],
                "elements": [e.to_command() for e in ctg.elements],
                "passed": passed,
                "voltage_min": opflow.voltage_min if opflow else 0.0,
                "voltage_max": opflow.voltage_max if opflow else 0.0,
                "max_line_loading_pct": opflow.max_line_loading_pct if opflow else 0.0,
                "violations": opflow.num_violations if opflow else 0,
                "status": opflow.convergence_status if opflow else "FAILED",
                "reason": reason,
            })

        reference_out = {
            "passed": ref_passed,
            "reason": ref_reason,
            "opflow": ref_opflow,
            "sim": ref_sim,
            "bus_limits": bus_limits,
            "first_opflow": first_opflow,
            "results_by_ctg": results_by_ctg,
            "build_errors": build_errors,
            "built_count": len(sim_tasks),
        }
        return reference_out, contingency_summaries

    def _handle_contingency_sweep(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a contingency (N-1/N-2) screen — one LLM action, whole study in Python.

        Enumerates outages drawn from the k nearest neighbor buses of a target,
        applies each outage set ON TOP OF THE CURRENT OPERATING POINT (so a prior
        `modify` that connected the load is included), re-solves OPFLOW, and judges
        pass/fail.

        Feasibility semantics: under OPFLOW the V-band (via set_all_bus_vlimits)
        and Rate A are in-solve hard constraints, so a contingency PASSES iff the
        post-contingency OPFLOW converges to a feasible re-dispatched operating
        point and FAILS iff it does not — the OPF-redispatch post-contingency
        model. The pre-contingency reference solve certifies the base operating
        point under the stated band.

        Like the boundary sweep, the whole screen is ONE LLM turn performing N
        internal solves; it does not consume N iterations of the LLM budget.
        """
        if self._config.search.application != "opflow":
            self._print(f"[Iter {iteration}] 'contingency' sweep is only supported for OPFLOW")
            self._error_feedback = (
                "The contingency sweep is only supported for the OPFLOW application. "
                "Use 'modify' for other applications."
            )
            return "error", True

        description = data.get("description", "Contingency screen")
        reasoning = data.get("reasoning", "")

        # --- validate parameters ---
        raw_target = data.get("target_bus")
        if raw_target is None:
            self._error_feedback = "contingency sweep requires a 'target_bus' (integer bus number)."
            return "error", True
        try:
            target_bus = int(raw_target)
        except (TypeError, ValueError):
            self._error_feedback = f"contingency 'target_bus' must be an integer, got {raw_target!r}."
            return "error", True

        try:
            neighbor_count = int(data.get("neighbor_count", 3))
        except (TypeError, ValueError):
            self._error_feedback = f"contingency 'neighbor_count' must be an integer, got {data.get('neighbor_count')!r}."
            return "error", True
        if neighbor_count < 1:
            self._error_feedback = "contingency 'neighbor_count' must be >= 1."
            return "error", True

        try:
            order = int(data.get("contingency_order", 1))
        except (TypeError, ValueError):
            self._error_feedback = f"contingency 'contingency_order' must be 1 or 2, got {data.get('contingency_order')!r}."
            return "error", True
        if order not in (1, 2):
            self._error_feedback = "contingency 'contingency_order' must be 1 (N-1) or 2 (N-2)."
            return "error", True

        components = data.get("components", ["branch", "gen", "load"])
        if not isinstance(components, list) or not components:
            self._error_feedback = (
                "contingency 'components' must be a non-empty list drawn from "
                "['branch', 'gen', 'load']."
            )
            return "error", True
        _valid_kinds = {"branch", "gen", "load"}
        if any(c not in _valid_kinds for c in components):
            self._error_feedback = (
                f"contingency 'components' may only contain {sorted(_valid_kinds)}; "
                f"got {components}."
            )
            return "error", True

        feasibility_spec = data.get("feasibility", {})
        vmin = feasibility_spec.get("Vmin", 0.9)
        vmax = feasibility_spec.get("Vmax", 1.1)

        # Optional relief phase (C.7): ordered list of relief measures to try on each
        # FAILED contingency. Absent → no relief phase (behaves exactly as C.5).
        relief_measures = data.get("relief_measures")
        if relief_measures is not None:
            if not isinstance(relief_measures, list) or not relief_measures:
                self._error_feedback = (
                    "contingency 'relief_measures' must be a non-empty ordered list drawn "
                    f"from {list(relief_search.MEASURE_ORDER)}."
                )
                return "error", True
            _bad = [m for m in relief_measures if m not in relief_search.MEASURE_ORDER]
            if _bad:
                self._error_feedback = (
                    f"contingency 'relief_measures' contains unknown measure(s) {_bad}; "
                    f"valid measures: {list(relief_search.MEASURE_ORDER)}."
                )
                return "error", True

        # Operating point: apply the screen on top of the CURRENT network (post-modify).
        net = self._current_network or self._base_network

        # --- enumerate ---
        try:
            contingencies = contingency.enumerate_contingencies(
                net, target_bus, neighbor_count, order, tuple(components),
            )
        except ValueError as exc:
            self._error_feedback = f"contingency enumeration error: {exc}"
            return "error", True

        if not contingencies:
            self._error_feedback = (
                f"No contingencies enumerated for target bus {target_bus} "
                f"(neighbors={neighbor_count}, components={components}). The neighbor "
                "buses may have no outage-eligible elements of the requested kinds."
            )
            return "error", True

        max_count = self._config.search.contingency_max_count
        if len(contingencies) > max_count:
            self._error_feedback = (
                f"Contingency screen would enumerate {len(contingencies)} contingencies, "
                f"exceeding the guard of {max_count}. Lower neighbor_count, reduce "
                f"contingency_order (2→1), or narrow components to shrink the study."
            )
            return "error", True

        neighbors = topology.k_nearest_by_hops(net, target_bus, neighbor_count)
        self._print(
            f'[Iter {iteration}] LLM action: contingency sweep — "{description}" '
            f"(target bus {target_bus}, N-{order}, {len(contingencies)} contingencies, "
            f"neighbors={[nb for nb, _ in neighbors]})"
        )

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # --- shared screen: reference solve + per-contingency parallel solves + judging ---
        reference, contingency_summaries = self._run_contingency_screen(
            net, contingencies, vmin, vmax, iteration=iteration, reference=True,
        )
        bus_limits = reference["bus_limits"]
        ref_sim = reference["sim"]
        ref_passed = reference["passed"]
        ref_reason = reference["reason"]
        results_by_ctg = reference["results_by_ctg"]

        if reference["built_count"] == 0:
            self._error_feedback = (
                "All contingencies failed to build outage commands. Errors:\n"
                + "\n".join(reference["build_errors"][:5])
            )
            return "error", True

        passed_count = sum(1 for s in contingency_summaries if s["passed"])
        failed_count = len(contingency_summaries) - passed_count
        self._latest_opflow = reference["opflow"] or reference["first_opflow"]

        # --- relief phase (C.7): search relief measures for each FAILED contingency ---
        if relief_measures:
            self._run_relief_phase(
                iteration, net, contingencies, contingency_summaries,
                relief_measures, vmin, vmax, bus_limits,
            )

        # --- LLM-facing view (token-bounded) ---
        self._latest_results_text = self._build_contingency_llm_view(
            target_bus=target_bus,
            neighbors=neighbors,
            order=order,
            components=components,
            vmin=vmin,
            vmax=vmax,
            contingency_summaries=contingency_summaries,
            passed_count=passed_count,
            failed_count=failed_count,
            ref_passed=ref_passed,
            ref_reason=ref_reason,
            threshold=self._config.search.sweep_full_table_threshold,
            top_n=self._config.search.sweep_llm_top_n,
            relief_measures=relief_measures,
        )

        # --- journal + cache ---
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        _representative_sim = ref_sim or next(iter(results_by_ctg.values()), None)
        _ctg_command = _multi_call_record(
            "contingency", len(contingencies), _representative_sim,
            "Executed once per enumerated contingency. Each outage set is written into "
            "the per-contingency netfile (via set_branch_status / set_gen_status / "
            "set_load), not passed as an ExaGO argument; only the -netfile path differs. "
            "The representative shown is the pre-contingency reference solve.",
        )
        self._journal.add_contingency(
            iteration=iteration,
            description=f"[contingency N-{order}] {description} (on current operating point)",
            target_bus=target_bus,
            neighbors=neighbors,
            order=order,
            contingency_summaries=contingency_summaries,
            passed_count=passed_count,
            failed_count=failed_count,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
            exago_command=_ctg_command,
        )
        self._store_sweep_cache(
            self._sweep_cache_key(data), description, len(contingencies),
            contingency_summaries, [], self._latest_results_text,
        )

        self._print(
            f"[Iter {iteration}] Contingency screen complete: "
            f"{passed_count} passed / {failed_count} failed of {len(contingencies)} "
            f"(N-{order}, target bus {target_bus})"
        )
        return "sweep", True

    def _handle_reserve_screen(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a hot-reserve / minimum N-1 generator security screen (C.8).

        Computes system hot reserve (Σ Pmax−Pg over on-units) from the SOLVED base
        operating point, then runs a system-wide N-1 generator-outage screen (each
        committed unit tripped, OPF re-solved) to verify each loss is coverable by
        redispatch within network limits (deliverability, not copperplate). The
        minimum hot reserve required for N-1 is the output of the largest committed
        unit — the worst single-generator loss. ONE LLM action performing
        1 + n_on internal solves; it does not consume the LLM iteration budget.
        """
        if self._config.search.application != "opflow":
            self._print(f"[Iter {iteration}] 'reserve' screen is only supported for OPFLOW")
            self._error_feedback = (
                "The reserve screen is only supported for the OPFLOW application. "
                "Use 'modify' for other applications."
            )
            return "error", True

        description = data.get("description", "Hot reserve / N-1 generator security")
        reasoning = data.get("reasoning", "")
        feasibility_spec = data.get("feasibility", {})
        vmin = feasibility_spec.get("Vmin", 0.9)
        vmax = feasibility_spec.get("Vmax", 1.1)
        minimize = bool(data.get("minimize", False))

        # Operating point: the reserve screen runs on the CURRENT network (the base
        # network when no prior `modify` ran — prompt 4 adds no load).
        net = self._current_network or self._base_network

        contingencies = contingency.all_generator_contingencies(net)
        if not contingencies:
            self._error_feedback = (
                "No in-service generators found; cannot run the reserve / N-1 "
                "generator security screen."
            )
            return "error", True

        max_count = self._config.search.contingency_max_count
        if len(contingencies) > max_count:
            self._error_feedback = (
                f"Reserve screen would enumerate {len(contingencies)} generator "
                f"outages, exceeding the guard of {max_count}."
            )
            return "error", True

        self._print(
            f'[Iter {iteration}] LLM action: reserve screen — "{description}" '
            f"({len(contingencies)} in-service generators, system-wide N-1)"
        )
        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # --- shared screen: base reference solve + per-unit N-1 solves + judging ---
        reference, contingency_summaries = self._run_contingency_screen(
            net, contingencies, vmin, vmax, iteration=iteration, reference=True,
        )
        base_result = reference["opflow"]
        if base_result is None or not reference["passed"]:
            self._error_feedback = (
                "Reserve screen: the base operating point is infeasible under the "
                f"band (Vmin={vmin}, Vmax={vmax}): "
                f"{reference['reason'] or 'did not converge'}. Cannot assess reserve."
            )
            return "error", True

        self._latest_opflow = base_result

        # --- reserve accounting from the solved base dispatch ---
        on_units = [g for g in base_result.generators if g.status == 1]
        if not on_units:
            self._error_feedback = (
                "Reserve screen: base solve reported no in-service generators."
            )
            return "error", True
        hot_reserve_available = sum(g.Pmax - g.Pg for g in on_units)
        largest_pg_gen = max(on_units, key=lambda g: g.Pg)
        largest_pmax_gen = max(on_units, key=lambda g: g.Pmax)
        n_on = len(on_units)

        passed_count = sum(1 for s in contingency_summaries if s["passed"])
        failed_count = len(contingency_summaries) - passed_count

        required_reserve_n1 = largest_pg_gen.Pg
        margin = hot_reserve_available - required_reserve_n1
        n1_secure = failed_count == 0

        reserve_meta = {
            "n_on": n_on,
            "hot_reserve_available": hot_reserve_available,
            "largest_pg": largest_pg_gen.Pg,
            "largest_pg_bus": largest_pg_gen.bus,
            "largest_pmax": largest_pmax_gen.Pmax,
            "largest_pmax_bus": largest_pmax_gen.bus,
            "required_reserve_n1": required_reserve_n1,
            "margin": margin,
            "n1_secure": n1_secure,
            "passed_count": passed_count,
            "failed_count": failed_count,
        }

        # --- optional minimization pass (C.8 Path A): greedy de-commitment ---
        # ``journal_summaries`` holds the per-unit N-1 summaries that become
        # ``explored_variants`` in the journal entry.  For the minimize path these
        # reflect the MINIMIZED commitment so the PDF evidence table is correct.
        journal_summaries = contingency_summaries
        if minimize:
            min_result = self._run_reserve_minimization(iteration, net, vmin, vmax)
            # Keep the base ``n1_secure``/counts as the full-commitment assessment;
            # store the minimized commitment's security under a distinct key so the
            # assessment table stays coherent. The ``minimize`` flag is added only on
            # this path so the reading-2 assessment journal stays byte-identical.
            reserve_meta.update({
                "minimize": True,
                "reserve_full": min_result.reserve_full,
                "min_reserve": min_result.min_reserve,
                "n_decommitted": len(min_result.decommitted),
                "decommitted": list(min_result.decommitted),
                "final_on_count": min_result.final_on_count,
                "lower_bound_pg": min_result.lower_bound_pg,
                "lower_bound_bus": min_result.lower_bound_bus,
                "n1_secure_min": min_result.n1_secure,
                "solves_used": min_result.solves_used,
                "hit_budget": min_result.hit_budget,
                "trajectory": list(min_result.trajectory),
            })
            description = (
                f"[reserve N-1 minimize] min feasible hot reserve = "
                f"{min_result.min_reserve:.0f} MW"
            )
            # Run one final N-1 screen on the MINIMIZED commitment so the PDF
            # evidence table shows the 41-unit (not 54-unit) outage list.
            if min_result.decommitted:
                try:
                    _decommit_cmds = [
                        {"action": "set_gen_status", "bus": b, "gen_id": g, "status": 0}
                        for (b, g) in min_result.decommitted
                    ]
                    _min_variant, _ = apply_modifications(
                        net,
                        [parse_command(c) for c in _decommit_cmds],
                        application="opflow",
                    )
                    _min_ctgs = contingency.all_generator_contingencies(_min_variant)
                    if _min_ctgs:
                        _, _min_summaries = self._run_contingency_screen(
                            _min_variant, _min_ctgs, vmin, vmax,
                            iteration=iteration, reference=False,
                        )
                        journal_summaries = _min_summaries
                except Exception:
                    pass  # fall back to full-commitment summaries
        else:
            description = f"[reserve N-1] {description} (on current operating point)"

        # --- LLM-facing view ---
        self._latest_results_text = self._build_reserve_llm_view(
            reserve_meta, contingency_summaries, vmin, vmax,
        )

        # --- journal + cache ---
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        _representative_sim = reference["sim"] or next(
            iter(reference["results_by_ctg"].values()), None
        )
        _reserve_command = _multi_call_record(
            "reserve", 1 + len(contingencies), _representative_sim,
            "Base operating-point solve plus one OPF solve per in-service generator "
            "(each unit tripped via set_gen_status in the per-solve netfile; only the "
            "-netfile path differs). The representative shown is the base solve.",
        )
        self._journal.add_reserve(
            iteration=iteration,
            description=description,
            reserve_meta=reserve_meta,
            contingency_summaries=journal_summaries,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
            exago_command=_reserve_command,
        )
        self._store_sweep_cache(
            self._sweep_cache_key(data), description, len(contingencies),
            journal_summaries, [], self._latest_results_text,
        )

        self._print(
            f"[Iter {iteration}] Reserve screen complete: "
            f"available {hot_reserve_available:.1f} MW, required (N-1) "
            f"{required_reserve_n1:.1f} MW, margin {margin:.1f} MW; "
            f"N-1 generator security {passed_count}/{n_on} feasible"
        )
        return "sweep", True

    def _run_reserve_minimization(
        self, iteration: int, net: MATNetwork, vmin: float, vmax: float,
    ) -> "reserve_search.ReserveMinResult":
        """Greedy security-constrained de-commitment to minimize N-1-secure hot reserve.

        Builds the base-solve and N-1-screen closures over ``net`` and delegates the
        accept/revert search to the pure ``reserve.minimize_hot_reserve``. Each trial
        de-commits a set of units (``set_gen_status=0``) and re-solves; the N-1 screen
        reuses the validated ``_run_contingency_screen`` over the remaining on-units.
        The result's greedy value is an upper bound; the lower bound is the largest
        remaining committed Pg at the final commitment.
        """
        predicate_fn = sweep_metrics.PREDICATES["standard"]
        vlimits_cmd = {"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax}
        solve_seq = {"n": 0}

        def _decommit_commands(off_units) -> list[dict]:
            return [
                {"action": "set_gen_status", "bus": b, "gen_id": g, "status": 0}
                for (b, g) in sorted(off_units)
            ]

        def _base_solve_fn(off_units):
            cmds = [vlimits_cmd] + _decommit_commands(off_units)
            try:
                variant, _ = apply_modifications(
                    net, [parse_command(c) for c in cmds], application="opflow",
                )
            except Exception:
                return None
            bus_limits = _bus_limits_from_network(variant)
            solve_seq["n"] += 1
            run_id = -(iteration * 1_000_000 + solve_seq["n"])
            sim = self._executor.run(
                variant, "opflow", run_id, self._build_extra_args(), None,
            )
            if sim is None:
                return None
            opflow = parse_simulation_result_for_app(
                sim, application="opflow", bus_limits=bus_limits,
            )
            passed, _ = predicate_fn(opflow, None, {"vmin": vmin, "vmax": vmax})
            return opflow if passed else None

        def _n1_screen_fn(off_units):
            cmds = _decommit_commands(off_units)
            if cmds:
                try:
                    variant, _ = apply_modifications(
                        net, [parse_command(c) for c in cmds], application="opflow",
                    )
                except Exception:
                    return False, 0, 0
            else:
                variant = net
            ctgs = contingency.all_generator_contingencies(variant)
            if not ctgs:
                return True, 0, 0  # no committed units left → vacuously N-1 secure
            _ref, summaries = self._run_contingency_screen(
                variant, ctgs, vmin, vmax, iteration=iteration, reference=False,
            )
            passed = sum(1 for s in summaries if s["passed"])
            total = len(summaries)
            return (passed == total), passed, total

        max_solves = self._config.search.reserve_max_solves
        self._print(
            f"[Iter {iteration}] Reserve minimization: greedy largest-Pmax-first "
            f"de-commitment (budget {max_solves} solves)..."
        )
        result = reserve_search.minimize_hot_reserve(
            net, vmin, vmax,
            base_solve_fn=_base_solve_fn,
            n1_screen_fn=_n1_screen_fn,
            max_solves=max_solves,
        )
        self._print(
            f"[Iter {iteration}] Reserve minimization complete: "
            f"min reserve {result.min_reserve:.1f} MW (full {result.reserve_full:.1f} MW), "
            f"{len(result.decommitted)} units de-committed, "
            f"lower bound {result.lower_bound_pg:.1f} MW"
            + (" [budget hit]" if result.hit_budget else "")
        )
        return result

    def _build_reserve_llm_view(
        self,
        reserve_meta: dict,
        contingency_summaries: list[dict],
        vmin: float,
        vmax: float,
    ) -> str:
        """Token-bounded LLM view of the hot-reserve / N-1 generator security screen."""
        m = reserve_meta
        n_on = m["n_on"]
        passed = m["passed_count"]
        failed = m["failed_count"]

        # Minimization result leads the view so the LLM answers with the minimum and
        # issues `complete` — no manual per-unit de-commitment, no invented number.
        if m.get("minimize"):
            secure_min = m.get("n1_secure_min", m["n1_secure"])
            lines = [
                "MINIMUM feasible hot reserve for N-1 (greedy largest-Pmax-first "
                "de-commitment; result is an UPPER BOUND on the true minimum).",
                f"Minimum feasible hot reserve = {m['min_reserve']:.1f} MW "
                f"(N-1 secure: {'yes' if secure_min else 'no'}).",
                f"Hot reserve at full commitment = {m['reserve_full']:.1f} MW; "
                f"de-committed {m['n_decommitted']} of {n_on} units "
                f"→ {m['final_on_count']} remain committed.",
                f"Bracket: lower bound {m['lower_bound_pg']:.1f} MW "
                f"(largest remaining committed unit, gen@{m['lower_bound_bus']}) "
                f"≤ minimum feasible reserve ≤ {m['min_reserve']:.1f} MW (greedy upper bound).",
                f"Feasibility model unchanged: OPF redispatch under Vmin={vmin}, Vmax={vmax} "
                "and Rate A; each remaining single-unit outage re-solved.",
            ]
            if m.get("hit_budget"):
                lines.append(
                    "Note: the solve budget was reached — the reported commitment is the "
                    "best found so far (the true minimum may be lower)."
                )
            if m["decommitted"]:
                shown = ", ".join(
                    f"gen@{b}#{g}" for (b, g) in m["decommitted"][:20]
                )
                more = "" if len(m["decommitted"]) <= 20 else f" (+{len(m['decommitted']) - 20} more)"
                lines.append(f"De-committed units (in order): {shown}{more}")
            lines.append(
                "This fully answers a 'minimum/minimize hot reserve for N-1' goal — "
                "issue `complete` with this minimum."
            )
            return "\n".join(lines)

        lines = [
            f"System hot-reserve / N-1 generator security ({n_on} committed units).",
            f"Feasibility band: Vmin={vmin}, Vmax={vmax} "
            "(OPF-redispatch model — a unit loss PASSES iff the post-outage OPFLOW "
            "converges feasibly, proving the loss is coverable by redispatch within "
            "network limits, not just on a copperplate).",
            f"Hot reserve available (Σ Pmax−Pg over on-units): "
            f"{m['hot_reserve_available']:.1f} MW",
            f"Largest committed unit: gen@{m['largest_pg_bus']} at {m['largest_pg']:.1f} MW "
            f"(capacity {m['largest_pmax']:.1f} MW"
            + (f", largest capacity gen@{m['largest_pmax_bus']}"
               if m['largest_pmax_bus'] != m['largest_pg_bus'] else "")
            + ")",
            f"Minimum hot reserve required for N-1 = {m['required_reserve_n1']:.1f} MW "
            "(worst single-unit loss)",
            f"Reserve margin = {m['margin']:.1f} MW",
            f"N-1 generator security: {passed}/{n_on} unit outages feasible",
        ]

        failed_summaries = [s for s in contingency_summaries if not s["passed"]]
        if failed_summaries:
            lines.append("")
            lines.append(
                f"NOT N-1 secure: {failed} unit outage(s) infeasible — the loss of "
                "these units is not coverable by redispatch within limits, so the "
                "arithmetic reserve margin is not deliverable and the required reserve "
                "is mis-located. Failed units:"
            )
            lines.append(
                f"{'contingency':<20} | {'reason':<22} | {'Vmin':>5} | {'Vmax':>5} | maxLoad%"
            )
            for s in failed_summaries:
                lines.append(
                    f"{s['label']:<20} | {(s.get('reason') or ''):<22} | "
                    f"{s['voltage_min']:>5.3f} | {s['voltage_max']:>5.3f} | "
                    f"{s['max_line_loading_pct']:>7.1f}"
                )
        else:
            lines.append("")
            lines.append(
                "N-1 SECURE: every single committed-generator loss is feasible under "
                "OPF redispatch within the band and thermal limits."
            )
        return "\n".join(lines)

    def _run_relief_phase(
        self,
        iteration: int,
        net: MATNetwork,
        contingencies: list,
        contingency_summaries: list[dict],
        relief_measures: list[str],
        vmin: float,
        vmax: float,
        bus_limits: dict,
    ) -> None:
        """Search relief measures for each FAILED contingency (C.7); attach 'relief' payloads.

        Uses the same executor solve as C.5 via an injected ``solve_fn``. A shared
        solve counter enforces ``relief_max_solves``: once the budget is spent,
        remaining failures are marked 'relief budget exhausted' rather than searched.
        """
        budget = self._config.search.relief_max_solves
        solve_counter = {"n": 0}
        extra_args = self._build_extra_args()

        def _solve_fn(mnet: MATNetwork):
            tag = -(iteration * 1_000_000 + solve_counter["n"])
            solve_counter["n"] += 1
            sim = self._executor.run(mnet, "opflow", tag, extra_args, None)
            if sim is None:
                return None
            return parse_simulation_result_for_app(
                sim, application="opflow", bus_limits=bus_limits,
            )

        failures = [i for i, s in enumerate(contingency_summaries) if not s["passed"]]
        if not failures:
            return

        if self._on_phase:
            self._on_phase(iteration, f"relief search ({len(failures)} failed contingencies)")
        self._print(
            f"[Iter {iteration}] Relief search over {len(failures)} failed contingencies "
            f"(measures: {relief_measures}, budget {budget} solves)..."
        )

        for i in failures:
            summary = contingency_summaries[i]
            if summary.get("status") == "BUILD_ERROR":
                summary["relief"] = {
                    "resolved": False, "measure": None,
                    "detail": "build error; cannot search relief", "attempts": [],
                }
                continue
            if solve_counter["n"] >= budget:
                summary["relief"] = {
                    "resolved": False, "measure": None,
                    "detail": "relief budget exhausted", "attempts": [],
                }
                continue
            result = relief_search.find_relief(
                net, contingencies[i], relief_measures, vmin, vmax,
                _solve_fn, self._config.search,
            )
            summary["relief"] = {
                "resolved": result.resolved,
                "measure": result.action.measure if result.action else None,
                "detail": result.action.detail if result.action else "no relief measure restored feasibility",
                "attempts": [[m, r] for m, r in result.attempts],
            }

        n_resolved = sum(
            1 for i in failures if contingency_summaries[i].get("relief", {}).get("resolved")
        )
        self._print(
            f"[Iter {iteration}] Relief search complete: "
            f"{n_resolved}/{len(failures)} failures resolved "
            f"({solve_counter['n']} relief solves used)"
        )

    def _build_contingency_llm_view(
        self,
        target_bus: int,
        neighbors: list[tuple[int, int]],
        order: int,
        components: list[str],
        vmin: float,
        vmax: float,
        contingency_summaries: list[dict],
        passed_count: int,
        failed_count: int,
        ref_passed: bool,
        ref_reason: str,
        threshold: int,
        top_n: int,
        relief_measures: list[str] | None = None,
    ) -> str:
        """Token-bounded LLM view of a contingency screen, mirroring the sweep view gating.

        At or below ``threshold`` contingencies, a full pass/fail table is shown.
        Above it, the FAILED set is listed in full (grouped by reason) with a capped
        sample of passers plus aggregate counts and a journal/PDF pointer — the LLM
        needs the complete failed set to answer, so failures are never truncated.
        """
        n_total = len(contingency_summaries)
        nbr_str = ", ".join(f"{nb}(h{hop})" for nb, hop in neighbors)
        comp_str = "/".join(components)
        ref_line = (
            "Pre-contingency reference: FEASIBLE under the band."
            if ref_passed
            else f"Pre-contingency reference: INFEASIBLE ({ref_reason or 'did not converge'}) "
                 "— base operating point does not hold under the band; interpret results with care."
        )
        header = [
            f"Contingency screen (N-{order}) on target bus {target_bus}: "
            f"{n_total} contingencies over nearest neighbors [{nbr_str}].",
            f"Components: {comp_str}.  Feasibility band: Vmin={vmin}, Vmax={vmax} "
            f"(OPF-redispatch model — PASS = post-outage OPFLOW converges feasibly).",
            f"PASSED: {passed_count} / {n_total}   FAILED: {failed_count} / {n_total}",
            ref_line,
            "",
        ]

        # Relief section (C.7) — appended to whichever table branch is used below.
        relief_lines = self._relief_section_lines(
            contingency_summaries, failed_count, relief_measures,
        )

        # --- full-table branch ---
        if threshold > 0 and n_total <= threshold:
            lines = header + [
                f"{'contingency':<28} | {'kinds':<14} | {'pass':>4} | "
                f"{'Vmin':>5} | {'Vmax':>5} | {'maxLoad%':>8} | reason",
            ]
            for s in contingency_summaries:
                pass_str = "yes" if s["passed"] else "no"
                kinds_str = "+".join(s["kinds"])
                lines.append(
                    f"{s['label']:<28} | {kinds_str:<14} | {pass_str:>4} | "
                    f"{s['voltage_min']:>5.3f} | {s['voltage_max']:>5.3f} | "
                    f"{s['max_line_loading_pct']:>8.1f} | {s.get('reason') or ''}"
                )
            lines += relief_lines
            return "\n".join(lines)

        # --- summarized branch (FAILED set never truncated) ---
        lines = list(header)
        failed = [s for s in contingency_summaries if not s["passed"]]
        passed = [s for s in contingency_summaries if s["passed"]]

        failed_by_reason: dict[str, list[str]] = defaultdict(list)
        for s in failed:
            failed_by_reason[s.get("reason") or "did not converge"].append(s["label"])
        lines.append(f"FAILED contingencies grouped by reason ({failed_count} total):")
        if failed_by_reason:
            for reason in sorted(failed_by_reason):
                labels = failed_by_reason[reason]
                lines.append(f"  {reason} ({len(labels)}): {labels}")
        else:
            lines.append("  (none)")
        lines.append("")

        sample = passed[:top_n]
        lines.append(
            f"Passed contingencies ({passed_count} total; showing {len(sample)}):"
        )
        lines.append(f"  {[s['label'] for s in sample]}")
        if passed_count > len(sample):
            lines.append(f"  ... and {passed_count - len(sample)} more (see journal).")
        lines.append("")
        lines.append(
            f"[Note: Full per-contingency pass/fail table ({n_total} rows) is stored in "
            "the journal. Only the complete failed set and a sample of passers are shown "
            "here to limit token usage — no failure is omitted from this view.]"
        )
        lines += relief_lines
        return "\n".join(lines)

    def _relief_section_lines(
        self,
        contingency_summaries: list[dict],
        failed_count: int,
        relief_measures: list[str] | None,
    ) -> list[str]:
        """Build the 'Relief for failed contingencies' view section (empty if no relief)."""
        if not relief_measures:
            return []
        lines = ["", f"Relief for failed contingencies (measures tried in order: {relief_measures}):"]
        if failed_count == 0:
            lines.append("  no failures; no relief required.")
            return lines
        lines.append(
            f"{'contingency':<28} | {'resolving measure':<20} | detail | (measures tried)"
        )
        for s in contingency_summaries:
            if s["passed"]:
                continue
            rel = s.get("relief") or {}
            if rel.get("resolved"):
                measure = rel.get("measure") or "?"
                detail = rel.get("detail") or ""
            else:
                measure = "UNRESOLVED"
                detail = rel.get("detail") or "no relief measure restored feasibility"
            tried = ", ".join(
                f"{m}{'✓' if r else '✗'}" for m, r in (rel.get("attempts") or [])
            )
            lines.append(f"{s['label']:<28} | {measure:<20} | {detail} | ({tried})")
        return lines

    def _handle_select(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'select' action — choose a variant from explore results."""
        if self._explore_cache is None:
            self._print(f"[Iter {iteration}] 'select' without prior 'explore'")
            self._error_feedback = (
                "No explore results to select from. Use 'explore' before 'select'."
            )
            return "error", True

        choice = data.get("choice", "")
        reasoning = data.get("reasoning", "")

        cache = self._explore_cache

        if choice not in cache.variants:
            available = sorted(cache.variants.keys())
            self._print(f"[Iter {iteration}] Invalid variant choice: '{choice}'")
            self._error_feedback = (
                f"Invalid variant '{choice}'. Available variants: {', '.join(available)}. "
                "Respond with a 'select' action choosing one of these."
            )
            return "error", True

        selected = cache.variants[choice]
        if selected.rejected:
            self._print(f"[Iter {iteration}] Cannot select rejected variant '{choice}'")
            available = sorted(
                lbl for lbl, v in cache.variants.items() if not v.rejected
            )
            self._error_feedback = (
                f"Variant '{choice}' was rejected (all commands were no-ops) and "
                f"cannot be selected. Selectable variants: {', '.join(available) or 'none'}."
            )
            return "error", True
        self._print(f'[Iter {iteration}] Selected variant {choice} — "{selected.description}"')

        # Update current state
        self._current_network = selected.modified_net
        self._latest_opflow = selected.opflow_result
        if selected.opflow_result is not None:
            self._opflow_results_cache[iteration] = selected.opflow_result
            self._latest_results_text = results_summary_for_app(
                selected.opflow_result,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=selected.modified_net.gencost if self._config.search.application == "pflow" else None,
            )

        # Build explored-variants summary for journal
        explored_variants = []
        gencost = (
            (cache.base_network_snapshot.gencost if cache.base_network_snapshot else None)
            if self._config.search.application == "pflow" else None
        )
        for lbl, v in cache.variants.items():
            entry = {"label": lbl, "description": v.description, "commands": v.raw_commands, "is_pareto": v.is_pareto}
            if v.rejected:
                entry["rejected"] = True
            if v.skipped_commands:
                skipped_summaries = []
                for cmd, reasons in v.skipped_commands:
                    cmd_name = type(cmd).__name__
                    reason_str = "; ".join(reasons)
                    skipped_summaries.append(f"{cmd_name}: {reason_str}")
                entry["skipped"] = skipped_summaries
            if v.opflow_result is not None:
                entry["feasible"] = (
                    v.opflow_result.feasibility_detail == "feasible"
                    and v.opflow_result.num_violations == 0
                )
                if gencost is not None:
                    try:
                        entry["cost"] = v.opflow_result.compute_generation_cost(gencost)
                    except Exception:
                        pass
            else:
                entry["feasible"] = False
            explored_variants.append(entry)

        # Add journal entry for the selected variant
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        self._journal.add_from_results(
            iteration=iteration,
            description=f"[select {choice}] {cache.description}",
            commands=selected.raw_commands,
            opflow_result=selected.opflow_result,
            sim_elapsed=selected.sim_result.elapsed_seconds if selected.sim_result else 0.0,
            llm_reasoning=f"Selected variant {choice}. {reasoning}",
            mode=cache.base_mode,
            steering_directive=active_directive,
            num_steps=self._tcopflow_num_steps,
            num_scenarios=self._sopflow_num_scenarios,
            explored_variants=explored_variants,
            gencost=selected.modified_net.gencost if self._config.search.application == "pflow" else None,
            exago_command=_single_call_record(selected.sim_result),
        )

        # Extract tracked metrics for multi-objective tracking
        if selected.opflow_result is not None and self._journal.latest:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            metrics = extract_all_metrics(selected.opflow_result, metric_names)
            # For PFLOW: preserve the computed cost we populated above —
            # extract_all_metrics would otherwise overwrite it with the
            # raw opflow.objective_value (0.0 for PFLOW).
            if (
                self._config.search.application == "pflow"
                and "generation_cost" in metrics
                and self._journal.latest.tracked_metrics is not None
                and "generation_cost" in self._journal.latest.tracked_metrics
            ):
                metrics["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
            if metrics:
                if self._journal.latest.tracked_metrics:
                    merged = dict(self._journal.latest.tracked_metrics)
                    merged.update(metrics)
                    if "generation_cost" in self._journal.latest.tracked_metrics:
                        merged["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
                    self._journal.latest.tracked_metrics = merged
                else:
                    self._journal.latest.tracked_metrics = metrics

        # Clear explore cache
        self._explore_cache = None

        return "select", True

    def _handle_complete(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'complete' action from the LLM."""
        findings = data.get("findings", {})
        reasoning = data.get("reasoning", "")
        summary_text = findings.get("summary", reasoning)

        self._print(f"[Iter {iteration}] LLM action: complete")
        self._print(f'[Iter {iteration}] Search completed: "{summary_text}"')

        self._journal.add_complete(
            iteration=iteration,
            summary=summary_text,
        )

        self._final_findings = findings
        return "complete", False

    def _handle_analyze(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle an 'analyze' action from the LLM."""
        query_type = (data.get("query_type") or "").strip().lower()
        if query_type:
            return self._handle_topology_analyze(iteration, data, query_type)

        query = data.get("query", "")

        self._print(f'[Iter {iteration}] LLM action: analyze — "{query}"')

        result_text = self._run_analysis_query(query)
        self._latest_results_text = result_text
        logger.info("Analysis query result: %s", result_text[:300])

        # Record analyze action in the journal
        self._journal.add_analysis(
            iteration=iteration,
            query=query,
            result_summary=result_text[:200],
        )

        return "analyze", True

    def _handle_topology_analyze(
        self, iteration: int, data: dict, query_type: str
    ) -> tuple[str, bool]:
        """Handle a structured topology query — deterministic, no LLM/backend call."""
        _VALID = ("nearest_neighbors", "incident_branches")
        if query_type not in _VALID:
            self._error_feedback = (
                f"Unknown query_type '{query_type}'. "
                f"Valid topology query types: {', '.join(_VALID)}."
            )
            return "error", True

        net = self._current_network or self._base_network
        if net is None:
            self._error_feedback = "No network loaded; cannot execute topology query."
            return "error", True

        raw_bus = data.get("bus")
        if raw_bus is None:
            self._error_feedback = "Topology query requires a 'bus' field (integer bus number)."
            return "error", True
        try:
            bus = int(raw_bus)
        except (TypeError, ValueError):
            self._error_feedback = f"'bus' must be an integer, got {raw_bus!r}."
            return "error", True

        if query_type == "nearest_neighbors":
            raw_k = data.get("k", 3)
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                self._error_feedback = f"'k' must be a positive integer, got {raw_k!r}."
                return "error", True

            try:
                neighbors = topology.k_nearest_by_hops(net, bus, k)
                total = topology.count_reachable(net, bus)
            except ValueError as exc:
                self._error_feedback = f"Topology query error: {exc}"
                return "error", True

            result_text = topology.format_nearest_neighbors_view(bus, k, neighbors, total)
            query_desc = f"nearest_neighbors bus={bus} k={k}"

        else:  # incident_branches
            try:
                branches = topology.incident_branches(net, bus)
            except ValueError as exc:
                self._error_feedback = f"Topology query error: {exc}"
                return "error", True

            result_text = topology.format_incident_branches_view(bus, branches)
            query_desc = f"incident_branches bus={bus}"

        self._latest_results_text = result_text
        self._print(f"[Iter {iteration}] LLM action: analyze (topology) — {query_desc}")
        logger.info("Topology query %s result:\n%s", query_desc, result_text)

        self._journal.add_analysis(
            iteration=iteration,
            query=query_desc,
            result_summary=result_text[:200],
        )

        return "analyze", True

    def _handle_set_load_factor(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'set_load_factor' action — update the session-level load scaling factor."""
        new_factor = data.get("factor")
        if not isinstance(new_factor, (int, float)) or new_factor <= 0:
            self._error_feedback = (
                f"Invalid factor for set_load_factor: {new_factor!r}. "
                "Must be a positive number (e.g., 1.23)."
            )
            return "error", True

        old_factor = self._session_load_factor
        self._session_load_factor = float(new_factor)
        self._journal.load_factor = self._session_load_factor
        self._print(
            f"[Iter {iteration}] LLM action: set_load_factor — "
            f"{old_factor} → {self._session_load_factor}"
        )
        # Rebuild system prompt to reflect the new load factor note
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt as _fmt_bench
        net_summary_text = network_summary(
            self._base_network,
            max_generators=self._config.report.network_summary_max_generators,
        )
        net_metadata_text = self._network_metadata_text
        benchmark_text = _fmt_bench(self._benchmark_result) if self._benchmark_result else None
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=benchmark_text,
            session_load_factor=self._session_load_factor,
        )
        self._error_feedback = (
            f"Load factor updated to {self._session_load_factor}×. "
            "All subsequent runs will automatically apply this scaling. "
            "Use 'explore' or 'modify' to continue the search."
        )
        return "set_load_factor", True

    def _run_analysis_query(self, query: str) -> str:
        """Execute an analysis query against the latest OPFLOW results.

        Handles pattern-matched queries for common analyses and falls back to
        an LLM sub-call for arbitrary questions.
        """
        if self._latest_opflow is None:
            return "No simulation results available to analyze."

        opf = self._latest_opflow
        q = query.lower()

        _dcopflow = self._config.search.application == "dcopflow"
        _tcopflow = self._config.search.application == "tcopflow"
        _sopflow = self._config.search.application == "sopflow"

        # ── TCOPFLOW time-period queries ──────────────────────────────────
        if _tcopflow and ("period" in q or "time step" in q or "timestep" in q or "temporal" in q):
            if not self._tcopflow_period_data:
                return "No per-period data available. Run a simulation first."
            lines = [f"Multi-period summary ({len(self._tcopflow_period_data)} periods):"]
            lines.append(
                f"{'Period':>6} | {'Load(MW)':>9} | {'Gen(MW)':>9} | "
                f"{'Vmin(pu)':>8} | {'Vmax(pu)':>8} | {'MaxLoad%':>8}"
            )
            lines.append("-" * 62)
            for p in self._tcopflow_period_data:
                lines.append(
                    f"{p['period']:>6} | {p['total_load_mw']:>9.1f} | "
                    f"{p['total_gen_mw']:>9.1f} | {p['voltage_min']:>8.3f} | "
                    f"{p['voltage_max']:>8.3f} | {p['max_line_loading_pct']:>7.1f}%"
                )
            peak = max(self._tcopflow_period_data, key=lambda p: p["total_load_mw"])
            worst_v = min(self._tcopflow_period_data, key=lambda p: p["voltage_min"])
            worst_load = max(self._tcopflow_period_data, key=lambda p: p["max_line_loading_pct"])
            lines.append(f"\nPeak load: period {peak['period']} ({peak['total_load_mw']:.1f} MW)")
            lines.append(f"Worst voltage: period {worst_v['period']} (Vmin={worst_v['voltage_min']:.3f} pu)")
            lines.append(f"Worst line loading: period {worst_load['period']} ({worst_load['max_line_loading_pct']:.1f}%)")
            return "\n".join(lines)

        # ── SOPFLOW scenario queries ─────────────────────────────────────
        # Only intercept queries that are purely about scenario/wind metadata.
        # If the query also asks about voltage, loading, or generators,
        # let it fall through to the standard pattern matchers so the LLM
        # gets useful data instead of a generic 4-line dead end.
        if _sopflow and ("scenario" in q or "wind" in q or "stochastic" in q):
            _wants_metrics = (
                "voltage" in q or "vm" in q or "bus" in q
                or "loading" in q or "line" in q or "branch" in q
                or "generator" in q or "load" in q or "loss" in q
                or "flow" in q or "slack" in q
            )
            if not _wants_metrics:
                wind_gens = [g for g in opf.generators if "wind" in g.fuel.lower() and g.status == 1]
                wind_info = ""
                if wind_gens:
                    wind_pg = sum(g.Pg for g in wind_gens)
                    wind_pmax = sum(g.Pmax for g in wind_gens)
                    wind_util = wind_pg / wind_pmax * 100 if wind_pmax > 0 else 0
                    wind_info = (
                        f"Wind generators: {len(wind_gens)} online, "
                        f"total {wind_pg:.2f} MW / {wind_pmax:.2f} MW capacity "
                        f"({wind_util:.0f}% utilization)"
                    )
                lines = [
                    f"SOPFLOW scenario summary ({self._sopflow_num_scenarios} scenarios):",
                    f"Solver: {opf.solver}",
                    wind_info,
                    f"Base-case dispatch cost: ${opf.objective_value:,.2f}",
                ]
                if opf.num_violations > 0:
                    lines.append(f"Constraints violated in base case: {opf.num_violations}")
                else:
                    lines.append("All scenarios satisfied network constraints (base case feasible).")
                lines.append(
                    "Note: per-scenario second-stage dispatch IS now available. The results "
                    "summary reports offered, dispatched (absorbed), and curtailed wind across "
                    "scenarios. Use scale_wind_scenario to raise offered wind: absorbed wind "
                    "rises then saturates at the network absorption capacity P*, with the "
                    "surplus curtailed."
                )
                return "\n".join(lines)
        m = re.search(r"voltage\s+below\s+([\d.]+)", q)
        if m:
            if _dcopflow:
                return (
                    "Voltage magnitude analysis is not available in DCOPFLOW. "
                    "In the DC approximation, all bus voltages are fixed at 1.0 pu. "
                    "Use phase angle or line loading queries instead."
                )
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm < threshold]
            if not buses:
                return f"No buses with voltage below {threshold} pu."
            lines = [f"Buses with Vm < {threshold} pu:"]
            for b in sorted(buses, key=lambda b: b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        m = re.search(r"voltage\s+above\s+([\d.]+)", q)
        if m:
            if _dcopflow:
                return (
                    "Voltage magnitude analysis is not available in DCOPFLOW. "
                    "In the DC approximation, all bus voltages are fixed at 1.0 pu. "
                    "Use phase angle or line loading queries instead."
                )
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm > threshold]
            if not buses:
                return f"No buses with voltage above {threshold} pu."
            lines = [f"Buses with Vm > {threshold} pu:"]
            for b in sorted(buses, key=lambda b: -b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        # ── Phase angle queries (DCOPFLOW) ────────────────────────────────
        if re.search(r"phase\s+angle|angle\s+profile|bus\s+angle", q):
            if not opf.buses:
                return "No bus data available."
            sorted_by_angle = sorted(opf.buses, key=lambda b: b.Va)
            lines = [f"Phase angle profile ({len(opf.buses)} buses):"]
            lines.append(f"  Min: {sorted_by_angle[0].Va:.3f}° (bus {sorted_by_angle[0].bus_id})")
            lines.append(f"  Max: {sorted_by_angle[-1].Va:.3f}° (bus {sorted_by_angle[-1].bus_id})")
            ref = min(opf.buses, key=lambda b: abs(b.Va))
            lines.append(f"  Ref: bus {ref.bus_id} ({ref.Va:.3f}°)")
            lines.append("  Most extreme angles:")
            for b in sorted_by_angle[:5]:
                lines.append(f"    Bus {b.bus_id}: Va={b.Va:.3f}°")
            if len(sorted_by_angle) > 5:
                lines.append("    ...")
                for b in sorted_by_angle[-3:]:
                    lines.append(f"    Bus {b.bus_id}: Va={b.Va:.3f}°")
            return "\n".join(lines)

        # ── Line loading ─────────────────────────────────────────────────
        if "most loaded" in q or "loaded lines" in q or "line loading" in q:
            loaded = []
            for br in opf.branches:
                if br.Slim > 0:
                    pct = max(br.Sf, br.St) / br.Slim * 100
                    loaded.append((pct, br))
            loaded.sort(key=lambda x: -x[0])
            lines = ["Most loaded lines:"]
            for pct, br in loaded[:10]:
                flow = max(br.Sf, br.St)
                lines.append(
                    f"  {br.from_bus}->{br.to_bus}: {pct:.1f}% "
                    f"({flow:.2f}/{br.Slim:.2f} MVA)"
                )
            return "\n".join(lines)

        # ── Generator summary ────────────────────────────────────────────
        if "generator" in q:
            lines = ["Generators:"]
            gencost_list = None
            ref_bus_ids = set()
            net_for_info = getattr(self, "_current_network", None) or getattr(self, "_base_network", None)
            if net_for_info is not None and hasattr(net_for_info, "gencost"):
                gencost_list = net_for_info.gencost
                ref_bus_ids = {b.bus_i for b in net_for_info.buses if b.type == 3}
            show_cost = "cost" in q or "coeff" in q or "detail" in q or "dispatch" in q
            for i, g in enumerate(sorted(opf.generators, key=lambda g: -g.Pg)):
                status = "ON" if g.status == 1 else "OFF"
                bus_type = " [SLACK]" if g.bus in ref_bus_ids else ""
                cost_str = ""
                if show_cost and gencost_list is not None and i < len(gencost_list):
                    gc = gencost_list[i]
                    if hasattr(gc, "coeffs") and gc.coeffs:
                        cost_str = f" cost_coeffs={gc.coeffs}"
                lines.append(
                    f"  Bus {g.bus}: {status} Pg={g.Pg:.2f} MW "
                    f"[{g.Pmin:.0f}-{g.Pmax:.0f}] fuel={g.fuel}{bus_type}{cost_str}"
                )
            return "\n".join(lines)

        # ── Voltage profile for a specific kV level ──────────────────────
        m = re.search(r"(\d+(?:\.\d+)?)\s*kv", q)
        if m and ("voltage profile" in q or "kv buses" in q or "buses" in q):
            target_kv = float(m.group(1))
            kv_buses = [b for b in opf.buses if abs(b.base_kv - target_kv) < 1.0]
            if not kv_buses:
                return f"No buses found at {target_kv} kV."
            lines = [f"Voltage profile for {target_kv} kV buses ({len(kv_buses)} buses):"]
            for b in sorted(kv_buses, key=lambda b: b.bus_id):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu, Va={b.Va:.2f}°")
            return "\n".join(lines)

        # ── Area summary ─────────────────────────────────────────────────
        m = re.search(r"area\s+(\d+)", q)
        if m and ("summary" in q or "area" in q):
            area_id = int(m.group(1))
            area_buses = [b for b in opf.buses if b.area == area_id]
            if not area_buses:
                return f"No buses found in area {area_id}."
            total_load = sum(b.Pd for b in area_buses)
            total_gen = sum(
                g.Pg for g in opf.generators
                if any(b.bus_id == g.bus for b in area_buses)
            )
            vm_vals = [b.Vm for b in area_buses]
            lines = [
                f"Area {area_id} Summary ({len(area_buses)} buses):",
                f"  Total load:       {total_load:.2f} MW",
                f"  Total generation: {total_gen:.2f} MW",
                f"  Voltage range:    {min(vm_vals):.4f} – {max(vm_vals):.4f} pu",
            ]
            return "\n".join(lines)

        # ── Cost breakdown by fuel type ──────────────────────────────────
        if "cost breakdown" in q or "generation cost" in q:
            from collections import defaultdict
            fuel_mw: dict[str, float] = defaultdict(float)
            for g in opf.generators:
                if g.status == 1:
                    fuel = g.fuel or "unknown"
                    fuel_mw[fuel] += g.Pg
            if self._config.search.application == "pflow":
                computed_cost = opf.compute_generation_cost(
                    self._current_network.gencost if self._current_network else self._base_network.gencost
                )
                lines = [f"Generation cost breakdown (computed: ${computed_cost:,.2f}):"]
            else:
                lines = [f"Generation cost breakdown (total: ${opf.objective_value:,.2f}):"]
            for fuel, mw in sorted(fuel_mw.items(), key=lambda x: -x[1]):
                lines.append(f"  {fuel:15s}: {mw:8.2f} MW")
            return "\n".join(lines)

        # ── Constraint margins ───────────────────────────────────────────
        if "constraint margin" in q or "binding constraint" in q:
            lines = ["Constraint Margins:"]
            # Voltage limits
            v_min_limit = 0.95
            v_max_limit = 1.05
            voltage_margins = []
            for b in opf.buses:
                margin_low = b.Vm - v_min_limit
                margin_high = v_max_limit - b.Vm
                voltage_margins.append((min(margin_low, margin_high), b))
            voltage_margins.sort(key=lambda x: x[0])
            lines.append("  Tightest voltage margins:")
            for margin, b in voltage_margins[:5]:
                lines.append(f"    Bus {b.bus_id}: Vm={b.Vm:.4f} pu (margin={margin:.4f})")
            # Line loading
            line_margins = []
            for br in opf.branches:
                if br.Slim > 0:
                    loading = max(br.Sf, br.St) / br.Slim * 100
                    line_margins.append((100 - loading, br, loading))
            line_margins.sort(key=lambda x: x[0])
            lines.append("  Lines closest to thermal limit:")
            for headroom, br, loading in line_margins[:5]:
                lines.append(
                    f"    {br.from_bus}->{br.to_bus}: {loading:.1f}% loaded "
                    f"(headroom={headroom:.1f}%)"
                )
            return "\n".join(lines)

        # ── Compare with base case ───────────────────────────────────────
        if "compare with base" in q or "changes from base" in q:
            if self._base_opflow_result is None:
                return "Base case results not available for comparison."
            base = self._base_opflow_result
            curr = opf
            lines = ["Comparison: current vs base case:"]
            if self._config.search.application == "pflow":
                base_cost = base.compute_generation_cost(self._base_network.gencost)
                curr_gencost = self._current_network.gencost if self._current_network else self._base_network.gencost
                curr_cost = curr.compute_generation_cost(curr_gencost)
                delta = curr_cost - base_cost
                pct = delta / base_cost * 100 if base_cost != 0 else 0
                lines.append(
                    f"  Computed cost: ${base_cost:,.2f} → ${curr_cost:,.2f}"
                    f"  ({delta:+,.2f}, {pct:+.1f}%)"
                )
            elif base.objective_value is not None and curr.objective_value is not None:
                delta = curr.objective_value - base.objective_value
                pct = delta / base.objective_value * 100 if base.objective_value != 0 else 0
                lines.append(
                    f"  Cost:       ${base.objective_value:,.2f} → ${curr.objective_value:,.2f}"
                    f"  ({delta:+,.2f}, {pct:+.1f}%)"
                )
            lines.append(
                f"  Voltage:    [{base.voltage_min:.4f}, {base.voltage_max:.4f}] → "
                f"[{curr.voltage_min:.4f}, {curr.voltage_max:.4f}] pu"
            )
            lines.append(
                f"  Generation: {base.total_gen_mw:.2f} → {curr.total_gen_mw:.2f} MW "
                f"({curr.total_gen_mw - base.total_gen_mw:+.2f})"
            )
            lines.append(
                f"  Max loading:{base.max_line_loading_pct:.1f}% → "
                f"{curr.max_line_loading_pct:.1f}% "
                f"({curr.max_line_loading_pct - base.max_line_loading_pct:+.1f}pp)"
            )
            return "\n".join(lines)

        # ── LLM fallback for unrecognized queries ────────────────────────
        logger.info("Analysis query not matched by patterns; using LLM fallback: %s", query)
        try:
            system = (
                "You are analyzing power grid simulation results. "
                "Answer the user's question based on the provided data. Be concise."
            )
            context = self._latest_results_text or "No results available."
            user = f"Question: {query}\n\nCurrent results:\n{context}"
            response = self._backend.complete(system, user)
            return response.raw_text
        except Exception as exc:
            logger.warning("LLM fallback for analysis failed: %s", exc)
            return (
                f"Query not matched and LLM fallback failed: {exc}\n"
                "Available pattern queries:\n"
                "  - 'voltage below/above X'\n"
                "  - 'most loaded lines'\n"
                "  - 'generators'\n"
                "  - '<N>kV voltage profile'\n"
                "  - 'area N summary'\n"
                "  - 'cost breakdown'\n"
                "  - 'constraint margins'\n"
                "  - 'compare with base'"
            )

    # ------------------------------------------------------------------
    # Multi-objective helpers
    # ------------------------------------------------------------------

    def _extract_initial_objectives(self, goal: str) -> None:
        """Use the LLM to extract tracked objectives from the initial goal."""
        try:
            sys_prompt, user_prompt = build_objective_extraction_prompt(
                text=goal,
                available_metrics=available_metrics_for_app(self._config.search.application),
                context="initial_goal",
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            parsed = parse_objective_extraction(response.raw_text)
            if parsed:
                for obj_data in parsed:
                    entry = ObjectiveEntry(
                        name=obj_data["name"],
                        direction=obj_data["direction"],
                        threshold=obj_data.get("threshold"),
                        priority=obj_data["priority"],
                        introduced_at=0,
                        source="initial",
                    )
                    self._journal.objective_registry.register(entry)
                self._print(
                    f"[Objectives] Registered {len(parsed)} objective(s): "
                    f"{', '.join(o['name'] for o in parsed)}"
                )
            else:
                self._journal.objective_registry.register(ObjectiveEntry(
                    name="generation_cost",
                    direction="minimize",
                    priority="primary",
                    introduced_at=0,
                    source="initial",
                ))
                self._print("[Objectives] Defaulted to generation_cost (minimize)")
        except Exception as exc:
            logger.warning("Failed to extract initial objectives: %s", exc)
            self._journal.objective_registry.register(ObjectiveEntry(
                name="generation_cost",
                direction="minimize",
                priority="primary",
                introduced_at=0,
                source="initial",
            ))

    def _extract_objectives_from_steering(self, directive: str, iteration: int) -> None:
        """Extract any new objectives from a steering directive."""
        try:
            sys_prompt, user_prompt = build_objective_extraction_prompt(
                text=directive,
                available_metrics=available_metrics_for_app(self._config.search.application),
                context="steering_directive",
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            parsed = parse_objective_extraction(response.raw_text)
            if parsed:
                for obj_data in parsed:
                    entry = ObjectiveEntry(
                        name=obj_data["name"],
                        direction=obj_data["direction"],
                        threshold=obj_data.get("threshold"),
                        priority=obj_data["priority"],
                        introduced_at=iteration,
                        source="steering",
                    )
                    self._journal.objective_registry.register(entry)
                self._print(
                    f"[Objectives] Steering added/updated {len(parsed)} objective(s): "
                    f"{', '.join(o['name'] for o in parsed)}"
                )
                self._backfill_metrics()
        except Exception as exc:
            logger.warning("Failed to extract objectives from steering: %s", exc)

    def _backfill_metrics(self) -> None:
        """Backfill tracked metrics for all past iterations from stored OPFLOW results.

        Called when new objectives are registered mid-search, so earlier
        iterations get the newly tracked metric values.
        """
        metric_names = [o.name for o in self._journal.objective_registry.objectives]
        for entry in self._journal.entries:
            if entry.mode == "analyze":
                continue
            opflow = self._opflow_results_cache.get(entry.iteration)
            if opflow is not None:
                entry.tracked_metrics = extract_all_metrics(opflow, metric_names)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_prompt(
        self,
        goal: str,
        latest_results_text: Optional[str],
        error_feedback: Optional[str] = None,
        steering_directives: list[dict] | None = None,
        current_iteration: Optional[int] = None,
    ) -> tuple[str, str]:
        """Assemble the system prompt and user prompt for the LLM."""
        journal_text = (
            self._journal.format_for_prompt() if len(self._journal) > 0 else None
        )

        # Multi-objective context
        multi_obj_text = None
        registry = self._journal.objective_registry
        if registry.objectives:
            parts = [registry.format_for_prompt()]
            mo_summary = self._journal.format_multi_objective_summary()
            if mo_summary:
                parts.append("")
                parts.append(mo_summary)
            multi_obj_text = "\n".join(parts)

        # If explore cache is active, inject variant comparison text
        explore_text = None
        if self._explore_cache is not None and self._latest_results_text is not None:
            explore_text = self._latest_results_text

        user_prompt = build_user_prompt(
            goal=goal,
            journal_text=journal_text,
            results_text=latest_results_text,
            error_feedback=error_feedback,
            steering_directives=steering_directives,
            multi_objective_text=multi_obj_text,
            explore_text=explore_text,
            session_best=self._journal.session_best,
            current_iteration=current_iteration,
            max_iterations=self._config.search.max_iterations,
            benchmark_result=self._benchmark_result,
        )
        return self._system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, session: SearchSession, elapsed_seconds: float) -> None:
        """Print summary and save journal."""
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens

        # --- Post-search goal classification via LLM ---
        goal_classification: Optional[dict] = None
        analysis_text: Optional[str] = None
        try:
            raw_stats = self._journal.summary_stats()
            sys_prompt, user_prompt = build_classification_prompts(
                goal=session.goal,
                termination_reason=session.termination_reason,
                stats=raw_stats,
                # Bounded digest avoids context overflow on large sweeps.
                journal_formatted=self._journal.format_for_classification(),
                total_tokens=total_tokens,
                objective_registry=self._journal.objective_registry.to_dict_list(),
                preference_history=self._journal.objective_registry.history,
                application=self._config.search.application,
                near_optimal_abs_tol=self._config.report.near_optimal_abs_tol,
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            analysis_text = response.raw_text
            valid_iters = {e.iteration for e in self._journal.entries}
            goal_classification = parse_goal_classification(analysis_text, valid_iters)
        except Exception as exc:
            logger.warning("Post-search goal classification failed: %s", exc)

        # Resolve stats with override if classification succeeded
        best_iter_override = (
            goal_classification["best_iteration"] if goal_classification else None
        )
        goal_type_override = (
            goal_classification["goal_type"] if goal_classification else None
        )
        stats = self._journal.summary_stats(
            best_iteration_override=best_iter_override,
            goal_type=goal_type_override,
        )

        # Store on session for downstream consumers (GUI, tests)
        session.goal_classification = goal_classification
        session.analysis_text = analysis_text
        session.objective_registry_data = self._journal.objective_registry.to_dict_list()
        session.preference_history = self._journal.objective_registry.history
        session.tcopflow_period_data = self._tcopflow_period_data if self._tcopflow_period_data else None
        session.tcopflow_dT_min = self._tcopflow_dT_min
        session.tcopflow_duration_min = self._tcopflow_duration_min
        session.tcopflow_is_coupling = self._tcopflow_is_coupling
        session.sopflow_num_scenarios = self._sopflow_num_scenarios

        # Always print the final summary (even in quiet mode)
        print()
        print("=" * 60)
        print("  AgentiGrid Search Complete")
        print("=" * 60)
        print(f"  Goal:           {session.goal}")
        print(f"  Application:    {session.application}")
        print(f"  Backend:        {self._backend.name()} ({self._config.llm.model})")
        print(
            f"  Iterations:     {stats['total_iterations']} "
            f"(of max {self._config.search.max_iterations})"
        )
        print(f"  Duration:       {elapsed_seconds:.1f} seconds")
        if total_tokens:
            print(
                f"  Tokens used:    ~{total_tokens:,} "
                f"(prompt: {self._total_prompt_tokens:,}, "
                f"completion: {self._total_completion_tokens:,})"
            )
        print(f"  Termination:    {session.termination_reason}")

        # Goal-aware best-solution line
        goal_type = stats.get("goal_type") or "cost_minimization"
        best_iter = stats["best_iteration"]
        best_obj = stats["best_objective"]
        rationale = (
            goal_classification["best_iteration_rationale"] if goal_classification else None
        )

        if best_obj is None:
            print("  Best solution:  N/A (no feasible solution found)")
        elif self._config.search.application == "pflow":
            best_entry = None
            for e in self._journal.entries:
                if e.iteration == best_iter:
                    best_entry = e
                    break
            feasible_str = "feasible" if best_entry and best_entry.feasible else "infeasible"
            print(
                f"  Best solution:  iteration {best_iter} ({feasible_str})"
            )
            if rationale:
                print(f"  Rationale:      {rationale}")
            print(f"  Search type:    {goal_type.replace('_', ' ')}")
        elif goal_type == "cost_minimization":
            print(
                f"  Best objective: ${best_obj:,.2f} "
                f"(iteration {best_iter})"
            )
        else:
            cost_str = f"${best_obj:,.2f}" if best_obj is not None else "N/A"
            print(f"  Best solution:  iteration {best_iter} — cost={cost_str}")
            if rationale:
                print(f"  Rationale:      {rationale}")
            print(f"  Search type:    {goal_type.replace('_', ' ')}")

        # Print findings if complete
        findings = getattr(self, "_final_findings", None)
        if findings:
            session.final_findings = findings
            summary = findings.get("summary", "")
            if summary:
                print(f"\n  Findings: {summary}")

        print("=" * 60)

        # Print journal table
        print()
        print(self._journal.format_for_prompt())

        # Print truncated analysis text
        if analysis_text:
            print()
            print("─" * 60)
            print("  Post-Search Analysis:")
            print("─" * 60)
            snippet = analysis_text[:500]
            if len(analysis_text) > 500:
                snippet += f"\n  ... ({len(analysis_text) - 500} chars truncated)"
            print(snippet)

        # Multi-objective summary
        registry = self._journal.objective_registry
        if registry.is_multi_objective:
            print()
            print("─" * 60)
            print("  Multi-Objective Summary:")
            print("─" * 60)
            for obj in registry.objectives:
                dir_str = obj.direction
                if obj.direction == "constraint" and obj.threshold is not None:
                    dir_str = f"constraint (\u2264 {obj.threshold})"
                print(f"  {obj.name}: {dir_str} [{obj.priority}] (from iter {obj.introduced_at})")
            if goal_classification and goal_classification.get("tradeoff_summary"):
                print()
                print(f"  Tradeoffs: {goal_classification['tradeoff_summary']}")
            if goal_classification and goal_classification.get("recommended_solutions"):
                recs = goal_classification["recommended_solutions"]
                if len(recs) > 1:
                    print(f"  Recommended solutions: iterations {recs}")

        # PFLOW vs OPFLOW benchmark
        if (
            self._config.search.application == "pflow"
            and self._config.search.benchmark_opflow
        ):
            try:
                from agentigrid.engine.benchmark import run_pflow_vs_opflow_benchmark

                best_iter_num = stats.get("best_iteration")
                goal_type_str = stats.get("goal_type") or goal_type
                pflow_best = self._opflow_results_cache.get(best_iter_num) if best_iter_num is not None else None
                bresult = run_pflow_vs_opflow_benchmark(
                    base_case_path=self._config.search.base_case,
                    pflow_journal=self._journal,
                    config=self._config,
                    goal_type=goal_type_str,
                    pflow_best_result=pflow_best,
                )
                session.benchmark_result = _benchmark_to_dict(bresult)
                self._journal.benchmark_result = _benchmark_to_dict(bresult)
                print()
                print(bresult.summary_text)
            except Exception as exc:
                logger.warning("PFLOW vs OPFLOW benchmark failed: %s", exc)

        # Save journal if configured
        if self._config.output.save_journal:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fmt = self._config.output.journal_format
            journal_path = (
                self._config.output.workdir / f"journal_{timestamp}.{fmt}"
            )
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "csv":
                self._journal.export_csv(journal_path)
            else:
                self._journal.export_json(journal_path)
            print(f"\nJournal saved to: {journal_path}")

    # ------------------------------------------------------------------
    # Session save/resume
    # ------------------------------------------------------------------

    def resume_from(self, save_dir: Path) -> SearchSession:
        """Resume a search from a saved session checkpoint.

        Loads the saved state (journal, objectives, network, steering),
        restores the controller's internal state, and continues the
        agent loop from the next iteration.

        Args:
            save_dir: Path to the saved session directory.

        Returns:
            Completed SearchSession.
        """
        from agentigrid.engine.session_io import load_session

        saved = load_session(save_dir)

        # Restore journal
        for entry in saved["journal_entries"]:
            self._journal.add_entry(entry)

        # Restore objective registry
        self._journal.objective_registry = saved["objective_registry"]

        # Restore steering state
        self._steering_history = saved["steering_history"]
        self._active_steering_directives = saved["active_steering_directives"]

        # Restore TCOPFLOW period data
        self._tcopflow_period_data = saved.get("tcopflow_period_data") or []
        self._tcopflow_dT_min = saved.get("tcopflow_dT_min", 0.0)
        self._tcopflow_duration_min = saved.get("tcopflow_duration_min", 0.0)
        self._tcopflow_is_coupling = saved.get("tcopflow_is_coupling", True)
        if self._tcopflow_period_data:
            self._tcopflow_num_steps = max(len(self._tcopflow_period_data), 0)
        else:
            self._tcopflow_num_steps = max(
                (e.num_steps for e in saved["journal_entries"] if e.num_steps > 0), default=0
            )

        # Restore SOPFLOW scenario count and override
        self._sopflow_num_scenarios = saved.get("sopflow_num_scenarios", 0)
        if not self._sopflow_num_scenarios:
            self._sopflow_num_scenarios = max(
                (e.num_scenarios for e in saved["journal_entries"] if e.num_scenarios > 0), default=0
            )
        _sopflow_scenario_str = saved.get("sopflow_scenario_override")
        self._sopflow_scenario_override = Path(_sopflow_scenario_str) if _sopflow_scenario_str else None

        # Restore TCOPFLOW profile overrides
        _saved_profile_overrides = saved.get("tcopflow_profile_overrides")
        if _saved_profile_overrides:
            self._tcopflow_profile_overrides = {k: Path(v) for k, v in _saved_profile_overrides.items()}
        else:
            self._tcopflow_profile_overrides = {}

        # Restore token counts
        self._total_prompt_tokens = saved["total_prompt_tokens"]
        self._total_completion_tokens = saved["total_completion_tokens"]

        # Parse base case and restore networks
        base_case = saved["base_case_path"]
        goal = saved["goal"]
        self._current_goal = goal

        self._base_network = parse_matpower(base_case)
        self._current_network = saved["current_network"] or self._base_network

        # Rebuild system prompt
        net_summary_text = network_summary(
            self._base_network,
            max_generators=self._config.report.network_summary_max_generators,
        )
        net_metadata_text = network_metadata(self._base_network)
        self._network_metadata_text = net_metadata_text
        # Restore load_factor from journal if saved
        if self._journal.load_factor is not None:
            self._session_load_factor = self._journal.load_factor
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=None,  # benchmark only shown in fresh sessions
            session_load_factor=self._session_load_factor,
        )

        self._latest_results_text = None

        # Handle explore cache: if an explore was in progress at save time,
        # clear it — the variant results cannot be fully restored, so the
        # LLM will need to re-explore from the current network state.
        self._explore_cache = None
        explore_info = saved.get("explore_cache_info")
        if explore_info and explore_info.get("was_active"):
            labels = explore_info.get("variant_labels", [])
            self._print(
                f"[Resume] Note: an explore action was in progress at save time "
                f"(iteration {explore_info.get('iteration', '?')}, "
                f"variants: {', '.join(labels)}). The variant results have been "
                f"discarded — the LLM will need to re-explore."
            )
            logger.info(
                "Cleared explore cache from saved session (variants: %s)",
                labels,
            )

        last_iteration = saved["last_iteration"]

        self._print(
            f"[Resume] Loaded session with {len(self._journal)} entries, "
            f"resuming from iteration {last_iteration + 1}"
        )

        # Notify callback for each restored entry so GUI can display them
        if self._on_iteration:
            for entry in saved["journal_entries"]:
                self._on_iteration(entry.iteration, entry, "restored", None)

        # Build session
        session_start = time.monotonic()
        session = SearchSession(
            goal=goal,
            application=saved["application"],
            base_case_path=base_case,
            config=self._config,
            journal=self._journal,
            start_time=datetime.now().isoformat(),
        )

        # Continue the agent loop from last_iteration + 1
        max_iter = self._config.search.max_iterations
        for iteration in range(last_iteration + 1, max_iter + 1):
            if self._stop_requested:
                session.termination_reason = "user_stopped"
                self._print("\nSearch stopped by user.")
                break
            action_type, should_continue = self._iteration(iteration, goal)
            if self._on_iteration:
                latest_entry = self._journal.latest
                if latest_entry:
                    self._on_iteration(iteration, latest_entry, action_type, self._latest_opflow)
            if not should_continue:
                if not session.termination_reason:
                    session.termination_reason = "completed"
                break
        else:
            session.termination_reason = "max_iterations"
            self._print(f"\nMax iterations ({max_iter}) reached.")

        if not session.termination_reason:
            session.termination_reason = "completed"

        # Final notification
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(
                    latest_entry.iteration, latest_entry,
                    session.termination_reason, self._latest_opflow,
                )

        session.end_time = datetime.now().isoformat()
        session.total_prompt_tokens = self._total_prompt_tokens
        session.total_completion_tokens = self._total_completion_tokens

        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                session.enforced_vmin = min(v[0] for v in limits.values())
                session.enforced_vmax = max(v[1] for v in limits.values())

        elapsed = time.monotonic() - session_start
        self._finalize(session, elapsed)
        return session

    def _serialize_explore_cache(self) -> dict | None:
        """Serialize explore cache metadata for session persistence.

        The full variant results (networks, simulation results) are not
        persisted because they're too large. On resume, the explore cache
        is cleared and the LLM starts fresh from the current network.
        We save only enough metadata to know an explore was in progress
        and log a warning on resume.
        """
        if self._explore_cache is None:
            return None
        cache = self._explore_cache
        return {
            "was_active": True,
            "description": cache.description,
            "reasoning": cache.reasoning,
            "iteration": cache.iteration,
            "variant_labels": list(cache.variants.keys()),
            "base_mode": cache.base_mode,
        }

    def save_session(self, save_dir: Path, config_path: Path | str | None = None) -> Path:
        """Save the current search state to disk for later resumption.

        Args:
            save_dir: Directory to save session files into.
            config_path: Path to the config YAML used (for reference).

        Returns:
            Path to the saved session directory.
        """
        from agentigrid.engine.session_io import save_session as _save_session

        last_entry = self._journal.latest
        last_iteration = last_entry.iteration if last_entry else 0

        enforced_vmin = None
        enforced_vmax = None
        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                enforced_vmin = min(v[0] for v in limits.values())
                enforced_vmax = max(v[1] for v in limits.values())

        return _save_session(
            save_dir=save_dir,
            goal=self._current_goal if hasattr(self, "_current_goal") else "",
            application=self._config.search.application,
            base_case_path=self._config.search.base_case,
            config_path=config_path,
            journal=self._journal,
            steering_history=self._steering_history,
            active_steering_directives=self._active_steering_directives,
            current_network=self._current_network,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            last_iteration=last_iteration,
            enforced_vmin=enforced_vmin,
            enforced_vmax=enforced_vmax,
            tcopflow_period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
            tcopflow_dT_min=self._tcopflow_dT_min,
            tcopflow_duration_min=self._tcopflow_duration_min,
            tcopflow_is_coupling=self._tcopflow_is_coupling,
            sopflow_num_scenarios=self._sopflow_num_scenarios,
            sopflow_scenario_override=str(self._sopflow_scenario_override) if self._sopflow_scenario_override else None,
            tcopflow_profile_overrides={k: str(v) for k, v in self._tcopflow_profile_overrides.items()} if self._tcopflow_profile_overrides else None,
            explore_cache_info=self._serialize_explore_cache(),
        )
