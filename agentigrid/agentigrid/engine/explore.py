"""Concurrent PFLOW explore/select data structures and result formatting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from agentigrid.engine.commands import ModCommand
from agentigrid.engine.executor import SimulationResult
from agentigrid.engine.journal import ObjectiveEntry
from agentigrid.engine.metric_extractor import extract_all_metrics
from agentigrid.engine.pareto import ParetoCandidate, pareto_filter
from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("agentigrid.engine.explore")


def _describe_command(cmd: ModCommand) -> str:
    """Return a compact abbreviation for a single command."""
    name = type(cmd).__name__
    if name == "ScaleAllLoads":
        return f"scale×{cmd.factor:g}"
    if name == "ScaleLoad":
        scope = (f"bus{cmd.bus}" if cmd.bus else
                 f"area{cmd.area}" if cmd.area else f"zone{cmd.zone}")
        return f"scale×{cmd.factor:g}({scope})"
    if name == "SetGenDispatch":
        return f"dispatch bus{cmd.bus}→{cmd.Pg:g}MW"
    if name == "SetGenVoltage":
        return f"Vg bus{cmd.bus}={cmd.Vg:g}"
    if name == "SetGenStatus":
        action = "commit" if cmd.status == 1 else "trip"
        return f"{action} bus{cmd.bus}"
    if name == "SetLoad":
        parts = []
        if cmd.Pd is not None:
            parts.append(f"Pd={cmd.Pd:g}")
        if cmd.Qd is not None:
            parts.append(f"Qd={cmd.Qd:g}")
        return f"load bus{cmd.bus}({','.join(parts)})"
    if name == "SetAllBusVLimits":
        lo = f"{cmd.Vmin:g}" if cmd.Vmin is not None else "?"
        hi = f"{cmd.Vmax:g}" if cmd.Vmax is not None else "?"
        return f"vlim[{lo}-{hi}]"
    if name == "SetBusVLimits":
        lo = f"{cmd.Vmin:g}" if cmd.Vmin is not None else "?"
        hi = f"{cmd.Vmax:g}" if cmd.Vmax is not None else "?"
        return f"vlim bus{cmd.bus}[{lo}-{hi}]"
    if name == "SetBranchStatus":
        action = "enable" if cmd.status == 1 else "disable"
        return f"{action} br{cmd.fbus}-{cmd.tbus}"
    if name == "SetBranchRate":
        return f"rate br{cmd.fbus}-{cmd.tbus}={cmd.rateA:g}"
    if name == "SetCostCoeffs":
        return f"cost bus{cmd.bus}"
    if name == "ScaleLoadProfile":
        return f"profile×{cmd.factor:g}"
    if name == "ScaleWindScenario":
        return f"wind×{cmd.factor:g}"
    if name == "SetTapRatio":
        return f"tap br{cmd.fbus}-{cmd.tbus}={cmd.ratio:g}"
    if name == "SetShuntSusceptance":
        return f"shunt bus{cmd.bus}={cmd.Bs:g}"
    if name == "SetPhaseShiftAngle":
        return f"phase br{cmd.fbus}-{cmd.tbus}={cmd.angle:g}deg"
    return name


def build_variant_description(
    commands: list[ModCommand],
    skipped_cmds: list[tuple[ModCommand, list[str]]],
) -> str:
    """Build a compact human-readable description for a variant.

    Marks skipped commands with [SKIP] inline and appends a parenthetical
    skip summary at the end. Targets ≤ 80 characters for the main part.

    Args:
        commands: All parsed commands for this variant (applied + skipped).
        skipped_cmds: Pairs of (cmd, reasons) for commands that were skipped.

    Returns:
        Human-readable description string.
    """
    if not commands:
        return "(no commands)"

    skipped_ids = {id(cmd) for cmd, _ in skipped_cmds}

    tokens: list[str] = []
    for cmd in commands:
        token = _describe_command(cmd)
        if id(cmd) in skipped_ids:
            token += " [SKIP]"
        tokens.append(token)

    cmd_summary = ", ".join(tokens)
    if len(cmd_summary) > 60:
        cmd_summary = cmd_summary[:57] + "..."

    if not skipped_cmds:
        return cmd_summary

    # Group skipped commands by their first reason for a compact suffix
    reason_tokens: dict[str, list[str]] = {}
    for cmd, reasons in skipped_cmds:
        reason = reasons[0] if reasons else "unknown"
        reason_tokens.setdefault(reason, []).append(_describe_command(cmd))

    n_total = len(skipped_cmds)
    reason_strs = []
    for reason, labels in reason_tokens.items():
        prefix = ", ".join(labels[:3])
        if len(labels) > 3:
            prefix += f", +{len(labels) - 3}"
        reason_strs.append(f"{prefix}: {reason[:35]}")

    suffix = f" ({n_total} SKIPPED: {'; '.join(reason_strs[:2])})"
    if len(suffix) > 70:
        suffix = f" ({n_total} SKIPPED)"

    return cmd_summary + suffix


@dataclass
class VariantResult:
    """Result of a single variant simulation within an explore action."""

    label: str
    description: str
    commands: list[ModCommand]
    raw_commands: list[dict]
    modified_net: MATNetwork
    sim_result: SimulationResult
    opflow_result: Optional[OPFLOWResult]
    is_pareto: bool = False
    skipped_commands: list[tuple[ModCommand, list[str]]] = field(default_factory=list)
    # rejected=True means the variant was not simulated because every one of
    # its commands would have been a no-op against the base network. This is
    # distinct from feasibility (a physics property of the simulated solution).
    rejected: bool = False
    # cost_equivalent_to: set to the label of a simpler variant that achieved
    # the same cost. Empty string when no cost equivalence is detected.
    cost_equivalent_to: str = ""


@dataclass
class ExploreCache:
    """Temporary storage for explore results awaiting a select action."""

    variants: dict[str, VariantResult] = field(default_factory=dict)
    description: str = ""
    reasoning: str = ""
    iteration: int = 0
    base_network_snapshot: Optional[MATNetwork] = None
    base_mode: str = "accumulative"


def variant_metrics(
    variant: VariantResult,
    objectives: list[ObjectiveEntry],
    gencost: list | None = None,
) -> dict[str, float]:
    """Extract metrics from a variant for Pareto comparison.

    For PFLOW, if no objectives are registered, uses computed generation
    cost as the default metric.
    """
    if variant.opflow_result is None:
        return {}

    metrics: dict[str, float] = {}

    if objectives:
        obj_names = [o.name for o in objectives]
        extracted = extract_all_metrics(variant.opflow_result, obj_names)
        metrics.update(extracted)

        # For PFLOW: override generation_cost with computed cost if gencost is available
        is_pflow = any(
            g.fuel.lower() == "" or g.fuel.lower() != ""
            for g in variant.opflow_result.generators[:1]
        )
        # Always compute generation cost for PFLOW since objective_value is 0.0
        if "generation_cost" in obj_names and gencost is not None:
            computed = variant.opflow_result.compute_generation_cost(gencost)
            if computed > 0:
                metrics["generation_cost"] = computed
    else:
        # No objectives registered: use computed generation cost as default
        if gencost is not None:
            computed = variant.opflow_result.compute_generation_cost(gencost)
            metrics["generation_cost"] = computed

    return metrics


def compute_pareto_labels(
    variants: dict[str, VariantResult],
    objectives: list[ObjectiveEntry],
    gencost: list | None = None,
) -> list[str]:
    """Compute Pareto-optimal labels and mark variants accordingly.

    Returns the list of Pareto-optimal labels. Variants in the dict
    are modified in-place: ``is_pareto`` is set on each.

    A variant is considered feasible only if:
    - The simulation converged (feasibility_detail == "feasible"), AND
    - There are zero constraint violations (num_violations == 0).

    This ensures that variants with voltage violations (e.g., Vmin < 0.95
    when tight limits are enforced via set_all_bus_vlimits) are classified
    as infeasible even though PFLOW reports convergence.
    """
    candidates: list[ParetoCandidate] = []
    for label, v in variants.items():
        if v.rejected:
            # Rejected variants never ran a physics solve — they cannot be
            # Pareto candidates. Exclude them entirely so the front is built
            # from real simulation outcomes only.
            continue
        feasible = (
            v.opflow_result is not None
            and v.opflow_result.feasibility_detail == "feasible"
            and v.opflow_result.num_violations == 0
        )
        metrics = variant_metrics(v, objectives, gencost)
        candidates.append(ParetoCandidate(
            label=label, metrics=metrics, feasible=feasible,
        ))

    pareto_labels = pareto_filter(candidates, objectives)

    for label, v in variants.items():
        v.is_pareto = label in pareto_labels

    return pareto_labels


def annotate_cost_equivalent_siblings(
    variants: dict[str, VariantResult],
    gencost: list | None = None,
) -> str | None:
    """Detect identical-cost feasible variants and annotate the more complex ones.

    When two or more feasible variants return the same rounded cost (to 2 dp),
    it usually means the extra commands in the more complex variant had no net
    effect. The variant with the fewest commands is treated as the reference;
    all others get an annotation added to their ``description`` and their
    ``cost_equivalent_to`` field set to the reference label.

    Args:
        variants: All variant results for this explore iteration.
        gencost: GenCost data for cost computation, or None.

    Returns:
        A batch-level warning string when siblings are found, or None.
    """
    # Collect costs for feasible variants
    costs: dict[str, float] = {}
    for label, v in variants.items():
        if v.rejected or v.opflow_result is None:
            continue
        if not (v.opflow_result.feasibility_detail == "feasible"
                and v.opflow_result.num_violations == 0):
            continue
        if gencost is not None:
            try:
                c = v.opflow_result.compute_generation_cost(gencost)
            except Exception:
                continue
        else:
            c = v.opflow_result.objective_value
        if c is not None and c > 0:
            costs[label] = c

    # Group by rounded cost
    cost_groups: dict[float, list[str]] = {}
    for label, c in costs.items():
        key = round(c, 2)
        cost_groups.setdefault(key, []).append(label)

    sibling_groups: list[tuple[str, list[str]]] = []
    for cost_key, labels in cost_groups.items():
        if len(labels) < 2:
            continue
        # Reference = fewest commands (applied + skipped)
        labels.sort(key=lambda lbl: len(variants[lbl].commands))
        ref_label = labels[0]
        others = labels[1:]
        for other in others:
            v = variants[other]
            ann = f" \u2190 same cost as {ref_label}; extra commands had no effect"
            if ann not in v.description:
                v.description += ann
            v.cost_equivalent_to = ref_label
        sibling_groups.append((ref_label, others))

    if not sibling_groups:
        return None

    parts: list[str] = []
    for ref_label, others in sibling_groups:
        labels_str = ", ".join(others)
        parts.append(
            f"\u26a0 Variant(s) {labels_str} returned the same cost as {ref_label} "
            f"\u2014 their additional commands had no effect. "
            "If this was due to skipped commands, reconsider those commands in the next explore."
        )
    return "\n".join(parts)


def format_variant_results(
    variants: dict[str, VariantResult],
    pareto_labels: list[str],
    gencost: list | None = None,
) -> str:
    """Format variant results as a comparative table for the LLM prompt.

    Shows a compact table with key metrics for each variant, with Pareto-optimal
    variants marked with a star. Includes guidance on how to select a variant.
    """
    if not variants:
        return "No variant results to display."

    lines: list[str] = []
    lines.append(f"=== Neighborhood Exploration Results ({len(variants)} variants) ===")
    lines.append("")

    # Header
    lines.append(
        f"{'Variant':<8} | {'Description':<28} | {'Cost($)':>12} | "
        f"{'Feas.':>5} | {'V_range':>13} | {'Max_load':>8} | "
        f"{'V viol':>5} | {'Pareto':>6}"
    )
    lines.append("-" * 100)

    for label, v in variants.items():
        desc = v.description[:28]
        if v.rejected:
            cost_str = "REJECTED"
            feas = "—"
            vr = "—"
            ml = "—"
            vv = "—"
        elif v.opflow_result is not None:
            opf = v.opflow_result
            cost_str = "N/A"
            if gencost is not None:
                try:
                    computed = opf.compute_generation_cost(gencost)
                    cost_str = f"${computed:,.2f}"
                except Exception:
                    pass

            feas = opf.feasibility_detail if opf.feasibility_detail else (
                "Yes" if opf.converged else "No"
            )

            if opf.voltage_min > 0 and opf.voltage_max > 0:
                vr = f"{opf.voltage_min:.3f}-{opf.voltage_max:.3f}"
            else:
                vr = "N/A"

            ml = f"{opf.max_line_loading_pct:.1f}%"
            vv = str(opf.num_violations)
        else:
            cost_str = "FAIL"
            feas = "No"
            vr = "N/A"
            ml = "N/A"
            vv = "-"

        pareto_mark = "  ★" if v.is_pareto else ""
        lines.append(
            f"{label:<8} | {desc:<28} | {cost_str:>12} | "
            f"{ feas:>5} | {vr:>13} | {ml:>8} | "
            f"{vv:>5} | {pareto_mark:>6}"
        )

        if v.skipped_commands:
            prefix = "REJECTED" if v.rejected else "Skipped"
            for cmd, reasons in v.skipped_commands:
                cmd_name = type(cmd).__name__
                reason_str = "; ".join(reasons)
                lines.append(f"         ⚠ {prefix} {cmd_name}: {reason_str}")

    lines.append("")

    # Pareto summary
    if pareto_labels:
        lines.append(f"Pareto-optimal variants (★): {', '.join(pareto_labels)}")
    else:
        lines.append("No Pareto-optimal variants (all infeasible or marginal).")

    lines.append("")
    lines.append(
        'To adopt a variant as the new current point, respond with:\n'
        '{"action": "select", "choice": "<label>", '
        '"reasoning": "why this variant is best"}\n\n'
        'You can also:\n'
        '- "explore" to evaluate a different neighborhood\n'
        '- "analyze" to inspect detailed results of any variant\n'
        '- "modify" to make a single-point change (clears explore state)\n'
        '- "complete" to end the search'
    )

    return "\n".join(lines)