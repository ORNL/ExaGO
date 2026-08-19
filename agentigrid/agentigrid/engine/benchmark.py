"""PFLOW vs OPFLOW benchmark: compare LLM-driven PFLOW search results against OPFLOW optimal solution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.engine.journal import SearchJournal
from agentigrid.parsers import parse_simulation_result_for_app
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult
from agentigrid.config import AppConfig

logger = logging.getLogger("agentigrid.engine.benchmark")


@dataclass
class DispatchComparison:
    bus: int
    fuel: str
    opflow_pg: float
    pflow_pg: float
    delta: float
    opflow_pmax: float


@dataclass
class LoadabilityResult:
    opflow_max_factor: float | None
    pflow_max_factor: float | None
    gap_pct: float | None
    detail: str


@dataclass
class BenchmarkResult:
    opflow_converged: bool
    opflow_objective: float | None
    pflow_best_computed_cost: float | None
    cost_gap_pct: float | None
    cost_gap_abs: float | None
    dispatch_comparison: list[DispatchComparison] = field(default_factory=list)
    loadability: LoadabilityResult | None = None
    opflow_result: OPFLOWResult | None = None
    pflow_best_result: OPFLOWResult | None = None
    summary_text: str = ""
    error: str | None = None


def _run_opflow_on_base_case(
    base_case_path: Path,
    config: AppConfig,
) -> OPFLOWResult | None:
    """Run OPFLOW on the given base case and return the parsed result.

    Returns None if OPFLOW fails to run or parse.
    """
    logger.info("Running OPFLOW benchmark on %s", base_case_path)
    try:
        net = parse_matpower(base_case_path)
    except Exception as exc:
        logger.warning("OPFLOW benchmark: failed to parse base case: %s", exc)
        return None

    executor = SimulationExecutor(config.exago, config.output)
    try:
        result = executor.run(net, application="opflow", iteration=-1)
    except Exception as exc:
        logger.warning("OPFLOW benchmark execution failed: %s", exc)
        return None

    if not result.success:
        logger.warning("OPFLOW benchmark did not succeed: %s", result.error_message)
        return None

    parsed = parse_simulation_result_for_app(result, "opflow")
    if parsed is None:
        logger.warning("OPFLOW benchmark output could not be parsed")
        return None

    return parsed


def run_pflow_vs_opflow_benchmark(
    base_case_path: Path,
    pflow_journal: SearchJournal,
    config: AppConfig,
    goal_type: str | None = None,
    pflow_best_result: OPFLOWResult | None = None,
) -> BenchmarkResult:
    """Compare PFLOW search results against OPFLOW optimal solution.

    Runs OPFLOW on the base case, computes generation cost for the best PFLOW
    iteration, and produces a structured comparison.

    Args:
        base_case_path: Path to the MATPOWER .m base case file.
        pflow_journal: The search journal from the PFLOW run.
        config: Application configuration (for OPFLOW executor settings).
        goal_type: The classified goal type (e.g., 'feasibility_boundary').
        pflow_best_result: The PFLOW result for the best iteration (from
            goal classification), used for dispatch and cost comparison.

    Returns:
        BenchmarkResult with comparison data, or BenchmarkResult with error set.
    """
    net = parse_matpower(base_case_path)

    opflow_result = _run_opflow_on_base_case(base_case_path, config)
    if opflow_result is None:
        return BenchmarkResult(
            opflow_converged=False,
            opflow_objective=None,
            pflow_best_computed_cost=None,
            cost_gap_pct=None,
            cost_gap_abs=None,
            error="OPFLOW benchmark failed to produce results",
        )

    opflow_cost = opflow_result.objective_value if opflow_result.objective_value is not None else None

    pflow_cost = None
    if pflow_best_result is not None and net.gencost:
        pflow_cost = pflow_best_result.compute_generation_cost(net.gencost)

    cost_gap_pct = None
    cost_gap_abs = None
    if opflow_cost is not None and pflow_cost is not None and opflow_cost != 0:
        cost_gap_abs = pflow_cost - opflow_cost
        cost_gap_pct = (pflow_cost - opflow_cost) / abs(opflow_cost) * 100

    dispatch_comparison = _build_dispatch_comparison(
        opflow_result, pflow_best_result,
    )

    loadability = None
    if goal_type == "feasibility_boundary":
        loadability = _compute_loadability_comparison(
            pflow_journal, base_case_path, config,
        )

    summary_text = _format_benchmark_summary(
        opflow_converged=opflow_result.converged,
        opflow_cost=opflow_cost,
        pflow_cost=pflow_cost,
        cost_gap_pct=cost_gap_pct,
        cost_gap_abs=cost_gap_abs,
        dispatch_comparison=dispatch_comparison,
        loadability=loadability,
    )

    return BenchmarkResult(
        opflow_converged=opflow_result.converged,
        opflow_objective=opflow_cost,
        pflow_best_computed_cost=pflow_cost,
        cost_gap_pct=cost_gap_pct,
        cost_gap_abs=cost_gap_abs,
        dispatch_comparison=dispatch_comparison,
        loadability=loadability,
        opflow_result=opflow_result,
        pflow_best_result=pflow_best_result,
        summary_text=summary_text,
    )



def _build_dispatch_comparison(
    opflow_result: OPFLOWResult,
    pflow_result: OPFLOWResult | None,
) -> list[DispatchComparison]:
    if pflow_result is None:
        return []

    pflow_gen_map = {g.bus: g for g in pflow_result.generators if g.status == 1}
    comparisons = []

    for opg in opflow_result.generators:
        if opg.status != 1:
            continue
        pgf = pflow_gen_map.get(opg.bus)
        pflow_pg = pgf.Pg if pgf is not None else 0.0
        delta = pflow_pg - opg.Pg
        comparisons.append(DispatchComparison(
            bus=opg.bus,
            fuel=opg.fuel,
            opflow_pg=opg.Pg,
            pflow_pg=pflow_pg,
            delta=delta,
            opflow_pmax=opg.Pmax,
        ))

    comparisons.sort(key=lambda c: abs(c.delta), reverse=True)
    return comparisons


def _compute_loadability_comparison(
    pflow_journal: SearchJournal,
    base_case_path: Path,
    config: AppConfig,
) -> LoadabilityResult | None:
    pflow_max_factor = _extract_pflow_max_factor(pflow_journal)
    if pflow_max_factor is None:
        return None

    opflow_max_factor = _find_opflow_max_loadability(base_case_path, config)
    if opflow_max_factor is None:
        return LoadabilityResult(
            opflow_max_factor=None,
            pflow_max_factor=pflow_max_factor,
            gap_pct=None,
            detail="OPFLOW loadability search did not converge",
        )

    gap_pct = (pflow_max_factor - opflow_max_factor) / opflow_max_factor * 100

    return LoadabilityResult(
        opflow_max_factor=opflow_max_factor,
        pflow_max_factor=pflow_max_factor,
        gap_pct=gap_pct,
        detail=f"OPFLOW \u03bb={opflow_max_factor:.4f}, PFLOW \u03bb={pflow_max_factor:.4f}",
    )


def _extract_pflow_max_factor(pflow_journal: SearchJournal) -> float | None:
    feasible_factors = []
    infeasible_factors = []

    for entry in pflow_journal.entries:
        if entry.iteration == 0:
            continue
        for cmd in entry.commands:
            if cmd.get("action") == "scale_all_loads":
                factor = cmd.get("factor")
                if factor is None:
                    continue
                if entry.feasible:
                    feasible_factors.append(factor)
                else:
                    infeasible_factors.append(factor)

    if feasible_factors:
        max_feasible = max(feasible_factors)
        if infeasible_factors:
            min_infeasible = min(infeasible_factors)
            return (max_feasible + min_infeasible) / 2
        return max_feasible

    return None


def _find_opflow_max_loadability(
    base_case_path: Path,
    config: AppConfig,
    max_search_iterations: int = 20,
    tolerance: float = 0.01,
) -> float | None:
    """Find maximum load scaling factor via OPFLOW binary search."""
    from agentigrid.engine.commands import ScaleAllLoads
    from agentigrid.engine.modifier import apply_modifications

    low = 1.0
    high = 3.0
    base_net = parse_matpower(base_case_path)
    executor = SimulationExecutor(config.exago, config.output)

    for _ in range(max_search_iterations):
        mid = (low + high) / 2
        cmd = ScaleAllLoads(factor=mid)
        try:
            modified, _ = apply_modifications(base_net, [cmd])
        except Exception:
            high = mid
            continue

        try:
            result = executor.run(modified, application="opflow", iteration=-1)
        except Exception:
            high = mid
            continue

        if not result.success:
            high = mid
            continue

        parsed = parse_simulation_result_for_app(result, "opflow")
        if parsed is not None and parsed.converged and parsed.feasibility_detail == "feasible":
            low = mid
        else:
            high = mid

        if high - low < tolerance:
            break

    return (low + high) / 2


def _format_benchmark_summary(
    opflow_converged: bool,
    opflow_cost: float | None,
    pflow_cost: float | None,
    cost_gap_pct: float | None,
    cost_gap_abs: float | None,
    dispatch_comparison: list[DispatchComparison],
    loadability: LoadabilityResult | None,
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  PFLOW vs OPFLOW Benchmark")
    lines.append("=" * 60)

    if not opflow_converged:
        lines.append("  OPFLOW did not converge - no comparison available.")
        return "\n".join(lines)

    if opflow_cost is not None:
        lines.append(f"  OPFLOW optimal cost:        ${opflow_cost:,.2f}")
    if pflow_cost is not None:
        lines.append(f"  Best PFLOW computed cost:   ${pflow_cost:,.2f}")
    if cost_gap_pct is not None:
        sign = "+" if cost_gap_pct >= 0 else ""
        lines.append(f"  Cost gap:                  {sign}{cost_gap_pct:.2f}%")
    if cost_gap_abs is not None:
        sign = "+" if cost_gap_abs >= 0 else ""
        lines.append(f"  Cost difference:           {sign}${cost_gap_abs:,.2f}")

    if dispatch_comparison:
        lines.append("")
        lines.append("  Dispatch comparison (sorted by |delta|):")
        lines.append(
            f"  {'Gen bus':<10} {'Fuel':<8} {'OPFLOW MW':>12} "
            f"{'PFLOW MW':>12} {'Delta MW':>12} {'% of Pmax':>10}"
        )
        for dc in dispatch_comparison[:10]:
            pct_pmax = (dc.delta / dc.opflow_pmax * 100) if dc.opflow_pmax > 0 else 0
            lines.append(
                f"  {dc.bus:<10} {dc.fuel:<8} {dc.opflow_pg:>12.2f} "
                f"{dc.pflow_pg:>12.2f} {dc.delta:>+12.2f} {pct_pmax:>+10.1f}%"
            )

    if loadability is not None:
        lines.append("")
        lines.append("  Loadability comparison:")
        if loadability.opflow_max_factor is not None:
            lines.append(f"  OPFLOW max load factor:    {loadability.opflow_max_factor:.4f}")
        if loadability.pflow_max_factor is not None:
            lines.append(f"  PFLOW max load factor:     {loadability.pflow_max_factor:.4f}")
        if loadability.gap_pct is not None:
            lines.append(f"  Boundary gap:              {loadability.gap_pct:+.2f}%")
        if loadability.detail:
            lines.append(f"  {loadability.detail}")

    lines.append("=" * 60)
    return "\n".join(lines)
