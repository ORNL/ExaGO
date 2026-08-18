"""Parser for SOPFLOW text output."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from agentigrid.parsers.opflow_parser import parse_opflow_output, _is_marginal_exit, _is_near_boundary
from agentigrid.parsers.opflow_results import OPFLOWResult
from agentigrid.parsers.sopflow_dispatch import all_scenarios_converged

logger = logging.getLogger("agentigrid.parsers.sopflow")

_NUM_SCENARIOS_RE = re.compile(r"^Number of scenarios\s+(\d+)", re.MULTILINE)
_MULTI_CONTINGENCY_RE = re.compile(r"^Multi-contingency scenarios\?\s+(\S+)", re.MULTILINE)
_CONTINGENCIES_PER_SCENARIO_RE = re.compile(r"^Contingencies per scenario\s+(\d+)", re.MULTILINE)
_LOAD_LOSS_RE = re.compile(r"^Load loss allowed\s+(\S+)", re.MULTILINE)
_POWER_IMBALANCE_RE = re.compile(r"^Power imbalance allowed\s+(\S+)", re.MULTILINE)
_IGNORE_LINEFLOW_RE = re.compile(r"^Ignore line flow constraints\s+(\S+)", re.MULTILINE)
_CONVERGENCE_RE = re.compile(r"^Convergence status\s+(.+)$", re.MULTILINE)
_OBJC_VALUE_RE = re.compile(r"^Objective value\s+(?:\(base\)\s+)?([\d.eE+-]+)", re.MULTILINE)
_SOLVER_RE = re.compile(r"^Solver\s+(\S+)", re.MULTILINE)

# Voltage band used only when no per-bus limits are available (case fallback).
_VMIN_FALLBACK = 0.9
_VMAX_FALLBACK = 1.1

# Marginal annotation reusing the "marginal convergence … use with caution" vocabulary
# of the SCOPFLOW / PFLOW / TCOPFLOW paths, specialized for EMPAR's advisory flags.
_EMPAR_MARGINAL_NOTE = (
    "Marginal: EMPAR reported partial per-scenario non-convergence, but the "
    "base-case solution checks (power balance, voltage band, no violations, "
    "positive objective) all pass — treated as feasible (marginal); use with caution."
)


def _enforced_voltage_band(
    bus_limits: dict[int, tuple[float, float]] | None,
) -> tuple[float, float]:
    """A single (Vmin, Vmax) band from the active per-bus limits, else case fallback.

    Uses the enforced envelope (min of the per-bus Vmin floors, max of the Vmax
    ceilings); for the usual uniform band this is exactly that band.
    """
    if bus_limits:
        vmins = [lim[0] for lim in bus_limits.values()]
        vmaxs = [lim[1] for lim in bus_limits.values()]
        if vmins and vmaxs:
            return min(vmins), max(vmaxs)
    return _VMIN_FALLBACK, _VMAX_FALLBACK


def sopflow_solution_feasible(
    result: OPFLOWResult,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> tuple[bool, dict]:
    """Judge SOPFLOW base-case feasibility from the SOLUTION, not from flags.

    EMPAR's header status and per-scenario ``mpc.converged`` flags are unreliable
    in both directions (falsely CONVERGED, falsely non-converged), so feasibility
    is decided from the parsed base-case metrics:

    - ``balanced``      generation within 5% of load (covers losses; catches the
                         old unsolved-echo where gen ran ~45% over load).
    - ``within_band``   aggregate voltage min/max inside the enforced band.
    - ``no_violations`` the parser found no bus/branch/power-balance violation.
    - ``has_objective`` a positive objective was parsed (not None / not 0).

    Returns ``(feasible, checks)`` where ``checks`` is the per-check breakdown.
    """
    load = result.total_load_mw
    gen = result.total_gen_mw
    balanced = abs(gen - load) / max(load, 1e-6) <= 0.05
    vmin_band, vmax_band = _enforced_voltage_band(bus_limits)
    within_band = (
        result.voltage_min >= vmin_band - 1e-3
        and result.voltage_max <= vmax_band + 1e-3
    )
    no_violations = result.num_violations == 0
    has_objective = result.objective_value is not None and result.objective_value > 0
    checks = {
        "balanced": balanced,
        "within_band": within_band,
        "no_violations": no_violations,
        "has_objective": has_objective,
    }
    return (balanced and within_band and no_violations and has_objective), checks


def parse_sopflow_output(
    stdout: str,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> tuple[OPFLOWResult, dict]:
    """Parse SOPFLOW text output.

    SOPFLOW prints the base-case solution using the same table format as
    OPFLOW, preceded by a stochastic-specific header. This function extracts
    SOPFLOW-specific metadata first, then delegates table parsing to the
    OPFLOW parser.

    Args:
        stdout: Complete stdout from a SOPFLOW run.
        bus_limits: Optional per-bus voltage limits for violation checking.

    Returns:
        Tuple of (OPFLOWResult, metadata dict).
        The metadata dict contains:
          - num_scenarios (int)
          - multi_contingency (bool)
          - is_coupling (bool)

    Raises:
        ValueError: If the output cannot be parsed.
    """
    if not stdout or "Stochastic Optimal Power Flow" not in stdout:
        raise ValueError("Output does not appear to be SOPFLOW output")

    metadata: dict = {}

    m = _NUM_SCENARIOS_RE.search(stdout)
    metadata["num_scenarios"] = int(m.group(1)) if m else 0

    m = _MULTI_CONTINGENCY_RE.search(stdout)
    metadata["multi_contingency"] = (m.group(1).upper() == "YES") if m else False

    m = _CONTINGENCIES_PER_SCENARIO_RE.search(stdout)
    metadata["contingencies_per_scenario"] = int(m.group(1)) if m else 0

    # is_coupling is NOT printed in SOPFLOW output; derive from config
    # defaulting to False (matching the default -sopflow_iscoupling 0)
    metadata["is_coupling"] = False

    m = _LOAD_LOSS_RE.search(stdout)
    metadata["load_loss_allowed"] = (m.group(1).upper() == "YES") if m else False

    m = _POWER_IMBALANCE_RE.search(stdout)
    metadata["power_imbalance_allowed"] = (m.group(1).upper() == "YES") if m else False

    m = _IGNORE_LINEFLOW_RE.search(stdout)
    metadata["ignore_lineflow"] = (m.group(1).upper() == "YES") if m else False

    metadata["convergence_status"] = ""
    m = _CONVERGENCE_RE.search(stdout)
    if m:
        metadata["convergence_status"] = m.group(1).strip()

    metadata["objective_value"] = None
    m = _OBJC_VALUE_RE.search(stdout)
    if m:
        try:
            metadata["objective_value"] = float(m.group(1))
        except ValueError:
            pass

    metadata["solver"] = ""
    m = _SOLVER_RE.search(stdout)
    if m:
        metadata["solver"] = m.group(1).strip()

    # The bus/branch/gen tables and objective value are identical to OPFLOW
    # format. The OPFLOW parser's header check ("Optimal Power Flow" in stdout)
    # passes because "Stochastic Optimal Power Flow" contains it.
    result = parse_opflow_output(stdout, bus_limits=bus_limits)

    # SOPFLOW may not produce an IPOPT EXIT message in the output, so the
    # OPFLOW parser may set converged=False even when SOPFLOW reports
    # CONVERGED in its header. Override based on convergence_status.
    conv_status = metadata.get("convergence_status", "")
    if conv_status.strip() == "CONVERGED":
        result.converged = True
    elif conv_status.strip() in ("DID NOT CONVERGE", "DIVERGED"):
        result.converged = False

    # SOPFLOW-specific feasibility logic.
    # Both IPOPT and EMPAR can be used. EMPAR with MPI is common for large cases.
    has_power_balance_violation = (
        result.losses_mw < 0 and result.total_load_mw > 0
    )

    if result.converged and not has_power_balance_violation:
        result.feasibility_detail = "feasible"
    elif has_power_balance_violation:
        result.feasibility_detail = "infeasible"
        result.converged = False
    elif not result.converged:
        if result.ipopt_exit_status and _is_marginal_exit(result.ipopt_exit_status):
            result.feasibility_detail = "marginal"
        elif _is_near_boundary(
            result.buses, result.branches, bus_limits,
            result.max_line_loading_pct, result.losses_mw, result.total_load_mw,
        ):
            result.feasibility_detail = "marginal"
        else:
            result.feasibility_detail = "infeasible"
    else:
        result.feasibility_detail = "infeasible"

    return result, metadata


def parse_sopflow_simulation_result(
    sim_result,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> Optional[tuple[OPFLOWResult, dict]]:
    """Parse a SOPFLOW SimulationResult.

    Returns (OPFLOWResult, metadata_dict) or None if parsing fails.
    """
    if not sim_result.success:
        logger.warning("SOPFLOW simulation did not succeed — skipping parse")
        return None

    try:
        result, metadata = parse_sopflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse SOPFLOW output: %s", exc)
        return None

    # Solution-based feasibility for EMPAR (and any non-IPOPT solver). EMPAR's
    # header status and per-scenario mpc.converged flags are unreliable in BOTH
    # directions — falsely CONVERGED at the header, and falsely non-converged per
    # scenario (an identical problem re-solved under IPOPT returns Optimal). So we
    # judge feasibility from the SOLUTION, never from the flags. IPOPT's status is
    # reliable and its solutions pass these same checks, so IPOPT is left untouched.
    solver = (metadata.get("solver") or result.solver or "").strip().upper()
    workdir = getattr(sim_result, "workdir", None)
    if solver != "IPOPT":
        feasible, checks = sopflow_solution_feasible(result, bus_limits)
        if feasible:
            result.converged = True
            result.convergence_status = "CONVERGED"
            result.feasibility_detail = "feasible"  # keep parsed objective_value
            metadata["convergence_status"] = "CONVERGED"
            metadata["sopflow_solution_feasible"] = True
            metadata["solution_checks"] = checks
            # Per-scenario flags are ADVISORY: if EMPAR flagged partial
            # non-convergence but the solution checks pass, keep feasible and
            # annotate as marginal — never downgrade a good solution to FAILED.
            if solver == "EMPAR" and workdir is not None:
                if all_scenarios_converged(Path(workdir)) is False:
                    logger.info(
                        "SOPFLOW/EMPAR flagged partial per-scenario non-convergence "
                        "in %s, but the base-case solution checks pass — feasible "
                        "(marginal).", workdir,
                    )
                    result.convergence_status = "CONVERGED (marginal)"
                    metadata["convergence_status"] = "CONVERGED (marginal)"
                    metadata["empar_marginal"] = True
                    metadata["marginal_note"] = _EMPAR_MARGINAL_NOTE
        else:
            # Genuine infeasibility (imbalanced, out-of-band, violations, or no
            # positive objective): never report feasible or a $0.00 base cost.
            logger.warning(
                "SOPFLOW/%s base case fails solution checks %s — marking DID NOT "
                "CONVERGE (objective N/A).", solver or "?", checks,
            )
            result.converged = False
            result.convergence_status = "DID NOT CONVERGE"
            result.feasibility_detail = "infeasible"
            result.objective_value = None  # NOT 0.0 — no feasible base cost exists
            metadata["convergence_status"] = "DID NOT CONVERGE"
            metadata["sopflow_solution_feasible"] = False
            metadata["solution_checks"] = checks

    return result, metadata
