"""Parser for SOPFLOW text output."""

from __future__ import annotations

import logging
import re
from typing import Optional

from agentigrid.parsers.opflow_parser import parse_opflow_output, _is_marginal_exit, _is_near_boundary
from agentigrid.parsers.opflow_results import OPFLOWResult

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
        return parse_sopflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse SOPFLOW output: %s", exc)
        return None