"""Parser for SCOPFLOW text output."""

from __future__ import annotations

import logging
import re
from typing import Optional

from agentigrid.parsers.opflow_parser import parse_opflow_output, _is_marginal_exit, _is_near_boundary
from agentigrid.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("agentigrid.parsers.scopflow")

_NUM_CONTINGENCIES_RE = re.compile(r"^Number of contingencies\s+(\d+)", re.MULTILINE)
_MULTI_PERIOD_RE = re.compile(r"^Multi-period contingencies\?\s+(\S+)", re.MULTILINE)


def parse_scopflow_output(
    stdout: str,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> tuple[OPFLOWResult, dict]:
    """Parse SCOPFLOW text output.

    SCOPFLOW prints the base case solution using the same table format as
    OPFLOW, so we delegate the bus/branch/generator parsing to the OPFLOW
    parser. This function extracts SCOPFLOW-specific metadata first.

    Args:
        stdout: Complete stdout from a SCOPFLOW run.
        bus_limits: Optional per-bus voltage limits for violation checking.

    Returns:
        Tuple of (OPFLOWResult for the base case, metadata dict).
        The metadata dict contains:
          - num_contingencies (int)
          - multi_period (bool)

    Raises:
        ValueError: If the output cannot be parsed.
    """
    if not stdout or "Security-Constrained Optimal Power Flow" not in stdout:
        raise ValueError("Output does not appear to be SCOPFLOW output")

    # Extract SCOPFLOW-specific metadata
    metadata: dict = {}

    m = _NUM_CONTINGENCIES_RE.search(stdout)
    metadata["num_contingencies"] = int(m.group(1)) if m else 0

    m = _MULTI_PERIOD_RE.search(stdout)
    metadata["multi_period"] = (m.group(1).upper() == "YES") if m else False

    # The bus/branch/gen tables and objective value are identical to OPFLOW
    # format. The OPFLOW parser's header check ("Optimal Power Flow" in stdout)
    # passes because "Security-Constrained Optimal Power Flow" contains it.
    result = parse_opflow_output(stdout, bus_limits=bus_limits)

    # SCOPFLOW with EMPAR solver does not produce an Ipopt EXIT message,
    # so the OPFLOW parser may set converged=False even when SCOPFLOW
    # reports CONVERGED in its header. Override based on convergence_status.
    if result.convergence_status == "CONVERGED":
        result.converged = True
    elif result.convergence_status in ("DID", "DID NOT CONVERGE", "DIVERGED"):
        result.converged = False

    # Recompute feasibility_detail for SCOPFLOW-specific logic.
    # EMPAR (solver=EMPAR) always reports CONVERGED but does not check
    # per-contingency convergence, so we rely on power balance and
    # structural violations instead.
    has_power_balance_violation = (
        result.losses_mw < 0 and result.total_load_mw > 0
    )
    is_empar = result.solver.strip().upper() == "EMPAR"

    if result.converged and not has_power_balance_violation:
        # For EMPAR, "CONVERGED" doesn't guarantee per-contingency convergence
        # but with no structural violations and positive losses, we accept it
        result.feasibility_detail = "feasible"
    elif has_power_balance_violation:
        result.feasibility_detail = "infeasible"
        result.converged = False
    elif result.convergence_status in ("DID", "DID NOT CONVERGE", "DIVERGED"):
        if result.ipopt_exit_status and _is_marginal_exit(result.ipopt_exit_status):
            result.feasibility_detail = "marginal"
        elif _is_near_boundary(
            result.buses, result.branches, bus_limits,
            result.max_line_loading_pct, result.losses_mw, result.total_load_mw,
        ):
            result.feasibility_detail = "marginal"
        else:
            result.feasibility_detail = "infeasible"
    elif is_empar and result.converged:
        # EMPAR CONVERGED with no issues — treat as feasible
        # (contingency-level convergence isn't verified by EMPAR)
        result.feasibility_detail = "feasible"
    else:
        result.feasibility_detail = "infeasible"

    return result, metadata


def parse_scopflow_simulation_result(
    sim_result,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> Optional[tuple[OPFLOWResult, dict]]:
    """Parse a SCOPFLOW SimulationResult.

    Returns (OPFLOWResult, metadata_dict) or None if parsing fails.
    """
    if not sim_result.success:
        logger.warning("SCOPFLOW simulation did not succeed — skipping parse")
        return None

    try:
        return parse_scopflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse SCOPFLOW output: %s", exc)
        return None
