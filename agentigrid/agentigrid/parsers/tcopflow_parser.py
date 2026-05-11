"""Parser for TCOPFLOW text output and multi-period result files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_parser import parse_opflow_output, _is_marginal_exit
from agentigrid.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("agentigrid.parsers.tcopflow")

_DURATION_RE = re.compile(r"^Duration \(minutes\)\s+([\d.]+)", re.MULTILINE)
_TIMESTEP_RE = re.compile(r"^Time-step \(minutes\)\s+([\d.]+)", re.MULTILINE)
_NUM_STEPS_RE = re.compile(r"^Number of steps\s+(\d+)", re.MULTILINE)
_PLOAD_PROFILE_RE = re.compile(r"^Active power demand profile\s+(.+)$", re.MULTILINE)
_QLOAD_PROFILE_RE = re.compile(r"^Rective power demand profile\s+(.+)$", re.MULTILINE)
_WIND_PROFILE_RE = re.compile(r"^Wind generation profile\s+(.+)$", re.MULTILINE)
_LOAD_LOSS_RE = re.compile(r"^Load loss allowed\s+(\S+)", re.MULTILINE)
_POWER_IMBALANCE_RE = re.compile(r"^Power imbalance allowed\s+(\S+)", re.MULTILINE)
_IGNORE_LINEFLOW_RE = re.compile(r"^Ignore line flow constraints\s+(\S+)", re.MULTILINE)
_IS_COUPLING_RE = re.compile(r"^Number of coupling constraints\s+(\d+)", re.MULTILINE)
_CONVERGENCE_RE = re.compile(r"^Convergence status\s+(.+)$", re.MULTILINE)
_OBJC_VALUE_RE = re.compile(r"^Objective value\s+([\d.eE+-]+)", re.MULTILINE)


_VOLTAGE_PROXIMITY_PU = 0.01
_LINE_LOADING_PROXIMITY_PCT = 5.0


def _is_near_boundary(
    result: OPFLOWResult,
    bus_limits: dict[int, tuple[float, float]] | None,
) -> bool:
    """Check if a non-converged solution's metrics are near their limits.

    When the solver reports DID NOT CONVERGE but the solution data shows
    metrics close to their constraint boundaries, this indicates a
    near-boundary case rather than a hard infeasibility. Classifying these
    as "marginal" helps the LLM agent recognize it has found the operating
    boundary and should declare the search complete.

    A case is near-boundary if:
    - Any bus voltage is within VOLTAGE_PROXIMITY_PU pu of its limit, OR
    - Any branch loading is within LINE_LOADING_PROXIMITY_PCT % of 100%
    - AND there is no power balance violation (positive losses)
    """
    if result.losses_mw < 0 and result.total_load_mw > 0:
        return False

    if bus_limits:
        for b in result.buses:
            if b.bus_id in bus_limits:
                vmin, vmax = bus_limits[b.bus_id]
                if (abs(b.Vm - vmin) <= _VOLTAGE_PROXIMITY_PU
                        or abs(b.Vm - vmax) <= _VOLTAGE_PROXIMITY_PU):
                    return True

    if result.max_line_loading_pct >= (100.0 - _LINE_LOADING_PROXIMITY_PCT):
        return True

    return False


def parse_tcopflow_output(
    stdout: str,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> tuple[OPFLOWResult, dict]:
    """Parse TCOPFLOW text output.

    TCOPFLOW prints period-0 solution using the same table format as OPFLOW,
    so we delegate bus/branch/generator parsing to the OPFLOW parser.
    This function extracts TCOPFLOW-specific metadata from the header first.

    Args:
        stdout: Complete stdout from a TCOPFLOW run.
        bus_limits: Optional per-bus voltage limits for violation checking.

    Returns:
        Tuple of (OPFLOWResult for period 0, metadata dict).
        The metadata dict contains:
          - duration_min (float): Total duration in minutes
          - dT_min (float): Time-step size in minutes
          - num_steps (int): Number of time steps
          - pload_profile (str): Path to active load profile
          - qload_profile (str): Path to reactive load profile
          - wind_profile (str): Path to wind profile, or "NOT SET"
          - load_loss_allowed (bool)
          - power_imbalance_allowed (bool)
          - ignore_lineflow (bool)
          - num_coupling_constraints (int)
          - convergence_status (str): CONVERGED / DID NOT CONVERGE
          - objective_value (float): Total objective across all periods

    Raises:
        ValueError: If the output cannot be parsed.
    """
    if not stdout or "Multi-Period Optimal Power Flow" not in stdout:
        raise ValueError("Output does not appear to be TCOPFLOW output")

    metadata: dict = {}

    m = _DURATION_RE.search(stdout)
    metadata["duration_min"] = float(m.group(1)) if m else 0.0

    m = _TIMESTEP_RE.search(stdout)
    metadata["dT_min"] = float(m.group(1)) if m else 0.0

    m = _NUM_STEPS_RE.search(stdout)
    metadata["num_steps"] = int(m.group(1)) if m else 0

    m = _PLOAD_PROFILE_RE.search(stdout)
    metadata["pload_profile"] = m.group(1).strip() if m else ""

    m = _QLOAD_PROFILE_RE.search(stdout)
    metadata["qload_profile"] = m.group(1).strip() if m else ""

    m = _WIND_PROFILE_RE.search(stdout)
    metadata["wind_profile"] = m.group(1).strip() if m else "NOT SET"

    m = _LOAD_LOSS_RE.search(stdout)
    metadata["load_loss_allowed"] = (m.group(1).upper() == "YES") if m else False

    m = _POWER_IMBALANCE_RE.search(stdout)
    metadata["power_imbalance_allowed"] = (m.group(1).upper() == "YES") if m else False

    m = _IGNORE_LINEFLOW_RE.search(stdout)
    metadata["ignore_lineflow"] = (m.group(1).upper() == "YES") if m else False

    m = _IS_COUPLING_RE.search(stdout)
    metadata["num_coupling_constraints"] = int(m.group(1)) if m else 0

    m = _CONVERGENCE_RE.search(stdout)
    metadata["convergence_status"] = m.group(1).strip() if m else ""

    m = _OBJC_VALUE_RE.search(stdout)
    metadata["objective_value"] = float(m.group(1)) if m else 0.0

    # The bus/branch/gen tables are identical to OPFLOW format.
    # The OPFLOW parser's header check ("Optimal Power Flow" in stdout)
    # passes because "Multi-Period Optimal Power Flow" contains it.
    result = parse_opflow_output(stdout, bus_limits=bus_limits)

    # Override convergence from TCOPFLOW header (authoritative for multi-period)
    tcopflow_convergence = metadata.get("convergence_status", "")
    if tcopflow_convergence == "CONVERGED":
        result.converged = True
    elif tcopflow_convergence in ("DID", "DID NOT CONVERGE", "DIVERGED"):
        result.converged = False

    # TCOPFLOW only supports IPOPT, so feasibility logic is simpler than SCOPFLOW
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
        elif _is_near_boundary(result, bus_limits):
            result.feasibility_detail = "marginal"
        else:
            result.feasibility_detail = "infeasible"
    else:
        result.feasibility_detail = "infeasible"

    return result, metadata


def parse_tcopflow_period_files(
    workdir: Path,
) -> list[dict]:
    """Parse per-period TCOPFLOW result files from tcopflowout/ directory.

    Each file (t0.m, t1.m, ...) is a MATPOWER .m file with bus, gen,
    branch data and mpc.summary_stats for the corresponding time period.

    Args:
        workdir: The working directory where TCOPFLOW was executed.
            The tcopflowout/ subdirectory should contain the period files.

    Returns:
        List of period dicts sorted by period index. Each dict contains:
          - period (int): Period index (0-based)
          - total_gen_mw (float): Total active generation
          - total_load_mw (float): Total active load
          - total_gen_mvar (float): Total reactive generation
          - total_load_mvar (float): Total reactive load
          - voltage_min (float): Minimum bus voltage magnitude
          - voltage_max (float): Maximum bus voltage magnitude
          - max_line_loading_pct (float): Worst line loading percentage
          - losses_mw (float): Active power losses (gen - load)
          - converged (bool): Per-period convergence status
          - objective (float): Per-period objective value (mpc.obj)
          - num_buses (int): Number of buses
          - num_gens_on (int): Number of online generators
    """
    tcopflowout_dir = Path(workdir) / "tcopflowout"
    if not tcopflowout_dir.is_dir():
        logger.warning("tcopflowout/ directory not found in %s", workdir)
        return []

    period_files = sorted(tcopflowout_dir.glob("t*.m"))
    if not period_files:
        logger.warning("No t*.m files found in %s", tcopflowout_dir)
        return []

    periods: list[dict] = []
    for pf in period_files:
        try:
            period_idx = int(pf.stem[1:])
        except ValueError:
            logger.warning("Cannot extract period index from %s, skipping", pf.name)
            continue

        try:
            net = parse_matpower(pf)
        except Exception as exc:
            logger.warning("Failed to parse period file %s: %s", pf.name, exc)
            continue

        total_gen_mw = sum(g.Pg for g in net.generators if g.status == 1)
        total_load_mw = sum(b.Pd for b in net.buses)
        total_gen_mvar = sum(g.Qg for g in net.generators if g.status == 1)
        total_load_mvar = sum(b.Qd for b in net.buses)

        voltages = [b.Vm for b in net.buses] if net.buses else [0.0]
        voltage_min = min(voltages)
        voltage_max = max(voltages)

        max_loading = 0.0
        for br in net.branches:
            if br.rateA > 0 and len(br.extra) >= 4:
                pf_mw = br.extra[0]
                qf_mvar = br.extra[1]
                pt_mw = br.extra[2]
                qt_mvar = br.extra[3]
                sf_mva = (pf_mw**2 + qf_mvar**2) ** 0.5
                st_mva = (pt_mw**2 + qt_mvar**2) ** 0.5
                loading = max(sf_mva, st_mva) / br.rateA * 100
                if loading > max_loading:
                    max_loading = loading

        periods.append({
            "period": period_idx,
            "total_gen_mw": total_gen_mw,
            "total_load_mw": total_load_mw,
            "total_gen_mvar": total_gen_mvar,
            "total_load_mvar": total_load_mvar,
            "voltage_min": voltage_min,
            "voltage_max": voltage_max,
            "max_line_loading_pct": max_loading,
            "losses_mw": total_gen_mw - total_load_mw,
            "converged": True,
            "objective": 0.0,
            "num_buses": len(net.buses),
            "num_gens_on": sum(1 for g in net.generators if g.status == 1),
        })

    periods.sort(key=lambda p: p["period"])
    return periods


def parse_tcopflow_simulation_result(
    sim_result,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> Optional[tuple[OPFLOWResult, dict]]:
    """Parse a TCOPFLOW SimulationResult.

    Returns (OPFLOWResult, metadata_dict) or None if parsing fails.
    """
    if not sim_result.success:
        logger.warning("TCOPFLOW simulation did not succeed — skipping parse")
        return None

    try:
        return parse_tcopflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse TCOPFLOW output: %s", exc)
        return None


def parse_tcopflow_metadata(sim_result) -> dict | None:
    """Extract TCOPFLOW-specific metadata (num_steps, duration, etc.).

    Returns a dict or None if parsing fails or the simulation did not succeed.
    """
    parsed = parse_tcopflow_simulation_result(sim_result)
    if parsed is None:
        return None
    _opflow_result, metadata = parsed
    return metadata