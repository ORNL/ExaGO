"""Parser for PFLOW (AC Power Flow) text output."""

from __future__ import annotations

import logging
import re
from typing import Optional

from agentigrid.parsers.opflow_parser import _is_near_boundary
from agentigrid.parsers.opflow_results import (
    BranchResult,
    BusResult,
    GenResult,
    OPFLOWResult,
)

logger = logging.getLogger("agentigrid.parsers.pflow")

_PFLOW_ITERATIONS_RE = re.compile(r"Number of iterations\s+(\d+)", re.MULTILINE)
_PFLOW_SOLVE_TIME_RE = re.compile(r"Solve Time \(sec\)\s+([\d.]+)", re.MULTILINE)
_CONVERGENCE_RE = re.compile(r"^Convergence status\s+(.+)$", re.MULTILINE)
_SOLVER_RE = re.compile(r"^Solver\s+(.+)$", re.MULTILINE)

_BUS_HEADER_RE = re.compile(r"^Bus\s+Pd\s+Pd", re.MULTILINE)
_BRANCH_HEADER_RE = re.compile(r"^From\s+To\s+Status\s+Sft", re.MULTILINE)
_GEN_HEADER_RE = re.compile(r"^Gen\s+Status\s+Fuel", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^-{10,}", re.MULTILINE)

_VOLTAGE_PROXIMITY_PU = 0.01
_LINE_LOADING_PROXIMITY_PCT = 5.0


def _parse_table_rows(text: str, start_pos: int) -> list[list[str]]:
    rows: list[list[str]] = []
    sep = _SEPARATOR_RE.search(text, start_pos)
    if not sep:
        return rows

    data_start = sep.end()
    started = False

    for line in text[data_start:].split("\n"):
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped.startswith("---"):
            if started:
                break
            continue
        started = True
        tokens = stripped.split()
        if tokens:
            rows.append(tokens)

    return rows


def parse_pflow_output(
    stdout: str,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> tuple[OPFLOWResult, dict]:
    """Parse PFLOW text output.

    PFLOW prints results using the same bus/branch/gen table format as OPFLOW,
    but its header, solver, and convergence indicators differ:
    - Header: "AC Power Flow" (not "Optimal Power Flow")
    - Solver: Newton-Rhapson (not IPOPT)
    - No EXIT: line from IPOPT
    - No "Objective value" line (power flow has no cost optimization)
    - Convergence via "Convergence status" line (CONVERGED / DID NOT CONVERGE)

    Args:
        stdout: Complete stdout from a PFLOW run.
        bus_limits: Optional per-bus voltage limits for violation checking.

    Returns:
        Tuple of (OPFLOWResult, metadata dict).
        The metadata dict contains:
          - solver (str)
          - convergence_status (str)

    Raises:
        ValueError: If the output cannot be parsed.
    """
    if not stdout or "AC Power Flow" not in stdout:
        raise ValueError("Output does not appear to be PFLOW output")

    metadata: dict = {}

    m = _CONVERGENCE_RE.search(stdout)
    convergence_status = m.group(1).strip() if m else "UNKNOWN"
    metadata["convergence_status"] = convergence_status

    converged = convergence_status == "CONVERGED"

    m = _SOLVER_RE.search(stdout)
    solver = m.group(1).strip() if m else ""
    metadata["solver"] = solver

    m = _PFLOW_ITERATIONS_RE.search(stdout)
    num_iterations = int(m.group(1)) if m else 0

    m = _PFLOW_SOLVE_TIME_RE.search(stdout)
    solve_time = float(m.group(1)) if m else 0.0

    buses: list[BusResult] = []
    m = _BUS_HEADER_RE.search(stdout)
    if m:
        for tokens in _parse_table_rows(stdout, m.start()):
            if len(tokens) >= 11:
                try:
                    buses.append(BusResult(
                        bus_id=int(tokens[0]),
                        Pd=float(tokens[1]),
                        Pd_loss=float(tokens[2]),
                        Qd=float(tokens[3]),
                        Qd_loss=float(tokens[4]),
                        Vm=float(tokens[5]),
                        Va=float(tokens[6]),
                        mult_Pmis=float(tokens[7]),
                        mult_Qmis=float(tokens[8]),
                        Pslack=float(tokens[9]),
                        Qslack=float(tokens[10]),
                    ))
                except (ValueError, IndexError):
                    logger.warning("Skipping unparseable bus row: %s", tokens)

    branches: list[BranchResult] = []
    m = _BRANCH_HEADER_RE.search(stdout)
    if m:
        for tokens in _parse_table_rows(stdout, m.start()):
            if len(tokens) >= 8:
                try:
                    branches.append(BranchResult(
                        from_bus=int(tokens[0]),
                        to_bus=int(tokens[1]),
                        status=int(tokens[2]),
                        Sf=float(tokens[3]),
                        St=float(tokens[4]),
                        Slim=float(tokens[5]),
                        mult_Sf=float(tokens[6]),
                        mult_St=float(tokens[7]),
                    ))
                except (ValueError, IndexError):
                    logger.warning("Skipping unparseable branch row: %s", tokens)

    generators: list[GenResult] = []
    m = _GEN_HEADER_RE.search(stdout)
    if m:
        for tokens in _parse_table_rows(stdout, m.start()):
            if len(tokens) >= 9:
                try:
                    generators.append(GenResult(
                        bus=int(tokens[0]),
                        status=int(tokens[1]),
                        fuel=tokens[2],
                        Pg=float(tokens[3]),
                        Qg=float(tokens[4]),
                        Pmin=float(tokens[5]),
                        Pmax=float(tokens[6]),
                        Qmin=float(tokens[7]),
                        Qmax=float(tokens[8]),
                    ))
                except (ValueError, IndexError):
                    logger.warning("Skipping unparseable gen row: %s", tokens)

    total_gen_mw = sum(g.Pg for g in generators if g.status == 1)
    total_load_mw = sum(b.Pd for b in buses)
    total_gen_mvar = sum(g.Qg for g in generators if g.status == 1)
    total_load_mvar = sum(b.Qd for b in buses)

    voltages = [b.Vm for b in buses] if buses else [0.0]
    voltage_min = min(voltages)
    voltage_max = max(voltages)
    voltage_mean = sum(voltages) / len(voltages)

    max_line_loading_pct = 0.0
    for br in branches:
        if br.Slim > 0:
            loading = max(br.Sf, br.St) / br.Slim * 100
            if loading > max_line_loading_pct:
                max_line_loading_pct = loading

    _V_FALLBACK_MIN = 0.95
    _V_FALLBACK_MAX = 1.05
    violations: list[str] = []
    for b in buses:
        if bus_limits is not None and b.bus_id in bus_limits:
            vmin, vmax = bus_limits[b.bus_id]
        else:
            vmin, vmax = _V_FALLBACK_MIN, _V_FALLBACK_MAX
        if b.Vm < vmin:
            violations.append(f"Bus {b.bus_id}: Vm={b.Vm:.3f} pu < {vmin} (undervoltage)")
        if b.Vm > vmax:
            violations.append(f"Bus {b.bus_id}: Vm={b.Vm:.3f} pu > {vmax} (overvoltage)")
    for br in branches:
        if br.Slim > 0:
            if br.Sf > br.Slim:
                violations.append(f"Branch {br.from_bus}-{br.to_bus}: Sf={br.Sf:.2f} > Slim={br.Slim:.2f} MVA")
            if br.St > br.Slim:
                violations.append(f"Branch {br.from_bus}-{br.to_bus}: St={br.St:.2f} > Slim={br.Slim:.2f} MVA")

    losses_mw = total_gen_mw - total_load_mw
    power_balance_mismatch_pct = 0.0
    if total_load_mw > 0:
        power_balance_mismatch_pct = losses_mw / total_load_mw * 100

    if losses_mw < 0 and total_load_mw > 0:
        violations.append(
            f"Power balance violation: generation ({total_gen_mw:.2f} MW) < "
            f"load ({total_load_mw:.2f} MW), losses = {losses_mw:.2f} MW"
        )

    has_power_balance_violation = losses_mw < 0 and total_load_mw > 0

    if converged and not has_power_balance_violation:
        feasibility_detail = "feasible"
    elif has_power_balance_violation:
        feasibility_detail = "infeasible"
        converged = False
    elif not converged:
        if _is_near_boundary(
            buses, branches, bus_limits,
            max_line_loading_pct, losses_mw, total_load_mw,
        ):
            feasibility_detail = "marginal"
        else:
            feasibility_detail = "infeasible"
    else:
        feasibility_detail = "infeasible"

    result = OPFLOWResult(
        converged=converged,
        objective_value=0.0,
        convergence_status=convergence_status,
        solver=solver,
        model="AC",
        objective_type="PowerFlow",
        num_iterations=num_iterations,
        solve_time=solve_time,
        buses=buses,
        branches=branches,
        generators=generators,
        total_gen_mw=total_gen_mw,
        total_load_mw=total_load_mw,
        total_gen_mvar=total_gen_mvar,
        total_load_mvar=total_load_mvar,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        voltage_mean=voltage_mean,
        max_line_loading_pct=max_line_loading_pct,
        num_violations=len(violations),
        violation_details=violations,
        losses_mw=losses_mw,
        power_balance_mismatch_pct=power_balance_mismatch_pct,
        ipopt_exit_status="",
        feasibility_detail=feasibility_detail,
    )

    return result, metadata


def parse_pflow_simulation_result(
    sim_result,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> Optional[tuple[OPFLOWResult, dict]]:
    """Parse a PFLOW SimulationResult.

    Returns (OPFLOWResult, metadata_dict) or None if parsing fails.
    """
    if not sim_result.success:
        logger.warning("PFLOW simulation did not succeed — skipping parse")
        return None

    try:
        return parse_pflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse PFLOW output: %s", exc)
        return None