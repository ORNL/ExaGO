"""Parser for OPFLOW text output."""

from __future__ import annotations

import logging
import re
from typing import Optional

from agentigrid.parsers.opflow_results import (
    BranchResult,
    BusResult,
    GenResult,
    OPFLOWResult,
)

logger = logging.getLogger("agentigrid.parsers.opflow")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ITERATIONS_RE = re.compile(r"Number of Iterations\.*:\s*(\d+)")
_SOLVE_TIME_RE = re.compile(r"Total seconds in IPOPT\s*=\s*([\d.]+)")
_EXIT_RE = re.compile(r"EXIT:\s*(.+)")
_MODEL_RE = re.compile(r"^Model\s+(\S+)", re.MULTILINE)
_SOLVER_RE = re.compile(r"^Solver\s+(\S+)", re.MULTILINE)
_OBJECTIVE_TYPE_RE = re.compile(r"^Objective\s+(\S+)", re.MULTILINE)
_CONVERGENCE_RE = re.compile(r"^Convergence status\s+(.+)$", re.MULTILINE)
_OBJ_VALUE_RE = re.compile(r"^Objective value\s+(?:\(base\)\s+)?([\d.eE+-]+)", re.MULTILINE)

_BUS_HEADER_RE = re.compile(r"^Bus\s+Pd\s+Pd", re.MULTILINE)
_BRANCH_HEADER_RE = re.compile(r"^From\s+To\s+Status\s+Sft", re.MULTILINE)
_GEN_HEADER_RE = re.compile(r"^Gen\s+Status\s+Fuel", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^-{10,}", re.MULTILINE)


_VOLTAGE_PROXIMITY_PU = 0.01
_LINE_LOADING_PROXIMITY_PCT = 5.0


def _is_near_boundary(
    buses: list[BusResult],
    branches: list[BranchResult],
    bus_limits: dict[int, tuple[float, float]] | None,
    max_line_loading_pct: float,
    losses_mw: float,
    total_load_mw: float,
) -> bool:
    """Check if a non-converged solution's metrics are near their limits."""
    if losses_mw < 0 and total_load_mw > 0:
        return False

    if bus_limits:
        for b in buses:
            if b.bus_id in bus_limits:
                vmin, vmax = bus_limits[b.bus_id]
                if (abs(b.Vm - vmin) <= _VOLTAGE_PROXIMITY_PU
                        or abs(b.Vm - vmax) <= _VOLTAGE_PROXIMITY_PU):
                    return True

    if max_line_loading_pct >= (100.0 - _LINE_LOADING_PROXIMITY_PCT):
        return True

    return False


def _parse_table_rows(text: str, start_pos: int) -> list[list[str]]:
    """Extract data rows between separator lines starting at *start_pos*.

    Finds the next separator line after start_pos, reads rows until
    the next separator or blank line, and returns tokenised rows.
    """
    # Find the separator line after the header
    sep = _SEPARATOR_RE.search(text, start_pos)
    if not sep:
        return []

    data_start = sep.end()
    rows: list[list[str]] = []
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


def _is_marginal_exit(exit_status: str) -> bool:
    """Check if an IPOPT exit status indicates a marginal solution.

    Marginal means the solver did not fully converge but the solution data may
    still be usable — e.g., maximum iterations exceeded or search direction too
    small. In contrast, "Infeasible Problem Detected" or "Diverging Iterates"
    clearly indicate infeasibility.
    """
    marginal_indicators = [
        "Maximum Number of Iterations Exceeded",
        "Maximum Iterations Exceeded",
        "Search Direction Becomes Too Small",
        "Solved To Acceptable Level",
    ]
    for indicator in marginal_indicators:
        if indicator in exit_status:
            return True
    return False


def parse_opflow_output(
    stdout: str,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> OPFLOWResult:
    """Parse OPFLOW text output into structured results.

    Args:
        stdout: The complete stdout text from an OPFLOW run.
        bus_limits: Optional mapping of ``bus_id -> (Vmin, Vmax)`` extracted
            from the input MATPOWER file.  When provided, violations are
            reported against these per-bus limits instead of the hardcoded
            0.9/1.1 fallback.

    Returns:
        OPFLOWResult with all sections populated.

    Raises:
        ValueError: If the output is not recognised as OPFLOW output.
    """
    if not stdout or "Optimal Power Flow" not in stdout:
        raise ValueError("Output does not appear to be OPFLOW output")

    # --- Section 1: Ipopt solver log ---
    m = _ITERATIONS_RE.search(stdout)
    num_iterations = int(m.group(1)) if m else 0

    m = _SOLVE_TIME_RE.search(stdout)
    solve_time = float(m.group(1)) if m else 0.0

    m = _EXIT_RE.search(stdout)
    ipopt_exit_status = m.group(1).strip() if m else ""
    converged = "Optimal Solution Found" in ipopt_exit_status or "Solved To Acceptable Level" in ipopt_exit_status

    # --- Section 2: OPFLOW summary ---
    m = _MODEL_RE.search(stdout)
    model = m.group(1) if m else ""

    m = _SOLVER_RE.search(stdout)
    solver = m.group(1) if m else ""

    m = _OBJECTIVE_TYPE_RE.search(stdout)
    objective_type = m.group(1) if m else ""

    m = _CONVERGENCE_RE.search(stdout)
    convergence_status = m.group(1).strip() if m else ("CONVERGED" if converged else "UNKNOWN")

    m = _OBJ_VALUE_RE.search(stdout)
    objective_value = float(m.group(1)) if m else 0.0

    # --- Section 3: Bus data ---
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

    # --- Section 4: Branch data ---
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

    # --- Section 5: Generator data ---
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

    # --- Derived metrics ---
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

    # Violations
    _V_FALLBACK_MIN = 0.9
    _V_FALLBACK_MAX = 1.1
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

    # Power balance check: generation must cover load + losses
    losses_mw = total_gen_mw - total_load_mw
    power_balance_mismatch_pct = 0.0
    if total_load_mw > 0:
        power_balance_mismatch_pct = losses_mw / total_load_mw * 100

    if losses_mw < 0 and total_load_mw > 0:
        violations.append(
            f"Power balance violation: generation ({total_gen_mw:.2f} MW) < "
            f"load ({total_load_mw:.2f} MW), losses = {losses_mw:.2f} MW"
        )

    # Determine feasibility_detail based on convergence, violations, and IPOPT status
    has_power_balance_violation = losses_mw < 0 and total_load_mw > 0

    if converged and not has_power_balance_violation:
        feasibility_detail = "feasible"
    elif has_power_balance_violation:
        feasibility_detail = "infeasible"
    elif converged:
        feasibility_detail = "infeasible"
    elif _is_marginal_exit(ipopt_exit_status):
        feasibility_detail = "marginal"
    elif _is_near_boundary(
        buses, branches, bus_limits,
        max_line_loading_pct, losses_mw, total_load_mw,
    ):
        feasibility_detail = "marginal"
    else:
        feasibility_detail = "infeasible"

    return OPFLOWResult(
        converged=converged,
        objective_value=objective_value,
        convergence_status=convergence_status,
        solver=solver,
        model=model,
        objective_type=objective_type,
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
        ipopt_exit_status=ipopt_exit_status,
        feasibility_detail=feasibility_detail,
    )


def parse_simulation_result(
    sim_result,
    bus_limits: dict[int, tuple[float, float]] | None = None,
) -> Optional[OPFLOWResult]:
    """Parse an OPFLOW SimulationResult into structured results.

    Args:
        sim_result: A SimulationResult from SimulationExecutor.run().
        bus_limits: Optional mapping of ``bus_id -> (Vmin, Vmax)`` used for
            violation checking.  Passed through to parse_opflow_output().

    Returns:
        OPFLOWResult, or None if the simulation failed or output can't be parsed.
    """
    if not sim_result.success:
        logger.warning("Simulation did not succeed — skipping parse")
        return None

    try:
        return parse_opflow_output(sim_result.stdout, bus_limits=bus_limits)
    except ValueError as exc:
        logger.warning("Failed to parse OPFLOW output: %s", exc)
        return None
