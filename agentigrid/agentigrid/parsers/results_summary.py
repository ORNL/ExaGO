"""Compact results summary generator for LLM prompts."""

from __future__ import annotations

from agentigrid.parsers.opflow_results import OPFLOWResult


def results_summary(result: OPFLOWResult) -> str:
    """Generate a compact text summary of OPFLOW results for the LLM.

    This is injected into the LLM prompt as "Section D — Latest Results".
    """
    lines: list[str] = []
    lines.append("=== OPFLOW Results ===")
    lines.append(f"Status: {result.convergence_status}")
    if result.feasibility_detail:
        lines.append(f"Feasibility: {result.feasibility_detail}")
    if result.ipopt_exit_status:
        lines.append(f"Solver exit: {result.ipopt_exit_status}")
    lines.append(f"Objective value (cost): ${result.objective_value:,.2f}")
    lines.append(f"Solver: {result.solver} ({result.num_iterations} iterations, {result.solve_time:.2f}s)")
    lines.append("")

    # Voltage profile
    lines.append("Voltage profile:")
    min_bus = min(result.buses, key=lambda b: b.Vm) if result.buses else None
    max_bus = max(result.buses, key=lambda b: b.Vm) if result.buses else None
    if min_bus and max_bus:
        lines.append(
            f"  Min: {result.voltage_min:.3f} pu (bus {min_bus.bus_id})  |  "
            f"Max: {result.voltage_max:.3f} pu (bus {max_bus.bus_id})  |  "
            f"Mean: {result.voltage_mean:.3f} pu"
        )
    # Buses near limits (within 0.005 of 0.9 or 1.1)
    near_limit = [
        b for b in result.buses
        if b.Vm <= 0.905 or b.Vm >= 1.095
    ]
    if near_limit:
        for b in near_limit[:5]:
            lines.append(f"  Bus {b.bus_id} at {b.Vm:.3f} pu")
    lines.append("")

    # Generation vs load
    losses = result.losses_mw
    gen_load_line = (
        f"Generation: {result.total_gen_mw:.2f} MW / "
        f"Load: {result.total_load_mw:.2f} MW / "
        f"Losses: {losses:.2f} MW"
    )
    if losses < 0 and result.total_load_mw > 0:
        gen_load_line += "  ** UNPHYSICAL — negative losses **"
    lines.append(gen_load_line)
    lines.append(f"Reactive: Gen {result.total_gen_mvar:.2f} MVAr / Load {result.total_load_mvar:.2f} MVAr")
    lines.append("")

    # Top 5 most loaded lines
    loaded = []
    for br in result.branches:
        if br.Slim > 0:
            pct = max(br.Sf, br.St) / br.Slim * 100
            loaded.append((pct, br))
    loaded.sort(key=lambda x: x[0], reverse=True)

    lines.append("Most loaded lines:")
    for i, (pct, br) in enumerate(loaded[:5]):
        flow = max(br.Sf, br.St)
        lines.append(f"  {i+1}. Bus {br.from_bus}->{br.to_bus}: {pct:.1f}% ({flow:.2f} / {br.Slim:.2f} MVA)")
    lines.append("")

    # Generators
    online = sum(1 for g in result.generators if g.status == 1)
    offline = sum(1 for g in result.generators if g.status == 0)
    lines.append(f"Generators: {online} online, {offline} offline")
    lines.append(f"Violations: {result.num_violations}")

    if result.violation_details:
        for v in result.violation_details[:5]:
            lines.append(f"  - {v}")

    return "\n".join(lines)
