"""Compact results summary generator for DCOPFLOW (DC-OPF) LLM prompts."""

from __future__ import annotations

from agentigrid.parsers.opflow_results import OPFLOWResult


def dcopflow_results_summary(result: OPFLOWResult) -> str:
    """Generate a compact text summary of DCOPFLOW results for the LLM.

    This is the DC-OPF equivalent of results_summary(). Key differences:
    - Skips voltage magnitude analysis (Vm is meaningless in DC — always ~1.0)
    - Shows phase angle (Va) statistics instead
    - Shows active power only (no reactive power)
    - Line loading IS meaningful in DC
    - Violations are line overloads only (no voltage violations in DC)
    """
    lines: list[str] = []
    lines.append("=== DCOPFLOW Results (DC Approximation) ===")
    lines.append(f"Status: {result.convergence_status}")
    if result.feasibility_detail:
        lines.append(f"Feasibility: {result.feasibility_detail}")
    lines.append(f"Objective value (cost): ${result.objective_value:,.2f}")
    lines.append(f"Solver: {result.solver} ({result.num_iterations} iterations, {result.solve_time:.2f}s)")
    lines.append("Note: DC approximation — voltages fixed at 1.0 pu, no reactive power.")
    lines.append("")

    # Phase angle profile (replaces voltage profile)
    if result.buses:
        min_angle_bus = min(result.buses, key=lambda b: b.Va)
        max_angle_bus = max(result.buses, key=lambda b: b.Va)
        # Reference bus: closest Va to 0
        ref_bus = min(result.buses, key=lambda b: abs(b.Va))
        lines.append("Phase angle profile:")
        lines.append(
            f"  Min: {min_angle_bus.Va:.3f}° (bus {min_angle_bus.bus_id})  |  "
            f"Max: {max_angle_bus.Va:.3f}° (bus {max_angle_bus.bus_id})  |  "
            f"Ref: bus {ref_bus.bus_id} ({ref_bus.Va:.3f}°)"
        )
        lines.append("")

    # Generation vs load (active power only — no reactive in DC)
    losses = result.losses_mw
    gen_load_line = (
        f"Generation: {result.total_gen_mw:.2f} MW / "
        f"Load: {result.total_load_mw:.2f} MW / "
        f"Losses: {losses:.2f} MW"
    )
    if losses < 0 and result.total_load_mw > 0:
        gen_load_line += "  ** UNPHYSICAL — negative losses **"
    lines.append(gen_load_line)
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
        lines.append(
            f"  {i+1}. Bus {br.from_bus}->{br.to_bus}: {pct:.1f}% "
            f"({flow:.2f} / {br.Slim:.2f} MVA)"
        )
    lines.append("")

    # Generators
    online = sum(1 for g in result.generators if g.status == 1)
    offline = sum(1 for g in result.generators if g.status == 0)
    lines.append(f"Generators: {online} online, {offline} offline")

    # Line violations only (no voltage violations in DC)
    line_violations = [v for v in result.violation_details if "line" in v.lower() or "branch" in v.lower() or "loading" in v.lower()]
    lines.append(f"Line violations: {len(line_violations)}")
    for v in line_violations[:5]:
        lines.append(f"  - {v}")

    return "\n".join(lines)
