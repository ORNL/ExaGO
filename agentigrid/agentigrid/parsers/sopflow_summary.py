"""Compact results summary for SOPFLOW (Stochastic OPF)."""

from __future__ import annotations

from agentigrid.parsers.opflow_results import OPFLOWResult


def sopflow_results_summary(result: OPFLOWResult, num_scenarios: int = 0) -> str:
    """Generate a compact text summary of SOPFLOW results for the LLM.

    This extends the OPFLOW summary with SOPFLOW-specific context:
    - Header identifies it as stochastic results
    - Notes number of scenarios
    - Explains that this is the expected-cost base-case dispatch
    - All other metrics (voltage, generation, line loading) same as OPFLOW
    """
    lines: list[str] = []
    lines.append("=== SOPFLOW Results (Stochastic OPF) ===")
    lines.append(f"Status: {result.convergence_status}")
    if result.feasibility_detail:
        lines.append(f"Feasibility: {result.feasibility_detail}")
    if result.ipopt_exit_status:
        lines.append(f"Solver exit: {result.ipopt_exit_status}")
    lines.append(f"Objective value (base case cost): ${result.objective_value:,.2f}")
    lines.append(f"Solver: {result.solver}")
    lines.append(f"Wind scenarios: {num_scenarios}")
    lines.append(
        "Note: This is the first-stage (here-and-now) dispatch that must "
    )
    lines.append(
        "satisfy constraints across all wind generation scenarios simultaneously."
    )
    if num_scenarios > 0:
        lines.append(
            "Per-scenario voltage/loading data is not available in SOPFLOW output."
        )
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
    near_limit = [b for b in result.buses if b.Vm <= 0.905 or b.Vm >= 1.095]
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
    lines.append(
        f"Reactive: Gen {result.total_gen_mvar:.2f} MVAr / "
        f"Load {result.total_load_mvar:.2f} MVAr"
    )
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
            f"  {i + 1}. Bus {br.from_bus}->{br.to_bus}: "
            f"{pct:.1f}% ({flow:.2f} / {br.Slim:.2f} MVA)"
        )
    lines.append("")

    # Generators
    online = sum(1 for g in result.generators if g.status == 1)
    offline = sum(1 for g in result.generators if g.status == 0)
    wind_gens = [g for g in result.generators if g.status == 1 and "wind" in g.fuel.lower()]
    lines.append(f"Generators: {online} online, {offline} offline")
    if wind_gens:
        wind_pg = sum(g.Pg for g in wind_gens)
        wind_pmax = sum(g.Pmax for g in wind_gens)
        wind_util = wind_pg / wind_pmax * 100 if wind_pmax > 0 else 0
        lines.append(f"  Wind generators: {len(wind_gens)} online, total {wind_pg:.2f} MW")
        lines.append(
            f"  Wind capacity utilization: {wind_pg:.2f} / {wind_pmax:.2f} MW "
            f"({wind_util:.0f}%)"
        )
        if wind_util >= 99.5:
            lines.append(
                "  WARNING: Wind generators are at maximum capacity in the base-case "
                "dispatch. scale_wind_scenario alone will NOT change the first-stage "
                "dispatch. Increase wind Pmax (set_gen_dispatch) or increase system "
                "stress (scale_all_loads) to see feasibility changes."
            )
    lines.append(f"Violations: {result.num_violations}")

    if result.violation_details:
        for v in result.violation_details[:5]:
            lines.append(f"  - {v}")

    return "\n".join(lines)