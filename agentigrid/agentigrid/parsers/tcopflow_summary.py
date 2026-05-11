"""Compact results summary generator for TCOPFLOW (Multi-Period OPF) LLM prompts."""

from __future__ import annotations

from agentigrid.parsers.opflow_results import OPFLOWResult


def tcopflow_results_summary(
    result: OPFLOWResult,
    num_steps: int = 0,
    duration_min: float = 0.0,
    dT_min: float = 0.0,
    is_coupling: bool = True,
    period_data: list[dict] | None = None,
) -> str:
    """Generate a compact text summary of TCOPFLOW results for the LLM.

    This extends the OPFLOW summary with TCOPFLOW-specific context:
    - Header identifies it as multi-period results
    - Shows temporal parameters (duration, time-steps, coupling)
    - Aggregated metrics across all periods when period_data available
    - Per-period mini-table for temporal overview
    - Period-0 detailed results (voltage, gen, lines)

    Args:
        result: Parsed OPFLOWResult (from period-0 stdout data).
        num_steps: Number of time steps in the horizon.
        duration_min: Total duration in minutes.
        dT_min: Time-step size in minutes.
        is_coupling: Whether generator ramp constraints are active.
        period_data: Optional list of per-period dicts from
            parse_tcopflow_period_files().
    """
    lines: list[str] = []
    lines.append("=== TCOPFLOW Results (Multi-Period OPF) ===")
    lines.append(f"Status: {result.convergence_status}")
    if result.feasibility_detail:
        lines.append(f"Feasibility: {result.feasibility_detail}")
    if result.ipopt_exit_status:
        lines.append(f"Solver exit: {result.ipopt_exit_status}")
    lines.append(f"Solver: {result.solver}")
    lines.append(f"Objective value (total): ${result.objective_value:,.2f}")
    lines.append("")

    # Temporal parameters
    coupling_str = "enabled (ramp constraints)" if is_coupling else "disabled"
    lines.append(f"Time horizon: {num_steps} steps, {duration_min:.0f} min "
                 f"(dT = {dT_min:.0f} min), coupling {coupling_str}")
    lines.append("")

    # Aggregated multi-period metrics
    if period_data and len(period_data) > 0:
        all_vmin = min(p["voltage_min"] for p in period_data)
        all_vmax = max(p["voltage_max"] for p in period_data)
        worst_loading = max(p["max_line_loading_pct"] for p in period_data)
        peak_load = max(p["total_load_mw"] for p in period_data)
        min_load = min(p["total_load_mw"] for p in period_data)
        total_gen_range = (
            min(p["total_gen_mw"] for p in period_data),
            max(p["total_gen_mw"] for p in period_data),
        )
        any_negative_losses = any(
            p["losses_mw"] < 0 and p["total_load_mw"] > 0 for p in period_data
        )

        lines.append("Aggregated metrics (all periods):")
        lines.append(
            f"  Voltage: {all_vmin:.3f} – {all_vmax:.3f} pu "
            f"(across all periods)"
        )
        lines.append(
            f"  Load range: {min_load:.1f} – {peak_load:.1f} MW"
        )
        lines.append(
            f"  Generation range: {total_gen_range[0]:.1f} – "
            f"{total_gen_range[1]:.1f} MW"
        )
        lines.append(f"  Worst line loading: {worst_loading:.1f}%")
        if any_negative_losses:
            lines.append(
                "  ** UNPHYSICAL — negative losses in at least one period **"
            )
        lines.append("")

        # Per-period mini-table
        lines.append("Per-period summary:")
        lines.append(
            f"{'Period':>6} | {'Load(MW)':>9} | {'Gen(MW)':>9} | "
            f"{'Vmin(pu)':>8} | {'Vmax(pu)':>8} | {'MaxLoad%':>8}"
        )
        lines.append("-" * 62)
        for p in period_data:
            loss_flag = " *" if p["losses_mw"] < 0 and p["total_load_mw"] > 0 else ""
            lines.append(
                f"{p['period']:>6} | {p['total_load_mw']:>9.1f} | "
                f"{p['total_gen_mw']:>9.1f} | {p['voltage_min']:>8.3f} | "
                f"{p['voltage_max']:>8.3f} | {p['max_line_loading_pct']:>7.1f}%"
                f"{loss_flag}"
            )
        lines.append("  * = negative losses (generation < load)")
        lines.append("")

    # Period-0 voltage profile (from result)
    lines.append("Period-0 voltage profile:")
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

    # Generation vs load (period-0)
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

    # Top 5 most loaded lines (period-0)
    loaded = []
    for br in result.branches:
        if br.Slim > 0:
            pct = max(br.Sf, br.St) / br.Slim * 100
            loaded.append((pct, br))
    loaded.sort(key=lambda x: x[0], reverse=True)

    lines.append("Most loaded lines (period-0):")
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
    lines.append(f"Generators: {online} online, {offline} offline")
    lines.append(f"Violations: {result.num_violations}")

    if result.violation_details:
        for v in result.violation_details[:5]:
            lines.append(f"  - {v}")

    return "\n".join(lines)