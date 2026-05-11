"""System prompt template for the LLM agent."""

from __future__ import annotations


def format_benchmark_for_prompt(benchmark_result: dict | None) -> str:
    """Format a BenchmarkResult dict into a compact prompt section (<25 lines).

    Args:
        benchmark_result: Dict from BenchmarkResult (serialized via _benchmark_to_dict),
            or None.

    Returns:
        Formatted benchmark string for injection into the system prompt.
    """
    if benchmark_result is None or not benchmark_result.get("opflow_converged"):
        return "Benchmark unavailable."

    lines = []
    opflow_cost = benchmark_result.get("opflow_objective")
    pflow_cost = benchmark_result.get("pflow_best_computed_cost")
    gap_pct = benchmark_result.get("cost_gap_pct")
    gap_abs = benchmark_result.get("cost_gap_abs")

    if opflow_cost is not None:
        lines.append(f"OPFLOW optimal cost:    ${opflow_cost:,.2f}")
    if pflow_cost is not None:
        lines.append(f"PFLOW baseline cost:    ${pflow_cost:,.2f}")
    if gap_pct is not None and gap_abs is not None:
        sign = "+" if gap_abs >= 0 else ""
        lines.append(f"Cost gap:               {sign}{gap_pct:.2f}%  ({sign}${gap_abs:,.2f})")

    loadability = benchmark_result.get("loadability")
    if loadability:
        opflow_lf = loadability.get("opflow_max_factor")
        pflow_lf = loadability.get("pflow_max_factor")
        lf_gap = loadability.get("gap_pct")
        if opflow_lf is not None:
            lines.append(f"OPFLOW max load factor: {opflow_lf:.4f}×")
        if pflow_lf is not None:
            lines.append(f"PFLOW max load factor:  {pflow_lf:.4f}×")
        if lf_gap is not None:
            lines.append(f"Loadability gap:        {lf_gap:+.1f}%")

    dispatch = benchmark_result.get("dispatch_comparison", [])
    if dispatch:
        top5 = sorted(dispatch, key=lambda d: abs(d.get("delta", 0)), reverse=True)[:5]
        max_delta = abs(top5[0].get("delta", 0)) if top5 else 0
        if max_delta < 0.01:
            lines.append("PFLOW and OPFLOW dispatch are identical.")
        else:
            lines.append("Top dispatch deviations (|delta| desc):")
            lines.append(f"  {'Bus':<6} {'Fuel':<8} {'OPFLOW MW':>10} {'PFLOW MW':>10} {'Delta MW':>10}")
            for dc in top5:
                lines.append(
                    f"  {dc['bus']:<6} {dc.get('fuel', '?'):<8} "
                    f"{dc.get('opflow_pg', 0):>10.1f} {dc.get('pflow_pg', 0):>10.1f} "
                    f"{dc.get('delta', 0):>+10.1f}"
                )

    # Contextual notes
    notes = []
    if gap_pct is not None and abs(gap_pct) < 0.1:
        notes.append(
            "Cost gap is negligible (< 0.1%). Cost reduction by redispatch is not "
            "meaningful for this network at this load level."
        )
    if loadability and loadability.get("gap_pct") is not None and loadability["gap_pct"] < -5:
        lf_gap_val = loadability["gap_pct"]
        notes.append(
            f"PFLOW loadability is significantly below OPFLOW ({lf_gap_val:.1f}%). "
            "Improving loadability requires generator commitment or voltage setpoint "
            "changes, not simple redispatch."
        )
    if dispatch:
        top1_delta = top5[0].get("delta", 0) if top5 else 0
        if abs(top1_delta) > 100:
            top1_bus = top5[0].get("bus", "?") if top5 else "?"
            notes.append(
                f"Generator at bus {top1_bus} is over-dispatched by "
                f"{abs(top1_delta):.0f} MW versus OPFLOW optimum. "
                "Consider redistributing this generation."
            )
    for note in notes:
        lines.append(f"Note: {note}")

    return "\n".join(lines)


def build_system_prompt(
    command_schema: str,
    network_summary: str,
    application: str = "opflow",
    search_mode: str = "standard",
    concurrent_pflow: bool = False,
    network_metadata: str | None = None,
    benchmark_text: str | None = None,
    session_load_factor: float | None = None,
) -> str:
    """Build the system prompt for the LLM agent.

    Args:
        command_schema: Output of command_schema_text().
        network_summary: Output of network_summary().
        application: ExaGO application name.
        search_mode: "standard" or "stress_test".
        concurrent_pflow: Enable explore/select actions for PFLOW.
        network_metadata: Optional output of network_metadata() with the
            static structural facts for this network (slack bus, must-run
            generators, cost-curve diversity). Inserted as Section G when
            provided.
        benchmark_text: Optional output of format_benchmark_for_prompt().
            Inserted as Section H (after network metadata) when provided.
        session_load_factor: When set, adds a note to the PFLOW section
            telling the LLM the load factor is auto-injected and it should
            not include scale_all_loads in its commands.

    Returns:
        Complete system prompt string.
    """
    if search_mode == "stress_test":
        return _build_stress_test_prompt(
            command_schema, network_summary, application, network_metadata,
        )
    return _build_standard_prompt(
        command_schema, network_summary, application, concurrent_pflow,
        network_metadata, benchmark_text, session_load_factor,
    )


_DC_OPF_SECTION = (
    "=== DC OPF Characteristics ===\n\n"
    "DCOPFLOW uses the DC power flow approximation:\n"
    "- All bus voltages are fixed at 1.0 pu \u2014 voltage magnitude is NOT an optimization variable.\n"
    "- Reactive power (Q) is ignored \u2014 only active power (P) is optimized.\n"
    "- Line flows are computed using the B-matrix (susceptance) and phase angles only.\n"
    "- The DC approximation is faster but less accurate than full AC OPF (OPFLOW).\n"
    "- Voltage-related commands (set_gen_voltage, set_bus_vlimits, set_all_bus_vlimits) have NO "
    "effect in DCOPFLOW. Do NOT use them.\n"
    "- Focus on: generator active power dispatch (Pg), load scaling, branch status, and cost curves.\n"
    "- DCOPFLOW is best used for fast screening, contingency ranking, and active power market analysis."
)

_AC_OPF_VOLTAGE_SECTION = (
    "=== OPF Voltage Control ===\n\n"
    "In OPFLOW (Optimal Power Flow), bus voltages are **optimization variables** \u2014 "
    "the solver picks the voltage at each bus to minimise cost within the bounds set "
    "by bus Vmin/Vmax. This means:\n"
    '- set_gen_voltage sets only an initial guess; OPFLOW will ignore it and solve '
    "for the optimal voltage.\n"
    '- To enforce voltage limits across the entire network, use set_all_bus_vlimits '
    '(command 11): {"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05}\n'
    '- To enforce voltage limits on a specific bus only, use set_bus_vlimits '
    '(command 10): {"action": "set_bus_vlimits", "bus": 10, "Vmin": 0.98, "Vmax": 1.02}\n'
    "- Use scale_all_loads / set_gen_dispatch to shift the operating point when "
    "limits alone are insufficient.\n\n"
    "=== Feasibility Classification ===\n\n"
    "Each iteration is classified as one of:\n"
    "- feasible: Simulation converged with no constraint violations. "
    "The solution is physically valid.\n"
    "- infeasible: Either the solver did not converge, or the solution has "
    "generation < load (negative losses). This is not a physically valid dispatch.\n"
    "- marginal: Solver did not fully converge but no violations were detected "
    "in the solution data. Use with caution \u2014 may serve as a boundary marker."
)

_SCOPFLOW_SECTION = (
    "=== Security-Constrained OPF Characteristics ===\n\n"
    "SCOPFLOW finds a preventive dispatch that satisfies the base case OPF "
    "constraints AND survives all contingencies in the contingency file simultaneously.\n"
    "- The results you see are the BASE CASE operating point \u2014 the dispatch that "
    "the system must use to be secure against all listed outages.\n"
    "- The cost is typically HIGHER than unconstrained OPFLOW because the dispatch "
    "must leave enough margin to handle any single contingency.\n"
    "- If SCOPFLOW is infeasible, it means NO dispatch exists that can survive all "
    "contingencies at the current network configuration.\n"
    "- Do NOT use set_branch_status to simulate contingencies \u2014 the contingency file "
    "already defines them. Disabling a branch in the base case permanently removes it "
    "from the topology (different from a contingency).\n"
    "- Useful modifications: load scaling, generator dispatch/status, voltage limits, "
    "cost curves \u2014 these change the operating point that SCOPFLOW must secure.\n"
    "- The 'security premium' is the cost difference between SCOPFLOW and OPFLOW \u2014 "
    "tracking this helps quantify the cost of reliability.\n\n"
    "=== SCOPFLOW Solver and Feasibility ===\n\n"
    "Two SCOPFLOW solvers are available:\n"
    "- IPOPT (single core): Solves the full SCOPFLOW problem monolithically. "
    "Reports DID NOT CONVERGE when no N-1-secure dispatch exists. Results are RELIABLE.\n"
    "- EMPAR (multi-core): Solves each contingency independently. ALWAYS reports CONVERGED "
    "regardless of whether individual contingencies actually converged. EMPAR does NOT "
    "properly enforce N-1 security \u2014 it only checks base-case feasibility. "
    "Results with EMPAR reflect base-case loadability only, NOT N-1-secure loadability. "
    "The loadability limit will appear significantly higher with EMPAR than IPOPT.\n\n"
    "When using EMPAR, results marked 'feasible' with CONVERGED status may still be "
    "N-1-INSECURE. For accurate N-1 security analysis, use IPOPT.\n\n"
    "=== Feasibility Classification ===\n\n"
    "Each iteration is classified as one of:\n"
    "- feasible: Simulation converged with no constraint violations. "
    "The solution is physically valid and can be used for decision-making.\n"
    "- infeasible: Either the solver did not converge, or the solution has "
    "generation < load (negative losses) meaning the dispatch cannot serve the demand. "
    "This iteration should be treated as a boundary marker, not a valid solution.\n"
    "- marginal: The solver did not fully converge (e.g., maximum iterations exceeded) "
    "but the solution data shows no constraint violations. The solution MAY be usable "
    "but should be treated with caution. It can serve as a boundary marker in "
    "feasibility searches."
)

_TCOPFLOW_SECTION = (
    "=== Multi-Period OPF Characteristics ===\n\n"
    "TCOPFLOW solves a multi-period AC optimal power flow problem over a time horizon:\n"
    "- The objective is to minimise TOTAL cost across ALL time periods.\n"
    "- Generator ramp constraints couple successive time periods: the change in "
    "generator output between adjacent periods is bounded by ramp limits.\n"
    "- TCOPFLOW reads per-bus per-period load values from CSV profile files "
    "(P and Q), NOT from the .m case file. The loads in the .m file are only "
    "used as period-0 initial values when no profile is provided.\n"
    "- Standard load commands (scale_all_loads, set_load, scale_load) modify "
    "the .m file but TCOPFLOW OVERRIDES these with profile data. They have "
    "LIMITED effect on TCOPFLOW results.\n"
    "- To actually change the demand level across all periods, use "
    "scale_load_profile (command 12): "
    '{"action": "scale_load_profile", "factor": 1.1}\n'
    "  This multiplies ALL values in both P and Q profile CSVs by the factor.\n"
    "- Modifications to the network topology (set_gen_status, set_branch_status, "
    "set_gen_dispatch, set_bus_vlimits, set_all_bus_vlimits, set_branch_rate, "
    "set_cost_coeffs) apply across ALL periods \u2014 they change the base network.\n\n"
    "=== TCOPFLOW Results Interpretation ===\n\n"
    "- The results summary shows AGGREGATED metrics across all periods:\n"
    "  - Worst voltage (min/max) across all periods\n"
    "  - Load and generation ranges across the time horizon\n"
    "  - Worst line loading across all periods\n"
    "  - A per-period table showing load, generation, voltage range, and "
    "max loading for each time step\n"
    "- The worst-case period (lowest voltage, highest loading) determines "
    "overall feasibility.\n"
    "- Period-0 details are shown for bus-level and branch-level inspection.\n"
    "- Use the 'analyze' action with keywords like 'period', 'timestep', or "
    "'temporal' to inspect per-period data.\n\n"
    "=== TCOPFLOW Solver ===\n\n"
"- TCOPFLOW only supports the IPOPT solver.\n"
"- Feasibility classification:\n"
"  - feasible: Converged with no violations\n"
"  - infeasible: Did not converge and metrics are far from limits, or generation < load\n"
"  - marginal: Did not fully converge BUT metrics are near their limits (e.g., voltage "
"within 0.01 pu of a bound, line loading within 5% of 100%). This indicates the "
"operating point is at or near the feasibility boundary — treat as a boundary marker.\n"
"- When a binary search produces consecutive 'marginal' or 'feasible/infeasible' "
    "oscillations with a gap < 1%, declare 'complete' — you have found the boundary."
)

_SOPFLOW_SECTION = (
    "=== Stochastic OPF Characteristics ===\n\n"
    "SOPFLOW (Stochastic Optimal Power Flow) solves a two-stage optimization:\n"
    "- **First stage (here-and-now):** Finds a base-case dispatch that must satisfy "
    "network constraints across ALL wind generation scenarios simultaneously.\n"
    "- **Second stage (wait-and-see):** For each wind scenario, the solver adjusts "
    "generation to accommodate the specific wind realization while respecting "
    "system limits.\n"
    "- The objective value is the expected cost (weighted across scenarios) or the "
    "base-case cost depending on the formulation.\n"
    "- If SOPFLOW is infeasible, it means NO dispatch exists that can handle all "
    "wind scenarios within network constraints at the current operating point.\n\n"
    "=== Wind Scenario File ===\n\n"
    "SOPFLOW reads wind generation scenarios from a CSV file (via -windgen flag), "
    "NOT from the .m case file.\n"
    "- Each scenario specifies a different possible wind generation level for "
    "each wind generator in the network.\n"
    "- Standard load commands (scale_all_loads, set_load) modify the .m file but "
    "do NOT change the wind scenario data.\n"
    "- To change the wind generation level across all scenarios, use "
    "scale_wind_scenario (command 13): "
    '{"action": "scale_wind_scenario", "factor": 1.5}\n'
    "  This multiplies all wind generation values in the scenario CSV by the "
    "factor. Factor > 1.0 increases wind penetration; factor < 1.0 reduces it.\n"
    "- The scenario file has two possible formats:\n"
    "  - Single-period: scenario_nr, <bus>_Wind_<id>..., weight\n"
    "  - Multi-period: sim_timestamp, scenario_nr, <bus>_Wind_<id>...\n"
    "- Non-numeric columns (scenario_nr, timestamp, weight) are preserved when "
    "scaling.\n\n"
    "=== Useful Modifications for SOPFLOW ===\n\n"
    "- scale_wind_scenario: Change wind penetration level (primary lever for "
    "SOPFLOW analysis).\n"
    "- set_all_bus_vlimits / set_bus_vlimits: Tighten or relax voltage constraints.\n"
    "- scale_all_loads / scale_load: Change demand level in the .m file (affects "
    "base-case constraints but not wind scenario data).\n"
    "- set_gen_status / set_gen_dispatch: Change generator commitment/dispatch.\n"
    "- set_branch_status / set_branch_rate: Modify network topology/capacity.\n"
    "- set_cost_coeffs: Change generation cost curves.\n\n"
    "=== Wind Generator Capacity Constraint ===\n\n"
    "Wind generators in the .m file have a maximum output (Pmax). In the "
    "first-stage dispatch, wind generators are often dispatched AT their Pmax. "
    "When this happens:\n"
    "- scale_wind_scenario with factor > 1.0 may NOT change the first-stage "
    "dispatch, because the wind generator is already at its capacity limit.\n"
    "- To increase wind's effective contribution, also increase the wind "
    "generator's Pmax using set_gen_dispatch (e.g., set Pmax to a higher value).\n"
    "- To test system robustness more effectively, combine scale_wind_scenario "
    "with scale_all_loads to increase demand while increasing wind.\n"
    "- Check the results summary for 'Pg/Pmax' utilization. If wind generators "
    "are at 100% capacity, scaling wind scenarios alone will not move the dispatch.\n\n"
    "=== SOPFLOW Results Interpretation ===\n\n"
    "- The results show the FIRST-STAGE (base-case) dispatch — the operating "
    "point that the system must commit to BEFORE knowing which wind scenario "
    "will materialise.\n"
    "- Wind generator output shown in results is the first-stage dispatch value, "
    "not the scenario values.\n"
    "- If the base case is feasible, all wind scenarios can be handled through "
    "second-stage adjustments.\n"
    "- If the base case is infeasible, no single dispatch can accommodate all "
    "scenarios — you may need to reduce wind penetration (scale_wind_scenario "
    "with factor < 1.0) or relax constraints.\n\n"
    "=== SOPFLOW Analysis Limitations ===\n\n"
    "SOPFLOW output only contains the first-stage (base-case) dispatch. "
    "Per-scenario voltage, loading, or generation data is NOT available through "
    "the `analyze` command — the solver only prints the base-case solution.\n"
    "Do NOT attempt to query per-scenario voltage or loading profiles.\n"
    "To explore how different wind levels affect feasibility:\n"
    "- Use `scale_wind_scenario` (factor > 1.0) to increase wind penetration "
    "and observe whether the base case becomes infeasible.\n"
    "- Use `scale_all_loads` combined with `scale_wind_scenario` to push the "
    "system toward its feasibility boundary.\n"
    "- The `analyze` command can report base-case voltage, line loading, and "
    "generator data (same as OPFLOW), but NOT per-scenario breakdowns.\n\n"
    "=== SOPFLOW Solver and Feasibility ===\n\n"
    "- SOPFLOW supports two solvers:\n"
    "  - IPOPT (single-core): Solves the full stochastic problem. Reports "
    "DID NOT CONVERGE when no dispatch satisfies all scenarios.\n"
    "  - EMPAR (multi-core): Decomposes the problem by scenario. Faster for "
    "large systems but may miss coupling constraints between scenarios.\n"
    "- Feasibility classification:\n"
    "  - feasible: Converged with no violations across all scenarios.\n"
    "  - infeasible: Did not converge and metrics are far from limits, or "
    "generation < load.\n"
    "  - marginal: Did not fully converge BUT metrics are near their limits "
    "(e.g., voltage within 0.01 pu of a bound, line loading within 5% of 100%). "
    "This indicates the operating point is at or near the feasibility boundary — "
    "treat as a boundary marker.\n"
    "- When a binary search produces consecutive 'marginal' or "
    "'feasible/infeasible' oscillations with a gap < 1%, declare 'complete' — "
    "you have found the boundary."
)

_PFLOW_SECTION_CORE = (
    "=== Power Flow (PFLOW) — Analysis, Not Optimization ===\n\n"
    "PFLOW solves the nonlinear power flow equations for a given network state. "
    "It does NOT optimise anything. There is no objective function, no cost minimisation, "
    "and no re-dispatch. The LLM is the optimiser: YOU control the search.\n\n"
    "=== CRITICAL: Voltage Limits Must Be Set Explicitly ===\n\n"
    "PFLOW does NOT enforce bus voltage limits — it only solves the power flow equations. "
    "Violation checking uses the Vmin/Vmax values in the network file. If the default limits "
    "are wide (e.g., 0.9/1.1), voltages outside the typical engineering range of 0.95–1.05 pu "
    "will NOT be flagged as violations.\n\n"
    "**IMPORTANT: As your FIRST action in any PFLOW search, issue "
    "set_all_bus_vlimits (command 11) to set the voltage limits you want to enforce.** "
    "For typical feasibility searches, use:\n"
    '{"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05}\n\n'
    "This ensures that voltages outside 0.95–1.05 pu are correctly reported as violations. "
    "Without this command, the search may treat infeasible operating points as feasible, "
    "leading to incorrect boundary estimates.\n\n"
    "=== Key Differences from OPFLOW ===\n\n"
    "- There is NO objective value. The 'computed generation cost' shown in results is "
    "calculated from your dispatch multiplied by the generator cost curves — it is a "
    "reporting metric, not something the solver minimises.\n"
    "- set_gen_voltage (command 6) DIRECTLY constrains the bus voltage. In OPFLOW, "
    "Vg is only an initial guess that the solver overrides. In PFLOW, the generator's "
    "voltage setpoint is a hard constraint — the solver enforces it. This is your "
    "primary tool for voltage control.\n"
    "- set_gen_dispatch (command 5) sets the generator's active power output directly. "
    "PFLOW will not re-dispatch generators — the dispatch you specify is the dispatch "
    "that will be solved.\n"
    "- PFLOW uses Newton-Raphson (not IPOPT). Convergence is reported as CONVERGED or "
    "DID NOT CONVERGE.\n\n"
    "=== New Commands for PFLOW ===\n\n"
    "In addition to the standard commands, PFLOW search has access to:\n"
    "- set_tap_ratio (command 14): Adjust transformer tap ratio. Only applies to branches "
    "that are transformers (ratio ≠ 0 in the base network). Typical range: 0.9–1.1.\n"
    "- set_shunt_susceptance (command 15): Modify shunt susceptance (Bs) at a bus. "
    "Positive Bs adds capacitive susceptance (raises voltage); negative adds inductive "
    "(lowers voltage).\n"
    "- set_phase_shift_angle (command 16): Adjust phase shifter angle (degrees). Only "
    "applies to branches that are phase shifters (angle ≠ 0 in the base network).\n\n"
    "=== Feasibility Classification ===\n\n"
    "Each iteration is classified as one of:\n"
    "- feasible: PFLOW converged with no constraint violations. The power flow solution "
    "is physically valid.\n"
    "- infeasible: PFLOW did not converge (DID NOT CONVERGE) or the solution has "
    "generation < load (negative losses). No valid power flow solution exists for "
    "the given dispatch and network state.\n"
    "- marginal: PFLOW did not fully converge but the solution data shows voltages "
    "within 0.01 pu of limits or line loading within 5% of thermal limits. This "
    "indicates the operating point is at or near the feasibility boundary — treat as "
    "a boundary marker.\n\n"
    "=== Important Reminders ===\n\n"
    "- PFLOW results include a 'Computed generation cost' line — this is NOT an "
    "optimised cost. It is simply Σ(Pg × cost_curve). Use it to evaluate different "
    "dispatches, but remember the solver did not minimise it.\n"
    "- When PFLOW does not converge, the solution data may be absent or unreliable. "
    "Treat non-convergence as a clear signal that the current network state is infeasible.\n"
    "- Use set_all_bus_vlimits to define the acceptable voltage range for feasibility "
    "checks, just like in OPFLOW."
)

_PFLOW_SEARCH_SEQUENTIAL = (
    "\n\n=== Search Heuristics for LLM-Driven Optimisation ===\n\n"
    "Since PFLOW does not optimise, you must implement the search strategy yourself:\n\n"
    "1. **Feasibility boundary (binary search):** To find the maximum load level before "
    "infeasibility, scale loads incrementally. When PFLOW reports DID NOT CONVERGE, "
    "reduce the scaling and try again. Binary search converges quickly — reduce the "
    "gap by half each iteration and declare 'complete' when the gap is below 1%.\n\n"
    "2. **Cost reduction (gradient-like):** To reduce generation cost while maintaining "
    "feasibility, reduce expensive generator outputs and increase cheaper ones while "
    "staying within Pmin/Pmax bounds. Check cost changes and feasibility after each "
    "adjustment.\n\n"
    "3. **Voltage improvement (iterative adjustment):** To improve voltage profile, "
    "adjust set_gen_voltage on generators near buses with poor voltage. PFLOW will "
    "enforce your setpoints. Increase Vg to raise bus voltages, decrease to lower them.\n\n"
    "4. **Thermal relief (selective modification):** To reduce line loading, redistribute "
    "generation using set_gen_dispatch, or disable overloaded lines with set_branch_status, "
    "or reduce load with scale_load or scale_all_loads.\n\n"
    "- Prefer 'fresh' mode for feasibility boundary searches (binary search). "
    "Prefer 'accumulative' mode for incremental cost/voltage tuning."
)

_PFLOW_SEARCH_CONCURRENT = (
    "\n\n=== Search Heuristics for Concurrent PFLOW ===\n\n"
    "Since PFLOW does not optimise, you must implement the search strategy. "
    "With concurrent PFLOW, use the 'explore' action to test multiple points "
    "simultaneously instead of testing one point per iteration:\n\n"
    "1. **Feasibility boundary (parallel search):** Instead of sequential binary search, "
    "use 'explore' to test 5-8 scaling factors spanning the search range in one round. "
    "Select the best feasible variant, then 'explore' a narrower range. This replaces "
    "N sequential iterations with log2(N) explore+select cycles.\n\n"
    "2. **Cost reduction (parallel dispatch comparison):** Use 'explore' to test "
    "multiple dispatch adjustments simultaneously — e.g., reducing different expensive "
    "generators. Select the variant with the lowest feasible cost.\n\n"
    "3. **Voltage improvement (parallel voltage sweep):** Use 'explore' to test "
    "multiple Vg values at key buses simultaneously. Select the variant with the "
    "best voltage profile.\n\n"
    "4. **Thermal relief (parallel):** Use 'explore' to test different generation "
    "redistributions and load reductions simultaneously.\n\n"
    "- Always use 'explore' as your primary search action. Use 'modify' ONLY for "
    "single-point changes when you are certain of the outcome."
)

_PFLOW_SEARCH_STRATEGY_GUIDANCE = (
    "\n\n=== Search Strategy Guidance ===\n\n"
    "**When to use `explore` (parallel branching):**\n"
    "Use `explore` when you want to test multiple independent hypotheses simultaneously. "
    "Each variant should represent a structurally different strategy, not just a parameter "
    "sweep of the same idea. `explore` is most valuable when you are uncertain which "
    "direction to take.\n\n"
    "**When to use `fresh` (single-step):**\n"
    "Use `fresh` when the direction is clear and you just need to take the next step — "
    "for example, during a binary search for the load feasibility boundary where each "
    "step's outcome determines the next.\n\n"
    "**Budget allocation heuristic:**\n"
    "Spend the first ~20% of your budget establishing a baseline and benchmark reference "
    "(if not already done), the middle ~60% on parallel explore iterations testing diverse "
    "strategies, and the final ~20% on refining the best direction found and confirming "
    "convergence.\n\n"
    "**Phase transitions:**\n"
    "If you have run 3+ consecutive explores and the session-best cost has not improved, "
    "consider: (a) that the search space has been exhausted for this network configuration, "
    "or (b) that a fundamentally different strategy is needed. Do not keep exploring the "
    "same direction with minor variations.\n\n"
    "**Analyze calls are expensive:**\n"
    "Each `analyze` call consumes one iteration from your budget. Reserve it for genuinely "
    "ambiguous situations. The Network Metadata section already contains the static facts "
    "(slack bus, cost coefficients, generator headroom) — do not use `analyze` to retrieve "
    "information already available there."
)

_PFLOW_VARIANT_READING_GUIDANCE = (
    "\n\n=== Reading Variant Results ===\n\n"
    "- If two variants return identical cost despite having different commands, "
    "the extra commands in the more complex variant had no effect. The most "
    "likely cause is a skipped command (check the description for [SKIP] and "
    "the skipped-command note). Do not repeat those commands in the next explore.\n"
    "- If your batch's cheapest variant is more expensive than the "
    "\"Session best\" shown above, your current search direction is regressing. "
    "Consider returning to a command set closer to the session-best commands, "
    "or explore a fundamentally different direction.\n"
    "- The \"Session best\" line shows the best cost ever found, including from "
    "non-selected variants. Use it as your primary cost reference, not the costs "
    "shown for prior selected iterations in the Search Journal."
)

_PFLOW_SECTION = _PFLOW_SECTION_CORE + _PFLOW_SEARCH_SEQUENTIAL

_EXPLORE_PFLOW_SECTION = (
    "\n\n=== Concurrent Neighborhood Search (Explore/Select) ===\n\n"
    "Concurrent PFLOW is ENABLED. You MUST use the 'explore' action as your "
    "primary search mechanism. The 'modify' action should only be used for "
    "single-point changes when you are certain of the outcome.\n\n"
    "The explore action evaluates multiple configurations simultaneously. "
    "The system will:\n"
    "- Run all variants concurrently\n"
    "- Compute a Pareto front based on tracked objectives\n"
    "- Present results with Pareto-optimal variants marked (★)\n\n"
    "After an 'explore', you MUST choose one of:\n"
    "- 'select' to adopt one variant as the new current point\n"
    "- 'explore' again to test a different neighborhood\n"
    "- 'analyze' to inspect results before choosing\n"
    "- 'complete' to end the search\n\n"
    "Required workflow:\n"
    "- Use 'explore' with 3-8 variants to test different parameter values\n"
    "- Use 'select' to adopt the best Pareto variant\n"
    "- Repeat: explore → select → explore → ...\n"
    "- Do NOT use 'modify' for search iterations — use 'explore' instead\n"
    "- You can 'explore' with 'mode': 'fresh' to test variants from the base case\n\n"
    "Explore is especially effective for:\n"
    "- Binary/bisection search: test multiple scaling factors in one round instead of one at a time\n"
    "- Voltage sweep: test different Vg values at key buses simultaneously\n"
    "- Dispatch sweep: test different Pg values at generators simultaneously\n"
    "- Load scaling sweep: test different scale_all_loads factors simultaneously\n"
    "- Combined variations: each variant is an independent set of modifications\n\n"
    "**CRITICAL: Every variant MUST include all commands that are required by the goal.** "
    "For example, if the goal says 'scale loads to 1.23', every variant must include "
    "scale_all_loads(factor=1.23). If the goal says 'enforce voltage limits 0.95-1.05', "
    "every variant must include set_all_bus_vlimits(Vmin=0.95, Vmax=1.05). "
    "Omitting these fixed commands in some variants makes the results incomparable.\n\n"
    "Example: For a feasibility boundary search, instead of testing one factor per "
    "iteration with 'modify', propose 5 factors spanning the search range. The system "
    "runs all 5 in parallel, identifies which are feasible, and you can then 'select' "
    "the best and 'explore' a narrower range. This replaces sequential binary search "
    "with parallel coordinate search, converging in fewer LLM round-trips."
)

_EXPLORE_PFLOW_SECTION_NO_CONCURRENT = (
    "\n\n=== Search Strategy ===\n\n"
    "Since concurrent PFLOW is not enabled, you must search one point at a time. "
    "Use the 'modify' action for each test. Recommended strategies:\n"
    "- For feasibility boundary searches: use binary search with 'fresh' mode\n"
    "- For incremental tuning: use 'accumulative' mode\n"
    "- Start with small changes, observe the effect, then adjust\n\n"
    "Tip: If you need to test many parameter values, ask the operator to enable "
    "concurrent PFLOW (--concurrent-pflow) to run multiple simulations in parallel."
)

def _app_section(
    application: str,
    concurrent_pflow: bool = False,
    session_load_factor: float | None = None,
) -> str:
    """Return the application-specific prompt section for the given app."""
    if application == "dcopflow":
        return _DC_OPF_SECTION
    if application == "scopflow":
        return _AC_OPF_VOLTAGE_SECTION + "\n\n" + _SCOPFLOW_SECTION
    if application == "tcopflow":
        return _AC_OPF_VOLTAGE_SECTION + "\n\n" + _TCOPFLOW_SECTION
    if application == "sopflow":
        return _AC_OPF_VOLTAGE_SECTION + "\n\n" + _SOPFLOW_SECTION
    if application == "pflow":
        load_factor_note = ""
        if session_load_factor is not None:
            load_factor_note = (
                f"\n\n=== Session Load Factor ===\n\n"
                f"The session load factor is **{session_load_factor}×** and is applied "
                f"automatically to every run. Do not include `scale_all_loads` in your "
                f"commands — it has already been applied. If you want to explore a "
                f"different load level, use the `set_load_factor` action:\n"
                f'{{"action": "set_load_factor", "factor": 1.35}}\n'
                f"This updates the session-level factor for all subsequent iterations."
            )
        if concurrent_pflow:
            result = (
                _PFLOW_SECTION_CORE
                + load_factor_note
                + _PFLOW_SEARCH_CONCURRENT
                + _EXPLORE_PFLOW_SECTION
                + _PFLOW_SEARCH_STRATEGY_GUIDANCE
                + _PFLOW_VARIANT_READING_GUIDANCE
            )
        else:
            result = (
                _PFLOW_SECTION_CORE
                + load_factor_note
                + _PFLOW_SEARCH_SEQUENTIAL
                + _EXPLORE_PFLOW_SECTION_NO_CONCURRENT
                + _PFLOW_SEARCH_STRATEGY_GUIDANCE
                + _PFLOW_VARIANT_READING_GUIDANCE
            )
        return result
    return _AC_OPF_VOLTAGE_SECTION


_DC_STRESS_TEST_SEVERITY = (
    "Rank contingencies by severity: infeasibility > high line loading > cost increase "
    "(no voltage violations in DC)."
)

_AC_STRESS_TEST_SEVERITY = (
    "Rank contingencies by severity: infeasibility > voltage violations > high line loading > cost increase."
)


def _build_standard_prompt(
    command_schema: str,
    network_summary: str,
    application: str,
    concurrent_pflow: bool = False,
    network_metadata: str | None = None,
    benchmark_text: str | None = None,
    session_load_factor: float | None = None,
) -> str:
    if concurrent_pflow and application == "pflow":
        _action_header = "You MUST respond with a single JSON object. Choose one of five actions:"
        _action_section = """
1. EXPLORE the neighborhood — evaluate multiple configurations concurrently (PREFERRED for search):
{{
  "action": "explore",
  "reasoning": "Why these variants are worth testing.",
  "mode": "fresh" or "accumulative",
  "description": "Short description of the exploration",
  "variants": [
    {{"label": "A", "commands": [{{"action": "set_gen_voltage", "bus": 1, "Vg": 1.02}}]}},
    {{"label": "B", "commands": [{{"action": "set_gen_voltage", "bus": 1, "Vg": 1.04}}]}},
    {{"label": "C", "commands": [{{"action": "set_gen_voltage", "bus": 1, "Vg": 1.06}}]}}
  ]
}}

2. SELECT a variant — after explore, adopt one of the evaluated points as the new current point:
{{
  "action": "select",
  "choice": "A",
  "reasoning": "Why this variant is the best choice for the next iteration."
}}

3. MODIFY the network — apply a single change and run a simulation (use ONLY when you are certain of the outcome):
{{
  "action": "modify",
  "reasoning": "Explanation of why these changes should help achieve the goal.",
  "mode": "fresh" or "accumulative",
  "description": "Short one-line description for the search journal",
  "commands": [{{"action": "...", ...}}]
}}

4. COMPLETE the search — when the goal is achieved or determined infeasible:
{{
  "action": "complete",
  "reasoning": "Explanation of why the search is done.",
  "findings": {{
    "summary": "Concise answer to the goal.",
    "details": "Supporting data and observations."
  }}
}}

5. ANALYZE results — request specific data before deciding:
{{
  "action": "analyze",
  "reasoning": "What information is needed and why.",
  "query": "e.g. buses with voltage below 0.95"
}}
"""
    else:
        _action_header = "You MUST respond with a single JSON object. Choose one of three actions:"
        _action_section = """
1. MODIFY the network — apply changes and run a simulation:
{{
  "action": "modify",
  "reasoning": "Explanation of why these changes should help achieve the goal.",
  "mode": "fresh" or "accumulative",
  "description": "Short one-line description for the search journal",
  "commands": [{{"action": "...", ...}}]
}}

2. COMPLETE the search — when the goal is achieved or determined infeasible:
{{
  "action": "complete",
  "reasoning": "Explanation of why the search is done.",
  "findings": {{
    "summary": "Concise answer to the goal.",
    "details": "Supporting data and observations."
  }}
}}

3. ANALYZE results — request specific data before deciding:
{{
  "action": "analyze",
  "reasoning": "What information is needed and why.",
  "query": "e.g. buses with voltage below 0.95"
}}
"""

    metadata_section = ""
    if network_metadata:
        metadata_section = f"\n=== Section G: Network Metadata ===\n\n{network_metadata}\n"

    benchmark_section = ""
    if benchmark_text:
        benchmark_section = f"\n=== Section H: Benchmark Reference (OPFLOW vs PFLOW) ===\n\n{benchmark_text}\n"

    return f"""\
You are a power systems analysis agent. You iteratively modify a power grid \
network and run {application.upper()} simulations to achieve a user-specified goal.

=== Section A: Available Commands ===

{command_schema}

=== Section B: Network Information ===

{network_summary}
{metadata_section}{benchmark_section}
=== Response Format ===

{_action_header}
{_action_section}
=== Rules ===

- Be systematic: start with small changes, observe the effect, then adjust.
- Explain your reasoning in every response.
- Respect physical bounds: generator Pmin/Pmax, voltage limits, thermal ratings.
- Use "fresh" mode to apply commands to the original base case network.
- Use "accumulative" mode to build on top of the previous iteration's network.
- Fresh mode is best for binary-search or parameter-sweep approaches.
- Accumulative mode is best for incremental refinement.
- Declare "complete" when you have a clear answer, when further iterations \
cannot improve the result, or when the goal is provably infeasible.
- When performing a binary search (e.g., finding a maximum scaling factor), \
declare "complete" as soon as the feasible/infeasible gap is below 1%. The \
last feasible value is your answer — further refinement wastes iterations \
without meaningful improvement.
- If the last 2-3 iterations all classify as "marginal" or oscillate between \
"feasible" and "infeasible" with a tiny gap, you are at the boundary — \
declare "complete" immediately.
- Do NOT repeat the same modification if it already failed.
- If a simulation diverges, try a smaller or different change.
- When multiple objectives are being tracked, explain tradeoffs between them \
in your reasoning. If you notice a tension between objectives (e.g., cost \
decreasing but voltage stability degrading), flag it explicitly.
- You may propose tracking additional metrics by including a "propose_objectives" \
field in your JSON response (optional): \
"propose_objectives": [{{"name": "<metric>", "direction": "minimize", "priority": "secondary"}}]
- The operator can accept or reject proposed objectives via steering.

{_app_section(application, concurrent_pflow, session_load_factor)}"""


def _build_stress_test_prompt(
    command_schema: str,
    network_summary: str,
    application: str,
    network_metadata: str | None = None,
) -> str:
    metadata_section = ""
    if network_metadata:
        metadata_section = f"\n=== Section G: Network Metadata ===\n\n{network_metadata}\n"

    return f"""\
You are a power systems security analyst performing adversarial stress testing \
on a power grid network. Your goal is to systematically identify critical \
contingencies — component outages that cause the most severe impact on \
system operation.

=== Section A: Available Commands ===

{command_schema}

=== Section B: Network Information ===

{network_summary}
{metadata_section}
=== Response Format ===

You MUST respond with a single JSON object. Choose one of three actions:

1. MODIFY the network — test a contingency by disabling component(s):
{{
  "action": "modify",
  "reasoning": "Why this contingency is worth testing.",
  "mode": "fresh",
  "description": "N-1: Line 42->87 outage",
  "contingency": {{
    "type": "N-1" or "N-2",
    "components": ["branch 42->87"]
  }},
  "commands": [{{"action": "set_branch_status", "fbus": 42, "tbus": 87, "status": 0}}]
}}

2. COMPLETE the search — report findings after sufficient testing:
{{
  "action": "complete",
  "reasoning": "Sufficient contingencies tested to characterize system vulnerability.",
  "findings": {{
    "summary": "Critical contingencies identified.",
    "critical_contingencies": [
      {{"components": ["branch X->Y"], "severity": "high", "impact": "description"}},
    ],
    "most_critical": "branch X->Y outage causes ...",
    "system_resilience": "overall assessment"
  }}
}}

3. ANALYZE results — request data before deciding the next contingency:
{{
  "action": "analyze",
  "reasoning": "Need line loading data to identify next candidate.",
  "query": "most loaded lines"
}}

=== Stress Testing Strategy ===

- ALWAYS use "fresh" mode — each contingency must be tested independently from the base case.
- Start with N-1 contingencies (single component outages).
- Focus on the most loaded lines first — they are the most likely to cause cascading issues when tripped.
- After testing key N-1 contingencies, consider N-2 combinations of the most impactful outages.
- For each contingency, assess: Did the system converge? How did cost change? \
Were there voltage violations? Which lines became overloaded?
- {_DC_STRESS_TEST_SEVERITY if application == "dcopflow" else _AC_STRESS_TEST_SEVERITY}
- Use the "analyze" action to inspect line loadings and identify the next candidate if needed.
- Declare "complete" once you've tested the most critical contingencies \
and can characterize the system's vulnerability profile.
- Do NOT test contingencies on lines with very low loading (<20%) — they are unlikely to be critical.

{_app_section(application)}"""
