"""User prompt template for each iteration."""

from __future__ import annotations

from typing import Optional


def _format_session_best(session_best: dict) -> str:
    """Format the session-best record for injection into the user prompt.

    Args:
        session_best: Dict with keys cost, iteration, variant_label, commands.

    Returns:
        A compact multi-line string summarising the all-time best.
    """
    cost = session_best.get("cost")
    iteration = session_best.get("iteration")
    label = session_best.get("variant_label", "?")
    commands = session_best.get("commands", [])

    cost_str = f"${cost:,.2f}" if cost is not None else "N/A"
    cmd_parts = []
    for cmd in commands[:6]:
        action = cmd.get("action", "?")
        # Compact representation: action + key parameter
        if action == "scale_all_loads":
            cmd_parts.append(f"scale×{cmd.get('factor', '?'):g}" if isinstance(cmd.get("factor"), (int, float)) else f"scale×{cmd.get('factor', '?')}")
        elif action == "set_gen_dispatch":
            cmd_parts.append(f"dispatch bus{cmd.get('bus', '?')}→{cmd.get('Pg', '?')}MW")
        elif action == "set_all_bus_vlimits":
            cmd_parts.append(f"vlim[{cmd.get('Vmin', '?')}-{cmd.get('Vmax', '?')}]")
        elif action == "set_gen_voltage":
            cmd_parts.append(f"Vg bus{cmd.get('bus', '?')}={cmd.get('Vg', '?')}")
        elif action == "set_gen_status":
            status = cmd.get("status", 1)
            act = "commit" if status == 1 else "trip"
            cmd_parts.append(f"{act} bus{cmd.get('bus', '?')}")
        else:
            cmd_parts.append(action)
    if len(commands) > 6:
        cmd_parts.append(f"+{len(commands) - 6} more")
    cmd_str = ", ".join(cmd_parts) if cmd_parts else "(no commands)"

    return (
        f"Session best (feasible): {cost_str}  [iter {iteration}, variant {label}]\n"
        f"  Commands: {cmd_str}"
    )


def _format_cost_reference_line(
    benchmark_result: Optional[dict],
    session_best: Optional[dict],
) -> Optional[str]:
    """Format the one-line cost reference injected before the journal.

    Shows OPFLOW optimal, PFLOW baseline, and session-best cost on one line.
    Returns None if no benchmark or session-best data is available.
    """
    parts = []

    if benchmark_result and benchmark_result.get("opflow_converged"):
        opflow_cost = benchmark_result.get("opflow_objective")
        pflow_cost = benchmark_result.get("pflow_best_computed_cost")
        gap_pct = benchmark_result.get("cost_gap_pct")
        if opflow_cost is not None:
            parts.append(f"OPFLOW optimal: ${opflow_cost:,.2f}")
        if pflow_cost is not None and gap_pct is not None:
            parts.append(f"PFLOW baseline: ${pflow_cost:,.2f} ({gap_pct:+.2f}% gap)")
        elif pflow_cost is not None:
            parts.append(f"PFLOW baseline: ${pflow_cost:,.2f}")

    if session_best is not None:
        cost = session_best.get("cost")
        iteration = session_best.get("iteration")
        label = session_best.get("variant_label", "?")
        if cost is not None:
            parts.append(f"Session best: ${cost:,.2f} [iter {iteration}, variant {label}]")

    return " | ".join(parts) if parts else None


def build_user_prompt(
    goal: str,
    journal_text: Optional[str],
    results_text: Optional[str],
    error_feedback: Optional[str] = None,
    steering_directives: list[dict] | None = None,
    multi_objective_text: Optional[str] = None,
    explore_text: Optional[str] = None,
    session_best: Optional[dict] = None,
    current_iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
    benchmark_result: Optional[dict] = None,
) -> str:
    """Build the user prompt for one iteration.

    Args:
        goal: Natural language search goal.
        journal_text: Output of journal.format_for_prompt(), or None.
        results_text: Output of results_summary(), or None.
        error_feedback: Error messages from previous iteration, if any.
        steering_directives: List of active steering directive dicts
            with keys "directive" and "mode", or None.
        multi_objective_text: Formatted multi-objective tracking context, or None.
        explore_text: Formatted variant comparison table from an explore action,
            or None. When set, replaces Section D in the prompt.
        session_best: Dict from SearchJournal.session_best — the best feasible
            cost found across all variants (not just selected ones). Injected
            before the explore table so the LLM can detect regressions.
        current_iteration: Current iteration number (1-based). Used for budget display.
        max_iterations: Total iteration budget. When set, shows remaining count.
        benchmark_result: Serialized BenchmarkResult dict. Used for the one-line
            cost reference injected before the journal.

    Returns:
        Complete user prompt string.
    """
    parts = []

    # Budget warning when remaining iterations ≤ 3
    if (
        current_iteration is not None
        and max_iterations is not None
        and max_iterations - current_iteration <= 3
    ):
        remaining = max_iterations - current_iteration
        parts.append(
            f"WARNING: {remaining} iteration(s) remaining. "
            "Prioritize consolidation: select the best-known state, confirm it is "
            "feasible, and report your findings. Avoid starting new exploration directions."
        )
        parts.append("")

    # Iteration counter
    if current_iteration is not None:
        if max_iterations is not None:
            remaining = max_iterations - current_iteration
            parts.append(
                f"Iteration: {current_iteration} / {max_iterations}  (remaining: {remaining})"
            )
        else:
            parts.append(f"Iteration: {current_iteration} (no budget limit)")

    parts.append(f"Goal: {goal}")

    # Cost reference one-liner (benchmark + session best on same line)
    cost_ref = _format_cost_reference_line(benchmark_result, session_best)
    if cost_ref:
        parts.append("")
        parts.append(cost_ref)

    if journal_text:
        parts.append("")
        parts.append("=== Section C: Search Journal ===")
        parts.append(journal_text)

    if multi_objective_text:
        parts.append("")
        parts.append("=== Section E: Multi-Objective Tracking ===")
        parts.append(multi_objective_text)

    if session_best is not None:
        parts.append("")
        parts.append(_format_session_best(session_best))

    if explore_text:
        parts.append("")
        parts.append(explore_text)
    elif results_text:
        parts.append("")
        parts.append("=== Section D: Latest Results ===")
        parts.append(results_text)

    if error_feedback:
        parts.append("")
        parts.append("=== Error Feedback ===")
        parts.append(error_feedback)

    if steering_directives:
        parts.append("")
        parts.append("=== Operator Directives ===")
        parts.append(
            "The user has provided the following steering instructions during the search.\n"
            "Incorporate these into your decision-making:"
        )
        for i, sd in enumerate(steering_directives, start=1):
            tag = "AUGMENT" if sd.get("mode") == "augment" else "REPLACE"
            parts.append(f'{i}. [{tag}] "{sd["directive"]}"')
        parts.append("")
        parts.append(
            "For AUGMENT directives, consider them alongside the original goal.\n"
            "For REPLACE directives, they supersede the original goal."
        )

    parts.append("")
    parts.append(
        "Based on the above, decide your next action. "
        "Respond with a single JSON object."
    )

    return "\n".join(parts)
