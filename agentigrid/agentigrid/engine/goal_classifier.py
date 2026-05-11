"""Shared post-search goal classification utilities.

Used by both the CLI path (AgentLoopController._finalize) and the GUI path
(SessionManager.get_summary_analysis) so the LLM prompt and JSON parsing logic
are defined in exactly one place.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("agentigrid.engine.goal_classifier")

_SYSTEM_PROMPT = (
    "You are an expert power systems analyst reviewing the results of an "
    "LLM-driven optimization search performed using an ExaGO application "
    "(OPFLOW, DCOPFLOW, SCOPFLOW, TCOPFLOW, SOPFLOW, or PFLOW). "
    "Provide a structured analytical summary of the search."
)

_GOAL_TYPE_DEFS = (
    "Goal type definitions:\n"
    "- cost_minimization: User wants to minimize generation cost."
    " Best = lowest cost among feasible.\n"
    "- feasibility_boundary: User wants to find the limit of a parameter before infeasibility."
    " Best = feasible iteration closest to the boundary. Marginal iterations"
    " (solver did not fully converge but no violations detected) can serve"
    " as boundary markers.\n"
    "- constraint_satisfaction: User wants to satisfy specific constraints."
    " Best = iteration that best satisfies them.\n"
    "- parameter_exploration: User is exploring what-if scenarios."
    " Best = most informative feasible iteration.\n"
)


def build_classification_prompts(
    goal: str,
    termination_reason: str,
    stats: dict[str, Any],
    journal_formatted: str,
    total_tokens: int,
    objective_registry: list[dict] | None = None,
    preference_history: list[dict] | None = None,
    application: str = "opflow",
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for post-search goal classification.

    The prompt requests both a structured analysis narrative *and* a JSON
    classification block, making it suitable for both CLI display and GUI
    reporting in a single LLM call.

    Args:
        goal: The original natural-language search goal.
        termination_reason: How the search ended (completed, max_iterations, etc.).
        stats: dict from ``SearchJournal.summary_stats()`` (before any override).
        journal_formatted: Output of ``SearchJournal.format_detailed()``.
        total_tokens: Total prompt + completion tokens used during the search.
        objective_registry: Optional list of tracked objective dicts.
        preference_history: Optional list of preference change dicts.
        application: ExaGO application name (e.g., "opflow", "tcopflow", "sopflow").

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    system_prompt = _SYSTEM_PROMPT
    _APP_CONTEXT = {
        "tcopflow": (
            "This search used TCOPFLOW (Multi-Period Optimal Power Flow). "
            "TCOPFLOW optimizes across multiple time periods with generator "
            "ramp constraints. The objective is total cost across all periods. "
            "When identifying the best iteration, consider temporal aspects: "
            "the worst-case period determines overall feasibility, and "
            "feasibility_boundary searches should consider the temporal horizon."
        ),
        "scopflow": (
            "This search used SCOPFLOW (Security-Constrained OPF). Results "
            "reflect base-case N-1 security analysis."
        ),
        "sopflow": (
            "This search used SOPFLOW (Stochastic Optimal Power Flow). "
            "SOPFLOW optimizes a two-stage problem: a first-stage base-case "
            "dispatch plus per-scenario second-stage corrections for wind "
            "generation uncertainty. The objective value is the expected cost "
            "across all scenarios (weighted base-case cost). Feasibility means "
            "all scenarios satisfy network constraints simultaneously. "
            "When identifying the best iteration, consider how well the dispatch "
            "handles wind variability across all scenarios."
        ),
        "pflow": (
            "This search used PFLOW (AC Power Flow — analysis, not optimization). "
            "PFLOW does NOT minimise cost — it solves the power flow equations for a "
            "given network state. The 'computed generation cost' is calculated from the "
            "dispatch multiplied by cost curves; it is NOT an optimised objective. "
            "The LLM was the optimiser, proposing changes and evaluating feasibility. "
            "When identifying the best iteration, consider: feasibility (CONVERGED vs "
            "DID NOT CONVERGE), voltage profile quality, line loading, and the computed "
            "generation cost as a secondary metric. For feasibility_boundary searches, "
            "the best iteration is the feasible one closest to the boundary (or the "
            "last feasible iteration before infeasibility). For cost-focused searches, "
            "the best iteration has the lowest computed cost among feasible results."
        ),
    }
    app_context = _APP_CONTEXT.get(application)
    if app_context:
        system_prompt = system_prompt + " " + app_context
    user_prompt = (
        f"Search goal: {goal}\n"
        f"Termination reason: {termination_reason}\n"
        f"Total iterations: {stats['total_iterations']}\n"
        f"Feasible: {stats['feasible_count']} / "
        f"Infeasible: {stats['infeasible_count']}"
    )
    if stats.get("marginal_count", 0) > 0:
        user_prompt += f" / Marginal: {stats['marginal_count']}"
    user_prompt += (
        f"\n"
        f"Lowest-cost feasible: {stats['best_objective']} "
        f"(iteration {stats['best_iteration']})\n"
        f"Tokens used: ~{total_tokens:,}\n"
        f"\n"
        f"=== Detailed Journal ===\n"
        f"{journal_formatted}\n"
    )

    if objective_registry:
        user_prompt += (
            f"\n=== Tracked Objectives ===\n"
            f"{json.dumps(objective_registry, indent=2)}\n"
        )
    if preference_history:
        user_prompt += (
            f"\n=== Preference History ===\n"
            f"{json.dumps(preference_history, indent=2)}\n"
        )

    user_prompt += (
        f"\n"
        f"Please provide:\n"
        f"\n"
        f"1. A structured analysis covering:\n"
        f"   a. Overall assessment — was the goal achieved?\n"
        f"   b. Search strategy analysis — what approach was taken?\n"
        f"   c. Convergence behavior — monotonic improvement, exploration, plateaus?\n"
        f"   d. Key modifications that had the most impact\n"
        f"   e. Potential further improvements\n"
        f"   f. Recommendations\n"
        f"\n"
        f"2. At the END of your response, include a JSON block wrapped in\n"
        f"   ```json ... ``` with the following structure:\n"
        f"\n"
        f"```json\n"
        f'{{\n'
        f'  "goal_type": "<one of: cost_minimization | feasibility_boundary'
        f' | constraint_satisfaction | parameter_exploration>",\n'
        f'  "best_iteration": <int — the iteration number that best answers'
        f" the user's goal>,\n"
        f'  "best_iteration_rationale": "<one sentence explaining why this'
        f' iteration is the best answer>",\n'
        f'  "is_multi_objective": <true or false>,\n'
        f'  "tradeoff_summary": "<one paragraph describing key tradeoffs between'
        f' objectives, or null if single-objective>",\n'
        f'  "recommended_solutions": [<list of iteration numbers for best tradeoff'
        f' options, or just one for single-objective>]\n'
        f'}}\n'
        f"```\n"
        f"\n"
        f"{_GOAL_TYPE_DEFS}"
    )
    return system_prompt, user_prompt


def parse_goal_classification(
    text: str,
    valid_iteration_numbers: set[int],
) -> dict[str, Any] | None:
    """Extract and validate the goal classification JSON block from an LLM response.

    Tries a fenced `` ```json ... ``` `` block first, then falls back to a
    bare ``{"goal_type": ...}`` pattern.  Returns *None* if parsing or
    validation fails so callers can fall back to cost-heuristic defaults.

    Args:
        text: Raw LLM response text.
        valid_iteration_numbers: Set of iteration numbers present in the journal.
            Used to reject hallucinated iteration numbers.

    Returns:
        Dict with keys ``goal_type``, ``best_iteration``,
        ``best_iteration_rationale``, or *None*.
    """
    # Try ```json ... ``` fenced block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # Fall back to bare JSON object containing a "goal_type" key
        match = re.search(r'\{\s*"goal_type"\s*:.*?\}', text, re.DOTALL)

    if match:
        try:
            json_str = match.group(1) if match.lastindex else match.group(0)
            data = json.loads(json_str)

            goal_type = data.get("goal_type", "cost_minimization")
            best_iter = data.get("best_iteration")
            rationale = data.get("best_iteration_rationale", "")

            if best_iter not in valid_iteration_numbers:
                logger.warning(
                    "LLM returned invalid best_iteration=%s (valid: %s); "
                    "falling back to cost heuristic",
                    best_iter,
                    sorted(valid_iteration_numbers),
                )
                return None

            logger.info(
                "Goal classification: type=%s, best_iter=%s, rationale=%s",
                goal_type,
                best_iter,
                rationale,
            )
            return {
                "goal_type": goal_type,
                "best_iteration": best_iter,
                "best_iteration_rationale": rationale,
                "is_multi_objective": data.get("is_multi_objective", False),
                "tradeoff_summary": data.get("tradeoff_summary"),
                "recommended_solutions": data.get("recommended_solutions", [best_iter]),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse goal classification JSON: %s", exc)

    logger.info("No goal classification found in LLM response; using cost heuristic")
    return None
