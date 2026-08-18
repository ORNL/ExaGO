"""Pareto front filter for concurrent PFLOW explore results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentigrid.engine.journal import ObjectiveEntry

logger = logging.getLogger("agentigrid.engine.pareto")


@dataclass
class ParetoCandidate:
    """A variant result evaluated for Pareto optimality."""

    label: str
    metrics: dict[str, float]
    feasible: bool


def _dominates(
    a: ParetoCandidate,
    b: ParetoCandidate,
    objectives: list[ObjectiveEntry],
) -> bool:
    """Return True if *a* dominates *b*.

    A dominates B when A is at least as good as B on all objectives
    and strictly better on at least one. For "minimize" objectives,
    lower is better; for "maximize", higher is better.

    Infeasible candidates can never dominate feasible ones, and vice versa:
    feasible always dominates infeasible.
    """
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return False

    if not objectives or not a.metrics or not b.metrics:
        return False

    at_least_one_better = False
    for obj in objectives:
        a_val = a.metrics.get(obj.name)
        b_val = b.metrics.get(obj.name)
        if a_val is None or b_val is None:
            continue

        if obj.direction == "minimize":
            if a_val > b_val:
                return False
            if a_val < b_val:
                at_least_one_better = True
        elif obj.direction == "maximize":
            if a_val < b_val:
                return False
            if a_val > b_val:
                at_least_one_better = True
        else:
            if a_val != b_val:
                return False

    return at_least_one_better


def pareto_filter(
    candidates: list[ParetoCandidate],
    objectives: list[ObjectiveEntry],
) -> list[str]:
    """Return labels of non-dominated candidates.

    Infeasible candidates are never Pareto-optimal (and can never
    dominate a feasible candidate). Among feasible candidates, standard
    Pareto dominance applies using the registered objectives.

    When no objectives are registered, the default comparison metric
    is ``generation_cost`` with direction ``minimize``. When a candidate
    lacks the required metrics, it is treated as incomparable (not
    Pareto-optimal).

    Args:
        candidates: List of ParetoCandidate objects with metrics.
        objectives: Registered objectives for comparison.

    Returns:
        List of labels of Pareto-optimal candidates.
    """
    if not candidates:
        return []

    effective_objectives = objectives
    if not effective_objectives:
        effective_objectives = [
            ObjectiveEntry(
                name="generation_cost",
                direction="minimize",
                priority="primary",
                introduced_at=0,
                source="default",
            )
        ]

    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        logger.info("Pareto filter: no feasible candidates among %d total", len(candidates))
        return []

    pareto_labels: list[str] = []
    for i, a in enumerate(feasible):
        dominated = False
        for j, b in enumerate(feasible):
            if i == j:
                continue
            if _dominates(b, a, effective_objectives):
                dominated = True
                break
        if not dominated:
            pareto_labels.append(a.label)

    logger.info(
        "Pareto filter: %d feasible of %d candidates, %d Pareto-optimal: %s",
        len(feasible), len(candidates), len(pareto_labels), pareto_labels,
    )
    return pareto_labels
