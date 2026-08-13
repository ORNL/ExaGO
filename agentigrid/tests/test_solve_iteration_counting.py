"""Non-solve entries (analyze/complete) must not be counted as failed solves.

A run with one real solve + an analyze query + a completion marker was written up
as a partial failure ("Infeasible: 2"). The shared `is_solve_iteration` predicate
excludes control entries from feasible/infeasible tallies, convergence charts, and
the completion narrative, while genuine infeasible SOLVES still count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentigrid.engine.journal import SearchJournal, is_solve_iteration
from agentigrid.engine.goal_classifier import build_classification_prompts
from agentigrid.parsers.opflow_results import OPFLOWResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
from charts import convergence_chart, voltage_range_chart  # noqa: E402


def _feasible_solve():
    r = OPFLOWResult(
        converged=True, objective_value=14313.0, convergence_status="CONVERGED",
        solver="EMPAR", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=1, solve_time=0.1,
    )
    r.total_gen_mw = 1000.0
    r.total_load_mw = 980.0
    r.voltage_min = 0.98
    r.voltage_max = 1.04
    r.num_violations = 0
    r.feasibility_detail = "feasible"
    r.losses_mw = 20.0
    return r


def _infeasible_solve():
    r = OPFLOWResult(
        converged=False, objective_value=None, convergence_status="DID NOT CONVERGE",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=1, solve_time=0.1,
    )
    r.num_violations = 3
    r.feasibility_detail = "infeasible"
    return r


def _run_journal():
    """iter0 real solve (feasible) + iter1 analyze (returned data) + iter2 complete."""
    j = SearchJournal()
    j.add_from_results(
        iteration=0, description="Base case", commands=[], opflow_result=_feasible_solve(),
        sim_elapsed=1.0, llm_reasoning="base", mode="fresh", num_scenarios=10,
        exago_command={"cwd": "/wd", "argv": ["exago", "-netfile", "x.m"]},
    )
    j.add_analysis(
        iteration=1, query="scenario_voltage_spread k=10",
        result_summary="Buses most affected: 152, 154, 155, 153, 151",
    )
    j.add_complete(iteration=2, summary="Grid handles all scenarios.")
    return j


# ---------------------------------------------------------------------------
# Change 1 — the predicate
# ---------------------------------------------------------------------------

def test_predicate_solve_vs_control():
    j = _run_journal()
    solve, analyze, complete = j.entries
    assert is_solve_iteration(solve) is True
    assert is_solve_iteration(analyze) is False
    assert is_solve_iteration(complete) is False


def test_predicate_solve_needs_exago_command():
    # A fresh-mode entry with no ExaGO run recorded is not a solve iteration.
    j = SearchJournal()
    j.add_from_results(
        iteration=0, description="no-run", commands=[], opflow_result=_feasible_solve(),
        sim_elapsed=1.0, llm_reasoning="", mode="fresh", exago_command=None,
    )
    assert is_solve_iteration(j.entries[0]) is False


def test_predicate_infeasible_solve_still_counts():
    j = SearchJournal()
    j.add_from_results(
        iteration=0, description="bad", commands=[], opflow_result=_infeasible_solve(),
        sim_elapsed=1.0, llm_reasoning="", mode="fresh",
        exago_command={"cwd": "/wd", "argv": ["exago"]},
    )
    assert is_solve_iteration(j.entries[0]) is True


# ---------------------------------------------------------------------------
# summary_stats solve-only keys
# ---------------------------------------------------------------------------

def test_summary_stats_solve_only_counts():
    j = _run_journal()
    s = j.summary_stats()
    assert s["solve_feasible_count"] == 1
    assert s["solve_infeasible_count"] == 0
    assert s["control_count"] == 2
    # Legacy keys are preserved (backward compatibility).
    assert s["feasible_count"] == 1
    assert s["infeasible_count"] == 2  # legacy still counts control entries


def test_summary_stats_infeasible_solve_counts_as_infeasible():
    j = SearchJournal()
    j.add_from_results(
        iteration=0, description="bad", commands=[], opflow_result=_infeasible_solve(),
        sim_elapsed=1.0, llm_reasoning="", mode="fresh",
        exago_command={"cwd": "/wd", "argv": ["exago"]},
    )
    j.add_complete(iteration=1, summary="done")
    s = j.summary_stats()
    assert s["solve_feasible_count"] == 0
    assert s["solve_infeasible_count"] == 1  # genuine infeasible solve counted
    assert s["control_count"] == 1


def test_summary_stats_empty_journal_has_solve_keys():
    s = SearchJournal().summary_stats()
    assert s["solve_feasible_count"] == 0
    assert s["solve_infeasible_count"] == 0
    assert s["control_count"] == 0


# ---------------------------------------------------------------------------
# Change 4 — narrative context (counts + labeling)
# ---------------------------------------------------------------------------

def test_narrative_counts_are_solve_only():
    j = _run_journal()
    _sys, user = build_classification_prompts(
        goal="handle all scenarios", termination_reason="complete",
        stats=j.summary_stats(), journal_formatted=j.format_for_classification(),
        total_tokens=0,
    )
    assert "Feasible solves: 1 / Infeasible solves: 0" in user
    assert "Analysis+control steps (not solves): 2" in user
    assert "Infeasible: 2" not in user  # must never claim 2 infeasible


def test_narrative_labels_non_solve_entries():
    fc = _run_journal().format_for_classification()
    # analyze query is labeled and its returned result is shown
    assert "analysis query" in fc
    assert "scenario_voltage_spread" in fc
    assert "152, 154, 155" in fc
    # completion marker labeled, not a failure
    assert "search completed by LLM" in fc
    # solve accounting present and correct
    assert "Real solve iterations: 1 (feasible 1, infeasible 0)" in fc


def test_narrative_fallback_without_solve_keys():
    # A hand-built stats dict lacking solve keys falls back to legacy counts.
    stats = {"total_iterations": 5, "feasible_count": 3, "infeasible_count": 2,
             "best_objective": 100.0, "best_iteration": 1}
    _sys, user = build_classification_prompts(
        goal="g", termination_reason="done", stats=stats,
        journal_formatted="", total_tokens=0,
    )
    assert "Feasible solves: 3 / Infeasible solves: 2" in user


# ---------------------------------------------------------------------------
# Change 3 — charts exclude control entries
# ---------------------------------------------------------------------------

def test_convergence_chart_excludes_control_entries():
    fig = convergence_chart(_run_journal())
    # Exactly one solve marker; no "No data" placeholder.
    marker_pts = sum(len(t.x) for t in fig.data if t.mode and "markers" in t.mode)
    assert marker_pts == 1
    assert not any("No data" in (a.text or "") for a in fig.layout.annotations)


def test_voltage_range_chart_excludes_control_entries():
    fig = voltage_range_chart(_run_journal())
    assert not any("No voltage" in (a.text or "") for a in fig.layout.annotations)
    assert len(fig.data) > 0


def test_charts_solve_only_can_be_disabled():
    # Backward-compatible escape hatch.
    j = _run_journal()
    fig = convergence_chart(j, solve_only=False)
    assert fig is not None  # does not raise
