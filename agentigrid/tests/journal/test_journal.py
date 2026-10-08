"""Tests for the Search Journal."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agentigrid.engine.journal import JournalEntry, SearchJournal
from agentigrid.parsers.opflow_results import OPFLOWResult


def _make_entry(iteration: int, obj: float = 1000.0, feasible: bool = True, **kw) -> JournalEntry:
    """Helper to create a JournalEntry with sensible defaults."""
    defaults = dict(
        iteration=iteration,
        description=f"Iteration {iteration} changes",
        commands=[{"command": "scale_all_loads", "scale_factor": 1.0 + iteration * 0.1}],
        objective_value=obj if feasible else None,
        feasible=feasible,
        convergence_status="CONVERGED" if feasible else "DIVERGED",
        violations_count=0,
        voltage_min=1.05,
        voltage_max=1.10,
        max_line_loading_pct=50.0 + iteration,
        total_gen_mw=500.0,
        total_load_mw=480.0,
        llm_reasoning="Test reasoning",
        mode="accumulative",
        elapsed_seconds=0.5,
        timestamp="2026-01-01T00:00:00",
    )
    defaults.update(kw)
    return JournalEntry(**defaults)


def _make_opflow_result(**kw) -> OPFLOWResult:
    """Helper to create an OPFLOWResult with sensible defaults."""
    defaults = dict(
        converged=True,
        objective_value=27557.57,
        convergence_status="CONVERGED",
        solver="IPOPT",
        model="POWER_BALANCE_POLAR",
        objective_type="MIN_GEN_COST",
        num_iterations=23,
        solve_time=0.042,
        total_gen_mw=520.0,
        total_load_mw=490.0,
        total_gen_mvar=100.0,
        total_load_mvar=80.0,
        voltage_min=1.062,
        voltage_max=1.100,
        voltage_mean=1.080,
        max_line_loading_pct=52.3,
        num_violations=0,
    )
    defaults.update(kw)
    return OPFLOWResult(**defaults)


# ===========================================================================
# Basic add/access
# ===========================================================================

class TestBasicOperations:

    def test_empty_journal(self):
        j = SearchJournal()
        assert len(j) == 0
        assert j.latest is None
        assert j.entries == []

    def test_add_entry(self):
        j = SearchJournal()
        e1 = _make_entry(1)
        e2 = _make_entry(2, obj=900.0)
        j.add_entry(e1)
        j.add_entry(e2)
        assert len(j) == 2
        assert j.latest is e2
        assert j.entries[0] is e1

    def test_entries_returns_copy(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1))
        entries = j.entries
        entries.clear()
        assert len(j) == 1


# ===========================================================================
# add_from_results
# ===========================================================================

class TestAddFromResults:

    def test_with_opflow_result(self):
        j = SearchJournal()
        opf = _make_opflow_result(objective_value=28000.0, voltage_min=1.05)
        entry = j.add_from_results(
            iteration=1,
            description="Base case",
            commands=[],
            opflow_result=opf,
            sim_elapsed=1.2,
            llm_reasoning="No changes",
            mode="fresh",
        )
        assert len(j) == 1
        assert entry.feasible is True
        assert entry.objective_value == 28000.0
        assert entry.voltage_min == 1.05
        assert entry.convergence_status == "CONVERGED"
        assert entry.mode == "fresh"

    def test_with_none_result(self):
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=2,
            description="Failed sim",
            commands=[{"command": "scale_all_loads", "scale_factor": 5.0}],
            opflow_result=None,
            sim_elapsed=0.1,
            llm_reasoning="Tried aggressive scaling",
            mode="accumulative",
        )
        assert entry.feasible is False
        assert entry.objective_value is None
        assert entry.convergence_status == "FAILED"
        assert entry.voltage_min == 0.0
        assert entry.voltage_max == 0.0

    def test_power_balance_violation_marks_infeasible(self):
        """Converged result with negative losses (gen < load) should be infeasible."""
        j = SearchJournal()
        opf = _make_opflow_result(
            converged=True,
            total_gen_mw=500.0,
            total_load_mw=1000.0,
            losses_mw=-500.0,
        )
        entry = j.add_from_results(
            iteration=1,
            description="Unphysical solution",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="Test",
            mode="fresh",
        )
        assert entry.feasible is False

    def test_positive_losses_still_feasible(self):
        """Converged result with positive losses should remain feasible."""
        j = SearchJournal()
        opf = _make_opflow_result(
            converged=True,
            total_gen_mw=520.0,
            total_load_mw=490.0,
            losses_mw=30.0,
        )
        entry = j.add_from_results(
            iteration=1,
            description="Good solution",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="Test",
            mode="fresh",
        )
        assert entry.feasible is True

    def test_zero_load_no_power_balance_violation(self):
        """Zero load should not cause a power balance violation flag."""
        j = SearchJournal()
        opf = _make_opflow_result(
            converged=True,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            losses_mw=0.0,
        )
        entry = j.add_from_results(
            iteration=1,
            description="Zero load",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="Test",
            mode="fresh",
        )
        assert entry.feasible is True

    def test_marginal_feasibility_detail(self):
        """DID NOT CONVERGE with marginal exit should be feasible=False, detail=marginal."""
        j = SearchJournal()
        opf = _make_opflow_result(
            converged=False,
            convergence_status="DID NOT CONVERGE",
            feasibility_detail="marginal",
        )
        entry = j.add_from_results(
            iteration=1,
            description="Marginal convergence",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="Test",
            mode="fresh",
        )
        assert entry.feasible is False
        assert entry.feasibility_detail == "marginal"

    def test_infeasible_power_balance(self):
        """Power balance violation should make feasible=False, detail=infeasible."""
        j = SearchJournal()
        opf = _make_opflow_result(
            converged=False,
            convergence_status="CONVERGED",
            total_gen_mw=500.0,
            total_load_mw=1000.0,
            losses_mw=-500.0,
            feasibility_detail="infeasible",
        )
        entry = j.add_from_results(
            iteration=1,
            description="Power balance violation",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="Test",
            mode="fresh",
        )
        assert entry.feasible is False
        assert entry.feasibility_detail == "infeasible"


# ===========================================================================
# add_from_results — PFLOW gencost-derived cost
# ===========================================================================

class TestAddFromResultsPFLOWCost:
    """PFLOW does not produce an objective value; cost must be computed
    from gencost and the dispatch. The entry's objective_value and
    tracked_metrics["generation_cost"] must reflect the computed value
    rather than the misleading 0.0 sentinel."""

    def _pflow_result(self, gen_pgs: list[float]) -> OPFLOWResult:
        from agentigrid.parsers.opflow_results import GenResult
        gens = [
            GenResult(
                bus=i + 1, status=1, fuel="COAL", Pg=pg, Qg=0.0,
                Pmin=0.0, Pmax=500.0, Qmin=-300.0, Qmax=300.0,
            )
            for i, pg in enumerate(gen_pgs)
        ]
        return _make_opflow_result(
            objective_value=0.0,
            solver="POWER_FLOW",
            objective_type="",
            generators=gens,
            feasibility_detail="feasible",
        )

    def _gencost(self, tuples: list[tuple[float, float, float]]):
        from agentigrid.parsers.matpower_model import GenCost
        return [
            GenCost(model=2, startup=0.0, shutdown=0.0, ncost=3, coeffs=list(t))
            for t in tuples
        ]

    def test_pflow_cost_populates_objective_value(self):
        """When gencost is supplied, objective_value reflects computed cost."""
        opf = self._pflow_result([100.0, 200.0])
        gc = self._gencost([(0.01, 5.0, 100.0), (0.02, 4.0, 200.0)])
        # 0.01*100^2 + 5*100 + 100 = 100 + 500 + 100 = 700
        # 0.02*200^2 + 4*200 + 200 = 800 + 800 + 200 = 1800
        # Total = 2500
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=1,
            description="PFLOW base",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="",
            mode="fresh",
            gencost=gc,
        )
        assert entry.objective_value == pytest.approx(2500.0)

    def test_pflow_cost_populates_tracked_metrics(self):
        """tracked_metrics['generation_cost'] must equal computed cost."""
        opf = self._pflow_result([100.0])
        gc = self._gencost([(0.01, 5.0, 100.0)])
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=1,
            description="PFLOW",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="",
            mode="fresh",
            gencost=gc,
        )
        assert entry.tracked_metrics is not None
        assert entry.tracked_metrics["generation_cost"] == pytest.approx(700.0)
        assert entry.tracked_metrics["generation_cost"] == entry.objective_value

    def test_pflow_no_gencost_falls_through_to_zero(self):
        """Without gencost, objective_value is the OPFLOWResult's own field."""
        opf = self._pflow_result([100.0])
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=1,
            description="PFLOW no gencost",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="",
            mode="fresh",
        )
        # Without gencost, falls through to opflow_result.objective_value (0.0)
        assert entry.objective_value == 0.0
        # tracked_metrics is not auto-populated without gencost
        assert entry.tracked_metrics is None

    def test_pflow_offline_generators_yield_none(self):
        """If all generators are offline, computed cost is 0.0 → entry sees None."""
        from agentigrid.parsers.opflow_results import GenResult
        opf = _make_opflow_result(
            objective_value=0.0,
            solver="POWER_FLOW",
            generators=[
                GenResult(
                    bus=1, status=0, fuel="COAL", Pg=0.0, Qg=0.0,
                    Pmin=0.0, Pmax=500.0, Qmin=-300.0, Qmax=300.0,
                ),
            ],
        )
        gc = self._gencost([(0.01, 5.0, 100.0)])
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=1,
            description="PFLOW all-offline",
            commands=[],
            opflow_result=opf,
            sim_elapsed=0.1,
            llm_reasoning="",
            mode="fresh",
            gencost=gc,
        )
        # compute_generation_cost returns 0.0 → treated as None
        assert entry.objective_value is None
        assert entry.tracked_metrics is None


# ===========================================================================
# format_for_prompt
# ===========================================================================

class TestFormatForPrompt:

    def test_empty(self):
        j = SearchJournal()
        text = j.format_for_prompt()
        assert "no iterations" in text

    def test_basic_format(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=27557.57))
        j.add_entry(_make_entry(2, obj=30000.00))
        j.add_entry(_make_entry(3, feasible=False))

        text = j.format_for_prompt()
        assert "3 iterations" in text
        assert "27,557.57" in text
        assert "30,000.00" in text
        assert "N/A" in text
        assert "Yes" in text
        assert "No" in text

    def test_format_alignment(self):
        """All rows should have the same pipe positions."""
        j = SearchJournal()
        for i in range(1, 4):
            j.add_entry(_make_entry(i, obj=1000.0 * i))

        text = j.format_for_prompt()
        lines = text.strip().split("\n")
        # Data rows start after header (line 0), blank (line 1), column header (line 2), separator (line 3)
        data_lines = [l for l in lines[4:] if l.strip()]
        for line in data_lines:
            assert line.count("|") == 5

    def test_max_entries(self):
        j = SearchJournal()
        for i in range(1, 11):
            j.add_entry(_make_entry(i, obj=1000.0 * i))

        text = j.format_for_prompt(max_entries=5)
        # Should include entry 1
        assert "   1 |" in text
        # Should have ellipsis
        assert "..." in text
        # Should include entries 7-10
        assert "   7 |" in text
        assert "  10 |" in text
        # Should NOT include entry 3
        assert "   3 |" not in text

    def test_max_entries_larger_than_total(self):
        j = SearchJournal()
        for i in range(1, 4):
            j.add_entry(_make_entry(i))

        text = j.format_for_prompt(max_entries=10)
        assert "..." not in text


# ===========================================================================
# format_detailed
# ===========================================================================

class TestFormatDetailed:

    def test_empty(self):
        j = SearchJournal()
        assert "empty" in j.format_detailed()

    def test_detailed_content(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=27000.0))
        text = j.format_detailed()
        assert "Iteration 1" in text
        assert "27,000.00" in text
        assert "CONVERGED" in text
        assert "scale_all_loads" in text


# ===========================================================================
# Export
# ===========================================================================

class TestExportJSON:

    def test_export_json(self, tmp_path: Path):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))
        j.add_entry(_make_entry(2, obj=2000.0))

        out = tmp_path / "journal.json"
        j.export_json(out)

        data = json.loads(out.read_text())
        assert "entries" in data
        assert len(data["entries"]) == 2
        assert data["entries"][0]["iteration"] == 1
        assert data["entries"][0]["objective_value"] == 1000.0
        assert data["entries"][1]["iteration"] == 2
        assert isinstance(data["entries"][0]["commands"], list)

    def test_export_json_empty(self, tmp_path: Path):
        j = SearchJournal()
        out = tmp_path / "journal.json"
        j.export_json(out)
        data = json.loads(out.read_text())
        assert data["entries"] == []
        assert "objective_registry" in data

    def test_export_json_includes_session_best(self, tmp_path: Path):
        """session_best is included as a top-level key in the JSON."""
        j = SearchJournal()
        j.update_session_best(label="A", iteration=3, cost=29924.90, commands=[{"action": "scale_all_loads", "factor": 1.2}])
        out = tmp_path / "journal.json"
        j.export_json(out)
        data = json.loads(out.read_text())
        assert "session_best" in data
        assert data["session_best"]["cost"] == pytest.approx(29924.90)
        assert data["session_best"]["iteration"] == 3
        assert data["session_best"]["variant_label"] == "A"

    def test_export_json_no_session_best_when_absent(self, tmp_path: Path):
        """session_best key is absent when no session best has been set."""
        j = SearchJournal()
        out = tmp_path / "journal.json"
        j.export_json(out)
        data = json.loads(out.read_text())
        assert "session_best" not in data


# ===========================================================================
# session_best tracking (Task 3)
# ===========================================================================

class TestSessionBest:

    def test_first_update_sets_best(self):
        j = SearchJournal()
        j.update_session_best("A", 1, 30000.0, [])
        assert j.session_best is not None
        assert j.session_best["cost"] == pytest.approx(30000.0)
        assert j.session_best["iteration"] == 1
        assert j.session_best["variant_label"] == "A"

    def test_cheaper_variant_replaces_best(self):
        j = SearchJournal()
        j.update_session_best("A", 1, 30000.0, [])
        j.update_session_best("B", 2, 29000.0, [])
        assert j.session_best["cost"] == pytest.approx(29000.0)
        assert j.session_best["variant_label"] == "B"

    def test_more_expensive_does_not_replace(self):
        j = SearchJournal()
        j.update_session_best("A", 1, 29000.0, [])
        j.update_session_best("B", 2, 31000.0, [])
        assert j.session_best["cost"] == pytest.approx(29000.0)
        assert j.session_best["variant_label"] == "A"

    def test_zero_cost_not_recorded(self):
        """0.0 is a sentinel for 'unknown cost', must not be stored."""
        j = SearchJournal()
        j.update_session_best("A", 1, 0.0, [])
        assert j.session_best is None

    def test_commands_stored_on_best(self):
        cmds = [{"action": "scale_all_loads", "factor": 1.2}]
        j = SearchJournal()
        j.update_session_best("A", 1, 5000.0, cmds)
        assert j.session_best["commands"] == cmds


# ===========================================================================
# session_best in user prompt
# ===========================================================================

class TestSessionBestInPrompt:

    def test_format_session_best_contains_cost(self):
        from agentigrid.prompts.user_prompt import _format_session_best
        sb = {"cost": 29924.90, "iteration": 5, "variant_label": "A", "commands": []}
        text = _format_session_best(sb)
        assert "29,924.90" in text
        assert "iter 5" in text
        assert "variant A" in text

    def test_session_best_injected_in_user_prompt(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        sb = {"cost": 12345.67, "iteration": 3, "variant_label": "B",
              "commands": [{"action": "scale_all_loads", "factor": 1.1}]}
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            session_best=sb,
        )
        assert "12,345.67" in prompt
        assert "Session best" in prompt

    def test_no_session_best_section_when_absent(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            session_best=None,
        )
        assert "Session best" not in prompt


class TestExportCSV:

    def test_export_csv(self, tmp_path: Path):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))
        j.add_entry(_make_entry(2, feasible=False))

        out = tmp_path / "journal.csv"
        j.export_csv(out)

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["iteration"] == "1"
        assert rows[0]["feasible"] == "True"
        # commands column should be valid JSON
        cmds = json.loads(rows[0]["commands"])
        assert isinstance(cmds, list)

    def test_export_csv_empty(self, tmp_path: Path):
        j = SearchJournal()
        out = tmp_path / "journal.csv"
        j.export_csv(out)
        assert out.read_text() == ""


# ===========================================================================
# Summary statistics
# ===========================================================================

class TestSummaryStats:

    def test_empty(self):
        j = SearchJournal()
        stats = j.summary_stats()
        assert stats["total_iterations"] == 0
        assert stats["best_objective"] is None
        assert stats["best_iteration"] is None
        assert stats["feasible_count"] == 0

    def test_all_feasible(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=3000.0))
        j.add_entry(_make_entry(2, obj=2000.0))
        j.add_entry(_make_entry(3, obj=2500.0))

        stats = j.summary_stats()
        assert stats["total_iterations"] == 3
        assert stats["best_objective"] == 2000.0
        assert stats["best_iteration"] == 2
        assert stats["feasible_count"] == 3
        assert stats["infeasible_count"] == 0
        assert stats["objective_trend"] == [3000.0, 2000.0, 2500.0]
        assert len(stats["voltage_range_trend"]) == 3

    def test_mixed_feasibility(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=5000.0))
        j.add_entry(_make_entry(2, feasible=False))
        j.add_entry(_make_entry(3, obj=3000.0))

        stats = j.summary_stats()
        assert stats["best_objective"] == 3000.0
        assert stats["best_iteration"] == 3
        assert stats["feasible_count"] == 2
        assert stats["infeasible_count"] == 1

    def test_all_infeasible(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, feasible=False))
        j.add_entry(_make_entry(2, feasible=False))

        stats = j.summary_stats()
        assert stats["best_objective"] is None
        assert stats["best_iteration"] is None
        assert stats["feasible_count"] == 0
        assert stats["infeasible_count"] == 2

    def test_goal_type_included(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))

        stats = j.summary_stats(goal_type="feasibility_boundary")
        assert stats["goal_type"] == "feasibility_boundary"

        stats_none = j.summary_stats()
        assert stats_none["goal_type"] is None

    def test_best_iteration_override(self):
        """Override should select a specific iteration as best, not lowest cost."""
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=27000.0))  # base case, lowest cost
        j.add_entry(_make_entry(1, obj=30000.0))
        j.add_entry(_make_entry(2, obj=35000.0))
        j.add_entry(_make_entry(3, obj=43000.0))  # highest cost but best for boundary search

        # Default: lowest cost wins
        stats_default = j.summary_stats()
        assert stats_default["best_iteration"] == 0
        assert stats_default["best_objective"] == 27000.0

        # Override: iteration 3 is the best answer
        stats_override = j.summary_stats(best_iteration_override=3)
        assert stats_override["best_iteration"] == 3
        assert stats_override["best_objective"] == 43000.0

    def test_best_iteration_override_invalid(self):
        """Invalid override iteration falls back to default cost heuristic."""
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=5000.0))
        j.add_entry(_make_entry(2, obj=3000.0))

        stats = j.summary_stats(best_iteration_override=99)
        # Should fall back to default (lowest cost)
        assert stats["best_iteration"] == 2
        assert stats["best_objective"] == 3000.0

    def test_best_iteration_override_infeasible_entry(self):
        """Override can select an infeasible entry (for boundary searches)."""
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=27000.0))
        j.add_entry(_make_entry(1, obj=35000.0))
        j.add_entry(_make_entry(2, feasible=False))  # infeasible

        # Override selects the infeasible entry
        stats = j.summary_stats(best_iteration_override=2)
        assert stats["best_iteration"] == 2


class TestAddAnalysis:
    def test_add_analysis_creates_entry(self):
        j = SearchJournal()
        entry = j.add_analysis(iteration=1, query="voltage profile", result_summary="Vmin=0.95pu")
        assert entry.iteration == 1
        assert entry.convergence_status == "ANALYSIS"
        assert entry.feasible is False
        assert entry.mode == "analyze"
        assert entry.objective_value is None
        assert "voltage profile" in entry.description

    def test_add_analysis_appears_in_entries(self):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_analysis(iteration=1, query="bus voltages", result_summary="OK")
        assert len(j.entries) == 2
        assert j.entries[1].convergence_status == "ANALYSIS"

    def test_add_analysis_in_format_detailed(self):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_analysis(iteration=1, query="bus voltages", result_summary="Vmin=0.95pu")
        detailed = j.format_detailed()
        assert "ANALYSIS" in detailed

    def test_add_analysis_in_csv_export(self, tmp_path):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_analysis(iteration=1, query="bus voltages", result_summary="OK")
        csv_path = tmp_path / "journal.csv"
        j.export_csv(csv_path)
        content = csv_path.read_text()
        assert "ANALYSIS" in content


class TestAddComplete:
    def test_add_complete_creates_entry(self):
        j = SearchJournal()
        entry = j.add_complete(iteration=3, summary="Search completed: system is feasible")
        assert entry.iteration == 3
        assert entry.convergence_status == "COMPLETE"
        assert entry.feasible is False
        assert entry.mode == "complete"
        assert entry.objective_value is None
        assert "completed" in entry.description.lower()

    def test_add_complete_appears_in_entries(self):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_entry(_make_entry(1, obj=900.0))
        j.add_complete(iteration=2, summary="Done")
        assert len(j.entries) == 3
        assert j.entries[2].convergence_status == "COMPLETE"

    def test_add_complete_in_format_detailed(self):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_complete(iteration=1, summary="Search completed")
        detailed = j.format_detailed()
        assert "COMPLETE" in detailed

    def test_add_complete_in_csv_export(self, tmp_path):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=1000.0))
        j.add_complete(iteration=1, summary="Done")
        csv_path = tmp_path / "journal.csv"
        j.export_csv(csv_path)
        content = csv_path.read_text()
        assert "COMPLETE" in content

    def test_add_complete_preserves_reasoning(self):
        j = SearchJournal()
        entry = j.add_complete(
            iteration=2,
            summary="System feasible with 1.5x wind scaling"
        )
        assert entry.llm_reasoning == "System feasible with 1.5x wind scaling"

    def test_add_complete_after_modifications(self):
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=3500.0))
        j.add_entry(_make_entry(1, obj=5000.0))
        j.add_complete(iteration=2, summary="Done")
        assert len(j.entries) == 3
        assert j.entries[0].convergence_status == "CONVERGED"
        assert j.entries[1].convergence_status == "CONVERGED"
        assert j.entries[2].convergence_status == "COMPLETE"


# ===========================================================================
# Prompt C: Feasibility flag fix (violations_count > 0 must flip feasible)
# ===========================================================================

class TestFeasibilityFlagFix:

    def test_feasible_with_no_violations(self):
        """feasibility_detail='feasible' + num_violations=0 → feasible=True."""
        j = SearchJournal()
        opf = _make_opflow_result(feasibility_detail="feasible", num_violations=0)
        entry = j.add_from_results(
            iteration=1, description="ok", commands=[], opflow_result=opf,
            sim_elapsed=0.1, llm_reasoning="", mode="fresh",
        )
        assert entry.feasible is True

    def test_infeasible_when_violations_present(self):
        """feasibility_detail='feasible' + num_violations=4 → feasible=False.

        This is the bug that was confirmed: thermal violations set
        violations_count > 0 but PFLOW still reports feasibility_detail='feasible'.
        """
        j = SearchJournal()
        opf = _make_opflow_result(
            feasibility_detail="feasible",
            num_violations=4,
            max_line_loading_pct=125.0,
        )
        entry = j.add_from_results(
            iteration=2, description="violations", commands=[], opflow_result=opf,
            sim_elapsed=0.1, llm_reasoning="", mode="fresh",
        )
        assert entry.feasible is False

    def test_infeasible_detail_stays_infeasible(self):
        """feasibility_detail='infeasible' → feasible=False regardless."""
        j = SearchJournal()
        opf = _make_opflow_result(feasibility_detail="infeasible", num_violations=0)
        entry = j.add_from_results(
            iteration=3, description="inf", commands=[], opflow_result=opf,
            sim_elapsed=0.1, llm_reasoning="", mode="fresh",
        )
        assert entry.feasible is False

    def test_marginal_detail_not_feasible(self):
        """feasibility_detail='marginal' → feasible=False."""
        j = SearchJournal()
        opf = _make_opflow_result(feasibility_detail="marginal", num_violations=0)
        entry = j.add_from_results(
            iteration=4, description="marginal", commands=[], opflow_result=opf,
            sim_elapsed=0.1, llm_reasoning="", mode="fresh",
        )
        assert entry.feasible is False


# ===========================================================================
# Prompt C: load_factor in journal
# ===========================================================================

class TestLoadFactorInJournal:

    def test_load_factor_default_none(self):
        j = SearchJournal()
        assert j.load_factor is None

    def test_load_factor_in_json_export(self, tmp_path):
        j = SearchJournal()
        j.load_factor = 1.23
        path = tmp_path / "journal.json"
        j.export_json(path)
        data = json.loads(path.read_text())
        assert data["load_factor"] == pytest.approx(1.23)

    def test_load_factor_absent_when_none(self, tmp_path):
        j = SearchJournal()
        path = tmp_path / "journal.json"
        j.export_json(path)
        data = json.loads(path.read_text())
        assert "load_factor" not in data

    def test_load_factor_update(self):
        j = SearchJournal()
        j.load_factor = 1.0
        j.load_factor = 1.35
        assert j.load_factor == pytest.approx(1.35)


# ===========================================================================
# Prompt C: iteration budget in user prompt
# ===========================================================================

class TestIterationBudgetInPrompt:
    """Tests for iteration counter and budget warning in build_user_prompt."""

    def test_iteration_counter_with_budget(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            current_iteration=5,
            max_iterations=20,
        )
        assert "Iteration: 5 / 20" in prompt
        assert "remaining: 15" in prompt

    def test_iteration_counter_no_budget(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            current_iteration=7,
            max_iterations=None,
        )
        assert "Iteration: 7" in prompt
        assert "no budget limit" in prompt

    def test_budget_warning_when_remaining_2(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            current_iteration=18,
            max_iterations=20,
        )
        assert "2 iteration(s) remaining" in prompt
        assert "Prioritize consolidation" in prompt

    def test_budget_warning_absent_when_remaining_10(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            current_iteration=10,
            max_iterations=20,
        )
        assert "Prioritize consolidation" not in prompt

    def test_budget_warning_absent_when_no_budget(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        prompt = build_user_prompt(
            goal="test",
            journal_text=None,
            results_text=None,
            current_iteration=19,
            max_iterations=None,
        )
        assert "Prioritize consolidation" not in prompt


# ===========================================================================
# Prompt C: format_benchmark_for_prompt
# ===========================================================================

class TestFormatBenchmarkForPrompt:

    REAL_BENCHMARK = {
        "opflow_converged": True,
        "opflow_objective": 27557.57,
        "pflow_best_computed_cost": 27564.26,
        "cost_gap_pct": 0.02,
        "cost_gap_abs": 6.69,
        "dispatch_comparison": [
            {"bus": 189, "fuel": "COAL", "opflow_pg": 383.4, "pflow_pg": 736.33,
             "delta": 352.93, "opflow_pmax": 569.15},
            {"bus": 100, "fuel": "GAS",  "opflow_pg": 200.0, "pflow_pg": 210.0,
             "delta": 10.0,  "opflow_pmax": 300.0},
        ],
        "loadability": {
            "opflow_max_factor": 1.6289,
            "pflow_max_factor": 1.35,
            "gap_pct": -17.12,
        },
    }

    def test_real_benchmark_contains_bus189(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        text = format_benchmark_for_prompt(self.REAL_BENCHMARK)
        assert "189" in text

    def test_real_benchmark_contains_delta(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        text = format_benchmark_for_prompt(self.REAL_BENCHMARK)
        assert "352" in text

    def test_real_benchmark_over_dispatched_note(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        text = format_benchmark_for_prompt(self.REAL_BENCHMARK)
        assert "over-dispatched" in text

    def test_not_converged_returns_unavailable(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        text = format_benchmark_for_prompt({"opflow_converged": False})
        assert text == "Benchmark unavailable."

    def test_none_returns_unavailable(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        text = format_benchmark_for_prompt(None)
        assert text == "Benchmark unavailable."

    def test_negligible_cost_gap_note(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        bench = {
            "opflow_converged": True,
            "opflow_objective": 10000.0,
            "pflow_best_computed_cost": 10000.5,
            "cost_gap_pct": 0.005,
            "cost_gap_abs": 0.5,
            "dispatch_comparison": [],
        }
        text = format_benchmark_for_prompt(bench)
        assert "negligible" in text

    def test_large_gap_no_negligible_note(self):
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
        bench = {
            "opflow_converged": True,
            "opflow_objective": 10000.0,
            "pflow_best_computed_cost": 10500.0,
            "cost_gap_pct": 5.0,
            "cost_gap_abs": 500.0,
            "dispatch_comparison": [],
        }
        text = format_benchmark_for_prompt(bench)
        assert "negligible" not in text
