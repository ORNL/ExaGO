"""Tests for multi-objective tracking (Phase 2.3)."""

from __future__ import annotations

import json
import pytest

from agentigrid.engine.journal import (
    JournalEntry, SearchJournal, ObjectiveEntry, ObjectiveRegistry,
)
from agentigrid.engine.metric_extractor import extract_metric, extract_all_metrics, available_metrics
from agentigrid.engine.objective_parser import parse_objective_extraction
from agentigrid.parsers.opflow_results import OPFLOWResult, BusResult, BranchResult, GenResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_opflow_result():
    """Create a minimal OPFLOWResult for testing metric extraction."""
    return OPFLOWResult(
        converged=True,
        objective_value=50000.0,
        convergence_status="CONVERGED",
        solver="IPOPT",
        model="POWER_BALANCE_POLAR",
        objective_type="MIN_GEN_COST",
        num_iterations=25,
        solve_time=1.5,
        buses=[
            BusResult(1, 50, 0, 20, 0, 1.02, 0, 0, 0, 0, 0),
            BusResult(2, 80, 0, 30, 0, 0.95, -5, 0, 0, 0, 0),
            BusResult(3, 30, 0, 10, 0, 1.04, -2, 0, 0, 0, 0),
        ],
        branches=[
            BranchResult(1, 2, 1, 80, 78, 100, 0, 0),
            BranchResult(2, 3, 1, 30, 28, 50, 0, 0),
        ],
        generators=[
            GenResult(1, 1, "coal", 100, 20, 10, 200, -50, 50),
            GenResult(2, 1, "gas", 65, 15, 5, 150, -30, 30),
        ],
        total_gen_mw=165.0,
        total_load_mw=160.0,
        total_gen_mvar=35.0,
        total_load_mvar=60.0,
        voltage_min=0.95,
        voltage_max=1.04,
        voltage_mean=1.003,
        max_line_loading_pct=80.0,
        num_violations=0,
    )


# ── ObjectiveRegistry Tests ──────────────────────────────────────────────────

class TestObjectiveRegistry:
    def test_register_and_retrieve(self):
        registry = ObjectiveRegistry()
        obj = ObjectiveEntry(name="generation_cost", direction="minimize")
        registry.register(obj)
        assert len(registry.objectives) == 1
        assert registry.objectives[0].name == "generation_cost"

    def test_register_updates_existing(self):
        registry = ObjectiveRegistry()
        obj1 = ObjectiveEntry(name="cost", direction="minimize", priority="primary")
        obj2 = ObjectiveEntry(name="cost", direction="minimize", priority="secondary")
        registry.register(obj1)
        registry.register(obj2)
        assert len(registry.objectives) == 1
        assert registry.objectives[0].priority == "secondary"

    def test_is_multi_objective(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        assert not registry.is_multi_objective
        registry.register(ObjectiveEntry(name="voltage", direction="minimize", priority="secondary"))
        assert registry.is_multi_objective

    def test_watch_doesnt_count_as_multi(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        registry.register(ObjectiveEntry(name="loading", direction="minimize", priority="watch"))
        assert not registry.is_multi_objective

    def test_reprioritize(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        result = registry.reprioritize("cost", "secondary", iteration=5)
        assert result is True
        assert registry.objectives[0].priority == "secondary"
        assert len(registry.history) == 2  # register + reprioritize

    def test_reprioritize_nonexistent(self):
        registry = ObjectiveRegistry()
        result = registry.reprioritize("nonexistent", "secondary", iteration=1)
        assert result is False

    def test_format_for_prompt(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        registry.register(ObjectiveEntry(
            name="loading", direction="minimize", priority="secondary",
        ))
        text = registry.format_for_prompt()
        assert "Tracked Objectives:" in text
        assert "cost" in text
        assert "loading" in text
        assert "primary" in text
        assert "secondary" in text

    def test_format_for_prompt_constraint(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(
            name="voltage", direction="constraint", threshold=0.95, priority="primary",
        ))
        text = registry.format_for_prompt()
        assert "constraint" in text
        assert "0.95" in text

    def test_format_for_prompt_empty(self):
        registry = ObjectiveRegistry()
        assert registry.format_for_prompt() == ""

    def test_to_dict_list(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        data = registry.to_dict_list()
        assert len(data) == 1
        assert data[0]["name"] == "cost"
        assert data[0]["direction"] == "minimize"

    def test_history_recorded(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="cost", direction="minimize", priority="primary"))
        assert len(registry.history) == 1
        assert registry.history[0]["action"] == "registered"

    def test_get_primary_secondary(self):
        registry = ObjectiveRegistry()
        registry.register(ObjectiveEntry(name="a", direction="minimize", priority="primary"))
        registry.register(ObjectiveEntry(name="b", direction="minimize", priority="secondary"))
        registry.register(ObjectiveEntry(name="c", direction="minimize", priority="watch"))
        assert len(registry.get_primary()) == 1
        assert len(registry.get_secondary()) == 1
        assert registry.get_primary()[0].name == "a"


# ── SearchJournal with ObjectiveRegistry ────────────────────────────────────

class TestSearchJournalObjectiveRegistry:
    def test_journal_has_registry(self):
        journal = SearchJournal()
        assert hasattr(journal, "objective_registry")
        assert isinstance(journal.objective_registry, ObjectiveRegistry)

    def test_format_multi_objective_summary_empty(self):
        journal = SearchJournal()
        assert journal.format_multi_objective_summary() == ""

    def test_format_multi_objective_summary_no_tracked_metrics(self):
        journal = SearchJournal()
        journal.objective_registry.register(
            ObjectiveEntry(name="cost", direction="minimize", priority="primary")
        )
        # Add entries without tracked_metrics
        journal.add_from_results(
            iteration=1, description="test", commands=[],
            opflow_result=None, sim_elapsed=1.0, llm_reasoning="r", mode="fresh",
        )
        assert journal.format_multi_objective_summary() == ""

    def test_format_multi_objective_summary_with_data(self):
        journal = SearchJournal()
        journal.objective_registry.register(
            ObjectiveEntry(name="generation_cost", direction="minimize", priority="primary")
        )
        entry = journal.add_from_results(
            iteration=1, description="t", commands=[], opflow_result=None,
            sim_elapsed=1.0, llm_reasoning="r", mode="fresh",
        )
        entry.tracked_metrics = {"generation_cost": 50000.0}
        entry2 = journal.add_from_results(
            iteration=2, description="t2", commands=[], opflow_result=None,
            sim_elapsed=1.0, llm_reasoning="r", mode="fresh",
        )
        entry2.tracked_metrics = {"generation_cost": 45000.0}
        text = journal.format_multi_objective_summary()
        assert "Multi-Objective Tracking" in text
        assert "generation_cost" in text
        assert "50000" in text
        assert "45000" in text
        assert "Trends" in text

    def test_summary_stats_includes_registry(self):
        journal = SearchJournal()
        journal.objective_registry.register(
            ObjectiveEntry(name="cost", direction="minimize", priority="primary")
        )
        stats = journal.summary_stats()
        assert "objective_registry" in stats
        assert "is_multi_objective" in stats
        assert stats["is_multi_objective"] is False

    def test_export_json_includes_registry(self, tmp_path):
        journal = SearchJournal()
        journal.objective_registry.register(
            ObjectiveEntry(name="cost", direction="minimize", priority="primary")
        )
        path = tmp_path / "journal.json"
        journal.export_json(path)
        data = json.loads(path.read_text())
        assert "objective_registry" in data
        assert "preference_history" in data
        assert "entries" in data

    def test_export_csv_includes_tracked_metrics(self, tmp_path):
        journal = SearchJournal()
        entry = journal.add_from_results(
            iteration=1, description="t", commands=[], opflow_result=None,
            sim_elapsed=1.0, llm_reasoning="r", mode="fresh",
        )
        entry.tracked_metrics = {"cost": 1234.5}
        path = tmp_path / "journal.csv"
        journal.export_csv(path)
        content = path.read_text()
        assert "tracked_metrics" in content
        assert "1234.5" in content


# ── JournalEntry tracked_metrics ─────────────────────────────────────────────

class TestJournalEntryTrackedMetrics:
    def test_entry_tracked_metrics_default_none(self):
        entry = JournalEntry(
            iteration=1, description="t", commands=[],
            objective_value=100.0, feasible=True, convergence_status="OK",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=95.0,
            llm_reasoning="r", mode="fresh", elapsed_seconds=1.0,
        )
        assert entry.tracked_metrics is None

    def test_entry_tracked_metrics_settable(self):
        entry = JournalEntry(
            iteration=1, description="t", commands=[],
            objective_value=100.0, feasible=True, convergence_status="OK",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=95.0,
            llm_reasoning="r", mode="fresh", elapsed_seconds=1.0,
        )
        entry.tracked_metrics = {"cost": 50000.0, "voltage": 0.95}
        assert entry.tracked_metrics["cost"] == 50000.0


# ── MetricExtractor Tests ─────────────────────────────────────────────────────

class TestMetricExtractor:
    def test_available_metrics_nonempty(self):
        metrics = available_metrics()
        assert len(metrics) > 0
        assert "generation_cost" in metrics
        assert "voltage_min" in metrics
        assert "max_line_loading_pct" in metrics

    def test_extract_generation_cost(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "generation_cost")
        assert val == pytest.approx(50000.0)

    def test_extract_total_generation_mw(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "total_generation_mw")
        assert val == pytest.approx(165.0)

    def test_extract_active_losses_mw(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "active_losses_mw")
        assert val == pytest.approx(5.0)  # 165 - 160

    def test_extract_voltage_min(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "voltage_min")
        assert val == pytest.approx(0.95)

    def test_extract_voltage_max(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "voltage_max")
        assert val == pytest.approx(1.04)

    def test_extract_voltage_deviation(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "voltage_deviation")
        # buses: 1.02, 0.95, 1.04 → deviations from 1.0: 0.02, 0.05, 0.04 → max = 0.05
        assert val == pytest.approx(0.05)

    def test_extract_voltage_range(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "voltage_range")
        assert val == pytest.approx(1.04 - 0.95)

    def test_extract_max_line_loading_pct(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "max_line_loading_pct")
        assert val == pytest.approx(80.0)

    def test_extract_mean_line_loading_pct(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "mean_line_loading_pct")
        # branch 1: max(80,78)/100*100 = 80%, branch 2: max(30,28)/50*100 = 60%
        assert val == pytest.approx(70.0)

    def test_extract_violation_count(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "violation_count")
        assert val == pytest.approx(0.0)

    def test_extract_total_reactive_gen_mvar(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "total_reactive_gen_mvar")
        assert val == pytest.approx(35.0)

    def test_extract_online_generator_count(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "online_generator_count")
        assert val == pytest.approx(2.0)

    def test_extract_generation_reserve_mw(self, sample_opflow_result):
        # gen1: Pmax=200, Pg=100 → reserve=100
        # gen2: Pmax=150, Pg=65 → reserve=85
        val = extract_metric(sample_opflow_result, "generation_reserve_mw")
        assert val == pytest.approx(185.0)

    def test_extract_unknown_metric_returns_none(self, sample_opflow_result):
        val = extract_metric(sample_opflow_result, "nonexistent_metric_xyz")
        assert val is None

    def test_extract_all_metrics(self, sample_opflow_result):
        names = ["generation_cost", "voltage_min", "voltage_max"]
        result = extract_all_metrics(sample_opflow_result, names)
        assert "generation_cost" in result
        assert "voltage_min" in result
        assert "voltage_max" in result
        assert result["generation_cost"] == pytest.approx(50000.0)

    def test_extract_all_metrics_skips_unknown(self, sample_opflow_result):
        names = ["generation_cost", "unknown_metric"]
        result = extract_all_metrics(sample_opflow_result, names)
        assert "generation_cost" in result
        assert "unknown_metric" not in result


# ── ObjectiveParser Tests ─────────────────────────────────────────────────────

class TestObjectiveParser:
    def test_parse_valid_json_response(self):
        response = '{"objectives": [{"name": "generation_cost", "direction": "minimize", "threshold": null, "priority": "primary"}]}'
        result = parse_objective_extraction(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "generation_cost"
        assert result[0]["direction"] == "minimize"
        assert result[0]["priority"] == "primary"

    def test_parse_fenced_json_block(self):
        response = '```json\n{"objectives": [{"name": "voltage_min", "direction": "maximize", "threshold": null, "priority": "secondary"}]}\n```'
        result = parse_objective_extraction(response)
        assert result is not None
        assert result[0]["name"] == "voltage_min"

    def test_parse_multi_objective(self):
        response = json.dumps({
            "objectives": [
                {"name": "generation_cost", "direction": "minimize", "priority": "primary"},
                {"name": "max_line_loading_pct", "direction": "minimize", "priority": "secondary"},
            ]
        })
        result = parse_objective_extraction(response)
        assert result is not None
        assert len(result) == 2

    def test_parse_constraint_with_threshold(self):
        response = json.dumps({
            "objectives": [
                {"name": "voltage_min", "direction": "constraint", "threshold": 0.95, "priority": "primary"},
            ]
        })
        result = parse_objective_extraction(response)
        assert result is not None
        assert result[0]["threshold"] == 0.95
        assert result[0]["direction"] == "constraint"

    def test_parse_invalid_direction_defaults_minimize(self):
        response = json.dumps({
            "objectives": [
                {"name": "cost", "direction": "wrong_direction", "priority": "primary"},
            ]
        })
        result = parse_objective_extraction(response)
        assert result is not None
        assert result[0]["direction"] == "minimize"

    def test_parse_invalid_priority_defaults_primary(self):
        response = json.dumps({
            "objectives": [
                {"name": "cost", "direction": "minimize", "priority": "invalid_priority"},
            ]
        })
        result = parse_objective_extraction(response)
        assert result is not None
        assert result[0]["priority"] == "primary"

    def test_parse_empty_objectives_returns_none(self):
        response = json.dumps({"objectives": []})
        result = parse_objective_extraction(response)
        assert result is None

    def test_parse_invalid_json_returns_none(self):
        result = parse_objective_extraction("this is not json at all")
        assert result is None

    def test_parse_objectives_without_name_skipped(self):
        response = json.dumps({
            "objectives": [
                {"direction": "minimize", "priority": "primary"},  # no name
                {"name": "cost", "direction": "minimize", "priority": "primary"},
            ]
        })
        result = parse_objective_extraction(response)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "cost"
