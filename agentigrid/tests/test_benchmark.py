"""Tests for PFLOW vs OPFLOW benchmark (Phase 4, Step 4.4)."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from agentigrid.engine.benchmark import (
    BenchmarkResult,
    DispatchComparison,
    LoadabilityResult,
    run_pflow_vs_opflow_benchmark,
    _build_dispatch_comparison,
    _format_benchmark_summary,
    _extract_pflow_max_factor,
)
from agentigrid.engine.journal import SearchJournal, JournalEntry
from agentigrid.parsers.opflow_results import OPFLOWResult, GenResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def opflow_result_converged():
    return OPFLOWResult(
        converged=True,
        objective_value=27557.57,
        convergence_status="CONVERGED",
        solver="IPOPT",
        model="AC",
        objective_type="MIN_GEN_COST",
        num_iterations=12,
        solve_time=0.02,
        buses=[],
        branches=[],
        generators=[
            GenResult(bus=1, status=1, fuel="COAL", Pg=100.0, Qg=10.0,
                      Pmin=10.0, Pmax=150.0, Qmin=-50.0, Qmax=50.0),
            GenResult(bus=2, status=1, fuel="GAS", Pg=80.0, Qg=5.0,
                      Pmin=5.0, Pmax=120.0, Qmin=-30.0, Qmax=30.0),
            GenResult(bus=3, status=1, fuel="WIND", Pg=90.0, Qg=0.0,
                      Pmin=0.0, Pmax=200.0, Qmin=-10.0, Qmax=10.0),
        ],
        total_gen_mw=270.0,
        total_load_mw=265.0,
        total_gen_mvar=15.0,
        total_load_mvar=10.0,
        voltage_min=0.99,
        voltage_max=1.05,
        voltage_mean=1.02,
        max_line_loading_pct=65.0,
        num_violations=0,
        violation_details=[],
        losses_mw=5.0,
        power_balance_mismatch_pct=1.9,
        ipopt_exit_status="Optimal Solution Found.",
        feasibility_detail="feasible",
    )


@pytest.fixture
def pflow_result_feasible():
    return OPFLOWResult(
        converged=True,
        objective_value=0.0,
        convergence_status="CONVERGED",
        solver="Newton-Rhapson",
        model="AC",
        objective_type="PowerFlow",
        num_iterations=3,
        solve_time=0.005,
        buses=[],
        branches=[],
        generators=[
            GenResult(bus=1, status=1, fuel="COAL", Pg=110.0, Qg=12.0,
                      Pmin=10.0, Pmax=150.0, Qmin=-50.0, Qmax=50.0),
            GenResult(bus=2, status=1, fuel="GAS", Pg=70.0, Qg=3.0,
                      Pmin=5.0, Pmax=120.0, Qmin=-30.0, Qmax=30.0),
            GenResult(bus=3, status=1, fuel="WIND", Pg=95.0, Qg=0.0,
                      Pmin=0.0, Pmax=200.0, Qmin=-10.0, Qmax=10.0),
        ],
        total_gen_mw=275.0,
        total_load_mw=265.0,
        total_gen_mvar=15.0,
        total_load_mvar=10.0,
        voltage_min=0.98,
        voltage_max=1.04,
        voltage_mean=1.01,
        max_line_loading_pct=68.0,
        num_violations=0,
        violation_details=[],
        losses_mw=10.0,
        power_balance_mismatch_pct=3.8,
        ipopt_exit_status="",
        feasibility_detail="feasible",
    )


@pytest.fixture
def pflow_journal_with_scaling():
    journal = SearchJournal()
    journal.add_entry(JournalEntry(
        iteration=0,
        description="Base case",
        commands=[],
        objective_value=0.0,
        feasible=True,
        convergence_status="CONVERGED",
        violations_count=0,
        voltage_min=1.001,
        voltage_max=1.04,
        max_line_loading_pct=65.0,
        total_gen_mw=269.0,
        total_load_mw=265.0,
        llm_reasoning="Base case",
        mode="fresh",
        elapsed_seconds=0.02,
    ))
    journal.add_entry(JournalEntry(
        iteration=1,
        description="Test factor 1.5",
        commands=[{"action": "scale_all_loads", "factor": 1.5}],
        objective_value=0.0,
        feasible=True,
        convergence_status="CONVERGED",
        violations_count=0,
        voltage_min=0.968,
        voltage_max=1.04,
        max_line_loading_pct=66.0,
        total_gen_mw=402.0,
        total_load_mw=397.5,
        llm_reasoning="Binary search lower bound",
        mode="fresh",
        elapsed_seconds=0.02,
        feasibility_detail="feasible",
        solver="Newton-Rhapson",
    ))
    journal.add_entry(JournalEntry(
        iteration=2,
        description="Test factor 2.0",
        commands=[{"action": "scale_all_loads", "factor": 2.0}],
        objective_value=0.0,
        feasible=False,
        convergence_status="CONVERGED",
        violations_count=5,
        voltage_min=0.916,
        voltage_max=1.04,
        max_line_loading_pct=85.0,
        total_gen_mw=541.0,
        total_load_mw=530.0,
        llm_reasoning="Binary search upper bound",
        mode="fresh",
        elapsed_seconds=0.02,
        feasibility_detail="infeasible",
        solver="Newton-Rhapson",
    ))
    journal.add_entry(JournalEntry(
        iteration=3,
        description="Test factor 1.75",
        commands=[{"action": "scale_all_loads", "factor": 1.75}],
        objective_value=0.0,
        feasible=False,
        convergence_status="CONVERGED",
        violations_count=3,
        voltage_min=0.945,
        voltage_max=1.04,
        max_line_loading_pct=68.0,
        total_gen_mw=471.0,
        total_load_mw=463.75,
        llm_reasoning="Binary search",
        mode="fresh",
        elapsed_seconds=0.02,
        feasibility_detail="infeasible",
        solver="Newton-Rhapson",
    ))
    return journal


# ---------------------------------------------------------------------------
# Test: dispatch comparison
# ---------------------------------------------------------------------------

class TestDispatchComparison:
    def test_matching_generators(self, opflow_result_converged, pflow_result_feasible):
        comparisons = _build_dispatch_comparison(opflow_result_converged, pflow_result_feasible)
        assert len(comparisons) == 3
        assert comparisons[0].bus == 1
        assert comparisons[0].opflow_pg == 100.0
        assert comparisons[0].pflow_pg == 110.0
        assert comparisons[0].delta == 10.0

    def test_sorted_by_abs_delta(self, opflow_result_converged, pflow_result_feasible):
        comparisons = _build_dispatch_comparison(opflow_result_converged, pflow_result_feasible)
        deltas = [abs(c.delta) for c in comparisons]
        assert deltas == sorted(deltas, reverse=True)

    def test_none_pflow_result(self, opflow_result_converged):
        comparisons = _build_dispatch_comparison(opflow_result_converged, None)
        assert comparisons == []


# ---------------------------------------------------------------------------
# Test: loadability factor extraction
# ---------------------------------------------------------------------------

class TestLoadabilityExtraction:
    def test_with_feasible_and_infeasible(self, pflow_journal_with_scaling):
        factor = _extract_pflow_max_factor(pflow_journal_with_scaling)
        assert factor is not None
        assert factor == pytest.approx(1.625, abs=0.01)

    def test_no_scale_commands(self):
        journal = SearchJournal()
        journal.add_entry(JournalEntry(
            iteration=0, description="Base", commands=[],
            objective_value=0.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=1.0, voltage_max=1.0,
            max_line_loading_pct=50.0, total_gen_mw=100.0,
            total_load_mw=100.0, llm_reasoning="", mode="fresh",
            elapsed_seconds=0.01, feasibility_detail="feasible",
            solver="PFLOW",
        ))
        factor = _extract_pflow_max_factor(journal)
        assert factor is None

    def test_only_feasible(self):
        journal = SearchJournal()
        journal.add_entry(JournalEntry(
            iteration=1, description="Test 1.5",
            commands=[{"action": "scale_all_loads", "factor": 1.5}],
            objective_value=0.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.97, voltage_max=1.04,
            max_line_loading_pct=66.0, total_gen_mw=400.0,
            total_load_mw=397.0, llm_reasoning="", mode="fresh",
            elapsed_seconds=0.01, feasibility_detail="feasible",
            solver="PFLOW",
        ))
        journal.add_entry(JournalEntry(
            iteration=2, description="Test 1.8",
            commands=[{"action": "scale_all_loads", "factor": 1.8}],
            objective_value=0.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.96, voltage_max=1.04,
            max_line_loading_pct=70.0, total_gen_mw=480.0,
            total_load_mw=477.0, llm_reasoning="", mode="fresh",
            elapsed_seconds=0.01, feasibility_detail="feasible",
            solver="PFLOW",
        ))
        factor = _extract_pflow_max_factor(journal)
        assert factor == 1.8


# ---------------------------------------------------------------------------
# Test: summary formatting
# ---------------------------------------------------------------------------

class TestBenchmarkSummary:
    def test_complete_summary(self):
        text = _format_benchmark_summary(
            opflow_converged=True,
            opflow_cost=27557.57,
            pflow_cost=28314.22,
            cost_gap_pct=2.75,
            cost_gap_abs=756.65,
            dispatch_comparison=[
                DispatchComparison(bus=1, fuel="COAL", opflow_pg=100.0,
                                  pflow_pg=112.3, delta=12.3, opflow_pmax=150.0),
                DispatchComparison(bus=2, fuel="GAS", opflow_pg=80.0,
                                  pflow_pg=70.1, delta=-9.9, opflow_pmax=120.0),
            ],
            loadability=None,
        )
        assert "OPFLOW optimal cost" in text
        assert "27,557.57" in text
        assert "28,314.22" in text
        assert "2.75%" in text
        assert "Dispatch comparison" in text
        assert "COAL" in text
        assert "GAS" in text

    def test_no_pflow_cost(self):
        text = _format_benchmark_summary(
            opflow_converged=True,
            opflow_cost=27557.57,
            pflow_cost=None,
            cost_gap_pct=None,
            cost_gap_abs=None,
            dispatch_comparison=[],
            loadability=None,
        )
        assert "OPFLOW optimal cost" in text
        assert "Cost gap" not in text

    def test_opflow_not_converged(self):
        text = _format_benchmark_summary(
            opflow_converged=False,
            opflow_cost=None,
            pflow_cost=None,
            cost_gap_pct=None,
            cost_gap_abs=None,
            dispatch_comparison=[],
            loadability=None,
        )
        assert "did not converge" in text

    def test_with_loadability(self):
        loadability = LoadabilityResult(
            opflow_max_factor=1.85,
            pflow_max_factor=1.70,
            gap_pct=-8.1,
            detail="OPFLOW lambda=1.8500, PFLOW lambda=1.7000",
        )
        text = _format_benchmark_summary(
            opflow_converged=True,
            opflow_cost=27557.57,
            pflow_cost=28000.0,
            cost_gap_pct=1.6,
            cost_gap_abs=442.43,
            dispatch_comparison=[],
            loadability=loadability,
        )
        assert "Loadability comparison" in text
        assert "1.8500" in text
        assert "1.7000" in text
        assert "-8.10%" in text


# ---------------------------------------------------------------------------
# Test: full benchmark with mocked executor
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    @patch("agentigrid.engine.benchmark.SimulationExecutor")
    def test_successful_benchmark(self, MockExecutor, opflow_result_converged, pflow_result_feasible):
        mock_instance = MagicMock()
        MockExecutor.return_value = mock_instance
        mock_instance.run.return_value = MagicMock(success=True, error_message=None)

        with patch("agentigrid.engine.benchmark.parse_simulation_result_for_app") as mock_parse:
            mock_parse.return_value = opflow_result_converged

            with patch("agentigrid.engine.benchmark.parse_matpower") as mock_net:
                mock_net.return_value = MagicMock(gencost=[])

                result = run_pflow_vs_opflow_benchmark(
                    base_case_path=MagicMock(),
                    pflow_journal=SearchJournal(),
                    config=MagicMock(
                        exago=MagicMock(timeout=60),
                        output=MagicMock(),
                    ),
                    pflow_best_result=pflow_result_feasible,
                )

        assert result.opflow_converged is True
        assert result.opflow_objective == 27557.57

    @patch("agentigrid.engine.benchmark.SimulationExecutor")
    def test_opflow_failure(self, MockExecutor):
        mock_instance = MagicMock()
        MockExecutor.return_value = mock_instance
        mock_instance.run.return_value = MagicMock(success=False, error_message="OPFLOW failed")

        with patch("agentigrid.engine.benchmark.parse_matpower") as mock_net:
            mock_net.return_value = MagicMock(gencost=[])

            result = run_pflow_vs_opflow_benchmark(
                base_case_path=MagicMock(),
                pflow_journal=SearchJournal(),
                config=MagicMock(
                    exago=MagicMock(timeout=60),
                    output=MagicMock(),
                ),
            )

        assert result.error is not None
        assert "failed" in result.error.lower()

    @patch("agentigrid.engine.benchmark.SimulationExecutor")
    def test_with_pflow_best_result(self, MockExecutor, opflow_result_converged, pflow_result_feasible):
        mock_instance = MagicMock()
        MockExecutor.return_value = mock_instance
        mock_instance.run.return_value = MagicMock(success=True, error_message=None)

        with patch("agentigrid.engine.benchmark.parse_simulation_result_for_app") as mock_parse:
            mock_parse.return_value = opflow_result_converged

            mock_gencost = MagicMock()
            mock_cost_entry = MagicMock(model=2, coeffs=[0.01, 20.0, 100.0])
            mock_gencost.__len__ = lambda self: 3
            mock_gencost.__getitem__ = lambda self, i: mock_cost_entry

            mock_net = MagicMock()
            mock_net.gencost = mock_gencost

            with patch("agentigrid.engine.benchmark.parse_matpower") as mock_parse_net:
                mock_parse_net.return_value = mock_net

                result = run_pflow_vs_opflow_benchmark(
                    base_case_path=MagicMock(),
                    pflow_journal=SearchJournal(),
                    config=MagicMock(
                        exago=MagicMock(timeout=60),
                        output=MagicMock(),
                    ),
                    pflow_best_result=pflow_result_feasible,
                )

        assert result.pflow_best_result is not None
        assert len(result.dispatch_comparison) == 3


# ---------------------------------------------------------------------------
# Test: benchmark_result on SearchJournal
# ---------------------------------------------------------------------------

class TestJournalBenchmarkResult:
    def test_default_none(self):
        journal = SearchJournal()
        assert journal.benchmark_result is None

    def test_set_and_get(self):
        journal = SearchJournal()
        br = {"opflow_converged": True, "opflow_objective": 100.0}
        journal.benchmark_result = br
        assert journal.benchmark_result == br

    def test_export_json_includes_benchmark(self, tmp_path):
        journal = SearchJournal()
        journal.add_entry(JournalEntry(
            iteration=0, description="Base", commands=[],
            objective_value=0.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=1.0, voltage_max=1.0,
            max_line_loading_pct=50.0, total_gen_mw=100.0,
            total_load_mw=100.0, llm_reasoning="", mode="fresh",
            elapsed_seconds=0.01,
        ))
        journal.benchmark_result = {
            "opflow_converged": True,
            "opflow_objective": 27557.57,
            "pflow_best_computed_cost": 28000.0,
            "cost_gap_pct": 1.6,
            "cost_gap_abs": 442.43,
        }
        out_path = tmp_path / "test_journal.json"
        journal.export_json(out_path)
        import json
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "benchmark_result" in data
        assert data["benchmark_result"]["opflow_converged"] is True

    def test_export_json_omits_benchmark_when_none(self, tmp_path):
        journal = SearchJournal()
        journal.add_entry(JournalEntry(
            iteration=0, description="Base", commands=[],
            objective_value=0.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=1.0, voltage_max=1.0,
            max_line_loading_pct=50.0, total_gen_mw=100.0,
            total_load_mw=100.0, llm_reasoning="", mode="fresh",
            elapsed_seconds=0.01,
        ))
        out_path = tmp_path / "test_journal.json"
        journal.export_json(out_path)
        import json
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "benchmark_result" not in data


# ---------------------------------------------------------------------------
# Test: OPFLOW feasibility check uses feasibility_detail, not .feasible
# ---------------------------------------------------------------------------

class TestFeasibilityCheck:
    def test_uses_feasibility_detail_not_feasible(self, opflow_result_converged):
        assert opflow_result_converged.converged is True
        assert opflow_result_converged.feasibility_detail == "feasible"
        assert not hasattr(opflow_result_converged, "feasible")

    def test_infeasible_result(self):
        result = OPFLOWResult(
            converged=False,
            objective_value=0.0,
            convergence_status="DID NOT CONVERGE",
            solver="IPOPT",
            model="AC",
            objective_type="MIN_GEN_COST",
            num_iterations=0,
            solve_time=0.0,
            feasibility_detail="infeasible",
        )
        assert result.feasibility_detail == "infeasible"
        assert result.converged is False


# ---------------------------------------------------------------------------
# Test: BenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResultDataclass:
    def test_defaults(self):
        br = BenchmarkResult(
            opflow_converged=True,
            opflow_objective=100.0,
            pflow_best_computed_cost=110.0,
            cost_gap_pct=10.0,
            cost_gap_abs=10.0,
        )
        assert br.dispatch_comparison == []
        assert br.loadability is None
        assert br.opflow_result is None
        assert br.pflow_best_result is None
        assert br.summary_text == ""
        assert br.error is None

    def test_with_error(self):
        br = BenchmarkResult(
            opflow_converged=False,
            opflow_objective=None,
            pflow_best_computed_cost=None,
            cost_gap_pct=None,
            cost_gap_abs=None,
            error="OPFLOW benchmark failed",
        )
        assert br.error == "OPFLOW benchmark failed"


class TestDispatchComparisonDataclass:
    def test_fields(self):
        dc = DispatchComparison(
            bus=1, fuel="COAL", opflow_pg=100.0, pflow_pg=110.0,
            delta=10.0, opflow_pmax=150.0,
        )
        assert dc.bus == 1
        assert dc.fuel == "COAL"
        assert dc.delta == 10.0


class TestLoadabilityResultDataclass:
    def test_fields(self):
        lr = LoadabilityResult(
            opflow_max_factor=1.85,
            pflow_max_factor=1.70,
            gap_pct=-8.1,
            detail="test",
        )
        assert lr.opflow_max_factor == 1.85
        assert lr.pflow_max_factor == 1.70
        assert lr.gap_pct == -8.1