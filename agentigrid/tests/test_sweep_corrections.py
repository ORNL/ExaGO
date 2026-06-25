"""Tests for the C.2/C.3 corrections (prompt-13/15/17 evidence).

Fix 1 — custom metrics are certified-gated (only CONVERGED candidates carry a metric).
Fix 2 — summary aggregation descends into sweep variant costs (best ≠ base case).
Fix 3 — reactive_adequacy forces Q (Qmin == Qmax) and records dispatched_q.
Fix 4 — identical sweeps are de-duplicated (system-prompt rule + session cache).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import AgentLoopController, _is_certified
from agentigrid.engine.executor import SimulationResult
from agentigrid.engine.journal import SearchJournal, JournalEntry
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult, BusResult, GenResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_CASE = DATA_DIR / "case_ACTIVSg200.m"
_has_base_case = BASE_CASE.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _busres(bus_id, vm):
    return BusResult(bus_id=bus_id, Pd=0, Pd_loss=0, Qd=0, Qd_loss=0,
                     Vm=vm, Va=0, mult_Pmis=0, mult_Qmis=0, Pslack=0, Qslack=0)


def _genres(bus, pg, qg):
    return GenResult(bus=bus, status=1, fuel="", Pg=pg, Qg=qg,
                     Pmin=0, Pmax=999, Qmin=-999, Qmax=999)


def _opflow(converged=True, obj=1000.0, violations=0, buses=None, gens=None):
    return OPFLOWResult(
        converged=converged, objective_value=obj,
        convergence_status="CONVERGED" if converged else "DID NOT CONVERGE",
        solver="IPOPT", model="PB", objective_type="MIN_COST",
        num_iterations=5, solve_time=0.1, buses=buses or [], generators=gens or [],
        voltage_min=0.95, voltage_max=1.05, max_line_loading_pct=50.0,
        num_violations=violations,
        feasibility_detail="feasible" if (converged and violations == 0) else "infeasible",
    )


def _sim():
    return SimulationResult(success=True, exit_code=0, stdout="ok", stderr="",
                            elapsed_seconds=0.1, input_file=Path("/tmp/x.m"),
                            application="opflow", error_message=None, workdir=Path("/tmp"))


def _make_config(tmp_path):
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(backend="openai", model="m", api_key_env="K", openai_base_url=None,
                      ollama_host="h", ollama_cloud_host=None, temperature=0.3, max_tokens=4096),
        search=SearchConfig(max_iterations=5, default_mode="accumulative",
                            base_case=BASE_CASE, gic_file=None, application="opflow"),
        output=OutputConfig(workdir=tmp_path / "wd", logs_dir=tmp_path / "logs",
                            save_journal=False, journal_format="json",
                            save_modified_files=False, verbose=False),
    )


def _controller(tmp_path):
    cfg = _make_config(tmp_path)
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        ex = MagicMock()
        ex.run.return_value = _sim()
        ex.run_parallel.side_effect = (
            lambda tasks, max_workers=4, thread_limit=None, on_progress=None:
            {i: _sim() for i in range(len(tasks))}
        )
        mock_exec_cls.return_value = ex
        c = AgentLoopController(cfg)
    c._base_network = parse_matpower(BASE_CASE)
    c._mock_executor = ex
    return c


def _variant(entry, bus):
    return next(v for v in entry.explored_variants if v["bus"] == bus)


# ===========================================================================
# Fix 1 — certified-gate custom metrics
# ===========================================================================

class TestCertifiedGateHelper:

    def test_converged_is_certified(self):
        assert _is_certified(_opflow(converged=True)) is True

    def test_non_converged_not_certified(self):
        assert _is_certified(_opflow(converged=False)) is False

    def test_none_not_certified(self):
        assert _is_certified(None) is False


@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestMetricCertifiedGate:

    def test_non_converged_metric_is_none(self, tmp_path):
        """The global-max ΔV belongs to a NON-converged bus; it must carry no
        trusted metric, and the converged bus keeps its (smaller) value."""
        c = _controller(tmp_path)
        base = _opflow(buses=[_busres(1, 1.00), _busres(2, 1.00), _busres(99, 1.00)])
        cand1 = _opflow(converged=True,  buses=[_busres(1, 1.01), _busres(2, 1.00), _busres(99, 1.03)])  # ΔV 0.03
        cand2 = _opflow(converged=False, buses=[_busres(1, 1.00), _busres(2, 1.00), _busres(99, 1.09)])  # ΔV 0.09 (uncertified)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=[base, cand1, cand2]):
            kind, ok = c._handle_sweep(1, {
                "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "metric": "max_delta_v",
                "description": "voltage step",
            })
        assert (kind, ok) == ("sweep", True)
        entry = c._journal.get_sweep_entry()
        v1, v2 = _variant(entry, 1), _variant(entry, 2)
        assert v1["feasible"] is True
        assert v1["metric_value"] == pytest.approx(0.03)
        # non-converged bus 2: no trusted metric, not feasible
        assert v2["feasible"] is False
        assert v2["metric_value"] is None
        # the reduction (max over feasible) is 0.03, never the uncertified 0.09
        feasible_metrics = [v["metric_value"] for v in entry.explored_variants
                            if v["feasible"] and v.get("metric_value") is not None]
        assert max(feasible_metrics) == pytest.approx(0.03)

    def test_all_converged_unchanged(self, tmp_path):
        """Regression: with all candidates converged, every metric is recorded."""
        c = _controller(tmp_path)
        base = _opflow(buses=[_busres(1, 1.00), _busres(2, 1.00)])
        cand1 = _opflow(converged=True, buses=[_busres(1, 1.02), _busres(2, 1.00)])  # 0.02
        cand2 = _opflow(converged=True, buses=[_busres(1, 1.00), _busres(2, 1.04)])  # 0.04
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=[base, cand1, cand2]):
            c._handle_sweep(1, {
                "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "metric": "max_delta_v",
            })
        entry = c._journal.get_sweep_entry()
        assert _variant(entry, 1)["metric_value"] == pytest.approx(0.02)
        assert _variant(entry, 2)["metric_value"] == pytest.approx(0.04)


# ===========================================================================
# Fix 2 — summary aggregation descends into sweep variants
# ===========================================================================

def _base_entry(cost):
    return JournalEntry(
        iteration=0, description="Base case", commands=[], objective_value=cost,
        feasible=True, convergence_status="CONVERGED", violations_count=0,
        voltage_min=0.95, voltage_max=1.05, max_line_loading_pct=50.0,
        total_gen_mw=0.0, total_load_mw=0.0, llm_reasoning="", mode="modify",
        elapsed_seconds=0.1,
    )


class TestSummaryDescendsIntoSweeps:

    def _journal_with_cost_sweep(self):
        j = SearchJournal()
        j._entries.append(_base_entry(27557.57))
        j.add_sweep(
            iteration=1, description="[sweep] siting", candidate_count=3,
            candidate_summaries=[
                {"bus": 50, "feasible": True, "cost": 27600.0},
                {"bus": 181, "feasible": True, "cost": 27367.73},
                {"bus": 90, "feasible": False, "cost": None},
            ],
            feasible_buses=[50, 181],
        )
        return j

    def test_cost_min_reports_sweep_optimum_not_base(self):
        j = self._journal_with_cost_sweep()
        stats = j.summary_stats(goal_type="cost_minimization")
        assert stats["best_objective"] == pytest.approx(27367.73)
        assert stats["best_iteration"] == 1
        assert stats["best_bus"] == 181

    def test_default_goal_type_also_descends(self):
        j = self._journal_with_cost_sweep()
        stats = j.summary_stats(goal_type=None)
        assert stats["best_objective"] == pytest.approx(27367.73)
        assert stats["best_bus"] == 181

    def test_metric_sweep_not_hijacked_by_cost(self):
        """A max_delta_v sweep carries variant costs but its reduction is the metric,
        not cost — the cost-min summary must ignore those variant costs."""
        j = SearchJournal()
        j._entries.append(_base_entry(27557.57))
        j.add_sweep(
            iteration=1, description="[sweep] dv", candidate_count=2,
            candidate_summaries=[
                {"bus": 5, "feasible": True, "cost": 100.0, "metric_name": "max_delta_v", "metric_value": 0.03},
                {"bus": 6, "feasible": True, "cost": 200.0, "metric_name": "max_delta_v", "metric_value": 0.05},
            ],
            feasible_buses=[5, 6],
        )
        stats = j.summary_stats(goal_type="cost_minimization")
        # variant costs (100/200) are ignored; base case stays the best cost
        assert stats["best_objective"] == pytest.approx(27557.57)
        assert stats["best_iteration"] == 0
        assert stats["best_bus"] is None

    def test_boundary_sweep_not_used_for_cost(self):
        j = SearchJournal()
        j._entries.append(_base_entry(27557.57))
        j.add_sweep(
            iteration=1, description="[boundary sweep] cap", candidate_count=2,
            candidate_summaries=[
                {"bus": 5, "feasible": True, "cost": None, "max_feasible_mw": 245.0},
                {"bus": 6, "feasible": True, "cost": None, "max_feasible_mw": 300.0},
            ],
            feasible_buses=[5, 6],
        )
        stats = j.summary_stats(goal_type="cost_minimization")
        assert stats["best_objective"] == pytest.approx(27557.57)
        assert stats["best_bus"] is None

    def test_non_cost_goal_does_not_descend(self):
        j = self._journal_with_cost_sweep()
        stats = j.summary_stats(goal_type="feasibility_boundary")
        # cost descent is gated off for non-cost goals → base remains
        assert stats["best_objective"] == pytest.approx(27557.57)
        assert stats["best_bus"] is None


# ===========================================================================
# Fix 3 — reactive_adequacy Q-forcing + dispatched_q audit
# ===========================================================================

@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestReactiveAdequacyAudit:

    def test_predicate_forces_qmin_equals_qmax(self, tmp_path):
        c = _controller(tmp_path)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0, "Qmax": 50.0},
            {}, "reactive_adequacy")
        assert mt["Qmin"] == 50.0 == mt["Qmax"]  # forced, not merely bounded

    def test_dispatched_q_recorded_at_adequate_bus(self, tmp_path):
        c = _controller(tmp_path)
        # added unit solves at Qg == Qmax target (50) at each adequate bus
        g_at = lambda bus: _opflow(converged=True, gens=[_genres(bus, 100.0, 50.0)])
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=[g_at(1), g_at(2)]):
            kind, ok = c._handle_sweep(1, {
                "mutation": {"action": "add_generator_at_bus", "capacity_mw": 100.0, "Qmax": 50.0},
                "feasibility_predicate": "reactive_adequacy",
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "description": "reactive adequacy",
            })
        assert (kind, ok) == ("sweep", True)
        entry = c._journal.get_sweep_entry()
        for bus in (1, 2):
            assert _variant(entry, bus)["dispatched_q"] == pytest.approx(50.0)


# ===========================================================================
# Fix 4 — redundant-sweep suppression
# ===========================================================================

class TestSystemPromptNudge:

    def test_no_redundant_sweep_rule_present(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        p = build_system_prompt("SCHEMA", "NET", application="opflow")
        assert "DO NOT RE-RUN AN IDENTICAL SWEEP" in p


@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestSweepDedup:

    def _run_sweep(self, c, pd=100.0):
        converged = _opflow(converged=True)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=converged):
            return c._handle_sweep(1, {
                "mutation": {"action": "add_load_at_bus", "Pd": pd},
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "description": "load sweep",
            })

    def test_identical_sweep_served_from_cache(self, tmp_path):
        c = _controller(tmp_path)
        self._run_sweep(c)
        solves_after_first = c._mock_executor.run_parallel.call_count
        assert solves_after_first == 1

        # identical re-request → cache hit, no new solves
        kind, ok = self._run_sweep(c)
        assert (kind, ok) == ("sweep", True)
        assert c._mock_executor.run_parallel.call_count == solves_after_first  # unchanged
        # journaled as cached
        assert "cached" in c._journal.get_sweep_entry().description.lower()

    def test_changed_parameter_still_executes(self, tmp_path):
        c = _controller(tmp_path)
        self._run_sweep(c, pd=100.0)
        first = c._mock_executor.run_parallel.call_count
        self._run_sweep(c, pd=200.0)  # different Pd → different signature
        assert c._mock_executor.run_parallel.call_count == first + 1
