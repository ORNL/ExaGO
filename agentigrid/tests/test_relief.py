"""Tests for the contingency relief-measure search (C.7, Option A).

Covers:
- priority order respected (first resolving measure returned; earlier ones logged failed)
- generator_redispatch pass-through (no command applied, never resolves)
- load_curtailment backstop (always resolves, reports minimum MW within tolerance)
- determinism (identical inputs → identical ReliefResult)
- handler: an N-2 run with seeded failures attaches per-failure relief entries to the
  view and journal, with no LLM/backend call inside the screen
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine import contingency as C
from agentigrid.engine import relief
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.executor import SimulationResult
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()

_CFG = SimpleNamespace(
    relief_tap_steps=[0.90, 0.95, 1.00, 1.05, 1.10],
    relief_curtail_tol_mw=1.0,
    relief_max_solves=2000,
)


def _opflow(feasible=True):
    return OPFLOWResult(
        converged=feasible,
        objective_value=1000.0,
        convergence_status="CONVERGED" if feasible else "DID NOT CONVERGE",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=10, solve_time=0.1,
        branches=[], buses=[],
        voltage_min=0.96, voltage_max=1.04, max_line_loading_pct=80.0,
        num_violations=0 if feasible else 1,
        feasibility_detail="feasible" if feasible else "infeasible",
    )


@pytest.fixture(scope="module")
def net118():
    return parse_matpower(IEEE118)


@pytest.fixture
def branch_contingency(net118):
    # A single branch outage at a neighbor of bus 77.
    return C.enumerate_contingencies(net118, 77, 3, order=1, components=("branch",))[0]


# ---------------------------------------------------------------------------
# Pure find_relief
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestFindRelief:

    def test_priority_order_line_switching_resolves(self, net118, branch_contingency):
        n_outaged = sum(1 for e in branch_contingency.elements if e.kind == "branch")

        def solve_fn(net):
            # Feasible only when an EXTRA branch (beyond the outage) is switched out.
            n_oos = sum(1 for b in net.branches if b.status == 0)
            return _opflow(feasible=n_oos > n_outaged)

        res = relief.find_relief(
            net118, branch_contingency,
            ["transformer_ratio", "generator_redispatch", "line_switching"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is True
        assert res.action.measure == "line_switching"
        # Earlier measures recorded as attempted-and-failed, in order.
        assert res.attempts[0] == ("transformer_ratio", False)
        assert res.attempts[1] == ("generator_redispatch", False)
        assert res.attempts[-1] == ("line_switching", True)

    def test_generator_redispatch_passthrough_never_resolves(self, net118, branch_contingency):
        calls = {"n": 0}

        def solve_fn(net):
            calls["n"] += 1
            return _opflow(feasible=True)  # would resolve anything that actually solves

        res = relief.find_relief(
            net118, branch_contingency, ["generator_redispatch"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is False
        assert res.attempts == [("generator_redispatch", False)]
        # Pass-through applies no command → performs no solve.
        assert calls["n"] == 0

    def test_load_curtailment_backstop_resolves_min_mw(self, net118, branch_contingency):
        focus = relief._focus_buses(branch_contingency)
        base = sum(b.Pd for b in net118.buses if b.bus_i in focus and (b.Pd or b.Qd))
        assert base > 0, "need load at focus buses for this test"
        thresh = 0.4 * base

        def solve_fn(net):
            curtailed = base - sum(b.Pd for b in net.buses if b.bus_i in focus)
            return _opflow(feasible=curtailed >= thresh - 1e-9)

        res = relief.find_relief(
            net118, branch_contingency, ["load_curtailment"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is True
        assert res.action.measure == "load_curtailment"
        # Minimum feasible curtailment ≈ threshold, within the bisection tolerance.
        curtailed_mw = base - sum(c["Pd"] for c in res.action.commands)
        assert abs(curtailed_mw - thresh) <= 2.0

    def test_load_curtailment_always_resolves_at_full(self, net118, branch_contingency):
        def solve_fn(net):
            # Feasible only when ALL local load removed (f == 1).
            total = sum(b.Pd for b in net.buses if b.bus_i in relief._focus_buses(branch_contingency))
            return _opflow(feasible=total <= 1e-6)

        res = relief.find_relief(
            net118, branch_contingency, ["load_curtailment"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is True

    def test_unresolved_when_nothing_helps(self, net118, branch_contingency):
        def solve_fn(net):
            return _opflow(feasible=False)  # nothing ever restores feasibility

        res = relief.find_relief(
            net118, branch_contingency,
            ["transformer_ratio", "line_switching", "load_curtailment"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is False
        assert res.action is None
        assert [m for m, _ in res.attempts] == [
            "transformer_ratio", "line_switching", "load_curtailment",
        ]

    def test_determinism(self, net118, branch_contingency):
        n_outaged = sum(1 for e in branch_contingency.elements if e.kind == "branch")

        def solve_fn(net):
            n_oos = sum(1 for b in net.branches if b.status == 0)
            return _opflow(feasible=n_oos > n_outaged)

        measures = ["transformer_ratio", "generator_redispatch", "line_switching"]
        r1 = relief.find_relief(net118, branch_contingency, measures, 0.9, 1.1, solve_fn, _CFG)
        r2 = relief.find_relief(net118, branch_contingency, measures, 0.9, 1.1, solve_fn, _CFG)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Handler integration
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(
            backend="openai", model="test-model", api_key_env="TEST_KEY",
            openai_base_url=None, ollama_host="http://localhost:11434",
            ollama_cloud_host=None, temperature=0.3, max_tokens=4096,
        ),
        search=SearchConfig(
            max_iterations=5, default_mode="accumulative",
            base_case=IEEE118, gic_file=None, application="opflow",
        ),
        output=OutputConfig(
            workdir=tmp_path / "wd", logs_dir=tmp_path / "logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False,
        ),
    )


def _sim_result():
    return SimulationResult(
        success=True, exit_code=0, stdout="ok", stderr="", elapsed_seconds=0.1,
        input_file=Path("/tmp/x.m"), application="opflow", error_message=None,
        workdir=Path("/tmp"),
    )


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestReliefHandler:

    def _controller(self, tmp_path):
        cfg = _make_config(tmp_path)
        backend_mock = MagicMock()
        with patch("agentigrid.engine.agent_loop.create_backend", return_value=backend_mock), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
            mock_executor = MagicMock()
            mock_executor.run.return_value = _sim_result()
            mock_executor.run_parallel.side_effect = (
                lambda tasks, max_workers=4, thread_limit=None, on_progress=None:
                {i: _sim_result() for i in range(len(tasks))}
            )
            mock_exec_cls.return_value = mock_executor
            controller = AgentLoopController(cfg)
        net = parse_matpower(IEEE118)
        controller._base_network = net
        controller._current_network = net
        return controller, backend_mock

    def test_relief_entries_attached_for_failures(self, tmp_path, net118):
        controller, backend_mock = self._controller(tmp_path)
        data = {
            "mode": "contingency", "target_bus": 77,
            "neighbor_count": 1, "contingency_order": 2, "components": ["branch"],
            "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            "relief_measures": ["line_switching", "load_curtailment"],
            "description": "N-2 + relief",
        }
        n = len(C.enumerate_contingencies(net118, 77, 1, 2, ("branch",)))
        assert n >= 3  # need at least 2 failures + some passers

        state = {"i": 0}

        def fake_parse(sim, application="opflow", bus_limits=None):
            k = state["i"]
            state["i"] += 1
            if k == 0:
                return _opflow(True)          # pre-contingency reference
            if 1 <= k <= n:
                return _opflow(k not in (1, 2))  # contingencies idx 0,1 fail
            return _opflow(True)              # relief solves → resolve

        calls_before = len(backend_mock.mock_calls)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=fake_parse):
            kind, ok = controller._handle_contingency_sweep(1, data)

        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        summaries = entry.explored_variants
        failed = [s for s in summaries if not s["passed"]]
        assert len(failed) == 2
        for s in failed:
            assert "relief" in s
            assert s["relief"]["resolved"] is True
            assert s["relief"]["measure"] == "line_switching"
        # View has a relief section.
        assert "Relief for failed contingencies" in controller._latest_results_text
        # No LLM/backend call inside the screen.
        assert len(backend_mock.mock_calls) == calls_before

    def test_no_relief_without_relief_measures_matches_c5(self, tmp_path, net118):
        """Without relief_measures, summaries carry no 'relief' key (C.5 behavior)."""
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "target_bus": 77,
                "neighbor_count": 1, "contingency_order": 1, "components": ["branch"],
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert all("relief" not in s for s in entry.explored_variants)
        assert "Relief for failed contingencies" not in controller._latest_results_text

    def test_invalid_relief_measure_rejected(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "target_bus": 77,
            "relief_measures": ["nonsense_measure"],
        })
        assert kind == "error"
        assert "relief_measures" in (controller._error_feedback or "")

    def test_relief_budget_exhausted(self, tmp_path, net118):
        import dataclasses
        controller, _ = self._controller(tmp_path)
        # Budget 0 → every failure marked exhausted, none searched.
        new_search = dataclasses.replace(controller._config.search, relief_max_solves=0)
        controller._config = dataclasses.replace(controller._config, search=new_search)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "target_bus": 77,
                "neighbor_count": 1, "contingency_order": 1, "components": ["branch"],
                "relief_measures": ["line_switching"],
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        failed = [s for s in entry.explored_variants if not s["passed"]]
        assert failed  # there are failures
        assert all(s["relief"]["detail"] == "relief budget exhausted" for s in failed)
