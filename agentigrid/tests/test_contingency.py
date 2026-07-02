"""Tests for the N-1/N-2 contingency screening engine (C.5).

Covers:
- pure enumeration: component pool, N-1 / N-2 counts, command shapes, determinism,
  ValueError guards, component filtering (branch-only)
- handler: _handle_contingency_sweep wiring — 15 contingencies enumerated, pass/fail
  derived from the standard predicate, journal recorded, no LLM/backend call inside
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine import contingency as C
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.executor import SimulationResult
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()


@pytest.fixture(scope="module")
def net118():
    return parse_matpower(IEEE118)


# ---------------------------------------------------------------------------
# Pure enumeration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestComponentPool:

    def test_branch_pool_deduped_endpoints(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        branches = [e for e in pool if e.kind == "branch"]
        assert len(branches) == 11
        assert all(e.ckt == 0 for e in branches)
        endpoint_set = {(min(e.fbus, e.tbus), max(e.fbus, e.tbus)) for e in branches}
        assert endpoint_set == {
            (47, 69), (49, 69), (68, 69), (69, 70), (69, 75), (69, 77),
            (70, 75), (74, 75), (75, 76), (75, 77), (76, 77),
        }

    def test_gen_pool(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        gen_buses = {e.bus for e in pool if e.kind == "gen"}
        assert gen_buses == {69, 76}  # 75 has no generator

    def test_load_pool(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        load_buses = {e.bus for e in pool if e.kind == "load"}
        assert load_buses == {75, 76}  # 69 has no load

    def test_total_pool_size(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        assert len(pool) == 15

    def test_branch_only_component_filter(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch",))
        assert len(pool) == 11
        assert all(e.kind == "branch" for e in pool)

    def test_determinism_label_sequence(self, net118):
        p1 = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        p2 = C.build_component_pool(net118, 77, 3, ("branch", "gen", "load"))
        assert [e.label() for e in p1] == [e.label() for e in p2]


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestEnumerate:

    def test_n1_count(self, net118):
        ctgs = C.enumerate_contingencies(net118, 77, 3, order=1)
        assert len(ctgs) == 15
        assert all(c.order == 1 for c in ctgs)

    def test_n2_count(self, net118):
        ctgs = C.enumerate_contingencies(net118, 77, 3, order=2)
        assert len(ctgs) == 105  # C(15, 2)
        assert all(c.order == 2 for c in ctgs)

    def test_branch_only_counts(self, net118):
        assert len(C.enumerate_contingencies(net118, 77, 3, 1, ("branch",))) == 11
        assert len(C.enumerate_contingencies(net118, 77, 3, 2, ("branch",))) == 55

    def test_order_out_of_range_raises(self, net118):
        with pytest.raises(ValueError):
            C.enumerate_contingencies(net118, 77, 3, order=3)

    def test_neighbor_count_zero_raises(self, net118):
        with pytest.raises(ValueError):
            C.enumerate_contingencies(net118, 77, 0, order=1)

    def test_determinism(self, net118):
        a = C.enumerate_contingencies(net118, 77, 3, order=2)
        b = C.enumerate_contingencies(net118, 77, 3, order=2)
        assert [c.label() for c in a] == [c.label() for c in b]


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestCommandShapes:

    def test_branch_to_command(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("branch",))
        cmd = pool[0].to_command()
        assert cmd["action"] == "set_branch_status"
        assert cmd["status"] == 0
        assert cmd["ckt"] == 0
        assert "fbus" in cmd and "tbus" in cmd

    def test_gen_to_command(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("gen",))
        cmd = pool[0].to_command()
        assert cmd == {
            "action": "set_gen_status", "bus": cmd["bus"],
            "gen_id": cmd["gen_id"], "status": 0,
        }
        assert cmd["gen_id"] == 0

    def test_load_to_command(self, net118):
        pool = C.build_component_pool(net118, 77, 3, ("load",))
        cmd = pool[0].to_command()
        assert cmd["action"] == "set_load"
        assert cmd["Pd"] == 0 and cmd["Qd"] == 0

    def test_contingency_label(self, net118):
        ctgs = C.enumerate_contingencies(net118, 77, 3, order=2, components=("branch",))
        assert ctgs[0].label().endswith(" out")
        assert " + " in ctgs[0].label()

    def test_contingency_commands_length_matches_order(self, net118):
        n2 = C.enumerate_contingencies(net118, 77, 3, order=2, components=("branch",))
        assert len(n2[0].commands()) == 2


# ---------------------------------------------------------------------------
# Handler
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


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestContingencyHandler:

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

    def test_n1_screen_records_15_all_pass(self, tmp_path):
        controller, backend_mock = self._controller(tmp_path)
        calls_before = len(backend_mock.mock_calls)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=True)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "target_bus": 77,
                "neighbor_count": 3, "contingency_order": 1,
                "components": ["branch", "gen", "load"],
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
                "description": "N-1 screen on bus 77 neighbors",
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.mode == "contingency"
        assert len(entry.explored_variants) == 15
        assert entry.contingency_meta["passed_count"] == 15
        assert entry.contingency_meta["failed_count"] == 0
        assert entry.contingency_meta["target_bus"] == 77
        assert entry.contingency_meta["order"] == 1
        # No LLM/backend call inside the screen.
        assert len(backend_mock.mock_calls) == calls_before

    def test_n1_screen_all_fail_when_infeasible(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "target_bus": 77,
                "neighbor_count": 3, "contingency_order": 1,
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.contingency_meta["passed_count"] == 0
        assert entry.contingency_meta["failed_count"] == 15
        assert "FAILED: 15 / 15" in controller._latest_results_text

    def test_n2_screen_records_105(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=True)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "target_bus": 77,
                "neighbor_count": 3, "contingency_order": 2,
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert len(entry.explored_variants) == 105
        assert entry.contingency_meta["order"] == 2

    def test_missing_target_bus_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "neighbor_count": 3,
        })
        assert kind == "error"
        assert "target_bus" in (controller._error_feedback or "")

    def test_bad_order_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "target_bus": 77, "contingency_order": 3,
        })
        assert kind == "error"
        assert "contingency_order" in (controller._error_feedback or "")

    def test_runaway_guard(self, tmp_path):
        import dataclasses
        controller, _ = self._controller(tmp_path)
        # Force a tiny guard (SearchConfig/AppConfig are frozen → rebuild them) so
        # N-2 (105 contingencies) trips it.
        new_search = dataclasses.replace(controller._config.search, contingency_max_count=50)
        controller._config = dataclasses.replace(controller._config, search=new_search)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "target_bus": 77,
            "neighbor_count": 3, "contingency_order": 2,
        })
        assert kind == "error"
        assert "exceeding the guard" in (controller._error_feedback or "")

    def test_dispatch_routes_contingency_mode(self, tmp_path):
        """_handle_sweep must route mode=contingency to the contingency handler."""
        controller, _ = self._controller(tmp_path)
        with patch.object(controller, "_handle_contingency_sweep",
                          return_value=("sweep", True)) as mock_h:
            controller._handle_sweep(1, {"mode": "contingency", "target_bus": 77})
        mock_h.assert_called_once()

    def test_non_opflow_rejected(self, tmp_path):
        import dataclasses
        controller, _ = self._controller(tmp_path)
        new_search = dataclasses.replace(controller._config.search, application="pflow")
        controller._config = dataclasses.replace(controller._config, search=new_search)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "target_bus": 77,
        })
        assert kind == "error"
