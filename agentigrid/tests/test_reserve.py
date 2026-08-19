"""Tests for the hot-reserve / minimum N-1 generator security screen (C.8).

Covers:
- the ``hot_reserve`` sweep metric (Σ Pmax−Pg over on-units; off-units excluded)
- ``all_generator_contingencies`` enumeration (system-wide, deterministic, gen-only)
- reserve accounting (required N-1 = largest committed Pg; margin; N-1-secure flag)
- handler: ``_handle_reserve_screen`` produces the right ``reserve_meta`` and journals
  it via ``add_reserve``, with no LLM/backend call inside the screen
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
from agentigrid.engine import sweep_metrics
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.executor import SimulationResult
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import GenResult, OPFLOWResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()


def _gen(bus, pg, pmax, status=1, pmin=0.0):
    return GenResult(
        bus=bus, status=status, fuel="COAL", Pg=pg, Qg=0.0,
        Pmin=pmin, Pmax=pmax, Qmin=-100.0, Qmax=100.0,
    )


def _opflow(feasible=True, generators=None):
    return OPFLOWResult(
        converged=feasible,
        objective_value=1000.0,
        convergence_status="CONVERGED" if feasible else "DID NOT CONVERGE",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=10, solve_time=0.1,
        branches=[], buses=[], generators=generators or [],
        voltage_min=0.96, voltage_max=1.04, max_line_loading_pct=80.0,
        num_violations=0 if feasible else 1,
        feasibility_detail="feasible" if feasible else "infeasible",
    )


# ---------------------------------------------------------------------------
# hot_reserve metric
# ---------------------------------------------------------------------------

class TestHotReserveMetric:

    def test_registered(self):
        assert "hot_reserve" in sweep_metrics.METRICS
        assert sweep_metrics.metric_direction("hot_reserve") == "minimize"

    def test_sum_over_on_units_only(self):
        gens = [
            _gen(1, pg=100.0, pmax=300.0, status=1),   # reserve 200
            _gen(2, pg=250.0, pmax=400.0, status=1),   # reserve 150
            _gen(3, pg=0.0, pmax=500.0, status=0),     # OFF — excluded
        ]
        res = _opflow(generators=gens)
        val = sweep_metrics.METRICS["hot_reserve"](res, None, {})
        assert val == pytest.approx(200.0 + 150.0)  # off unit's 500 excluded

    def test_none_candidate(self):
        assert sweep_metrics.METRICS["hot_reserve"](None, None, {}) is None


# ---------------------------------------------------------------------------
# all_generator_contingencies
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestAllGeneratorContingencies:

    @pytest.fixture(scope="class")
    def net118(self):
        return parse_matpower(IEEE118)

    def test_count_equals_in_service_gens(self, net118):
        n_on = sum(1 for g in net118.generators if g.status == 1)
        ctgs = C.all_generator_contingencies(net118)
        assert len(ctgs) == n_on
        assert n_on > 0

    def test_all_order_one_gen(self, net118):
        for ctg in C.all_generator_contingencies(net118):
            assert ctg.order == 1
            (e,) = ctg.elements
            assert e.kind == "gen"
            assert e.bus == e.neighbor_bus
            assert e.hop == 0

    def test_deterministic_sorted_by_bus_genid(self, net118):
        ctgs = C.all_generator_contingencies(net118)
        keys = [(ctg.elements[0].bus, ctg.elements[0].gen_id) for ctg in ctgs]
        assert keys == sorted(keys)
        # And reproducible across calls.
        assert keys == [
            (ctg.elements[0].bus, ctg.elements[0].gen_id)
            for ctg in C.all_generator_contingencies(net118)
        ]

    def test_gen_id_matches_per_bus_index(self, net118):
        # gen_id is the position among generators at the bus (modifier semantics).
        by_bus: dict[int, list[int]] = {}
        for ctg in C.all_generator_contingencies(net118):
            e = ctg.elements[0]
            by_bus.setdefault(e.bus, []).append(e.gen_id)
        for bus, ids in by_bus.items():
            expected = [
                i for i, g in enumerate(
                    [g for g in net118.generators if g.bus == bus]
                ) if g.status == 1
            ]
            assert ids == expected


# ---------------------------------------------------------------------------
# Reserve accounting
# ---------------------------------------------------------------------------

class TestReserveAccounting:

    def test_required_is_largest_pg_and_margin(self):
        on_units = [
            _gen(10, pg=100.0, pmax=300.0),
            _gen(20, pg=250.0, pmax=400.0),   # largest Pg
            _gen(30, pg=50.0, pmax=200.0),
        ]
        available = sweep_metrics.METRICS["hot_reserve"](
            _opflow(generators=on_units), None, {}
        )
        assert available == pytest.approx(200.0 + 150.0 + 150.0)  # 500
        required = max(g.Pg for g in on_units)
        assert required == pytest.approx(250.0)
        assert available - required == pytest.approx(250.0)


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
class TestReserveHandler:

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

    def test_reserve_meta_and_no_backend_call(self, tmp_path):
        controller, backend_mock = self._controller(tmp_path)

        # Three committed units (+ one off-line unit that must be excluded).
        base_gens = [
            _gen(10, pg=100.0, pmax=300.0, status=1),
            _gen(20, pg=250.0, pmax=400.0, status=1),   # largest Pg AND Pmax
            _gen(30, pg=50.0, pmax=200.0, status=1),
            _gen(40, pg=0.0, pmax=500.0, status=0),     # OFF — excluded
        ]
        # Screen exactly these three in-service units.
        fake_ctgs = [
            C.Contingency(elements=(C.OutageElement(
                kind="gen", neighbor_bus=b, hop=0, bus=b, gen_id=0),))
            for b in (10, 20, 30)
        ]

        state = {"i": 0}

        def fake_parse(sim, application="opflow", bus_limits=None):
            k = state["i"]
            state["i"] += 1
            if k == 0:
                return _opflow(True, generators=base_gens)  # base reference solve
            # contingencies: unit @20's loss is infeasible → 1 failure
            return _opflow(k != 2)

        calls_before = len(backend_mock.mock_calls)
        with patch("agentigrid.engine.agent_loop.contingency.all_generator_contingencies",
                   return_value=fake_ctgs), \
             patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=fake_parse):
            kind, ok = controller._handle_reserve_screen(1, {
                "mode": "reserve", "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
                "description": "Min N-1 hot reserve",
            })

        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.mode == "reserve"
        assert entry.convergence_status == "CONTINGENCY"
        rm = entry.reserve_meta
        assert rm["n_on"] == 3
        assert rm["hot_reserve_available"] == pytest.approx(500.0)
        assert rm["largest_pg"] == pytest.approx(250.0)
        assert rm["largest_pg_bus"] == 20
        assert rm["largest_pmax"] == pytest.approx(400.0)
        assert rm["required_reserve_n1"] == pytest.approx(250.0)
        assert rm["margin"] == pytest.approx(250.0)
        assert rm["passed_count"] == 2
        assert rm["failed_count"] == 1
        assert rm["n1_secure"] is False
        # View is populated and no LLM/backend call happened inside the screen.
        assert "hot-reserve" in (controller._latest_results_text or "").lower()
        assert len(backend_mock.mock_calls) == calls_before

    def test_secure_when_all_units_feasible(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        base_gens = [
            _gen(10, pg=100.0, pmax=300.0, status=1),
            _gen(20, pg=80.0, pmax=400.0, status=1),
        ]
        fake_ctgs = [
            C.Contingency(elements=(C.OutageElement(
                kind="gen", neighbor_bus=b, hop=0, bus=b, gen_id=0),))
            for b in (10, 20)
        ]
        state = {"i": 0}

        def fake_parse(sim, application="opflow", bus_limits=None):
            k = state["i"]
            state["i"] += 1
            if k == 0:
                return _opflow(True, generators=base_gens)
            return _opflow(True)  # every unit loss feasible

        with patch("agentigrid.engine.agent_loop.contingency.all_generator_contingencies",
                   return_value=fake_ctgs), \
             patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=fake_parse):
            kind, ok = controller._handle_reserve_screen(1, {
                "mode": "reserve", "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert (kind, ok) == ("sweep", True)
        rm = controller._journal.entries[-1].reserve_meta
        assert rm["n1_secure"] is True
        assert rm["failed_count"] == 0
        assert rm["required_reserve_n1"] == pytest.approx(100.0)  # largest Pg

    def test_base_infeasible_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            kind, ok = controller._handle_reserve_screen(1, {
                "mode": "reserve", "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert kind == "error"
        assert "base operating point" in (controller._error_feedback or "")
