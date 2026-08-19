"""Tests for C2 (dispatchable / cost-curve generators) and C3 (metric/predicate registry).

C2 — prompt 13: min-cost siting of a dispatchable generator under economic dispatch.
C3 — prompts 15 & 17: custom per-candidate metric (max_delta_v) and feasibility
predicate (reactive_adequacy), selected by name from a verified registry.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine import sweep_metrics
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.commands import AddGeneratorAtBus, parse_command
from agentigrid.engine.executor import SimulationResult
from agentigrid.engine.modifier import (
    apply_modifications, _median_existing_cost_coeffs,
)
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult, BusResult, GenResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_CASE = DATA_DIR / "case_ACTIVSg200.m"
_has_base_case = BASE_CASE.exists()


# ---------------------------------------------------------------------------
# Synthetic result helpers
# ---------------------------------------------------------------------------

def _busres(bus_id, vm):
    return BusResult(bus_id=bus_id, Pd=0, Pd_loss=0, Qd=0, Qd_loss=0,
                     Vm=vm, Va=0, mult_Pmis=0, mult_Qmis=0, Pslack=0, Qslack=0)


def _genres(bus, pg, qg=0.0):
    return GenResult(bus=bus, status=1, fuel="", Pg=pg, Qg=qg,
                     Pmin=0, Pmax=999, Qmin=-999, Qmax=999)


def _opflow(converged=True, obj=1000.0, violations=0, buses=None, gens=None,
            vmin=0.95, vmax=1.05, max_load=50.0):
    return OPFLOWResult(
        converged=converged, objective_value=obj,
        convergence_status="CONVERGED" if converged else "DID NOT CONVERGE",
        solver="IPOPT", model="PB", objective_type="MIN_COST",
        num_iterations=5, solve_time=0.1,
        buses=buses or [], generators=gens or [],
        voltage_min=vmin, voltage_max=vmax, max_line_loading_pct=max_load,
        num_violations=violations,
        feasibility_detail="feasible" if (converged and violations == 0) else "infeasible",
    )


# ===========================================================================
# C2 — dispatchable / cost-curve generator primitive
# ===========================================================================

@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestC2GeneratorPrimitive:

    @pytest.fixture(scope="class")
    def net(self):
        return parse_matpower(BASE_CASE)

    def test_dispatchable_unit_bounds_and_cost(self, net):
        cmd = AddGeneratorAtBus(bus=net.buses[10].bus_i, capacity_mw=200.0, dispatchable=True)
        m, _ = apply_modifications(net, [cmd], application="opflow")
        g = m.generators[-1]
        assert g.Pmin == 0.0 and g.Pmax == 200.0  # dispatchable
        # a gencost row was appended and stays aligned
        assert len(m.gencost) == len(m.generators)
        gc = m.gencost[-1]
        # default median-existing curve is non-trivial (not the zero curve)
        assert any(c != 0.0 for c in gc.coeffs)

    def test_fixed_injection_regression(self, net):
        """C.1/hosting behaviour unchanged: forced injection pins Pmin == Pmax."""
        cmd = AddGeneratorAtBus(bus=net.buses[10].bus_i, capacity_mw=150.0, dispatchable=False)
        m, _ = apply_modifications(net, [cmd], application="opflow")
        g = m.generators[-1]
        assert g.Pmin == g.Pmax == 150.0
        # fixed injection → zero cost curve (cost irrelevant, Pg pinned)
        assert m.gencost[-1].coeffs == [0.0, 0.0]

    def test_explicit_cost_coeffs(self, net):
        cmd = AddGeneratorAtBus(bus=net.buses[3].bus_i, capacity_mw=100.0,
                                dispatchable=True, cost_coeffs=[0.01, 35.0, 0.0])
        m, _ = apply_modifications(net, [cmd], application="opflow")
        assert m.gencost[-1].coeffs == [0.01, 35.0, 0.0]

    def test_median_existing_is_mid_merit(self, net):
        coeffs = _median_existing_cost_coeffs(net)
        assert len(coeffs) == 3
        # linear term (c1, $/MWh) should be a positive, plausible mid-merit value
        assert coeffs[1] > 0


class TestMedianCostFallback:

    def test_fallback_when_no_polynomial_curves(self):
        from agentigrid.parsers.matpower_model import MATNetwork, GenCost
        net = MATNetwork(
            casename="x", version="2", baseMVA=100.0, buses=[], generators=[],
            branches=[], gencost=[GenCost(model=1, startup=0, shutdown=0, ncost=2, coeffs=[0, 0, 1, 1])],
            header_comments="",
        )
        coeffs = _median_existing_cost_coeffs(net)
        assert coeffs == [0.0, 40.0, 0.0]  # fallback mid-merit linear


# ===========================================================================
# C3 — metric / predicate registry
# ===========================================================================

class TestRegistry:

    def test_defaults_registered(self):
        assert "cost" in sweep_metrics.METRICS
        assert "max_delta_v" in sweep_metrics.METRICS
        assert "standard" in sweep_metrics.PREDICATES
        assert "reactive_adequacy" in sweep_metrics.PREDICATES

    def test_metric_needs_base(self):
        assert sweep_metrics.metric_needs_base("max_delta_v") is True
        assert sweep_metrics.metric_needs_base("cost") is False
        assert sweep_metrics.metric_needs_base(None) is False

    def test_metric_direction(self):
        assert sweep_metrics.metric_direction("max_delta_v") == "maximize"
        assert sweep_metrics.metric_direction("cost") == "minimize"

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            sweep_metrics.get_metric("nonexistent")

    def test_unknown_predicate_raises(self):
        with pytest.raises(KeyError):
            sweep_metrics.get_predicate("nonexistent")

    def test_register_metric_extensible(self):
        sweep_metrics.register_metric("test_tmp", lambda c, b, ctx: 1.0,
                                      needs_base=True, direction="maximize")
        try:
            assert sweep_metrics.metric_needs_base("test_tmp") is True
            assert sweep_metrics.metric_direction("test_tmp") == "maximize"
        finally:
            sweep_metrics.METRICS.pop("test_tmp", None)
            sweep_metrics._METRICS_NEED_BASE.discard("test_tmp")
            sweep_metrics._METRIC_DIRECTION.pop("test_tmp", None)


class TestMaxDeltaVMetric:

    def test_true_max_absolute_difference(self):
        base = _opflow(buses=[_busres(1, 1.00), _busres(2, 1.00), _busres(3, 1.00)])
        cand = _opflow(buses=[_busres(1, 1.01), _busres(2, 0.93), _busres(3, 1.02)])
        # max |ΔV| = |0.93 - 1.00| = 0.07 at bus 2
        assert sweep_metrics._max_delta_v_metric(cand, base, {}) == pytest.approx(0.07)

    def test_max_bus_need_not_be_candidate_bus(self):
        """The largest step can occur at a bus other than the one switched."""
        base = _opflow(buses=[_busres(10, 1.00), _busres(99, 1.00)])
        cand = _opflow(buses=[_busres(10, 1.005), _busres(99, 0.95)])
        ctx = {"bus": 10}  # candidate bus is 10, but max ΔV is at bus 99
        assert sweep_metrics._max_delta_v_metric(cand, base, ctx) == pytest.approx(0.05)

    def test_none_when_base_missing(self):
        cand = _opflow(buses=[_busres(1, 1.0)])
        assert sweep_metrics._max_delta_v_metric(cand, None, {}) is None


class TestReactiveAdequacyPredicate:

    def test_adequate_when_feasible(self):
        cand = _opflow(converged=True, violations=0)
        ok, reason = sweep_metrics._reactive_adequacy_predicate(cand, None, {})
        assert ok is True
        assert reason == ""

    def test_inadequate_on_violation_with_reason(self):
        cand = _opflow(converged=True, violations=1)
        cand.violation_details = ["Bus 5: Vm=0.88 pu < 0.90 (undervoltage)"]
        ok, reason = sweep_metrics._reactive_adequacy_predicate(cand, None, {})
        assert ok is False
        assert "Pmax" in reason and "Qmax" in reason
        assert "0.88" in reason

    def test_inadequate_on_nonconvergence(self):
        cand = _opflow(converged=False)
        ok, reason = sweep_metrics._reactive_adequacy_predicate(cand, None, {})
        assert ok is False
        assert "did not converge" in reason.lower()


class TestStandardPredicateUnchanged:

    def test_feasible(self):
        ok, reason = sweep_metrics._standard_predicate(_opflow(converged=True, violations=0), None, {})
        assert ok is True and reason == ""

    def test_converged_but_violation(self):
        ok, reason = sweep_metrics._standard_predicate(_opflow(converged=True, violations=2), None, {})
        assert ok is False and reason == "constraint violation"

    def test_not_converged(self):
        ok, reason = sweep_metrics._standard_predicate(_opflow(converged=False), None, {})
        assert ok is False and reason == "did not converge"

    def test_none_candidate(self):
        ok, reason = sweep_metrics._standard_predicate(None, None, {})
        assert ok is False and reason == "did not converge"


# ===========================================================================
# C2/C3 — _augment_generator_mutation
# ===========================================================================

def _make_config(tmp_path, **search_over):
    search_kw = dict(
        max_iterations=5, default_mode="accumulative",
        base_case=BASE_CASE if _has_base_case else tmp_path / "d.m",
        gic_file=None, application="opflow",
    )
    search_kw.update(search_over)
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(backend="openai", model="m", api_key_env="K", openai_base_url=None,
                      ollama_host="h", ollama_cloud_host=None, temperature=0.3, max_tokens=4096),
        search=SearchConfig(**search_kw),
        output=OutputConfig(workdir=tmp_path / "wd", logs_dir=tmp_path / "logs",
                            save_journal=False, journal_format="json",
                            save_modified_files=False, verbose=False),
    )


def _controller(tmp_path, **search_over):
    cfg = _make_config(tmp_path, **search_over)
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_exec_cls.return_value = MagicMock()
        c = AgentLoopController(cfg)
    return c


class TestAugmentGeneratorMutation:

    def test_dispatchable_default_applied(self, tmp_path):
        c = _controller(tmp_path, added_gen_dispatchable_default=True)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0}, {}, None)
        assert mt["dispatchable"] is True

    def test_entity_dispatchable_overrides_config(self, tmp_path):
        c = _controller(tmp_path, added_gen_dispatchable_default=False)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0},
            {"entity_dispatchable": True}, None)
        assert mt["dispatchable"] is True

    def test_explicit_cost_coeffs_forwarded(self, tmp_path):
        c = _controller(tmp_path)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0},
            {"entity_cost_coeffs": [0.01, 30.0, 0.0]}, None)
        assert mt["cost_coeffs"] == [0.01, 30.0, 0.0]

    def test_explicit_strategy_requires_coeffs(self, tmp_path):
        c = _controller(tmp_path, added_gen_cost_strategy="explicit", added_gen_dispatchable_default=True)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0}, {}, None)
        assert mt is None  # error
        assert "explicit" in (c._error_feedback or "")

    def test_reactive_adequacy_pins_q(self, tmp_path):
        c = _controller(tmp_path)
        mt = c._augment_generator_mutation(
            {"action": "add_generator_at_bus", "capacity_mw": 100.0, "Qmax": 60.0},
            {}, "reactive_adequacy")
        assert mt["Qmin"] == 60.0 == mt["Qmax"]  # Q pinned to Qmax

    def test_non_generator_mutation_untouched(self, tmp_path):
        c = _controller(tmp_path)
        original = {"action": "add_load_at_bus", "Pd": 100.0}
        mt = c._augment_generator_mutation(original, {}, None)
        assert mt == original


# ===========================================================================
# Integration — sweep handler with metric / predicate (mocked solves)
# ===========================================================================

def _sim():
    return SimulationResult(success=True, exit_code=0, stdout="ok", stderr="",
                            elapsed_seconds=0.1, input_file=Path("/tmp/x.m"),
                            application="opflow", error_message=None, workdir=Path("/tmp"))


@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestSweepHandlerIntegration:

    def _controller_with_exec(self, tmp_path):
        cfg = _make_config(tmp_path)
        with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
            mock_executor = MagicMock()
            mock_executor.run.return_value = _sim()
            mock_executor.run_parallel.side_effect = (
                lambda tasks, max_workers=4, thread_limit=None, on_progress=None:
                {i: _sim() for i in range(len(tasks))}
            )
            mock_exec_cls.return_value = mock_executor
            c = AgentLoopController(cfg)
        c._base_network = parse_matpower(BASE_CASE)
        return c

    def test_unknown_metric_clean_error(self, tmp_path):
        c = self._controller_with_exec(tmp_path)
        kind, _ = c._handle_sweep(1, {
            "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
            "candidate_set": {"type": "bus_list", "buses": [1, 2]},
            "metric": "totally_made_up",
        })
        assert kind == "error"
        assert "totally_made_up" in (c._error_feedback or "")

    def test_unknown_predicate_clean_error(self, tmp_path):
        c = self._controller_with_exec(tmp_path)
        kind, _ = c._handle_sweep(1, {
            "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
            "candidate_set": {"type": "bus_list", "buses": [1, 2]},
            "feasibility_predicate": "made_up",
        })
        assert kind == "error"
        assert "made_up" in (c._error_feedback or "")

    def test_max_delta_v_sweep_records_metric(self, tmp_path):
        c = self._controller_with_exec(tmp_path)
        base = _opflow(buses=[_busres(1, 1.0), _busres(2, 1.0)])
        cand = _opflow(buses=[_busres(1, 1.02), _busres(2, 0.97)])  # max ΔV = 0.03
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=[base, cand, cand]):  # base solve, then 2 candidates
            kind, ok = c._handle_sweep(1, {
                "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "metric": "max_delta_v",
                "description": "voltage step",
            })
        assert (kind, ok) == ("sweep", True)
        entry = c._journal.get_sweep_entry()
        variants = entry.explored_variants
        assert all(v.get("metric_name") == "max_delta_v" for v in variants)
        assert all(v.get("metric_value") == pytest.approx(0.03) for v in variants)

    def test_default_sweep_has_no_metric_keys(self, tmp_path):
        """Regression: a sweep with no metric/predicate journals an unchanged payload."""
        c = self._controller_with_exec(tmp_path)
        cand = _opflow(converged=True, violations=0)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=cand):
            c._handle_sweep(1, {
                "mutation": {"action": "add_load_at_bus", "Pd": 100.0},
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
            })
        variants = c._journal.get_sweep_entry().explored_variants
        for v in variants:
            assert "metric_value" not in v
            assert "metric_name" not in v
            assert "predicate_name" not in v
            assert "dispatched_pg" not in v

    def test_dispatchable_sweep_records_pg(self, tmp_path):
        c = self._controller_with_exec(tmp_path)
        # candidate result includes the added unit (last gen at the bus) dispatching 120 MW
        def _cand_for(bus):
            return _opflow(converged=True, violations=0, gens=[_genres(bus, 120.0)])
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=[_cand_for(1), _cand_for(2)]):
            kind, ok = c._handle_sweep(1, {
                "mutation": {"action": "add_generator_at_bus", "capacity_mw": 200.0},
                "entity_dispatchable": True,
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "description": "siting",
            })
        assert (kind, ok) == ("sweep", True)
        variants = c._journal.get_sweep_entry().explored_variants
        assert all(v.get("dispatched_pg") == 120.0 for v in variants)
