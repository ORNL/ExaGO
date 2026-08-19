"""Tests for the per-candidate boundary (hosting-capacity) sweep — C.1.

Covers:
- system-average power factor + PF-spec resolution
- candidate-mutation helper (constant-PF load ray, generator fixed injection)
- binding-constraint identification at the max-feasible point
- the pure bisection algorithm (synthetic feasibility oracle)
- the token-bounded boundary LLM view
- executor.map_callables (parallel candidate callables)
- _handle_boundary_sweep orchestration: journal wiring, one-LLM-turn invariant,
  base-infeasibility abort
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import (
    AgentLoopController,
    _ProbeOutcome,
    _bisect_boundary,
    _build_boundary_llm_view,
    _identify_binding,
    _mutate_candidate_network,
    _system_average_tan_phi,
    _tan_phi_from_pf_spec,
)
from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import (
    OPFLOWResult, BranchResult, BusResult,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_CASE = DATA_DIR / "case_ACTIVSg200.m"
_has_base_case = BASE_CASE.exists()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

class _Bus:
    def __init__(self, pd, qd):
        self.Pd = pd
        self.Qd = qd


class _Net:
    def __init__(self, buses):
        self.buses = buses


def _opflow(
    converged=True, max_load=50.0, vmin=0.95, vmax=1.05,
    branches=None, buses=None,
) -> OPFLOWResult:
    return OPFLOWResult(
        converged=converged,
        objective_value=1000.0,
        convergence_status="CONVERGED" if converged else "DID NOT CONVERGE",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=10, solve_time=0.1,
        branches=branches or [], buses=buses or [],
        voltage_min=vmin, voltage_max=vmax, max_line_loading_pct=max_load,
        num_violations=0,
        feasibility_detail="feasible" if converged else "infeasible",
    )


def _branch(fb, tb, sf, slim):
    return BranchResult(from_bus=fb, to_bus=tb, status=1, Sf=sf, St=sf,
                        Slim=slim, mult_Sf=0.0, mult_St=0.0)


def _busres(bus_id, vm):
    return BusResult(bus_id=bus_id, Pd=0, Pd_loss=0, Qd=0, Qd_loss=0,
                     Vm=vm, Va=0, mult_Pmis=0, mult_Qmis=0, Pslack=0, Qslack=0)


# ---------------------------------------------------------------------------
# System PF + PF-spec
# ---------------------------------------------------------------------------

class TestSystemPF:

    def test_system_average_tan_phi(self):
        net = _Net([_Bus(100, 30), _Bus(50, 20)])
        # ΣQd/ΣPd = 50/150
        assert _system_average_tan_phi(net) == pytest.approx(50.0 / 150.0)

    def test_zero_total_load_guarded(self):
        net = _Net([_Bus(0, 0), _Bus(0, 5)])
        assert _system_average_tan_phi(net) == 0.0

    def test_pf_spec_system_average(self):
        assert _tan_phi_from_pf_spec("system_average", 0.42) == 0.42
        assert _tan_phi_from_pf_spec(None, 0.42) == 0.42

    def test_pf_spec_unity(self):
        assert _tan_phi_from_pf_spec("unity", 0.42) == 0.0

    def test_pf_spec_numeric(self):
        import math
        assert _tan_phi_from_pf_spec(0.95, 0.42) == pytest.approx(math.tan(math.acos(0.95)))

    def test_pf_spec_numeric_string(self):
        import math
        assert _tan_phi_from_pf_spec("0.9", 0.42) == pytest.approx(math.tan(math.acos(0.9)))


# ---------------------------------------------------------------------------
# Candidate mutation (real ACTIVSg200 network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestMutateCandidate:

    @pytest.fixture(scope="class")
    def net(self):
        return parse_matpower(BASE_CASE)

    def test_load_pf_ray(self, net):
        tan_avg = _system_average_tan_phi(net)
        bus = net.buses[10].bus_i
        m = _mutate_candidate_network(net, bus, "load", 100.0, tan_avg, 0.9, 1.1, 0.4)
        before = next(b for b in net.buses if b.bus_i == bus)
        after = next(b for b in m.buses if b.bus_i == bus)
        d_pd = after.Pd - before.Pd
        d_qd = after.Qd - before.Qd
        assert d_pd == pytest.approx(100.0)
        # ΔQ / ΔP must equal the system-average tan(phi)
        assert d_qd / d_pd == pytest.approx(tan_avg)

    def test_load_unity_pf_no_reactive(self, net):
        bus = net.buses[10].bus_i
        m = _mutate_candidate_network(net, bus, "load", 100.0, 0.0, 0.9, 1.1, 0.4)
        before = next(b for b in net.buses if b.bus_i == bus)
        after = next(b for b in m.buses if b.bus_i == bus)
        assert (after.Qd - before.Qd) == pytest.approx(0.0)

    def test_generator_fixed_injection(self, net):
        bus = net.buses[10].bus_i
        m = _mutate_candidate_network(net, bus, "generator", 100.0, 0.0, 0.9, 1.1, 0.4)
        g = m.generators[-1]
        assert g.Pmin == g.Pmax == 100.0  # forced injection
        assert g.Qmax == pytest.approx(40.0)   # 0.4 * 100
        assert g.Qmin == pytest.approx(-40.0)
        # gencost stays aligned with generators
        assert len(m.gencost) == len(m.generators)

    def test_base_network_untouched(self, net):
        n_gens_before = len(net.generators)
        _mutate_candidate_network(net, net.buses[5].bus_i, "generator", 50.0, 0.0, 0.9, 1.1, 0.4)
        assert len(net.generators) == n_gens_before


# ---------------------------------------------------------------------------
# Binding-constraint identification
# ---------------------------------------------------------------------------

class TestIdentifyBinding:

    def test_thermal_binding_with_element(self):
        branches = [_branch(29, 30, 99.7, 100.0), _branch(1, 2, 40.0, 100.0)]
        o = _opflow(max_load=99.7, vmin=0.97, vmax=1.03, branches=branches)
        result = _identify_binding(o, 0.9, 1.1)
        assert result.startswith("thermal")
        assert "29->30" in result

    def test_voltage_low_binding_with_bus(self):
        buses = [_busres(5, 0.901), _busres(6, 1.00)]
        o = _opflow(max_load=40.0, vmin=0.901, vmax=1.00, buses=buses)
        result = _identify_binding(o, 0.9, 1.1)
        assert result.startswith("voltage")
        assert "bus 5" in result
        assert "Vmin" in result

    def test_voltage_high_binding_with_bus(self):
        buses = [_busres(7, 1.099), _busres(8, 1.00)]
        o = _opflow(max_load=40.0, vmin=1.00, vmax=1.099, buses=buses)
        result = _identify_binding(o, 0.9, 1.1)
        assert result.startswith("voltage")
        assert "bus 7" in result
        assert "Vmax" in result

    def test_type_only_when_no_arrays(self):
        o = _opflow(max_load=99.9, vmin=0.95, vmax=1.05)
        result = _identify_binding(o, 0.9, 1.1)
        assert result.startswith("thermal")

    def test_none_opflow_is_convergence(self):
        assert _identify_binding(None, 0.9, 1.1) == "convergence"


# ---------------------------------------------------------------------------
# Bisection algorithm (synthetic oracle)
# ---------------------------------------------------------------------------

class TestBisectBoundary:

    def _oracle(self, true_boundary):
        def solve(delta):
            feasible = delta <= true_boundary
            return _ProbeOutcome(
                feasible=feasible,
                voltage_min=0.96, voltage_max=1.04,
                max_line_loading_pct=99.0 if feasible else 105.0,
                status="CONVERGED" if feasible else "DID NOT CONVERGE",
                binding="thermal @ 99.0%" if feasible else "",
            )
        return solve

    def test_finds_boundary_within_tol(self):
        r = _bisect_boundary(self._oracle(137.5), 50.0, 2000.0, 1.0, 24)
        assert abs(r["max_feasible_mw"] - 137.5) <= 1.0
        assert r["max_feasible_mw"] <= 137.5  # never reports an infeasible value
        assert r["binding_constraint"] == "thermal @ 99.0%"

    def test_probes_bounded(self):
        r = _bisect_boundary(self._oracle(137.5), 50.0, 2000.0, 1.0, 24)
        assert r["probes_used"] <= 24

    def test_probe_budget_respected_tight(self):
        r = _bisect_boundary(self._oracle(137.5), 50.0, 2000.0, 0.0001, 6)
        assert r["probes_used"] <= 6

    def test_cap_not_reached(self):
        # boundary above the cap → report cap with a note
        r = _bisect_boundary(self._oracle(5000.0), 50.0, 2000.0, 1.0, 24)
        assert r["max_feasible_mw"] == 2000.0
        assert "cap" in r["note"]

    def test_first_probe_infeasible(self):
        # even 50 MW infeasible → boundary bisected down toward 0
        r = _bisect_boundary(self._oracle(10.0), 50.0, 2000.0, 1.0, 24)
        assert r["max_feasible_mw"] <= 10.0

    def test_convergence_as_infeasible(self):
        """Non-convergence is the infeasible signal — a probe that does not
        converge caps the bracket exactly like a physical violation."""
        calls = {"n": 0}

        def solve(delta):
            calls["n"] += 1
            converged = delta <= 200.0
            return _ProbeOutcome(
                feasible=converged,
                status="CONVERGED" if converged else "DID NOT CONVERGE",
                binding="voltage @ 0.900 pu" if converged else "",
            )

        r = _bisect_boundary(solve, 50.0, 2000.0, 1.0, 24)
        assert abs(r["max_feasible_mw"] - 200.0) <= 1.0


# ---------------------------------------------------------------------------
# Token-bounded boundary LLM view
# ---------------------------------------------------------------------------

def _bcand(bus, mfm, binding="thermal @ 99%", reason=""):
    return {
        "bus": bus, "feasible": mfm is not None, "max_feasible_mw": mfm,
        "binding_constraint": binding, "probes_used": 7,
        "voltage_min": 0.95, "voltage_max": 1.05, "max_line_loading_pct": 99.0,
        "reason": reason,
    }


class TestBoundaryLLMView:

    def test_full_table_below_threshold(self):
        cands = [_bcand(i, 100.0 + i) for i in range(10)]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=25, threshold=250)
        # full table: every bus present, no pointer line
        for i in range(10):
            assert str(100 + i) in txt or str(i) in txt
        assert "journal" not in txt.lower()

    def test_summary_above_threshold_is_bounded(self):
        cands = [_bcand(i, 100.0 + i) for i in range(300)]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=25, threshold=250)
        # ranked rows bounded at top_n
        ranked = [l for l in txt.splitlines() if l.strip() and l.strip()[0].isdigit() and "|" in l]
        assert len(ranked) == 25
        assert "journal" in txt.lower()
        assert "stats" in txt.lower()

    def test_summary_ranks_capacity_descending(self):
        cands = [_bcand(1, 100.0), _bcand(2, 300.0), _bcand(3, 200.0)]
        cands += [_bcand(100 + i, 10.0) for i in range(300)]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=3, threshold=250)
        ranked = [l for l in txt.splitlines() if l.strip() and l.strip()[0].isdigit() and "|" in l]
        # highest capacity (bus 2, 300 MW) ranked first
        assert ranked[0].split("|")[0].strip() == "2"

    def test_size_bound_2000(self):
        cands = [_bcand(i, 100.0 + i) for i in range(2000)]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=25, threshold=250)
        ranked = [l for l in txt.splitlines() if l.strip() and l.strip()[0].isdigit() and "|" in l]
        assert len(ranked) == 25

    def test_undetermined_grouped(self):
        cands = [_bcand(i, 100.0 + i) for i in range(260)]
        cands += [_bcand(900, None, reason="bisection error"),
                  _bcand(901, None, reason="bisection error")]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=25, threshold=250)
        assert "Undetermined" in txt
        assert "bisection error" in txt

    def test_threshold_zero_forces_summary(self):
        cands = [_bcand(i, 100.0 + i) for i in range(5)]
        txt = _build_boundary_llm_view(cands, "boundary load", top_n=25, threshold=0)
        assert "journal" in txt.lower()


# ---------------------------------------------------------------------------
# executor.map_callables
# ---------------------------------------------------------------------------

class TestMapCallables:

    def _executor(self, tmp_path):
        exago = ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        )
        out = OutputConfig(
            workdir=tmp_path / "wd", logs_dir=tmp_path / "logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False,
        )
        return SimulationExecutor(exago, out)

    def test_preserves_index_order(self, tmp_path):
        ex = self._executor(tmp_path)
        fns = [(lambda i=i: i * 10) for i in range(5)]
        results = ex.map_callables(fns, max_workers=3)
        assert results == {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}

    def test_captures_exceptions(self, tmp_path):
        ex = self._executor(tmp_path)

        def boom():
            raise ValueError("kaboom")

        results = ex.map_callables([lambda: 1, boom], max_workers=2)
        assert results[0] == 1
        assert isinstance(results[1], Exception)

    def test_progress_callback_counts(self, tmp_path):
        ex = self._executor(tmp_path)
        seen = []
        ex.map_callables(
            [(lambda i=i: i) for i in range(4)], max_workers=2,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        assert len(seen) == 4
        assert seen[-1] == (4, 4)

    def test_empty_returns_empty(self, tmp_path):
        ex = self._executor(tmp_path)
        assert ex.map_callables([]) == {}


# ---------------------------------------------------------------------------
# _handle_boundary_sweep orchestration (mocked solves)
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
            base_case=BASE_CASE, gic_file=None, application="opflow",
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


@pytest.mark.skipif(not _has_base_case, reason="ACTIVSg200 base case not found")
class TestBoundaryHandlerIntegration:

    def _controller(self, tmp_path):
        cfg = _make_config(tmp_path)
        with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
            mock_executor = MagicMock()
            mock_executor.run.return_value = _sim_result()
            # map_callables runs the (patched) bisection callables synchronously
            mock_executor.map_callables.side_effect = (
                lambda fns, max_workers=4, on_progress=None: {i: fn() for i, fn in enumerate(fns)}
            )
            mock_exec_cls.return_value = mock_executor
            controller = AgentLoopController(cfg)
        controller._base_network = parse_matpower(BASE_CASE)
        return controller

    def test_boundary_sweep_journals_and_one_turn(self, tmp_path):
        controller = self._controller(tmp_path)

        # Canned per-candidate bisection results.
        def fake_bisect(bus, *a, **k):
            return {
                "max_feasible_mw": 100.0 + bus, "binding_constraint": "thermal: line 1->2 @ 99%",
                "probes_used": 6, "voltage_min": 0.95, "voltage_max": 1.05,
                "max_line_loading_pct": 99.0, "bus": bus, "entity": "load", "note": "",
            }
        controller._bisect_candidate = fake_bisect

        feasible_base = _opflow(converged=True)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=feasible_base):
            kind, ok = controller._handle_boundary_sweep(1, {
                "entity": "load",
                "power_factor": "system_average",
                "candidate_set": {"type": "bus_list", "buses": [1, 2, 3]},
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
                "description": "max load hosting",
            })

        assert (kind, ok) == ("sweep", True)
        # One journal entry for the whole sweep (one LLM turn, not N).
        sweep_entry = controller._journal.get_sweep_entry()
        assert sweep_entry is not None
        assert sweep_entry.candidate_count == 3
        variants = sweep_entry.explored_variants
        assert len(variants) == 3
        assert all("max_feasible_mw" in v for v in variants)
        assert {v["bus"] for v in variants} == {1, 2, 3}
        # determined buses recorded
        assert sorted(sweep_entry.feasible_buses) == [1, 2, 3]
        # LLM-facing text mentions hosting capacity
        assert controller._latest_results_text is not None
        assert "boundary" in controller._latest_results_text.lower()

    def test_base_infeasible_aborts(self, tmp_path):
        controller = self._controller(tmp_path)
        infeasible_base = _opflow(converged=False)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=infeasible_base):
            kind, ok = controller._handle_boundary_sweep(1, {
                "entity": "load",
                "candidate_set": {"type": "bus_list", "buses": [1, 2]},
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert kind == "error"
        assert "base case is infeasible" in (controller._error_feedback or "")

    def test_invalid_entity_rejected(self, tmp_path):
        controller = self._controller(tmp_path)
        kind, ok = controller._handle_boundary_sweep(1, {
            "entity": "shunt",
            "candidate_set": {"type": "bus_list", "buses": [1]},
        })
        assert kind == "error"
        assert "entity" in (controller._error_feedback or "")


# ---------------------------------------------------------------------------
# C.1 correction — binding-constraint identification (duals + margin-vs-base)
# ---------------------------------------------------------------------------

from agentigrid.engine.agent_loop import _identify_binding, _bisect_boundary, _ProbeOutcome


def _branch_ends(fb, tb, sf, st, slim, mult_sf=0.0, mult_st=0.0):
    """Branch with distinct from/to flows and optional flow-limit multipliers."""
    return BranchResult(from_bus=fb, to_bus=tb, status=1, Sf=sf, St=st,
                        Slim=slim, mult_Sf=mult_sf, mult_St=mult_st)


class TestBindingThermalGlobalMax:

    def test_reports_global_max_not_local(self):
        """The most-loaded line (100% via its To-end, NOT incident to the injection
        bus) must be named, not a local line at 96%."""
        branches = [
            _branch_ends(1, 124, 96.0, 90.0, 100.0),    # local to injection bus 1, 96%
            _branch_ends(66, 158, 50.0, 100.0, 100.0),  # remote line, 100% on the To-end
        ]
        o = _opflow(max_load=100.0, vmin=0.97, vmax=1.03, branches=branches)
        label = _identify_binding(o, 0.9, 1.1)
        assert "66->158" in label
        assert "1->124" not in label
        assert "100.0%" in label

    def test_uses_both_ends_for_loading(self):
        """Loading must use max(|Sf|,|St|): a line at 100% on the To-end is binding
        even though its From-end reads 96%."""
        branches = [_branch_ends(5, 9, 96.0, 100.0, 100.0)]
        o = _opflow(max_load=100.0, branches=branches)
        label = _identify_binding(o, 0.9, 1.1)
        assert "100.0%" in label and "5->9" in label

    def test_not_thermal_when_below_limit_and_no_duals(self):
        """No line at the limit and no duals → no spurious thermal label."""
        branches = [_branch_ends(1, 2, 80.0, 82.0, 100.0)]
        o = _opflow(max_load=82.0, vmin=0.96, vmax=1.04, branches=branches, buses=[_busres(1, 1.00)])
        label = _identify_binding(o, 0.9, 1.1)
        assert "thermal:" not in label  # falls back, but not a fabricated thermal binder


class TestBindingDuals:

    def test_largest_multiplier_chosen(self):
        """With duals present, the genuinely-active line (largest |multiplier|) is
        chosen even if another line reads a marginally higher loading."""
        branches = [
            _branch_ends(10, 11, 99.0, 99.0, 100.0, mult_sf=0.0, mult_st=0.0),   # at limit, slack
            _branch_ends(20, 21, 95.0, 95.0, 100.0, mult_sf=12.5, mult_st=0.0),  # active
        ]
        o = _opflow(max_load=99.0, branches=branches)
        label = _identify_binding(o, 0.9, 1.1)
        assert "20->21" in label          # active line wins
        assert "10->11" not in label      # zero-multiplier line ignored

    def test_zero_multiplier_pinned_line_ignored(self):
        branches = [_branch_ends(3, 4, 100.0, 100.0, 100.0, mult_sf=0.0, mult_st=0.0)]
        o = _opflow(max_load=100.0, branches=branches)
        # no duals anywhere → value-based path still names it (it IS at 100%)
        label = _identify_binding(o, 0.9, 1.1)
        assert "3->4" in label


class TestBindingVoltageMarginVsBase:

    def test_pinned_voltage_excluded(self):
        """A bus at Vmax=1.10 in BOTH base and boundary (pinned PV setpoint) must
        NOT be reported as binding."""
        base = _opflow(buses=[_busres(100, 1.100), _busres(5, 1.00)],
                       branches=[_branch_ends(1, 2, 50.0, 50.0, 100.0)])
        cand = _opflow(buses=[_busres(100, 1.100), _busres(5, 1.02)],
                       branches=[_branch_ends(1, 2, 70.0, 70.0, 100.0)])
        label = _identify_binding(cand, 0.90, 1.10, base_opflow=base)
        assert "bus 100" not in label  # pinned setpoint excluded
        assert "Vmax" not in label

    def test_collapsed_margin_reported(self):
        """A bus whose Vmin margin collapses from positive (base) to ~0 (boundary)
        IS reported."""
        base = _opflow(buses=[_busres(7, 0.95)])   # margin 0.05 above Vmin
        cand = _opflow(buses=[_busres(7, 0.901)])  # margin ~0 at boundary
        label = _identify_binding(cand, 0.90, 1.10, base_opflow=base)
        assert "bus 7" in label and "Vmin" in label

    def test_both_thermal_and_voltage_reported(self):
        """If a line is at 100% AND a voltage bound newly collapses, report both."""
        base = _opflow(buses=[_busres(7, 0.95)], branches=[_branch_ends(1, 2, 50.0, 50.0, 100.0)])
        cand = _opflow(
            buses=[_busres(7, 0.901)],
            branches=[_branch_ends(8, 9, 100.0, 100.0, 100.0)],
            max_load=100.0,
        )
        label = _identify_binding(cand, 0.90, 1.10, base_opflow=base)
        assert "thermal:" in label and "8->9" in label
        assert "voltage:" in label and "bus 7" in label


class TestBindingDoesNotAffectCapacity:

    def test_capacity_invariant_to_binding_label(self):
        """The bisection consumes only `.feasible`; the binding string must not move
        any max_feasible_mw. Two oracles with identical feasibility but different
        binding labels must yield the same capacity."""
        def make_oracle(binding_str):
            def solve(delta):
                feasible = delta <= 137.5
                return _ProbeOutcome(
                    feasible=feasible,
                    max_line_loading_pct=99.0 if feasible else 105.0,
                    binding=binding_str if feasible else "",
                )
            return solve

        r1 = _bisect_boundary(make_oracle("thermal: line 1->2 @ 99.0%"), 50.0, 2000.0, 1.0, 24)
        r2 = _bisect_boundary(make_oracle("voltage: bus 7 Vmin @ 0.901 pu"), 50.0, 2000.0, 1.0, 24)
        assert r1["max_feasible_mw"] == r2["max_feasible_mw"]
