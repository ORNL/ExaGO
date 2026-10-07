"""Tests for the N-1/N-2 contingency screening engine (C.5).

Covers:
- substation scope: transformers merge buses (0 hops), lines are hops, tiers grow
  from both ends of a changed branch, out-of-service branches are not walked
- neighbor search: the five affected-element lists (buses, branches, gens, loads,
  shunts) with tiers, changed element excluded, branches leaving the last tier
  excluded
- outage pool: joining the lists, N-1 / N-2 counts, command shapes, determinism,
  ValueError guards
- handler: _handle_contingency_sweep wiring — pass/fail derived from the standard
  predicate, journal recorded, no LLM/backend call inside
- verdict: the change passes N-k only if every contingency passes; the LLM view
  leads with the verdict and a table of every failed contingency (plus, for N-2,
  failures counted per element)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine import contingency as C
from agentigrid.engine import topology as T
from agentigrid.engine.commands import parse_command
from agentigrid.engine.modifier import apply_modifications
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.executor import SimulationResult
from agentigrid.parsers.matpower_model import Branch, Bus, Generator, MATNetwork
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.opflow_results import OPFLOWResult

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exago" / "examples"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()


_LOAD_77 = C.ChangedElement(kind="load", bus=77)


@pytest.fixture(scope="module")
def net118():
    return parse_matpower(IEEE118)


# ---------------------------------------------------------------------------
# Synthetic grid (hand-checkable)
# ---------------------------------------------------------------------------
#
#   tier 0          tier 1                 tier 2        tier 3     tier 4
#   [1 =T= 2] --- [11]  (2 circuits) ---- [21] -------- [31] ----- [41]
#        \
#         ------- [12 =T= 13] ----------- [22]
#        \
#         ...x... 50   (1-50 out of service)
#
#   =T= transformer (ratio != 0, same substation, 0 hops); --- line (1 hop).
#   Load at 1 and 21; in-service gen at 2 and 31, out-of-service gen at 11;
#   shunt at 11 (Bs) and 41 (Gs).

def _bus(bus_i, Pd=0.0, Qd=0.0, Gs=0.0, Bs=0.0):
    return Bus(
        bus_i=bus_i, type=1, Pd=Pd, Qd=Qd, Gs=Gs, Bs=Bs,
        area=1, Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9,
    )


def _line(f, t, status=1):
    return Branch(f, t, 0.01, 0.1, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, status, -30.0, 30.0)


def _xfmr(f, t):
    return Branch(f, t, 0.0, 0.05, 0.0, 100.0, 100.0, 100.0, 1.0, 0.0, 1, -30.0, 30.0)


def _gen(bus, status=1):
    return Generator(bus, 50.0, 0.0, 50.0, -50.0, 1.0, 100.0, status, 100.0, 0.0)


@pytest.fixture
def grid():
    return MATNetwork(
        casename="synthetic", version="2", baseMVA=100.0,
        buses=[
            _bus(1, Pd=50.0, Qd=10.0), _bus(2), _bus(11, Bs=10.0), _bus(12), _bus(13),
            _bus(21, Pd=20.0), _bus(22), _bus(31), _bus(41, Gs=5.0), _bus(50),
        ],
        generators=[_gen(2), _gen(11, status=0), _gen(31)],
        branches=[
            _xfmr(1, 2), _line(1, 11), _line(1, 11), _line(1, 12), _xfmr(12, 13),
            _line(11, 21), _line(12, 22), _line(21, 31), _line(31, 41), _line(1, 50, status=0),
        ],
        gencost=[], header_comments="",
    )


LOAD_AT_1 = C.ChangedElement(kind="load", bus=1)


def _pool(net, changed, depth, components=C.OUTAGE_KINDS):
    return C.outages_from(C.find_affected_elements(net, changed, depth), components)


def _labels(pool):
    return [e.label() for e in pool]


class TestSubstations:

    def test_transformer_is_ratio_nonzero(self, grid):
        assert [T.is_transformer(br) for br in grid.branches] == [
            True, False, False, False, True, False, False, False, False, False,
        ]

    def test_transformers_merge_buses(self, grid):
        sub = T.substation_map(grid)
        assert sub[2] == sub[1] == 1
        assert sub[13] == sub[12] == 12
        assert sub[11] == 11

    def test_no_transformers_each_bus_own_substation(self):
        net = MATNetwork(
            casename="flat", version="2", baseMVA=100.0,
            buses=[_bus(1), _bus(2), _bus(3)], generators=[],
            branches=[_line(1, 2), _line(2, 3)], gencost=[], header_comments="",
        )
        assert T.substation_map(net) == {1: 1, 2: 2, 3: 3}

    def test_tiers_count_lines_not_transformers(self, grid):
        scope = T.substations_within(grid, [1], 2)
        assert scope.tiers == [[1], [11, 12], [21, 22]]
        # 13 sits behind a transformer at tier-1 substation 12 → tier 1, not 2.
        assert scope.tier_of[13] == 1
        assert scope.buses == [1, 2, 11, 12, 13, 21, 22]

    def test_out_of_service_line_not_walked(self, grid):
        scope = T.substations_within(grid, [1], 4)
        assert 50 not in scope.substation_of

    def test_depth_zero_is_start_substation_only(self, grid):
        assert T.substations_within(grid, [2], 0).buses == [1, 2]

    def test_branch_start_grows_both_sides(self, grid):
        scope = T.substations_within(grid, [11, 21], 1)
        assert scope.tiers == [[11, 21], [1, 31]]

    def test_stops_when_grid_exhausted(self, grid):
        scope = T.substations_within(grid, [1], 10)
        assert len(scope.tiers) == 5  # tiers 0..4, then nothing new

    def test_errors(self, grid):
        with pytest.raises(ValueError):
            T.substations_within(grid, [999], 1)
        with pytest.raises(ValueError):
            T.substations_within(grid, [1], -1)
        with pytest.raises(ValueError):
            T.substations_within(grid, [], 1)


class TestAffectedElements:
    """The neighbor search output, checked list by list against the sketch above."""

    def test_five_lists_depth2(self, grid):
        a = C.find_affected_elements(grid, LOAD_AT_1, 2)
        assert a.buses == [
            {"bus": 1, "substation": 1, "tier": 0}, {"bus": 2, "substation": 1, "tier": 0},
            {"bus": 11, "substation": 11, "tier": 1}, {"bus": 12, "substation": 12, "tier": 1},
            {"bus": 13, "substation": 12, "tier": 1}, {"bus": 21, "substation": 21, "tier": 2},
            {"bus": 22, "substation": 22, "tier": 2},
        ]
        assert a.branches == [
            {"fbus": 1, "tbus": 2, "ckt": 0, "tier": 0},
            {"fbus": 1, "tbus": 11, "ckt": 0, "tier": 0},
            {"fbus": 1, "tbus": 11, "ckt": 1, "tier": 0},
            {"fbus": 1, "tbus": 12, "ckt": 0, "tier": 0},
            {"fbus": 12, "tbus": 13, "ckt": 0, "tier": 1},
            {"fbus": 11, "tbus": 21, "ckt": 0, "tier": 1},
            {"fbus": 12, "tbus": 22, "ckt": 0, "tier": 1},
        ]
        assert a.gens == [{"bus": 2, "gen_id": 0, "tier": 0}]
        assert a.loads == [{"bus": 21, "tier": 2}]          # load@1 is the changed one
        assert a.shunts == [{"bus": 11, "tier": 1}]

    def test_tiers_and_substations(self, grid):
        a = C.find_affected_elements(grid, LOAD_AT_1, 2)
        assert a.tiers == [[1], [11, 12], [21, 22]]
        assert a.substations == [(1, 0), (11, 1), (12, 1), (21, 2), (22, 2)]

    def test_to_dict_has_exactly_five_lists(self, grid):
        d = C.find_affected_elements(grid, LOAD_AT_1, 2).to_dict()
        assert list(d) == ["buses", "branches", "gens", "loads", "shunts"]

    def test_pool_joins_the_lists(self, grid):
        a = C.find_affected_elements(grid, LOAD_AT_1, 2)
        pool = C.outages_from(a)
        n = len(a.branches) + len(a.gens) + len(a.loads) + len(a.shunts)
        assert len(pool) == n == 10


class TestOutagePool:

    def test_pool_depth2_load_changed(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        assert _labels(pool) == [
            "branch 1-2", "branch 1-11", "branch 1-11 (ckt 1)", "branch 1-12",
            "branch 11-21", "branch 12-13", "branch 12-22",
            "gen@2",
            "load@21",
            "shunt@11",
        ]

    def test_changed_load_excluded_other_elements_kept(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        assert "load@1" not in _labels(pool)
        # Elements in the changed element's own substation are still candidates.
        assert "gen@2" in _labels(pool) and "branch 1-2" in _labels(pool)

    def test_branch_leaving_last_tier_excluded(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        assert "branch 21-31" not in _labels(pool)
        assert "branch 21-31" in _labels(_pool(grid, LOAD_AT_1, 3))

    def test_out_of_service_gen_excluded(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2, ("gen",))
        assert _labels(pool) == ["gen@2"]

    def test_added_gen_excluded_existing_gen_kept(self, grid):
        net, changed = C.apply_mutation(
            grid, {"action": "add_generator_at_bus", "bus": 2, "capacity_mw": 30.0})
        assert changed == C.ChangedElement(kind="gen", bus=2, gen_id=1)
        labels = _labels(_pool(net, changed, 2))
        assert "gen@2#1" not in labels
        assert "gen@2" in labels and "load@1" in labels

    def test_disconnected_branch_excluded_and_both_sides(self, grid):
        net, changed = C.apply_mutation(
            grid, {"action": "set_branch_status", "fbus": 21, "tbus": 11, "status": 0})
        affected = C.find_affected_elements(net, changed, 1)
        assert affected.tiers == [[11, 21], [1, 31]]
        labels = _labels(C.outages_from(affected))
        assert "branch 11-21" not in labels
        assert "branch 21-31" in labels and "branch 1-11" in labels
        assert "gen@31" in labels and "gen@2" in labels

    def test_connected_line_is_walked_and_excluded(self, grid):
        # 1-50 is out of service in the base grid; connecting it brings 50 into tier 0.
        net, changed = C.apply_mutation(
            grid, {"action": "set_branch_status", "fbus": 1, "tbus": 50, "status": 1})
        affected = C.find_affected_elements(net, changed, 0)
        assert affected.tiers == [[1, 50]]
        assert "branch 1-50" not in _labels(C.outages_from(affected))

    def test_parallel_circuit_only_that_circuit(self, grid):
        net, changed = C.apply_mutation(
            grid, {"action": "set_branch_status", "fbus": 1, "tbus": 11, "ckt": 1, "status": 0})
        labels = _labels(_pool(net, changed, 1))
        assert "branch 1-11" in labels and "branch 1-11 (ckt 1)" not in labels

    def test_component_filter(self, grid):
        pool = _pool(grid, LOAD_AT_1, 4, ("shunt",))
        assert _labels(pool) == ["shunt@11", "shunt@41"]

    def test_element_placement(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        by_label = {e.label(): e for e in pool}
        assert by_label["branch 11-21"].tier == 1
        assert by_label["load@21"].tier == 2
        assert by_label["gen@2"].tier == 0

    def test_n1_and_n2_counts(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        assert len(C.enumerate_contingencies(pool, 1)) == 10
        assert len(C.enumerate_contingencies(pool, 2)) == 45  # C(10, 2)

    def test_order_out_of_range_raises(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        with pytest.raises(ValueError):
            C.enumerate_contingencies(pool, 3)

    def test_determinism(self, grid):
        a = C.enumerate_contingencies(_pool(grid, LOAD_AT_1, 3), 2)
        b = C.enumerate_contingencies(_pool(grid, LOAD_AT_1, 3), 2)
        assert [c.label() for c in a] == [c.label() for c in b]


class TestApplyMutation:

    @pytest.mark.parametrize("raw,expected", [
        ({"action": "add_load_at_bus", "bus": 21, "Pd": 50.0}, C.ChangedElement(kind="load", bus=21)),
        ({"action": "set_load", "bus": 21, "Pd": 0.0}, C.ChangedElement(kind="load", bus=21)),
        ({"action": "set_gen_status", "bus": 2, "status": 0}, C.ChangedElement(kind="gen", bus=2, gen_id=0)),
        ({"action": "set_branch_status", "fbus": 12, "tbus": 22, "status": 0},
         C.ChangedElement(kind="branch", fbus=12, tbus=22, ckt=0)),
    ])
    def test_changed_element(self, grid, raw, expected):
        assert C.apply_mutation(grid, raw)[1] == expected

    def test_mutation_applied_to_copy_only(self, grid):
        net, _ = C.apply_mutation(grid, {"action": "add_load_at_bus", "bus": 21, "Pd": 50.0})
        assert {b.bus_i: b.Pd for b in net.buses}[21] == 70.0
        assert {b.bus_i: b.Pd for b in grid.buses}[21] == 20.0

    def test_added_generator_is_last_at_bus(self, grid):
        grid.generators.append(_gen(2))
        _, changed = C.apply_mutation(
            grid, {"action": "add_generator_at_bus", "bus": 2, "capacity_mw": 10.0})
        assert changed.gen_id == 2

    @pytest.mark.parametrize("raw", [
        None,
        {"action": "scale_all_loads", "factor": 1.1},       # not an allowed mutation
        {"action": "add_load_at_bus"},                      # missing bus
        {"action": "add_load_at_bus", "bus": 999, "Pd": 1.0},
        {"action": "set_gen_status", "bus": 12, "status": 0},   # no generator there
        {"action": "set_gen_status", "bus": 2, "gen_id": 5, "status": 0},
        {"action": "set_branch_status", "fbus": 1, "tbus": 31, "status": 0},
        {"action": "set_branch_status", "fbus": 1, "tbus": 11, "ckt": 2, "status": 0},
    ])
    def test_invalid_mutations_raise(self, grid, raw):
        with pytest.raises(ValueError):
            C.apply_mutation(grid, raw)


class TestCommandShapes:

    def test_outage_commands(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2)
        by_kind = {e.kind: e.to_command() for e in pool}
        assert by_kind["branch"]["action"] == "set_branch_status"
        assert by_kind["branch"]["status"] == 0
        assert by_kind["gen"] == {"action": "set_gen_status", "bus": 2, "gen_id": 0, "status": 0}
        assert by_kind["load"] == {"action": "set_load", "bus": 21, "Pd": 0, "Qd": 0}
        assert by_kind["shunt"] == {"action": "set_shunt", "bus": 11, "Gs": 0, "Bs": 0}

    def test_outage_commands_apply(self, grid):
        pool = _pool(grid, LOAD_AT_1, 4, ("shunt", "load"))
        cmds = [parse_command(e.to_command()) for e in pool]
        out, _ = apply_modifications(grid, cmds, application="opflow")
        by_bus = {b.bus_i: b for b in out.buses}
        assert (by_bus[11].Gs, by_bus[11].Bs) == (0, 0)
        assert (by_bus[41].Gs, by_bus[41].Bs) == (0, 0)
        assert (by_bus[21].Pd, by_bus[21].Qd) == (0, 0)
        assert by_bus[1].Pd == 50.0  # changed load untouched

    def test_contingency_label_and_commands(self, grid):
        pool = _pool(grid, LOAD_AT_1, 2, ("branch",))
        n2 = C.enumerate_contingencies(pool, 2)
        assert n2[0].label().endswith(" out") and " + " in n2[0].label()
        assert len(n2[0].commands()) == 2


# ---------------------------------------------------------------------------
# Real IEEE 118-bus case
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestIEEE118:

    def test_scope_contains_tiers_around_77(self, net118):
        scope = T.substations_within(net118, [77], 3)
        assert scope.tiers[0] == [T.substation_map(net118)[77]]
        assert len(scope.tiers) == 4

    def test_pool_has_no_branch_leaving_scope(self, net118):
        affected = C.find_affected_elements(net118, C.ChangedElement(kind="load", bus=77), 3)
        pool = C.outages_from(affected)
        in_scope = {b["bus"] for b in affected.buses}
        for e in pool:
            if e.kind == "branch":
                assert e.fbus in in_scope and e.tbus in in_scope
        assert "load@77" not in _labels(pool)

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

    def _n_expected(self, depth, order):
        pool = _pool(parse_matpower(IEEE118), _LOAD_77, depth)
        return len(C.enumerate_contingencies(pool, order))

    def test_n1_screen_all_pass(self, tmp_path):
        controller, backend_mock = self._controller(tmp_path)
        n = self._n_expected(1, 1)
        calls_before = len(backend_mock.mock_calls)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=True)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 1,
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
                "description": "N-1 screen around bus 77",
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.mode == "contingency"
        assert len(entry.explored_variants) == n
        assert entry.contingency_meta["passed_count"] == n
        assert entry.contingency_meta["failed_count"] == 0
        assert entry.contingency_meta["mutation"] == {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0}
        assert entry.contingency_meta["changed_element"] == {"type": "load", "bus": 77}
        assert entry.contingency_meta["substation_depth"] == 1
        aff = entry.contingency_meta["affected"]
        assert set(aff) == {"buses", "branches", "gens", "loads", "shunts"}
        assert {"bus": 77, "substation": 77, "tier": 0} in aff["buses"]
        assert all(ld["bus"] != 77 for ld in aff["loads"])
        assert entry.contingency_meta["order"] == 1
        assert all("load@77" not in v["label"] for v in entry.explored_variants)
        assert "around changed element load@77" in controller._latest_results_text
        assert entry.contingency_meta["passes"] is True
        assert entry.feasible is True
        assert f"VERDICT: load@77 passes N-1: all {n} contingencies passed." in (
            controller._latest_results_text
        )
        # No LLM/backend call inside the screen.
        assert len(backend_mock.mock_calls) == calls_before

    def test_n1_screen_all_fail_when_infeasible(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        n = self._n_expected(1, 1)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 1,
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.contingency_meta["passed_count"] == 0
        assert entry.contingency_meta["failed_count"] == n
        assert f"FAILED: {n} / {n}" in controller._latest_results_text
        # The reference solve is infeasible too, so the verdict names that first.
        assert entry.contingency_meta["passes"] is False
        assert entry.feasible is False
        assert "does NOT pass N-1: the operating point with the change is already infeasible" in (
            controller._latest_results_text
        )
        assert f"FAILED contingencies ({n} of {n}):" in controller._latest_results_text

    def test_n2_screen_count(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        n = self._n_expected(1, 2)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=True)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 2,
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert len(entry.explored_variants) == n
        assert entry.contingency_meta["order"] == 2
        assert "Elements in failed contingencies" not in controller._latest_results_text

    def test_n2_failures_counted_per_element(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=False)):
            controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 2,
            })
        assert "Elements in failed contingencies" in controller._latest_results_text

    def test_view_one_failure(self, tmp_path):
        """One failed contingency out of three: verdict fails and the table names it."""
        controller, _ = self._controller(tmp_path)
        net = parse_matpower(IEEE118)
        affected = C.find_affected_elements(net, _LOAD_77, 1)
        ctgs = C.enumerate_contingencies(C.outages_from(affected), 1)[:3]
        summaries = [
            {"label": c.label(), "kinds": [e.kind for e in c.elements],
             "tiers": [e.tier for e in c.elements], "passed": i != 1,
             "voltage_min": 0.95, "voltage_max": 1.05, "max_line_loading_pct": 80.0,
             "reason": "" if i != 1 else "did not converge"}
            for i, c in enumerate(ctgs)
        ]
        passes, verdict = C.screen_verdict(_LOAD_77, 1, 3, 1, True)
        assert passes is False
        text = controller._build_contingency_llm_view(
            changed=_LOAD_77, affected=affected, order=1, components=list(C.OUTAGE_KINDS),
            vmin=0.9, vmax=1.1, contingencies=ctgs, contingency_summaries=summaries,
            passed_count=2, failed_count=1, ref_passed=True, ref_reason="",
            verdict=verdict, threshold=0, top_n=25,
        )
        assert "VERDICT: load@77 does NOT pass N-1: 1 of 3 contingencies failed." in text
        failed_block = text.split("FAILED contingencies (1 of 3):")[1].split("\n\n")[0]
        assert ctgs[1].label() in failed_block
        assert ctgs[0].label() not in failed_block

    def test_default_depth_is_3(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        with patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(feasible=True)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert entry.contingency_meta["substation_depth"] == 3
        assert len(entry.explored_variants) == self._n_expected(3, 1)

    def test_missing_mutation_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {"mode": "contingency"})
        assert kind == "error"
        assert "mutation" in (controller._error_feedback or "")

    def test_invalid_mutation_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 99999, "Pd": 50.0},
        })
        assert kind == "error"
        assert "99999" in (controller._error_feedback or "")

    def test_bad_depth_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            "substation_depth": -1,
        })
        assert kind == "error"
        assert "substation_depth" in (controller._error_feedback or "")

    def test_bad_order_errors(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            "contingency_order": 3,
        })
        assert kind == "error"
        assert "contingency_order" in (controller._error_feedback or "")

    def test_runaway_guard(self, tmp_path):
        import dataclasses
        controller, _ = self._controller(tmp_path)
        # Force a tiny guard (SearchConfig/AppConfig are frozen → rebuild them) so
        # the N-2 study trips it.
        new_search = dataclasses.replace(controller._config.search, contingency_max_count=50)
        controller._config = dataclasses.replace(controller._config, search=new_search)
        n = self._n_expected(1, 2)
        assert n > 50
        with patch.object(controller, "_run_contingency_screen") as mock_screen:
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 2,
            })
        # Stops without solving and says the study cannot be completed; not an error
        # that invites the agent to shrink the study.
        assert (kind, ok) == ("sweep", True)
        mock_screen.assert_not_called()
        verdict = (
            f"VERDICT: The number of contingencies ({n}) exceeds 50, "
            "the study cannot be completed with existing resources."
        )
        assert verdict in controller._latest_results_text
        entry = controller._journal.entries[-1]
        assert entry.contingency_meta["verdict"] == verdict
        assert entry.contingency_meta["passes"] is False
        assert entry.explored_variants == []

    def test_dispatch_routes_contingency_mode(self, tmp_path):
        """_handle_sweep must route mode=contingency to the contingency handler."""
        controller, _ = self._controller(tmp_path)
        with patch.object(controller, "_handle_contingency_sweep",
                          return_value=("sweep", True)) as mock_h:
            controller._handle_sweep(1, {"mode": "contingency",
                                         "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0}})
        mock_h.assert_called_once()

    def test_non_opflow_rejected(self, tmp_path):
        import dataclasses
        controller, _ = self._controller(tmp_path)
        new_search = dataclasses.replace(controller._config.search, application="pflow")
        controller._config = dataclasses.replace(controller._config, search=new_search)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
        })
        assert kind == "error"


class TestAddedLoad:
    """The load a mutation adds is the development under study; relief protects it."""

    def test_add_load_returns_increase_only(self):
        net = parse_matpower(IEEE118) if _has_118 else None
        if net is None:
            pytest.skip("ieee_118_bus_v10.m not available")
        mutated, changed = C.apply_mutation(net, {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0})
        dp, dq = C.added_load(net, mutated, changed)[77]
        assert dp == pytest.approx(50.0)

    def test_generator_mutation_protects_nothing(self):
        net = parse_matpower(IEEE118) if _has_118 else None
        if net is None:
            pytest.skip("ieee_118_bus_v10.m not available")
        mutated, changed = C.apply_mutation(net, {"action": "add_generator_at_bus", "bus": 77, "capacity_mw": 50.0})
        assert C.added_load(net, mutated, changed) == {}


class TestVerdict:
    """The change passes N-k only if every contingency passes."""

    def test_all_pass(self):
        assert C.screen_verdict(_LOAD_77, 1, 25, 0, True) == (
            True, "VERDICT: load@77 passes N-1: all 25 contingencies passed.",
        )

    def test_one_failure_fails(self):
        passes, line = C.screen_verdict(_LOAD_77, 1, 25, 1, True)
        assert passes is False
        assert line == "VERDICT: load@77 does NOT pass N-1: 1 of 25 contingencies failed."

    def test_infeasible_reference_fails(self):
        passes, line = C.screen_verdict(_LOAD_77, 2, 25, 0, False, "voltage out of band")
        assert passes is False
        assert "does NOT pass N-2" in line and "voltage out of band" in line


class TestFailedElementCounts:

    def test_counts_and_order(self):
        a = C.OutageElement(kind="branch", tier=0, fbus=1, tbus=2, ckt=0)
        b = C.OutageElement(kind="branch", tier=1, fbus=2, tbus=3, ckt=0)
        g = C.OutageElement(kind="gen", tier=1, bus=3, gen_id=0)
        ctgs = C.enumerate_contingencies([a, b, g], 2)  # a+b, a+g, b+g
        rows = C.failed_element_counts(ctgs, [False, False, True])
        assert [(e.label(), n, of) for e, n, of in rows] == [
            ("branch 1-2", 2, 2), ("branch 2-3", 1, 2), ("gen@3", 1, 2),
        ]

    def test_no_failures(self):
        a = C.OutageElement(kind="load", tier=0, bus=5)
        assert C.failed_element_counts(C.enumerate_contingencies([a], 1), [True]) == []


class TestDescribeScope:
    """The one-line scope text shown in the launcher and PDF report."""

    def test_new_meta(self, grid):
        from agentigrid.engine.journal import describe_contingency_scope
        a = C.find_affected_elements(grid, LOAD_AT_1, 2)
        text = describe_contingency_scope({
            "changed_element": LOAD_AT_1.to_dict(), "substation_depth": 2, "affected": a.to_dict(),
        })
        assert text == (
            "around load at bus 1; 5 substations in 3 tier(s) within substation depth 2 "
            "(7 buses, 7 branches, 1 generators, 1 loads, 1 shunts)"
        )

    def test_meta_without_affected_record(self):
        from agentigrid.engine.journal import describe_contingency_scope
        text = describe_contingency_scope({"order": 1})
        assert text.startswith("study scope not recorded")
