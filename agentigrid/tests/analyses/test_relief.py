"""Tests for the contingency relief-measure search (C.7, Option A).

Covers:
- priority order respected (first resolving measure returned; earlier ones logged failed)
- generator_redispatch is not a measure (committed units already redispatch in OPF)
- line_switching switches nearby lines out, and open ones in (never the outaged ones)
- load_curtailment sheds tier by tier around the outage, at most
  relief_shed_max_fraction of each bus, bisecting the share in the first tier that works;
  the load the mutation added is never shed
- relief guard: relief_measures is dropped unless the goal (or a steering directive)
  asks for relief; a dropped request is served from the plain screen's cache
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
from agentigrid.exago_engine import contingency as C
from agentigrid.exago_engine import relief
from agentigrid.exago_engine import topology as T
from agentigrid.exago_engine.agent_loop import AgentLoopController
from agentigrid.exago_engine.executor import SimulationResult
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.exago_parsers.opflow_results import OPFLOWResult

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exago" / "examples"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()

_CFG = SimpleNamespace(
    relief_tap_steps=[0.90, 0.95, 1.00, 1.05, 1.10],
    relief_curtail_tol_mw=1.0,
    relief_shed_max_fraction=0.10,
    relief_shed_max_depth=3,
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
    # A single branch outage (47-69) in the substations around a load at bus 77.
    affected = C.find_affected_elements(net118, C.ChangedElement(kind="load", bus=77), 3)
    pool = C.outages_from(affected, ("branch",))
    ctgs = C.enumerate_contingencies(pool, order=1)
    return next(c for c in ctgs if c.label() == "branch 47-69 out")


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
            ["transformer_ratio", "line_switching"],
            0.9, 1.1, solve_fn, _CFG,
        )
        assert res.resolved is True
        assert res.action.measure == "line_switching"
        assert res.action.detail.startswith("switch out branch")
        # Earlier measures recorded as attempted-and-failed, in order.
        assert res.attempts == [("transformer_ratio", False), ("line_switching", True)]

    def test_generator_redispatch_is_not_a_measure(self):
        assert "generator_redispatch" not in relief.MEASURE_ORDER

    def test_line_switching_switches_open_line_in(self, net118, branch_contingency):
        """An open line at the contingency's buses is switched back in."""
        import copy
        net = copy.deepcopy(net118)
        focus = relief._focus_buses(branch_contingency)
        outaged = relief._outaged_branch_keys(branch_contingency)
        spare, key = next(
            (br, k) for br, k in relief._incident_branches_sorted(net, focus) if k not in outaged
        )
        spare.status = 0  # a normally-open line near the outage

        def solve_fn(n):
            br = next(b for b in n.branches if (b.fbus, b.tbus) == (spare.fbus, spare.tbus))
            return _opflow(feasible=br.status == 1)

        res = relief.find_relief(net, branch_contingency, ["line_switching"], 0.9, 1.1, solve_fn, _CFG)
        assert res.resolved is True
        assert res.action.detail == f"switch in branch {spare.fbus}-{spare.tbus}"
        assert res.action.commands[0]["status"] == 1

    def test_line_switching_never_switches_outaged_line_back_in(self, net118, branch_contingency):
        outaged = relief._outaged_branch_keys(branch_contingency)
        seen = []

        def solve_fn(n):
            for b in n.branches:
                key = (min(b.fbus, b.tbus), max(b.fbus, b.tbus), 0)
                if key in outaged:
                    seen.append(b.status)
            return _opflow(feasible=False)

        relief.find_relief(net118, branch_contingency, ["line_switching"], 0.9, 1.1, solve_fn, _CFG)
        assert seen and all(st == 0 for st in seen)

    def _tiers(self, net, contingency):
        """Load buses per tier around the contingency, as the shedding search sees them."""
        scope = T.substations_within(net, sorted(relief._focus_buses(contingency)), 3)
        tiers = [[] for _ in range(4)]
        for b in sorted(net.buses, key=lambda b: b.bus_i):
            if b.Pd > 0 and b.bus_i in scope.tier_of:
                tiers[scope.tier_of[b.bus_i]].append(b.bus_i)
        return tiers

    @staticmethod
    def _shed(base_net, net, buses):
        base = {b.bus_i: b.Pd for b in base_net.buses}
        return sum(base[b.bus_i] - b.Pd for b in net.buses if b.bus_i in buses)

    def test_load_shedding_resolves_in_tier0(self, net118, branch_contingency):
        tiers = self._tiers(net118, branch_contingency)
        assert tiers[0], "need load in tier 0 for this test"
        tier0_pd = sum(b.Pd for b in net118.buses if b.bus_i in tiers[0])
        thresh = 0.04 * tier0_pd  # 4% at tier 0 is enough

        def solve_fn(net):
            return _opflow(feasible=self._shed(net118, net, tiers[0]) >= thresh - 1e-9)

        res = relief.find_relief(net118, branch_contingency, ["load_curtailment"], 0.9, 1.1, solve_fn, _CFG)
        assert res.resolved is True
        assert res.action.measure == "load_curtailment"
        assert {c["bus"] for c in res.action.commands} == set(tiers[0])
        shed = sum(
            next(b.Pd for b in net118.buses if b.bus_i == c["bus"]) - c["Pd"]
            for c in res.action.commands
        )
        assert abs(shed - thresh) <= 2.0
        assert "tier 0" in res.action.detail and "tier 1" not in res.action.detail

    def test_load_shedding_moves_to_next_tier(self, net118, branch_contingency):
        """10% at tier 0 is not enough, so tier 0 stays at 10% and tier 1 is added."""
        tiers = self._tiers(net118, branch_contingency)
        assert tiers[0] and tiers[1]
        tier0_pd = sum(b.Pd for b in net118.buses if b.bus_i in tiers[0])
        tier1_pd = sum(b.Pd for b in net118.buses if b.bus_i in tiers[1])
        need = 0.10 * tier0_pd + 0.05 * tier1_pd

        def solve_fn(net):
            return _opflow(feasible=self._shed(net118, net, tiers[0] + tiers[1]) >= need - 1e-9)

        res = relief.find_relief(net118, branch_contingency, ["load_curtailment"], 0.9, 1.1, solve_fn, _CFG)
        assert res.resolved is True
        pd = {b.bus_i: b.Pd for b in net118.buses}
        by_bus = {c["bus"]: c["Pd"] for c in res.action.commands}
        assert set(by_bus) == set(tiers[0] + tiers[1])
        for b in tiers[0]:
            assert by_bus[b] == pytest.approx(0.9 * pd[b])
        for b in tiers[1]:
            assert 0.9 * pd[b] <= by_bus[b] < pd[b]
        assert "tier 1" in res.action.detail

    def test_load_shedding_never_exceeds_cap(self, net118, branch_contingency):
        """Unresolved when 10% at every tier is not enough; no bus is ever shed more."""
        pd = {b.bus_i: b.Pd for b in net118.buses}
        worst = {"share": 0.0}

        def solve_fn(net):
            for b in net.buses:
                if pd[b.bus_i] > 0:
                    worst["share"] = max(worst["share"], 1 - b.Pd / pd[b.bus_i])
            return _opflow(feasible=False)

        res = relief.find_relief(net118, branch_contingency, ["load_curtailment"], 0.9, 1.1, solve_fn, _CFG)
        assert res.resolved is False
        assert worst["share"] <= 0.10 + 1e-12

    def test_load_shedding_never_sheds_added_load(self, net118, branch_contingency):
        """Protected (added) load stays; only the bus's other load counts toward the 10%."""
        tiers = self._tiers(net118, branch_contingency)
        bus = tiers[0][0]
        pd = {b.bus_i: b.Pd for b in net118.buses}
        protected = {bus: (0.5 * pd[bus], 0.0)}  # half of this bus's load is "new"
        lowest = {"pd": pd[bus]}

        def solve_fn(net):
            lowest["pd"] = min(lowest["pd"], next(b.Pd for b in net.buses if b.bus_i == bus))
            return _opflow(feasible=False)

        res = relief.find_relief(
            net118, branch_contingency, ["load_curtailment"], 0.9, 1.1, solve_fn, _CFG,
            protected_load=protected,
        )
        assert res.resolved is False
        # At most 10% of the unprotected half is shed: Pd never drops below 0.95 * Pd.
        assert lowest["pd"] == pytest.approx(0.5 * pd[bus] + 0.9 * 0.5 * pd[bus])

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

        measures = ["transformer_ratio", "line_switching", "load_curtailment"]
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
        with patch("agentigrid.exago_engine.agent_loop.create_backend", return_value=backend_mock), \
             patch("agentigrid.exago_engine.agent_loop.SimulationExecutor") as mock_exec_cls:
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
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            "substation_depth": 1, "contingency_order": 2, "components": ["branch"],
            "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            "relief_measures": ["line_switching", "load_curtailment"],
            "description": "N-2 + relief",
        }
        affected = C.find_affected_elements(net118, C.ChangedElement(kind="load", bus=77), 1)
        pool = C.outages_from(affected, ("branch",))
        n = len(C.enumerate_contingencies(pool, 2))
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
        with patch("agentigrid.exago_engine.agent_loop.parse_simulation_result_for_app",
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
        with patch("agentigrid.exago_engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 1, "components": ["branch"],
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        assert all("relief" not in s for s in entry.explored_variants)
        assert "Relief for failed contingencies" not in controller._latest_results_text

    def test_mutation_load_is_protected_in_handler(self, tmp_path, net118):
        """End to end: relief never sheds the 50 MW the mutation adds at bus 77."""
        controller, _ = self._controller(tmp_path)
        base_pd77 = next(b.Pd for b in net118.buses if b.bus_i == 77)
        state = {"i": 0}
        lowest = {"pd": float("inf")}
        real_solve = controller._executor.run

        def fake_parse(sim, application="opflow", bus_limits=None):
            state["i"] += 1
            return _opflow(state["i"] == 1)  # reference passes, everything else fails

        def fake_run(net, *a, **k):
            lowest["pd"] = min(lowest["pd"], next(b.Pd for b in net.buses if b.bus_i == 77))
            return real_solve(net, *a, **k)

        controller._executor.run = fake_run
        with patch("agentigrid.exago_engine.agent_loop.parse_simulation_result_for_app",
                   side_effect=fake_parse):
            controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 1, "components": ["branch"],
                "relief_measures": ["load_curtailment"],
            })
        assert lowest["pd"] != float("inf"), "relief solves ran"
        assert lowest["pd"] >= 50.0 + 0.9 * base_pd77 - 1e-9

    def test_relief_request_not_served_from_plain_screen_cache(self, tmp_path):
        """A screen with relief_measures is a different study; it must not reuse the cache."""
        controller, _ = self._controller(tmp_path)
        base = {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            "substation_depth": 1, "contingency_order": 1, "components": ["branch"],
        }
        with_relief = dict(base, relief_measures=["line_switching"])
        assert controller._sweep_cache_key(base) != controller._sweep_cache_key(with_relief)

    _SCREEN = {
        "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
        "substation_depth": 1, "contingency_order": 1, "components": ["branch"],
    }

    def _sweep(self, controller, data):
        with patch("agentigrid.exago_engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            return controller._handle_sweep(1, dict(data))

    def test_guard_drops_relief_when_goal_does_not_ask(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        controller._current_goal = "Connect a 50 MW load to bus 77 and test N-1 contingencies."
        kind, _ = self._sweep(controller, dict(self._SCREEN, relief_measures=["line_switching"]))
        assert kind == "sweep"
        entry = controller._journal.entries[-1]
        assert all("relief" not in v for v in entry.explored_variants)
        assert "relief_measures ignored" in controller._latest_results_text
        assert "Relief for failed contingencies" not in controller._latest_results_text

    def test_guard_keeps_relief_when_goal_asks(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        controller._current_goal = "Connect a 50 MW load to bus 77, test N-1 and suggest relief measures."
        self._sweep(controller, dict(self._SCREEN, relief_measures=["line_switching"]))
        entry = controller._journal.entries[-1]
        failed = [v for v in entry.explored_variants if not v["passed"]]
        assert failed and all("relief" in v for v in failed)
        assert "relief_measures ignored" not in controller._latest_results_text

    def test_guard_keeps_relief_when_steering_asks(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        controller._current_goal = "Connect a 50 MW load to bus 77 and test N-1 contingencies."
        controller._active_steering_directives = [{"directive": "Also try relief measures."}]
        self._sweep(controller, dict(self._SCREEN, relief_measures=["line_switching"]))
        assert "relief_measures ignored" not in controller._latest_results_text

    def test_guard_dropped_request_served_from_cache(self, tmp_path):
        """Plain screen, then the same screen with unrequested relief: no new solves."""
        controller, _ = self._controller(tmp_path)
        controller._current_goal = "Connect a 50 MW load to bus 77 and test N-1 contingencies."
        self._sweep(controller, self._SCREEN)
        with patch.object(controller, "_handle_contingency_sweep") as screen:
            self._sweep(controller, dict(self._SCREEN, relief_measures=["line_switching"]))
        screen.assert_not_called()
        assert "relief_measures ignored" in controller._latest_results_text

    def test_generator_redispatch_rejected(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
            "relief_measures": ["generator_redispatch", "load_curtailment"],
        })
        assert kind == "error"
        assert "generator_redispatch" in (controller._error_feedback or "")

    def test_invalid_relief_measure_rejected(self, tmp_path):
        controller, _ = self._controller(tmp_path)
        kind, ok = controller._handle_contingency_sweep(1, {
            "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
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
        with patch("agentigrid.exago_engine.agent_loop.parse_simulation_result_for_app",
                   return_value=_opflow(False)):
            kind, ok = controller._handle_contingency_sweep(1, {
                "mode": "contingency", "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0},
                "substation_depth": 1, "contingency_order": 1, "components": ["branch"],
                "relief_measures": ["line_switching"],
            })
        assert (kind, ok) == ("sweep", True)
        entry = controller._journal.entries[-1]
        failed = [s for s in entry.explored_variants if not s["passed"]]
        assert failed  # there are failures
        assert all(s["relief"]["detail"] == "relief budget exhausted" for s in failed)
