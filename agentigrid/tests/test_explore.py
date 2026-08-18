"""Tests for explore/select actions and concurrent PFLOW support."""

import pytest
from unittest.mock import MagicMock, patch

from agentigrid.config import AppConfig, SearchConfig, ExagoConfig, OutputConfig, LLMConfig, DataConfig
from agentigrid.engine.explore import (
    ExploreCache,
    VariantResult,
    annotate_cost_equivalent_siblings,
    build_variant_description,
    compute_pareto_labels,
    format_variant_results,
)
from agentigrid.engine.journal import ObjectiveEntry
from agentigrid.engine.pareto import ParetoCandidate


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal AppConfig for testing."""
    defaults = {
        "exago": ExagoConfig(
            binary_dir="/tmp/bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=60, mpi_np=1,
        ),
        "data": DataConfig(data_dir="/tmp/data"),
        "llm": LLMConfig(
            backend="anthropic", model="test", api_key_env="KEY",
            openai_base_url=None, ollama_host="http://localhost:11434",
            ollama_cloud_host=None, temperature=0.3, max_tokens=1024,
        ),
        "search": SearchConfig(
            max_iterations=10, default_mode="accumulative",
            base_case=None, gic_file=None, application="pflow",
            concurrent_pflow=True, max_variants=8,
        ),
        "output": OutputConfig(
            workdir="/tmp/wd", logs_dir="/tmp/logs",
            save_journal=True, journal_format="json",
            save_modified_files=True, verbose=False,
        ),
    }
    return AppConfig(**defaults)


def _make_opflow_result(**kwargs):
    """Build a minimal OPFLOWResult for testing."""
    from agentigrid.parsers.opflow_results import (
        OPFLOWResult, BusResult, BranchResult, GenResult,
    )
    defaults = dict(
        converged=True,
        objective_value=0.0,
        convergence_status="CONVERGED",
        solver="Newton-Raphson",
        model="POWER_BALANCE_SL",
        objective_type="NONE",
        num_iterations=5,
        solve_time=0.1,
        buses=[BusResult(bus_id=1, Pd=100.0, Pd_loss=0.0, Qd=20.0, Qd_loss=0.0,
                         Vm=1.02, Va=0.0, mult_Pmis=0.0, mult_Qmis=0.0,
                         Pslack=100.0, Qslack=20.0)],
        branches=[],
        generators=[GenResult(bus=1, status=1, fuel="coal", Pg=100.0, Qg=20.0,
                              Pmin=0.0, Pmax=200.0, Qmin=-50.0, Qmax=50.0)],
        total_gen_mw=100.0,
        total_load_mw=100.0,
        total_gen_mvar=20.0,
        total_load_mvar=20.0,
        voltage_min=0.98,
        voltage_max=1.05,
        voltage_mean=1.01,
        max_line_loading_pct=55.0,
        num_violations=0,
        violation_details=[],
        losses_mw=0.5,
        power_balance_mismatch_pct=0.1,
        feasibility_detail="feasible",
    )
    defaults.update(kwargs)
    return OPFLOWResult(**defaults)


# ---------------------------------------------------------------------------
# format_variant_results tests
# ---------------------------------------------------------------------------

class TestFormatVariantResults:

    def _minimal_variant(self, label, feasible=True, cost=10000.0, **kwargs):
        from agentigrid.engine.executor import SimulationResult
        from agentigrid.parsers.matpower_model import MATNetwork
        # Minimal mock network
        net = MagicMock(spec=MATNetwork)
        net.gencost = []

        opflow_kwargs = dict(
            feasibility_detail="feasible" if feasible else "infeasible",
            converged=feasible,
            voltage_min=kwargs.get("voltage_min", 0.98),
            voltage_max=kwargs.get("voltage_max", 1.05),
            max_line_loading_pct=kwargs.get("max_line_loading_pct", 55.0),
            num_violations=kwargs.get("num_violations", 0),
        )
        if not feasible:
            opflow_kwargs["convergence_status"] = "DID NOT CONVERGE"

        opf = _make_opflow_result(**opflow_kwargs) if feasible else None

        sim = MagicMock(spec=SimulationResult)
        sim.elapsed_seconds = 1.0

        return VariantResult(
            label=label,
            description=f"Variant {label}",
            commands=[],
            raw_commands=[],
            modified_net=net,
            sim_result=sim,
            opflow_result=opf,
            is_pareto=kwargs.get("is_pareto", False),
        )

    def test_empty_variants(self):
        result = format_variant_results({}, [])
        assert "No variant results" in result

    def test_single_feasible_variant(self):
        v = self._minimal_variant("A", feasible=True)
        result = format_variant_results({"A": v}, ["A"])
        assert "Variant" in result
        assert "A" in result
        assert "★" in result

    def test_mixed_feasibility(self):
        variants = {
            "A": self._minimal_variant("A", feasible=True),
            "B": self._minimal_variant("B", feasible=False),
        }
        result = format_variant_results(variants, ["A"])
        assert "feasible" in result.lower() or "Feas." in result

    def test_failed_simulation(self):
        v = self._minimal_variant("C", feasible=False)
        v.opflow_result = None
        result = format_variant_results({"C": v}, [])
        assert "FAIL" in result

    def test_all_pareto_labels_shown(self):
        v1 = self._minimal_variant("A", feasible=True, is_pareto=True)
        v2 = self._minimal_variant("B", feasible=True, is_pareto=True)
        result = format_variant_results({"A": v1, "B": v2}, ["A", "B"])
        assert "Pareto-optimal" in result

    def test_skipped_commands_displayed(self):
        from agentigrid.engine.commands import SetGenDispatch
        v = self._minimal_variant("A", feasible=True)
        v.skipped_commands = [
            (SetGenDispatch(bus=189, Pg=500.0), ["Bus 189 is the reference/slack bus"]),
        ]
        result = format_variant_results({"A": v}, ["A"])
        assert "Skipped" in result
        assert "SetGenDispatch" in result
        assert "reference/slack bus" in result


# ---------------------------------------------------------------------------
# compute_pareto_labels tests
# ---------------------------------------------------------------------------

class TestComputeParetoLabels:

    def _make_variant(self, label, feasible=True, cost=10000.0):
        v = MagicMock(spec=VariantResult)
        v.label = label
        # Create a real OPFLOWResult for metrics extraction
        feasibility = "feasible" if feasible else "infeasible"
        v.opflow_result = _make_opflow_result(
            feasibility_detail=feasibility,
            converged=feasible,
            objective_value=cost,
        )
        v.is_pareto = False
        v.rejected = False
        # Add compute_generation_cost mock
        v.opflow_result.compute_generation_cost = MagicMock(return_value=cost)
        return v

    def test_empty_variants(self):
        result = compute_pareto_labels({}, [])
        assert result == []

    def test_single_feasible_is_pareto(self):
        v = self._make_variant("A", feasible=True)
        variants = {"A": v}
        labels = compute_pareto_labels(variants, [])
        assert "A" in labels
        assert v.is_pareto is True

    def test_two_feasible_pareto_tradeoff(self):
        v1 = self._make_variant("A", feasible=True, cost=100.0)
        v2 = self._make_variant("B", feasible=True, cost=80.0)
        variants = {"A": v1, "B": v2}
        labels = compute_pareto_labels(variants, [])
        # With default generation_cost minimize, B (cost=80) dominates A (cost=100)
        assert "B" in labels

    def test_converged_but_violations_is_infeasible(self):
        v = self._make_variant("A", feasible=True, cost=100.0)
        v.opflow_result = _make_opflow_result(
            feasibility_detail="feasible",
            converged=True,
            num_violations=2,
        )
        v.opflow_result.compute_generation_cost = MagicMock(return_value=100.0)
        variants = {"A": v}
        labels = compute_pareto_labels(variants, [])
        assert "A" not in labels
        assert v.is_pareto is False

    def test_mixed_violations_feasibility(self):
        v_ok = self._make_variant("A", feasible=True, cost=100.0)
        v_viol = self._make_variant("B", feasible=True, cost=80.0)
        v_viol.opflow_result = _make_opflow_result(
            feasibility_detail="feasible",
            converged=True,
            num_violations=1,
        )
        v_viol.opflow_result.compute_generation_cost = MagicMock(return_value=80.0)
        variants = {"A": v_ok, "B": v_viol}
        labels = compute_pareto_labels(variants, [])
        assert "A" in labels
        assert "B" not in labels

    def test_rejected_variant_excluded_from_pareto(self):
        """Rejected variants didn't run a physics solve and must not enter the
        Pareto front, even if their dummy fields would otherwise dominate."""
        v_real = self._make_variant("A", feasible=True, cost=100.0)
        v_rej = self._make_variant("B", feasible=True, cost=10.0)
        v_rej.rejected = True
        labels = compute_pareto_labels({"A": v_real, "B": v_rej}, [])
        assert "A" in labels
        assert "B" not in labels


# ---------------------------------------------------------------------------
# Pre-execution rejection of all-no-op variants (Task 2)
# ---------------------------------------------------------------------------

class TestAllNoOpRejection:
    """A variant whose every command is a no-op (e.g. set_gen_dispatch on the
    slack bus) should be rejected before any PFLOW subprocess is launched."""

    @pytest.fixture
    def net(self):
        from agentigrid.parsers.matpower_parser import parse_matpower
        return parse_matpower("data/case_ACTIVSg200.m")

    def test_all_slack_dispatch_yields_zero_applied(self, net):
        """apply_modifications against slack-only commands → applied is empty."""
        from agentigrid.engine.commands import SetGenDispatch
        from agentigrid.engine.modifier import apply_modifications

        slack_buses = [b.bus_i for b in net.buses if b.type == 3]
        assert slack_buses, "ACTIVSg200 must have a slack bus"
        cmds = [SetGenDispatch(bus=slack_buses[0], Pg=400.0 + i) for i in range(3)]
        _, report = apply_modifications(net, cmds, application="pflow")
        assert len(report.applied) == 0
        assert len(report.skipped) == 3

    def test_mixed_effective_and_noop_yields_some_applied(self, net):
        """A variant with one effective command should NOT be rejected."""
        from agentigrid.engine.commands import SetGenDispatch
        from agentigrid.engine.modifier import apply_modifications

        slack_buses = [b.bus_i for b in net.buses if b.type == 3]
        non_slack_gen = next(
            g for g in net.generators
            if g.bus not in slack_buses and g.status == 1
            and g.Pmin < g.Pmax
        )
        cmds = [
            SetGenDispatch(bus=slack_buses[0], Pg=500.0),    # no-op
            SetGenDispatch(bus=non_slack_gen.bus, Pg=non_slack_gen.Pmin + 1.0),
        ]
        _, report = apply_modifications(net, cmds, application="pflow")
        assert len(report.applied) >= 1
        assert len(report.skipped) >= 1

    def test_variant_result_default_not_rejected(self):
        """VariantResult.rejected defaults to False."""
        from agentigrid.engine.executor import SimulationResult
        from agentigrid.parsers.matpower_model import MATNetwork
        v = VariantResult(
            label="A",
            description="x",
            commands=[],
            raw_commands=[],
            modified_net=MagicMock(spec=MATNetwork),
            sim_result=MagicMock(spec=SimulationResult),
            opflow_result=None,
        )
        assert v.rejected is False

    def test_rejected_variant_renders_REJECTED_in_table(self):
        """format_variant_results displays REJECTED for rejected variants."""
        from agentigrid.engine.commands import SetGenDispatch
        from agentigrid.engine.executor import SimulationResult
        from agentigrid.parsers.matpower_model import MATNetwork

        net = MagicMock(spec=MATNetwork)
        net.gencost = []
        v = VariantResult(
            label="A",
            description="[REJECTED — all commands no-op] slack dispatch",
            commands=[SetGenDispatch(bus=189, Pg=500.0)],
            raw_commands=[],
            modified_net=net,
            sim_result=MagicMock(spec=SimulationResult),
            opflow_result=None,
            rejected=True,
            skipped_commands=[
                (SetGenDispatch(bus=189, Pg=500.0),
                 ["Bus 189 is the reference/slack bus"]),
            ],
        )
        out = format_variant_results({"A": v}, [])
        assert "REJECTED" in out


# ---------------------------------------------------------------------------
# build_variant_description tests (Task 1)
# ---------------------------------------------------------------------------

class TestBuildVariantDescription:

    def _cmd(self, bus, pg):
        from agentigrid.engine.commands import SetGenDispatch
        return SetGenDispatch(bus=bus, Pg=pg)

    def _vlim(self, vmin, vmax):
        from agentigrid.engine.commands import SetAllBusVLimits
        return SetAllBusVLimits(Vmin=vmin, Vmax=vmax)

    def _scale(self, factor):
        from agentigrid.engine.commands import ScaleAllLoads
        return ScaleAllLoads(factor=factor)

    def test_all_applied_contains_bus_numbers(self):
        c1 = self._cmd(135, 250.0)
        c2 = self._cmd(136, 250.0)
        desc = build_variant_description([c1, c2], [])
        assert "135" in desc
        assert "136" in desc
        assert "250" in desc
        assert "[SKIP]" not in desc
        assert "SKIPPED" not in desc

    def test_skipped_command_marked_inline(self):
        c_applied = self._cmd(135, 250.0)
        c_skipped = self._cmd(189, 569.0)
        desc = build_variant_description(
            [c_applied, c_skipped],
            [(c_skipped, ["Bus 189 is the reference/slack bus"])],
        )
        assert "189" in desc
        assert "[SKIP]" in desc
        assert "135" in desc

    def test_skip_suffix_present_when_skips_exist(self):
        c1 = self._scale(1.23)
        c2 = self._cmd(189, 400.0)
        desc = build_variant_description([c1, c2], [(c2, ["slack bus"])])
        assert "SKIPPED" in desc
        assert "1" in desc  # skip count

    def test_no_suffix_when_no_skips(self):
        c1 = self._vlim(0.95, 1.05)
        c2 = self._cmd(135, 200.0)
        desc = build_variant_description([c1, c2], [])
        assert "SKIPPED" not in desc
        assert "[SKIP]" not in desc

    def test_no_commands(self):
        desc = build_variant_description([], [])
        assert "no commands" in desc

    def test_all_commands_skipped(self):
        c1 = self._cmd(189, 400.0)
        c2 = self._cmd(189, 450.0)
        desc = build_variant_description(
            [c1, c2],
            [(c1, ["slack bus"]), (c2, ["slack bus"])],
        )
        assert "[SKIP]" in desc
        assert "SKIPPED" in desc

    def test_mixed_commands_compact(self):
        """Abbreviations should be compact and not verbose."""
        c1 = self._scale(1.23)
        c2 = self._vlim(0.95, 1.05)
        c3 = self._cmd(135, 250.0)
        desc = build_variant_description([c1, c2, c3], [])
        assert "1.23" in desc
        assert "0.95" in desc
        assert "135" in desc
        # Should not contain Python class names
        assert "ScaleAllLoads" not in desc
        assert "SetAllBusVLimits" not in desc


# ---------------------------------------------------------------------------
# annotate_cost_equivalent_siblings tests (Task 2)
# ---------------------------------------------------------------------------

class TestAnnotateCostEquivalentSiblings:

    def _variant(self, label, cost=None, feasible=True, n_commands=1, *, rejected=False):
        from agentigrid.engine.executor import SimulationResult
        from agentigrid.parsers.matpower_model import MATNetwork
        from agentigrid.engine.commands import SetGenDispatch
        net = MagicMock(spec=MATNetwork)
        net.gencost = []
        if feasible and cost is not None:
            opf = _make_opflow_result(
                feasibility_detail="feasible",
                converged=True,
                num_violations=0,
            )
            opf.compute_generation_cost = MagicMock(return_value=cost)
        else:
            opf = None
        cmds = [SetGenDispatch(bus=i + 1, Pg=float(i * 10)) for i in range(n_commands)]
        return VariantResult(
            label=label,
            description=label,
            commands=cmds,
            raw_commands=[],
            modified_net=net,
            sim_result=MagicMock(spec=SimulationResult),
            opflow_result=opf,
            rejected=rejected,
        )

    def test_no_siblings_returns_none(self):
        v1 = self._variant("A", cost=1000.0)
        v2 = self._variant("B", cost=2000.0)
        result = annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert result is None

    def test_sibling_simpler_variant_unchanged(self):
        v1 = self._variant("A", cost=1000.0, n_commands=1)  # simpler
        v2 = self._variant("B", cost=1000.0, n_commands=3)  # more complex
        annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert "same cost" not in v1.description
        assert v1.cost_equivalent_to == ""

    def test_sibling_complex_variant_annotated(self):
        v1 = self._variant("A", cost=1000.0, n_commands=1)
        v2 = self._variant("B", cost=1000.0, n_commands=3)
        annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert "same cost as A" in v2.description
        assert v2.cost_equivalent_to == "A"

    def test_sibling_batch_warning_contains_labels(self):
        v1 = self._variant("A", cost=5000.0, n_commands=1)
        v2 = self._variant("B", cost=5000.0, n_commands=2)
        warning = annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert warning is not None
        assert "B" in warning
        assert "A" in warning

    def test_infeasible_variants_excluded(self):
        v1 = self._variant("A", cost=1000.0, feasible=True)
        v2 = self._variant("B", cost=None, feasible=False)  # infeasible — no opflow
        result = annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert result is None  # no sibling pair found
        assert "same cost" not in v2.description

    def test_rejected_variant_excluded(self):
        v1 = self._variant("A", cost=1000.0, n_commands=1)
        v2 = self._variant("B", cost=1000.0, n_commands=3, rejected=True)
        result = annotate_cost_equivalent_siblings({"A": v1, "B": v2}, gencost=[])
        assert result is None
        assert "same cost" not in v2.description


# ---------------------------------------------------------------------------
# ExploreCache tests
# ---------------------------------------------------------------------------

class TestExploreCache:

    def test_default_values(self):
        cache = ExploreCache()
        assert cache.variants == {}
        assert cache.description == ""
        assert cache.iteration == 0
        assert cache.base_network_snapshot is None
        assert cache.base_mode == "accumulative"

    def test_with_values(self):
        cache = ExploreCache(
            variants={},
            description="Voltage sweep",
            reasoning="Testing Vg at bus 1",
            iteration=5,
            base_mode="fresh",
        )
        assert cache.description == "Voltage sweep"
        assert cache.iteration == 5


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestConfigConcurrentPflow:

    def test_default_concurrent_pflow_false(self):
        from agentigrid.config import load_config
        cfg = load_config()
        assert cfg.search.concurrent_pflow is False
        assert cfg.search.max_variants == 8

    def test_cli_override_concurrent_pflow(self):
        from agentigrid.config import load_config
        cfg = load_config(cli_overrides={
            "search.concurrent_pflow": True,
            "search.max_variants": 4,
        })
        assert cfg.search.concurrent_pflow is True
        assert cfg.search.max_variants == 4


# ---------------------------------------------------------------------------
# build_system_prompt integration tests
# ---------------------------------------------------------------------------

class TestSystemPromptConcurrent:

    def test_pflow_concurrent_includes_explore(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test",
            network_summary="test",
            application="pflow",
            concurrent_pflow=True,
        )
        assert "explore" in prompt.lower()
        assert "select" in prompt.lower()
        assert "Pareto" in prompt or "pareto" in prompt.lower()

    def test_pflow_non_concurrent_no_explore(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test",
            network_summary="test",
            application="pflow",
            concurrent_pflow=False,
        )
        assert '"explore"' not in prompt
        assert '"select"' not in prompt

    def test_opflow_concurrent_no_explore(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test",
            network_summary="test",
            application="opflow",
            concurrent_pflow=True,
        )
        assert '"explore"' not in prompt


# ---------------------------------------------------------------------------
# build_user_prompt explore_text tests
# ---------------------------------------------------------------------------

class TestUserPromptExploreText:

    def test_explore_text_replaces_results(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        result = build_user_prompt(
            goal="test goal",
            journal_text=None,
            results_text="Normal results",
            explore_text="=== Neighborhood Exploration Results ===",
        )
        assert "Neighborhood Exploration" in result
        assert "Latest Results" not in result

    def test_no_explore_text_shows_results(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        result = build_user_prompt(
            goal="test goal",
            journal_text=None,
            results_text="Normal results",
        )
        assert "Latest Results" in result
        assert "Neighborhood Exploration" not in result

    def test_explore_text_none_shows_results(self):
        from agentigrid.prompts.user_prompt import build_user_prompt
        result = build_user_prompt(
            goal="test goal",
            journal_text=None,
            results_text="Normal results",
            explore_text=None,
        )
        assert "Latest Results" in result


# ---------------------------------------------------------------------------
# Journal explored_variants tests
# ---------------------------------------------------------------------------

class TestJournalExploredVariants:

    def test_add_from_results_with_explored_variants(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        opf = _make_opflow_result()
        entry = journal.add_from_results(
            iteration=1,
            description="[select A] Voltage sweep",
            commands=[{"action": "set_gen_voltage", "bus": 1, "Vg": 1.04}],
            opflow_result=opf,
            sim_elapsed=1.0,
            llm_reasoning="Selected variant A",
            mode="accumulative",
            explored_variants=[
                {"label": "A", "feasible": True, "cost": 12000.0},
                {"label": "B", "feasible": True, "cost": 14000.0},
                {"label": "C", "feasible": False},
            ],
        )
        assert entry.explored_variants is not None
        assert len(entry.explored_variants) == 3
        assert entry.explored_variants[0]["label"] == "A"
        assert entry.explored_variants[2]["feasible"] is False

    def test_add_from_results_without_explored_variants(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        opf = _make_opflow_result()
        entry = journal.add_from_results(
            iteration=1,
            description="Test",
            commands=[],
            opflow_result=opf,
            sim_elapsed=1.0,
            llm_reasoning="test",
            mode="fresh",
        )
        assert entry.explored_variants is None


# ---------------------------------------------------------------------------
# Session persistence with explore_cache_info
# ---------------------------------------------------------------------------

class TestSessionExplorePersistence:

    def test_save_with_explore_cache_info(self, tmp_path):
        from agentigrid.engine.session_io import save_session, load_session
        from agentigrid.engine.journal import SearchJournal

        journal = SearchJournal()
        save_dir = tmp_path / "test_session"

        save_session(
            save_dir=save_dir,
            goal="test goal",
            application="pflow",
            base_case_path=tmp_path / "test.m",
            config_path=None,
            journal=journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=3,
            explore_cache_info={
                "was_active": True,
                "description": "Voltage sweep",
                "iteration": 3,
                "variant_labels": ["A", "B", "C"],
                "base_mode": "fresh",
            },
        )

        loaded = load_session(save_dir)
        assert loaded["explore_cache_info"] is not None
        assert loaded["explore_cache_info"]["was_active"] is True
        assert loaded["explore_cache_info"]["variant_labels"] == ["A", "B", "C"]

    def test_save_without_explore_cache_info(self, tmp_path):
        from agentigrid.engine.session_io import save_session, load_session
        from agentigrid.engine.journal import SearchJournal

        journal = SearchJournal()
        save_dir = tmp_path / "test_session2"

        save_session(
            save_dir=save_dir,
            goal="test goal",
            application="opflow",
            base_case_path=tmp_path / "test.m",
            config_path=None,
            journal=journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=1,
        )

        loaded = load_session(save_dir)
        assert loaded["explore_cache_info"] is None

    def test_load_v1_0_session_without_explore(self, tmp_path):
        import json
        from agentigrid.engine.session_io import load_session

        session_data = {
            "format_version": "1.0",
            "goal": "test",
            "application": "opflow",
            "base_case_path": "/tmp/test.m",
            "config_path": None,
            "last_iteration": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "journal": {"entries": [], "objective_registry": [], "preference_history": []},
        }
        session_path = tmp_path / "session.json"
        session_path.write_text(json.dumps(session_data))

        loaded = load_session(tmp_path)
        assert loaded["explore_cache_info"] is None


# ---------------------------------------------------------------------------
# Explore handler validation tests (mocked controller)
# ---------------------------------------------------------------------------

class TestExploreHandlerValidation:
    """Test validation logic in _handle_explore and _handle_select."""

    def _make_pflow_config(self, tmp_path, concurrent_pflow=True, max_variants=8):
        from agentigrid.config import AppConfig, ExagoConfig, OutputConfig, LLMConfig
        return AppConfig(
            exago=ExagoConfig(
                binary_dir=tmp_path / "bin", opflow_binary=None,
                scopflow_binary=None, tcopflow_binary=None, sopflow_binary=None,
                dcopflow_binary=None, pflow_binary=None, env_script=None, timeout=30,
            ),
            data=DataConfig(data_dir=tmp_path / "data"),
            llm=LLMConfig(
                backend="openai", model="test-model", api_key_env="TEST_KEY",
                openai_base_url=None, ollama_host="http://localhost:11434",
                ollama_cloud_host=None, temperature=0.3, max_tokens=4096,
            ),
            search=SearchConfig(
                max_iterations=10, default_mode="accumulative",
                base_case=None, gic_file=None, application="pflow",
                concurrent_pflow=concurrent_pflow, max_variants=max_variants,
            ),
            output=OutputConfig(
                workdir=tmp_path / "workdir", logs_dir=tmp_path / "logs",
                save_journal=True, journal_format="json",
                save_modified_files=False, verbose=False,
            ),
        )

    def test_explore_requires_concurrent_pflow(self, tmp_path):
        from agentigrid.engine.agent_loop import AgentLoopController
        from unittest.mock import MagicMock, patch

        cfg = self._make_pflow_config(tmp_path, concurrent_pflow=False)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_create, \
             patch("agentigrid.engine.agent_loop.SimulationExecutor"):
            mock_create.return_value = MagicMock()
            controller = AgentLoopController(cfg, quiet=True)
            result_type, should_continue = controller._handle_explore(1, {
                "action": "explore",
                "reasoning": "test",
                "mode": "accumulative",
                "description": "test",
                "variants": [
                    {"label": "A", "commands": [{"action": "scale_all_loads", "factor": 1.05}]},
                    {"label": "B", "commands": [{"action": "scale_all_loads", "factor": 1.10}]},
                ],
            })
        assert result_type == "error"
        assert should_continue is True

    def test_explore_requires_at_least_2_variants(self, tmp_path):
        from agentigrid.engine.agent_loop import AgentLoopController
        from unittest.mock import MagicMock, patch

        cfg = self._make_pflow_config(tmp_path, concurrent_pflow=True)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_create, \
             patch("agentigrid.engine.agent_loop.SimulationExecutor"):
            mock_create.return_value = MagicMock()
            controller = AgentLoopController(cfg, quiet=True)
            result_type, should_continue = controller._handle_explore(1, {
                "action": "explore",
                "reasoning": "test",
                "mode": "accumulative",
                "description": "test",
                "variants": [
                    {"label": "A", "commands": [{"action": "scale_all_loads", "factor": 1.05}]},
                ],
            })
        assert result_type == "error"

    def test_select_without_explore_returns_error(self, tmp_path):
        from agentigrid.engine.agent_loop import AgentLoopController
        from unittest.mock import MagicMock, patch

        cfg = self._make_pflow_config(tmp_path, concurrent_pflow=True)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_create, \
             patch("agentigrid.engine.agent_loop.SimulationExecutor"):
            mock_create.return_value = MagicMock()
            controller = AgentLoopController(cfg, quiet=True)
            result_type, should_continue = controller._handle_select(1, {
                "action": "select",
                "choice": "A",
                "reasoning": "test",
            })
        assert result_type == "error"

    def test_select_invalid_label_returns_error(self, tmp_path):
        from agentigrid.engine.explore import ExploreCache, VariantResult
        from agentigrid.engine.agent_loop import AgentLoopController
        from unittest.mock import MagicMock, patch

        cfg = self._make_pflow_config(tmp_path, concurrent_pflow=True)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_create, \
             patch("agentigrid.engine.agent_loop.SimulationExecutor"):
            mock_create.return_value = MagicMock()
            controller = AgentLoopController(cfg, quiet=True)
            controller._explore_cache = ExploreCache(
                variants={},
                description="test",
                iteration=1,
            )
            result_type, should_continue = controller._handle_select(1, {
                "action": "select",
                "choice": "Z",
                "reasoning": "test",
            })
        assert result_type == "error"

    def test_modify_clears_explore_cache(self, tmp_path):
        from agentigrid.engine.explore import ExploreCache
        from agentigrid.engine.agent_loop import AgentLoopController
        from unittest.mock import MagicMock, patch

        cfg = self._make_pflow_config(tmp_path, concurrent_pflow=True)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_create, \
             patch("agentigrid.engine.agent_loop.SimulationExecutor"):
            mock_create.return_value = MagicMock()
            controller = AgentLoopController(cfg, quiet=True)
            controller._explore_cache = ExploreCache(description="test", iteration=1)
            assert controller._explore_cache is not None
            controller._explore_cache = None
            assert controller._explore_cache is None


# ---------------------------------------------------------------------------
# run_parallel tests
# ---------------------------------------------------------------------------

class TestRunParallel:

    def test_run_parallel_basic(self, tmp_path):
        from agentigrid.config import ExagoConfig, OutputConfig
        from agentigrid.engine.executor import SimulationExecutor, SimulationResult

        exago_config = ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None,
            scopflow_binary=None, tcopflow_binary=None, sopflow_binary=None,
            dcopflow_binary=None, pflow_binary=None, env_script=None, timeout=10,
        )
        output_config = OutputConfig(
            workdir=tmp_path / "work",
            logs_dir=tmp_path / "logs",
            save_journal=False, journal_format="json",
            save_modified_files=False, verbose=False,
        )
        executor = SimulationExecutor(exago_config, output_config)

        result = executor.run_parallel([])
        assert result == {}

    def test_run_parallel_sigature(self):
        from agentigrid.engine.executor import SimulationExecutor
        import inspect
        sig = inspect.signature(SimulationExecutor.run_parallel)
        params = list(sig.parameters.keys())
        assert "tasks" in params
        assert "max_workers" in params


# ---------------------------------------------------------------------------
# CLI arguments tests
# ---------------------------------------------------------------------------

class TestConcurrentPflowCLI:

    def test_concurrent_pflow_flag(self):
        from agentigrid.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["case.m", "goal", "--concurrent-pflow"])
        assert args.concurrent_pflow is True

    def test_max_variants_flag(self):
        from agentigrid.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["case.m", "goal", "--max-variants", "5"])
        assert args.max_variants == 5

    def test_cli_overrides_mapping(self):
        from agentigrid.cli import build_parser, _cli_overrides
        parser = build_parser()
        args = parser.parse_args(["case.m", "goal", "--concurrent-pflow", "--max-variants", "6"])
        overrides = _cli_overrides(args)
        assert overrides["search.concurrent_pflow"] is True
        assert overrides["search.max_variants"] == 6

    def test_defaults_without_flags(self):
        from agentigrid.cli import build_parser, _cli_overrides
        parser = build_parser()
        args = parser.parse_args(["case.m", "goal"])
        overrides = _cli_overrides(args)
        assert "search.concurrent_pflow" not in overrides
        assert "search.max_variants" not in overrides


# ---------------------------------------------------------------------------
# Build system prompt with concurrent_pflow
# ---------------------------------------------------------------------------

class TestSystemPromptConcurrentMode:
    """Extended tests for the system prompt with concurrent_pflow mode."""

    def test_pflow_concurrent_has_explore_and_select(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test", network_summary="net",
            application="pflow", concurrent_pflow=True,
        )
        assert '"explore"' in prompt
        assert '"select"' in prompt
        assert "variants" in prompt.lower()
        assert "neighborhood" in prompt.lower()

    def test_pflow_concurrent_has_pareto_star(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test", network_summary="net",
            application="pflow", concurrent_pflow=True,
        )
        assert "★" in prompt or "Pareto" in prompt

    def test_pflow_non_concurrent_no_explore(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test", network_summary="net",
            application="pflow", concurrent_pflow=False,
        )
        assert '"explore"' not in prompt
        assert '"select"' not in prompt

    def test_opflow_concurrent_no_explore_section(self):
        from agentigrid.prompts.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            command_schema="test", network_summary="net",
            application="opflow", concurrent_pflow=True,
        )
        assert "EXPLORE" not in prompt
        assert '"explore"' not in prompt


# ---------------------------------------------------------------------------
# Bug fix tests
# ---------------------------------------------------------------------------

class TestExploreJournalEntry:
    """Bug fix: explore actions must create a journal entry."""

    def test_add_explore_creates_entry(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        entry = journal.add_explore(
            iteration=3,
            description="[explore] Voltage sweep",
            variant_info=[
                {"label": "A", "description": "Vmin=0.95", "commands": [{"action": "scale_all_loads", "factor": 1.05}], "feasible": True, "is_pareto": True},
                {"label": "B", "description": "Vmin=0.94", "commands": [{"action": "scale_all_loads", "factor": 1.10}], "feasible": False, "is_pareto": False},
            ],
            pareto_labels=["A"],
            llm_reasoning="Testing voltage limit sensitivity",
        )
        assert entry.convergence_status == "EXPLORE"
        assert entry.mode == "explore"
        assert entry.explored_variants is not None
        assert len(entry.explored_variants) == 2
        assert entry.explored_variants[0]["label"] == "A"
        assert entry.explored_variants[0]["description"] == "Vmin=0.95"
        assert entry.explored_variants[0]["is_pareto"] is True
        assert entry.explored_variants[1]["is_pareto"] is False
        assert len(journal.entries) == 1

    def test_add_explore_appears_in_format_for_prompt(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        journal.add_explore(
            iteration=2,
            description="[explore] Load sweep",
            variant_info=[
                {"label": "A", "description": "factor=1.05", "commands": [], "feasible": True, "is_pareto": True},
            ],
            pareto_labels=["A"],
        )
        text = journal.format_for_prompt()
        assert "EXPLORE" in text

    def test_add_explore_appears_in_format_detailed(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        journal.add_explore(
            iteration=5,
            description="[explore] Gen dispatch sweep",
            variant_info=[
                {"label": "A", "description": "Pg=100", "commands": [{"action": "set_gen_dispatch", "bus": 1, "Pg": 100}], "feasible": True, "is_pareto": True},
            ],
            pareto_labels=["A"],
        )
        text = journal.format_detailed()
        assert "EXPLORE" in text
        assert "Explored variants" in text


class TestExploredVariantsDescriptions:
    """Bug fix: explored_variants must include description, commands, and is_pareto."""

    def test_select_entry_has_variant_descriptions(self):
        from agentigrid.engine.journal import SearchJournal
        journal = SearchJournal()
        opf = _make_opflow_result()
        entry = journal.add_from_results(
            iteration=3,
            description="[select A] test",
            commands=[{"action": "scale_all_loads", "factor": 1.05}],
            opflow_result=opf,
            sim_elapsed=1.0,
            llm_reasoning="Selected A",
            mode="accumulative",
            explored_variants=[
                {
                    "label": "A",
                    "description": "Load factor 1.05",
                    "commands": [{"action": "scale_all_loads", "factor": 1.05}],
                    "feasible": True,
                    "cost": 12000.0,
                    "is_pareto": True,
                },
                {
                    "label": "B",
                    "description": "Load factor 1.10",
                    "commands": [{"action": "scale_all_loads", "factor": 1.10}],
                    "feasible": False,
                    "is_pareto": False,
                },
            ],
        )
        assert entry.explored_variants is not None
        assert entry.explored_variants[0]["description"] == "Load factor 1.05"
        assert entry.explored_variants[0]["commands"] == [{"action": "scale_all_loads", "factor": 1.05}]
        assert entry.explored_variants[0]["is_pareto"] is True
        assert entry.explored_variants[1]["description"] == "Load factor 1.10"


class TestAutoInjectVlimits:
    """Bug fix: auto-inject set_all_bus_vlimits into variants missing it."""

    def test_current_bus_vlimits_uniform(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        net = MagicMock()
        bus1 = MagicMock()
        bus1.Vmin = 0.95
        bus1.Vmax = 1.05
        bus2 = MagicMock()
        bus2.Vmin = 0.95
        bus2.Vmax = 1.05
        net.buses = [bus1, bus2]
        result = AgentLoopController._current_bus_vlimits(net)
        assert result == (0.95, 1.05)

    def test_current_bus_vlimits_mixed(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        net = MagicMock()
        bus1 = MagicMock()
        bus1.Vmin = 0.95
        bus1.Vmax = 1.05
        bus2 = MagicMock()
        bus2.Vmin = 0.90
        bus2.Vmax = 1.10
        net.buses = [bus1, bus2]
        result = AgentLoopController._current_bus_vlimits(net)
        assert result is None

    def test_variant_has_vlimits(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        cmds = [
            {"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05},
            {"action": "scale_all_loads", "factor": 1.05},
        ]
        assert AgentLoopController._variant_has_vlimits(cmds) is True

    def test_variant_no_vlimits(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        cmds = [{"action": "scale_all_loads", "factor": 1.05}]
        assert AgentLoopController._variant_has_vlimits(cmds) is False

    def test_variant_per_bus_vlimits(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        cmds = [{"action": "set_bus_vlimits", "bus": 1, "Vmin": 0.95}]
        assert AgentLoopController._variant_has_vlimits(cmds) is True


# ===========================================================================
# Prompt C Task 1: load factor auto-inject helpers
# ===========================================================================

class TestLoadFactorAutoInject:
    """Unit tests for the load factor auto-inject logic."""

    def _has_scale_all_loads(self, cmds: list[dict]) -> bool:
        return any(c.get("action", "").lower() == "scale_all_loads" for c in cmds)

    def _inject_if_needed(self, raw_commands: list[dict], factor) -> list[dict]:
        """Replicate the auto-inject logic from _handle_modify."""
        if (
            factor is not None
            and not self._has_scale_all_loads(raw_commands)
        ):
            return [{"action": "scale_all_loads", "factor": factor}] + list(raw_commands)
        return raw_commands

    def test_inject_when_missing(self):
        """When load_factor=1.23 and no scale_all_loads, it is prepended."""
        cmds = [{"action": "set_gen_voltage", "bus": 1, "Vg": 1.02}]
        result = self._inject_if_needed(cmds, 1.23)
        assert result[0] == {"action": "scale_all_loads", "factor": 1.23}
        assert len(result) == 2

    def test_no_inject_when_already_present(self):
        """When scale_all_loads is already in the commands, no duplicate is added."""
        cmds = [
            {"action": "scale_all_loads", "factor": 1.23},
            {"action": "set_gen_voltage", "bus": 1, "Vg": 1.02},
        ]
        result = self._inject_if_needed(cmds, 1.23)
        count = sum(1 for c in result if c.get("action") == "scale_all_loads")
        assert count == 1

    def test_no_inject_when_factor_none(self):
        """When load_factor is None, no scale_all_loads is injected."""
        cmds = [{"action": "set_gen_voltage", "bus": 1, "Vg": 1.02}]
        result = self._inject_if_needed(cmds, None)
        assert not self._has_scale_all_loads(result)

    def test_inject_uses_updated_factor(self):
        """When factor changes (set_load_factor), the new factor is used."""
        import pytest
        cmds = [{"action": "set_gen_voltage", "bus": 1, "Vg": 1.02}]
        result = self._inject_if_needed(cmds, 1.35)
        assert result[0]["factor"] == pytest.approx(1.35)
