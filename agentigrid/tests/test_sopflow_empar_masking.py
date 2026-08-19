"""SOPFLOW EMPAR-masking fix + curtailable-wind base normalization.

Part A: trust the per-scenario scen_*.m files over EMPAR's optimistic header, so a
run whose second-stage subproblems failed is never reported as feasible / $0.00.
Part B: model base-case wind as curtailable (Pmin -> 0) so sub-nameplate scenarios
are feasible; gated by search.sopflow_curtailable_wind_base.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
    load_config,
)
from agentigrid.parsers import all_scenarios_converged
from agentigrid.parsers.opflow_results import OPFLOWResult
import agentigrid.parsers.sopflow_parser as SP
from agentigrid.parsers.sopflow_parser import parse_sopflow_simulation_result
from agentigrid.parsers.sopflow_summary import sopflow_results_summary

_DATA = Path(__file__).resolve().parent.parent / "data"
_ACTIVSG200 = _DATA / "case_ACTIVSg200.m"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_scen_dir(tmp_path: Path, converged_flags: list[int]) -> Path:
    """Create workdir/sopflowout/scen_<i>.m each carrying mpc.converged = flag."""
    wd = tmp_path / "iter"
    out = wd / "sopflowout"
    out.mkdir(parents=True)
    for i, flag in enumerate(converged_flags):
        (out / f"scen_{i}.m").write_text(
            f"function mpc = scen_{i}\nmpc.baseMVA = 100;\n"
            f"mpc.obj = 0;\nmpc.converged = {flag};\n",
            encoding="utf-8",
        )
    return wd


_BAND = {b: (0.95, 1.05) for b in range(1, 5)}  # enforced 0.95–1.05 band


def _result(solver="EMPAR", converged=True, obj=14392.24, gen=1000.0, load=980.0,
            vmin=0.98, vmax=1.04, num_violations=0):
    """A solved base-case result. Defaults describe a genuinely feasible solution
    (balanced, in-band, no violations, positive objective)."""
    r = OPFLOWResult(
        converged=converged, objective_value=obj, convergence_status="CONVERGED",
        solver=solver, model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=1, solve_time=0.1,
    )
    r.feasibility_detail = "feasible"
    r.num_violations = num_violations
    r.total_gen_mw = gen
    r.total_load_mw = load
    r.voltage_min = vmin
    r.voltage_max = vmax
    r.losses_mw = gen - load
    return r


def _run_parse(tmp_path, solver, converged_flags, result=None, bus_limits=_BAND):
    wd = _make_scen_dir(tmp_path, converged_flags)
    sim = SimpleNamespace(success=True, stdout="x", workdir=wd)
    res = result if result is not None else _result(solver=solver)

    def fake_parse(stdout, bus_limits=None):
        return res, {"solver": solver, "num_scenarios": len(converged_flags)}

    with patch.object(SP, "parse_sopflow_output", side_effect=fake_parse):
        return parse_sopflow_simulation_result(sim, bus_limits=bus_limits)


# --------------------------------------------------------------------------
# all_scenarios_converged
# --------------------------------------------------------------------------

def test_all_converged_true(tmp_path):
    wd = _make_scen_dir(tmp_path, [1, 1, 1])
    assert all_scenarios_converged(wd) is True


def test_all_converged_false_when_any_zero(tmp_path):
    wd = _make_scen_dir(tmp_path, [1, 0, 1])
    assert all_scenarios_converged(wd) is False


def test_all_converged_none_when_no_files(tmp_path):
    (tmp_path / "iter" / "sopflowout").mkdir(parents=True)
    assert all_scenarios_converged(tmp_path / "iter") is None


def test_all_converged_none_when_no_sopflowout(tmp_path):
    assert all_scenarios_converged(tmp_path / "missing") is None


def test_all_converged_missing_flag_is_not_converged(tmp_path):
    wd = tmp_path / "iter"
    out = wd / "sopflowout"
    out.mkdir(parents=True)
    (out / "scen_0.m").write_text("function mpc = scen_0\nmpc.baseMVA = 100;\n", encoding="utf-8")
    assert all_scenarios_converged(wd) is False  # fail-safe: cannot certify


# --------------------------------------------------------------------------
# Change 1: solution-based feasibility (flags are advisory only)
# --------------------------------------------------------------------------

def test_empar_bad_flags_good_solution_is_feasible_marginal(tmp_path):
    # EMPAR flagged scenarios non-converged, but the SOLUTION is good -> feasible,
    # annotated marginal, with the real objective kept (NOT FAILED, NOT $0.00).
    res, meta = _run_parse(tmp_path, "EMPAR", [1, 0, 1])
    assert res.converged is True
    assert res.feasibility_detail == "feasible"
    assert res.objective_value == pytest.approx(14392.24)
    assert "marginal" in res.convergence_status.lower()
    assert meta["empar_marginal"] is True
    assert "marginal_note" in meta


def test_empar_bad_flags_bad_solution_is_infeasible(tmp_path):
    # EMPAR non-converged flags AND a bad solution (imbalanced echo, obj 0) ->
    # infeasible, objective None.
    bad = _result(solver="EMPAR", obj=0.0, gen=1450.0, load=1000.0, vmin=1.0, vmax=1.0)
    res, meta = _run_parse(tmp_path, "EMPAR", [0, 0, 0], result=bad)
    assert res.converged is False
    assert res.convergence_status == "DID NOT CONVERGE"
    assert res.feasibility_detail == "infeasible"
    assert res.objective_value is None  # NOT 0.0
    assert meta["sopflow_solution_feasible"] is False


def test_empar_all_converged_good_solution_feasible_not_marginal(tmp_path):
    res, meta = _run_parse(tmp_path, "EMPAR", [1, 1, 1])
    assert res.converged is True
    assert res.feasibility_detail == "feasible"
    assert res.objective_value == pytest.approx(14392.24)
    assert res.convergence_status == "CONVERGED"
    assert "empar_marginal" not in meta


def test_empar_out_of_band_is_infeasible(tmp_path):
    # Genuinely infeasible: base sits at 1.06–1.10, band is 0.95–1.05.
    oob = _result(solver="EMPAR", vmin=1.06, vmax=1.10, num_violations=2)
    res, meta = _run_parse(tmp_path, "EMPAR", [1, 1, 1], result=oob)
    assert res.converged is False
    assert res.feasibility_detail == "infeasible"
    assert res.objective_value is None


def test_empar_imbalanced_is_infeasible(tmp_path):
    # Gen ~45% over load (the old unsolved-echo case) -> imbalanced -> infeasible.
    imb = _result(solver="EMPAR", gen=1450.0, load=1000.0)
    res, meta = _run_parse(tmp_path, "EMPAR", [1, 1, 1], result=imb)
    assert res.feasibility_detail == "infeasible"
    assert res.objective_value is None


def test_ipopt_untouched_even_with_failed_scen(tmp_path):
    # IPOPT is reliable — solution-based override must NOT run for it.
    ipopt = _result(solver="IPOPT", obj=13250.78)
    res, meta = _run_parse(tmp_path, "IPOPT", [0, 0, 0], result=ipopt)
    assert res.converged is True
    assert res.feasibility_detail == "feasible"
    assert res.objective_value == pytest.approx(13250.78)
    assert "sopflow_solution_feasible" not in meta  # override skipped for IPOPT


def test_empar_good_solution_no_scen_files_feasible(tmp_path):
    # No scen files -> no marginal annotation, but the solution still decides.
    (tmp_path / "iter" / "sopflowout").mkdir(parents=True)
    sim = SimpleNamespace(success=True, stdout="x", workdir=tmp_path / "iter")

    def fake_parse(stdout, bus_limits=None):
        return _result(solver="EMPAR"), {"solver": "EMPAR", "num_scenarios": 0}

    with patch.object(SP, "parse_sopflow_output", side_effect=fake_parse):
        res, meta = parse_sopflow_simulation_result(sim, bus_limits=_BAND)
    assert res.feasibility_detail == "feasible"
    assert res.convergence_status == "CONVERGED"
    assert "empar_marginal" not in meta


def test_summary_handles_none_objective():
    # An infeasible result has objective_value None — the summary must not crash.
    r = _result(converged=False, obj=None)
    r.convergence_status = "DID NOT CONVERGE"
    r.feasibility_detail = "infeasible"
    text = sopflow_results_summary(r, num_scenarios=10)
    assert "N/A" in text
    assert "$0.00" not in text


def test_summary_shows_marginal_note():
    r = _result(solver="EMPAR")
    r.convergence_status = "CONVERGED (marginal)"
    text = sopflow_results_summary(r, num_scenarios=10)
    assert "marginal" in text.lower()
    assert "use with caution" in text.lower()


@pytest.mark.skipif(not _ACTIVSG200.exists(), reason="case_ACTIVSg200.m not available")
def test_real_empar_masked_workdir(tmp_path):
    # If a real EMPAR-masked workdir is on disk, the flat baseline must be flagged.
    real = Path("workdir/iter_000_20260801_161221")
    if not (real / "sopflowout").is_dir():
        pytest.skip("real EMPAR-masked workdir not present")
    assert all_scenarios_converged(real) is False


# --------------------------------------------------------------------------
# Part B: config flag + curtailable-wind base normalization
# --------------------------------------------------------------------------

def test_config_flag_default_true():
    assert load_config(None).search.sopflow_curtailable_wind_base is True


def test_config_flag_empty_search_section(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("search: {}\n", encoding="utf-8")
    assert load_config(cfg_file).search.sopflow_curtailable_wind_base is True


def test_config_flag_override_false():
    cfg = load_config(None, cli_overrides={"search.sopflow_curtailable_wind_base": False})
    assert cfg.search.sopflow_curtailable_wind_base is False


def _controller(tmp_path, application="sopflow", flag=True):
    from agentigrid.engine.agent_loop import AgentLoopController
    cfg = AppConfig(
        exago=ExagoConfig(binary_dir=tmp_path/"bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None, pflow_binary=None,
            env_script=None, timeout=30),
        data=DataConfig(data_dir=tmp_path/"data"),
        llm=LLMConfig(backend="openai", model="m", api_key_env="K", openai_base_url=None,
            ollama_host="h", ollama_cloud_host=None, temperature=0.3, max_tokens=10),
        search=SearchConfig(max_iterations=5, default_mode="fresh", base_case=None,
            gic_file=None, application=application, sopflow_curtailable_wind_base=flag),
        output=OutputConfig(workdir=tmp_path/"wd", logs_dir=tmp_path/"logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False),
    )
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor"):
        return AgentLoopController(cfg)


@pytest.mark.skipif(not _ACTIVSG200.exists(), reason="case_ACTIVSg200.m not available")
def test_partb_lowers_wind_pmin_when_on(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    ctrl = _controller(tmp_path, flag=True)
    ctrl._base_network = parse_matpower(_ACTIVSG200)
    ctrl._normalize_sopflow_wind_base()
    wind_buses = {65, 104, 105, 114, 115, 147}
    wind_gens = [g for g in ctrl._base_network.generators if g.bus in wind_buses]
    assert wind_gens, "expected wind generators in ACTIVSg200"
    assert all(g.Pmin == 0.0 for g in wind_gens)
    assert all(g.Pmax > 0.0 for g in wind_gens)  # Pmax preserved


@pytest.mark.skipif(not _ACTIVSG200.exists(), reason="case_ACTIVSg200.m not available")
def test_partb_noop_when_off(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    ctrl = _controller(tmp_path, flag=False)
    ctrl._base_network = parse_matpower(_ACTIVSG200)
    before = {g.bus: g.Pmin for g in ctrl._base_network.generators}
    ctrl._normalize_sopflow_wind_base()
    after = {g.bus: g.Pmin for g in ctrl._base_network.generators}
    assert before == after  # must-run bounds untouched


@pytest.mark.skipif(not _ACTIVSG200.exists(), reason="case_ACTIVSg200.m not available")
def test_partb_noop_for_non_sopflow(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    ctrl = _controller(tmp_path, application="opflow", flag=True)
    ctrl._base_network = parse_matpower(_ACTIVSG200)
    before = {id(g): g.Pmin for g in ctrl._base_network.generators}
    ctrl._normalize_sopflow_wind_base()
    after = {id(g): g.Pmin for g in ctrl._base_network.generators}
    assert before == after  # non-SOPFLOW untouched
