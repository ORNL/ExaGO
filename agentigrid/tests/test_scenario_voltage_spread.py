"""Per-bus wind-variability analysis for SOPFLOW.

Covers:
- compute_scenario_voltage_spread: per-bus Vm spread across scenario files, sorted
  by v_range desc, guarded (<2 files / bad workdir -> None)
- the report "Wind Variability by Bus" subsection (picks a feasible iteration with
  real variation, graceful-skips when no scenario data on disk)
- the analyze query_type scenario_voltage_spread handler
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.parsers import compute_scenario_voltage_spread

# report_generator lives in the (non-package) launcher/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
from report_generator import ReportGenerator  # noqa: E402
from reportlab.platypus import Table, Paragraph  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic scenario-file fixtures (minimal MATPOWER, only what the parser needs)
# --------------------------------------------------------------------------

def _write_scen(path: Path, vm_by_bus: dict[int, float]) -> None:
    """Write a minimal MATPOWER .m with the given per-bus Vm (col 8)."""
    lines = ["function mpc = scen", "mpc.baseMVA = 100;", "mpc.bus = ["]
    for bus, vm in vm_by_bus.items():
        # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
        lines.append(f"\t{bus}\t1\t0\t0\t0\t0\t1\t{vm}\t0\t115\t1\t1.1\t0.9;")
    lines.append("];")
    lines.append("mpc.gen = [")
    lines.append("\t1\t10\t0\t50\t-50\t1\t100\t1\t100\t0;")
    lines.append("];")
    lines.append("mpc.branch = [")
    lines.append("\t1\t2\t0.01\t0.1\t0\t100\t100\t100\t0\t0\t1\t-360\t360;")
    lines.append("];")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_workdir(tmp_path: Path, scenarios: list[dict[int, float]]) -> Path:
    wd = tmp_path / "iter_000"
    out = wd / "sopflowout"
    out.mkdir(parents=True)
    for i, vm_by_bus in enumerate(scenarios):
        _write_scen(out / f"scen_{i}.m", vm_by_bus)
    return wd


# --------------------------------------------------------------------------
# compute_scenario_voltage_spread
# --------------------------------------------------------------------------

def test_spread_ranks_by_range(tmp_path):
    # bus 2 swings most (0.95..1.05), bus 3 medium, bus 1 flat.
    wd = _make_workdir(tmp_path, [
        {1: 1.00, 2: 0.95, 3: 1.00},
        {1: 1.00, 2: 1.05, 3: 1.02},
        {1: 1.00, 2: 1.00, 3: 0.99},
    ])
    rows = compute_scenario_voltage_spread(wd)
    assert rows is not None
    assert [r["bus"] for r in rows] == [2, 3, 1]  # descending v_range
    top = rows[0]
    assert top["bus"] == 2
    assert top["v_min"] == pytest.approx(0.95)
    assert top["v_max"] == pytest.approx(1.05)
    assert top["v_range"] == pytest.approx(0.10)
    assert top["n_scenarios"] == 3
    assert top["v_std"] > 0
    # Flat bus has zero range/std.
    assert rows[-1]["bus"] == 1
    assert rows[-1]["v_range"] == pytest.approx(0.0)
    assert rows[-1]["v_std"] == pytest.approx(0.0)


def test_spread_needs_two_scenarios(tmp_path):
    wd = _make_workdir(tmp_path, [{1: 1.0, 2: 1.0}])  # only 1 scen file
    assert compute_scenario_voltage_spread(wd) is None


def test_spread_missing_workdir_returns_none(tmp_path):
    assert compute_scenario_voltage_spread(tmp_path / "nope") is None


def test_spread_no_sopflowout_returns_none(tmp_path):
    (tmp_path / "iter_x").mkdir()
    assert compute_scenario_voltage_spread(tmp_path / "iter_x") is None


def test_spread_skips_unparseable_but_uses_rest(tmp_path):
    wd = _make_workdir(tmp_path, [{1: 1.0, 2: 0.9}, {1: 1.0, 2: 1.1}])
    # Add a junk third file — must be skipped, not crash.
    (wd / "sopflowout" / "scen_2.m").write_text("not matpower at all", encoding="utf-8")
    rows = compute_scenario_voltage_spread(wd)
    assert rows is not None
    top = next(r for r in rows if r["bus"] == 2)
    assert top["n_scenarios"] == 2  # junk file skipped
    assert top["v_range"] == pytest.approx(0.2)


def test_spread_deterministic(tmp_path):
    wd = _make_workdir(tmp_path, [{1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}])  # all flat -> tie
    a = compute_scenario_voltage_spread(wd)
    b = compute_scenario_voltage_spread(wd)
    assert [r["bus"] for r in a] == [r["bus"] for r in b] == [1, 2]  # tie-break by bus


# --------------------------------------------------------------------------
# Report subsection
# --------------------------------------------------------------------------

def _entry(iteration, cwd, feasible=True, mode="fresh"):
    return SimpleNamespace(
        iteration=iteration, mode=mode, feasible=feasible,
        exago_command={"cwd": cwd, "argv": ["exago"]},
        solver="IPOPT", objective_value=1.0, voltage_min=0.98, voltage_max=1.05,
        max_line_loading_pct=50.0, violations_count=0, total_gen_mw=1.0,
        total_load_mw=1.0, feasibility_detail="",
    )


def _session(entries, application="sopflow"):
    return SimpleNamespace(journal=SimpleNamespace(entries=entries),
                           application=application)


def _para_texts(elements):
    return [el.text for el in elements if isinstance(el, Paragraph)]


def test_report_prefers_iteration_with_variation(tmp_path):
    flat = _make_workdir(tmp_path / "a", [{1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}])
    varied = _make_workdir(tmp_path / "b", [{1: 1.0, 2: 0.9}, {1: 1.0, 2: 1.1}])
    # iter 0 flat, iter 1 varied — both feasible. Helper must pick the varied one.
    session = _session([_entry(0, str(flat)), _entry(1, str(varied))])
    res = ReportGenerator()._sopflow_voltage_spread(session)
    assert res is not None
    entry, rows = res
    assert entry.iteration == 1
    assert rows[0]["bus"] == 2 and rows[0]["v_range"] == pytest.approx(0.2)


def test_report_section_renders_variability_table(tmp_path):
    varied = _make_workdir(tmp_path, [
        {b: (1.0 + (0.01 if b == 5 else 0.0)) for b in range(1, 20)},
        {b: (1.0 - (0.02 if b == 5 else 0.0)) for b in range(1, 20)},
    ])
    session = _session([_entry(1, str(varied))])
    els = ReportGenerator()._build_sopflow_stochastic_section(session, 2)
    tables = [t for t in els if isinstance(t, Table)]
    var = [t for t in tables if t._cellvalues[0][0] == "Bus"]
    assert len(var) == 1
    assert var[0]._cellvalues[0] == [
        "Bus", "V_min (pu)", "V_max (pu)", "V_range (pu)", "V_std (pu)",
    ]
    # bus 5 swings most -> ranked first.
    assert var[0]._cellvalues[1][0] == "5"
    texts = _para_texts(els)
    assert any("Wind Variability by Bus" in t for t in texts)
    assert any("most wind-affected" in t for t in texts)


def test_report_section_graceful_when_no_scenarios():
    # No exago_command cwd on disk -> caption, no crash, no variability table.
    e = SimpleNamespace(iteration=1, mode="fresh", feasible=True, exago_command=None,
                        solver="IPOPT", objective_value=1.0, voltage_min=0.98,
                        voltage_max=1.05, max_line_loading_pct=50.0,
                        violations_count=0, total_gen_mw=1.0, total_load_mw=1.0,
                        feasibility_detail="")
    els = ReportGenerator()._build_sopflow_stochastic_section(_session([e]), 5)
    tables = [t for t in els if isinstance(t, Table)]
    assert all(t._cellvalues[0][0] != "Bus" for t in tables)  # no variability table
    texts = _para_texts(els)
    assert any("Wind Variability by Bus" in t for t in texts)
    assert any("could not be recomputed" in t for t in texts)


def test_report_section_never_raises_on_bad_workdir():
    session = _session([_entry(1, "/definitely/not/here")])
    els = ReportGenerator()._build_sopflow_stochastic_section(session, 5)
    assert any("could not be recomputed" in t for t in _para_texts(els))


# --------------------------------------------------------------------------
# analyze query_type handler
# --------------------------------------------------------------------------

def _controller(tmp_path, application="sopflow"):
    from agentigrid.config import (
        AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
    )
    from agentigrid.engine.agent_loop import AgentLoopController

    cfg = AppConfig(
        exago=ExagoConfig(binary_dir=tmp_path/"bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None, pflow_binary=None,
            env_script=None, timeout=30),
        data=DataConfig(data_dir=tmp_path/"data"),
        llm=LLMConfig(backend="openai", model="m", api_key_env="K", openai_base_url=None,
            ollama_host="h", ollama_cloud_host=None, temperature=0.3, max_tokens=10),
        search=SearchConfig(max_iterations=5, default_mode="fresh", base_case=None,
            gic_file=None, application=application),
        output=OutputConfig(workdir=tmp_path/"wd", logs_dir=tmp_path/"logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False),
    )
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor"):
        return AgentLoopController(cfg)


def _push_sim_entry(ctrl, cwd, iteration=1):
    from agentigrid.engine.journal import JournalEntry
    ctrl._journal._entries.append(JournalEntry(
        iteration=iteration, description="sopflow", commands=[], objective_value=1.0,
        feasible=True, convergence_status="CONVERGED", violations_count=0,
        voltage_min=0.98, voltage_max=1.05, max_line_loading_pct=50.0,
        total_gen_mw=1.0, total_load_mw=1.0, llm_reasoning="", mode="fresh",
        elapsed_seconds=1.0, solver="IPOPT",
        exago_command={"cwd": str(cwd), "argv": ["exago"]},
    ))


def test_analyze_handler_returns_ranked_table(tmp_path):
    wd = _make_workdir(tmp_path, [{1: 1.0, 2: 0.95, 3: 1.0}, {1: 1.0, 2: 1.05, 3: 1.02}])
    ctrl = _controller(tmp_path)
    _push_sim_entry(ctrl, wd)
    kind, cont = ctrl._handle_scenario_voltage_spread(5, {
        "query_type": "scenario_voltage_spread", "k": 2,
    })
    assert (kind, cont) == ("analyze", True)
    txt = ctrl._latest_results_text
    assert "most affected by wind variability" in txt
    assert "top 2 of 3 buses" in txt
    # bus 2 has the biggest spread -> appears first in the table body.
    body = txt.splitlines()[3:]
    assert body[0].split("|")[0].strip() == "2"
    # journaled as an ANALYSIS entry carrying the query in its description.
    last = ctrl._journal.entries[-1]
    assert last.mode == "analyze"
    assert last.description == "Analysis: scenario_voltage_spread k=2"


def test_analyze_handler_rejects_non_sopflow(tmp_path):
    ctrl = _controller(tmp_path, application="opflow")
    kind, cont = ctrl._handle_scenario_voltage_spread(1, {"query_type": "scenario_voltage_spread"})
    assert kind == "error"
    assert "only available for the SOPFLOW" in ctrl._error_feedback


def test_analyze_handler_no_workdir(tmp_path):
    ctrl = _controller(tmp_path)  # no sim entries -> no cwd
    kind, cont = ctrl._handle_scenario_voltage_spread(1, {"query_type": "scenario_voltage_spread"})
    assert kind == "error"
    assert "No per-scenario second-stage voltages" in ctrl._error_feedback


def test_analyze_dispatch_routes_query_type(tmp_path):
    wd = _make_workdir(tmp_path, [{1: 1.0, 2: 0.9}, {1: 1.0, 2: 1.1}])
    ctrl = _controller(tmp_path)
    _push_sim_entry(ctrl, wd)
    # Go through the public analyze dispatch, not the handler directly.
    kind, cont = ctrl._handle_analyze(3, {"query_type": "scenario_voltage_spread", "k": 5})
    assert kind == "analyze"
    assert "wind variability" in ctrl._latest_results_text
