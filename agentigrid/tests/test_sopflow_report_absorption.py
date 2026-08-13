"""Wind Absorption (Second Stage) block in the SOPFLOW PDF report section.

Guards the report-time recomputation of second-stage wind absorption:
- rows are collected from real simulation iterations with an on-disk workdir
- a missing/cleaned workdir (or a compute exception) degrades to a caption, never
  an error, so PDF generation always completes
- non-SOPFLOW reports are unaffected
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# report_generator lives in the (non-package) launcher/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launcher"))
from report_generator import ReportGenerator  # noqa: E402

from reportlab.platypus import Table, Paragraph  # noqa: E402


def _entry(iteration, mode="fresh", cwd="/wd", scenfile="/wd/scen.csv",
           argv_has_scenfile=True, command=None):
    argv = ["exago", "-netfile", "x.m"]
    if argv_has_scenfile and scenfile is not None:
        argv += ["-scenfile", scenfile]
    cmd = {"cwd": cwd, "argv": argv}
    if command is not None:
        cmd["command"] = command
    return SimpleNamespace(
        iteration=iteration, mode=mode, exago_command=cmd,
        feasible=True, objective_value=1000.0, voltage_min=0.96,
        voltage_max=1.04, max_line_loading_pct=80.0, violations_count=0,
        total_gen_mw=500.0, total_load_mw=490.0, feasibility_detail="",
        solver="IPOPT",
    )


def _session(entries, application="sopflow", num_scenarios=5):
    journal = SimpleNamespace(entries=entries)
    return SimpleNamespace(journal=journal, application=application,
                           sopflow_num_scenarios=num_scenarios)


_WA = {
    "num_scenarios": 5, "total_available_mw": 8000.0,
    "total_dispatched_mw": 1200.0, "total_curtailment_mw": 6800.0,
    "curtailment_pct": 85.0, "avg_dispatched_mw": 240.0,
}


# --------------------------------------------------------------------------
# _sopflow_absorption_rows
# --------------------------------------------------------------------------

def test_absorption_rows_from_argv():
    session = _session([_entry(1), _entry(2)])
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=_WA) as m:
        rows = ReportGenerator()._sopflow_absorption_rows(session)
    assert len(rows) == 2
    assert [e.iteration for e, _ in rows] == [1, 2]
    # cwd + scenfile were passed through to the recompute.
    (call_cwd, call_scen) = m.call_args[0]
    assert str(call_cwd) == "/wd"
    assert str(call_scen).endswith("scen.csv")


def test_absorption_rows_fallback_to_command_string():
    # argv lacks -scenfile; the flat command string carries it instead.
    e = _entry(1, argv_has_scenfile=False,
               command="exago -netfile x.m -scenfile /wd/from_cmd.csv")
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=_WA) as m:
        rows = ReportGenerator()._sopflow_absorption_rows(_session([e]))
    assert len(rows) == 1
    assert str(m.call_args[0][1]).endswith("from_cmd.csv")


def test_absorption_rows_skip_non_simulation_modes():
    session = _session([_entry(1, mode="sweep"), _entry(2, mode="explore")])
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=_WA):
        rows = ReportGenerator()._sopflow_absorption_rows(session)
    assert rows == []


def test_absorption_rows_skip_missing_command():
    e = SimpleNamespace(iteration=1, mode="fresh", exago_command=None)
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=_WA):
        rows = ReportGenerator()._sopflow_absorption_rows(_session([e]))
    assert rows == []


def test_absorption_rows_none_result_skipped():
    # workdir gone / files missing → compute returns None → no row.
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=None):
        rows = ReportGenerator()._sopflow_absorption_rows(_session([_entry(1)]))
    assert rows == []


def test_absorption_rows_exception_guarded():
    # A bad workdir must never propagate out of report rendering.
    with patch("agentigrid.parsers.compute_wind_absorption",
               side_effect=RuntimeError("bad workdir")):
        rows = ReportGenerator()._sopflow_absorption_rows(_session([_entry(1)]))
    assert rows == []


# --------------------------------------------------------------------------
# _build_sopflow_stochastic_section
# --------------------------------------------------------------------------

def _para_texts(elements):
    return [el.text for el in elements if isinstance(el, Paragraph)]


def test_section_renders_absorption_table():
    session = _session([_entry(1), _entry(2)])
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=_WA):
        elements = ReportGenerator()._build_sopflow_stochastic_section(session, 5)

    tables = [el for el in elements if isinstance(el, Table)]
    # base-case summary table + the new absorption table.
    assert len(tables) == 2
    abs_table = tables[-1]
    header = [c for c in abs_table._cellvalues[0]]
    assert header == ["Iter", "Offered (MW)", "Dispatched (MW)",
                      "Curtailed (MW)", "Curtailed (%)"]
    # Two data rows with formatted values.
    assert abs_table._cellvalues[1][1] == "8,000.0"
    assert abs_table._cellvalues[1][4] == "85.0"
    assert len(abs_table._cellvalues) == 3  # header + 2 iterations

    texts = _para_texts(elements)
    assert any("Wind Absorption (Second Stage)" in t for t in texts)
    assert any("absorption ceiling" in t for t in texts)


def test_section_graceful_when_no_rows():
    session = _session([_entry(1)])
    with patch("agentigrid.parsers.compute_wind_absorption", return_value=None):
        elements = ReportGenerator()._build_sopflow_stochastic_section(session, 5)

    tables = [el for el in elements if isinstance(el, Table)]
    assert len(tables) == 1  # only the base-case summary table
    texts = _para_texts(elements)
    assert any("Wind Absorption (Second Stage)" in t for t in texts)
    assert any("could not be recomputed" in t for t in texts)


def test_section_never_raises_on_bad_workdir():
    session = _session([_entry(1, cwd="/definitely/not/here")])
    with patch("agentigrid.parsers.compute_wind_absorption",
               side_effect=OSError("gone")):
        elements = ReportGenerator()._build_sopflow_stochastic_section(session, 5)
    # Falls back to the "not available" caption, no exception.
    assert any("could not be recomputed" in t for t in _para_texts(elements))
