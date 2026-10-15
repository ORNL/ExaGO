"""Tests for reading a GridKit run's status from exit code, console and CSV."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agentigrid.gridkit_parsers.case_parser import build_element_map, column_owners, load_case
from agentigrid.gridkit_parsers.status_parser import parse_run_status

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "applications" / "gridkit" / "DynamicSimulation"
IEEE39 = ROOT / "data" / "gridkit" / "examples" / "IEEE39.case.json"

# Console text copied from real DynamicSimulation runs on IEEE39.
_WARN = (
    "[\x1b[33;1mWARNING\x1b[0m] GridKit was not built with Enzyme. "
    "Falling back to dense Jacobians in PhasorDynamics.\n"
)
OK_STDOUT = _WARN + _WARN + "\n\nComplete in 1.33171 seconds\n"
FAIL_STDOUT = (
    _WARN + _WARN
    + "[\x1b[1;31mERROR\x1b[0m] Function IDASolve failed with flag IDA_TOO_MUCH_WORK!\n"
    + "[\x1b[1;31mERROR\x1b[0m] DynamicSimulation failed: Method in Ida class failed!\n"
)
FAIL_STDERR = (
    "[ERROR][rank 0][/home/u/GridKit/build/deps/sundials-src/src/idas/idas.c:2878]"
    "[IDASolve] At t = 4e-05, mxstep steps taken before reaching tout.\n"
)


def _csv(path: Path, times: list[float]) -> Path:
    lines = ["t,Genrou_g_omega"] + [f"{t:.16e},0.0" for t in times]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestSolved:
    def test_clean_run(self, tmp_path):
        s = parse_run_status(0, OK_STDOUT, "", _csv(tmp_path / "o.csv", [0.0, 1.0, 2.0]), tmax=2.0)
        assert s.solved
        assert s.runtime_s == pytest.approx(1.33171)
        assert s.reached_tmax and s.csv_rows == 3 and s.last_time_s == 2.0
        assert s.stderr_empty and not s.errors
        assert s.reason == "solved: reached t = 2 s in 1.33 s"

    def test_contingency_analysis_complete_line(self, tmp_path):
        # ContingencyAnalysis prints the duration as "<s>s", not "<s> seconds".
        s = parse_run_status(0, _WARN + "\n\nComplete in 1.91939s\n", "",
                             _csv(tmp_path / "o.csv", [0.0, 2.0]), tmax=2.0)
        assert s.solved and s.runtime_s == pytest.approx(1.91939)

    def test_warnings_kept_once(self, tmp_path):
        s = parse_run_status(0, OK_STDOUT, "", _csv(tmp_path / "o.csv", [2.0]), tmax=2.0)
        assert s.warnings == [
            "[WARNING] GridKit was not built with Enzyme. "
            "Falling back to dense Jacobians in PhasorDynamics."
        ]

    def test_without_csv_or_tmax(self):
        s = parse_run_status(0, OK_STDOUT, "")
        assert s.solved and s.reached_tmax is None


class TestNotSolved:
    def test_solver_failure(self, tmp_path):
        s = parse_run_status(1, FAIL_STDOUT, FAIL_STDERR, _csv(tmp_path / "o.csv", [0.0]), tmax=2.0)
        assert not s.solved
        assert (s.solver_function, s.solver_flag) == ("IDASolve", "IDA_TOO_MUCH_WORK")
        assert s.failed_at_s == pytest.approx(4e-05)
        assert not s.stderr_empty and not s.completed_line and s.reached_tmax is False
        assert s.reason.startswith("solver failed: IDASolve IDA_TOO_MUCH_WORK at t = 4e-05 s")
        assert "CSV stops at t = 0 s of 2 s (1 rows)" in s.reason

    def test_timeout(self, tmp_path):
        s = parse_run_status(None, "", "", tmp_path / "none.csv", tmax=2.0)
        assert not s.solved
        assert s.reason.startswith("did not finish")
        assert "no output CSV" in s.reason

    def test_exit_zero_but_stopped_early(self, tmp_path):
        s = parse_run_status(0, OK_STDOUT, "", _csv(tmp_path / "o.csv", [0.0, 1.0]), tmax=2.0)
        assert not s.solved
        assert s.reason.startswith("not confirmed as solved")

    def test_exit_zero_but_stderr_not_empty(self):
        s = parse_run_status(0, OK_STDOUT, "something went wrong\n")
        assert not s.solved and "stderr not empty" in s.reason

    def test_exit_zero_but_error_line(self):
        s = parse_run_status(0, OK_STDOUT + "[ERROR] Study output file not usable\n", "")
        assert not s.solved
        assert s.reason.startswith("failed: [ERROR] Study output file not usable")


@pytest.mark.skipif(
    not (BINARY.exists() and IEEE39.exists()),
    reason="GridKit DynamicSimulation or IEEE39 case not linked",
)
def test_live_ieee39_fault(tmp_path):
    """One real run: fault at bus 16 (element 0), every CSV column mapped."""
    shutil.copy(IEEE39, tmp_path / "IEEE39.case.json")
    emap = build_element_map(load_case(IEEE39), IEEE39)
    solver = {
        "system_model_file": "IEEE39.case.json",
        "dt_monitor": 0.05,
        "tmax": 1.0,
        "output_file": "out.csv",
        "events": [
            {"time": 0.2, "type": "fault_on", "element_id": 0},
            {"time": 0.3, "type": "fault_off", "element_id": 0},
        ],
    }
    (tmp_path / "study.solver.json").write_text(json.dumps(solver))
    proc = subprocess.run(
        [str(BINARY), "study.solver.json"], cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
    )
    s = parse_run_status(proc.returncode, proc.stdout, proc.stderr, tmp_path / "out.csv", tmax=1.0)
    assert s.solved, s.reason

    header = (tmp_path / "out.csv").read_text().splitlines()[0].split(",")
    owners = column_owners(emap, header[1:])
    assert all(o is not None for o in owners.values())
    assert {o["bus"] for o in owners.values()} == {m["bus"] for m in emap["machines"]}
