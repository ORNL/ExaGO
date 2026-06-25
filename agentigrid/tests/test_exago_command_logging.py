"""Tests for per-iteration ExaGO invocation logging into the JSON journal.

Covers:
- SimulationExecutor.run() captures argv / shell_command / env_overrides / cwd
  (incl. the mpirun prefix and the env-script shell wrapper, thread pinning).
- The single-call and multi-call (sweep/explore) record shapes.
- JournalEntry.exago_command serializes via export_json and export_csv.
- The field is NOT present in the LLM-facing text (format_detailed / format_for_prompt).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import ExagoConfig, OutputConfig
from agentigrid.engine.agent_loop import _single_call_record, _multi_call_record
from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.engine.journal import SearchJournal
from agentigrid.parsers import parse_matpower

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACTIVSG200 = DATA_DIR / "case_ACTIVSg200.m"
ENV_SCRIPT_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "configs" / "env_setup.sh.template"
)
_has_test_file = ACTIVSG200.exists()


def _make_config(tmp_path, binary_dir, env_script=None, mpi_np=1):
    exago = ExagoConfig(
        binary_dir=binary_dir, opflow_binary=None, scopflow_binary=None,
        tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
        pflow_binary=None, env_script=env_script, timeout=120, mpi_np=mpi_np,
    )
    output = OutputConfig(
        workdir=tmp_path / "workdir", logs_dir=tmp_path / "logs", save_journal=True,
        journal_format="json", save_modified_files=True, verbose=False,
    )
    return exago, output


def _fake_bin(bin_dir: Path, name: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    b = bin_dir / name
    b.write_text("#!/bin/bash\necho OK; exit 0\n")
    b.chmod(0o755)
    return b


# ---------------------------------------------------------------------------
# Executor capture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_test_file, reason="ACTIVSg200 not available")
class TestExecutorCapture:

    def test_argv_and_cwd_captured(self, tmp_path):
        _fake_bin(tmp_path / "bin", "opflow")
        exago, output = _make_config(tmp_path, tmp_path / "bin")
        ex = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            res = ex.run(net, application="opflow", iteration=0)
        assert res.argv and res.argv[0].endswith("opflow")
        assert "-netfile" in res.argv
        assert res.cwd is not None and Path(res.cwd).name.startswith("iter_")
        # no env script, no thread limit → these stay empty
        assert res.shell_command is None
        assert res.env_overrides is None

    def test_thread_limit_env_overrides_captured(self, tmp_path):
        _fake_bin(tmp_path / "bin", "opflow")
        exago, output = _make_config(tmp_path, tmp_path / "bin")
        ex = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            res = ex.run(net, application="opflow", iteration=1, thread_limit=1)
        assert res.env_overrides is not None
        assert res.env_overrides.get("OMP_NUM_THREADS") == "1"

    def test_mpirun_prefix_captured_in_argv(self, tmp_path):
        _fake_bin(tmp_path / "bin", "scopflow")
        exago, output = _make_config(tmp_path, tmp_path / "bin", mpi_np=4)
        ex = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            res = ex.run(net, application="scopflow", iteration=0)
        assert res.argv[:3] == ["mpirun", "-np", "4"]

    def test_env_script_shell_command_captured(self, tmp_path):
        _fake_bin(tmp_path / "bin", "opflow")
        env_script = tmp_path / "env.sh"
        env_script.write_text("#!/bin/bash\ntrue\n")
        exago, output = _make_config(tmp_path, tmp_path / "bin", env_script=env_script)
        ex = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            res = ex.run(net, application="opflow", iteration=2, thread_limit=1)
        assert res.shell_command is not None
        assert "source" in res.shell_command and str(env_script) in res.shell_command
        assert "OMP_NUM_THREADS=1" in res.shell_command  # thread pinning in the wrapper

    def test_binary_not_found_has_empty_argv_but_cwd(self, tmp_path):
        # binary_dir without an opflow binary → early return
        exago, output = _make_config(tmp_path, tmp_path / "empty_bin")
        (tmp_path / "empty_bin").mkdir()
        ex = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        res = ex.run(net, application="opflow", iteration=0)
        assert res.success is False
        assert res.argv == []
        assert res.shell_command is None
        assert res.cwd is not None  # run_dir was created


# ---------------------------------------------------------------------------
# Record shapes
# ---------------------------------------------------------------------------

def _sim(argv=None, app="opflow", shell=None, env=None, cwd="/wd/iter_000"):
    return SimulationResult(
        success=True, exit_code=0, stdout="", stderr="", elapsed_seconds=0.1,
        input_file=Path("/wd/x.m"), application=app, error_message=None,
        workdir=Path("/wd"),
        argv=argv if argv is not None else ["/bin/opflow", "-netfile", "/wd/x.m", "-print_output"],
        shell_command=shell, env_overrides=env, cwd=cwd,
    )


class TestRecordShapes:

    def test_single_call_record_shape(self):
        rec = _single_call_record(_sim())
        assert rec["mode"] == "single"
        assert rec["application"] == "opflow"
        assert "/bin/opflow" in rec["command"] and "-netfile" in rec["command"]
        assert rec["argv"][0] == "/bin/opflow"
        assert rec["shell_command"] is None
        assert rec["env_overrides"] is None
        assert rec["cwd"] == "/wd/iter_000"

    def test_single_call_record_none_for_missing(self):
        assert _single_call_record(None) is None
        assert _single_call_record(_sim(argv=[])) is None  # no captured argv

    def test_single_call_record_preserves_wrapper(self):
        rec = _single_call_record(_sim(
            shell="source env.sh && export OMP_NUM_THREADS=1 && /bin/opflow -netfile x.m",
            env={"OMP_NUM_THREADS": "1"},
        ))
        assert "source env.sh" in rec["shell_command"]
        assert rec["env_overrides"]["OMP_NUM_THREADS"] == "1"

    def test_multi_call_record_shape(self):
        rec = _multi_call_record("sweep", 200, _sim(), "Executed once per candidate bus.")
        assert rec["mode"] == "sweep"
        assert rec["candidate_count"] == 200
        assert rec["representative"]["mode"] == "single"
        assert rec["note"] and "candidate" in rec["note"]

    def test_multi_call_record_none_without_representative(self):
        assert _multi_call_record("sweep", 5, None, "note") is None


# ---------------------------------------------------------------------------
# Journal serialization + LLM-facing exclusion
# ---------------------------------------------------------------------------

class TestJournalSerialization:

    def _journal_with_entries(self):
        j = SearchJournal()
        # single-call modify entry
        j.add_from_results(
            iteration=1, description="modify", commands=[{"action": "scale_all_loads", "factor": 1.1}],
            opflow_result=None, sim_elapsed=0.1, llm_reasoning="r", mode="accumulative",
            exago_command=_single_call_record(_sim()),
        )
        # sweep entry
        j.add_sweep(
            iteration=2, description="[sweep] siting", candidate_count=200,
            candidate_summaries=[{"bus": 1, "feasible": True, "cost": 100.0}],
            feasible_buses=[1],
            exago_command=_multi_call_record(
                "sweep", 200, _sim(), "Executed once per candidate bus (netfile differs)."
            ),
        )
        return j

    def test_single_call_entry_command_has_binary_and_netfile(self):
        j = self._journal_with_entries()
        entry = j.entries[0]
        assert entry.exago_command["mode"] == "single"
        assert "opflow" in entry.exago_command["command"]
        assert "-netfile" in entry.exago_command["command"]

    def test_sweep_entry_record(self):
        j = self._journal_with_entries()
        sweep = j.get_sweep_entry()
        rec = sweep.exago_command
        assert rec["mode"] == "sweep"
        assert rec["candidate_count"] == 200
        assert rec["note"]  # non-empty

    def test_export_json_includes_field(self, tmp_path):
        j = self._journal_with_entries()
        out = tmp_path / "journal.json"
        j.export_json(out)
        data = json.loads(out.read_text())
        entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
        recs = [e.get("exago_command") for e in entries]
        assert any(r and r.get("mode") == "single" for r in recs)
        assert any(r and r.get("mode") == "sweep" for r in recs)

    def test_export_csv_roundtrips_field(self, tmp_path):
        j = self._journal_with_entries()
        out = tmp_path / "journal.csv"
        j.export_csv(out)  # must not raise (fieldnames updated)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert "exago_command" in rows[0]
        decoded = [json.loads(r["exago_command"]) for r in rows]
        assert any(d and d.get("mode") == "single" for d in decoded)
        assert any(d and d.get("mode") == "sweep" for d in decoded)

    def test_field_absent_from_llm_facing_text(self):
        j = self._journal_with_entries()
        detailed = j.format_detailed()
        for_prompt = j.format_for_prompt()
        assert "exago_command" not in detailed
        assert "exago_command" not in for_prompt
        # the captured binary path / netfile flag must not leak into LLM text either
        assert "-netfile" not in detailed
        assert "-netfile" not in for_prompt
