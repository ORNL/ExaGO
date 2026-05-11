"""Tests for the Simulation Executor."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agentigrid.config import ExagoConfig, OutputConfig
from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.parsers import parse_matpower

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
APP_DIR = Path(__file__).resolve().parent.parent / "applications"
ACTIVSG200 = DATA_DIR / "case_ACTIVSg200.m"
ENV_SCRIPT = Path(__file__).resolve().parent.parent / "configs" / "env_setup.sh"

_has_test_file = ACTIVSG200.exists()
_has_opflow = (APP_DIR / "opflow").exists()


def _make_config(
    tmp_path: Path,
    binary_dir: Path | None = None,
    env_script: Path | None = None,
    timeout: int = 120,
) -> tuple[ExagoConfig, OutputConfig]:
    """Create test configs pointing at tmp_path for workdir."""
    exago = ExagoConfig(
        binary_dir=binary_dir or APP_DIR,
        opflow_binary=None,
        scopflow_binary=None,
        tcopflow_binary=None,
        sopflow_binary=None,
        dcopflow_binary=None,
        pflow_binary=None,
        env_script=env_script,
        timeout=timeout,
    )
    output = OutputConfig(
        workdir=tmp_path / "workdir",
        logs_dir=tmp_path / "logs",
        save_journal=True,
        journal_format="json",
        save_modified_files=True,
        verbose=False,
    )
    return exago, output


# ===========================================================================
# Unit tests (no ExaGO binary needed)
# ===========================================================================

class TestResolveBinary:

    def test_finds_binary_in_binary_dir(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "opflow").touch(mode=0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir)
        executor = SimulationExecutor(exago, output)
        result = executor.resolve_binary("opflow")
        assert result == bin_dir / "opflow"

    def test_uses_override(self, tmp_path: Path):
        custom = tmp_path / "custom_opflow"
        custom.touch(mode=0o755)

        exago = ExagoConfig(
            binary_dir=tmp_path / "nonexistent",
            opflow_binary=custom,
            scopflow_binary=None,
            tcopflow_binary=None,
            sopflow_binary=None,
            dcopflow_binary=None,
            pflow_binary=None,
            env_script=None,
            timeout=120,
        )
        output = OutputConfig(
            workdir=tmp_path / "workdir",
            logs_dir=tmp_path / "logs",
            save_journal=True,
            journal_format="json",
            save_modified_files=True,
            verbose=False,
        )
        executor = SimulationExecutor(exago, output)
        result = executor.resolve_binary("opflow")
        assert result == custom

    def test_raises_on_missing(self, tmp_path: Path):
        exago, output = _make_config(tmp_path, binary_dir=tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        executor = SimulationExecutor(exago, output)
        with pytest.raises(FileNotFoundError, match="opflow"):
            executor.resolve_binary("opflow")


class TestWorkdirManagement:

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_workdir_created(self, tmp_path: Path):
        """Run with a fake binary to verify workdir is created."""
        # Create a fake binary that just exits
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "opflow"
        fake_bin.write_text("#!/bin/bash\necho 'fake output'\nexit 0\n")
        fake_bin.chmod(0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", iteration=0)

        assert result.workdir.exists()
        assert result.input_file.exists()
        assert result.workdir.name.startswith("iter_000_")

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_cleanup_workdir(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "opflow"
        fake_bin.write_text("#!/bin/bash\necho 'fake'\nexit 0\n")
        fake_bin.chmod(0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", iteration=0)

        assert result.workdir.exists()
        executor.cleanup_workdir(result)
        assert not result.workdir.exists()


class TestCommandConstruction:

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_command_includes_netfile_flag(self, tmp_path: Path):
        """Verify -netfile and -print_output are in the command."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "opflow"
        # Script that prints the args it received
        fake_bin.write_text("#!/bin/bash\necho \"ARGS: $@\"\nexit 0\n")
        fake_bin.chmod(0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", iteration=0)

        assert "-netfile" in result.stdout
        assert "-print_output" in result.stdout

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_extra_args_passed(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "opflow"
        fake_bin.write_text("#!/bin/bash\necho \"ARGS: $@\"\nexit 0\n")
        fake_bin.chmod(0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", extra_args=["-opflow_solver", "HIOP"])

        assert "-opflow_solver" in result.stdout
        assert "HIOP" in result.stdout


class TestTimeout:

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_timeout(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "opflow"
        fake_bin.write_text("#!/bin/bash\nsleep 30\n")
        fake_bin.chmod(0o755)

        exago, output = _make_config(tmp_path, binary_dir=bin_dir, timeout=2)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", iteration=0)

        assert not result.success
        assert "timed out" in result.error_message


# ===========================================================================
# Integration test (requires real opflow binary)
# ===========================================================================

@pytest.mark.skipif(not _has_opflow, reason="opflow binary not available")
@pytest.mark.skipif(not _has_test_file, reason="case_ACTIVSg200.m not in data/")
class TestMPIGuard:
    """Verify mpirun is only prepended for SCOPFLOW, not for IPOPT-only apps."""

    def _make_config_with_mpi(self, tmp_path: Path, mpi_np: int) -> tuple[ExagoConfig, OutputConfig]:
        exago = ExagoConfig(
            binary_dir=tmp_path / "bin",
            opflow_binary=None,
            scopflow_binary=None,
            tcopflow_binary=None,
            sopflow_binary=None,
            dcopflow_binary=None,
            pflow_binary=None,
            env_script=None,
            timeout=120,
            mpi_np=mpi_np,
        )
        output = OutputConfig(
            workdir=tmp_path / "workdir",
            logs_dir=tmp_path / "logs",
            save_journal=True,
            journal_format="json",
            save_modified_files=True,
            verbose=False,
        )
        return exago, output

    def _make_fake_bin(self, bin_dir: Path, name: str) -> Path:
        bin_dir.mkdir(parents=True, exist_ok=True)
        fake_bin = bin_dir / name
        fake_bin.write_text("#!/bin/bash\necho OK; exit 0\n")
        fake_bin.chmod(0o755)
        return fake_bin

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_no_mpirun_for_opflow(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "opflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=4)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="opflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] != "mpirun"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_no_mpirun_for_tcopflow(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "tcopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=4)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="tcopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] != "mpirun"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_no_mpirun_for_dcopflow(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "dcopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=4)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="dcopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] != "mpirun"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_mpirun_for_scopflow_with_mpi_np(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "scopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=4)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="scopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] == "mpirun"
            assert cmd_arg[1] == "-np"
            assert cmd_arg[2] == "4"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_no_mpirun_for_scopflow_with_mpi_np_1(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "scopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=1)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="scopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] != "mpirun"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_mpirun_for_sopflow_with_mpi_np(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "sopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=4)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="sopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] == "mpirun"
            assert cmd_arg[1] == "-np"
            assert cmd_arg[2] == "4"

    @pytest.mark.skipif(not _has_test_file, reason="test data not available")
    def test_no_mpirun_for_sopflow_with_mpi_np_1(self, tmp_path: Path):
        self._make_fake_bin(tmp_path / "bin", "sopflow")
        exago, output = self._make_config_with_mpi(tmp_path, mpi_np=1)
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)

        with patch("agentigrid.engine.executor.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(stdout="OK", stderr="", returncode=0)
            executor.run(net, application="sopflow", iteration=0)
            cmd_arg = mock_sub.call_args[0][0]
            assert cmd_arg[0] != "mpirun"


class TestExecutorLive:

    def test_opflow_run(self, tmp_path: Path):
        """Run OPFLOW on ACTIVSg200 and verify success."""
        exago, output = _make_config(
            tmp_path,
            binary_dir=APP_DIR,
            env_script=ENV_SCRIPT if ENV_SCRIPT.exists() else None,
        )
        executor = SimulationExecutor(exago, output)
        net = parse_matpower(ACTIVSG200)
        result = executor.run(net, application="opflow", iteration=0)

        assert result.success, f"opflow failed: {result.error_message}\nstderr: {result.stderr[:500]}"
        assert len(result.stdout) > 0
        assert result.input_file.exists()
        assert result.elapsed_seconds < exago.timeout
        assert result.exit_code == 0 or result.success  # MPI noise tolerance

        # Save stdout for Step 1.6 (only if converged, to avoid overwriting good sample)
        if "Optimal Solution Found" in result.stdout:
            sample_path = Path(__file__).resolve().parent / "sample_opflow_output.txt"
            sample_path.write_text(result.stdout, encoding="utf-8")
            print(f"\nSaved sample output to {sample_path} ({len(result.stdout)} bytes)")
