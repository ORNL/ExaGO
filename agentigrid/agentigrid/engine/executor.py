"""Simulation executor — invokes ExaGO application binaries."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from agentigrid.config import ExagoConfig, OutputConfig
from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.matpower_writer import write_matpower

logger = logging.getLogger("agentigrid.engine.executor")


@dataclass
class SimulationResult:
    """Raw result from an ExaGO simulation run."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    input_file: Path
    application: str
    error_message: Optional[str]
    workdir: Path


# ---------------------------------------------------------------------------
# Per-application command builders
# ---------------------------------------------------------------------------

def _default_cmd_builder(
    binary: Path, input_file: Path, extra_args: list[str] | None,
) -> list[str]:
    """Build command for most ExaGO applications (opflow, dcopflow, pflow, etc.)."""
    cmd = [str(binary), "-netfile", str(input_file), "-print_output"]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _tcopflow_cmd_builder(
    binary: Path, input_file: Path, extra_args: list[str] | None,
) -> list[str]:
    """Build command for TCOPFLOW (includes -save_output for multi-period results)."""
    cmd = [str(binary), "-netfile", str(input_file), "-print_output", "-save_output"]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


_CMD_BUILDERS: dict[str, Callable] = {
    "opflow": _default_cmd_builder,
    "dcopflow": _default_cmd_builder,
    "pflow": _default_cmd_builder,
    "scopflow": _default_cmd_builder,
    "tcopflow": _tcopflow_cmd_builder,
    "sopflow": _default_cmd_builder,
}


# ---------------------------------------------------------------------------
# Application-specific binary attribute names on ExagoConfig
# ---------------------------------------------------------------------------

_BINARY_ATTR = {
    "opflow": "opflow_binary",
    "scopflow": "scopflow_binary",
    "tcopflow": "tcopflow_binary",
    "sopflow": "sopflow_binary",
    "dcopflow": "dcopflow_binary",
    "pflow": "pflow_binary",
}


class SimulationExecutor:
    """Invokes ExaGO applications and captures results."""

    def __init__(self, exago_config: ExagoConfig, output_config: OutputConfig) -> None:
        """Initialise with config. Warn if binary_dir doesn't exist."""
        self._exago = exago_config
        self._output = output_config

        if not exago_config.binary_dir.exists():
            logger.warning("binary_dir does not exist: %s", exago_config.binary_dir)

        self._env_script: Path | None = exago_config.env_script
        if self._env_script and not self._env_script.exists():
            logger.warning("env_script does not exist: %s", self._env_script)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_binary(self, application: str) -> Path:
        """Resolve the path to the binary for the given application.

        Checks:
        1. Application-specific override (e.g., ``exago_config.opflow_binary``).
        2. ``{binary_dir}/{application}``.

        Raises:
            FileNotFoundError: If the binary doesn't exist at the resolved path.
        """
        # Check application-specific override
        attr = _BINARY_ATTR.get(application)
        if attr:
            override: Path | None = getattr(self._exago, attr, None)
            if override is not None:
                if not override.exists():
                    raise FileNotFoundError(
                        f"Binary override for '{application}' not found: {override}"
                    )
                return override

        # Fall back to binary_dir / application
        binary = self._exago.binary_dir / application
        if not binary.exists():
            raise FileNotFoundError(
                f"Binary for '{application}' not found at {binary}. "
                f"Copy or symlink the ExaGO binary there, or set "
                f"exago.{application}_binary in the config."
            )
        return binary

    def run(
        self,
        network: MATNetwork,
        application: str = "opflow",
        iteration: int = 0,
        extra_args: list[str] | None = None,
    ) -> SimulationResult:
        """Run an ExaGO application with the given network.

        Args:
            network: The (possibly modified) network to simulate.
            application: ExaGO application name.
            iteration: Current search iteration number.
            extra_args: Additional command-line arguments for the binary.

        Returns:
            SimulationResult with captured output.
        """
        # 1. Create working directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._output.workdir / f"iter_{iteration:03d}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 2. Write network to .m file
        input_file = run_dir / f"{network.casename}.m"
        write_matpower(network, input_file)

        # 3. Resolve binary
        try:
            binary = self.resolve_binary(application)
        except FileNotFoundError as exc:
            return SimulationResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                elapsed_seconds=0.0,
                input_file=input_file,
                application=application,
                error_message=str(exc),
                workdir=run_dir,
            )

        # 4. Build command
        builder = _CMD_BUILDERS.get(application, _default_cmd_builder)
        cmd = builder(binary, input_file, extra_args)

        if self._exago.mpi_np > 1 and application in ("scopflow", "sopflow"):
            cmd = ["mpirun", "-np", str(self._exago.mpi_np)] + cmd

        # 5/6. Execute
        logger.info("Running: %s", " ".join(cmd))
        t0 = time.monotonic()

        try:
            if self._env_script:
                shell_cmd = f"source {shlex.quote(str(self._env_script))} && {' '.join(shlex.quote(c) for c in cmd)}"
                proc = subprocess.run(
                    shell_cmd,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=self._exago.timeout,
                    cwd=run_dir,
                )
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._exago.timeout,
                    cwd=run_dir,
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            error_msg = f"Simulation timed out after {self._exago.timeout} seconds"
            logger.error(error_msg)
            return SimulationResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=error_msg,
                elapsed_seconds=elapsed,
                input_file=input_file,
                application=application,
                error_message=error_msg,
                workdir=run_dir,
            )

        elapsed = time.monotonic() - t0

        # 7. Determine success
        # MPI cleanup errors are benign if stdout has real output
        success = proc.returncode == 0
        error_message = None

        if not success:
            # Check if MPI cleanup error is the only issue
            has_output = len(proc.stdout.strip()) > 100
            mpi_noise = ("MPI_" in proc.stderr or "mpi" in proc.stderr.lower())
            if has_output and mpi_noise and proc.returncode != 0:
                logger.warning(
                    "Non-zero exit (%d) but stdout has output and stderr is MPI noise — treating as success",
                    proc.returncode,
                )
                success = True
            else:
                error_message = proc.stderr.strip() or f"Non-zero exit code: {proc.returncode}"
                logger.error("Simulation failed (exit %d): %s", proc.returncode, error_message[:200])

        logger.info(
            "Simulation %s in %.1fs (exit %d, stdout %d bytes)",
            "succeeded" if success else "FAILED",
            elapsed, proc.returncode, len(proc.stdout),
        )

        return SimulationResult(
            success=success,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
            input_file=input_file,
            application=application,
            error_message=error_message,
            workdir=run_dir,
        )

    def run_parallel(
        self,
        tasks: list[tuple[MATNetwork, str, int, list[str] | None]],
        max_workers: int = 4,
    ) -> dict[int, SimulationResult]:
        """Run multiple simulations concurrently via thread pool.

        Each simulation is an independent subprocess invocation; the GIL
        is released during I/O wait, so threads provide true concurrency.

        Args:
            tasks: List of (network, application, iteration, extra_args) tuples.
                Use distinct ``iteration`` values with variant labels to avoid
                directory collisions (e.g., pass negative iteration numbers or
                offset them so each variant gets its own workdir).
            max_workers: Maximum number of concurrent simulations.

        Returns:
            Dict mapping task index (0-based) to SimulationResult,
            preserving the input order.
        """
        if not tasks:
            return {}

        logger.info(
            "run_parallel: submitting %d simulation(s) with max_workers=%d",
            len(tasks), max_workers,
        )

        results: dict[int, SimulationResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx: dict = {}
            for idx, (network, application, iteration, extra_args) in enumerate(tasks):
                future = pool.submit(
                    self.run, network, application, iteration, extra_args,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("run_parallel: task %d raised %s: %s", idx, type(exc).__name__, exc)
                    results[idx] = SimulationResult(
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=str(exc),
                        elapsed_seconds=0.0,
                        input_file=Path("."),
                        application="unknown",
                        error_message=f"Parallel task {idx} failed: {exc}",
                        workdir=Path("."),
                    )

        return results

    def cleanup_workdir(self, result: SimulationResult) -> None:
        """Remove the working directory for a simulation run.

        Only intended to be called if ``output_config.save_modified_files``
        is False.
        """
        import shutil

        if result.workdir.exists():
            shutil.rmtree(result.workdir)
            logger.debug("Cleaned up workdir: %s", result.workdir)
