"""Run GridKit DynamicSimulation: one run folder per fault, several at once."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.gridkit_parsers.status_parser import RunStatus, parse_run_status

logger = logging.getLogger("agentigrid.gridkit_engine.executor")

SOLVER_FILE = "study.solver.json"
OUTPUT_FILE = "output.csv"


@dataclass
class RunJob:
    """One GridKit run: a fault (or none, for the base case) on a case copy."""

    run_dir: Path
    case_file: Path                  # case copy, shared by the jobs of one screen
    tmax_s: float
    dt_monitor_s: float
    element_id: Optional[int] = None  # GridKit bus-fault index; None = no fault
    start_s: float = 0.0
    clear_s: float = 0.0
    bus: Optional[int] = None


@dataclass
class RunResult:
    job: RunJob
    status: RunStatus
    csv_path: Path
    elapsed_s: float


def solver_file_content(job: RunJob) -> dict:
    """The solver .json for a job (paths relative to the run folder)."""
    events = []
    if job.element_id is not None:
        events = [
            {"time": job.start_s, "type": "fault_on", "element_id": job.element_id},
            {"time": job.clear_s, "type": "fault_off", "element_id": job.element_id},
        ]
    return {
        "system_model_file": os.path.relpath(job.case_file, job.run_dir),
        "dt_monitor": job.dt_monitor_s,
        "tmax": job.tmax_s,
        "output_file": OUTPUT_FILE,
        "events": events,
    }


def run_one(job: RunJob, settings: GridkitSettings) -> RunResult:
    """Write the solver file, run DynamicSimulation in the run folder, keep
    stdout/stderr, and read the run status."""
    job.run_dir.mkdir(parents=True, exist_ok=True)
    (job.run_dir / SOLVER_FILE).write_text(json.dumps(solver_file_content(job), indent=1))
    binary = settings.dynamic_simulation.resolve()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(binary), SOLVER_FILE], cwd=job.run_dir,
            capture_output=True, text=True, timeout=settings.timeout_s,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except OSError as exc:
        exit_code, stdout, stderr = None, "", f"[ERROR] cannot run {binary}: {exc}"
    elapsed = time.monotonic() - start
    (job.run_dir / "stdout.txt").write_text(stdout)
    (job.run_dir / "stderr.txt").write_text(stderr)
    csv_path = job.run_dir / OUTPUT_FILE
    status = parse_run_status(exit_code, stdout, stderr, csv_path, job.tmax_s)
    logger.debug("GridKit run %s: %s", job.run_dir.name, status.reason)
    return RunResult(job, status, csv_path, elapsed)


def run_many(jobs: list[RunJob], settings: GridkitSettings) -> list[RunResult]:
    """Run *jobs* in parallel (settings.workers); results in job order."""
    if not jobs:
        return []
    workers = settings.resolved_workers(len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: run_one(j, settings), jobs))
