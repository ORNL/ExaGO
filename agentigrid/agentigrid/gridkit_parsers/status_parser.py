"""Solver status of a GridKit run, read from its exit code, console and CSV.

GridKit writes no status column and no log file. What a run leaves behind:

  exit code     0 = finished (or, with a reference file, matched it); 1 = failed
  stdout        [WARNING] / [ERROR] lines from GridKit, and on success
                "Complete in <s> seconds"
  stderr        empty on success; SUNDIALS messages on failure, e.g.
                "[ERROR][rank 0][...][IDASolve] At t = 4e-05, mxstep steps ..."
  CSV           one row per recorded time; a failed run stops early

This module checks all of them so no single signal is trusted alone. It is a
stop-gap until GridKit reports its status directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agentigrid.gridkit_parsers.status")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# DynamicSimulation: "Complete in 2.57 seconds"; ContingencyAnalysis: "Complete in 1.91939s"
_COMPLETE_RE = re.compile(r"Complete in\s+([\d.]+(?:[eE][+-]?\d+)?)\s*(?:seconds|s)\b")
_FLAG_RE = re.compile(r"Function\s+(\w+)\s+failed with flag\s+(\w+)")
_FAILED_AT_RE = re.compile(r"\bAt t\s*=\s*([\d.eE+-]+)")
_ERROR_TAG = "[ERROR]"
_WARNING_TAG = "[WARNING]"

_TMAX_REL_TOL = 1e-6


@dataclass
class RunStatus:
    """What one GridKit run tells us about how it ended."""

    solved: bool
    reason: str
    exit_code: Optional[int] = None
    completed_line: bool = False        # "Complete in ... seconds" printed
    runtime_s: Optional[float] = None   # from that line
    stderr_empty: bool = True
    solver_function: Optional[str] = None  # e.g. "IDASolve"
    solver_flag: Optional[str] = None      # e.g. "IDA_TOO_MUCH_WORK"
    failed_at_s: Optional[float] = None    # time where SUNDIALS stopped
    csv_found: bool = False
    csv_rows: int = 0
    last_time_s: Optional[float] = None
    tmax_s: Optional[float] = None
    reached_tmax: Optional[bool] = None    # None when tmax or CSV not given
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # unique lines


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in _ANSI_RE.sub("", text or "").splitlines() if line.strip()]


def _read_csv_tail(csv_path: Path) -> tuple[int, Optional[float]]:
    """Number of data rows and the time in the last row."""
    rows = 0
    last_line = ""
    with csv_path.open(encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            if line.strip():
                rows += 1
                last_line = line
    if not last_line:
        return rows, None
    try:
        return rows, float(last_line.split(",", 1)[0])
    except ValueError:
        return rows, None


def parse_run_status(
    exit_code: Optional[int],
    stdout: str,
    stderr: str,
    csv_path: Path | str | None = None,
    tmax: Optional[float] = None,
) -> RunStatus:
    """Decide whether a GridKit run solved, and why not if it did not.

    A run counts as solved only if all of these hold:
      exit code 0, "Complete in" printed, no [ERROR] line, stderr empty,
      and (when *csv_path* and *tmax* are given) the CSV reaches *tmax*.

    *exit_code* None means the run did not finish (for example a timeout).
    """
    out_lines = _clean_lines(stdout)
    err_lines = _clean_lines(stderr)

    errors = [l for l in out_lines + err_lines if l.startswith(_ERROR_TAG)]
    warnings = list(dict.fromkeys(l for l in out_lines + err_lines if l.startswith(_WARNING_TAG)))

    status = RunStatus(
        solved=False,
        reason="",
        exit_code=exit_code,
        stderr_empty=not err_lines,
        tmax_s=tmax,
        errors=errors,
        warnings=warnings,
    )

    all_text = "\n".join(out_lines + err_lines)
    m = _COMPLETE_RE.search(all_text)
    if m:
        status.completed_line = True
        status.runtime_s = float(m.group(1))
    m = _FLAG_RE.search(all_text)
    if m:
        status.solver_function, status.solver_flag = m.group(1), m.group(2)
    m = _FAILED_AT_RE.search(all_text)
    if m:
        status.failed_at_s = float(m.group(1))

    if csv_path is not None:
        csv_path = Path(csv_path)
        if csv_path.exists():
            status.csv_found = True
            status.csv_rows, status.last_time_s = _read_csv_tail(csv_path)
            if tmax is not None and status.last_time_s is not None:
                tol = _TMAX_REL_TOL * max(1.0, abs(tmax))
                status.reached_tmax = status.last_time_s >= tmax - tol
            elif tmax is not None:
                status.reached_tmax = False

    status.solved = (
        exit_code == 0
        and status.completed_line
        and not errors
        and status.stderr_empty
        and status.reached_tmax is not False
        and (csv_path is None or status.csv_found)
    )
    status.reason = _reason(status, csv_path is not None)
    return status


def _reason(s: RunStatus, csv_expected: bool) -> str:
    """One line for logs and for the LLM."""
    if s.solved:
        text = "solved"
        if s.last_time_s is not None:
            text += f": reached t = {s.last_time_s:g} s"
        if s.runtime_s is not None:
            text += f" in {s.runtime_s:.2f} s"
        return text

    if s.exit_code is None:
        head = "did not finish (no exit code, e.g. timeout)"
    elif s.solver_flag:
        head = f"solver failed: {s.solver_function} {s.solver_flag}"
        if s.failed_at_s is not None:
            head += f" at t = {s.failed_at_s:g} s"
    elif s.errors:
        head = f"failed: {s.errors[0]}"
    elif s.exit_code != 0:
        head = f"failed with exit code {s.exit_code}"
    else:
        head = "not confirmed as solved"

    details = []
    if s.exit_code not in (None, 0):
        details.append(f"exit code {s.exit_code}")
    if not s.completed_line:
        details.append("no 'Complete in' line")
    if not s.stderr_empty:
        details.append("stderr not empty")
    if csv_expected and not s.csv_found:
        details.append("no output CSV")
    elif s.reached_tmax is False:
        last = "none" if s.last_time_s is None else f"{s.last_time_s:g} s"
        details.append(f"CSV stops at t = {last} of {s.tmax_s:g} s ({s.csv_rows} rows)")
    return head + (" (" + "; ".join(details) + ")" if details else "")
