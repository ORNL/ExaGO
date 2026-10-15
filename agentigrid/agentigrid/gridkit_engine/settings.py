"""GridKit run settings.

Taken from the ``gridkit:`` section of the one shared config (binary, timeout,
solver .json values, default fault start). LLM, output folders, iteration
limit and worker count come from the other config sections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agentigrid.config import AppConfig, GridkitConfig

_DEFAULTS = GridkitConfig()
DEFAULT_BINARY_DIR = _DEFAULTS.binary_dir   # relative to cwd, like exago.binary_dir
DEFAULT_TIMEOUT_S = float(_DEFAULTS.timeout)
MAX_AUTO_WORKERS = 8


@dataclass(frozen=True)
class GridkitSettings:
    binary_dir: Path = DEFAULT_BINARY_DIR
    dynamicsimulation_binary: Optional[Path] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    workers: int = 0                # 0 = auto: min(cpu count, MAX_AUTO_WORKERS)
    workdir: Path = Path("./workdir/gridkit")
    exago_workdir: Path = Path("./workdir/exago")   # where "latest_steady_state" is looked up
    logs_dir: Path = Path("./logs/gridkit")
    max_iterations: int = 10
    tmax_s: float = _DEFAULTS.study.tmax
    dt_monitor_s: float = _DEFAULTS.study.dt_monitor
    fault_start_s: float = _DEFAULTS.fault.start
    solver_options: dict[str, Any] = field(default_factory=_DEFAULTS.study.solver_options)

    @property
    def dynamic_simulation(self) -> Path:
        return self.dynamicsimulation_binary or self.binary_dir / "DynamicSimulation"

    @property
    def contingency_analysis(self) -> Path:
        return self.binary_dir / "ContingencyAnalysis"

    def study_defaults(self) -> dict[str, float]:
        """FaultStudy fields the config sets when the LLM gives no value."""
        return {"start_s": self.fault_start_s, "tmax_s": self.tmax_s,
                "dt_monitor_s": self.dt_monitor_s}

    def resolved_workers(self, n_jobs: int) -> int:
        auto = min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
        return max(1, min(n_jobs, self.workers or auto))

    @classmethod
    def from_config(cls, cfg: AppConfig, **overrides) -> "GridkitSettings":
        """Settings from the shared config: outputs go to a ``gridkit/`` subfolder."""
        gk = cfg.gridkit
        base = dict(
            binary_dir=gk.binary_dir,
            dynamicsimulation_binary=gk.dynamicsimulation_binary,
            timeout_s=float(gk.timeout),
            workdir=Path(cfg.output.workdir) / "gridkit",
            exago_workdir=Path(cfg.output.workdir) / "exago",
            logs_dir=Path(cfg.output.logs_dir) / "gridkit",
            max_iterations=cfg.search.max_iterations,
            workers=cfg.search.sweep_max_workers,
            tmax_s=gk.study.tmax,
            dt_monitor_s=gk.study.dt_monitor,
            fault_start_s=gk.fault.start,
            solver_options=gk.study.solver_options(),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)
