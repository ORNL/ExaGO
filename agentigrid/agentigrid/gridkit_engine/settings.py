"""GridKit run settings.

The one shared config has no ``gridkit:`` section yet (draft kept in the
session notes until the GridKit team's solver .json change lands). Until
then these defaults apply; LLM, output folders, iteration limit and worker
count come from the existing config sections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentigrid.config import AppConfig

DEFAULT_BINARY_DIR = Path("./applications/gridkit")   # relative to cwd, like exago.binary_dir
DEFAULT_TIMEOUT_S = 600.0
MAX_AUTO_WORKERS = 8


@dataclass(frozen=True)
class GridkitSettings:
    binary_dir: Path = DEFAULT_BINARY_DIR
    timeout_s: float = DEFAULT_TIMEOUT_S
    workers: int = 0                # 0 = auto: min(cpu count, MAX_AUTO_WORKERS)
    workdir: Path = Path("./workdir/gridkit")
    logs_dir: Path = Path("./logs/gridkit")
    max_iterations: int = 10

    @property
    def dynamic_simulation(self) -> Path:
        return self.binary_dir / "DynamicSimulation"

    def resolved_workers(self, n_jobs: int) -> int:
        auto = min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
        return max(1, min(n_jobs, self.workers or auto))

    @classmethod
    def from_config(cls, cfg: AppConfig, **overrides) -> "GridkitSettings":
        """Settings from the shared config: outputs go to a ``gridkit/`` subfolder."""
        base = dict(
            workdir=Path(cfg.output.workdir) / "gridkit",
            logs_dir=Path(cfg.output.logs_dir) / "gridkit",
            max_iterations=cfg.search.max_iterations,
            workers=cfg.search.sweep_max_workers,
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)
