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
from agentigrid.gridkit_engine.sweep_metrics import CheckOptions

_DEFAULTS = GridkitConfig()
DEFAULT_BINARY_DIR = _DEFAULTS.binary_dir   # relative to cwd, like exago.binary_dir
DEFAULT_TIMEOUT_S = float(_DEFAULTS.timeout)
# Thread-count variables set for each GridKit run when settings.threads is given.
THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
MAX_AUTO_WORKERS = 8


@dataclass(frozen=True)
class GridkitSettings:
    binary_dir: Path = DEFAULT_BINARY_DIR
    dynamicsimulation_binary: Optional[Path] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    threads: Optional[int] = _DEFAULTS.threads   # per run; None = inherit the environment
    damping_in_verdict: bool = _DEFAULTS.damping_in_verdict
    damping_method: str = _DEFAULTS.damping_method
    final_voltage_window_s: float = _DEFAULTS.final_voltage_window_s
    final_voltage_margin_pu: float = _DEFAULTS.final_voltage_margin_pu
    short_circuit_info: bool = _DEFAULTS.short_circuit_info
    weak_bus_percentile: float = _DEFAULTS.weak_bus_percentile
    final_voltage_pre_fault_delta_pu: float = _DEFAULTS.final_voltage_pre_fault_delta_pu
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

    @property
    def info_checks(self) -> frozenset[str]:
        """Checks computed and reported but kept out of the verdict (status "info")."""
        return frozenset() if self.damping_in_verdict else frozenset({"damping"})

    @property
    def check_options(self) -> CheckOptions:
        """How the damping and final-voltage checks are computed (all off by default)."""
        return CheckOptions(self.damping_method, float(self.final_voltage_window_s),
                            float(self.final_voltage_margin_pu), bool(self.short_circuit_info),
                            float(self.weak_bus_percentile),
                            final_voltage_pre_fault_delta_pu=float(self.final_voltage_pre_fault_delta_pu))

    def subprocess_env(self) -> Optional[dict[str, str]]:
        """Environment for a GridKit process: ours with the thread variables set
        to *threads*, or None (inherit unchanged) when threads is None. Several
        runs go at once, so threads per run times workers should fit the cores."""
        if self.threads is None:
            return None
        return {**os.environ, **{v: str(self.threads) for v in THREAD_ENV_VARS}}

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
            threads=gk.threads,
            damping_in_verdict=gk.damping_in_verdict,
            damping_method=gk.damping_method,
            final_voltage_window_s=gk.final_voltage_window_s,
            final_voltage_margin_pu=gk.final_voltage_margin_pu,
            short_circuit_info=gk.short_circuit_info,
            weak_bus_percentile=gk.weak_bus_percentile,
            final_voltage_pre_fault_delta_pu=gk.final_voltage_pre_fault_delta_pu,
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
