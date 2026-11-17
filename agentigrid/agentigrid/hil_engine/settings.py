"""HIL run settings, from the ``hil:`` section of the shared config.

LLM, output folders and the iteration limit come from the other sections, as for
GridKit; ExaGO results are looked up in the ExaGO work folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentigrid.config import AppConfig, HilConfig

_DEFAULTS = HilConfig()
REST_STATES = ("meshed", "radial", "stopped")


@dataclass(frozen=True)
class HilSettings:
    typhoon_dir: Path = _DEFAULTS.typhoon_dir
    model: Optional[Path] = None
    compiled_model: Optional[Path] = None
    device: str = _DEFAULTS.device
    configuration_id: str = _DEFAULTS.configuration_id
    t_exec: float = _DEFAULTS.t_exec
    settle_s: float = _DEFAULTS.settle_s
    read_samples: int = _DEFAULTS.read_samples
    read_interval_s: float = _DEFAULTS.read_interval_s
    tol_pct: float = _DEFAULTS.tol_pct
    tol_deg: float = _DEFAULTS.tol_deg
    rest_state: str = _DEFAULTS.rest_state
    close_started_thcc: bool = _DEFAULTS.close_started_thcc
    startup_timeout: float = _DEFAULTS.startup_timeout
    workdir: Path = Path("./workdir/hil")
    exago_workdir: Path = Path("./workdir/exago")   # where "latest_steady_state" is looked up
    logs_dir: Path = Path("./logs/hil")
    max_iterations: int = 10

    @property
    def thcc_executable(self) -> Path:
        return self.typhoon_dir / "typhoon_hil.exe"

    def compiled_model_file(self) -> Path:
        """The .cpd Typhoon writes next to the schematic when it compiles it."""
        if self.compiled_model:
            return self.compiled_model
        if self.model is None:
            raise ValueError("No Typhoon model given (hil.model or the case argument).")
        return self.model.parent / f"{self.model.stem} Target files" / f"{self.model.stem}.cpd"

    @classmethod
    def from_config(cls, cfg: AppConfig, **overrides) -> "HilSettings":
        """Settings from the shared config: outputs go to a ``hil/`` subfolder."""
        h = cfg.hil
        base = dict(
            typhoon_dir=h.typhoon_dir, model=h.model, compiled_model=h.compiled_model,
            device=h.device, configuration_id=str(h.configuration_id), t_exec=float(h.t_exec),
            settle_s=float(h.settle_s), read_samples=int(h.read_samples),
            read_interval_s=float(h.read_interval_s), tol_pct=float(h.tol_pct),
            tol_deg=float(h.tol_deg), rest_state=h.rest_state,
            close_started_thcc=bool(h.close_started_thcc), startup_timeout=float(h.startup_timeout),
            workdir=Path(cfg.output.workdir) / "hil",
            exago_workdir=Path(cfg.output.workdir) / "exago",
            logs_dir=Path(cfg.output.logs_dir) / "hil",
            max_iterations=cfg.search.max_iterations,
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        if base["rest_state"] not in REST_STATES:
            raise ValueError(f"hil.rest_state must be one of {REST_STATES}, not {base['rest_state']!r}")
        return cls(**base)
