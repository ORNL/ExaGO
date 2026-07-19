"""YAML configuration loading, merging, and validation for AgentiGrid."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("agentigrid.config")

# ---------------------------------------------------------------------------
# Default values (mirrors configs/default_config.yaml)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "exago": {
        "binary_dir": "./applications",
        "opflow_binary": None,
        "scopflow_binary": None,
        "tcopflow_binary": None,
        "sopflow_binary": None,
        "dcopflow_binary": None,
        "pflow_binary": None,
        "env_script": None,
        "timeout": 600,
        "mpi_np": 1,
    },
    "data": {
        "data_dir": "./data",
    },
    "llm": {
        "backend": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "openai_base_url": None,
        "ollama_host": "http://localhost:11434",
        "ollama_cloud_host": None,
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "search": {
        "max_iterations": 20,
        "default_mode": "accumulative",
        "base_case": None,
        "gic_file": None,
        "ctgc_file": None,
        "pload_profile": None,
        "qload_profile": None,
        "wind_profile": None,
        "tcopflow_duration": 1.0,
        "tcopflow_dT": 60.0,
        "tcopflow_iscoupling": 1,
        "scenario_file": None,
        "sopflow_solver": "IPOPT",
        "sopflow_iscoupling": 0,
        "application": "opflow",
        "search_mode": "standard",
        "concurrent_pflow": False,
        "max_variants": 8,
        "sweep_max_workers": 0,
        "sweep_llm_top_n": 25,
        "sweep_full_table_threshold": 250,
        "contingency_max_count": 500,
        "benchmark_opflow": False,
        "boundary_initial_mw": 50.0,
        "boundary_max_mw": 2000.0,
        "boundary_tol_mw": 1.0,
        "boundary_max_probes": 24,
        "boundary_gen_q_frac": 0.4,
        "boundary_power_factor_default": "system_average",
        "added_gen_cost_strategy": "median_existing",
        "added_gen_dispatchable_default": False,
        "switched_load_mw": 100.0,
        "load_factor": None,
        "relief_tap_steps": [0.90, 0.95, 1.00, 1.05, 1.10],
        "relief_curtail_tol_mw": 1.0,
        "relief_max_solves": 2000,
        "reserve_max_solves": 4000,
    },
    "output": {
        "workdir": "./workdir",
        "logs_dir": "./logs",
        "save_journal": True,
        "journal_format": "json",
        "save_modified_files": True,
        "verbose": False,
    },
    "report": {
        "cost_min_top_k": 10,
        "near_optimal_abs_tol": 5.0,
        "network_summary_max_generators": 40,
    },
}

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExagoConfig:
    binary_dir: Path
    opflow_binary: Optional[Path]
    scopflow_binary: Optional[Path]
    tcopflow_binary: Optional[Path]
    sopflow_binary: Optional[Path]
    dcopflow_binary: Optional[Path]
    pflow_binary: Optional[Path]
    env_script: Optional[Path]
    timeout: int
    mpi_np: int = 1


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    api_key_env: str
    openai_base_url: Optional[str]
    ollama_host: str
    ollama_cloud_host: Optional[str]
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int
    default_mode: str
    base_case: Optional[Path]
    gic_file: Optional[Path]
    application: str
    search_mode: str = "standard"  # "standard" or "stress_test"
    ctgc_file: Optional[Path] = None  # Contingency file for SCOPFLOW
    pload_profile: Optional[Path] = None  # Active load profile for TCOPFLOW
    qload_profile: Optional[Path] = None  # Reactive load profile for TCOPFLOW
    wind_profile: Optional[Path] = None  # Wind generation profile for TCOPFLOW
    tcopflow_duration: float = 1.0  # Duration in hours for TCOPFLOW
    tcopflow_dT: float = 60.0  # Time-step in minutes for TCOPFLOW
    tcopflow_iscoupling: int = 1  # Ramp coupling (0=off, 1=on) for TCOPFLOW
    scenario_file: Optional[Path] = None  # Wind scenario CSV for SOPFLOW
    sopflow_solver: str = "IPOPT"  # Solver for SOPFLOW: IPOPT or EMPAR
    sopflow_iscoupling: int = 0  # Coupling between first/second stage (0=off, 1=on) for SOPFLOW
    benchmark_opflow: bool = False  # Run OPFLOW benchmark after PFLOW search
    concurrent_pflow: bool = False  # Enable concurrent explore/select for PFLOW
    max_variants: int = 8  # Max variants per explore action (range: 2-16)
    sweep_max_workers: int = 0  # Max concurrent solves for OPFLOW sweep; 0 = auto = min(cpu_count, 16)
    sweep_llm_top_n: int = 25  # Top-N feasible buses ranked by primary objective in summarized LLM view
    sweep_full_table_threshold: int = 250  # Candidate count above which LLM view switches to summary; 0 = always summarize
    contingency_max_count: int = 500  # Runaway guard: max enumerated contingencies per screen (N-2 blowup protection)
    # Boundary (hosting-capacity) search — per-candidate bisection on injection magnitude (OPFLOW only)
    boundary_initial_mw: float = 50.0  # Initial probe / exponential-bracketing start (MW)
    boundary_max_mw: float = 2000.0  # Bracketing cap; if still feasible at cap, report >= cap (MW)
    boundary_tol_mw: float = 1.0  # Bisection stops when the feasible/infeasible gap is below this (MW)
    boundary_max_probes: int = 24  # Max OPFLOW solves per candidate bisection
    boundary_gen_q_frac: float = 0.4  # Generator reactive band as fraction of ΔP (Qmin/Qmax = ∓frac·ΔP)
    boundary_power_factor_default: str = "system_average"  # Load PF: "system_average" | <0..1> | "unity"
    # Added-generator economics (C2) — dispatchable sweeps for min-cost siting
    added_gen_cost_strategy: str = "median_existing"  # "median_existing" (mid-merit) | "explicit"
    added_gen_dispatchable_default: bool = False  # Default mode for add_generator_at_bus sweeps (False = fixed injection)
    # Custom sweep metric (C3) — prompt-15 switched-in load block size
    switched_load_mw: float = 100.0  # MW load block switched in at a candidate bus for max_delta_v
    load_factor: Optional[float] = None  # Session-level load scaling factor (PFLOW only)
    # Contingency relief (C.7) — bounds for the per-failure relief-measure search
    relief_tap_steps: list = field(default_factory=lambda: [0.90, 0.95, 1.00, 1.05, 1.10])  # transformer tap ratios tried
    relief_curtail_tol_mw: float = 1.0  # Load-curtailment bisection stops below this MW gap
    relief_max_solves: int = 2000  # Total relief solves budget across all failed contingencies
    # Reserve minimization (C.8-revised Path A) — solve budget for the greedy de-commitment search
    reserve_max_solves: int = 4000  # Total OPF solves budget across the greedy reserve-minimization pass


@dataclass(frozen=True)
class OutputConfig:
    workdir: Path
    logs_dir: Path
    save_journal: bool
    journal_format: str
    save_modified_files: bool
    verbose: bool


@dataclass(frozen=True)
class ReportConfig:
    cost_min_top_k: int = 10        # Top-K cheapest buses shown in cost-min sweep ranking
    near_optimal_abs_tol: float = 5.0  # $/h threshold below which top candidates are equivalently optimal
    network_summary_max_generators: int = 40  # max generators listed in the
        # network summary sent to the LLM; larger networks truncate to the
        # top-N by Pmax. Bounds system-prompt size at scale.


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable configuration for AgentiGrid."""

    exago: ExagoConfig
    data: DataConfig
    llm: LLMConfig
    search: SearchConfig
    output: OutputConfig
    report: ReportConfig = field(default_factory=ReportConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Expand ${VAR} references in string values."""
    if not isinstance(value, str):
        return value
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_path(value: Any, root: Path) -> Optional[Path]:
    """Resolve a path relative to *root*, or return None."""
    if value is None:
        return None
    p = Path(_expand_env_vars(str(value)))
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _build_section(cls: type, raw: dict, root: Path, path_fields: set[str]) -> Any:
    """Instantiate a frozen dataclass section, resolving paths."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        val = raw.get(f.name)
        val = _expand_env_vars(val)
        if f.name in path_fields:
            val = _resolve_path(val, root)
        kwargs[f.name] = val
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration from a YAML file, merge with defaults and CLI overrides.

    Args:
        path: Path to the YAML config file. If *None*, only defaults are used.
        cli_overrides: Flat dict of CLI overrides. Keys use dot notation
            (e.g. ``"llm.backend"``).

    Returns:
        A frozen :class:`AppConfig` instance.
    """
    raw: dict[str, Any] = {}
    # Resolve relative paths against cwd (where the user invokes the tool),
    # not the config file's parent — the default YAML values like
    # "./applications" are written relative to the project root.
    config_root = Path.cwd()

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as fh:
                raw = yaml.safe_load(fh) or {}
            logger.debug("Loaded config from %s", path)
        else:
            logger.warning("Config file not found: %s — using defaults", path)

    # Merge: defaults <- file <- CLI overrides
    merged = _deep_merge(DEFAULTS, raw)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split(".")
            target = merged
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        logger.debug("Applied CLI overrides: %s", cli_overrides)

    # Build frozen dataclass sections
    exago_path_fields = {
        "binary_dir", "opflow_binary", "scopflow_binary", "tcopflow_binary",
        "sopflow_binary", "dcopflow_binary", "pflow_binary", "env_script",
    }
    exago = _build_section(ExagoConfig, merged["exago"], config_root, exago_path_fields)

    data = _build_section(DataConfig, merged["data"], config_root, {"data_dir"})

    llm = _build_section(LLMConfig, merged["llm"], config_root, set())

    search_path_fields = {"base_case", "gic_file", "ctgc_file", "pload_profile", "qload_profile", "wind_profile", "scenario_file"}
    search = _build_section(SearchConfig, merged["search"], config_root, search_path_fields)

    output = _build_section(OutputConfig, merged["output"], config_root, {"workdir", "logs_dir"})

    report = _build_section(ReportConfig, merged.get("report", {}), config_root, set())

    cfg = AppConfig(exago=exago, data=data, llm=llm, search=search, output=output, report=report)

    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    """Emit warnings for missing paths; does not raise."""
    if not cfg.exago.binary_dir.exists():
        logger.warning(
            "ExaGO binary_dir does not exist: %s (binaries may not be available)",
            cfg.exago.binary_dir,
        )
    if not cfg.data.data_dir.exists():
        logger.warning("Data directory does not exist: %s", cfg.data.data_dir)

    valid_backends = {"openai", "anthropic", "ollama", "ollama-cloud"}
    if cfg.llm.backend not in valid_backends:
        logger.warning("Unknown LLM backend '%s' (expected one of %s)", cfg.llm.backend, valid_backends)

    valid_apps = {"opflow", "scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"}
    if cfg.search.application not in valid_apps:
        logger.warning("Unknown application '%s' (expected one of %s)", cfg.search.application, valid_apps)

    valid_modes = {"accumulative", "fresh"}
    if cfg.search.default_mode not in valid_modes:
        logger.warning("Unknown mode '%s' (expected one of %s)", cfg.search.default_mode, valid_modes)

    valid_search_modes = {"standard", "stress_test"}
    if cfg.search.search_mode not in valid_search_modes:
        logger.warning(
            "Unknown search_mode '%s' (expected one of %s)",
            cfg.search.search_mode, valid_search_modes,
        )

    if cfg.search.application == "scopflow" and cfg.search.ctgc_file is None:
        logger.warning(
            "SCOPFLOW requires a contingency file (-ctgcfile). "
            "Set search.ctgc_file in config or use --ctgc on the CLI."
        )
    if cfg.search.ctgc_file is not None and not cfg.search.ctgc_file.exists():
        logger.warning(
            "Contingency file does not exist: %s", cfg.search.ctgc_file
        )

    if cfg.exago.mpi_np > 1 and cfg.search.application not in ("scopflow", "sopflow"):
        logger.warning(
            "mpi_np=%d is set but application '%s' only supports single-core "
            "IPOPT execution. mpi_np will be ignored. "
            "Only SCOPFLOW and SOPFLOW support multi-core via EMPAR.",
            cfg.exago.mpi_np, cfg.search.application,
        )

    if cfg.search.application == "tcopflow":
        if cfg.search.pload_profile is None:
            logger.warning(
                "TCOPFLOW requires an active load profile (-tcopflow_ploadprofile). "
                "Set search.pload_profile in config or use --pload-profile on the CLI."
            )
        if cfg.search.qload_profile is None:
            logger.warning(
                "TCOPFLOW requires a reactive load profile (-tcopflow_qloadprofile). "
                "Set search.qload_profile in config or use --qload-profile on the CLI."
            )
        if cfg.search.pload_profile is not None and not cfg.search.pload_profile.exists():
            logger.warning(
                "Active load profile does not exist: %s", cfg.search.pload_profile
            )
        if cfg.search.qload_profile is not None and not cfg.search.qload_profile.exists():
            logger.warning(
                "Reactive load profile does not exist: %s", cfg.search.qload_profile
            )
        if cfg.search.wind_profile is not None and not cfg.search.wind_profile.exists():
            logger.warning(
                "Wind generation profile does not exist: %s", cfg.search.wind_profile
            )

    if cfg.search.application == "sopflow":
        if cfg.search.scenario_file is None:
            logger.warning(
                "SOPFLOW requires a scenario file (-windgen). "
                "Set search.scenario_file in config or use --scenario-file on the CLI."
            )
        if cfg.search.scenario_file is not None and not cfg.search.scenario_file.exists():
            logger.warning(
                "Scenario file does not exist: %s", cfg.search.scenario_file
            )
        valid_solvers = {"IPOPT", "EMPAR"}
        if cfg.search.sopflow_solver not in valid_solvers:
            logger.warning(
                "Unknown SOPFLOW solver '%s' (expected one of %s)",
                cfg.search.sopflow_solver, valid_solvers,
            )
        if cfg.search.sopflow_solver == "EMPAR" and cfg.exago.mpi_np < 2:
            logger.warning(
                "SOPFLOW with EMPAR solver typically requires mpi_np >= 2 "
                "(currently mpi_np=%d).", cfg.exago.mpi_np,
            )

    if cfg.search.concurrent_pflow and cfg.search.application != "pflow":
        logger.warning(
            "concurrent_pflow is only applicable for the 'pflow' application; "
            "setting will be ignored for '%s'.", cfg.search.application,
        )
    if cfg.search.max_variants < 2:
        logger.warning(
            "max_variants=%d is below the minimum of 2; explore actions require at least 2 variants.",
            cfg.search.max_variants,
        )
    if cfg.search.max_variants > 16:
        logger.warning(
            "max_variants=%d is above 16; this may cause excessive prompt size and slow simulation.",
            cfg.search.max_variants,
        )
    if cfg.search.sweep_max_workers < 0:
        logger.warning(
            "sweep_max_workers=%d is negative; treating as auto (min(cpu_count, 16)).",
            cfg.search.sweep_max_workers,
        )
