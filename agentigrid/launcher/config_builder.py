"""Configuration builder for the AgentiGrid Launcher.

Bridges Streamlit GUI widget values to the AgentiGrid configuration system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def get_project_root() -> Path:
    """Return the AgentiGrid project root directory.

    The project root is the parent of the launcher/ directory.
    This function works regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


def scan_data_files(data_dir: Path | None = None) -> list[Path]:
    """Find all MATPOWER .m files in the data directory.

    Args:
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of .m file paths (absolute).
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.m"))


def scan_config_files(configs_dir: Path | None = None) -> list[Path]:
    """Find all YAML config files in the configs directory.

    Args:
        configs_dir: Path to configs directory. Defaults to <project_root>/configs.

    Returns:
        Sorted list of .yaml file paths (absolute).
    """
    if configs_dir is None:
        configs_dir = get_project_root() / "configs"
    if not configs_dir.is_dir():
        return []
    # Include both .yaml and .yml, exclude templates
    files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    # Filter out template files
    files = [f for f in files if not f.name.endswith(".template")]
    return sorted(files)


def scan_contingency_files(data_dir: Path | None = None) -> list[Path]:
    """Find all .cont contingency files in the data directory.

    Args:
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of .cont file paths (absolute).
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.cont"))


def scan_profile_files(data_dir: Path | None = None) -> list[Path]:
    """Find all load profile CSV files in the data directory.

    Looks for files matching *_load_P.csv and *_load_Q.csv.

    Args:
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of profile CSV file paths (absolute).
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*_load_*.csv"))


def scan_scenario_files(data_dir: Path | None = None) -> list[Path]:
    """Find all wind scenario CSV files in the data directory.

    Looks for files matching *_scenarios.csv and *_10_scenarios.csv.

    Args:
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of scenario CSV file paths (absolute).
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []
    singles = sorted(data_dir.glob("*_scenarios.csv"))
    tens = sorted(data_dir.glob("*_10_scenarios.csv"))
    return sorted(set(singles + tens))


_KNOWN_CASE_SUFFIXES = ("mod",)


def _case_stem_variants(case_stem: str) -> list[str]:
    """Generate variant case name stems for profile matching.

    Implements the layered fallback convention:
    1. Full case name stem (e.g., "case9mod")
    2. Strip known suffixes (e.g., "case9mod" → "case9")
    3. Strip trailing numeric segments (e.g., "case_ACTIVSg200" → "case_ACTIVSg200")

    Args:
        case_stem: The case file stem (filename without .m extension).

    Returns:
        List of candidate stems in priority order (no duplicates).
    """
    variants = [case_stem]
    for suffix in _KNOWN_CASE_SUFFIXES:
        if case_stem.endswith(suffix):
            stripped = case_stem[: -len(suffix)].rstrip("_")
            if stripped and stripped != case_stem:
                variants.append(stripped)
    return list(dict.fromkeys(variants))


def match_profiles_for_case(
    case_path: Path,
    data_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """Find load profile CSV files matching a base case name.

    Uses layered fallback matching:
    1. Try exact prefix: {case_stem}_*load_P.csv
    2. Strip known suffixes: e.g., case9mod → case9_*load_P.csv
    3. If no matches, return all available profiles as fallback.

    Args:
        case_path: Path to the .m base case file.
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Dict with keys "pload" and "qload", each containing a list of
        matching profile paths sorted by name. Empty lists if no matches.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return {"pload": [], "qload": []}

    all_p = sorted(data_dir.glob("*_load_P.csv"))
    all_q = sorted(data_dir.glob("*_load_Q.csv"))

    case_stem = case_path.stem
    variants = _case_stem_variants(case_stem)

    for stem in variants:
        p_matches = sorted(
            p for p in all_p if p.name.startswith(stem + "_") or p.name.startswith(stem)
        )
        q_matches = sorted(
            q for q in all_q if q.name.startswith(stem + "_") or q.name.startswith(stem)
        )
        if p_matches or q_matches:
            return {"pload": p_matches, "qload": q_matches}

    # Fallback: return all profiles
    return {"pload": all_p, "qload": all_q}


def match_scenarios_for_case(
    case_path: Path,
    data_dir: Path | None = None,
) -> list[Path]:
    """Find wind scenario CSV files matching a base case name.

    Uses layered fallback matching:
    1. Try exact prefix: {case_stem}_*scenarios.csv
    2. Strip known suffixes: e.g., case9mod_gen3_wind → case9_*scenarios.csv
    3. If no matches, return all available scenario files as fallback.

    Args:
        case_path: Path to the .m base case file.
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of matching scenario file paths.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []

    all_scenarios = scan_scenario_files(data_dir)
    case_stem = case_path.stem
    variants = _case_stem_variants(case_stem)

    for stem in variants:
        matches = sorted(
            s for s in all_scenarios
            if s.name.startswith(stem + "_") or s.name.startswith(stem)
        )
        if matches:
            return matches

    return all_scenarios


def load_example_goals() -> list[dict[str, str]]:
    """Load preset search goals from assets/example_goals.yaml.

    Returns:
        List of dicts with 'label' and 'goal' keys.
        Returns empty list if file not found.
    """
    goals_path = Path(__file__).resolve().parent / "assets" / "example_goals.yaml"
    if not goals_path.exists():
        return []
    with open(goals_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("goals", [])


# Default model names per backend
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:7b",
    "ollama-cloud": "qwen2.5:7b",
}

# Available backends
BACKENDS: list[str] = ["anthropic", "openai", "ollama", "ollama-cloud"]

# Available applications
APPLICATIONS: list[str] = ["opflow", "dcopflow", "scopflow", "tcopflow", "sopflow", "pflow"]

# Future applications (shown as disabled in the UI)
FUTURE_APPLICATIONS: list[str] = []

# Available search modes
MODES: list[str] = ["accumulative", "fresh"]

# Available search mode types
SEARCH_MODES: list[str] = ["standard", "stress_test"]


def build_config_overrides(
    base_case: str | Path,
    backend: str,
    model: str,
    temperature: float,
    application: str,
    default_mode: str,
    max_iterations: int,
    gic_file: str | Path | None = None,
    ctgc_file: str | Path | None = None,
    ollama_host: str | None = None,
    ollama_cloud_host: str | None = None,
    openai_base_url: str | None = None,
    verbose: bool = False,
    search_mode: str = "standard",
    mpi_np: int = 1,
    pload_profile: str | Path | None = None,
    qload_profile: str | Path | None = None,
    wind_profile: str | Path | None = None,
    tcopflow_duration: float = 1.0,
    tcopflow_dT: float = 60.0,
    tcopflow_iscoupling: int = 1,
    scenario_file: str | Path | None = None,
    sopflow_solver: str = "IPOPT",
    sopflow_iscoupling: int = 0,
    benchmark_opflow: bool = False,
    concurrent_pflow: bool = False,
    max_variants: int = 8,
    load_factor: float | None = None,
) -> dict[str, Any]:
    """Build a CLI-style overrides dict from GUI widget values.

    The returned dict uses dot-notation keys compatible with
    ``agentigrid.config.load_config(cli_overrides=...)``.

    Args:
        base_case: Path to the .m base case file.
        backend: LLM backend name (anthropic, openai, ollama, ollama-cloud).
        model: Model name/identifier.
        temperature: LLM temperature (0.0-1.0).
        application: ExaGO application name (e.g., "opflow").
        default_mode: Search mode ("accumulative" or "fresh").
        max_iterations: Maximum search iterations.
        gic_file: Optional path to .gic file.
        ollama_host: Optional Ollama host URL.
        ollama_cloud_host: Optional Ollama cloud host URL.
        openai_base_url: Optional OpenAI base URL.
        verbose: Enable verbose output.

    Returns:
        Dict with dot-notation keys for load_config(cli_overrides=...).
    """
    overrides: dict[str, Any] = {
        "search.base_case": str(base_case),
        "search.application": application,
        "search.default_mode": default_mode,
        "search.max_iterations": max_iterations,
        "search.search_mode": search_mode,
        "llm.backend": backend,
        "llm.model": model,
        "llm.temperature": temperature,
        "output.verbose": verbose,
        "exago.mpi_np": mpi_np,
    }

    if gic_file:
        overrides["search.gic_file"] = str(gic_file)

    if ctgc_file:
        overrides["search.ctgc_file"] = str(ctgc_file)

    if ollama_host:
        overrides["llm.ollama_host"] = ollama_host

    if ollama_cloud_host:
        overrides["llm.ollama_cloud_host"] = ollama_cloud_host

    if openai_base_url:
        overrides["llm.openai_base_url"] = openai_base_url

    if pload_profile:
        overrides["search.pload_profile"] = str(pload_profile)

    if qload_profile:
        overrides["search.qload_profile"] = str(qload_profile)

    if wind_profile:
        overrides["search.wind_profile"] = str(wind_profile)

    if tcopflow_duration != 1.0:
        overrides["search.tcopflow_duration"] = tcopflow_duration

    if tcopflow_dT != 60.0:
        overrides["search.tcopflow_dT"] = tcopflow_dT

    if tcopflow_iscoupling != 1:
        overrides["search.tcopflow_iscoupling"] = tcopflow_iscoupling

    if scenario_file:
        overrides["search.scenario_file"] = str(scenario_file)

    if sopflow_solver != "IPOPT":
        overrides["search.sopflow_solver"] = sopflow_solver

    if sopflow_iscoupling != 0:
        overrides["search.sopflow_iscoupling"] = sopflow_iscoupling

    if benchmark_opflow:
        overrides["search.benchmark_opflow"] = True

    if concurrent_pflow:
        overrides["search.concurrent_pflow"] = True

    if max_variants != 8:
        overrides["search.max_variants"] = max_variants

    if load_factor is not None:
        overrides["search.load_factor"] = load_factor

    return overrides


def get_default_config_path() -> Path:
    """Return the default configuration file path.

    Returns:
        Absolute path to configs/default_config.yaml.
    """
    return get_project_root() / "configs" / "default_config.yaml"
