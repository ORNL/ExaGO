"""Session save/resume — serialize and deserialize search state."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agentigrid.engine.journal import (
    JournalEntry, ObjectiveEntry, ObjectiveRegistry, SearchJournal,
)
from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.matpower_writer import write_matpower
from agentigrid.parsers.matpower_parser import parse_matpower

logger = logging.getLogger("agentigrid.engine.session_io")

SESSION_FORMAT_VERSION = "1.1"


def save_session(
    save_dir: Path,
    goal: str,
    application: str,
    base_case_path: Path,
    config_path: Path | str | None,
    journal: SearchJournal,
    steering_history: list[dict],
    active_steering_directives: list[dict],
    current_network: MATNetwork | None,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    last_iteration: int,
    termination_reason: str = "paused",
    enforced_vmin: float | None = None,
    enforced_vmax: float | None = None,
    tcopflow_period_data: list[dict] | None = None,
    tcopflow_dT_min: float = 0.0,
    tcopflow_duration_min: float = 0.0,
    tcopflow_is_coupling: bool = True,
    sopflow_num_scenarios: int = 0,
    sopflow_scenario_override: str | None = None,
    tcopflow_profile_overrides: dict[str, str] | None = None,
    benchmark_result: dict | None = None,
    explore_cache_info: dict | None = None,
) -> Path:
    """Save a search session to disk for later resumption.

    Args:
        save_dir: Directory to save the session files into. Created if needed.
        goal: The search goal text.
        application: ExaGO application name.
        base_case_path: Path to the original base case .m file.
        config_path: Path to the config YAML used.
        journal: The SearchJournal with all entries so far.
        steering_history: List of steering directive dicts.
        active_steering_directives: Currently active steering directives.
        current_network: The modified network state (last iteration's network).
        total_prompt_tokens: Cumulative prompt tokens used.
        total_completion_tokens: Cumulative completion tokens used.
        last_iteration: The last completed iteration number.
        termination_reason: Why the session was saved (e.g., "paused", "user_stopped").
        enforced_vmin: Minimum voltage limit enforced in current network.
        enforced_vmax: Maximum voltage limit enforced in current network.

    Returns:
        Path to the saved session directory.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save current network state
    network_file = None
    if current_network is not None:
        network_file = "current_network.m"
        write_matpower(current_network, save_dir / network_file)

    # Build session data
    session_data = {
        "format_version": SESSION_FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(),
        "goal": goal,
        "application": application,
        "base_case_path": str(base_case_path),
        "config_path": str(config_path) if config_path else None,
        "last_iteration": last_iteration,
        "termination_reason": termination_reason,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "enforced_vmin": enforced_vmin,
        "enforced_vmax": enforced_vmax,
        "current_network_file": network_file,
        "steering_history": steering_history,
        "active_steering_directives": active_steering_directives,
        "tcopflow_period_data": tcopflow_period_data,
        "tcopflow_dT_min": tcopflow_dT_min,
        "tcopflow_duration_min": tcopflow_duration_min,
        "tcopflow_is_coupling": tcopflow_is_coupling,
        "sopflow_num_scenarios": sopflow_num_scenarios,
        "sopflow_scenario_override": sopflow_scenario_override,
        "tcopflow_profile_overrides": tcopflow_profile_overrides,
        "benchmark_result": benchmark_result,
        "explore_cache_info": explore_cache_info,
        "journal": {
            "entries": [asdict(e) for e in journal.entries],
            "objective_registry": journal.objective_registry.to_dict_list(),
            "preference_history": journal.objective_registry.history,
        },
    }

    session_path = save_dir / "session.json"
    session_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
    logger.info("Session saved to %s (%d entries)", save_dir, len(journal))
    return save_dir


def load_session(save_dir: Path) -> dict[str, Any]:
    """Load a saved session from disk.

    Args:
        save_dir: Directory containing session.json and optional current_network.m.

    Returns:
        Dict with all session data needed for resumption:
        - "goal", "application", "base_case_path", "config_path"
        - "journal_entries" (list of JournalEntry)
        - "objective_registry" (ObjectiveRegistry, populated)
        - "steering_history", "active_steering_directives"
        - "current_network" (MATNetwork or None)
        - "last_iteration", "total_prompt_tokens", "total_completion_tokens"
        - "enforced_vmin", "enforced_vmax"

    Raises:
        FileNotFoundError: If session.json doesn't exist.
        ValueError: If the format version is unsupported.
    """
    session_path = save_dir / "session.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Session file not found: {session_path}")

    raw = json.loads(session_path.read_text(encoding="utf-8"))

    version = raw.get("format_version", "0")
    if version not in ("1.0", SESSION_FORMAT_VERSION):
        raise ValueError(
            f"Unsupported session format version '{version}' "
            f"(expected '{SESSION_FORMAT_VERSION}')"
        )

    # Reconstruct journal entries
    journal_entries = []
    for entry_data in raw.get("journal", {}).get("entries", []):
        entry = JournalEntry(
            iteration=entry_data["iteration"],
            description=entry_data["description"],
            commands=entry_data.get("commands", []),
            objective_value=entry_data.get("objective_value"),
            feasible=entry_data.get("feasible", False),
            convergence_status=entry_data.get("convergence_status", "UNKNOWN"),
            violations_count=entry_data.get("violations_count", 0),
            voltage_min=entry_data.get("voltage_min", 0.0),
            voltage_max=entry_data.get("voltage_max", 0.0),
            max_line_loading_pct=entry_data.get("max_line_loading_pct", 0.0),
            total_gen_mw=entry_data.get("total_gen_mw", 0.0),
            total_load_mw=entry_data.get("total_load_mw", 0.0),
            llm_reasoning=entry_data.get("llm_reasoning", ""),
            mode=entry_data.get("mode", "fresh"),
            elapsed_seconds=entry_data.get("elapsed_seconds", 0.0),
            timestamp=entry_data.get("timestamp", ""),
            steering_directive=entry_data.get("steering_directive"),
            tracked_metrics=entry_data.get("tracked_metrics"),
            feasibility_detail=entry_data.get("feasibility_detail", ""),
            solver=entry_data.get("solver", ""),
            num_steps=entry_data.get("num_steps", 0),
            num_scenarios=entry_data.get("num_scenarios", 0),
            explored_variants=entry_data.get("explored_variants"),
            # Previously dropped on resume — restore so sweep/contingency/reserve
            # metadata survives a save/load round-trip (dataclass defaults keep
            # older sessions loadable).
            candidate_count=entry_data.get("candidate_count", 0),
            feasible_buses=entry_data.get("feasible_buses"),
            exago_command=entry_data.get("exago_command"),
            contingency_meta=entry_data.get("contingency_meta"),
            reserve_meta=entry_data.get("reserve_meta"),
        )
        journal_entries.append(entry)

    # Reconstruct objective registry
    registry = ObjectiveRegistry()
    for obj_data in raw.get("journal", {}).get("objective_registry", []):
        registry.register(ObjectiveEntry(
            name=obj_data["name"],
            direction=obj_data["direction"],
            threshold=obj_data.get("threshold"),
            priority=obj_data.get("priority", "primary"),
            introduced_at=obj_data.get("introduced_at", 0),
            source=obj_data.get("source", "initial"),
        ))
    # Restore history (register() creates its own history entries, so we
    # need to replace with the saved history)
    saved_history = raw.get("journal", {}).get("preference_history", [])
    registry._history = saved_history

    # Load current network if saved
    current_network = None
    network_file = raw.get("current_network_file")
    if network_file:
        network_path = save_dir / network_file
        if network_path.exists():
            current_network = parse_matpower(network_path)

    return {
        "goal": raw["goal"],
        "application": raw["application"],
        "base_case_path": Path(raw["base_case_path"]),
        "config_path": raw.get("config_path"),
        "journal_entries": journal_entries,
        "objective_registry": registry,
        "steering_history": raw.get("steering_history", []),
        "active_steering_directives": raw.get("active_steering_directives", []),
        "current_network": current_network,
        "last_iteration": raw.get("last_iteration", 0),
        "total_prompt_tokens": raw.get("total_prompt_tokens", 0),
        "total_completion_tokens": raw.get("total_completion_tokens", 0),
        "enforced_vmin": raw.get("enforced_vmin"),
        "enforced_vmax": raw.get("enforced_vmax"),
        "termination_reason": raw.get("termination_reason", "paused"),
        "tcopflow_period_data": raw.get("tcopflow_period_data"),
        "tcopflow_dT_min": raw.get("tcopflow_dT_min", 0.0),
        "tcopflow_duration_min": raw.get("tcopflow_duration_min", 0.0),
        "tcopflow_is_coupling": raw.get("tcopflow_is_coupling", True),
        "sopflow_num_scenarios": raw.get("sopflow_num_scenarios", 0),
        "sopflow_scenario_override": raw.get("sopflow_scenario_override"),
        "tcopflow_profile_overrides": raw.get("tcopflow_profile_overrides"),
        "benchmark_result": raw.get("benchmark_result"),
        "explore_cache_info": raw.get("explore_cache_info"),
    }
