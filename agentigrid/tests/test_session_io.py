"""Tests for session save/resume (Phase 2.4)."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from agentigrid.engine.session_io import save_session, load_session, SESSION_FORMAT_VERSION
from agentigrid.engine.journal import (
    JournalEntry, SearchJournal, ObjectiveEntry, ObjectiveRegistry,
)


@pytest.fixture
def sample_journal():
    journal = SearchJournal()
    journal.objective_registry.register(
        ObjectiveEntry(name="generation_cost", direction="minimize", priority="primary")
    )
    for i in range(3):
        entry = JournalEntry(
            iteration=i,
            description=f"iter {i}",
            commands=[{"action": "scale_all_loads", "factor": 1.0 + i * 0.05}] if i > 0 else [],
            objective_value=50000 - i * 1000,
            feasible=True,
            convergence_status="CONVERGED",
            violations_count=0,
            voltage_min=0.95,
            voltage_max=1.05,
            max_line_loading_pct=70.0,
            total_gen_mw=100,
            total_load_mw=95,
            llm_reasoning=f"reasoning {i}",
            mode="fresh",
            elapsed_seconds=1.0,
            tracked_metrics={"generation_cost": 50000 - i * 1000},
        )
        journal.add_entry(entry)
    return journal


class TestSessionSaveLoad:
    def test_save_creates_files(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_session"
        save_session(
            save_dir=save_dir,
            goal="test goal",
            application="opflow",
            base_case_path=Path("/data/case.m"),
            config_path="configs/default.yaml",
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            last_iteration=2,
        )
        assert (save_dir / "session.json").exists()

    def test_save_and_load_roundtrip(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_session"
        save_session(
            save_dir=save_dir,
            goal="minimize cost",
            application="opflow",
            base_case_path=Path("/data/case.m"),
            config_path="configs/default.yaml",
            journal=sample_journal,
            steering_history=[{"iteration": 1, "directive": "test", "mode": "augment"}],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            last_iteration=2,
        )

        loaded = load_session(save_dir)

        assert loaded["goal"] == "minimize cost"
        assert loaded["application"] == "opflow"
        assert loaded["last_iteration"] == 2
        assert loaded["total_prompt_tokens"] == 1000
        assert len(loaded["journal_entries"]) == 3
        assert loaded["journal_entries"][0].iteration == 0
        assert loaded["journal_entries"][2].objective_value == 48000
        assert len(loaded["steering_history"]) == 1

    def test_load_restores_objective_registry(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_session"
        save_session(
            save_dir=save_dir,
            goal="test",
            application="opflow",
            base_case_path=Path("/data/case.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=2,
        )

        loaded = load_session(save_dir)
        registry = loaded["objective_registry"]
        assert len(registry.objectives) == 1
        assert registry.objectives[0].name == "generation_cost"

    def test_load_restores_tracked_metrics(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_session"
        save_session(
            save_dir=save_dir,
            goal="test",
            application="opflow",
            base_case_path=Path("/data/case.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=2,
        )

        loaded = load_session(save_dir)
        assert loaded["journal_entries"][0].tracked_metrics is not None
        assert loaded["journal_entries"][0].tracked_metrics["generation_cost"] == 50000

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nonexistent")

    def test_format_version_in_file(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_session"
        save_session(
            save_dir=save_dir,
            goal="test",
            application="opflow",
            base_case_path=Path("/data/case.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=2,
        )
        data = json.loads((save_dir / "session.json").read_text())
        assert data["format_version"] == SESSION_FORMAT_VERSION

    def test_sopflow_scenario_override_roundtrip(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_sopflow_session"
        save_session(
            save_dir=save_dir,
            goal="find max wind penetration",
            application="sopflow",
            base_case_path=Path("/data/case9mod_gen3_wind.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=2,
            sopflow_num_scenarios=10,
            sopflow_scenario_override="/workdir/scenarios_iter_002/case9_10_scenarios.csv",
        )
        loaded = load_session(save_dir)
        assert loaded["sopflow_num_scenarios"] == 10
        assert loaded["sopflow_scenario_override"] == "/workdir/scenarios_iter_002/case9_10_scenarios.csv"

    def test_sopflow_no_override_roundtrip(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_sopflow_no_override"
        save_session(
            save_dir=save_dir,
            goal="test",
            application="sopflow",
            base_case_path=Path("/data/case9mod_gen3_wind.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=0,
        )
        loaded = load_session(save_dir)
        assert loaded["sopflow_scenario_override"] is None
        assert loaded["sopflow_num_scenarios"] == 0

    def test_tcopflow_profile_overrides_roundtrip(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_tcopflow_session"
        save_session(
            save_dir=save_dir,
            goal="find temporal loadability",
            application="tcopflow",
            base_case_path=Path("/data/case9mod.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=2,
            tcopflow_period_data=[{"period": 0, "total_load_mw": 265.0}],
            tcopflow_dT_min=15.0,
            tcopflow_duration_min=60.0,
            tcopflow_is_coupling=True,
            tcopflow_profile_overrides={
                "pload_profile": "/workdir/profiles_iter_002/case9_load_P.csv",
                "qload_profile": "/workdir/profiles_iter_002/case9_load_Q.csv",
            },
        )
        loaded = load_session(save_dir)
        assert loaded["tcopflow_profile_overrides"] is not None
        assert loaded["tcopflow_profile_overrides"]["pload_profile"] == "/workdir/profiles_iter_002/case9_load_P.csv"
        assert loaded["tcopflow_profile_overrides"]["qload_profile"] == "/workdir/profiles_iter_002/case9_load_Q.csv"

    def test_tcopflow_no_overrides_roundtrip(self, tmp_path, sample_journal):
        save_dir = tmp_path / "test_tcopflow_no_overrides"
        save_session(
            save_dir=save_dir,
            goal="test",
            application="tcopflow",
            base_case_path=Path("/data/case9mod.m"),
            config_path=None,
            journal=sample_journal,
            steering_history=[],
            active_steering_directives=[],
            current_network=None,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            last_iteration=0,
        )
        loaded = load_session(save_dir)
        assert loaded["tcopflow_profile_overrides"] is None
