"""Tests for the gridkit: section of the shared config and how the GridKit
code uses it (settings, solver .json, fault-study defaults, schema text)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from agentigrid.config import GridkitConfig, load_config
from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.commands import parse_fault_study
from agentigrid.gridkit_engine.executor import RunJob, solver_file_content
from agentigrid.gridkit_engine.schema_description import command_schema_text
from agentigrid.gridkit_engine.settings import GridkitSettings

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SOLVER_KEYS = {"rel_tol", "abs_tol", "dt_fixed", "max_steps", "max_order"}


def _write(tmp_path: Path, gridkit: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"gridkit": gridkit}))
    return path


class TestGridkitSection:
    def test_defaults_match_current_code(self):
        gk = load_config(None).gridkit
        assert gk.timeout == 600
        assert gk.study.tmax == criteria.SIM_LENGTH_S
        assert gk.fault.start == criteria.FAULT_START_S
        assert gk.study.dt_monitor == 0.01
        assert gk.binary_dir.name == "gridkit"

    @pytest.mark.parametrize("name", ["default_config.yaml.template", "local_config.yaml.template"])
    def test_templates_have_both_tools(self, name):
        raw = yaml.safe_load((CONFIGS / name).read_text())
        assert "exago" in raw and "gridkit" in raw
        gk = load_config(CONFIGS / name).gridkit
        assert gk.study == GridkitConfig().study
        assert gk.fault == GridkitConfig().fault

    def test_partial_section_keeps_other_defaults(self, tmp_path):
        gk = load_config(_write(tmp_path, {"study": {"tmax": 12.0}, "timeout": 90})).gridkit
        assert gk.study.tmax == 12.0
        assert gk.study.max_order == 5
        assert gk.timeout == 90
        assert gk.fault.start == 1.0

    def test_empty_subsection_keeps_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("gridkit:\n  study:\n  fault:\n")
        gk = load_config(path).gridkit
        assert gk.study == GridkitConfig().study
        assert gk.fault == GridkitConfig().fault

    def test_paths_resolved(self, tmp_path):
        gk = load_config(_write(tmp_path, {"dynamicsimulation_binary": "bin/DS"})).gridkit
        assert gk.binary_dir.is_absolute()
        assert gk.dynamicsimulation_binary == (Path.cwd() / "bin/DS").resolve()

    def test_unknown_key_warns(self, tmp_path, caplog):
        # GridKit itself ignores a misspelled solver key without a word.
        with caplog.at_level(logging.WARNING, logger="agentigrid.config"):
            load_config(_write(tmp_path, {"study": {"rel_tool": 1e-6}}))
        assert "rel_tool" in caplog.text

    def test_bad_max_order_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="agentigrid.config"):
            load_config(_write(tmp_path, {"study": {"max_order": 9}}))
        assert "max_order" in caplog.text


class TestSettingsFromConfig:
    def test_values_carried(self, tmp_path):
        cfg = load_config(_write(tmp_path, {
            "timeout": 90, "study": {"tmax": 12.0, "rel_tol": 1e-6}, "fault": {"start": 0.5},
        }))
        s = GridkitSettings.from_config(cfg)
        assert s.timeout_s == 90.0
        assert s.tmax_s == 12.0
        assert s.fault_start_s == 0.5
        assert s.solver_options["rel_tol"] == 1e-6
        assert set(s.solver_options) == SOLVER_KEYS
        assert s.study_defaults() == {"start_s": 0.5, "tmax_s": 12.0, "dt_monitor_s": 0.01}

    def test_binary_override(self, tmp_path):
        cfg = load_config(_write(tmp_path, {"dynamicsimulation_binary": "/opt/gk/DS"}))
        assert GridkitSettings.from_config(cfg).dynamic_simulation == Path("/opt/gk/DS")
        assert GridkitSettings.from_config(load_config(None)).dynamic_simulation.name == "DynamicSimulation"

    def test_keyword_override_wins(self):
        s = GridkitSettings.from_config(load_config(None), binary_dir=Path("/x"), max_iterations=3)
        assert s.binary_dir == Path("/x") and s.max_iterations == 3


class TestUse:
    def test_solver_file_has_config_keys(self, tmp_path):
        job = RunJob(tmp_path / "run", tmp_path / "case.json", 15.0, 0.01, element_id=3,
                     start_s=1.0, clear_s=1.1)
        content = solver_file_content(job, GridkitSettings().solver_options)
        assert SOLVER_KEYS <= set(content)
        assert content["tmax"] == 15.0 and len(content["events"]) == 2
        assert SOLVER_KEYS.isdisjoint(solver_file_content(job))

    def test_fault_study_defaults(self):
        s = parse_fault_study({"poi": 2}, {"start_s": 0.5, "tmax_s": 12.0, "dt_monitor_s": 0.02})
        assert (s.start_s, s.tmax_s, s.dt_monitor_s) == (0.5, 12.0, 0.02)
        s = parse_fault_study({"poi": 2, "tmax_s": 14.0}, {"tmax_s": 12.0})
        assert s.tmax_s == 14.0      # the LLM's value wins over the config default

    def test_schema_shows_config_defaults(self):
        text = command_schema_text(start_s=0.5, tmax_s=12.0)
        assert "start_s (0.5)" in text and "tmax_s (12.0)" in text
