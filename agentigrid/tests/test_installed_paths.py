"""Test resolution of installed configuration and cases."""

from pathlib import Path

from agentigrid.cli import build_parser
from agentigrid.config import load_config
from launcher.config_builder import (
    get_data_dir,
    get_default_config_path,
    match_profiles_for_case,
    match_scenarios_for_case,
    scan_data_files,
)


def test_installed_defaults_and_case_discovery(tmp_path, monkeypatch):
    config_dir = tmp_path / "share/exago/agentigrid/configs"
    config_dir.mkdir(parents=True)
    data_dir = tmp_path / "share/exago/datafiles"
    case_dir = data_dir / "case9"
    case_dir.mkdir(parents=True)
    case = case_dir / "case9mod_gen3_wind.m"
    case.touch()
    for name in ("load_P.csv", "load_Q.csv", "10_scenarios_9bus.csv"):
        (case_dir / name).touch()
    other = data_dir / "case_other"
    other.mkdir()
    (other / "load_P.csv").touch()
    (other / "scenarios_other.csv").touch()
    config = config_dir / "default_config.yaml"
    config.write_text(
        f"exago:\n  binary_dir: '{tmp_path / 'bin'}'\n"
        f"data:\n  data_dir: '{data_dir}'\n"
        'output:\n  workdir: "${EXAGO_STATE_DIR}/workdir"\n'
    )
    monkeypatch.setenv("AGENTIGRID_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("EXAGO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    assert Path(build_parser().parse_args([]).config) == config
    assert get_default_config_path() == config
    assert get_data_dir() == data_dir
    assert scan_data_files() == [case]
    assert match_profiles_for_case(case) == {
        "pload": [case_dir / "load_P.csv"], "qload": [case_dir / "load_Q.csv"],
    }
    assert match_scenarios_for_case(case) == [case_dir / "10_scenarios_9bus.csv"]
    cfg = load_config(config)
    assert cfg.exago.binary_dir == tmp_path / "bin"
    assert cfg.output.workdir == tmp_path / "state/workdir"


def test_flat_examples_do_not_mix_other_cases(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    for name in ("case9mod.m", "case9_load_P.csv", "case200_load_P.csv",
                 "case9_scenarios.csv", "10_scenarios_ACTIVSg200.csv",
                 "10_scenarios_ACTIVSg2000.csv"):
        (examples / name).touch()
    profiles = match_profiles_for_case(examples / "case9mod.m", tmp_path)
    assert profiles["pload"] == [examples / "case9_load_P.csv"]
    assert match_scenarios_for_case(examples / "case9mod.m", tmp_path) == [
        examples / "case9_scenarios.csv",
    ]
    assert match_scenarios_for_case(examples / "case_ACTIVSg200.m", tmp_path) == [
        examples / "10_scenarios_ACTIVSg200.csv",
    ]
