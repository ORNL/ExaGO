"""gridkit.damping_in_verdict: damping in the verdict (true, default) or reported only (false).

With false the damping check is still computed and reported (value, machine, detail),
its status is "info", and it never decides a fault's result or the screen verdict.
"""

from __future__ import annotations

import math

import pytest
import yaml

from agentigrid.config import load_config
from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.agent_loop import _outcome_table
from agentigrid.gridkit_engine.journal import GridkitJournal
from agentigrid.gridkit_engine.schema_description import command_schema_text
from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.gridkit_engine.sweep_metrics import FaultOutcome, as_info, run_checks, screen_verdict
from agentigrid.gridkit_parsers.status_parser import RunStatus

CHECKS = ["angle_stability", "voltage_recovery", "damping", "final_voltage"]
INFO_TEXT = ("damping (informational only: the peak-decay estimate is not reliable on "
             "multi-mode responses; not part of the verdict)")
OK = RunStatus(solved=True, reason="solved")


class _Series:
    """Two machines, the second swinging with slowly growing amplitude (damping fails);
    voltages fine, angle change small."""

    def __init__(self):
        self.t = [i * 0.01 for i in range(1501)]
        swing = [0.0 if t < 1.1 else math.radians(2.0) * math.exp(0.02 * (t - 1.1))
                 * math.sin(2 * math.pi * 0.8 * (t - 1.1)) for t in self.t]
        self._delta = {"g1": [0.3] * len(self.t), "g2": [0.5 + s for s in swing]}
        self._vm = {1: [1.0] * len(self.t), 2: [0.99] * len(self.t)}

    def device_variable(self, name, classes=None):
        return self._delta

    def bus_variable(self, name):
        return self._vm


def _write(tmp_path, gridkit):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"gridkit": gridkit}))
    return path


def _settings(tmp_path, flag):
    return GridkitSettings.from_config(load_config(_write(tmp_path, {"damping_in_verdict": flag})))


def test_default_is_true():
    s = GridkitSettings.from_config(load_config(None))
    assert s.damping_in_verdict is True and s.info_checks == frozenset()


@pytest.mark.parametrize("flag,info", [(True, frozenset()), (False, frozenset({"damping"}))])
def test_config_carried(tmp_path, flag, info):
    s = _settings(tmp_path, flag)
    assert s.damping_in_verdict is flag and s.info_checks == info


def test_true_unchanged(tmp_path):
    checks = run_checks(_Series(), CHECKS, 1.1, _settings(tmp_path, True).info_checks)
    status = {c.name: c.status for c in checks}
    assert status == {"angle_stability": "pass", "voltage_recovery": "pass",
                      "damping": "fail", "final_voltage": "pass"}
    out = FaultOutcome(7, OK, checks)
    assert out.status == "fail"
    text = screen_verdict("Case", [out], "note")
    assert text.startswith("VERDICT: Case does NOT pass: 1 of 1 bus faults failed")
    assert "INFORMATIONAL" not in text
    assert "Damping (info)" not in _outcome_table([out])


def test_false_info_and_excluded(tmp_path):
    checks = run_checks(_Series(), CHECKS, 1.1, _settings(tmp_path, False).info_checks)
    damping = next(c for c in checks if c.name == "damping")
    assert damping.status == "info"
    assert damping.data["status_if_judged"] == "fail"
    assert damping.value is not None and damping.value < criteria.DAMPING_RATIO_MIN
    assert damping.data["machine"] in ("g1", "g2") and "lowest damping ratio" in damping.detail
    out = FaultOutcome(7, OK, checks)
    assert out.status == "pass"
    text = screen_verdict("Case", [out], "note")
    assert text.startswith("VERDICT: Case passes all 1 bus faults")
    assert "INFORMATIONAL, not part of the verdict: " + INFO_TEXT in text
    assert f"bus 7: damping - {damping.detail}" in text
    table = _outcome_table([out])
    assert "Damping (info)" in table and INFO_TEXT in table
    assert "pass" in table.splitlines()[3]


def test_false_unjudged_damping_does_not_block(tmp_path):
    nt = criteria.CheckResult("damping", "not_tested", None, 0.03, "run too short", "src")
    out = FaultOutcome(3, OK, [as_info(nt)])
    assert out.status == "pass"
    assert "NOT CONFIRMED" not in screen_verdict("Case", [out], "note")


def test_false_failing_other_check_still_fails():
    fail = criteria.CheckResult("voltage_recovery", "fail", 0.6, 0.7, "0.600 pu at bus 7", "src")
    info = criteria.CheckResult("damping", "info", 0.05, 0.03, "5%", "src")
    text = screen_verdict("Case", [FaultOutcome(7, OK, [fail, info])], "note")
    assert "does NOT pass" in text
    failed_table = text.split("FAILED")[1].split("INFORMATIONAL")[0]
    assert "voltage_recovery" in failed_table and "damping" not in failed_table


@pytest.mark.parametrize("flag", [True, False])
def test_schema_text(flag):
    info = frozenset() if flag else frozenset({"damping"})
    text = command_schema_text(info_checks=info)
    assert ("damping: damping ratio >= 3%" in text) is flag
    assert (INFO_TEXT in text) is not flag
    assert "angle_stability: machines stay in synchronism" in text


@pytest.mark.parametrize("flag", [True, False])
def test_journal_records_flag(tmp_path, flag):
    j = GridkitJournal("goal", tmp_path / "case.json", tmp_path)
    j.damping_in_verdict = flag
    assert j.to_dict()["damping_in_verdict"] is flag
