"""gridkit.damping_method / final_voltage_window_s / final_voltage_margin_pu (all off by default).

The options change HOW the damping and final-voltage checks are computed, never which
checks exist. With the defaults everything must be exactly as before.
Selma Grebovic, Nov 2026 (IPST 2027 re-assessment of sweep1).
"""

from __future__ import annotations

import math

import pytest
import yaml

from agentigrid.config import load_config
from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.journal import GridkitJournal
from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.gridkit_engine.sweep_metrics import (DEFAULT_OPTIONS, CheckOptions, check_lines,
                                                     info_label, run_checks)

CLEAR = 1.0 + 6.25 / 60
T = [i * 0.01 for i in range(1501)]          # 15 s, 10 ms, as GridKit records


def _mode(t, zeta, f_hz, amp_deg, phase=0.0):
    """One decaying oscillation after clearing, in rad."""
    if t < CLEAR:
        return 0.0
    w = 2 * math.pi * f_hz
    sigma = zeta * w / math.sqrt(1 - zeta * zeta)
    return math.radians(amp_deg) * math.exp(-sigma * (t - CLEAR)) * math.sin(w * (t - CLEAR) + phase)


def _deltas(*modes_per_machine):
    """Machine g0 at rest, one machine per mode list."""
    out = {"g0": [0.3] * len(T)}
    for i, modes in enumerate(modes_per_machine, start=1):
        out[f"g{i}"] = [0.5 + sum(_mode(t, *m) for m in modes) for t in T]
    return out


class _Series:
    def __init__(self, deltas, vms):
        self.t, self._d, self._v = T, deltas, vms

    def device_variable(self, name, classes=None):
        return self._d

    def bus_variable(self, name):
        return self._v


def _write(tmp_path, gridkit):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"gridkit": gridkit}))
    return path


# ---------------- defaults: nothing changes ----------------

def test_defaults_are_off():
    s = GridkitSettings.from_config(load_config(None))
    assert s.check_options == CheckOptions() == DEFAULT_OPTIONS
    assert s.check_options.is_default()
    assert (s.damping_method, s.final_voltage_window_s, s.final_voltage_margin_pu) == ("peak", 0.0, 0.0)


def test_default_descriptions_unchanged():
    assert check_lines() == check_lines(frozenset(), DEFAULT_OPTIONS)
    assert "damping: damping ratio >= 3% (PJM M-14B G.2.2); run >= 10 s" in check_lines()
    assert "final_voltage: bus voltages 0.90-1.10 pu at end of run" in check_lines()
    assert info_label("damping") == info_label("damping", DEFAULT_OPTIONS)


def test_default_damping_is_peak_method():
    d = _deltas([(0.05, 0.8, 3.0)])
    a = criteria.check_damping(T, d, CLEAR)
    b = criteria.check_damping(T, d, CLEAR, method="peak")
    assert (a.status, a.value, a.detail) == (b.status, b.value, b.detail)
    assert "method" not in a.data


def test_default_final_voltage_unchanged():
    v = {1: [1.0] * len(T), 2: [1.05] * (len(T) - 1) + [1.1003]}
    a = criteria.check_final_voltage(v)
    b = criteria.check_final_voltage(v, t=T, window_s=0.0, margin_pu=0.0)
    assert (a.status, a.value, a.detail, a.data) == (b.status, b.value, b.detail, b.data)
    assert a.status == "fail" and a.detail.startswith("end of run:")


def test_journal_unchanged_by_default(tmp_path):
    j = GridkitJournal("goal", tmp_path / "case.json", tmp_path)
    assert "check_options" not in j.to_dict()
    j.check_options = CheckOptions("modal", 3.0, 1e-5).to_dict()
    assert j.to_dict()["check_options"]["damping_method"] == "modal"


# ---------------- config ----------------

def test_config_carried(tmp_path):
    cfg = load_config(_write(tmp_path, {"damping_method": "modal", "final_voltage_window_s": 3,
                                        "final_voltage_margin_pu": 1e-5}))
    o = GridkitSettings.from_config(cfg).check_options
    assert o == CheckOptions("modal", 3.0, 1e-5) and not o.is_default()


def test_config_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {"damping_method": "prony"}))


def test_unknown_method_in_check():
    with pytest.raises(ValueError):
        criteria.check_damping(T, _deltas([(0.05, 0.8, 3.0)]), CLEAR, method="prony")


# ---------------- modal damping ----------------

def test_modal_low_damping_fails():
    r = criteria.check_damping(T, _deltas([(0.02, 0.6, 5.0)]), CLEAR, method="modal")
    assert r.status == "fail"
    assert r.value == pytest.approx(0.02, abs=0.005)
    assert r.data["method"] == "modal" and r.data["frequency_hz"] == pytest.approx(0.6, abs=0.02)
    assert "matrix pencil" in r.detail


def test_modal_well_damped_passes():
    r = criteria.check_damping(T, _deltas([(0.08, 0.5, 4.0)]), CLEAR, method="modal")
    assert r.status == "pass"
    assert r.value == pytest.approx(0.08, abs=0.01)


def test_modal_two_close_modes_peak_method_misreads():
    """Two well-damped close modes (beat): the peak method reports low damping,
    the matrix pencil finds both modes at their true damping (as on the 10k case).
    The angle is taken relative to the machine average, so with two machines the
    swing seen is half of the one written here."""
    d = _deltas([(0.07, 0.40, 4.0), (0.07, 0.46, 4.0, math.pi)])
    peak = criteria.check_damping(T, d, CLEAR)
    modal = criteria.check_damping(T, d, CLEAR, method="modal")
    assert peak.status == "fail" and peak.value < 0.03          # false alarm of the peak method
    assert modal.status == "pass"
    assert modal.value == pytest.approx(0.07, abs=0.005)


def test_modal_small_swings_settled():
    r = criteria.check_damping(T, _deltas([(0.01, 0.7, 0.3)]), CLEAR, method="modal")
    assert r.status == "pass" and r.value is None
    assert r.data["machines_measured"] == 0


def test_modal_short_run_not_tested():
    short = [t for t in T if t <= 5.0]
    d = {k: v[:len(short)] for k, v in _deltas([(0.05, 0.8, 3.0)]).items()}
    assert criteria.check_damping(short, d, CLEAR, method="modal").status == "not_tested"


# ---------------- final voltage window and margin ----------------

def _osc_voltage(mean, amp):
    """Oscillation still present at the end; at t = 15 s it is at its maximum."""
    return [mean + (amp * math.cos(2 * math.pi * 0.7 * (t - 15.0)) if t > 2 else 0.0) for t in T]


def test_window_removes_oscillation_phase():
    v = {1: [1.0] * len(T), 2: _osc_voltage(1.0995, 0.001)}
    assert v[2][-1] > 1.10                                   # last sample happens to be above
    last = criteria.check_final_voltage(v, t=T)
    win = criteria.check_final_voltage(v, t=T, window_s=3.0)
    assert last.status == "fail" and win.status == "pass"
    assert win.detail.startswith("mean of the last 3 s") and win.data["window_s"] == 3.0


def test_margin_separates_solver_tolerance():
    v = {1: [1.0] * len(T), 2: [1.100004] * len(T)}          # at the limit within ACOPF tolerance
    assert criteria.check_final_voltage(v).status == "fail"
    m = criteria.check_final_voltage(v, margin_pu=1e-5)
    assert m.status == "pass" and m.data["n_outside"] == 0 and "margin" in m.detail
    v[3] = [1.1004] * len(T)                                  # a real exceedance stays a failure
    assert criteria.check_final_voltage(v, margin_pu=1e-5).status == "fail"


# ---------------- through the registry ----------------

def test_options_reach_the_checks():
    s = _Series(_deltas([(0.07, 0.40, 4.0), (0.07, 0.46, 4.0, math.pi)]),
                {1: [1.0] * len(T), 2: _osc_voltage(1.0995, 0.001)})
    names = ["damping", "final_voltage"]
    old = {c.name: c for c in run_checks(s, names, CLEAR)}
    new = {c.name: c for c in run_checks(s, names, CLEAR, frozenset(), CheckOptions("modal", 3.0, 0.0))}
    assert "method" not in old["damping"].data and new["damping"].data["method"] == "modal"
    assert old["final_voltage"].status == "fail" and new["final_voltage"].status == "pass"


def test_descriptions_change_only_when_on():
    on = check_lines(frozenset(), CheckOptions("modal", 3.0, 1e-5))
    assert any(line.startswith("damping: lowest damping ratio >= 3%") and "matrix pencil" in line for line in on)
    assert "final_voltage: bus voltages 0.90-1.10 pu with margin 1e-05 pu, mean of the last 3 s" in on
    assert [line.split(":")[0] for line in on] == [line.split(":")[0] for line in check_lines()]
