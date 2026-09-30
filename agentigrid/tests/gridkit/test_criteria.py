"""Tests for the PJM pass/fail values and checks on GridKit time series."""

from __future__ import annotations

import math

import pytest

from agentigrid.gridkit_engine import criteria as c

DT = 0.01
T15 = [round(i * DT, 10) for i in range(int(15 / DT) + 1)]


def _damped(zeta: float, amp_deg: float = 10.0, f_hz: float = 1.0, t=T15, start: float = 0.0):
    wn = 2 * math.pi * f_hz
    wd = wn * math.sqrt(1 - zeta ** 2)
    return [
        math.radians(amp_deg * math.exp(-zeta * wn * (ti - start)) * math.cos(wd * (ti - start)))
        if ti >= start else math.radians(amp_deg)
        for ti in t
    ]


def _flat(value: float, t=T15):
    return [value] * len(t)


class TestPJMValues:
    def test_values_from_manual(self):
        assert c.V_RECOVERY_MIN_PU == 0.70 and c.V_RECOVERY_AFTER_S == 2.5
        assert c.DAMPING_RATIO_MIN == 0.03
        assert c.MARGIN_PRIMARY_CYCLES == 0.25
        assert c.STUDY_LOAD_LEVEL == "peak"

    def test_clearing_time(self):
        assert c.clearing_time_s(6.0) == pytest.approx(6.25 / 60)


class TestDampingRatio:
    @pytest.mark.parametrize("zeta", [0.01, 0.03, 0.10])
    def test_measures_known_ratio(self, zeta):
        x = [math.degrees(v) for v in _damped(zeta)]
        measured, peaks, _ = c.damping_ratio(T15, x)
        assert measured == pytest.approx(zeta, abs=0.002)
        assert peaks > 10

    def test_check_uses_worst_machine(self):
        deltas = {"g1": _damped(0.10), "g2": [-v for v in _damped(0.02)]}
        r = c.check_damping(T15, deltas, clear_time_s=0.0)
        assert r.status == "fail"
        assert r.value < c.DAMPING_RATIO_MIN

    def test_well_damped_passes(self):
        deltas = {"g1": _damped(0.08), "g2": [-v for v in _damped(0.08)]}
        assert c.check_damping(T15, deltas, clear_time_s=0.0).status == "pass"

    def test_short_run_not_tested(self):
        t5 = T15[: int(5 / DT) + 1]
        deltas = {"g1": _damped(0.08, t=t5), "g2": _flat(0.0, t5)}
        assert c.check_damping(t5, deltas, clear_time_s=0.0).status == "not_tested"

    def test_no_swing_passes(self):
        deltas = {"g1": _flat(0.1), "g2": _flat(0.2)}
        assert c.check_damping(T15, deltas, clear_time_s=1.0).status == "pass"


class TestAngleStability:
    def test_within_limit(self):
        r = c.check_angle_stability(T15, {"a": _flat(0.0), "b": _flat(math.radians(100))})
        assert r.status == "pass" and r.value == pytest.approx(100)

    def test_loss_of_synchronism(self):
        slip = [math.radians(20 * ti) for ti in T15]  # 300 deg by 15 s
        r = c.check_angle_stability(T15, {"a": _flat(0.0), "b": slip})
        assert r.status == "fail" and "synchronism" in r.detail


class TestVoltage:
    @staticmethod
    def _dip(recover_to: float, clear: float = 1.1):
        return [0.3 if 1.0 <= ti <= clear else (recover_to if ti > clear else 1.0) for ti in T15]

    def test_recovery_pass_and_fail(self):
        assert c.check_voltage_recovery(T15, {1: self._dip(0.95)}, 1.1).status == "pass"
        r = c.check_voltage_recovery(T15, {1: self._dip(0.65)}, 1.1)
        assert r.status == "fail" and r.value == pytest.approx(0.65)

    def test_recovery_run_too_short(self):
        t = T15[:300]
        assert c.check_voltage_recovery(t, {1: [1.0] * 300}, 1.1).status == "not_tested"

    def test_comed_envelope_ignores_sample_at_clearing(self):
        assert c.check_voltage_envelope(T15, {1: self._dip(0.97)}, 1.1).status == "pass"
        assert c.check_voltage_envelope(T15, {1: self._dip(0.85)}, 1.1).status == "fail"

    def test_final_voltage(self):
        assert c.check_final_voltage({1: [1.0, 0.95]}).status == "pass"
        assert c.check_final_voltage({1: [1.0, 0.88]}).status == "fail"
