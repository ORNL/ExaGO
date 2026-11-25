"""gridkit.damping_method / final_voltage_window_s / final_voltage_margin_pu / final_voltage_pre_fault_delta_pu (all off by default).

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


# ---------------- short-circuit strength as information (patch v2) ----------------

def _three_bus_case():
    """Source bus 3 (machine Xdpp 0.02 pu) feeding bus 1 through 0.01+j0.10 pu and bus 2 through j0.05 pu."""
    return {"buses": [{"number": n, "name": f"B{n}", "params": {"kv": 138.0}, "init": {"Vr": 1.0, "Vi": 0.0}}
                      for n in (1, 2, 3)],
            "devices": [
                {"class": "Branch", "ports": {"bus1": 3, "bus2": 1}, "params": {"R": 0.01, "X": 0.10, "B": 0.0}},
                {"class": "Branch", "ports": {"bus1": 3, "bus2": 2}, "params": {"R": 0.0, "X": 0.05, "B": 0.0}},
                {"class": "Genrou", "ports": {"bus": 3}, "params": {"mva": 100.0, "Ra": 0.0, "Xdpp": 0.02}},
                {"class": "BusFault", "ports": {"bus": 1}, "params": {}}]}


def test_ssc_off_by_default():
    s = GridkitSettings.from_config(load_config(None))
    assert s.short_circuit_info is False and s.check_options.short_circuit_info is False
    assert s.check_options.is_default()


def test_ssc_config_carried(tmp_path):
    cfg = load_config(_write(tmp_path, {"short_circuit_info": True, "weak_bus_percentile": 10}))
    o = GridkitSettings.from_config(cfg).check_options
    assert o.short_circuit_info is True and o.weak_bus_percentile == 10.0 and not o.is_default()
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {"weak_bus_percentile": 0}))


def test_ssc_three_bus_by_hand():
    pytest.importorskip("scipy")
    from agentigrid.gridkit_engine import ssc
    info = ssc.strength(_three_bus_case(), [1, 2], weak_percentile=50.0)
    # Zth(1) = 0.01 + j(0.10 + 0.02), Zth(2) = j(0.05 + 0.02), Zth(3) = j0.02
    assert info[1]["zth_pu"] == pytest.approx(abs(complex(0.01, 0.12)), rel=1e-9)
    assert info[1]["ssc_mva"] == pytest.approx(100 / abs(complex(0.01, 0.12)), rel=1e-9)
    assert info[1]["x_over_r"] == pytest.approx(12.0, rel=1e-9)
    assert info[2]["ssc_mva"] == pytest.approx(100 / 0.07, rel=1e-9)
    assert info[1]["n_same_kv"] == 3
    assert info[1]["percentile"] == 0.0 and info[1]["weak"] is True          # weakest of the three
    assert info[2]["percentile"] == pytest.approx(100 / 3) and info[2]["weak"] is True
    lines = ssc.report_lines(info)
    assert lines[0].startswith("SHORT-CIRCUIT STRENGTH (information only, not part of the verdict)")
    assert any(line.startswith("  bus 1: Ssc 830 MVA") for line in lines)
    assert "does not predict the outcome of the stability checks" in lines[-1]


def test_ssc_cache_reused():
    pytest.importorskip("scipy")
    from agentigrid.gridkit_engine import ssc
    cache = {}
    a = ssc.strength(_three_bus_case(), [1], cache=cache)
    assert 138.0 in cache and len(cache[138.0]) == 3
    b = ssc.strength(_three_bus_case(), [1], cache=cache)
    assert a == b


def test_ssc_missing_scipy_reported(monkeypatch):
    from agentigrid.gridkit_engine import ssc

    def no_scipy(case):
        raise ImportError("no scipy", name="scipy")
    monkeypatch.setattr(ssc, "_ybus_lu", no_scipy)
    info = ssc.strength(_three_bus_case(), [1])
    assert "error" in info and "scipy" in info["error"]
    assert "not available" in "\n".join(ssc.report_lines(info))


def test_ssc_zero_impedance_branch_reported_not_raised():
    pytest.importorskip("scipy")
    from agentigrid.gridkit_engine import ssc
    case = _three_bus_case()
    case["devices"][1]["params"].update(R=0.0, X=0.0)                # bus 3 - bus 2: zero impedance
    cache = {}
    info = ssc.strength(case, [1, 2], cache=cache)
    assert set(info) == {"error"}
    assert info["error"].startswith("not available (ZeroDivisionError")
    assert cache == {}
    lines = ssc.report_lines(info)
    assert lines[1] == f"  {info['error']}"


def test_ssc_unexpected_error_reported(monkeypatch):
    from agentigrid.gridkit_engine import ssc

    def singular(case):
        raise RuntimeError("Factor is exactly singular")
    monkeypatch.setattr(ssc, "_ybus_lu", singular)
    info = ssc.strength(_three_bus_case(), [1])
    assert info == {"error": "not available (RuntimeError: Factor is exactly singular)"}


def test_modal_undetermined_asks_for_30_s(monkeypatch):
    """Undetermined at 15 s: the message tells the LLM to repeat the bus with tmax_s = 30."""
    from agentigrid.gridkit_engine import modal
    monkeypatch.setattr(modal, "damping_ratio_modal_robust",
                        lambda *a, **k: (0.045, [(0.72, 0.045, 2.0)], "undetermined"))
    r = criteria.check_damping(T, _deltas([(0.05, 0.7, 4.0)]), CLEAR, method="modal")
    assert r.status == "not_tested"
    assert "repeat this bus with tmax_s = 30" in r.detail and "report both results" in r.detail
    assert r.data["undetermined"]


def test_modal_repeated_time_stamps(monkeypatch):
    """Duplicated rows (event times written twice, or every row twice) give the same
    finite result as the clean series instead of a zero median step (OverflowError)."""
    pytest.importorskip("numpy")
    from agentigrid.gridkit_engine import modal
    x = [math.degrees(_mode(t, 0.05, 0.7, 4.0)) for t in T]
    ref, _ = modal.damping_ratio_modal(T, x, 1.0, t_ref=CLEAR)
    assert ref == pytest.approx(0.05, abs=0.005)

    i_f, i_c = T.index(1.0), min(range(len(T)), key=lambda i: abs(T[i] - CLEAR))
    t_ev = T[:i_f] + [T[i_f]] + T[i_f:i_c] + [T[i_c]] + T[i_c:]       # fault and clearing rows twice
    x_ev = x[:i_f] + [x[i_f]] + x[i_f:i_c] + [x[i_c]] + x[i_c:]
    t_all = [t for t in T for _ in (0, 1)]                              # every row twice
    x_all = [v for v in x for _ in (0, 1)]
    for tt, xx in ((t_ev, x_ev), (t_all, x_all)):
        z, modes = modal.damping_ratio_modal(tt, xx, 1.0, t_ref=CLEAR)
        assert z == pytest.approx(ref, rel=1e-9) and modes

    assert modal.damping_ratio_modal([1.0] * 50, list(range(50)), 0.0) == (None, [])


def test_modal_one_b_mode_confirms_only_one_a_mode(monkeypatch):
    """Two window-a modes near one window-b mode: only the closer one is confirmed,
    the other stays unconfirmed (and, below the threshold, makes the result undetermined)."""
    from agentigrid.gridkit_engine import modal
    a = [(0.70, 0.050, 5.0), (0.72, 0.020, 4.0)]
    b = [(0.705, 0.045, 5.0)]
    fits = iter([(0.02, a), (0.045, b)] * 2)
    monkeypatch.setattr(modal, "damping_ratio_modal", lambda *args, **kw: next(fits))
    kw = dict(f_tol=0.05, zeta_tol=0.03)

    zeta, confirmed, status = modal.damping_ratio_modal_robust(T, T, CLEAR, **kw)
    assert len(confirmed) == 1
    assert confirmed[0][0] == pytest.approx(0.7025) and confirmed[0][1] == pytest.approx(0.0475)
    assert zeta == pytest.approx(0.0475) and status == "ok"

    zeta, confirmed, status = modal.damping_ratio_modal_robust(T, T, CLEAR, threshold=0.03, **kw)
    assert len(confirmed) == 1 and status == "undetermined"          # 0.72 Hz, 2 % seen in window a only


def test_journal_row_gets_ssc_only_when_on():
    loop = pytest.importorskip("agentigrid.gridkit_engine.agent_loop")
    row = {"bus": 1, "status": "pass"}
    assert loop._with_ssc(row, None, 1) is row                       # option off: row unchanged
    info = {1: {"ssc_mva": 830.5}}
    assert loop._with_ssc(row, info, 1)["short_circuit"] == {"ssc_mva": 830.5}
    assert "error" in loop._with_ssc(row, {"error": "not available"}, 1)["short_circuit"]


# ---------------- final voltage relative to the pre-fault value (patch v3) ----------------

def _held(v0, v_end):
    """Bus at v0 before the fault, a dip during the fault, ending at v_end."""
    return [v0 if t < 1.0 else (0.5 if t < CLEAR else v_end) for t in T]


def test_pre_fault_off_by_default():
    s = GridkitSettings.from_config(load_config(None))
    assert s.final_voltage_pre_fault_delta_pu == 0.0
    assert s.check_options.final_voltage_pre_fault_delta_pu == 0.0 and s.check_options.is_default()


def test_pre_fault_bus_held_at_limit_passes():
    # held at 1.10 by the OPF, ends 5e-5 pu higher: a "violation" only because of the operating point
    v = {1: [1.0] * len(T), 2: _held(1.10, 1.10005)}
    assert criteria.check_final_voltage(v, t=T, window_s=3.0, margin_pu=1e-5).status == "fail"
    r = criteria.check_final_voltage(v, t=T, window_s=3.0, margin_pu=1e-5, pre_fault_delta_pu=1e-3)
    assert r.status == "pass"
    assert r.data["n_outside"] == 1 and r.data["n_violations"] == 0 and r.data["n_outside_as_before_fault"] == 1
    assert "relative to the pre-fault value" in r.detail


def test_pre_fault_new_violation_fails():
    v = {1: [1.0] * len(T), 2: _held(1.05, 1.12)}            # inside before, outside after
    r = criteria.check_final_voltage(v, t=T, window_s=3.0, margin_pu=1e-5, pre_fault_delta_pu=1e-3)
    assert r.status == "fail" and r.data["n_violations"] == 1 and "worst: bus 2" in r.detail


def test_pre_fault_worsened_beyond_delta_fails():
    v = {1: [1.0] * len(T), 2: _held(1.10, 1.103)}            # at the limit, then 3e-3 pu higher
    assert criteria.check_final_voltage(v, t=T, window_s=3.0, pre_fault_delta_pu=1e-3).status == "fail"
    assert criteria.check_final_voltage(v, t=T, window_s=3.0, pre_fault_delta_pu=5e-3).status == "pass"


def test_pre_fault_lower_limit_symmetric():
    v = {1: [1.0] * len(T), 2: _held(0.90, 0.8995), 3: _held(0.95, 0.89)}
    r = criteria.check_final_voltage(v, t=T, window_s=3.0, pre_fault_delta_pu=1e-3)
    assert r.status == "fail" and r.data["n_violations"] == 1 and r.data["n_outside_as_before_fault"] == 1
    del v[3]
    assert criteria.check_final_voltage(v, t=T, window_s=3.0, pre_fault_delta_pu=1e-3).status == "pass"


def test_pre_fault_config_and_description(tmp_path):
    cfg = load_config(_write(tmp_path, {"final_voltage_window_s": 3.0, "final_voltage_margin_pu": 1e-5,
                                        "final_voltage_pre_fault_delta_pu": 1e-3}))
    o = GridkitSettings.from_config(cfg).check_options
    assert o.final_voltage_pre_fault_delta_pu == 1e-3 and not o.is_default()
    assert o.to_dict()["final_voltage_pre_fault_delta_pu"] == 1e-3
    line = [x for x in check_lines(frozenset(), o) if x.startswith("final_voltage:")][0]
    assert "relative to the pre-fault value" in line and "0.001 pu" in line
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {"final_voltage_pre_fault_delta_pu": -1e-3}))


def test_pre_fault_through_the_registry():
    s = _Series(_deltas([(0.07, 0.40, 4.0)]), {1: [1.0] * len(T), 2: _held(1.10, 1.10005)})
    v2 = {c.name: c for c in run_checks(s, ["final_voltage"], CLEAR, frozenset(), CheckOptions("peak", 3.0, 1e-5))}
    v3 = {c.name: c for c in run_checks(s, ["final_voltage"], CLEAR, frozenset(),
                                        CheckOptions("peak", 3.0, 1e-5, final_voltage_pre_fault_delta_pu=1e-3))}
    assert v2["final_voltage"].status == "fail" and v3["final_voltage"].status == "pass"
