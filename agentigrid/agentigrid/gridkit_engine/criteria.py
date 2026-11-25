"""PJM pass/fail values for transient-stability studies, and the checks.

Every value names its source. Where PJM gives no number, the value is an
AgentiGrid default and says so; those may be changed, PJM values may not
(except where a stricter Transmission Owner criterion applies).

Sources:
  M-14B  PJM Manual 14B, Revision 59, effective 2026-04-22
         (Attachment G stability criteria, Attachment I TPL-001 events)

Study conventions for GridKit (decided 2026-09-29):
  - Peak load case.
  - Fault = bus short circuit to ground (GridKit BusFault, positive
    sequence, i.e. a balanced fault). Per M-14B Att. I footnote 2, a 3-phase
    study meeting the criteria is sufficient evidence for SLG as well.
  - No approximations: a PJM test that needs an element GridKit does not
    have is skipped, not imitated. Skipped for now: stuck breaker and zone-2
    clearing (element removal), high-speed reclosing (line trip/close),
    relay tripping by swings (no relay models), large-load and IBR
    ride-through (no such load/protection models), system frequency
    (no bus frequency output).

Time series conventions (GridKit output):
  delta  rotor angle, rad        omega  speed deviation, pu of nominal
  Vm     bus voltage, pu
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Study set-up values
# ---------------------------------------------------------------------------

STUDY_LOAD_LEVEL = "peak"             # decision 2026-09-29 (M-14B G.1 allows peak)
SIM_LENGTH_S = 15.0                   # M-14B G.2.2 damping test period "normally 10 to 15 s"
FAULT_START_S = 1.0                   # AgentiGrid default: 1 s of steady state first
NORMAL_CLEARING_CYCLES = 6.0          # M-14B G.9.4 (ComEd) "normal clearing = 6 cycles";
                                      #   PJM uses actual TO clearing times when known
FAULT_R_PU = 0.0                      # AgentiGrid default, as in GridKit IEEE39 case
FAULT_X_PU = 0.01                     # AgentiGrid default, as in GridKit IEEE39 case

# Clearing-time margin added to the nominal primary clearing time, cycles
# (M-14B G.2.2 Margins, 3-phase normally cleared). The stuck-breaker and
# zone-2 margins are not used: those tests remove elements, which GridKit
# cannot do yet (skipped, see module note).
MARGIN_PRIMARY_CYCLES = 0.25

# ---------------------------------------------------------------------------
# Pass/fail values
# ---------------------------------------------------------------------------

# Angle stability: M-14B G.2.2 requires transient stability (machines stay in
# synchronism) but gives no angle. AgentiGrid default: loss of synchronism
# when the rotor-angle difference between any two machines changes by more
# than 180 degrees from its pre-fault (initial) value. The absolute spread is
# not tested: in large cases most of it is already there in steady state.
ANGLE_SPREAD_MAX_DEG = 180.0

# Transient voltage recovery at BES buses (M-14B G.2.2):
# "shall recover to a minimum of 0.7 p.u. after 2.5 seconds" after clearing.
V_RECOVERY_MIN_PU = 0.70
V_RECOVERY_AFTER_S = 2.5

# ComEd zone dynamic voltage recovery envelope (M-14B G.9.4), time after
# clearing (s) -> minimum voltage (pu). Last point: steady-state minimum,
# "typically 92-95%"; 0.92 used here.
COMED_DVR_ENVELOPE = ((0.0, 0.70), (20.0 / 60.0, 0.80), (0.5, 0.90), (1.5, 0.92))

# Damping (M-14B G.2.2): positive damping of the machine-angle envelope with
# a 3% damping margin.
DAMPING_RATIO_MIN = 0.03
# AgentiGrid defaults for measuring damping:
DAMPING_MIN_AMPLITUDE_DEG = 1.0       # smaller swings count as settled (raised from 0.5 on
                                      #   2026-09-29: ACTIVSg200 swings of 0.64 deg from two
                                      #   beating modes gave false fits; mode separation
                                      #   (Prony/matrix pencil) deferred)
DAMPING_MIN_PEAKS = 3                 # fewer peaks: not enough to measure
DAMPING_METHODS = ("peak", "modal")   # AgentiGrid option gridkit.damping_method (default "peak")
DAMPING_MIN_RUN_S = 10.0              # M-14B G.2.2 test period 10-15 s; a 5 s run gave
                                      #   false failures on IEEE39 that vanish at 10-15 s

# Voltage after the event (M-14B Att. I: PJM Operations limits, by TO zone).
# AgentiGrid default band, same as the ExaGO side.
FINAL_V_MIN_PU = 0.90
FINAL_V_MAX_PU = 1.10

def clearing_time_s(nominal_cycles: float, margin_cycles: float = MARGIN_PRIMARY_CYCLES,
                    freq_hz: float = 60.0) -> float:
    """Fault duration in seconds: nominal clearing plus PJM margin."""
    return (nominal_cycles + margin_cycles) / freq_hz


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of one PJM check on one run."""

    name: str
    status: str               # "pass", "fail", "not_tested", or "info" (reported, not judged)
    value: Optional[float]
    limit: Optional[float]
    detail: str
    source: str
    data: dict = field(default_factory=dict)   # every number in *detail*, unrounded (journal)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def num(x: float, fmt: str, unit: str = "") -> str:
    """*x* rounded with *fmt* for reading, then its full value in brackets, so a
    rounded 1.100 that is really 1.10036 stays visible."""
    return f"{x:{fmt}}{unit} [{float(x)!r}{unit}]"


def _series_after(t: Sequence[float], x: Sequence[float], t0: float) -> list[tuple[float, float]]:
    return [(ti, xi) for ti, xi in zip(t, x) if ti >= t0]


def _unwrap_rad(x: Sequence[float]) -> list[float]:
    """Angle series in rad without 2*pi jumps (same idea as numpy.unwrap).

    GridKit integrates delta, so its output is already continuous; this only
    protects the check against a wrapped angle from another source.
    """
    if not x:
        return []
    out = [x[0]]
    for prev, cur in zip(x, x[1:]):
        step = cur - prev
        step -= 2.0 * math.pi * round(step / (2.0 * math.pi))
        out.append(out[-1] + step)
    return out


def check_angle_stability(
    t: Sequence[float],
    deltas: Mapping[str, Sequence[float]],
    limit_deg: float = ANGLE_SPREAD_MAX_DEG,
) -> CheckResult:
    """Largest change of the angle between any two machines over the run.

    The angle between two machines in steady state depends on the dispatch
    and on the size of the grid (about 108 deg in ACTIVSg10k), so it says
    nothing about stability. Only the change caused by the event is tested:

        d_i(t) = delta_i(t) - delta_i(0)        change of machine i
        change(t) = max_i d_i(t) - min_i d_i(t)  largest pairwise change

    A common drift of all machines (a frequency change) cancels out.
    The run must start from a steady state (see check_steady_state).
    The steady-state spread is reported for information only.
    """
    src = "M-14B G.2.2 (angle limit: AgentiGrid default)"
    # Never runs after a failed steady state: GridkitAgentLoop only screens from a start whose no-fault run passed check_steady_state (_Start.ok; run() ends with base_case_failed otherwise).
    if len(deltas) < 2:
        return CheckResult("angle_stability", "not_tested", None, limit_deg,
                           "fewer than two machine angles recorded", src)
    names = list(deltas)
    # Continuous angles and their pre-event (steady-state) values.
    angles = {n: _unwrap_rad(deltas[n]) for n in names}
    start = {n: angles[n][0] for n in names}
    initial_spread = math.degrees(max(start.values()) - min(start.values()))

    worst, worst_t, worst_pair = -1.0, None, (None, None)
    for i, ti in enumerate(t):
        change = {n: angles[n][i] - start[n] for n in names}
        hi = max(change, key=change.get)   # machine that advanced most
        lo = min(change, key=change.get)   # machine that fell back most
        spread = math.degrees(change[hi] - change[lo])
        if spread > worst:
            worst, worst_t, worst_pair = spread, ti, (hi, lo)
    ok = worst <= limit_deg
    return CheckResult(
        "angle_stability", "pass" if ok else "fail", worst, limit_deg,
        f"largest change of machine angle difference {num(worst, '.1f', ' deg')} "
        f"at t = {worst_t:g} s (machines {worst_pair[0]} and {worst_pair[1]}); "
        f"steady-state spread {num(initial_spread, '.1f', ' deg')} (information only)"
        + ("" if ok else " (loss of synchronism)"),
        src,
        {"spread_deg": worst, "t_s": worst_t, "machines": list(worst_pair),
         "initial_spread_deg": initial_spread},
    )


def check_voltage_recovery(
    t: Sequence[float],
    vms: Mapping[object, Sequence[float]],
    clear_time_s: float,
    min_pu: float = V_RECOVERY_MIN_PU,
    after_s: float = V_RECOVERY_AFTER_S,
) -> CheckResult:
    """Every bus at or above *min_pu* from *after_s* after clearing to the end."""
    src = "M-14B G.2.2 Acceptable Transient Voltage Recovery"
    t_check = clear_time_s + after_s
    if not vms:
        return CheckResult("voltage_recovery", "not_tested", None, min_pu,
                           "no bus voltages recorded", src)
    if not t or t[-1] < t_check:
        return CheckResult("voltage_recovery", "not_tested", None, min_pu,
                           f"run ends before t = {t_check:g} s", src)
    worst_v, worst_bus, worst_t = math.inf, None, None
    for bus, v in vms.items():
        for ti, vi in _series_after(t, v, t_check):
            if vi < worst_v:
                worst_v, worst_bus, worst_t = vi, bus, ti
    ok = worst_v >= min_pu
    return CheckResult(
        "voltage_recovery", "pass" if ok else "fail", worst_v, min_pu,
        f"lowest voltage from {after_s:g} s after clearing: {num(worst_v, '.3f', ' pu')} "
        f"at bus {worst_bus}, t = {worst_t:g} s",
        src,
        {"lowest_pu": worst_v, "bus": worst_bus, "t_s": worst_t},
    )


def check_voltage_envelope(
    t: Sequence[float],
    vms: Mapping[object, Sequence[float]],
    clear_time_s: float,
    envelope: Sequence[tuple[float, float]] = COMED_DVR_ENVELOPE,
    name: str = "comed_voltage_envelope",
    source: str = "M-14B G.9.4 (ComEd zone only)",
) -> CheckResult:
    """Voltage stays above a stepped recovery envelope after clearing."""
    last_step = envelope[-1][0]
    if not vms or not t or t[-1] < clear_time_s + last_step:
        return CheckResult(name, "not_tested", None, None,
                           "no voltages or run too short for the envelope", source)
    worst_margin, worst = math.inf, None
    for bus, v in vms.items():
        # The sample at the clearing time itself still holds the faulted value.
        for ti, vi in ((ti, vi) for ti, vi in zip(t, v) if ti > clear_time_s + 1e-9):
            dt = ti - clear_time_s
            vmin = max(vm for ts, vm in envelope if dt >= ts)
            if vi - vmin < worst_margin:
                worst_margin, worst = vi - vmin, (bus, ti, vi, vmin)
    ok = worst_margin >= 0
    bus, ti, vi, vmin = worst
    return CheckResult(
        name, "pass" if ok else "fail", vi, vmin,
        f"closest to envelope: bus {bus} at t = {ti:g} s, {num(vi, '.3f', ' pu')} vs minimum {vmin:.2f} pu",
        source,
        {"v_pu": vi, "bus": bus, "t_s": ti, "envelope_min_pu": vmin, "margin_pu": worst_margin},
    )


def check_final_voltage(
    vms: Mapping[object, Sequence[float]],
    vmin: float = FINAL_V_MIN_PU,
    vmax: float = FINAL_V_MAX_PU,
    t: Optional[Sequence[float]] = None,
    window_s: float = 0.0,
    margin_pu: float = 0.0,
    pre_fault_delta_pu: float = 0.0,
) -> CheckResult:
    """Bus voltages at the end of the run within the steady-state band.

    Default: the last sample of every bus, band vmin-vmax (unchanged behaviour).
    AgentiGrid options (both off by default):
      window_s > 0   the mean of each bus voltage over the last *window_s* seconds
                     (needs *t*) instead of the last sample, so an oscillation that is
                     still present does not decide the result by its phase;
      margin_pu > 0  the band widened to vmin - margin_pu ... vmax + margin_pu, e.g.
                     10 x the ACOPF tolerance, so a voltage held at a limit only within
                     the solver tolerance is not reported as a violation.
      pre_fault_delta_pu > 0
                     judged relative to the pre-fault value (first sample, as the angle
                     check): the fault must not create a violation, nor worsen a voltage
                     that was already at or beyond a limit before the fault by more than
                     this, i.e. each bus must end within
                     min(vmin - margin, V_pre - d) ... max(vmax + margin, V_pre + d).
                     A bus that the operating point (e.g. the AC OPF) holds at a limit is
                     then not reported because of tiny post-fault movements.
    """
    src = "M-14B Att. I (band: AgentiGrid default)"
    if not vms:
        return CheckResult("final_voltage", "not_tested", None, None, "no bus voltages recorded", src)
    use_window = window_s > 0 and t is not None and len(t) > 1
    if use_window:
        t_from = t[-1] - window_s
        k0 = next(i for i, ti in enumerate(t) if ti >= t_from - 1e-12)
        finals = {bus: sum(v[k0:]) / len(v[k0:]) for bus, v in vms.items()}
        where = f"mean of the last {window_s:g} s"
    else:
        finals = {bus: v[-1] for bus, v in vms.items()}
        where = "end of run"
    lo_lim, hi_lim = vmin - margin_pu, vmax + margin_pu
    lo_bus = min(finals, key=finals.get)
    hi_bus = max(finals, key=finals.get)
    band = f"band {vmin}-{vmax} pu" if margin_pu == 0 else \
        f"band {vmin}-{vmax} pu with margin {margin_pu:g} pu ({lo_lim:.6g}-{hi_lim:.6g} pu)"
    outside = [b for b, v in finals.items() if v < lo_lim or v > hi_lim]
    if pre_fault_delta_pu > 0:
        pre = {bus: v[0] for bus, v in vms.items()}
        bad = [b for b in outside
               if finals[b] > max(hi_lim, pre[b] + pre_fault_delta_pu)
               or finals[b] < min(lo_lim, pre[b] - pre_fault_delta_pu)]
        ok = not bad
        held = len(outside) - len(bad)
        band += (f"; relative to the pre-fault value (tolerance {pre_fault_delta_pu:g} pu): "
                 f"{len(bad)} bus(es) violate it, {held} outside the band only as before the fault")
        if bad:
            w = max(bad, key=lambda b: abs(finals[b] - pre[b]))
            band += f" (worst: bus {w}, {num(pre[w], '.4f')} -> {num(finals[w], '.4f')} pu)"
    else:
        ok = finals[lo_bus] >= lo_lim and finals[hi_bus] <= hi_lim
    data = {"lowest_pu": finals[lo_bus], "lowest_bus": lo_bus,
            "highest_pu": finals[hi_bus], "highest_bus": hi_bus, "vmin_pu": vmin, "vmax_pu": vmax}
    if use_window or margin_pu or pre_fault_delta_pu:
        data.update({"window_s": window_s if use_window else 0.0, "margin_pu": margin_pu,
                     "n_outside": len(outside)})
    if pre_fault_delta_pu > 0:
        data.update({"pre_fault_delta_pu": pre_fault_delta_pu, "n_violations": len(bad),
                     "n_outside_as_before_fault": held,
                     "max_abs_change_from_pre_fault_pu": max((abs(finals[b] - pre[b]) for b in finals), default=0.0)})
    return CheckResult(
        "final_voltage", "pass" if ok else "fail", finals[lo_bus], vmin,
        f"{where}: lowest {num(finals[lo_bus], '.3f', ' pu')} (bus {lo_bus}), "
        f"highest {num(finals[hi_bus], '.3f', ' pu')} (bus {hi_bus}), {band}",
        src, data,
    )


def damping_ratio(t: Sequence[float], x: Sequence[float]) -> tuple[Optional[float], int, float]:
    """Damping ratio of an oscillation from the decay of its peak envelope.

    Uses the size of successive peaks and valleys (half cycles) around the
    signal's final level, fits ln(size) against half-cycle number, and turns
    the decay per cycle (logarithmic decrement d) into zeta = d / sqrt(4 pi^2 + d^2).

    Returns (zeta or None if not measurable, number of peaks, largest swing).
    """
    n = len(x)
    if n < 5:
        return None, 0, 0.0
    tail = x[int(0.8 * n):]
    level = sum(tail) / len(tail)
    y = [xi - level for xi in x]
    peaks = [
        abs(y[i]) for i in range(1, n - 1)
        if (y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 0)
        or (y[i] < y[i - 1] and y[i] <= y[i + 1] and y[i] < 0)
    ]
    largest = max((abs(v) for v in y), default=0.0)
    if len(peaks) < 2:
        return None, len(peaks), largest
    k = list(range(len(peaks)))
    ln_a = [math.log(max(p, 1e-12)) for p in peaks]
    k_mean = sum(k) / len(k)
    a_mean = sum(ln_a) / len(ln_a)
    slope = sum((ki - k_mean) * (ai - a_mean) for ki, ai in zip(k, ln_a)) / sum(
        (ki - k_mean) ** 2 for ki in k
    )
    d = -2.0 * slope  # two half cycles per cycle
    return d / math.sqrt(4 * math.pi ** 2 + d ** 2), len(peaks), largest


def check_damping(
    t: Sequence[float],
    deltas: Mapping[str, Sequence[float]],
    clear_time_s: float,
    min_ratio: float = DAMPING_RATIO_MIN,
    min_amplitude_deg: float = DAMPING_MIN_AMPLITUDE_DEG,
    min_peaks: int = DAMPING_MIN_PEAKS,
    min_run_s: float = DAMPING_MIN_RUN_S,
    method: str = "peak",
) -> CheckResult:
    """Damping ratio of each machine's angle relative to the machine average.

    Measured after fault clearing; the worst machine decides.
    method "peak" (default): decay of the peak envelope (damping_ratio).
    method "modal" (AgentiGrid option): robust matrix pencil, see _check_damping_modal.
    """
    if method not in DAMPING_METHODS:
        raise ValueError(f"damping method '{method}' unknown; use one of {DAMPING_METHODS}")
    if method == "modal":
        return _check_damping_modal(t, deltas, clear_time_s, min_ratio, min_amplitude_deg, min_run_s)
    src = "M-14B G.2.2 Acceptable Damping (3% margin)"
    if len(deltas) < 2:
        return CheckResult("damping", "not_tested", None, min_ratio,
                           "fewer than two machine angles recorded", src)
    if not t or t[-1] < min_run_s:
        return CheckResult("damping", "not_tested", None, min_ratio,
                           f"run ends at {t[-1] if t else 0:g} s; damping needs at least "
                           f"{min_run_s:g} s", src)
    idx = [i for i, ti in enumerate(t) if ti >= clear_time_s]
    tt = [t[i] for i in idx]
    names = list(deltas)
    avg = [sum(deltas[nm][i] for nm in names) / len(names) for i in idx]
    worst: Optional[tuple[float, str]] = None
    unmeasured: list[str] = []
    for nm in names:
        rel = [math.degrees(deltas[nm][i] - a) for i, a in zip(idx, avg)]
        zeta, npk, largest = damping_ratio(tt, rel)
        if largest < min_amplitude_deg:
            continue  # settled: nothing to damp
        if zeta is None or npk < min_peaks:
            unmeasured.append(nm)
            continue
        if worst is None or zeta < worst[0]:
            worst = (zeta, nm)
    if worst is None:
        if unmeasured:
            return CheckResult("damping", "not_tested", None, min_ratio,
                               f"too few swings to measure for {', '.join(unmeasured[:5])}; "
                               "run longer", src)
        return CheckResult("damping", "pass", None, min_ratio,
                           f"all swings below {min_amplitude_deg} deg after clearing", src)
    zeta, nm = worst
    ok = zeta >= min_ratio and not unmeasured
    detail = f"lowest damping ratio {num(100 * zeta, '.1f', '%')} (machine {nm})"
    if unmeasured:
        detail += f"; not measurable for {', '.join(unmeasured[:5])}"
    status = "pass" if ok else ("fail" if zeta < min_ratio else "not_tested")
    return CheckResult("damping", status, zeta, min_ratio, detail, src,
                       {"ratio": zeta, "machine": nm, "unmeasured": unmeasured})


def _check_damping_modal(
    t: Sequence[float],
    deltas: Mapping[str, Sequence[float]],
    clear_time_s: float,
    min_ratio: float,
    min_amplitude_deg: float,
    min_run_s: float,
) -> CheckResult:
    """Damping by the robust matrix pencil (modal.damping_ratio_modal_robust).

    Same signal as the peak method: each machine's angle relative to the machine
    average, in degrees; machines whose largest swing after clearing is below
    *min_amplitude_deg* count as settled. For every other machine the matrix pencil
    is fitted in two windows (clearing + 1 s and + 2 s); a mode counts only if it is
    found in both (|df| <= 0.05 Hz, |dzeta| <= 1.5 points). Status:
      fail        a confirmed mode has a damping ratio below *min_ratio*
      not_tested  no confirmed mode below *min_ratio*, but for some machine a mode
                  below *min_ratio* was seen in one window only ("undetermined")
      pass        otherwise
    Modes with an amplitude below *min_amplitude_deg* are not considered.
    """
    src = "M-14B G.2.2 Acceptable Damping (3% margin); matrix pencil: AgentiGrid option"
    try:
        import numpy as np                  # only needed for this option (not a package requirement)
        from agentigrid.gridkit_engine import modal
    except ImportError:
        return CheckResult("damping", "not_tested", None, min_ratio,
                           "damping_method 'modal' needs numpy (pip install numpy)", src)
    if len(deltas) < 2:
        return CheckResult("damping", "not_tested", None, min_ratio,
                           "fewer than two machine angles recorded", src)
    if not t or t[-1] < min_run_s:
        return CheckResult("damping", "not_tested", None, min_ratio,
                           f"run ends at {t[-1] if t else 0:g} s; damping needs at least "
                           f"{min_run_s:g} s", src)
    tt = np.asarray(t, float)
    names = list(deltas)
    D = np.asarray([deltas[n] for n in names], float).T
    rel = np.degrees(D - D.mean(axis=1, keepdims=True))
    k = tt >= clear_time_s
    worst: Optional[tuple[float, str, float]] = None
    undetermined: list[str] = []
    measured = 0
    for j, nm in enumerate(names):
        seg = rel[k, j]
        if len(seg) < 5:
            continue
        level = seg[int(0.8 * len(seg)):].mean()
        if np.max(np.abs(seg - level)) < min_amplitude_deg:
            continue  # settled: nothing to damp
        measured += 1
        zeta, modes, status = modal.damping_ratio_modal_robust(
            tt, rel[:, j], clear_time_s, min_amplitude=min_amplitude_deg, threshold=min_ratio)
        if status == "undetermined":
            undetermined.append(nm)
        if zeta is not None and (worst is None or zeta < worst[0]):
            f = min(modes, key=lambda m: m[1])[0] if modes else float("nan")
            worst = (zeta, nm, f)
    data = {"method": "modal", "machines_measured": measured, "undetermined": undetermined}
    if worst is not None:
        data.update({"ratio": worst[0], "machine": worst[1], "frequency_hz": worst[2]})
    if worst is not None and worst[0] < min_ratio:
        return CheckResult("damping", "fail", worst[0], min_ratio,
                           f"lowest confirmed damping ratio {num(100 * worst[0], '.1f', '%')} "
                           f"(machine {worst[1]}, {worst[2]:.2f} Hz; matrix pencil)", src, data)
    if undetermined:
        low = "" if worst is None else \
            f"; lowest confirmed {num(100 * worst[0], '.1f', '%')} (machine {worst[1]})"
        return CheckResult("damping", "not_tested", None if worst is None else worst[0], min_ratio,
                           f"not determined for {', '.join(undetermined[:5])}: a mode below "
                           f"{100 * min_ratio:g}% seen in one fitting window only{low}; "
                           "the record is too short: repeat this bus with tmax_s = 30 and report "
                           "both results (PJM test period is 10-15 s)", src, data)
    if worst is None:
        return CheckResult("damping", "pass", None, min_ratio,
                           f"no confirmed mode of {min_amplitude_deg} deg or more below "
                           f"{100 * min_ratio:g}% ({measured} machines swing >= {min_amplitude_deg} deg; "
                           "matrix pencil)", src, data)
    return CheckResult("damping", "pass", worst[0], min_ratio,
                       f"lowest confirmed damping ratio {num(100 * worst[0], '.1f', '%')} "
                       f"(machine {worst[1]}, {worst[2]:.2f} Hz; matrix pencil)", src, data)


# Pre-fault steady state: the no-fault run of the case copy must stay still,
# otherwise swings after a fault cannot be told apart from drift of the case
# itself. AgentiGrid defaults.
STEADY_ANGLE_TOL_DEG = 0.1
STEADY_V_TOL_PU = 0.001


def check_steady_state(
    t: Sequence[float],
    deltas: Mapping[str, Sequence[float]],
    vms: Mapping[object, Sequence[float]],
    angle_tol_deg: float = STEADY_ANGLE_TOL_DEG,
    v_tol_pu: float = STEADY_V_TOL_PU,
) -> CheckResult:
    """No-fault run: machine angles (relative to their average) and bus
    voltages must not move by more than the tolerances."""
    src = "AgentiGrid base-case check (no PJM number)"
    if len(deltas) < 2 or not vms or not t:
        return CheckResult("steady_state", "not_tested", None, None,
                           "machine angles and bus voltages must both be recorded", src)
    names = list(deltas)
    n = len(t)
    avg = [sum(deltas[m][i] for m in names) / len(names) for i in range(n)]
    angle_move = max(
        math.degrees(max(deltas[m][i] - avg[i] for i in range(n))
                     - min(deltas[m][i] - avg[i] for i in range(n)))
        for m in names
    )
    v_move = max(max(v) - min(v) for v in vms.values())
    ok = angle_move <= angle_tol_deg and v_move <= v_tol_pu
    return CheckResult(
        "steady_state", "pass" if ok else "fail", angle_move, angle_tol_deg,
        f"without a fault: angles move {num(angle_move, '.3f', ' deg')} (limit {angle_tol_deg}), "
        f"voltages move {num(v_move, '.4f', ' pu')} (limit {v_tol_pu})",
        src,
        {"angle_move_deg": angle_move, "v_move_pu": v_move},
    )

# PJM tests the agent does not run, because GridKit lacks the element they
# need (no approximations). Every screen verdict lists them.
SKIPPED_TESTS = (
    ("NEW", "connecting the requested generator or load itself",
     "needs its dynamic model data and a consistent initial state; screens test the existing "
     "system at the POI"),
    ("D1", "SLG fault with stuck breaker", "needs element removal (line/generator trip)"),
    ("D2", "SLG fault with zone-2 clearing", "needs element removal (line/generator trip)"),
    ("D3", "high-speed reclosing", "needs line trip and re-close events"),
    ("D4", "NERC P2-P7 events", "needs element removal"),
    ("D5", "line maintenance outage before the fault", "needs element removal"),
    ("D6/C4", "relay tripping by power swings", "no relay models"),
    ("R-G2/R-G3", "wind / IBR ride-through", "no plant protection or trip models"),
    ("R-L1/R-L2", "large-load ride-through", "no power-electronic load model, no bus frequency"),
)
