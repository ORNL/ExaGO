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
from dataclasses import dataclass
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
# when the spread between any two machine angles exceeds 180 degrees.
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
    status: str               # "pass", "fail" or "not_tested"
    value: Optional[float]
    limit: Optional[float]
    detail: str
    source: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _series_after(t: Sequence[float], x: Sequence[float], t0: float) -> list[tuple[float, float]]:
    return [(ti, xi) for ti, xi in zip(t, x) if ti >= t0]


def check_angle_stability(
    t: Sequence[float],
    deltas: Mapping[str, Sequence[float]],
    limit_deg: float = ANGLE_SPREAD_MAX_DEG,
) -> CheckResult:
    """Largest spread between machine rotor angles over the run."""
    src = "M-14B G.2.2 (angle limit: AgentiGrid default)"
    if len(deltas) < 2:
        return CheckResult("angle_stability", "not_tested", None, limit_deg,
                           "fewer than two machine angles recorded", src)
    names = list(deltas)
    worst, worst_t = -1.0, None
    for i, ti in enumerate(t):
        vals = [deltas[n][i] for n in names]
        spread = math.degrees(max(vals) - min(vals))
        if spread > worst:
            worst, worst_t = spread, ti
    ok = worst <= limit_deg
    return CheckResult(
        "angle_stability", "pass" if ok else "fail", worst, limit_deg,
        f"largest machine angle spread {worst:.1f} deg at t = {worst_t:g} s"
        + ("" if ok else " (loss of synchronism)"),
        src,
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
        f"lowest voltage from {after_s:g} s after clearing: {worst_v:.3f} pu "
        f"at bus {worst_bus}, t = {worst_t:g} s",
        src,
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
        f"closest to envelope: bus {bus} at t = {ti:g} s, {vi:.3f} pu vs minimum {vmin:.2f} pu",
        source,
    )


def check_final_voltage(
    vms: Mapping[object, Sequence[float]],
    vmin: float = FINAL_V_MIN_PU,
    vmax: float = FINAL_V_MAX_PU,
) -> CheckResult:
    """Bus voltages at the end of the run within the steady-state band."""
    src = "M-14B Att. I (band: AgentiGrid default)"
    if not vms:
        return CheckResult("final_voltage", "not_tested", None, None, "no bus voltages recorded", src)
    finals = {bus: v[-1] for bus, v in vms.items()}
    lo_bus = min(finals, key=finals.get)
    hi_bus = max(finals, key=finals.get)
    ok = finals[lo_bus] >= vmin and finals[hi_bus] <= vmax
    return CheckResult(
        "final_voltage", "pass" if ok else "fail", finals[lo_bus], vmin,
        f"end of run: lowest {finals[lo_bus]:.3f} pu (bus {lo_bus}), "
        f"highest {finals[hi_bus]:.3f} pu (bus {hi_bus}), band {vmin}-{vmax} pu",
        src,
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
) -> CheckResult:
    """Damping ratio of each machine's angle relative to the machine average.

    Measured after fault clearing; the worst machine decides.
    """
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
    detail = f"lowest damping ratio {100 * zeta:.1f}% (machine {nm})"
    if unmeasured:
        detail += f"; not measurable for {', '.join(unmeasured[:5])}"
    status = "pass" if ok else ("fail" if zeta < min_ratio else "not_tested")
    return CheckResult("damping", status, zeta, min_ratio, detail, src)


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
        f"without a fault: angles move {angle_move:.3f} deg (limit {angle_tol_deg}), "
        f"voltages move {v_move:.4f} pu (limit {v_tol_pu})",
        src,
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
