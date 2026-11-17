"""Comparison of HIL readings with ExaGO, bus by bus, and the verdict.

The verdict is decided here, not by the LLM. Limits (``hil.tol_pct``,
``hil.tol_deg``) cover the known difference between ExaGO's pure circuit and the
HIL model, which splits lines with TLM couplings (study plan, step 3b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agentigrid.hil_engine.connection import BusReading
from agentigrid.hil_engine.from_exago import ExagoState
from agentigrid.hil_engine.network import HilNetwork


@dataclass
class Comparison:
    name: str                    # e.g. "radial start" / "tie 12-22 closed"
    reference: str               # ExaGO label
    max_pct: float
    max_pct_bus: int
    max_deg: float
    max_deg_bus: int
    passed: bool
    rows: list[dict] = field(default_factory=list)

    def line(self, tol_pct: float, tol_deg: float) -> str:
        return (f"{self.name} vs {self.reference}: largest {self.max_pct:+.4f} % at bus {self.max_pct_bus}, "
                f"{self.max_deg:+.4f} deg at bus {self.max_deg_bus} (limits {tol_pct} %, {tol_deg} deg): "
                f"{'PASS' if self.passed else 'FAIL'}")


def compare(name: str, reading: BusReading, ref: ExagoState, net: HilNetwork,
            tol_pct: float, tol_deg: float) -> Comparison:
    rows = []
    for b in net.buses:
        v, d = reading.volts[b], reading.degrees[b]
        rv, rd = ref.volts[b], ref.degrees[b]
        rows.append({"bus": b, "hil_V": v, "exago_V": rv, "err_pct": 100 * (v - rv) / rv,
                     "hil_deg": d, "exago_deg": rd, "err_deg": d - rd})
    wp = max(rows, key=lambda r: abs(r["err_pct"]))
    wd = max(rows, key=lambda r: abs(r["err_deg"]))
    passed = all(abs(r["err_pct"]) <= tol_pct and abs(r["err_deg"]) <= tol_deg for r in rows)
    return Comparison(name, ref.label, wp["err_pct"], wp["bus"], wd["err_deg"], wd["bus"], passed, rows)


@dataclass
class StudyResult:
    verdict: str
    text: str                                    # shown to the LLM
    passed: Optional[bool]                       # None: no comparison asked
    comparisons: list[Comparison] = field(default_factory=list)
    radial: Optional[BusReading] = None
    closed: Optional[BusReading] = None


def _changes(net: HilNetwork, before: BusReading, after: BusReading, n: int = 6) -> list[dict]:
    rows = [{"bus": b, "dV": after.volts[b] - before.volts[b],
             "ddeg": after.degrees[b] - before.degrees[b]} for b in net.buses]
    return sorted(rows, key=lambda r: abs(r["dV"]), reverse=True)[:n]


def _table(net: HilNetwork, radial: BusReading, closed: BusReading,
           ex_radial: Optional[ExagoState], ex_closed: Optional[ExagoState]) -> str:
    head = f"{'Bus':>4} | {'HIL radial V':>12} {'HIL tie V':>10} {'change V':>9} | {'HIL tie deg':>11}"
    if ex_closed:
        head += f" | {'ExaGO tie V':>11} {'diff %':>8} {'ExaGO deg':>9} {'diff deg':>9}"
    if ex_radial:
        head += f" | {'ExaGO rad V':>11} {'diff %':>8}"
    lines = [head, "-" * len(head)]
    for b in net.buses:
        rv, cv = radial.volts[b], closed.volts[b]
        s = f"{b:>4} | {rv:>12.2f} {cv:>10.2f} {cv - rv:>+9.2f} | {closed.degrees[b]:>11.4f}"
        if ex_closed:
            ev, ed = ex_closed.volts[b], ex_closed.degrees[b]
            s += f" | {ev:>11.2f} {100 * (cv - ev) / ev:>+8.4f} {ed:>9.4f} {closed.degrees[b] - ed:>+9.4f}"
        if ex_radial:
            ev = ex_radial.volts[b]
            s += f" | {ev:>11.2f} {100 * (rv - ev) / ev:>+8.4f}"
        lines.append(s)
    return "\n".join(lines)


def study_result(net: HilNetwork, tie: tuple[int, int], radial: BusReading, closed: BusReading,
                 switch_ok: bool, ex_closed: Optional[ExagoState], ex_radial: Optional[ExagoState],
                 tol_pct: float, tol_deg: float, notes: list[str]) -> StudyResult:
    name = f"{tie[0]}-{tie[1]}"
    comps: list[Comparison] = []
    if ex_closed is not None:
        comps.append(compare(f"HIL with tie {name} closed", closed, ex_closed, net, tol_pct, tol_deg))
        if ex_radial is not None:
            comps.append(compare("HIL radial start", radial, ex_radial, net, tol_pct, tol_deg))

    chg = _changes(net, radial, closed)
    big = max(chg, key=lambda r: abs(r["dV"]))
    if not switch_ok:
        verdict = f"VERDICT: FAIL — the HIL did not accept closing tie {name}."
        passed: Optional[bool] = False
    elif ex_closed is None:
        verdict = (f"VERDICT: NOT COMPARED — tie {name} closed on the HIL (no ExaGO comparison "
                   f"asked); largest change {big['dV']:+.1f} V at bus {big['bus']}.")
        passed = None
    else:
        passed = all(c.passed for c in comps)
        main = comps[0]
        verdict = (f"VERDICT: {'PASS' if passed else 'FAIL'} — HIL with tie {name} closed "
                   f"{'matches' if main.passed else 'does NOT match'} {ex_closed.label} within "
                   f"{tol_pct} % / {tol_deg} deg at all {len(net.buses)} buses (largest "
                   f"{main.max_pct:+.4f} % at bus {main.max_pct_bus}, {main.max_deg:+.4f} deg at bus "
                   f"{main.max_deg_bus})")
        if len(comps) > 1:
            rad = comps[1]
            verdict += (f"; radial start {'matches' if rad.passed else 'does NOT match'} ExaGO's radial "
                        f"solution (largest {rad.max_pct:+.4f} % at bus {rad.max_pct_bus}, "
                        f"{rad.max_deg:+.4f} deg at bus {rad.max_deg_bus})")
        verdict += "."

    out = [verdict, ""]
    out += [f"Notes: {n}" for n in notes]
    out.append(f"HIL readings: radial start at simulation time {radial.sim_time:.2f} s, "
               f"tie {name} closed at {closed.sim_time:.2f} s.")
    for c in comps:
        out.append(c.line(tol_pct, tol_deg))
    if ex_closed is not None and ex_radial is None:
        out.append("Radial start: no ExaGO radial solution in the same session, not compared.")
    out.append("Largest voltage changes on closing the tie (HIL): " + ", ".join(
        f"bus {r['bus']} {r['dV']:+.1f} V ({r['ddeg']:+.3f} deg)" for r in chg))
    out += ["", "ALL BUSES (V line-to-neutral, phase a; angles in degrees):",
            _table(net, radial, closed, ex_radial, ex_closed)]
    return StudyResult(verdict, "\n".join(out), passed, comps, radial, closed)
