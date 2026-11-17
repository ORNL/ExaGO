"""The Typhoon HIL in a router session: repeat a steady state's switching on the HIL and compare.

``hil_step`` takes the latest or the best steady state S. The HIL can only switch
ties, and only close them (live opening gives a wrong state on this HIL), so S must
have been reached from the input case by closing ties only.

- S built on no earlier state, or with no new tie: restart the simulation with S's
  ties closed, read all buses, compare with S.
- S built on state P with new ties: restart with P's ties closed, read (compared
  with P), close the new ties live one after the other, read (compared with S).

Readings: settle, then several readings averaged (``hil.*`` settings); limits
``hil.tol_pct`` and ``hil.tol_deg``; the verdict is decided here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from agentigrid.hil_engine.check import Comparison, compare
from agentigrid.hil_engine.connection import BusReading, HilDevice, TyphoonHil
from agentigrid.hil_engine.from_exago import ExagoState, load_state
from agentigrid.hil_engine.network import HilNetwork, describe, load_network
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.logging_setup import log_scope
from agentigrid.router_engine.adapters.base import ActionResult, Context, ToolAdapter
from agentigrid.router_engine.state import SteadyState, StateStore

logger = logging.getLogger("agentigrid.router_engine.adapters.hil")

NEAR_LIMIT = 0.9          # a difference at or above this share of its limit is flagged


def _ties_text(ties) -> str:
    return ", ".join(f"{a}-{b}" for a, b in sorted(ties)) or "none"


def _reading_dict(r: BusReading) -> dict:
    return {"sim_time": r.sim_time, "volts": r.volts, "degrees": r.degrees}


class HilAdapter(ToolAdapter):
    name = "hil"
    title = "Typhoon hardware-in-the-loop (HIL) simulator, real time"
    synonyms = "hardware in the loop, HIL, Typhoon, real-time simulator, the hardware"
    actions = ("hil_step",)

    def __init__(self, model: Path, settings: HilSettings, device: HilDevice | None = None,
                 network: HilNetwork | None = None):
        self._model = Path(model)
        self._settings = settings
        self._device = device
        self._net = network
        self._open = False
        self._done: dict[int, Optional[bool]] = {}     # state id -> passed (None: skipped)
        self._used = False

    # ------------------------------------------------------------------
    def ready(self) -> Optional[str]:
        if self._net is None:
            if not self._model.exists():
                return f"Typhoon model {self._model} does not exist"
            self._net = load_network(self._model)
        return None

    @property
    def network(self) -> HilNetwork:
        return self._net

    def card(self) -> str:
        return f"""\
TOOL: {self.title}
{describe(self._net)}
Action:
{{"action": "hil_step",
 "check": "<what the latest result shows>",
 "decision": "proceed",
 "start_from": "latest" | "best",
 "reasoning": "<why, from your check>"}}
- hil_step repeats on the HIL the switching that produced the chosen ExaGO steady
  state, then compares every bus with ExaGO (limits {self._settings.tol_pct} % on voltage
  magnitude and {self._settings.tol_deg} degrees on angle). If that state was built on an
  earlier one, the HIL starts with the earlier state's ties closed, is compared with
  it, closes the new ties live and is compared with the new state.
- The HIL can only close the ties listed above. A steady state with any other
  change (a line out, a load or generator change, a tie opened) cannot be run on
  the HIL: AgentiGrid reports it as SKIPPED with the reason.
- Each steady state is run on the HIL once."""

    def _chain(self, s: SteadyState, store: StateStore) -> list[SteadyState]:
        out = []
        while s is not None:
            out.append(s)
            s = store.states[s.parent] if s.parent is not None else None
        return out

    def _why_not(self, s: SteadyState, store: StateStore) -> Optional[str]:
        """None if the HIL can reach *s* by closing ties, else the reason."""
        tie_pairs = {t.buses for t in self._net.ties}
        for st in self._chain(s, store):
            for c in st.commands:
                act = str(c.get("action", "")).lower()
                if act == "set_branch_status":
                    pair = tuple(sorted((int(c.get("fbus", 0)), int(c.get("tbus", 0)))))
                    if pair not in tie_pairs:
                        return f"{st.name} switches {pair[0]}-{pair[1]}, which is not a tie on the HIL"
                    if int(c.get("status", 1)) == 0:
                        return (f"{st.name} opens tie {pair[0]}-{pair[1]}; opening a tie live is not "
                                "available on this HIL (it gives a wrong steady state)")
                else:
                    return f"{st.name} has an ExaGO change the HIL cannot make ({act})"
        return None

    def pending(self, store: StateStore) -> Optional[str]:
        if not self._used or not store.states:
            return None
        s = store.latest()
        if s.sid in self._done or self._why_not(s, store):
            return None
        return f"the latest steady state {s.name} has not been run on the HIL"

    # ------------------------------------------------------------------
    def handle(self, action: str, data: dict, ctx: Context) -> ActionResult:
        with log_scope(logger, "Router HIL action", turn=ctx.turn):
            return self._step(data, ctx)

    def _step(self, data: dict, ctx: Context) -> ActionResult:
        store = ctx.store
        s, why = store.choose(data.get("start_from"))
        if s is None:
            return ActionResult(feedback=f"hil_step: no steady state to run: {why}. Call exago_solve first.")
        if s.sid in self._done:
            return ActionResult(feedback=f"hil_step: {s.name} was already run on the HIL; "
                                         "make a new steady state first, or complete")
        self._used = True
        reason = self._why_not(s, store)
        ex_s = load_state(s.solved_file, self._net, f"ExaGO {s.name} ({s.label})")
        if not reason and not ex_s.usable:
            reason = "the HIL cannot build its network: " + "; ".join(ex_s.problems)
        if reason:
            self._done[s.sid] = None
            text = f"hil_step SKIPPED for {s.name}: {reason}."
            return ActionResult(text=text, verdict=text, record={"state": s.name},
                                row={"tool": "HIL", "state": s.name, "what": s.label, "result": text})
        tol_pct, tol_deg = self._settings.tol_pct, self._settings.tol_deg
        p = store.states[s.parent] if s.parent is not None else None
        ex_p: Optional[ExagoState] = None
        if p is not None:
            ex_p = load_state(p.solved_file, self._net, f"ExaGO {p.name} ({p.label})")
        new_ties = sorted(ex_s.closed_ties - ex_p.closed_ties) if ex_p else []
        if not self._open:
            ctx.print(f"[HIL] Opening the HIL ({self._device_describe()})...")
            self._open = True
            self._device_get().open()
        dev = self._device_get()
        t0 = time.monotonic()
        commands: list[dict] = []
        readings: dict = {}
        comps: list[Comparison] = []
        ok = True
        if not new_ties:
            ctx.print(f"[Turn {ctx.turn}] HIL {s.name}: start with ties closed: {_ties_text(ex_s.closed_ties)}...")
            dev.restart(ex_s.closed_ties)
            r = dev.read_buses()
            commands.append({"command": "restart", "ties_closed": [f"{a}-{b}" for a, b in sorted(ex_s.closed_ties)],
                             "sim_time": r.sim_time})
            readings["state"] = _reading_dict(r)
            comps.append(compare(f"HIL {s.name}", r, ex_s, self._net, tol_pct, tol_deg))
            changes = []
        else:
            names = ", ".join(f"{a}-{b}" for a, b in new_ties)
            ctx.print(f"[Turn {ctx.turn}] HIL {s.name}: start as {p.name} (ties closed: "
                      f"{_ties_text(ex_p.closed_ties)}), then close {names} live...")
            dev.restart(ex_p.closed_ties)
            before = dev.read_buses()
            commands.append({"command": "restart", "ties_closed": [f"{a}-{b}" for a, b in sorted(ex_p.closed_ties)],
                             "sim_time": before.sim_time})
            comps.append(compare(f"HIL start ({p.name})", before, ex_p, self._net, tol_pct, tol_deg))
            for tie in new_ties:
                accepted = dev.close_tie(tie)
                t = self._net.tie(*tie)
                commands.append({"command": "close_tie", "tie": f"{tie[0]}-{tie[1]}", "switch": t.switch,
                                 "scada_input": t.scada_input, "value": 1.0, "accepted": accepted})
                ok = ok and accepted
                if not accepted:
                    break
            after = dev.read_buses()
            commands[-1]["sim_time_read"] = after.sim_time
            readings["start"] = _reading_dict(before)
            readings["closed"] = _reading_dict(after)
            comps.append(compare(f"HIL {s.name} (after closing {names})", after, ex_s, self._net, tol_pct, tol_deg))
            changes = sorted(({"bus": b, "dV": after.volts[b] - before.volts[b],
                               "ddeg": after.degrees[b] - before.degrees[b]} for b in self._net.buses),
                             key=lambda x: abs(x["dV"]), reverse=True)[:6]
        stop = None
        if not ok:
            passed = False
            verdict = f"VERDICT HIL {s.name}: FAIL — the HIL did not accept closing a tie."
            stop = f"the HIL refused a switch command ({s.name})"
        else:
            passed = all(c.passed for c in comps)
            last = comps[-1]
            verdict = (f"VERDICT HIL {s.name}: {'PASS' if passed else 'FAIL'} — HIL "
                       f"{'matches' if last.passed else 'does NOT match'} ExaGO {s.name} within {tol_pct} % / "
                       f"{tol_deg} deg at all {len(self._net.buses)} buses (largest {last.max_pct:+.4f} % at bus "
                       f"{last.max_pct_bus}, {last.max_deg:+.4f} deg at bus {last.max_deg_bus})")
            if len(comps) > 1:
                st = comps[0]
                verdict += (f"; start state {'matches' if st.passed else 'does NOT match'} ExaGO {p.name} "
                            f"(largest {st.max_pct:+.4f} % at bus {st.max_pct_bus}, {st.max_deg:+.4f} deg at bus "
                            f"{st.max_deg_bus})")
            verdict += "."
        self._done[s.sid] = passed
        attention = []
        for c in comps:
            for share, what in ((abs(c.max_pct) / tol_pct, f"magnitude {c.max_pct:+.4f} % at bus {c.max_pct_bus}"),
                                (abs(c.max_deg) / tol_deg, f"angle {c.max_deg:+.4f} deg at bus {c.max_deg_bus}")):
                if share >= NEAR_LIMIT:
                    attention.append(f"{c.name}: {what} is at {100 * share:.0f} % of its limit")
        out = [verdict, f"Steady state used: {s.name} ({why}).", ""] + [c.line(tol_pct, tol_deg) for c in comps]
        if attention:
            out.append("Near the limit (at or above " + f"{100 * NEAR_LIMIT:.0f} %): " + "; ".join(attention))
        if changes:
            out.append("Largest voltage changes on closing (HIL): " + ", ".join(
                f"bus {x['bus']} {x['dV']:+.1f} V ({x['ddeg']:+.3f} deg)" for x in changes))
        out += ["", f"ALL BUSES, HIL against ExaGO {s.name} (V line-to-neutral, phase a; degrees):",
                self._table(comps[-1])]
        hil_file = ctx.session_dir / "hil" / f"{s.name}.json"
        hil_file.parent.mkdir(parents=True, exist_ok=True)
        hil_file.write_text(json.dumps({"state": s.name, "label": s.label, "commands": commands,
                                        "readings": readings, "comparisons": [asdict(c) for c in comps],
                                        "verdict": verdict}, indent=1, default=str))
        row = {"tool": "HIL", "state": s.name, "what": s.label, "passed": passed,
               "result": (f"{'PASS' if passed else 'FAIL'}: "
                          + "; ".join(f"{c.name} {c.max_pct:+.4f} % (b{c.max_pct_bus}) {c.max_deg:+.4f} deg "
                                      f"(b{c.max_deg_bus})" for c in comps))}
        return ActionResult(text="\n".join(out), verdict=verdict, passed=passed, stop=stop,
                            record={"state": s.name, "commands": commands, "hil_file": str(hil_file),
                                    "elapsed_s": time.monotonic() - t0}, row=row, attention=attention)

    def _table(self, c: Comparison) -> str:
        head = f"{'Bus':>4} | {'HIL V':>9} {'ExaGO V':>9} {'diff %':>8} | {'HIL deg':>9} {'ExaGO deg':>9} {'diff deg':>9}"
        rows = [head, "-" * len(head)]
        for r in c.rows:
            rows.append(f"{r['bus']:>4} | {r['hil_V']:>9.2f} {r['exago_V']:>9.2f} {r['err_pct']:>+8.4f} | "
                        f"{r['hil_deg']:>9.4f} {r['exago_deg']:>9.4f} {r['err_deg']:>+9.4f}")
        return "\n".join(rows)

    def _device_get(self) -> HilDevice:
        if self._device is None:
            self._device = TyphoonHil(self._settings, self._net)
        return self._device

    def _device_describe(self) -> str:
        return self._device_get().describe()

    def close(self) -> None:
        if self._open and self._device is not None:
            self._device.shutdown()
