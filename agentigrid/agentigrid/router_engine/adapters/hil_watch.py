"""The Typhoon HIL in watch mode: the simulation keeps running, a scenario changes it, the agent watches.

Used in a router session without ExaGO (``--tool router --watch-scenario <file>``).
The agent starts the simulation once (``hil_start``); from then on AgentiGrid reads
all buses at fixed report times (``report_interval_s`` after the start, then every
interval) and every action runs until the next report, whose readings are the
action's result:

- ``hil_start``  start the simulation radial (all ties open), first report;
- ``wait``       nothing to do, next report;
- ``close_tie``  close one tie live now, next report.

The scenario (a JSON file) sets SCADA inputs at the start and at fixed times after
it (for example a load step). It runs inside AgentiGrid's waits and is never shown
to the agent: the agent sees only the readings. AgentiGrid does not judge the
readings either (no limit, no flags); judging them is the agent's task.

Ties are only closed live, never opened (live opening gives a wrong state on this
HIL). The watch ends after ``duration_s``; complete is refused until then.

Outputs in the session folder: ``status.csv`` (one row per report: time, lowest
voltage, ties closed, every bus in pu), ``events.csv`` (scenario inputs and the
agent's switchings with their times) and ``watch/report_<k>.json`` (full readings).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentigrid.hil_engine.commands import OPEN_REFUSED, REFUSED_ACTIONS, _pair
from agentigrid.hil_engine.connection import BusReading, HilDevice, TyphoonHil
from agentigrid.hil_engine.network import HilNetwork, describe, load_network
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.logging_setup import log_scope
from agentigrid.router_engine.adapters.base import ActionResult, Context, ToolAdapter

logger = logging.getLogger("agentigrid.router_engine.adapters.hil_watch")


@dataclass
class WatchEvent:
    t_s: float
    inputs: dict[str, float]
    label: str = ""


@dataclass
class WatchScenario:
    report_interval_s: float = 60.0
    duration_s: float = 600.0
    at_start: dict[str, float] = field(default_factory=dict)
    events: list[WatchEvent] = field(default_factory=list)
    name: str = ""

    @classmethod
    def load(cls, path: Path) -> "WatchScenario":
        d = json.loads(Path(path).read_text())
        return cls(report_interval_s=float(d.get("report_interval_s", 60)),
                   duration_s=float(d.get("duration_s", 600)),
                   at_start={k: float(v) for k, v in (d.get("at_start") or {}).items()},
                   events=sorted((WatchEvent(float(e["t_s"]), {k: float(v) for k, v in e["set"].items()},
                                             str(e.get("label", ""))) for e in d.get("events", [])),
                                 key=lambda e: e.t_s),
                   name=Path(path).name)

    @property
    def n_reports(self) -> int:
        return int(self.duration_s // self.report_interval_s)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


class HilWatchAdapter(ToolAdapter):
    name = "hil"
    title = "Typhoon hardware-in-the-loop (HIL) simulator, real time, watch mode"
    synonyms = "hardware in the loop, HIL, Typhoon, real-time simulator, the hardware, the simulation"
    actions = ("hil_start", "wait", "close_tie")
    standalone = True              # works without ExaGO steady states

    def __init__(self, model: Path, settings: HilSettings, scenario: WatchScenario,
                 device: HilDevice | None = None, network: HilNetwork | None = None,
                 clock=time.monotonic, sleep=time.sleep):
        self._model = Path(model)
        self._settings = settings
        self._sc = scenario
        self._device = device
        self._net = network
        self._clock = clock
        self._sleep = sleep
        self._t0: Optional[float] = None
        self._k = 0                                   # reports made
        self._next_event = 0
        self._ties: set[tuple[int, int]] = set()
        self._agent_events: list[str] = []            # since the last report
        self._last_pu: Optional[dict[int, float]] = None
        self._trend: list[tuple[str, float, int]] = []
        self._dir: Optional[Path] = None
        self._open = False

    # ------------------------------------------------------------------
    def ready(self) -> Optional[str]:
        if self._net is None:
            if not self._model.exists():
                return f"Typhoon model {self._model} does not exist"
            self._net = load_network(self._model)
        missing = [n for n in self._inputs() if not n]
        return f"scenario input names missing: {missing}" if missing else None

    def _inputs(self) -> list[str]:
        return list(self._sc.at_start) + [k for e in self._sc.events for k in e.inputs]

    def start(self, session_dir: Path, goal: str, quiet: bool) -> None:
        self._dir = Path(session_dir)

    @property
    def over(self) -> bool:
        return self._t0 is not None and self._k >= self._sc.n_reports

    def card(self) -> str:
        tie_pairs = {t.buses for t in self._net.ties}
        lines = ", ".join(f"{a}-{b}" for a, b in sorted(set(self._net.lines.values()) - tie_pairs))
        n, iv = self._sc.n_reports, self._sc.report_interval_s
        return f"""\
TOOL: {self.title}
{describe(self._net)}
Lines (bus-bus, no switch): {lines}
Watch mode: the simulation is started once and keeps running; it is not restarted.
After the start AgentiGrid reads all {len(self._net.buses)} buses every {iv:g} s ({n} reports, the
watch ends at {_mmss(n * iv)}). Every action below runs until the next report; that
report is the action's result. Conditions on the network may change while it runs.
Actions:
{{"action": "hil_start", "reasoning": "<why>"}}
    start the simulation in the normal state (radial, all ties open).
{{"action": "wait", "check": "<what the latest report shows>", "decision": "proceed",
 "reasoning": "<why, from your check>"}}
    do nothing until the next report.
{{"action": "close_tie", "tie": [<bus>, <bus>], "check": "<what the latest report shows>",
 "decision": "proceed", "reasoning": "<why this tie, from your check>"}}
    close one tie switch now, while the simulation runs; the next report shows the result.
- Only the tie switches listed above can be operated, and only closed: opening a tie
  while the simulation runs is not available on this HIL.
- Before the watch ends, complete is refused: keep reporting with wait or close_tie."""

    # ------------------------------------------------------------------
    def pending(self, store) -> Optional[str]:
        if self._t0 is None:
            return "the HIL simulation has not been started (hil_start)"
        if not self.over:
            left = self._sc.n_reports - self._k
            return f"the watch runs until {_mmss(self._sc.duration_s)} ({left} report(s) left)"
        return None

    def blocks_complete(self) -> Optional[str]:
        return self.pending(None)

    def overall(self) -> str:
        ties = ", ".join(f"{a}-{b}" for a, b in sorted(self._ties)) or "none"
        return (f"OVERALL: WATCH SESSION — {self._k} of {self._sc.n_reports} reports, ties closed: {ties}. "
                "AgentiGrid does not judge the readings in watch mode.")

    # ------------------------------------------------------------------
    def handle(self, action: str, data: dict, ctx: Context) -> ActionResult:
        with log_scope(logger, "Router HIL watch action", turn=ctx.turn, action=action):
            if action == "hil_start":
                if self._t0 is not None:
                    return ActionResult(feedback="hil_start refused: the simulation is already running; "
                                                 "use wait or close_tie")
                return self._start(ctx)
            if self._t0 is None:
                return ActionResult(feedback=f"{action} refused: start the simulation first with hil_start")
            if self.over:
                return ActionResult(feedback=f"{action} refused: the watch ended at "
                                             f"{_mmss(self._sc.duration_s)}; finish with complete")
            if action == "wait":
                return self._report(ctx, "wait")
            return self._close(data, ctx)

    def _dev(self) -> HilDevice:
        if self._device is None:
            self._device = TyphoonHil(self._settings, self._net)
        return self._device

    def _now(self) -> float:
        return self._clock() - self._t0

    def _event(self, kind: str, detail: str, t: Optional[float] = None) -> None:
        t = self._now() if t is None else t
        new = not (self._dir / "events.csv").exists()
        with (self._dir / "events.csv").open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["t_s", "time", "kind", "detail"])
            w.writerow([f"{t:.2f}", _mmss(t), kind, detail])

    def _start(self, ctx: Context) -> ActionResult:
        dev = self._dev()
        ctx.print(f"[HIL] Opening the HIL ({dev.describe()})...")
        dev.open()
        self._open = True
        for name, v in self._sc.at_start.items():
            if not dev.set_input(name, v):
                return ActionResult(stop=f"the HIL refused input {name}", text=f"Setting {name} failed",
                                    verdict=f"HIL start FAILED: setting {name} was refused")
        dev.restart(set())
        self._t0 = self._clock()
        self._event("start", "simulation started radial; " + "; ".join(f"{k}={v:g}" for k, v in self._sc.at_start.items()))
        ctx.print(f"[Turn {ctx.turn}] HIL started (radial); scenario {self._sc.name}")
        return self._report(ctx, "hil_start")

    def _close(self, data: dict, ctx: Context) -> ActionResult:
        if str(data.get("action", "")).lower() in REFUSED_ACTIONS:
            return ActionResult(feedback=OPEN_REFUSED)
        try:
            tie = _pair(data.get("tie"))
        except (ValueError, TypeError) as exc:
            return ActionResult(feedback=f'close_tie refused: {exc}. Example: {{"action": "close_tie", "tie": [12, 22]}}')
        if self._net.tie(*tie) is None:
            return ActionResult(feedback=f"close_tie refused: {tie[0]}-{tie[1]} is not a tie switch; "
                                         f"the tie switches are {self._net.tie_names()}")
        if tie in self._ties:
            return ActionResult(feedback=f"close_tie refused: tie {tie[0]}-{tie[1]} is already closed")
        t = self._now()
        ok = self._dev().close_tie(tie)
        self._event("agent", f"close tie {tie[0]}-{tie[1]}: " + ("accepted" if ok else "REFUSED by the HIL"), t)
        if not ok:
            text = f"The HIL refused to close tie {tie[0]}-{tie[1]} at {_mmss(t)}."
            return ActionResult(text=text, verdict=text, stop=f"the HIL refused closing {tie[0]}-{tie[1]}")
        self._ties.add(tie)
        self._agent_events.append(f"you closed tie {tie[0]}-{tie[1]} at {_mmss(t)}")
        ctx.print(f"[Turn {ctx.turn}] Closed tie {tie[0]}-{tie[1]} live at {_mmss(t)}")
        return self._report(ctx, f"close_tie {tie[0]}-{tie[1]}")

    # ------------------------------------------------------------------
    def _apply_due(self) -> None:
        dev = self._dev()
        while self._next_event < len(self._sc.events) and self._sc.events[self._next_event].t_s <= self._now():
            e = self._sc.events[self._next_event]
            late = self._now() - e.t_s
            ok = all(dev.set_input(k, v) for k, v in e.inputs.items())
            self._event("scenario", f"{e.label or 'inputs'}: " + "; ".join(f"{k}={v:g}" for k, v in e.inputs.items())
                        + ("" if ok else " (REFUSED)") + (f" (applied {late:.1f} s late)" if late > 1 else ""))
            self._next_event += 1

    def _report(self, ctx: Context, what: str) -> ActionResult:
        target = (self._k + 1) * self._sc.report_interval_s
        late = self._now() - target
        while self._now() < target:
            self._apply_due()
            nxt = min([target] + [e.t_s for e in self._sc.events[self._next_event:]])
            self._sleep(max(0.0, min(nxt - self._now(), 1.0)))
        self._apply_due()
        t = self._now()
        r = self._dev().read_buses()
        self._k += 1
        return self._make_report(ctx, what, t, r, late)

    def _make_report(self, ctx: Context, what: str, t: float, r: BusReading, late: float) -> ActionResult:
        vb = self._net.v_ln
        pu = {b: r.volts[b] / vb for b in self._net.buses}
        bmin = min(pu, key=pu.get)
        ties = ", ".join(f"{a}-{b}" for a, b in sorted(self._ties)) or "none"
        n = self._sc.n_reports
        head = f"REPORT {self._k} of {n} at {_mmss(t)} after the start (simulation time {r.sim_time:.1f} s)"
        out = [head + (f"; taken {late:.0f} s late because your reply came after the report time" if late > 1 else ""),
               f"Ties closed: {ties}",
               f"Your switching since the last report: {'; '.join(self._agent_events) or 'none'}",
               f"Lowest voltage: {pu[bmin]:.4f} pu at bus {bmin}",
               "Open ties, voltage at each end: " + "; ".join(
                   f"{t_.buses[0]}-{t_.buses[1]}: bus {t_.buses[0]} {pu[t_.buses[0]]:.4f} pu, "
                   f"bus {t_.buses[1]} {pu[t_.buses[1]]:.4f} pu" for t_ in self._net.ties if t_.buses not in self._ties)
               or "Open ties: none"]
        if self._trend:
            out.append("Earlier reports (time: lowest voltage, bus): " + "; ".join(
                f"{tt}: {v:.4f} pu, bus {b}" for tt, v, b in self._trend))
        if self.over:
            out.append("This was the last report: the watch has ended. Finish with complete.")
        head2 = f"{'Bus':>4} | {'V (L-N)':>9} | {'V pu':>7} | {'angle deg':>9} | {'change since last report, pu':>28}"
        out += ["", "ALL BUSES (phase a):", head2, "-" * len(head2)]
        for b in self._net.buses:
            d = "" if self._last_pu is None else f"{pu[b] - self._last_pu[b]:+.4f}"
            out.append(f"{b:>4} | {r.volts[b]:>9.2f} | {pu[b]:>7.4f} | {r.degrees[b]:>9.4f} | {d:>28}")
        text = "\n".join(out)
        # records
        self._trend.append((_mmss(t), pu[bmin], bmin))
        self._last_pu = pu
        acts = "; ".join(self._agent_events)
        self._agent_events = []
        new = not (self._dir / "status.csv").exists()
        with (self._dir / "status.csv").open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["report", "t_s", "time", "sim_time_s", "action", "ties_closed", "vmin_pu", "vmin_bus"]
                           + [f"V{b}_pu" for b in self._net.buses])
            w.writerow([self._k, f"{t:.2f}", _mmss(t), f"{r.sim_time:.2f}", what, ties, f"{pu[bmin]:.6f}", bmin]
                       + [f"{pu[b]:.6f}" for b in self._net.buses])
        rep = self._dir / "watch" / f"report_{self._k:02d}.json"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps({"report": self._k, "t_s": t, "sim_time": r.sim_time, "action": what,
                                   "ties_closed": sorted(self._ties), "volts": r.volts, "degrees": r.degrees,
                                   "pu": pu, "agent_switching": acts, "late_s": max(0.0, late)}, indent=1))
        verdict = f"Report {self._k} at {_mmss(t)}: lowest voltage {pu[bmin]:.4f} pu at bus {bmin}; ties closed: {ties}"
        row = {"tool": "HIL", "state": f"R{self._k}", "what": what,
               "result": f"{_mmss(t)}: Vmin {pu[bmin]:.4f} pu (b{bmin}); ties closed: {ties}"}
        ctx.print(f"[Turn {ctx.turn}] {verdict}")
        return ActionResult(text=text, verdict=verdict, row=row,
                            record={"report": self._k, "t_s": t, "report_file": str(rep)})

    def close(self) -> None:
        if self._open and self._device is not None:
            try:
                for name in self._inputs():
                    if name.endswith(".Enable") or name.startswith("Load_step"):
                        self._device.set_input(name, 0.0)
            except Exception:                        # noqa: BLE001 - clean-up
                logger.exception("Could not switch the scenario loads off")
            self._device.shutdown()
