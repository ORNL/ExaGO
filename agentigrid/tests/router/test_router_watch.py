"""Router watch sessions (--tool router --watch-scenario): HIL only, a scenario changes the running
simulation, the agent watches the reports. Scripted LLM, fake HIL, fake clock; no Typhoon, no LLM call.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agentigrid.config import load_config
from agentigrid.hil_engine.connection import BusReading, HilDevice
from agentigrid.hil_engine.network import load_network
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.router_engine.adapters.hil_watch import HilWatchAdapter, WatchEvent, WatchScenario
from agentigrid.router_engine.agent_loop import RouterAgentLoop
from tests.hil.test_agent_loop import ScriptedBackend

MODEL = Path("/home/shilongphone/Study_loop/models/ieee33/ieee 33 bus system.tse")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs the Study_loop IEEE 33 model")

ANALYSIS = "analysis text"
LOAD = "VarLoad716.Enable"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeWatchHil(HilDevice):
    """Bus 18 at 0.925 pu radial, 0.915 with the load on, 0.940 once tie 18-33 or 9-15 is closed."""

    def __init__(self, v_ln, clock):
        self.v_ln, self.clock = v_ln, clock
        self.inputs: dict = {}
        self.closed: set = set()
        self.log: list = []

    def open(self):
        self.log.append("open")

    def restart(self, closed_ties):
        self.closed = set(closed_ties)
        self.log.append(("restart", sorted(closed_ties)))

    def set_input(self, name, value):
        self.inputs[name] = value
        self.log.append(("set", name, value, round(self.clock() - 1000.0, 1)))
        return True

    def close_tie(self, tie):
        self.closed.add(tuple(tie))
        self.log.append(("close_tie", tuple(tie)))
        self.clock.sleep(5)
        return True

    def read_buses(self):
        self.clock.sleep(5)
        v18 = 0.925
        if self.inputs.get(LOAD):
            v18 = 0.915
        if self.closed & {(18, 33), (9, 15)}:
            v18 = 0.940
        pu = {b: 0.95 for b in range(1, 34)}
        pu[1] = 1.0
        pu[18] = v18
        return BusReading(self.clock() - 1000.0, {b: v * self.v_ln for b, v in pu.items()},
                          {b: 0.0 for b in pu})

    def shutdown(self):
        self.log.append("shutdown")


SCENARIO = WatchScenario(report_interval_s=60, duration_s=300, at_start={LOAD: 0.0},
                         events=[WatchEvent(150, {LOAD: 1.0}, "load step")], name="test.json")


def _ck(action, **kw):
    return {"action": action, "check": "c", "decision": "proceed", "reasoning": "r", **kw}


def _run(tmp_path, replies, scenario=SCENARIO, max_turns=20):
    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "workdir"),
                                           "output.logs_dir": str(tmp_path / "logs")})
    net = load_network(MODEL)
    clock = Clock()
    device = FakeWatchHil(net.v_ln, clock)
    adapter = HilWatchAdapter(MODEL, HilSettings.from_config(cfg, model=MODEL, rest_state="stopped"), scenario,
                              device=device, network=net, clock=clock, sleep=clock.sleep)
    backend = ScriptedBackend(list(replies) + [ANALYSIS])
    session = RouterAgentLoop(cfg, [adapter], max_turns=max_turns, backend=backend, quiet=True).run(
        "start the HIL and watch it; close a tie if a voltage falls below 0.92 pu")
    journal = json.loads(session.journal_path.read_text())
    return session, journal, backend, device


def _errors(journal):
    return [e["description"] for e in journal["entries"] if e["action"] == "error"]


# reports at 1:00 .. 5:00; the load step at 2:30 shows in report 3; the tie is closed after it
GOOD = [{"action": "hil_start", "reasoning": "r"}, _ck("wait"), _ck("wait"), _ck("close_tie", tie=[18, 33]),
        _ck("wait"), _ck("complete", decision="stop", summary="s", details="d")]


def test_watch_session_reports_on_time_and_hides_the_scenario(tmp_path):
    session, journal, backend, device = _run(tmp_path, GOOD)
    assert session.termination_reason == "completed" and not _errors(journal)
    rows = list(csv.DictReader((session.session_dir / "status.csv").open()))
    assert [int(r["report"]) for r in rows] == [1, 2, 3, 4, 5]
    assert [round(float(r["t_s"])) for r in rows] == [60, 120, 180, 240, 300]
    assert [r["vmin_bus"] for r in rows][:3] == ["18", "18", "18"]
    assert float(rows[1]["V18_pu"]) == pytest.approx(0.925) and float(rows[2]["V18_pu"]) == pytest.approx(0.915)
    assert float(rows[3]["V18_pu"]) == pytest.approx(0.940) and rows[3]["ties_closed"] == "18-33"
    # the load step ran at its time, inside a wait, and the agent never saw it
    assert ("set", LOAD, 1.0, 150.0) in device.log
    events = (session.session_dir / "events.csv").read_text()
    assert "load step" in events and "close tie 18-33: accepted" in events
    prompts = "\n".join(p for _, p in backend.prompts[:-1]) + backend.prompts[0][0]
    assert "load step" not in prompts and "VarLoad" not in prompts and "0.92" not in backend.prompts[0][0]
    # the analysis gets the scenario events
    assert "load step" in backend.prompts[-1][1]
    assert device.log[-1] == "shutdown" and ("set", LOAD, 0.0, pytest.approx(300, abs=20)) in device.log[-3:]


def test_report_lists_open_ties_with_both_end_voltages_and_trend(tmp_path):
    _, _, backend, _ = _run(tmp_path, GOOD)
    third = backend.prompts[3][1]          # prompt after report 3 (close_tie comes from it)
    assert "REPORT 3 of 5 at 03:00" in third
    assert "Lowest voltage: 0.9150 pu at bus 18" in third
    assert "18-33: bus 18 0.9150 pu, bus 33 0.9500 pu" in third
    assert "Earlier reports (time: lowest voltage, bus): 01:00: 0.9250 pu, bus 18; 02:00: 0.9250 pu, bus 18" in third
    fourth = backend.prompts[4][1]
    assert "you closed tie 18-33 at 03:" in fourth and "Ties closed: 18-33" in fourth
    assert "18-33:" not in fourth.split("Open ties, voltage at each end:")[1].splitlines()[0]


def test_complete_refused_until_the_watch_ends(tmp_path):
    done = _ck("complete", decision="stop", summary="s", details="d")
    replies = [{"action": "hil_start", "reasoning": "r"}, done, done, _ck("wait"), _ck("wait"), _ck("wait"),
               _ck("wait"), done]
    session, journal, *_ = _run(tmp_path, replies)
    errs = _errors(journal)
    assert len(errs) == 2 and all("complete refused: the watch runs until 05:00" in e for e in errs)
    assert session.termination_reason == "completed"


def test_bad_switching_refused(tmp_path):
    replies = [{"action": "hil_start", "reasoning": "r"}, _ck("close_tie", tie=[16, 17]), _ck("close_tie", tie=[18, 33]),
               _ck("close_tie", tie=[33, 18]), _ck("open_tie", tie=[18, 33]), _ck("wait"), _ck("wait"), _ck("wait"),
               _ck("complete", decision="stop", summary="s", details="d")]
    session, journal, _, device = _run(tmp_path, replies)
    errs = _errors(journal)
    assert any("16-17 is not a tie switch" in e for e in errs)
    assert any("tie 18-33 is already closed" in e for e in errs)
    assert any("Unknown action 'open_tie'" in e for e in errs)
    assert [x for x in device.log if x[0] == "close_tie"] == [("close_tie", (18, 33))]


def test_watch_prompt_has_no_exago_and_no_objectives_call(tmp_path):
    session, journal, backend, _ = _run(tmp_path, GOOD)
    system = backend.prompts[0][0]
    assert "HOW A WATCH SESSION WORKS" in system and "exago_solve" not in system
    assert "Lines (bus-bus, no switch): 1-2" in system and "18-33" not in system.split("Lines (bus-bus")[1].splitlines()[0]
    phases = [c["phase"] for c in journal["token_usage"]["per_call"]]
    assert "objective_extraction" not in phases and phases[-1] == "analysis"
    assert "OVERALL: WATCH SESSION — 5 of 5 reports, ties closed: 18-33" in session.overall_verdict


def test_late_reply_takes_the_report_at_once_and_says_so(tmp_path):
    class SlowBackend(ScriptedBackend):
        def __init__(self, replies, clock):
            super().__init__(replies)
            self.clock = clock

        def complete(self, system_prompt, user_prompt, temperature=None):
            if len(self.prompts) == 1:
                self.clock.sleep(90)       # the reply after report 1 comes after report 2's time
            return super().complete(system_prompt, user_prompt, temperature)

    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "w"), "output.logs_dir": str(tmp_path / "l")})
    net = load_network(MODEL)
    clock = Clock()
    device = FakeWatchHil(net.v_ln, clock)
    adapter = HilWatchAdapter(MODEL, HilSettings.from_config(cfg, model=MODEL), SCENARIO, device=device,
                              network=net, clock=clock, sleep=clock.sleep)
    backend = SlowBackend(GOOD + [ANALYSIS], clock)
    session = RouterAgentLoop(cfg, [adapter], backend=backend, quiet=True).run("watch")
    rows = list(csv.DictReader((session.session_dir / "status.csv").open()))
    assert float(rows[1]["t_s"]) > 120
    assert "taken" in backend.prompts[2][1] and "late" in backend.prompts[2][1]
