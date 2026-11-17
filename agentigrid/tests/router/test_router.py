"""Router sessions (--tool router) with a scripted LLM, real PFLOW, a fake HIL and a fake GridKit.

Uses the study's IEEE 33 model and ExaGO case, and the human reference solutions
(Study_loop/results/by_hand) as the fake HIL's answers. No Typhoon, no GridKit, no LLM call.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import pytest

from agentigrid.config import load_config
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.hil_engine.connection import BusReading, HilDevice
from agentigrid.hil_engine.network import load_network
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.router_engine.adapters.exago import ExagoAdapter
from agentigrid.router_engine.adapters.gridkit import GridkitAdapter
from agentigrid.router_engine.adapters.hil import HilAdapter
from agentigrid.router_engine.agent_loop import RouterAgentLoop
from agentigrid.router_engine.state import solved_metrics

from tests.hil.test_agent_loop import ScriptedBackend

STUDY = Path("/home/shilongphone/Study_loop")
MODEL = STUDY / "models" / "ieee33" / "ieee 33 bus system.tse"
CASE = STUDY / "inputs" / "ieee33_typhoon_radial_g.m"
BY_HAND = STUDY / "results" / "by_hand"
PFLOW = Path(__file__).resolve().parents[2] / "applications" / "exago" / "pflow"
SEQUENCE = [(8, 21), (9, 15), (12, 22), (18, 33), (25, 29)]
TIES = set(SEQUENCE)

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and CASE.exists() and BY_HAND.exists() and PFLOW.exists()),
    reason="needs the Study_loop IEEE 33 model, case, human reference and the PFLOW binary")


def _ref(k: int) -> Path:
    return BY_HAND / ("g_0_open.m" if k == 0 else f"g_{k}_closed.m")


def _ties_of(path: Path) -> frozenset:
    return frozenset(tuple(sorted((b.fbus, b.tbus))) for b in parse_matpower(path).branches
                     if b.status and tuple(sorted((b.fbus, b.tbus))) in TIES)


class FakeHil(HilDevice):
    """Answers with the human-reference ExaGO voltages of the current tie state (times *scale*)."""

    def __init__(self, v_ln: float, scale: float = 1.0, accept: bool = True):
        self.v_ln, self.scale, self.accept = v_ln, scale, accept
        self.states = {}
        for k in range(6):
            case = parse_matpower(_ref(k))
            self.states[_ties_of(_ref(k))] = ({b.bus_i: b.Vm for b in case.buses},
                                              {b.bus_i: b.Va for b in case.buses})
        self.closed: set = set()
        self.log: list = []
        self.t = 0.0

    def open(self):
        self.log.append("open")

    def restart(self, closed_ties):
        self.closed = set(closed_ties)
        self.log.append(("restart", sorted(closed_ties)))

    def close_tie(self, tie):
        self.log.append(("close_tie", tuple(tie)))
        if self.accept:
            self.closed.add(tuple(tie))
        return self.accept

    def read_buses(self):
        self.t += 10.0
        vm, va = self.states[frozenset(self.closed)]
        return BusReading(self.t, {b: v * self.v_ln * self.scale for b, v in vm.items()}, dict(va))

    def describe(self):
        return "fake HIL"

    def shutdown(self):
        self.log.append("shutdown")


class FakeGridkitLoop:
    """Stands in for GridkitAgentLoop's external interface."""

    def __init__(self, ok=True, verdict="VERDICT: PASS — all 3 bus faults stable."):
        self.ok, self.verdict, self.calls = ok, verdict, []

    def network_text(self):
        return "GridKit test case: 3 buses, 1 machine"

    def open_external(self, case_file, goal, session_dir):
        self.calls.append(("open", Path(case_file).name))

    def use_steady_state(self, source, iteration, name):
        self.calls.append(("use", name, Path(source.solved_file).name))
        return self.ok, f"operating point {name}"

    def fault_screen(self, iteration, data):
        self.calls.append(("screen", dict(data)))
        return self.verdict + "\nrest of the table", None

    def export_journal(self):
        self.calls.append(("export",))


NO_OBJECTIVES = {"objectives": []}
MIN_LOSSES = {"objectives": [{"name": "active_losses_mw", "direction": "minimize", "priority": "primary"}]}


def _close(tie, status=1):
    return {"action": "set_branch_status", "fbus": tie[0], "tbus": tie[1], "status": status}


def _solve(*cmds, start="latest", check="c-ok", decision="proceed"):
    return {"action": "exago_solve", "start_from": start, "commands": list(cmds),
            "check": check, "decision": decision, "reasoning": f"r-exago-{len(cmds)}"}


def _hil(start="latest", check="c-ok", decision="proceed"):
    return {"action": "hil_step", "start_from": start, "check": check, "decision": decision,
            "reasoning": "r-hil"}


DONE = {"action": "complete", "check": "c-last", "decision": "stop", "reasoning": "r-done",
        "summary": "S", "details": "D"}
ANALYSIS = "1. What was done\nAnalysis text A."


def full_script():
    out = [NO_OBJECTIVES, _solve(), _hil()]
    for t in SEQUENCE:
        out += [_solve(_close(t)), _hil()]
    return out + [DONE]


def _run(tmp_path, replies, goal="iterate between ExaGO and Typhoon HIL for each step", scale=1.0,
         accept=True, pflow=PFLOW, max_turns=20, hil=True, gridkit=None, case=CASE):
    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "workdir"),
                                           "output.logs_dir": str(tmp_path / "logs")})
    net = load_network(MODEL)
    device = FakeHil(net.v_ln, scale=scale, accept=accept)
    adapters = [ExagoAdapter(case, {"pflow": pflow})]
    if hil:
        adapters.append(HilAdapter(MODEL, HilSettings.from_config(cfg, model=MODEL), device=device, network=net))
    if gridkit is not None:
        adapters.append(GridkitAdapter(tmp_path / "case.json", GridkitSettings.from_config(cfg), loop=gridkit))
        (tmp_path / "case.json").write_text("{}")
    backend = ScriptedBackend(list(replies) + [ANALYSIS])
    session = RouterAgentLoop(cfg, adapters, max_turns=max_turns, backend=backend, quiet=True).run(goal)
    journal = json.loads(session.journal_path.read_text())
    return session, journal, backend, device


def _errors(journal):
    return [e["description"] for e in journal["entries"] if e["action"] == "error"]


def _hil_rows(session):
    return [r for r in session.rows if r["tool"] == "HIL"]


# --------------------------------------------------------------------------- full sequence

def test_full_sequence_passes_and_matches_human_reference(tmp_path):
    session, journal, backend, device = _run(tmp_path, full_script())
    assert session.termination_reason == "completed"
    assert [r["state"] for r in _hil_rows(session)] == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert all(r["passed"] for r in _hil_rows(session))
    assert session.overall_verdict.startswith("OVERALL VERDICT: PASS — all 6 checks passed")
    assert _errors(journal) == []
    for k in range(6):
        ours = parse_matpower(session.session_dir / "exago" / f"S{k}" / "pflowout.m")
        ref = parse_matpower(_ref(k))
        assert max(abs(a.Vm - b.Vm) for a, b in zip(ours.buses, ref.buses)) < 1e-6
        assert max(abs(a.Va - b.Va) for a, b in zip(ours.buses, ref.buses)) < 1e-6
    restarts = [e for e in device.log if isinstance(e, tuple) and e[0] == "restart"]
    closes = [e[1] for e in device.log if isinstance(e, tuple) and e[0] == "close_tie"]
    assert restarts[0] == ("restart", []) and restarts[1] == ("restart", [])
    assert restarts[3] == ("restart", [(8, 21), (9, 15)])
    assert closes == SEQUENCE
    assert device.log[0] == "open" and device.log[-1] == "shutdown"
    log = session.log_path.read_text()
    assert "LLM reasoning: r-exago-1" in log and "Final details:\nD" in log
    assert "OVERALL VERDICT: PASS" in session.summary_path.read_text()
    assert (session.session_dir / "hil" / "S5.json").exists()
    prompt = (session.session_dir / "system_prompt.txt").read_text()
    assert "ACPF" in prompt and "hardware in the loop" in prompt and "set_branch_status" in prompt
    assert "Tie_sw_3" in prompt


def test_exago_step_shows_load_and_losses(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), DONE])
    text = next(e["result_text"] for e in journal["entries"] if e["action"] == "exago_solve")
    m = re.search(r"Generation ([\d.]+) MW; load ([\d.]+) MW .* losses ([\d.]+) MW", text)
    assert m and abs(float(m.group(3)) - 0.157) < 0.002


def test_mismatch_fails_and_gate_stops_the_sequence(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _solve(_close((8, 21))), DONE],
                                scale=1.01)
    assert [r["passed"] for r in _hil_rows(session)] == [False]
    assert any("the last check FAILED" in e for e in _errors(journal))
    assert len(journal["steady_states"]) == 1
    assert session.overall_verdict.startswith("OVERALL VERDICT: FAIL")


def test_goal_may_ask_to_continue_after_a_failure(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _solve(_close((8, 21))), _hil(), DONE],
                                scale=1.01, goal="close 8-21 and compare; continue even if a comparison fails")
    assert [r["passed"] for r in _hil_rows(session)] == [False, False]
    assert _errors(journal) == []


def test_missing_check_refused(tmp_path):
    no_check = {"action": "hil_step", "start_from": "latest", "reasoning": "r"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), no_check, _hil(), DONE])
    assert any('"check" is missing' in e for e in _errors(journal))
    assert len(_hil_rows(session)) == 1


def test_stop_decision_with_a_tool_action_refused(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(decision="stop"), DONE])
    assert any('your decision is "stop"' in e for e in _errors(journal))


def test_transcript_and_analysis_written(tmp_path):
    session, journal, backend, _ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(check="S0 fine"), DONE])
    lines = (session.session_dir / "llm_transcript.jsonl").read_text().splitlines()
    calls = [json.loads(l) for l in lines]
    assert [c["phase"] for c in calls] == ["objective_extraction", "search", "search", "search", "analysis"]
    assert "LATEST RESULT" in calls[2]["user_prompt"] and '"check"' not in calls[1]["user_prompt"]
    assert 'containing "action", "check", "decision"' in calls[2]["user_prompt"]
    assert 'containing "action", "reasoning"' in calls[1]["user_prompt"]
    assert (session.session_dir / "analysis.txt").read_text().strip() == ANALYSIS
    assert "S0 fine" in calls[-1]["user_prompt"]          # the agent's checks go to the analysis
    log = session.log_path.read_text()
    assert "Agent check: S0 fine" in log and "=== Final analysis" in log
    assert journal["analysis"] == ANALYSIS


def test_near_limit_is_flagged(tmp_path):
    # the fake HIL reads 0.046 % high everywhere: 92 % of the 0.05 % limit, still PASS
    session, journal, backend, _ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), DONE], scale=1.00046)
    hil = next(e for e in journal["entries"] if e["action"] == "hil_step")
    assert hil["verdict"].startswith("VERDICT HIL S0: PASS")
    assert any("magnitude" in a and "of its limit" in a for a in hil["attention"])
    assert "Near the limit in the latest result" in backend.prompts[3][1]


# --------------------------------------------------------------------------- order and repeats

def test_hil_before_exago_refused(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _hil(), _solve(), _hil(), DONE])
    assert any("no steady state to run" in e for e in _errors(journal))
    assert len(_hil_rows(session)) == 1


def test_same_state_twice_on_hil_refused(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _hil(), DONE])
    assert any("already run on the HIL" in e for e in _errors(journal))


def test_solve_without_commands_twice_refused(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _solve(), DONE])
    assert any("no commands" in e for e in _errors(journal))


def test_complete_before_pending_hil_refused_once(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _solve(_close((8, 21))), DONE,
                                           _hil(), DONE])
    assert any("complete refused" in e for e in _errors(journal))
    assert len(_hil_rows(session)) == 2 and session.termination_reason == "completed"
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _solve(_close((8, 21))), DONE, DONE])
    assert session.termination_reason == "completed"
    assert "has not been run on the HIL" in session.overall_verdict


def test_no_pending_when_hil_never_used(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _solve(_close((8, 21))), DONE])
    assert _errors(journal) == [] and session.termination_reason == "completed"
    assert session.overall_verdict.startswith("OVERALL VERDICT: NO CHECKS")


# --------------------------------------------------------------------------- what the HIL cannot do

def test_load_change_skipped_on_hil(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(),
                                           _solve({"action": "scale_all_loads", "factor": 1.05}), _hil(), DONE])
    row = _hil_rows(session)[0]
    assert "SKIPPED" in row["result"] and "cannot make" in row["result"]


def test_opening_a_tie_skipped_on_hil(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _solve(_close((8, 21))),
                                           _solve(_close((8, 21), 0)), _hil(), DONE])
    assert "opens tie 8-21" in _hil_rows(session)[0]["result"]


def test_two_ties_at_once_closed_live(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _solve(_close((8, 21)), _close((9, 15))),
                                           _hil(), DONE])
    # two ties in one ExaGO step: the HIL closes both live, one after the other
    row = _hil_rows(session)[0]
    assert row["passed"] is True


# --------------------------------------------------------------------------- best

def test_best_follows_the_goals_objective(tmp_path):
    replies = [MIN_LOSSES, _solve(), _solve(_close((8, 21))), _solve(_close((9, 15))),
               _hil(start="best"), DONE]
    session, journal, *_ = _run(tmp_path, replies, goal="minimize losses")
    losses = {f"S{k}": solved_metrics(session.session_dir / "exago" / f"S{k}" / "pflowout.m")["active_losses_mw"]
              for k in range(3)}
    best = min(losses, key=losses.get)
    assert _hil_rows(session)[0]["state"] == best
    assert session.objectives[0]["name"] == "active_losses_mw"


def test_best_without_objective_refused(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(start="best"), DONE])
    assert any("states no objective" in e for e in _errors(journal))


def test_exago_from_best(tmp_path):
    replies = [MIN_LOSSES, _solve(), _solve(_close((8, 21))), _solve(_close((9, 15)), start="best"), DONE]
    session, journal, *_ = _run(tmp_path, replies, goal="minimize losses", hil=False)
    states = journal["steady_states"]
    assert states[2]["parent"] == 1          # S1 (8-21) has lower losses than S0


# --------------------------------------------------------------------------- stops and limits

def test_hil_refusal_stops_study(tmp_path):
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), _solve(_close((8, 21))), _hil(),
                                           _solve(_close((9, 15))), DONE], accept=False)
    assert "study stopped" in session.overall_verdict
    assert any("the study was stopped" in e for e in _errors(journal))


def test_non_convergence_stops_study(tmp_path):
    fake = tmp_path / "pflow"
    fake.write_text("#!/bin/sh\necho 'PFLOW DID NOT CONVERGE'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    session, journal, backend, device = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), DONE], pflow=fake)
    assert journal["steady_states"] == []
    assert "open" not in device.log
    assert any("the study was stopped" in e for e in _errors(journal))


def test_turn_limit(tmp_path):
    session, *_ = _run(tmp_path, full_script(), max_turns=4)
    assert session.termination_reason == "max_turns"
    assert len(_hil_rows(session)) == 2


# --------------------------------------------------------------------------- tools

def test_hil_tool_not_offered_without_model(tmp_path):
    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "w")})
    adapters = [ExagoAdapter(CASE, {"pflow": PFLOW}),
                HilAdapter(tmp_path / "missing.tse", HilSettings.from_config(cfg))]
    backend = ScriptedBackend([NO_OBJECTIVES, _solve(), _hil(), DONE])
    session = RouterAgentLoop(cfg, adapters, backend=backend, quiet=True).run("solve")
    assert session.tools == ["exago"]
    journal = json.loads(session.journal_path.read_text())
    assert any("Unknown action 'hil_step'" in e for e in _errors(journal))


def test_no_exago_case_not_run(tmp_path):
    session, journal, backend, _ = _run(tmp_path, [NO_OBJECTIVES], case=tmp_path / "missing.m")
    assert session.termination_reason == "not_run"
    assert backend.prompts == []


def test_gridkit_screen_from_latest_state(tmp_path):
    gk = FakeGridkitLoop()
    screen = {"action": "gridkit_fault_screen", "start_from": "latest", "all_buses": True,
              "description": "all buses", "check": "c", "decision": "proceed", "reasoning": "r-gk"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _solve(_close((8, 21))), screen, DONE],
                                hil=False, gridkit=gk)
    assert ("use", "S1", "pflowout.m") in gk.calls
    sent = next(c[1] for c in gk.calls if c[0] == "screen")
    sent = {k: v for k, v in sent.items() if k not in ("check", "decision")}
    assert sent["action"] == "fault_screen" and sent["start_from"] == "latest_steady_state"
    assert sent["all_buses"] is True
    assert session.overall_verdict.startswith("OVERALL VERDICT: PASS — all 1 checks")
    assert gk.calls[-1] == ("export",)


def test_gridkit_unusable_state_not_run(tmp_path):
    gk = FakeGridkitLoop(ok=False)
    screen = {"action": "gridkit_fault_screen", "start_from": "latest", "all_buses": True,
              "check": "c", "decision": "proceed"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), screen, DONE], hil=False, gridkit=gk)
    assert "NOT RUN" in session.rows[-1]["result"]
    assert not any(c[0] == "screen" for c in gk.calls)


def test_all_three_tools_in_one_prompt(tmp_path):
    gk = FakeGridkitLoop()
    screen = {"action": "gridkit_fault_screen", "start_from": "latest", "buses": [18],
              "check": "c", "decision": "proceed"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), screen, _solve(_close((8, 21))),
                                           _hil(), screen, DONE], gridkit=gk)
    assert [r["tool"] for r in session.rows] == ["ExaGO", "HIL", "GridKit", "ExaGO", "HIL", "GridKit"]
    assert session.overall_verdict.startswith("OVERALL VERDICT: PASS — all 4 checks")
    prompt = (session.session_dir / "system_prompt.txt").read_text()
    assert "gridkit_fault_screen" in prompt and "hil_step" in prompt and "exago_solve" in prompt


def test_reply_without_action_gets_a_specific_message(tmp_path):
    no_action = {"check": "c", "decision": "proceed", "reasoning": "r"}
    session, journal, backend, _ = _run(tmp_path, [NO_OBJECTIVES, _solve(), _hil(), no_action, DONE])
    msg = next(e for e in _errors(journal) if "no \"action\"" in e)
    assert '"action": "exago_solve"' in msg and '"check"' in msg


def test_missing_check_message_shows_the_shape(tmp_path):
    no_check = {"action": "hil_step", "start_from": "latest", "reasoning": "r"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), no_check, _hil(), DONE])
    msg = next(e for e in _errors(journal) if "check" in e)
    assert '{"action": "hil_step", "check":' in msg


def test_three_refusals_in_a_row_stop_the_session(tmp_path):
    no_action = {"check": "c", "decision": "proceed", "reasoning": "r"}
    session, journal, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), no_action, no_action, no_action, DONE])
    assert session.termination_reason == "format_errors"
    assert len([e for e in journal["entries"] if e["action"] == "error"]) == 3


def test_every_action_template_carries_check_and_decision(tmp_path):
    session, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), DONE], gridkit=FakeGridkitLoop())
    prompt = (session.session_dir / "system_prompt.txt").read_text()
    for act in ("exago_solve", "hil_step", "gridkit_fault_screen", "complete"):
        i = prompt.index(f'{{"action": "{act}"')
        assert '"check"' in prompt[i:i + 200] and '"decision"' in prompt[i:i + 200], act
        assert "not needed" not in prompt[i:i + 200], act


def test_turn_prompt_ends_with_reply_shape_after_first_result(tmp_path):
    session, *_ = _run(tmp_path, [NO_OBJECTIVES, _solve(), DONE])
    calls = [json.loads(l) for l in (session.session_dir / "llm_transcript.jsonl").read_text().splitlines()]
    search = [c for c in calls if c["phase"] == "search"]
    assert "Shape of the reply" not in search[0]["user_prompt"]
    last = search[1]["user_prompt"].splitlines()[-1]
    assert last.startswith("Shape of the reply") and '"check"' in last and '"decision"' in last
