"""HIL agent loop with a scripted backend and a fake HIL device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import load_config
from agentigrid.hil_engine.agent_loop import HilAgentLoop
from agentigrid.hil_engine.network import load_network
from agentigrid.hil_engine.settings import HilSettings

from tests.hil.helpers import FakeHil, exago_workdir, network

STUDY_MODEL = Path("/home/shilongphone/Study_loop/models/ieee33/ieee 33 bus system.tse")


class ScriptedBackend(LLMBackend):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system_prompt, user_prompt, temperature=None):
        self.prompts.append((system_prompt, user_prompt))
        reply = self.replies.pop(0)
        is_json = isinstance(reply, dict)
        return LLMResponse(raw_text=json.dumps(reply) if is_json else reply,
                           json_data=reply if is_json else None,
                           json_error=None if is_json else "not JSON",
                           model="scripted", backend="scripted", prompt_tokens=10, completion_tokens=5)

    def name(self):
        return "scripted"

    def supports_json_mode(self):
        return True


DONE = {"action": "complete", "summary": "s", "details": "d"}


def _run(tmp_path, replies, goal="close lin 2-4", device=None, exago=True, **wd):
    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "workdir"),
                                           "output.logs_dir": str(tmp_path / "logs")})
    settings = HilSettings.from_config(cfg, max_iterations=5,
                                       exago_workdir=exago_workdir(tmp_path, **wd) if exago else tmp_path / "none")
    backend = ScriptedBackend(replies)
    device = device or FakeHil()
    session = HilAgentLoop(cfg, settings=settings, backend=backend, device=device, quiet=True).run(
        "small.tse", goal, network=network())
    journal = json.loads(session.journal_path.read_text())
    return session, journal, backend, device


def _entries(journal, action):
    return [e for e in journal["entries"] if e["action"] == action]


def test_close_and_compare_with_latest_steady_state(tmp_path):
    session, journal, backend, device = _run(tmp_path, [
        {"action": "close_tie", "tie": "lin 2-4", "start_from": "latest_steady_state"}, DONE],
        goal="Starting from the most recent steady state, close lin 2-4 and compare")
    assert session.termination_reason == "completed"
    study = _entries(journal, "close_tie")[0]
    assert study["verdict"].startswith("VERDICT: PASS")
    assert [c["passed"] for c in study["comparisons"]] == [True, True]   # tie closed, radial start
    assert study["commands"][1]["scada_input"] == "Tie_line_1" and study["commands"][1]["accepted"]
    # every study starts radial, the tie is closed live, the rest state is set at the end
    assert ("restart", []) in device.log and ("close_tie", (2, 4)) in device.log
    assert device.log[-1] == "shutdown"
    system = backend.prompts[0][0]
    assert "close lin 12-22" in system and "Tie_sw_1" in system and "STARTING STATE" in system


def test_hil_off_by_more_than_the_limit_fails(tmp_path):
    _, journal, _, _ = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4], "start_from": "latest_steady_state"}, DONE],
        device=FakeHil(scale=1.001))          # 0.1 % > 0.05 %
    assert _entries(journal, "close_tie")[0]["verdict"].startswith("VERDICT: FAIL")


def test_goal_wants_steady_state_but_action_does_not(tmp_path):
    _, journal, backend, _ = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4]},
        {"action": "close_tie", "tie": [2, 4], "start_from": "latest_steady_state"}, DONE],
        goal="From the latest steady state, close tie 2-4")
    assert "latest_steady_state" in _entries(journal, "error")[0]["description"]
    assert "ERROR IN YOUR LAST REPLY" in backend.prompts[1][1]
    assert _entries(journal, "close_tie")[0]["verdict"].startswith("VERDICT: PASS")


def test_exago_network_differs_from_the_study(tmp_path):
    _, journal, _, device = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4], "start_from": "latest_steady_state"}, DONE],
        tie_status=0)                         # ExaGO did not close the tie
    assert "NOT RUN" in _entries(journal, "close_tie")[0]["verdict"]
    assert not any(isinstance(x, tuple) and x[0] == "close_tie" for x in device.log)


def test_exago_line_out_is_skipped(tmp_path):
    _, journal, _, _ = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4], "start_from": "latest_steady_state"}, DONE],
        line_23_status=0)
    assert "SKIPPED" in _entries(journal, "close_tie")[0]["verdict"]


def test_no_exago_journal(tmp_path):
    _, journal, _, _ = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4], "start_from": "latest_steady_state"}, DONE], exago=False)
    assert "cannot be used" in _entries(journal, "close_tie")[0]["verdict"]


def test_radial_only_and_identical_repeat(tmp_path):
    _, journal, _, device = _run(tmp_path, [
        {"action": "close_tie", "tie": [2, 4]}, {"action": "close_tie", "tie": [4, 2]}, DONE],
        goal="close the tie between 4 and 2")
    first, second = _entries(journal, "close_tie")
    assert first["verdict"].startswith("VERDICT: NOT COMPARED")
    assert second["result_text"].startswith("IDENTICAL")
    assert sum(1 for x in device.log if isinstance(x, tuple) and x[0] == "close_tie") == 1


def test_open_and_non_tie_are_refused(tmp_path):
    _, journal, _, device = _run(tmp_path, [
        {"action": "open_tie", "tie": [2, 4]}, {"action": "close_tie", "tie": [2, 3]}, DONE],
        goal="open 2-4, then close 2-3")
    assert "SKIPPED" in _entries(journal, "refused")[0]["verdict"]
    assert "ordinary line" in _entries(journal, "close_tie")[0]["verdict"]
    assert not any(isinstance(x, tuple) and x[0] == "close_tie" for x in device.log)


def test_switch_not_accepted(tmp_path):
    _, journal, _, _ = _run(tmp_path, [{"action": "close_tie", "tie": [2, 4]}, DONE],
                            device=FakeHil(accept=False))
    assert "did not accept" in _entries(journal, "close_tie")[0]["verdict"]


@pytest.mark.skipif(not STUDY_MODEL.exists(), reason="IEEE 33 Typhoon model copy not present")
def test_ieee33_model_ties():
    net = load_network(STUDY_MODEL)
    assert len(net.buses) == 33 and net.feeder_switch == "Feeder sw"
    assert {(t.name, t.scada_input) for t in net.ties} == {
        ("8-21", "Tie_line_3"), ("9-15", "Tie_line_2"), ("12-22", "Tie_line_5"),
        ("18-33", "Tie_line_4"), ("25-29", "Tie_line_1")}
    assert net.tie(22, 12).line == "Line 735"
