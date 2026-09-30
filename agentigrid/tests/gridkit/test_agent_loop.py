"""GridKit agent loop with a scripted backend and real GridKit runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import load_config
from agentigrid.gridkit_engine.agent_loop import GridkitAgentLoop
from agentigrid.gridkit_engine.settings import GridkitSettings

ROOT = Path(__file__).resolve().parents[2]
BINARY_DIR = ROOT / "applications" / "gridkit"
IEEE39 = ROOT / "data" / "gridkit" / "examples" / "IEEE39.case.json"

pytestmark = pytest.mark.skipif(
    not ((BINARY_DIR / "DynamicSimulation").exists() and IEEE39.exists()),
    reason="GridKit DynamicSimulation or IEEE39 case not linked",
)


class ScriptedBackend(LLMBackend):
    """Returns the given replies in order and keeps the prompts it was sent."""

    def __init__(self, replies: list[dict | str]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system_prompt, user_prompt, temperature=None):
        self.prompts.append((system_prompt, user_prompt))
        reply = self.replies.pop(0)
        is_json = isinstance(reply, dict)
        return LLMResponse(
            raw_text=json.dumps(reply) if is_json else reply,
            json_data=reply if is_json else None,
            json_error=None if is_json else "not JSON",
            model="scripted", backend="scripted", prompt_tokens=10, completion_tokens=5,
        )

    def name(self):
        return "scripted"

    def supports_json_mode(self):
        return True


SCREEN = {"action": "fault_screen", "description": "POI 16", "poi": 16, "hops": 0,
          "tmax_s": 4.0, "checks": ["angle_stability", "voltage_recovery"]}


def _loop(tmp_path, replies):
    cfg = load_config(None, cli_overrides={
        "output.workdir": str(tmp_path / "workdir"), "output.logs_dir": str(tmp_path / "logs"),
    })
    settings = GridkitSettings.from_config(cfg, binary_dir=BINARY_DIR, max_iterations=6)
    backend = ScriptedBackend(replies)
    return GridkitAgentLoop(cfg, settings=settings, backend=backend, quiet=True), backend


def test_session_end_to_end(tmp_path):
    loop, backend = _loop(tmp_path, [
        {"action": "fault_screen", "poi": 999},                  # rejected: no such bus
        "I think we should fault bus 16.",                       # not JSON
        SCREEN,                                                  # runs GridKit
        SCREEN,                                                  # identical: served from cache
        {"action": "complete", "summary": "VERDICT: ...", "details": "none"},
    ])
    session = loop.run(IEEE39, "Fault bus 16 and tell me if the system survives.")

    assert session.termination_reason == "completed"
    assert session.final_answer["summary"] == "VERDICT: ..."
    assert (session.session_dir / "element_map.json").exists()

    journal = json.loads(session.journal_path.read_text())
    actions = [e["action"] for e in journal["entries"]]
    assert actions == ["base_case", "error", "error", "fault_screen", "fault_screen", "complete"]
    assert "steady state: pass" in journal["entries"][0]["verdict"].lower()
    assert "POI bus 999 is not in the case" in journal["entries"][1]["description"]

    screen = journal["entries"][3]
    assert screen["verdict"].startswith("VERDICT: IEEE39 around POI bus 16")
    assert "passes all 1 bus faults" in screen["verdict"]
    assert screen["rows"][0]["bus"] == 16 and screen["rows"][0]["solved"]
    assert (Path(screen["run_dir"]) / "bus_16" / "output.csv").exists()
    assert (Path(screen["run_dir"]) / "bus_16" / "study.solver.json").exists()
    assert journal["entries"][4]["description"].endswith("(cached)")

    # What the LLM saw
    system_prompt = backend.prompts[0][0]
    assert "fault_screen" in system_prompt and "SKIPPED" in system_prompt
    assert "Steady state: pass" in system_prompt
    assert "POI bus 999 is not in the case" in backend.prompts[1][1]
    assert "not a valid JSON object" in backend.prompts[2][1]
    assert "already ran in iteration 3" in backend.prompts[4][1]
    assert "D3: high-speed reclosing" in backend.prompts[4][1]


def test_stops_after_three_non_json_replies(tmp_path):
    loop, _ = _loop(tmp_path, ["a", "b", "c"])
    session = loop.run(IEEE39, "anything")
    assert session.termination_reason == "parse_failures"
    assert session.final_answer is None


def test_stops_at_first_llm_api_error(tmp_path):
    loop, backend = _loop(tmp_path, ['Anthropic API error: "Could not resolve authentication method."'])
    session = loop.run(IEEE39, "anything")
    assert session.termination_reason == "llm_error"
    assert "authentication" in session.error
    assert len(backend.prompts) == 1
