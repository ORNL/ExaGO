"""Every LLM call of a GridKit session is counted (all under ``search``)."""

from __future__ import annotations

import json

from agentigrid.backends.base import LLMResponse
from tests.gridkit.test_agent_loop import IEEE39, ScriptedBackend, _loop, pytestmark  # noqa: F401


def _ledger_matches_backend(session, backend) -> None:
    ledger = session.token_ledger
    assert len(ledger) == len(backend.prompts)
    assert set(ledger.by_phase) == {"search"}
    assert session.total_prompt_tokens == ledger.total_prompt
    assert session.total_completion_tokens == ledger.total_completion
    assert ledger.total == sum(
        (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in ledger.calls
    )


def test_parse_failures_counted(tmp_path):
    loop, backend = _loop(tmp_path, ["a", "b", "c"])
    session = loop.run(IEEE39, "anything")
    assert session.termination_reason == "parse_failures"
    _ledger_matches_backend(session, backend)
    assert session.token_ledger.total == 3 * 15
    assert [c["iteration"] for c in session.token_ledger.calls] == [1, 2, 3]


def test_journal_block_and_missing_usage(tmp_path):
    loop, backend = _loop(tmp_path, [])

    replies = iter(["not json", {"action": "complete", "summary": "VERDICT: ok", "details": ""}])

    def complete(system_prompt, user_prompt, temperature=None):
        backend.prompts.append((system_prompt, user_prompt))
        reply = next(replies)
        is_json = isinstance(reply, dict)
        return LLMResponse(raw_text=json.dumps(reply) if is_json else reply,
                           json_data=reply if is_json else None,
                           json_error=None if is_json else "not JSON",
                           model="scripted", backend="scripted",
                           prompt_tokens=None if not is_json else 20,      # usage missing once
                           completion_tokens=None if not is_json else 8)

    backend.complete = complete
    session = loop.run(IEEE39, "anything")
    assert session.termination_reason == "completed"
    _ledger_matches_backend(session, backend)
    assert session.token_ledger.calls_without_usage == 1

    journal = json.loads(session.journal_path.read_text())
    assert journal["token_usage"] == session.token_ledger.to_dict()
    assert journal["token_usage"]["calls"] == 2
    assert journal["token_usage"]["total_tokens"] == 28
