"""Unit tests for agentigrid.token_usage.TokenLedger."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentigrid.backends.base import LLMResponse
from agentigrid.token_usage import PHASES, TokenLedger


def _resp(prompt, completion, model="m", backend="b") -> LLMResponse:
    return LLMResponse(raw_text="", json_data=None, json_error=None, model=model,
                       backend=backend, prompt_tokens=prompt, completion_tokens=completion)


def _ledger() -> TokenLedger:
    ledger = TokenLedger()
    ledger.record("objective_extraction", _resp(50, 20), 0)
    ledger.record("search", _resp(100, 40), 1)
    ledger.record("search", _resp(120, 60), 2)
    ledger.record("analysis", _resp(300, 200))
    return ledger


def test_totals_and_phases():
    ledger = _ledger()
    assert len(ledger) == 4
    assert ledger.total_prompt == 570
    assert ledger.total_completion == 320
    assert ledger.total == 890
    assert list(ledger.by_phase) == ["objective_extraction", "search", "analysis"]
    assert ledger.by_phase["search"] == {
        "calls": 2, "prompt_tokens": 220, "completion_tokens": 100,
        "total_tokens": 320, "calls_without_usage": 0,
        "prompt_chars": 0, "completion_chars": 0,
    }
    assert sum(p["total_tokens"] for p in ledger.by_phase.values()) == ledger.total


def test_none_usage_kept_per_call_and_counted():
    ledger = TokenLedger()
    ledger.record("search", _resp(None, None), 1)       # e.g. API error
    ledger.record("search", _resp(10, None), 2)         # partial usage
    ledger.record("search", _resp(10, 5), 3)
    assert ledger.calls[0]["prompt_tokens"] is None
    assert ledger.calls[1]["completion_tokens"] is None
    assert ledger.total_prompt == 20 and ledger.total_completion == 5
    assert ledger.calls_without_usage == 2
    assert ledger.by_phase["search"]["calls_without_usage"] == 2
    assert "2 calls without usage data" in ledger.summary_text()


def test_partial_response_objects_never_fail():
    ledger = TokenLedger()
    ledger.record("analysis", SimpleNamespace(raw_text="x"))
    ledger.record("analysis", MagicMock())
    assert len(ledger) == 2
    assert ledger.calls_without_usage == 2
    json.dumps(ledger.to_dict())                        # no mock objects leak into the export


def test_unknown_phase_rejected():
    with pytest.raises(ValueError):
        TokenLedger().record("planning", _resp(1, 1))


def test_to_dict_round_trip():
    ledger = _ledger()
    ledger.record("search", _resp(None, None), 3)
    data = json.loads(json.dumps(ledger.to_dict()))
    assert data["calls"] == 5
    assert data["total_tokens"] == 890
    assert data["calls_without_usage"] == 1
    assert data["by_phase"]["search"]["calls"] == 3
    back = TokenLedger.from_dict(data)
    assert back.calls == ledger.calls
    assert back.to_dict() == data


def test_from_dict_empty_and_legacy():
    assert len(TokenLedger.from_dict(None)) == 0
    assert len(TokenLedger.from_dict({})) == 0
    legacy = TokenLedger.from_legacy_totals(1000, 500)
    assert legacy.total == 1500
    assert legacy.by_phase["search"]["calls"] == 1
    assert legacy.calls[0]["carried_over"] is True
    assert len(TokenLedger.from_legacy_totals(0, 0)) == 0
    # carried_over survives a round trip
    assert TokenLedger.from_dict(legacy.to_dict()).calls[0]["carried_over"] is True


def test_summary_text_format():
    text = _ledger().summary_text()
    assert text == (
        "Tokens: 890 total (570 prompt + 320 completion) in 4 LLM calls; "
        "objective extraction 70, search 320, analysis 500"
    )
    assert TokenLedger().summary_text() == "Tokens: 0 total (0 prompt + 0 completion) in 0 LLM calls"


def test_phase_names():
    assert PHASES == ("objective_extraction", "search", "analysis", "steering", "report", "other")


def test_char_totals_equal_sum_over_calls():
    ledger = TokenLedger()
    r1 = _resp(10, 5); r1.raw_text = "abcd"
    r2 = _resp(20, 6); r2.raw_text = "xy"
    ledger.record("search", r1, 1, prompt_chars=120)
    ledger.record("search", r2, 2, prompt_chars=80)
    ledger.record("analysis", _resp(30, 7), prompt_chars=None)    # call site without a length
    ledger.record("other", SimpleNamespace())                        # no raw_text at all

    assert [c["prompt_chars"] for c in ledger.calls] == [120, 80, None, None]
    assert [c["completion_chars"] for c in ledger.calls] == [4, 2, 0, None]
    assert ledger.total_prompt_chars == sum(c["prompt_chars"] or 0 for c in ledger.calls) == 200
    assert ledger.total_completion_chars == sum(c["completion_chars"] or 0 for c in ledger.calls) == 6
    assert ledger.by_phase["search"]["prompt_chars"] == 200
    assert ledger.by_phase["search"]["completion_chars"] == 6
    assert sum(a["prompt_chars"] for a in ledger.by_phase.values()) == ledger.total_prompt_chars
    data = ledger.to_dict()
    assert data["prompt_chars"] == 200 and data["completion_chars"] == 6
    assert TokenLedger.from_dict(data).calls == ledger.calls


def test_old_dict_without_char_fields_loads():
    old = {"per_call": [{"phase": "search", "iteration": 1, "model": "m", "backend": "b",
                         "prompt_tokens": 10, "completion_tokens": 5}]}
    ledger = TokenLedger.from_dict(old)
    assert ledger.calls[0]["prompt_chars"] is None
    assert ledger.calls[0]["completion_chars"] is None
    assert ledger.total == 15 and ledger.total_prompt_chars == 0
    assert ledger.to_dict()["by_phase"]["search"]["prompt_chars"] == 0
