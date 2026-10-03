"""Iteration count shown to users: agent iterations plus the base case."""

from __future__ import annotations

from agentigrid.engine.journal import SearchJournal, format_iteration_count

from tests.journal.test_journal import _make_entry


def _journal(iterations, discarded=()):
    j = SearchJournal()
    for i in iterations:
        j.add_entry(_make_entry(i))
    j.discarded_actions = [{"iteration": i, "kind": "rejected"} for i in discarded]
    return j


def test_base_case_is_not_counted_as_an_agent_iteration():
    stats = _journal([0, 1, 2]).summary_stats()
    assert stats["total_iterations"] == 3          # unchanged: journal entries
    assert stats["llm_iterations"] == 2
    assert stats["has_base_case"] is True
    assert format_iteration_count(stats) == "2 + base case"


def test_rejected_iterations_still_count():
    # max 4: iteration 3 was rejected (no journal entry, kept in discarded_actions)
    stats = _journal([0, 1, 2, 4], discarded=[3]).summary_stats()
    assert stats["total_iterations"] == 4
    assert stats["llm_iterations"] == 4
    assert format_iteration_count(stats) == "4 + base case"


def test_several_entries_for_one_iteration_count_once():
    stats = _journal([0, 1, 1, 2]).summary_stats()
    assert stats["llm_iterations"] == 2


def test_no_base_case():
    stats = _journal([1, 2]).summary_stats()
    assert format_iteration_count(stats) == "2"


def test_old_stats_without_new_keys_fall_back():
    assert format_iteration_count({"total_iterations": 5}) == "5"


def test_empty_journal():
    stats = SearchJournal().summary_stats()
    assert stats["llm_iterations"] == 0 and stats["has_base_case"] is False
    assert format_iteration_count(stats) == "0"
