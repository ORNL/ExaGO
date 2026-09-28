"""Tests for interactive steering in AgentLoopController."""

from __future__ import annotations

import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.prompts.user_prompt import build_user_prompt


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_controller() -> AgentLoopController:
    """Create a minimal AgentLoopController with a mocked config and backend."""
    cfg = MagicMock()
    cfg.llm.backend = "openai"
    cfg.llm.model = "gpt-4o"
    with patch("agentigrid.engine.agent_loop.create_backend"), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor"):
        ctrl = AgentLoopController(cfg)
    return ctrl


# ── Steering Queue ────────────────────────────────────────────────────────────

class TestSteeringQueue:

    def test_inject_directive_augment(self):
        ctrl = _make_controller()
        ctrl.inject_steering("Focus on voltage stability", mode="augment")
        item = ctrl._steering_queue.get_nowait()
        assert item["directive"] == "Focus on voltage stability"
        assert item["mode"] == "augment"

    def test_inject_directive_replace(self):
        ctrl = _make_controller()
        ctrl.inject_steering("Minimize cost only", mode="replace")
        item = ctrl._steering_queue.get_nowait()
        assert item["mode"] == "replace"

    def test_inject_default_mode_is_augment(self):
        ctrl = _make_controller()
        ctrl.inject_steering("some directive")
        item = ctrl._steering_queue.get_nowait()
        assert item["mode"] == "augment"

    def test_multiple_directives_queued(self):
        ctrl = _make_controller()
        ctrl.inject_steering("First directive")
        ctrl.inject_steering("Second directive")
        items = []
        while not ctrl._steering_queue.empty():
            items.append(ctrl._steering_queue.get_nowait())
        assert len(items) == 2
        assert items[0]["directive"] == "First directive"
        assert items[1]["directive"] == "Second directive"


# ── Pause / Resume ────────────────────────────────────────────────────────────

class TestPauseResume:

    def test_not_paused_initially(self):
        ctrl = _make_controller()
        assert not ctrl.is_paused()

    def test_pause_sets_paused(self):
        ctrl = _make_controller()
        ctrl.pause()
        assert ctrl.is_paused()

    def test_resume_clears_paused(self):
        ctrl = _make_controller()
        ctrl.pause()
        ctrl.resume()
        assert not ctrl.is_paused()

    def test_pause_callback_called(self):
        calls = []
        ctrl = _make_controller()
        ctrl._on_pause_state = lambda p: calls.append(p)
        ctrl.pause()
        assert calls == [True]
        ctrl.resume()
        assert calls == [True, False]

    def test_pause_event_blocks_wait(self):
        ctrl = _make_controller()
        ctrl.pause()
        # Start a thread that waits on the event
        unblocked = threading.Event()
        def waiter():
            ctrl._pause_event.wait()
            unblocked.set()
        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        assert not unblocked.is_set()
        ctrl.resume()
        t.join(timeout=1.0)
        assert unblocked.is_set()


# ── Steering History ──────────────────────────────────────────────────────────

class TestSteeringHistory:

    def test_history_empty_initially(self):
        ctrl = _make_controller()
        assert ctrl.steering_history == []

    def test_history_populated_after_drain(self):
        ctrl = _make_controller()
        ctrl.inject_steering("directive 1", mode="augment")
        ctrl.inject_steering("directive 2", mode="replace")
        # Simulate drain at iteration boundary
        ctrl._active_steering_directives.clear()
        while not ctrl._steering_queue.empty():
            item = ctrl._steering_queue.get_nowait()
            if item["mode"] == "replace":
                ctrl._active_steering_directives.clear()
            ctrl._active_steering_directives.append(item)
            ctrl._steering_history.append({"iteration": 1, **item})
        assert len(ctrl.steering_history) == 2
        assert ctrl.steering_history[0]["directive"] == "directive 1"

    def test_replace_clears_active_directives(self):
        ctrl = _make_controller()
        ctrl.inject_steering("first augment", mode="augment")
        ctrl.inject_steering("replace everything", mode="replace")
        # Drain
        while not ctrl._steering_queue.empty():
            item = ctrl._steering_queue.get_nowait()
            if item["mode"] == "replace":
                ctrl._active_steering_directives.clear()
            ctrl._active_steering_directives.append(item)
            ctrl._steering_history.append({"iteration": 1, **item})
        # Only the replace directive should remain active
        assert len(ctrl._active_steering_directives) == 1
        assert ctrl._active_steering_directives[0]["mode"] == "replace"


# ── Prompt Integration ────────────────────────────────────────────────────────

class TestSteeringInPrompt:

    def test_steering_section_included(self):
        directives = [
            {"directive": "Focus on voltage", "mode": "augment"},
            {"directive": "Ignore cost", "mode": "replace"},
        ]
        prompt = build_user_prompt(
            goal="Minimize cost",
            journal_text=None,
            results_text=None,
            steering_directives=directives,
        )
        assert "Operator Directives" in prompt
        assert "[AUGMENT]" in prompt
        assert "[REPLACE]" in prompt
        assert "Focus on voltage" in prompt
        assert "Ignore cost" in prompt

    def test_no_steering_section_when_none(self):
        prompt = build_user_prompt(
            goal="Minimize cost",
            journal_text=None,
            results_text=None,
            steering_directives=None,
        )
        assert "Operator Directives" not in prompt

    def test_no_steering_section_when_empty(self):
        prompt = build_user_prompt(
            goal="Minimize cost",
            journal_text=None,
            results_text=None,
            steering_directives=[],
        )
        assert "Operator Directives" not in prompt

    def test_steering_section_position(self):
        """Steering section should appear after journal/results but before the action request."""
        directives = [{"directive": "test", "mode": "augment"}]
        prompt = build_user_prompt(
            goal="goal",
            journal_text="journal",
            results_text="results",
            steering_directives=directives,
        )
        op_idx = prompt.index("Operator Directives")
        action_idx = prompt.index("decide your next action")
        assert op_idx < action_idx

    def test_augment_semantics_note(self):
        directives = [{"directive": "test", "mode": "augment"}]
        prompt = build_user_prompt(
            goal="goal",
            journal_text=None,
            results_text=None,
            steering_directives=directives,
        )
        assert "alongside the original goal" in prompt

    def test_replace_semantics_note(self):
        directives = [{"directive": "test", "mode": "replace"}]
        prompt = build_user_prompt(
            goal="goal",
            journal_text=None,
            results_text=None,
            steering_directives=directives,
        )
        assert "supersede the original goal" in prompt
