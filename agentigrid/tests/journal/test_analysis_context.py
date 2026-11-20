"""The post-search analysis sees the latest results and the final findings.

The journal digest has no per-bus values: in the AgentiGrid 4 pilot the analysis
said the bus-70030 voltage "cannot be confirmed" although the search had reported
it. The prompt now carries the last results summary and the findings too.
"""

from __future__ import annotations

from agentigrid.exago_engine.goal_classifier import (
    MAX_LATEST_RESULTS_CHARS,
    build_classification_prompts,
)

STATS = {"total_iterations": 2, "feasible_count": 2, "infeasible_count": 0,
         "best_objective": 2488236.09, "best_iteration": 1}


def _user(**kw):
    _sys, user = build_classification_prompts(
        goal="Add load at bus 70030 and report its voltage", termination_reason="complete",
        stats=STATS, journal_formatted="digest", total_tokens=0, **kw)
    return user


def test_old_prompt_unchanged_without_context():
    user = _user()
    assert "=== Latest Results" not in user
    assert "=== Final Findings" not in user


def test_latest_results_and_findings_included():
    user = _user(latest_results_text="Bus 70030: Vm 0.937255 pu",
                 final_findings={"summary": "bus 70030 voltage is 0.937 pu"})
    assert "=== Latest Results" in user and "Bus 70030: Vm 0.937255 pu" in user
    assert "=== Final Findings" in user and "bus 70030 voltage is 0.937 pu" in user
    # both come after the digest and before the instructions
    assert user.index("digest") < user.index("=== Final Findings") < user.index("Please provide:")


def test_long_results_cut():
    user = _user(latest_results_text="x" * (MAX_LATEST_RESULTS_CHARS + 500))
    assert "x" * MAX_LATEST_RESULTS_CHARS + "\n[... cut]" in user
    assert "x" * (MAX_LATEST_RESULTS_CHARS + 1) not in user
