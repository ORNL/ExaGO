"""Tests for Pareto front filter."""

import pytest

from agentigrid.engine.journal import ObjectiveEntry
from agentigrid.engine.pareto import ParetoCandidate, _dominates, pareto_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj(name: str, direction: str = "minimize") -> ObjectiveEntry:
    return ObjectiveEntry(
        name=name, direction=direction, priority="primary", introduced_at=0, source="test",
    )


def _cand(label: str, feasible: bool, **metrics: float) -> ParetoCandidate:
    return ParetoCandidate(label=label, metrics=metrics, feasible=feasible)


# ---------------------------------------------------------------------------
# _dominates tests
# ---------------------------------------------------------------------------

class TestDominates:

    def test_feasible_dominates_infeasible(self):
        a = _cand("A", feasible=True, cost=100.0)
        b = _cand("B", feasible=False, cost=200.0)
        assert _dominates(a, b, [_obj("cost")]) is True

    def test_infeasible_does_not_dominate_feasible(self):
        a = _cand("A", feasible=False, cost=100.0)
        b = _cand("B", feasible=True, cost=200.0)
        assert _dominates(a, b, [_obj("cost")]) is False

    def test_infeasible_does_not_dominate_infeasible(self):
        a = _cand("A", feasible=False, cost=100.0)
        b = _cand("B", feasible=False, cost=200.0)
        assert _dominates(a, b, [_obj("cost")]) is False

    def test_minimize_a_better_on_all(self):
        a = _cand("A", feasible=True, cost=100.0)
        b = _cand("B", feasible=True, cost=200.0)
        assert _dominates(a, b, [_obj("cost")]) is True

    def test_minimize_a_worse(self):
        a = _cand("A", feasible=True, cost=200.0)
        b = _cand("B", feasible=True, cost=100.0)
        assert _dominates(a, b, [_obj("cost")]) is False

    def test_minimize_equal_cost_no_dominance(self):
        a = _cand("A", feasible=True, cost=100.0)
        b = _cand("B", feasible=True, cost=100.0)
        assert _dominates(a, b, [_obj("cost")]) is False

    def test_maximize_direction(self):
        a = _cand("A", feasible=True, voltage=1.05)
        b = _cand("B", feasible=True, voltage=0.98)
        assert _dominates(a, b, [_obj("voltage", "maximize")]) is True
        assert _dominates(b, a, [_obj("voltage", "maximize")]) is False

    def test_multi_objective_tradeoff(self):
        a = _cand("A", feasible=True, cost=100.0, loading=90.0)
        b = _cand("B", feasible=True, cost=120.0, loading=70.0)
        objs = [_obj("cost"), _obj("loading")]
        assert _dominates(a, b, objs) is False
        assert _dominates(b, a, objs) is False

    def test_multi_objective_strict_dominance(self):
        a = _cand("A", feasible=True, cost=80.0, loading=70.0)
        b = _cand("B", feasible=True, cost=100.0, loading=80.0)
        objs = [_obj("cost"), _obj("loading")]
        assert _dominates(a, b, objs) is True
        assert _dominates(b, a, objs) is False

    def test_missing_metric_no_dominance(self):
        a = _cand("A", feasible=True, cost=100.0)
        b = _cand("B", feasible=True, cost=200.0)
        assert _dominates(a, b, [_obj("loading")]) is False

    def test_empty_objectives_no_dominance(self):
        a = _cand("A", feasible=True, cost=100.0)
        b = _cand("B", feasible=True, cost=200.0)
        assert _dominates(a, b, []) is False


# ---------------------------------------------------------------------------
# pareto_filter tests
# ---------------------------------------------------------------------------

class TestParetoFilter:

    def test_empty_input(self):
        assert pareto_filter([], []) == []

    def test_single_feasible(self):
        c = _cand("A", feasible=True, cost=100.0)
        result = pareto_filter([c], [_obj("cost")])
        assert result == ["A"]

    def test_all_infeasible(self):
        cands = [
            _cand("A", feasible=False, cost=100.0),
            _cand("B", feasible=False, cost=200.0),
        ]
        result = pareto_filter(cands, [_obj("cost")])
        assert result == []

    def test_mixed_feasibility_only_feasible_on_pareto(self):
        cands = [
            _cand("A", feasible=True, cost=100.0),
            _cand("B", feasible=False, cost=50.0),
        ]
        result = pareto_filter(cands, [_obj("cost")])
        assert "A" in result
        assert "B" not in result

    def test_two_feasible_one_dominates(self):
        cands = [
            _cand("A", feasible=True, cost=100.0),
            _cand("B", feasible=True, cost=200.0),
        ]
        result = pareto_filter(cands, [_obj("cost")])
        assert result == ["A"]

    def test_two_feasible_tradeoff(self):
        cands = [
            _cand("A", feasible=True, cost=100.0, loading=90.0),
            _cand("B", feasible=True, cost=120.0, loading=70.0),
        ]
        objs = [_obj("cost"), _obj("loading")]
        result = pareto_filter(cands, objs)
        assert set(result) == {"A", "B"}

    def test_three_feasible_one_dominated(self):
        cands = [
            _cand("A", feasible=True, cost=100.0, loading=80.0),
            _cand("B", feasible=True, cost=150.0, loading=70.0),
            _cand("C", feasible=True, cost=200.0, loading=90.0),
        ]
        objs = [_obj("cost"), _obj("loading")]
        result = pareto_filter(cands, objs)
        assert "A" in result
        assert "B" in result
        assert "C" not in result

    def test_no_objectives_uses_default_cost(self):
        cands = [
            _cand("A", feasible=True, generation_cost=100.0),
            _cand("B", feasible=True, generation_cost=80.0),
            _cand("C", feasible=True, generation_cost=120.0),
        ]
        result = pareto_filter(cands, [])
        assert result == ["B"]

    def test_no_objectives_all_equal_cost(self):
        cands = [
            _cand("A", feasible=True, generation_cost=100.0),
            _cand("B", feasible=True, generation_cost=100.0),
        ]
        result = pareto_filter(cands, [])
        assert set(result) == {"A", "B"}

    def test_constraint_objective(self):
        cost_obj = _obj("cost", "minimize")
        voltage_obj = ObjectiveEntry(
            name="max_voltage", direction="constraint",
            priority="secondary", introduced_at=0, source="test",
            threshold=1.05,
        )
        cands = [
            _cand("A", feasible=True, cost=100.0, max_voltage=1.03),
            _cand("B", feasible=True, cost=90.0, max_voltage=1.07),
        ]
        result = pareto_filter(cands, [cost_obj, voltage_obj])
        assert "A" in result
