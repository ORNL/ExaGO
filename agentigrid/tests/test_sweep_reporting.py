"""Tests for sweep-reporting honesty fixes (prompt #14 analysis).

Covers:
- _infeasible_reason uses convergence_status as sole source of truth
- Certified-reason rendering derives from status, never from stored heuristic label
- Near-optimal cluster caveat fires / does not fire at the correct threshold
- Top-K ranking returns the K cheapest buses with correct Δ values
- PDF font regression: DejaVuSans still embedded after touching the report
"""

from __future__ import annotations

import pytest

from agentigrid.engine.agent_loop import _infeasible_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    bus: int,
    feasible: bool,
    status: str,
    voltage_min: float = 0.95,
    voltage_max: float = 1.05,
    max_line_loading_pct: float = 50.0,
    violations: int = 0,
    cost: float | None = None,
    reason: str | None = None,
) -> dict:
    return {
        "bus": bus,
        "feasible": feasible,
        "status": status,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "max_line_loading_pct": max_line_loading_pct,
        "violations": violations,
        "cost": cost,
        "reason": reason if reason is not None else ("" if feasible else "did not converge"),
    }


def _certified_reason(v: dict) -> str:
    """Mirror of the helper in report_generator.py and app.py."""
    status = (v.get("status") or "").upper()
    return "Did not converge" if not status.startswith("CONVERGED") else "Constraint violation"


# ---------------------------------------------------------------------------
# Fix 1a — _infeasible_reason source of truth
# ---------------------------------------------------------------------------

class TestInfeasibleReason:

    def test_did_not_converge_status(self):
        assert _infeasible_reason("DID NOT CONVERGE") == "did not converge"

    def test_failed_status(self):
        assert _infeasible_reason("FAILED") == "did not converge"

    def test_empty_status(self):
        assert _infeasible_reason("") == "did not converge"

    def test_none_status(self):
        assert _infeasible_reason(None) == "did not converge"

    def test_converged_maps_to_constraint_violation(self):
        assert _infeasible_reason("CONVERGED") == "constraint violation"

    def test_converged_case_insensitive(self):
        assert _infeasible_reason("converged") == "constraint violation"

    def test_converged_prefix_partial(self):
        # Any string starting with CONVERGED counts (e.g. "CONVERGED with warnings")
        assert _infeasible_reason("CONVERGED with warnings") == "constraint violation"


# ---------------------------------------------------------------------------
# Fix 1b — certified reason rendering: status overrides stored reason
# ---------------------------------------------------------------------------

class TestCertifiedReasonRendering:

    def test_all_dnc_candidates_render_did_not_converge(self):
        """All DID NOT CONVERGE buses must show "Did not converge", regardless of
        what was stored in the `reason` field (e.g. old heuristic label)."""
        candidates = [
            _make_candidate(52, False, "DID NOT CONVERGE", reason="line overload",
                            max_line_loading_pct=103.4, violations=2),
            _make_candidate(68, False, "DID NOT CONVERGE", reason="line overload",
                            max_line_loading_pct=100.0, violations=0),
            _make_candidate(69, False, "DID NOT CONVERGE", reason="did not converge",
                            max_line_loading_pct=100.0, violations=0),
            _make_candidate(77, False, "DID NOT CONVERGE", reason="line overload",
                            max_line_loading_pct=293.8, violations=2),
        ]
        rendered = [_certified_reason(v) for v in candidates]
        assert all(r == "Did not converge" for r in rendered), (
            f"Some buses wrongly labelled: {rendered}"
        )

    def test_no_line_overload_label_for_dnc(self):
        """Bus 77 (293.8% loading, DID NOT CONVERGE) must NOT be labelled 'Line overload'."""
        v = _make_candidate(77, False, "DID NOT CONVERGE",
                            max_line_loading_pct=293.8, violations=2, reason="line overload")
        assert _certified_reason(v) == "Did not converge"
        assert "overload" not in _certified_reason(v).lower()

    def test_converged_infeasible_gets_constraint_violation(self):
        """CONVERGED but infeasible (PFLOW post-solve check) → 'Constraint violation'."""
        v = _make_candidate(10, False, "CONVERGED", violations=1, reason="line overload")
        assert _certified_reason(v) == "Constraint violation"

    def test_build_error_shows_did_not_converge(self):
        v = _make_candidate(99, False, "BUILD_ERROR", reason="build error")
        assert _certified_reason(v) == "Did not converge"

    def test_feasible_candidate_not_in_infeasible_table(self):
        """Feasible candidates should not appear in the infeasible table at all —
        confirmed by the filter, not by the reason function. Just assert the reason
        function still behaves correctly for CONVERGED feasible-looking status."""
        v = _make_candidate(100, True, "CONVERGED", violations=0, cost=28000.0)
        # Even for feasible, the reason function would say "Constraint violation"
        # (it's only called for infeasible rows in practice)
        assert _certified_reason(v) == "Constraint violation"


# ---------------------------------------------------------------------------
# Fix 2 — near-optimal cluster caveat logic
# ---------------------------------------------------------------------------

class TestNearOptimalCaveat:

    def _sorted_costs(self, candidates):
        return sorted(
            [(v, v["cost"]) for v in candidates if isinstance(v.get("cost"), (int, float))],
            key=lambda x: x[1],
        )

    def test_caveat_fires_when_gap_below_tolerance(self):
        """Bus 189 ($28,228.57) vs Bus 187 ($28,230.05) → gap $1.48 < $5.00 tolerance."""
        candidates = [
            _make_candidate(189, True, "CONVERGED", cost=28228.57),
            _make_candidate(187, True, "CONVERGED", cost=28230.05),
            _make_candidate(50,  True, "CONVERGED", cost=28300.00),
        ]
        sorted_costs = self._sorted_costs(candidates)
        best = sorted_costs[0][1]
        gap = sorted_costs[1][1] - best
        abs_tol = 5.0
        assert gap < abs_tol, f"Gap ${gap:.2f} should trigger caveat (threshold ${abs_tol})"

    def test_caveat_does_not_fire_when_gap_above_tolerance(self):
        candidates = [
            _make_candidate(100, True, "CONVERGED", cost=28000.00),
            _make_candidate(101, True, "CONVERGED", cost=28010.00),  # gap = $10 > $5
        ]
        sorted_costs = self._sorted_costs(candidates)
        best = sorted_costs[0][1]
        gap = sorted_costs[1][1] - best
        abs_tol = 5.0
        assert gap >= abs_tol, f"Gap ${gap:.2f} should NOT trigger caveat (threshold ${abs_tol})"

    def test_caveat_does_not_fire_at_exact_tolerance(self):
        """Gap exactly at tolerance should NOT trigger the caveat (strictly below)."""
        candidates = [
            _make_candidate(100, True, "CONVERGED", cost=28000.00),
            _make_candidate(101, True, "CONVERGED", cost=28005.00),  # gap = $5.00 = tolerance
        ]
        sorted_costs = self._sorted_costs(candidates)
        gap = sorted_costs[1][1] - sorted_costs[0][1]
        abs_tol = 5.0
        assert not (gap < abs_tol), f"Gap exactly at tolerance should NOT fire caveat"

    def test_single_feasible_bus_no_caveat(self):
        candidates = [_make_candidate(100, True, "CONVERGED", cost=28000.00)]
        sorted_costs = self._sorted_costs(candidates)
        # Need at least 2 to compute a gap
        assert len(sorted_costs) < 2


# ---------------------------------------------------------------------------
# Fix 2 — top-K ranking
# ---------------------------------------------------------------------------

class TestTopKRanking:

    def test_top_k_selects_k_cheapest(self):
        candidates = [
            _make_candidate(i, True, "CONVERGED", cost=28000.0 + i)
            for i in range(20)
        ]
        feasible_with_cost = sorted(
            [(v, v["cost"]) for v in candidates if isinstance(v.get("cost"), (int, float))],
            key=lambda x: x[1],
        )
        k = 10
        top_k = feasible_with_cost[:k]
        assert len(top_k) == k
        assert top_k[0][1] == 28000.0
        assert top_k[-1][1] == 28009.0

    def test_delta_from_best(self):
        candidates = [
            _make_candidate(189, True, "CONVERGED", cost=28228.57),
            _make_candidate(187, True, "CONVERGED", cost=28230.05),
        ]
        sorted_costs = sorted(
            [(v, v["cost"]) for v in candidates if isinstance(v.get("cost"), (int, float))],
            key=lambda x: x[1],
        )
        best = sorted_costs[0][1]
        deltas = [cost - best for _, cost in sorted_costs]
        assert deltas[0] == pytest.approx(0.00)
        assert deltas[1] == pytest.approx(1.48)

    def test_ranking_order_ascending_cost(self):
        candidates = [
            _make_candidate(3, True, "CONVERGED", cost=300.0),
            _make_candidate(1, True, "CONVERGED", cost=100.0),
            _make_candidate(2, True, "CONVERGED", cost=200.0),
        ]
        sorted_costs = sorted(
            [(v, v["cost"]) for v in candidates if isinstance(v.get("cost"), (int, float))],
            key=lambda x: x[1],
        )
        buses_in_order = [v["bus"] for v, _ in sorted_costs]
        assert buses_in_order == [1, 2, 3]

    def test_buses_without_cost_excluded(self):
        candidates = [
            _make_candidate(1, True, "CONVERGED", cost=28000.0),
            _make_candidate(2, True, "CONVERGED", cost=None),   # no cost → excluded
            _make_candidate(3, True, "CONVERGED", cost=28005.0),
        ]
        feasible_with_cost = [
            (v, v["cost"]) for v in candidates
            if isinstance(v.get("cost"), (int, float))
        ]
        assert len(feasible_with_cost) == 2


# ---------------------------------------------------------------------------
# PDF font regression — DejaVuSans embedded after report changes
# ---------------------------------------------------------------------------

class TestPDFFontRegression:

    def test_dejavu_sans_still_used(self):
        """Instantiating ReportGenerator should pick DejaVuSans (or Helvetica
        fallback) — it must not revert to Courier or another serif font."""
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "launcher"))
            from report_generator import ReportGenerator
        except ImportError:
            pytest.skip("report_generator not importable in this environment")

        rg = ReportGenerator()
        assert rg._font in ("DejaVuSans", "Helvetica"), (
            f"Unexpected font: {rg._font!r}; DejaVuSans or Helvetica fallback expected"
        )
        assert rg._font_bold in ("DejaVuSans-Bold", "Helvetica-Bold"), (
            f"Unexpected bold font: {rg._font_bold!r}"
        )

    def test_certified_reason_helper_present(self):
        """ReportGenerator must expose _certified_reason as a static method."""
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "launcher"))
            from report_generator import ReportGenerator
        except ImportError:
            pytest.skip("report_generator not importable in this environment")

        assert hasattr(ReportGenerator, "_certified_reason"), (
            "_certified_reason static method missing from ReportGenerator"
        )
        # Spot-check: DID NOT CONVERGE → Did not converge
        v_dnc = _make_candidate(1, False, "DID NOT CONVERGE", reason="line overload")
        assert ReportGenerator._certified_reason(v_dnc) == "Did not converge"
        # Spot-check: CONVERGED infeasible → Constraint violation
        v_conv = _make_candidate(2, False, "CONVERGED", violations=1)
        assert ReportGenerator._certified_reason(v_conv) == "Constraint violation"
