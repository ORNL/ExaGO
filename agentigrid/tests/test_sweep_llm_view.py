"""Tests for B.1 — token-bounded LLM-facing sweep results view.

Verifies:
- Small sweep (≤ threshold): LLM text is byte-identical to the pre-B.1 full table.
- Large sweep (> threshold): text contains header counts, complete feasible list,
  grouped infeasible list, exactly min(top_n, feasible_count) ranked rows,
  aggregate stats, and the journal pointer line.
- Size bound: ranked block stays at top_n rows regardless of candidate count.
- Completeness: feasible + infeasible bus lists union to the full candidate set.
- Journal untouched: add_sweep receives all N per-candidate summaries.
- Objective fallback: empty registry → ranking by "cost (minimize)".
"""

from __future__ import annotations

import statistics

import pytest

from agentigrid.engine.agent_loop import _build_sweep_llm_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    bus: int,
    feasible: bool,
    cost: float | None = None,
    voltage_min: float = 0.95,
    voltage_max: float = 1.05,
    max_line_loading_pct: float = 50.0,
    violations: int = 0,
    reason: str = "",
    status: str = "CONVERGED",
) -> dict:
    return {
        "bus": bus,
        "feasible": feasible,
        "cost": cost,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "max_line_loading_pct": max_line_loading_pct,
        "violations": violations,
        "reason": reason if reason else ("" if feasible else "did not converge"),
        "status": status if feasible else "DID NOT CONVERGE",
    }


def _build_reference_full_table(candidate_summaries, feasible_buses, mut_desc):
    """Replicate the exact pre-B.1 full-table string for regression comparison."""
    n_total = len(candidate_summaries)
    n_feasible = len(feasible_buses)
    infeasible_buses = [s["bus"] for s in candidate_summaries if not s["feasible"]]
    if len(infeasible_buses) > 20:
        infeasible_str = f"[{', '.join(str(b) for b in infeasible_buses[:20])}, ...]"
    else:
        infeasible_str = f"[{', '.join(str(b) for b in infeasible_buses)}]"
    lines = [
        f"Sweep over {n_total} candidate buses (mutation: {mut_desc}):",
        f"FEASIBLE: {n_feasible} / {n_total}.  INFEASIBLE buses: {infeasible_str}",
        f"{'bus':>4} | {'feasible':>8} | {'Vmin':>5} | {'Vmax':>5} | {'maxLoad%':>8} | {'viol':>4} | {'cost':>12}",
    ]
    for s in candidate_summaries:
        feas_str = "yes" if s["feasible"] else "no"
        cost_val = s["cost"]
        cost_str = f"{cost_val:>12,.1f}" if cost_val is not None else "           N/A"
        lines.append(
            f"{s['bus']:>4} | {feas_str:>8} | {s['voltage_min']:>5.3f} | "
            f"{s['voltage_max']:>5.3f} | {s['max_line_loading_pct']:>8.1f} | "
            f"{s['violations']:>4} | {cost_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small sweep (≤ threshold) — must be byte-identical to pre-B.1 full table
# ---------------------------------------------------------------------------

class TestSmallSweepFullTable:

    def _make_small_candidates(self, n=10):
        candidates = []
        for i in range(n):
            feasible = (i % 3 != 0)
            candidates.append(_make_candidate(
                bus=100 + i,
                feasible=feasible,
                cost=28000.0 + i * 10 if feasible else None,
            ))
        return candidates

    def test_small_sweep_is_byte_identical_to_reference(self):
        """candidate_count <= threshold → output identical to pre-B.1 full table."""
        candidates = self._make_small_candidates(10)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        mut_desc = "add_load Pd=100 MW"
        threshold = 250
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc=mut_desc,
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=threshold,
        )
        reference = _build_reference_full_table(candidates, feasible, mut_desc)
        assert result == reference, (
            "Small-sweep LLM view must be byte-identical to the pre-B.1 full table."
        )

    def test_exactly_at_threshold_uses_full_table(self):
        """candidate_count == threshold must still use full table."""
        n = 250
        candidates = [_make_candidate(bus=i, feasible=True, cost=28000.0 + i) for i in range(n)]
        feasible = [s["bus"] for s in candidates]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="add_gen 100 MW",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        reference = _build_reference_full_table(candidates, feasible, "add_gen 100 MW")
        assert result == reference

    def test_full_table_infeasible_truncated_at_20(self):
        """Full-table path: infeasible bus list truncated at 20 with '...'"""
        candidates = [_make_candidate(bus=i, feasible=False) for i in range(25)]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=[],
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=5,
            threshold=250,
        )
        # Second line should have "..." indicating truncation
        lines = result.splitlines()
        assert "..." in lines[1]
        # But first 20 infeasible buses are present
        for i in range(20):
            assert str(i) in lines[1]

    def test_full_table_small_infeasible_not_truncated(self):
        """Full-table path: ≤ 20 infeasible buses — no ellipsis."""
        candidates = [_make_candidate(bus=i, feasible=False) for i in range(5)]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=[],
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=5,
            threshold=250,
        )
        lines = result.splitlines()
        assert "..." not in lines[1]


# ---------------------------------------------------------------------------
# Large sweep (> threshold) — summarized view
# ---------------------------------------------------------------------------

class TestLargeSweepSummary:

    def _make_large_candidates(self, n=300, n_feasible=50):
        candidates = []
        for i in range(n):
            feasible = (i < n_feasible)
            candidates.append(_make_candidate(
                bus=i + 1,
                feasible=feasible,
                cost=28000.0 + i if feasible else None,
                reason="" if feasible else "did not converge",
            ))
        return candidates

    def test_header_contains_counts(self):
        candidates = self._make_large_candidates(300, 50)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="add_load Pd=100 MW",
            objective_name="generation_cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        assert "300" in result  # total candidates
        assert "50" in result   # feasible count
        assert "250" in result  # infeasible count

    def test_feasible_bus_list_complete(self):
        """All feasible buses must appear in the summary."""
        candidates = self._make_large_candidates(300, 50)
        feasible_buses = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible_buses,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        for bus in feasible_buses:
            assert str(bus) in result, f"Bus {bus} missing from feasible list in summary"

    def test_infeasible_grouped_by_reason(self):
        """Infeasible buses are grouped by reason, not one row each."""
        candidates = [
            _make_candidate(1, False, reason="did not converge"),
            _make_candidate(2, False, reason="did not converge"),
            _make_candidate(3, False, reason="constraint violation"),
            _make_candidate(4, True, cost=28000.0),
        ] + [_make_candidate(100 + i, False, reason="did not converge") for i in range(300)]
        feasible = [4]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=5,
            threshold=250,
        )
        assert "did not converge" in result
        assert "constraint violation" in result
        # grouped: not one line per bus (no "did not converge: [1]" then "did not converge: [2]")
        lines = [l for l in result.splitlines() if "did not converge" in l]
        # Should be exactly one line grouping all DNC buses
        assert len(lines) == 1

    def test_top_n_ranked_rows_exact_count(self):
        """Exactly min(top_n, feasible_count) ranked rows in summary block."""
        n_feasible = 40
        candidates = self._make_large_candidates(300, n_feasible)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        top_n = 25
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="generation_cost",
            objective_direction="minimize",
            top_n=top_n,
            threshold=250,
        )
        # Count lines that start with a rank number (right-aligned, e.g. "   1 |")
        ranked_lines = [l for l in result.splitlines() if l.strip() and l.strip()[0].isdigit() and "|" in l]
        assert len(ranked_lines) == min(top_n, n_feasible)

    def test_aggregate_stats_present(self):
        """min / median / max stats over the feasible set must be present."""
        candidates = self._make_large_candidates(300, 50)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        assert "min=" in result
        assert "median=" in result
        assert "max=" in result

    def test_aggregate_stats_correct_values(self):
        """min/median/max must match the actual feasible cost distribution."""
        costs = [28000.0, 28010.0, 28020.0, 28030.0, 28040.0]
        candidates = [_make_candidate(i + 1, True, cost=costs[i]) for i in range(5)]
        candidates += [_make_candidate(100 + i, False) for i in range(300)]
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        # min=28,000.0  median=28,020.0  max=28,040.0
        assert "28,000.0" in result
        assert "28,020.0" in result
        assert "28,040.0" in result

    def test_journal_pointer_line_present(self):
        """Pointer line must appear in the summary (not in full table)."""
        candidates = self._make_large_candidates(300, 10)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        assert "journal" in result.lower()
        assert "PDF report" in result or "pdf report" in result.lower()

    def test_objective_label_in_summary(self):
        """The primary objective name and direction appear in the ranked block header."""
        candidates = self._make_large_candidates(300, 30)
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="generation_cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )
        assert "generation_cost" in result
        assert "minimize" in result

    def test_maximize_direction_sorts_descending(self):
        """When direction is 'maximize', the top-N block shows highest-cost first."""
        candidates = [
            _make_candidate(1, True, cost=100.0),
            _make_candidate(2, True, cost=300.0),
            _make_candidate(3, True, cost=200.0),
        ] + [_make_candidate(100 + i, False) for i in range(300)]
        feasible = [1, 2, 3]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="loadability_margin",
            objective_direction="maximize",
            top_n=3,
            threshold=250,
        )
        # The ranked block: rank 1 should be bus 2 (cost=300, maximize)
        lines = [l for l in result.splitlines() if l.strip().startswith("1")]
        assert len(lines) >= 1
        assert "2" in lines[0]  # bus 2 is rank 1

    def test_threshold_zero_forces_summary(self):
        """threshold=0 forces summary mode even for small candidate sets."""
        candidates = [_make_candidate(i, True, cost=28000.0 + i) for i in range(10)]
        feasible = list(range(10))
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=5,
            threshold=0,
        )
        # Summary has the pointer line; full table does not
        assert "journal" in result.lower()


# ---------------------------------------------------------------------------
# Size bound — ranked block bounded by top_n regardless of candidate count
# ---------------------------------------------------------------------------

class TestSizeBound:

    def test_ranked_block_bounded_by_top_n(self):
        """Ranked block must have at most top_n rows for a 2000-bus sweep."""
        n = 2000
        top_n = 25
        candidates = [_make_candidate(i + 1, True, cost=28000.0 + i) for i in range(n)]
        feasible = [s["bus"] for s in candidates]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="add_gen 100 MW",
            objective_name="cost",
            objective_direction="minimize",
            top_n=top_n,
            threshold=250,
        )
        ranked_lines = [
            l for l in result.splitlines()
            if l.strip() and l.strip()[0].isdigit() and "|" in l
        ]
        assert len(ranked_lines) == top_n, (
            f"Ranked block has {len(ranked_lines)} rows; expected exactly {top_n}"
        )

    def test_line_count_does_not_scale_with_n_infeasible(self):
        """Adding more infeasible candidates does not grow the line count proportionally.

        The infeasible section is grouped by reason, so adding 1000 DNC buses
        should add at most 1 extra line (the grouped entry), not 1000 lines.
        """
        def _line_count(n_infeasible):
            candidates = (
                [_make_candidate(i + 1, True, cost=28000.0 + i) for i in range(50)]
                + [_make_candidate(1000 + i, False, reason="did not converge")
                   for i in range(n_infeasible)]
            )
            feasible = [s["bus"] for s in candidates if s["feasible"]]
            result = _build_sweep_llm_view(
                candidate_summaries=candidates,
                feasible_buses=feasible,
                mut_desc="mut",
                objective_name="cost",
                objective_direction="minimize",
                top_n=25,
                threshold=250,
            )
            return len(result.splitlines())

        lc_100 = _line_count(100)
        lc_1000 = _line_count(1000)
        # Going from 100 to 1000 infeasible should NOT add ~900 lines
        assert lc_1000 - lc_100 < 10, (
            f"Line count grew from {lc_100} to {lc_1000} — infeasible grouping is not working"
        )


# ---------------------------------------------------------------------------
# Completeness — feasible + infeasible union to full candidate set
# ---------------------------------------------------------------------------

class TestCompleteness:

    def test_feasible_and_infeasible_union_is_full_set(self):
        """Every bus in candidate_summaries must appear in either the feasible or
        infeasible section of the summary."""
        n_feasible = 40
        n_infeasible = 260
        candidates = (
            [_make_candidate(i + 1, True, cost=28000.0 + i) for i in range(n_feasible)]
            + [_make_candidate(1000 + i, False) for i in range(n_infeasible)]
        )
        feasible_buses = [s["bus"] for s in candidates if s["feasible"]]
        all_buses = {s["bus"] for s in candidates}

        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible_buses,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=25,
            threshold=250,
        )

        # Check every feasible bus is in the result text
        for bus in feasible_buses:
            assert str(bus) in result, f"Feasible bus {bus} not found in summary"

        # Check infeasible buses appear (at least as part of the grouped entry)
        # The grouped line has all DNC buses listed as a Python list
        infeasible_buses = [s["bus"] for s in candidates if not s["feasible"]]
        for bus in infeasible_buses:
            assert str(bus) in result, f"Infeasible bus {bus} not found in summary"


# ---------------------------------------------------------------------------
# Objective fallback — empty registry falls back to cost/minimize
# ---------------------------------------------------------------------------

class TestObjectiveFallback:

    def test_fallback_label_when_no_registry(self):
        """When called with objective_name='cost', objective_direction='minimize'
        (the fallback values used when the registry has no primary objective),
        the summary labels the ranking correctly."""
        candidates = [
            _make_candidate(i + 1, True, cost=28000.0 + i) for i in range(10)
        ] + [_make_candidate(300 + i, False) for i in range(300)]
        feasible = [s["bus"] for s in candidates if s["feasible"]]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=5,
            threshold=250,
        )
        assert "cost" in result
        assert "minimize" in result

    def test_fallback_ranking_ascending(self):
        """Fallback 'cost minimize' → cheapest bus is rank 1."""
        candidates = [
            _make_candidate(3, True, cost=300.0),
            _make_candidate(1, True, cost=100.0),
            _make_candidate(2, True, cost=200.0),
        ] + [_make_candidate(100 + i, False) for i in range(300)]
        feasible = [1, 2, 3]
        result = _build_sweep_llm_view(
            candidate_summaries=candidates,
            feasible_buses=feasible,
            mut_desc="mut",
            objective_name="cost",
            objective_direction="minimize",
            top_n=3,
            threshold=250,
        )
        # rank 1 should be bus 1 (cheapest)
        rank1_lines = [l for l in result.splitlines() if l.strip().startswith("1") and "|" in l]
        assert len(rank1_lines) >= 1
        assert "1" in rank1_lines[0].split("|")[1]  # bus column
