"""Tests for agentigrid.engine.topology — hop-distance graph primitive (C.6).

Covers:
- build_adjacency: undirected, in-service-only, every bus present, parallel collapse
- k_nearest_by_hops: BFS order, tie-break by bus number, truncation to k
- hop_distance: identity, direct neighbor, two-hop, unreachable -> None, unknown -> ValueError
- incident_branches: stable file order, parallel circuits preserved, out-of-service exclusion
- count_reachable: total reachable count
- format_nearest_neighbors_view / format_incident_branches_view: content checks
- out-of-service toggle: synthetic 3-bus network
- determinism: two successive calls return identical results
- handler-level: _handle_topology_analyze sets _latest_results_text correctly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.engine.topology import (
    build_adjacency,
    count_reachable,
    format_incident_branches_view,
    format_nearest_neighbors_view,
    hop_distance,
    incident_branches,
    k_nearest_by_hops,
)
from agentigrid.parsers.matpower_model import Branch, Bus, GenCost, Generator, MATNetwork
from agentigrid.parsers.matpower_parser import parse_matpower

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IEEE118 = DATA_DIR / "ieee_118_bus_v10.m"
_has_118 = IEEE118.exists()


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _bus(bus_i: int) -> Bus:
    return Bus(
        bus_i=bus_i, type=1, Pd=0.0, Qd=0.0, Gs=0.0, Bs=0.0,
        area=1, Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9,
    )


def _branch(fbus: int, tbus: int, status: int = 1, rateA: float = 100.0) -> Branch:
    return Branch(
        fbus=fbus, tbus=tbus, r=0.01, x=0.1, b=0.0,
        rateA=rateA, rateB=100.0, rateC=100.0,
        ratio=0.0, angle=0.0, status=status,
        angmin=-30.0, angmax=30.0,
    )


def _tiny_net(
    buses: list[int],
    branches: list[tuple[int, int, int]],
) -> MATNetwork:
    """Build a minimal MATNetwork from bus ids and (fbus, tbus, status) triples."""
    return MATNetwork(
        casename="tiny", version="2", baseMVA=100.0,
        buses=[_bus(b) for b in buses],
        generators=[],
        branches=[_branch(f, t, s) for f, t, s in branches],
        gencost=[],
        header_comments="",
    )


# ---------------------------------------------------------------------------
# Synthetic out-of-service exclusion (3 buses, 2 branches, one out of service)
# ---------------------------------------------------------------------------

class TestOutOfServiceExclusion:

    @pytest.fixture
    def net(self):
        # 1 --(in-service)--> 2 --(out-of-service, status=0)--> 3
        return _tiny_net([1, 2, 3], [(1, 2, 1), (2, 3, 0)])

    def test_out_of_service_branch_excluded_by_default(self, net):
        adj = build_adjacency(net)
        assert 3 not in adj[2]
        assert 2 not in adj[3]

    def test_out_of_service_branch_included_when_flag_set(self, net):
        adj = build_adjacency(net, include_out_of_service=True)
        assert 3 in adj[2]
        assert 2 in adj[3]

    def test_isolated_bus_present_as_empty_key(self, net):
        adj = build_adjacency(net)
        assert 3 in adj
        assert adj[3] == set()

    def test_k_nearest_excludes_oos_by_default(self, net):
        # bus 3 is unreachable from bus 1 without OOS
        neighbors = k_nearest_by_hops(net, 1, 10)
        neighbor_buses = [nb for nb, _ in neighbors]
        assert 3 not in neighbor_buses

    def test_k_nearest_includes_oos_when_flag_set(self, net):
        neighbors = k_nearest_by_hops(net, 1, 10, include_out_of_service=True)
        neighbor_buses = [nb for nb, _ in neighbors]
        assert 3 in neighbor_buses

    def test_incident_excludes_oos_by_default(self, net):
        inc = incident_branches(net, 2)
        assert all(br.status == 1 for br in inc)
        assert len(inc) == 1  # only the 1-2 branch

    def test_incident_includes_oos_when_flag_set(self, net):
        inc = incident_branches(net, 2, include_out_of_service=True)
        assert len(inc) == 2


# ---------------------------------------------------------------------------
# Real IEEE 118-bus case
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def net118():
    return parse_matpower(IEEE118)


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestBuildAdjacency:

    def test_adj_77(self, net118):
        adj = build_adjacency(net118)
        assert adj[77] == {69, 75, 76, 78, 80, 82}

    def test_adj_10(self, net118):
        adj = build_adjacency(net118)
        assert adj[10] == {9}

    def test_adj_40(self, net118):
        adj = build_adjacency(net118)
        assert adj[40] == {37, 42}

    def test_every_bus_present(self, net118):
        adj = build_adjacency(net118)
        for b in net118.buses:
            assert b.bus_i in adj


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestKNearestByHops:

    def test_k_nearest_77_k3(self, net118):
        result = k_nearest_by_hops(net118, 77, 3)
        assert result == [(69, 1), (75, 1), (76, 1)]

    def test_k_nearest_40_k3(self, net118):
        result = k_nearest_by_hops(net118, 40, 3)
        assert result == [(37, 1), (42, 1), (15, 2)]

    def test_k_nearest_10_k3(self, net118):
        result = k_nearest_by_hops(net118, 10, 3)
        assert result == [(9, 1), (8, 2), (5, 3)]

    def test_k_nearest_77_large_k_fewer_than_k_results(self, net118):
        result = k_nearest_by_hops(net118, 77, 100)
        assert len(result) < 100
        # Strictly ordered by (hop, bus_id)
        for i in range(len(result) - 1):
            nb1, h1 = result[i]
            nb2, h2 = result[i + 1]
            assert (h1, nb1) < (h2, nb2)

    def test_raises_for_unknown_bus(self, net118):
        with pytest.raises(ValueError, match="9999"):
            k_nearest_by_hops(net118, 9999, 3)

    def test_raises_for_k_zero(self, net118):
        with pytest.raises(ValueError):
            k_nearest_by_hops(net118, 77, 0)

    def test_determinism(self, net118):
        r1 = k_nearest_by_hops(net118, 77, 3)
        r2 = k_nearest_by_hops(net118, 77, 3)
        assert r1 == r2


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestHopDistance:

    def test_identity(self, net118):
        assert hop_distance(net118, 77, 77) == 0

    def test_direct_neighbor(self, net118):
        assert hop_distance(net118, 77, 69) == 1

    def test_two_hops(self, net118):
        assert hop_distance(net118, 40, 15) == 2

    def test_unreachable_returns_none(self, net118):
        # build a net with an isolated bus to guarantee unreachability
        net_isolated = _tiny_net([1, 2, 99], [(1, 2, 1)])
        assert hop_distance(net_isolated, 1, 99) is None

    def test_unknown_src_raises(self, net118):
        with pytest.raises(ValueError):
            hop_distance(net118, 9999, 77)

    def test_unknown_dst_raises(self, net118):
        with pytest.raises(ValueError):
            hop_distance(net118, 77, 9999)


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestIncidentBranches:

    def test_bus_77_has_seven_entries(self, net118):
        inc = incident_branches(net118, 77)
        assert len(inc) == 7

    def test_bus_77_all_in_service(self, net118):
        inc = incident_branches(net118, 77)
        assert all(br.status == 1 for br in inc)

    def test_bus_77_two_parallel_80_circuits(self, net118):
        inc = incident_branches(net118, 77)
        parallel = [(br.fbus, br.tbus) for br in inc if 80 in (br.fbus, br.tbus)]
        assert len(parallel) == 2

    def test_raises_for_unknown_bus(self, net118):
        with pytest.raises(ValueError, match="9999"):
            incident_branches(net118, 9999)


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestCountReachable:

    def test_count_reachable_from_77(self, net118):
        total = count_reachable(net118, 77)
        assert total == 99  # 100-bus case, 77 can reach all others


# ---------------------------------------------------------------------------
# View formatters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestFormatters:

    def test_nearest_neighbors_view_bus_77(self, net118):
        neighbors = k_nearest_by_hops(net118, 77, 3)
        total = count_reachable(net118, 77)
        text = format_nearest_neighbors_view(77, 3, neighbors, total)
        assert "69" in text
        assert "75" in text
        assert "76" in text
        assert "99" in text  # total reachable count

    def test_nearest_neighbors_view_short_result_note(self, net118):
        net_tiny = _tiny_net([1, 2], [(1, 2, 1)])
        neighbors = k_nearest_by_hops(net_tiny, 1, 5)
        total = count_reachable(net_tiny, 1)
        text = format_nearest_neighbors_view(1, 5, neighbors, total)
        assert "Only 1 neighbor" in text

    def test_incident_branches_view_bus_77_has_7_rows(self, net118):
        branches = incident_branches(net118, 77)
        text = format_incident_branches_view(77, branches)
        # 1 header row + 7 data rows = 8 non-empty data lines before the note
        data_rows = [
            line for line in text.splitlines()
            if "|" in line and "fbus" not in line
        ]
        assert len(data_rows) == 7

    def test_incident_branches_view_contains_parallel_note(self, net118):
        branches = incident_branches(net118, 77)
        text = format_incident_branches_view(77, branches)
        assert "Parallel circuits" in text or "parallel" in text.lower()

    def test_incident_branches_view_empty_bus(self):
        net_tiny = _tiny_net([1, 2, 3], [(1, 2, 1)])
        branches = incident_branches(net_tiny, 3)  # isolated
        text = format_incident_branches_view(3, branches)
        assert "No in-service" in text


# ---------------------------------------------------------------------------
# Handler-level: _handle_topology_analyze
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestHandlerLevel:

    def _make_controller(self, net118):
        """Minimal stub that exposes only the attributes _handle_topology_analyze reads."""
        from agentigrid.engine.agent_loop import AgentLoopController

        ctrl = AgentLoopController.__new__(AgentLoopController)
        ctrl._config = MagicMock()
        ctrl._base_network = net118
        ctrl._current_network = net118
        ctrl._latest_results_text = None
        ctrl._error_feedback = None
        ctrl._latest_opflow = None
        ctrl._journal = MagicMock()
        ctrl._print = lambda *a, **k: None
        return ctrl

    def test_nearest_neighbors_sets_results_text(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "nearest_neighbors", "bus": 77, "k": 3},
            query_type="nearest_neighbors",
        )
        assert result == ("analyze", True)
        assert ctrl._latest_results_text is not None
        text = ctrl._latest_results_text
        assert "69" in text and "75" in text and "76" in text

    def test_incident_branches_sets_results_text(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "incident_branches", "bus": 77},
            query_type="incident_branches",
        )
        assert result == ("analyze", True)
        assert ctrl._latest_results_text is not None
        assert "77" in ctrl._latest_results_text

    def test_unknown_query_type_returns_error(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "full_adjacency"},
            query_type="full_adjacency",
        )
        assert result[0] == "error"
        assert ctrl._error_feedback is not None

    def test_unknown_bus_returns_error(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "nearest_neighbors", "bus": 9999, "k": 3},
            query_type="nearest_neighbors",
        )
        assert result[0] == "error"

    def test_no_backend_call_made(self, net118):
        """Structured topology query must not invoke any backend."""
        ctrl = self._make_controller(net118)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_backend:
            ctrl._handle_topology_analyze(
                iteration=1,
                data={"action": "analyze", "query_type": "nearest_neighbors", "bus": 77, "k": 3},
                query_type="nearest_neighbors",
            )
            mock_backend.assert_not_called()

    def test_free_text_analyze_path_unaffected(self, net118):
        """When query_type is absent, _handle_analyze must follow the original path."""
        ctrl = self._make_controller(net118)
        ctrl._run_analysis_query = MagicMock(return_value="mock result text")
        result = ctrl._handle_analyze(
            iteration=1,
            data={"action": "analyze", "query": "buses with voltage below 0.95"},
        )
        assert result == ("analyze", True)
        ctrl._run_analysis_query.assert_called_once_with("buses with voltage below 0.95")
