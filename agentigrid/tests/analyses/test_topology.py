"""Tests for agentigrid.engine.topology and the structured topology analyze queries.

Covers:
- incident_branches: stable file order, parallel circuits preserved, out-of-service exclusion
- format_incident_branches_view: content checks
- handler-level: _handle_topology_analyze for affected_elements (the contingency
  neighbor search) and incident_branches sets _latest_results_text correctly

Substation grouping and tiers (substation_map, substations_within) are covered
in test_contingency.py alongside the neighbor search that uses them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.engine import contingency as C
from agentigrid.engine.topology import format_incident_branches_view, incident_branches
from agentigrid.parsers.matpower_model import Branch, Bus, MATNetwork
from agentigrid.parsers.matpower_parser import parse_matpower

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "exago" / "examples"
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


# ---------------------------------------------------------------------------
# View formatters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
class TestFormatters:

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

    def _affected(self, ctrl, **extra):
        data = {"action": "analyze", "query_type": "affected_elements",
                "mutation": {"action": "add_load_at_bus", "bus": 77, "Pd": 50.0}, **extra}
        return ctrl._handle_topology_analyze(
            iteration=1, data=data, query_type="affected_elements",
        )

    def test_affected_elements_matches_the_contingency_search(self, net118):
        ctrl = self._make_controller(net118)
        assert self._affected(ctrl, substation_depth=1) == ("analyze", True)
        text = ctrl._latest_results_text
        expected = C.find_affected_elements(net118, C.ChangedElement(kind="load", bus=77), 1)
        assert f"branches ({len(expected.branches)})" in text
        assert f"gens ({len(expected.gens)})" in text
        assert f"loads ({len(expected.loads)})" in text
        assert f"shunts ({len(expected.shunts)})" in text
        assert "tier 0: [77]" in text

    def test_affected_elements_default_depth(self, net118):
        ctrl = self._make_controller(net118)
        assert self._affected(ctrl) == ("analyze", True)
        assert f"substation depth {C.DEFAULT_SUBSTATION_DEPTH}" in ctrl._latest_results_text

    def test_affected_elements_requires_mutation(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "affected_elements", "bus": 77},
            query_type="affected_elements",
        )
        assert result[0] == "error"
        assert "mutation" in ctrl._error_feedback

    def test_affected_elements_unknown_bus_returns_error(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "affected_elements",
                  "mutation": {"action": "add_load_at_bus", "bus": 9999, "Pd": 1.0}},
            query_type="affected_elements",
        )
        assert result[0] == "error"
        assert "9999" in ctrl._error_feedback

    def test_affected_elements_bad_depth_returns_error(self, net118):
        ctrl = self._make_controller(net118)
        assert self._affected(ctrl, substation_depth=-1)[0] == "error"

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

    def test_incident_branches_unknown_bus_returns_error(self, net118):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": "incident_branches", "bus": 9999},
            query_type="incident_branches",
        )
        assert result[0] == "error"

    @pytest.mark.parametrize("query_type", ["full_adjacency", "k_hop_buses"])
    def test_unknown_query_type_returns_error(self, net118, query_type):
        ctrl = self._make_controller(net118)
        result = ctrl._handle_topology_analyze(
            iteration=1,
            data={"action": "analyze", "query_type": query_type, "bus": 77},
            query_type=query_type,
        )
        assert result[0] == "error"
        assert "affected_elements" in ctrl._error_feedback

    def test_no_backend_call_made(self, net118):
        """Structured topology query must not invoke any backend."""
        ctrl = self._make_controller(net118)
        with patch("agentigrid.engine.agent_loop.create_backend") as mock_backend:
            self._affected(ctrl)
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
