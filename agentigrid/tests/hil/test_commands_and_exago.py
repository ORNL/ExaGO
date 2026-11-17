"""close_tie parsing and checks; reading ExaGO steady states for the HIL."""

from __future__ import annotations

import pytest

from agentigrid.hil_engine.commands import parse_close_tie, validate_close_tie
from agentigrid.hil_engine.from_exago import latest_state, load_state, radial_reference

from tests.hil.helpers import RADIAL_VM, TIE_VM, V_LN, exago_workdir, network, write_m


@pytest.mark.parametrize("tie", [[2, 4], [4, 2], ["4", "2"], "2-4", "4 - 2", "line 2-4"])
def test_tie_any_form_any_order(tie):
    assert parse_close_tie({"action": "close_tie", "tie": tie}).tie == (2, 4)


@pytest.mark.parametrize("raw,match", [
    ({"action": "close_tie"}, "needs"),
    ({"action": "close_tie", "tie": [2]}, "exactly two"),
    ({"action": "close_tie", "tie": [2, 4], "start_from": "meshed"}, "start_from"),
    ({"action": "close_tie", "tie": [2, 4], "speed": 1}, "Unknown"),
])
def test_bad_requests(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_close_tie(raw)


def test_only_ties_can_be_closed():
    net = network()
    assert validate_close_tie(parse_close_tie({"tie": [4, 2]}), net) is None
    msg = validate_close_tie(parse_close_tie({"tie": [2, 3]}), net)
    assert "ordinary line (Line 702) with no switch" in msg and "2-4" in msg
    assert "no line between" in validate_close_tie(parse_close_tie({"tie": [1, 4]}), net)
    assert "does not exist" in validate_close_tie(parse_close_tie({"tie": [2, 9]}), net)


def test_exago_state_reads_ties_either_order(tmp_path):
    net = network()
    st = load_state(write_m(tmp_path / "a.m", TIE_VM, {b: 0 for b in TIE_VM}, 1), net, "x")
    assert st.usable and st.closed_ties == {(2, 4)}
    assert st.volts[4] == pytest.approx(TIE_VM[4] * V_LN)


def test_line_out_cannot_be_built_on_the_hil(tmp_path):
    st = load_state(write_m(tmp_path / "a.m", TIE_VM, {b: 0 for b in TIE_VM}, 1, line_23_status=0),
                    network(), "x")
    assert not st.usable and "line 2-3 is out of service" in st.problems[0]


def test_latest_and_radial_reference(tmp_path):
    net = network()
    source, st = latest_state(exago_workdir(tmp_path), net)
    assert source.iteration == 1 and st.closed_ties == {(2, 4)}
    radial = radial_reference(source, net)
    assert radial is not None and not radial.closed_ties
    assert radial.volts[4] == pytest.approx(RADIAL_VM[4] * V_LN)


def test_no_radial_reference(tmp_path):
    net = network()
    source, _ = latest_state(exago_workdir(tmp_path, with_radial=False), net)
    assert radial_reference(source, net) is None
