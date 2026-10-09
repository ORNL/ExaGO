"""Regression: rewritten cases must keep the "};" terminator on cell arrays.

ExaGO's MATPOWER reader (src/ps/psreaddata.cpp) finds the end of mpc.genfuel
by searching for the literal "};". The writer used to emit a bare "}", so ExaGO
silently ignored the fuel types of every case AgentiGrid rewrote: ramp rates
stayed 0 (SCOPFLOW could not redispatch in contingencies and reported local
infeasibility on ACTIVSg200) and wind/solar units stayed fixed at Pmax.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.matpower_writer import write_matpower

ROOT = Path(__file__).resolve().parents[2]
ACTIVSG200 = ROOT.parent / "datafiles" / "case_ACTIVSg200.m"


def _exago_genfuel_entries(path: Path) -> list[str] | None:
    """Mimic psreaddata.cpp: genfuel starts after the 'mpc.genfuel' line and
    ends at the first later line containing '};'. None = end never found."""
    lines = path.read_text().splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and "mpc.genfuel" in line:
            start = i + 1
        elif start is not None and "};" in line:
            end = i
            break
    if start is None or end is None:
        return None
    return [m.group(1) for l in lines[start:end] if (m := re.search(r"'([^']*)'", l))]


@pytest.mark.skipif(not ACTIVSG200.exists(), reason="ExaGO datafiles not present")
def test_roundtrip_keeps_genfuel_readable_by_exago(tmp_path):
    original = _exago_genfuel_entries(ACTIVSG200)
    assert original and "wind" in original
    out = tmp_path / "rt.m"
    write_matpower(parse_matpower(ACTIVSG200), out)
    assert _exago_genfuel_entries(out) == original
    # every cell array closes with "};", every matrix with "];"
    closers = [l.strip() for l in out.read_text().splitlines() if l.strip()[:1] in ("}", "]")]
    assert closers and all(c in ("};", "];") for c in closers)


def test_writer_terminates_legacy_raw_sections(tmp_path):
    """Networks parsed before the fix (e.g. restored sessions) stored '}'."""
    net = MATNetwork(casename="c", version="2", baseMVA=100.0, buses=[], generators=[],
                     branches=[], gencost=[], header_comments="function mpc = c\n",
                     extra_sections={"genfuel": "mpc.genfuel = {\n\t'wind';\n}",
                                     "other": "mpc.other = [\n\t1 2;\n]"})
    out = tmp_path / "c.m"
    write_matpower(net, out)
    text = out.read_text()
    assert "\n};\n" in text and "\n];\n" in text
    assert _exago_genfuel_entries(out) == ["wind"]


@pytest.mark.skipif(not ACTIVSG200.exists(), reason="ExaGO datafiles not present")
def test_double_roundtrip_is_stable(tmp_path):
    a, b = tmp_path / "a.m", tmp_path / "b.m"
    write_matpower(parse_matpower(ACTIVSG200), a)
    write_matpower(parse_matpower(a), b)
    assert "};;" not in b.read_text() and "];;" not in b.read_text()
    assert _exago_genfuel_entries(b) == _exago_genfuel_entries(ACTIVSG200)
