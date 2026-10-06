"""MATPOWER reader/writer hardening: no silent misreads, stable round trips.

Covers five defects found while specifying the parser:
  1. malformed rows were skipped with only a log warning (a bus could vanish);
  2. valid-but-unsupported MATLAB syntax was misread (two rows on one line
     became one garbage row; comma-separated rows were all dropped);
  3. the writer crashed on Inf/NaN, which the reader accepts;
  4. every read/write round trip added another header line;
  5. MATPOWER case version '1' was accepted although ExaGO reads version 2.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from agentigrid.parsers.matpower_model import Branch, Bus, GenCost, Generator, MATNetwork
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.matpower_writer import write_matpower

ROOT = Path(__file__).resolve().parents[2]
ACTIVSG200 = ROOT.parent / "datafiles" / "case_ACTIVSg200.m"

BUS = "\t1\t3\t0\t0\t0\t0\t1\t1\t0\t230\t1\t1.1\t0.9;\n\t2\t1\t90\t30\t0\t0\t1\t1\t0\t230\t1\t1.1\t0.9;"
GEN = "\t1\t0\t0\t300\t-300\t1\t100\t1\t250\t10;"
BRANCH = "\t1\t2\t0.01\t0.1\t0\t100\t100\t100\t0\t0\t1\t-360\t360;"
GENCOST = "\t2\t0\t0\t3\t0.11\t5\t150;"


def _case(*, version="2", bus=BUS, gen=GEN, branch=BRANCH, gencost=GENCOST, extra="",
          drop: tuple[str, ...] = ()) -> str:
    parts = {
        "head": "function mpc = tiny\n%TINY test case\n\n%% MATPOWER Case Format : Version 2\n",
        "version": f"mpc.version = '{version}';\n",
        "base": "\n%% system MVA base\nmpc.baseMVA = 100;\n",
        "bus": f"\n%% bus data\nmpc.bus = [\n{bus}\n];\n",
        "gen": f"\n%% generator data\nmpc.gen = [\n{gen}\n];\n",
        "branch": f"\n%% branch data\nmpc.branch = [\n{branch}\n];\n",
        "gencost": f"\n%% generator cost data\nmpc.gencost = [\n{gencost}\n];\n",
        "extra": f"\n{extra}\n" if extra else "",
    }
    return "".join(v for k, v in parts.items() if k not in drop)


def _write(tmp_path: Path, text: str, name: str = "c.m", crlf: bool = False) -> Path:
    p = tmp_path / name
    data = text.replace("\n", "\r\n") if crlf else text
    p.write_bytes(data.encode("utf-8"))
    return p


# --- 5. version ------------------------------------------------------------

def test_version_1_rejected(tmp_path):
    with pytest.raises(ValueError, match="version"):
        parse_matpower(_write(tmp_path, _case(version="1")))


def test_missing_version_defaults_to_2(tmp_path):
    text = _case().replace("mpc.version = '2';\n", "")
    assert parse_matpower(_write(tmp_path, text)).version == "2"


# --- 1./2. malformed rows and unsupported syntax ---------------------------

@pytest.mark.parametrize(
    "kwargs, section",
    [
        ({"bus": BUS.replace("\n", " ")}, "bus"),                          # two rows, one line
        ({"bus": BUS.replace("\t", ",")}, "bus"),                          # commas
        ({"gen": "\t1\t0\t0\t300\t-300 ...\n\t1\t100\t1\t250\t10;"}, "gen"),  # continuation
        ({"bus": BUS + "\n\t3\tX\t0\t0\t0\t0\t1\t1\t0\t230\t1\t1.1\t0.9;"}, "bus"),  # non-numeric
        ({"bus": "\t1\t3\t0\t0;"}, "bus"),                                 # too few columns
        ({"gen": "\t1\t0\t0;"}, "gen"),
        ({"branch": "\t1\t2\t0.01;"}, "branch"),
        ({"gencost": "\t2\t0\t0;"}, "gencost"),
        ({"bus": BUS + "\n\t3" + "\t1" * 17 + ";"}, "bus"),               # 18 bus columns
        ({"bus": BUS.replace("\t1\t3\t", "\t1.5\t3\t", 1)}, "bus"),         # non-integral bus_i
        ({"gen": GEN.replace("\t100\t1\t", "\t100\tInf\t")}, "gen"),        # Inf status
        ({"branch": BRANCH.replace("\t0\t1\t-360", "\t0\tNaN\t-360")}, "branch"),  # NaN status
    ],
    ids=["rows-one-line", "commas", "continuation", "non-numeric", "short-bus", "short-gen",
         "short-branch", "short-gencost", "bus-18-cols", "int-col-1.5", "int-col-inf",
         "int-col-nan"],
)
def test_malformed_input_raises_naming_section(tmp_path, kwargs, section):
    with pytest.raises(ValueError, match=rf"mpc\.{section}\b"):
        parse_matpower(_write(tmp_path, _case(**kwargs)))


def test_non_numeric_row_reports_row_number(tmp_path):
    bad = BUS + "\n\t3\tX\t0\t0\t0\t0\t1\t1\t0\t230\t1\t1.1\t0.9;"
    with pytest.raises(ValueError, match=r"mpc\.bus row 3"):
        parse_matpower(_write(tmp_path, _case(bus=bad)))


def test_unterminated_block_raises(tmp_path):
    text = _case(drop=("gencost",)).replace("\n];\n\n%% branch data", "\n\n%% branch data")
    with pytest.raises(ValueError, match="terminated"):
        parse_matpower(_write(tmp_path, text))


def test_missing_bus_section_raises(tmp_path):
    with pytest.raises(ValueError, match="mpc.bus"):
        parse_matpower(_write(tmp_path, _case(drop=("bus",))))


def test_missing_gen_and_branch_give_empty_lists(tmp_path):
    net = parse_matpower(_write(tmp_path, _case(drop=("gen", "branch", "gencost"))))
    assert net.generators == [] and net.branches == [] and net.gencost == []


def test_partial_bus_solution_columns(tmp_path):
    net = parse_matpower(_write(tmp_path, _case(bus=BUS.replace("0.9;", "0.9\t6.5;", 1))))
    b = net.buses[0]
    assert b.lam_P == 6.5 and b.lam_Q is None and b.mu_Vmax is None and b.mu_Vmin is None


def test_integral_floats_accepted_in_integer_columns(tmp_path):
    net = parse_matpower(_write(tmp_path, _case(bus=BUS.replace("\t1\t3\t", "\t1.0\t3.0\t", 1))))
    assert net.buses[0].bus_i == 1 and isinstance(net.buses[0].bus_i, int)


# --- line endings ------------------------------------------------------------

def test_crlf_reads_like_lf_and_writer_emits_lf(tmp_path):
    lf = parse_matpower(_write(tmp_path, _case(), "lf.m"))
    crlf = parse_matpower(_write(tmp_path, _case(), "crlf.m", crlf=True))
    assert crlf.buses == lf.buses and crlf.header_comments == lf.header_comments
    out = tmp_path / "out.m"
    write_matpower(crlf, out)
    assert b"\r" not in out.read_bytes()


# --- 3. Inf / NaN ----------------------------------------------------------

def test_inf_nan_round_trip(tmp_path):
    net = parse_matpower(_write(tmp_path, _case()))
    net.branches[0].rateA = math.inf
    net.generators[0].Qmin = -math.inf
    net.buses[1].Pd = math.nan
    out = tmp_path / "out.m"
    write_matpower(net, out)
    back = parse_matpower(out)
    assert back.branches[0].rateA == math.inf
    assert back.generators[0].Qmin == -math.inf
    assert math.isnan(back.buses[1].Pd)


def test_solution_columns_keep_precision(tmp_path):
    net = parse_matpower(_write(tmp_path, _case(bus=BUS.replace("0.9;", "0.9\t6.5\t0\t1e-07\t0.000123456789;", 1))))
    out = tmp_path / "out.m"
    write_matpower(net, out)
    b = parse_matpower(out).buses[0]
    assert math.isclose(b.mu_Vmax, 1e-07, rel_tol=1e-9)
    assert math.isclose(b.mu_Vmin, 0.000123456789, rel_tol=1e-9)


# --- 4. idempotent writing ---------------------------------------------------

EXTRAS = (
    "%% generator fuel type\nmpc.genfuel = {\n\t'coal';\n};\n\n"
    "mpc.areas = [\n\t1\t1;\n];"
)


def test_round_trip_text_is_idempotent(tmp_path):
    src = _write(tmp_path, _case(extra=EXTRAS), crlf=True)
    a, b, c = (tmp_path / n for n in ("a.m", "b.m", "c.m"))
    write_matpower(parse_matpower(src), a)
    write_matpower(parse_matpower(a), b)
    write_matpower(parse_matpower(b), c)
    assert a.read_text() == b.read_text() == c.read_text()
    assert a.read_text().count("MATPOWER Case Format") == 1


@pytest.mark.skipif(not ACTIVSG200.exists(), reason="ExaGO datafiles not present")
def test_real_case_round_trip_is_idempotent(tmp_path):
    a, b = tmp_path / "a.m", tmp_path / "b.m"
    write_matpower(parse_matpower(ACTIVSG200), a)
    write_matpower(parse_matpower(a), b)
    assert a.read_text() == b.read_text()


def test_section_comments_stay_with_their_section(tmp_path):
    out = tmp_path / "out.m"
    write_matpower(parse_matpower(_write(tmp_path, _case(extra=EXTRAS))), out)
    text = out.read_text()
    assert text.index("%% generator data") < text.index("mpc.gen =")
    assert text.index("];", text.index("mpc.bus =")) < text.index("%% generator data")
    assert text.index("%% generator fuel type") < text.index("mpc.genfuel =")


# --- casename / function line ------------------------------------------------

def _net(header: str, casename: str = "mine") -> MATNetwork:
    return MATNetwork(
        casename=casename, version="2", baseMVA=100.0,
        buses=[Bus(1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9)],
        generators=[Generator(1, 0.0, 0.0, 300.0, -300.0, 1.0, 100.0, 1, 250.0, 10.0)],
        branches=[Branch(1, 1, 0.01, 0.1, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 1, -360.0, 360.0)],
        gencost=[GenCost(2, 0.0, 0.0, 3, [0.11, 5.0, 150.0])],
        header_comments=header,
    )


def test_casename_wins_over_header_function_line(tmp_path):
    out = tmp_path / "out.m"
    write_matpower(_net("function mpc = other\n%comment\n", casename="mine"), out)
    text = out.read_text()
    assert "function mpc = mine" in text and "function mpc = other" not in text
    assert text.count("function mpc") == 1
    assert parse_matpower(out).casename == "mine"


def test_empty_header_writes_function_line_first(tmp_path):
    out = tmp_path / "out.m"
    write_matpower(_net(""), out)
    assert out.read_text().splitlines()[0] == "function mpc = mine"
    assert parse_matpower(out).casename == "mine"


def test_writer_does_not_mutate_its_input(tmp_path):
    net = parse_matpower(_write(tmp_path, _case(extra=EXTRAS)))
    before = copy.deepcopy(net)
    write_matpower(net, tmp_path / "out.m")
    assert net == before
