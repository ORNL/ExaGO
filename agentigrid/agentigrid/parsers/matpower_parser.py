"""MATPOWER .m file parser."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agentigrid.parsers.matpower_model import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
)

logger = logging.getLogger("agentigrid.parsers.matpower")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"^function\s+mpc\s*=\s*(\w+)", re.MULTILINE)
_VERSION_RE = re.compile(r"mpc\.version\s*=\s*'([^']+)'")
_BASEMVA_RE = re.compile(r"mpc\.baseMVA\s*=\s*([\d.]+)")
# Match  mpc.<section> = [  or  mpc.<section> = {
_SECTION_RE = re.compile(r"^mpc\.(\w+)\s*=\s*([{\[])", re.MULTILINE)


def _parse_float(s: str) -> float:
    """Parse a string as float, handling MATLAB Inf/NaN."""
    sl = s.lower().rstrip(";")
    if sl == "inf":
        return float("inf")
    if sl == "-inf":
        return float("-inf")
    if sl == "nan":
        return float("nan")
    return float(sl)


def _split_data_row(line: str) -> list[str]:
    """Split a MATPOWER data row into tokens, stripping comments and semicolons."""
    # Remove inline comment
    idx = line.find("%")
    if idx >= 0:
        line = line[:idx]
    # Remove trailing semicolons and whitespace
    line = line.rstrip().rstrip(";").rstrip()
    return line.split()


def _extract_section_block(text: str, start: int, bracket: str) -> tuple[str, int]:
    """Extract text from *start* until the matching closing bracket.

    Returns (block_content, end_position).
    """
    close = "]" if bracket == "[" else "}"
    # Find the end of the section
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == bracket:
            depth += 1
        elif ch == close:
            depth -= 1
        i += 1
    return text[start:i - 1], i


def _collect_section_comment(text: str, section_start: int) -> str:
    """Walk backwards from *section_start* to collect preceding comment/blank lines."""
    lines = text[:section_start].split("\n")
    comment_lines: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("%") or stripped == "":
            comment_lines.append(line)
        else:
            break
    comment_lines.reverse()
    return "\n".join(comment_lines)


# ---------------------------------------------------------------------------
# Row → dataclass converters
# ---------------------------------------------------------------------------

def _row_to_bus(values: list[float]) -> Bus:
    base = dict(
        bus_i=int(values[0]),
        type=int(values[1]),
        Pd=values[2],
        Qd=values[3],
        Gs=values[4],
        Bs=values[5],
        area=int(values[6]),
        Vm=values[7],
        Va=values[8],
        baseKV=values[9],
        zone=int(values[10]),
        Vmax=values[11],
        Vmin=values[12],
    )
    if len(values) > 13:
        base["lam_P"] = values[13]
    if len(values) > 14:
        base["lam_Q"] = values[14]
    if len(values) > 15:
        base["mu_Vmax"] = values[15]
    if len(values) > 16:
        base["mu_Vmin"] = values[16]
    return Bus(**base)


def _row_to_gen(values: list[float]) -> Generator:
    return Generator(
        bus=int(values[0]),
        Pg=values[1],
        Qg=values[2],
        Qmax=values[3],
        Qmin=values[4],
        Vg=values[5],
        mBase=values[6],
        status=int(values[7]),
        Pmax=values[8],
        Pmin=values[9],
        extra=list(values[10:]),
    )


def _row_to_branch(values: list[float]) -> Branch:
    return Branch(
        fbus=int(values[0]),
        tbus=int(values[1]),
        r=values[2],
        x=values[3],
        b=values[4],
        rateA=values[5],
        rateB=values[6],
        rateC=values[7],
        ratio=values[8],
        angle=values[9],
        status=int(values[10]),
        angmin=values[11],
        angmax=values[12],
        extra=list(values[13:]),
    )


def _row_to_gencost(values: list[float]) -> GenCost:
    return GenCost(
        model=int(values[0]),
        startup=values[1],
        shutdown=values[2],
        ncost=int(values[3]),
        coeffs=list(values[4:]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_matpower(path: Path) -> MATNetwork:
    """Parse a MATPOWER .m file into a MATNetwork object.

    Args:
        path: Path to the .m file.

    Returns:
        MATNetwork with all data sections populated.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a valid MATPOWER case.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MATPOWER file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # --- Function name ---
    m = _FUNC_RE.search(text)
    if not m:
        raise ValueError(f"Cannot find 'function mpc = ...' in {path}")
    casename = m.group(1)

    # --- Version ---
    m = _VERSION_RE.search(text)
    version = m.group(1) if m else "2"

    # --- baseMVA ---
    m = _BASEMVA_RE.search(text)
    if not m:
        raise ValueError(f"Cannot find mpc.baseMVA in {path}")
    baseMVA = float(m.group(1))

    # --- Header comments (everything before mpc.version or mpc.baseMVA) ---
    first_mpc = text.find("mpc.")
    header_end = text.rfind("\n", 0, first_mpc) + 1 if first_mpc > 0 else 0
    header_comments = text[:header_end]

    # --- Find all sections ---
    buses: list[Bus] = []
    generators: list[Generator] = []
    branches: list[Branch] = []
    gencost: list[GenCost] = []
    extra_sections: dict[str, str] = {}

    # Collect section comment blocks for known sections
    _section_comments: dict[str, str] = {}

    for match in _SECTION_RE.finditer(text):
        section_name = match.group(1)
        bracket = match.group(2)
        block_start = match.end()

        # Skip scalar assignments (version, baseMVA) already handled
        if section_name in ("version", "baseMVA"):
            continue

        block_content, _ = _extract_section_block(text, block_start, bracket)

        # Collect preceding comment
        line_start = text.rfind("\n", 0, match.start()) + 1
        comment = _collect_section_comment(text, line_start)
        _section_comments[section_name] = comment

        if bracket == "{":
            # Cell array (bus_name, gentype, genfuel, etc.) → store as raw
            raw_text = comment + "\n" + text[match.start():block_start] + block_content + ("}" if bracket == "{" else "]")
            extra_sections[section_name] = raw_text
            continue

        # Numeric matrix section — parse rows
        rows: list[list[float]] = []
        for line in block_content.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            tokens = _split_data_row(line)
            if not tokens:
                continue
            try:
                rows.append([_parse_float(t) for t in tokens])
            except ValueError:
                logger.warning("Skipping unparseable row in mpc.%s: %s", section_name, line.strip())

        if section_name == "bus":
            buses = [_row_to_bus(r) for r in rows]
            logger.debug("Parsed %d buses", len(buses))
        elif section_name == "gen":
            generators = [_row_to_gen(r) for r in rows]
            logger.debug("Parsed %d generators", len(generators))
        elif section_name == "branch":
            branches = [_row_to_branch(r) for r in rows]
            logger.debug("Parsed %d branches", len(branches))
        elif section_name == "gencost":
            gencost = [_row_to_gencost(r) for r in rows]
            logger.debug("Parsed %d gencost entries", len(gencost))
        else:
            # Unknown numeric section → store as raw
            raw_text = comment + "\n" + text[match.start():block_start] + block_content + "]"
            extra_sections[section_name] = raw_text
            logger.debug("Stored unknown section mpc.%s as raw text", section_name)

    # Store section comments on the network for the writer
    net = MATNetwork(
        casename=casename,
        version=version,
        baseMVA=baseMVA,
        buses=buses,
        generators=generators,
        branches=branches,
        gencost=gencost,
        header_comments=header_comments,
        extra_sections=extra_sections,
    )
    # Attach section comments as a private attribute for the writer
    net._section_comments = _section_comments  # type: ignore[attr-defined]

    logger.info(
        "Parsed %s: %d buses, %d generators, %d branches, %d gencost",
        casename, len(buses), len(generators), len(branches), len(gencost),
    )
    return net
