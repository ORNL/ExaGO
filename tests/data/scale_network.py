#!/usr/bin/env python3
"""
Scale a MATPOWER (.m) network by replicating it n_scale times.

Usage
-----
    python scale_network_m.py <n_scale> <direction> [options]

Arguments
---------
    n_scale    Total number of grid copies, including the original (≥ 1).
    direction  'v'    – stack copies vertically   (connect buses 27-15, 115-37, 26-38, 25-42)
               'h'    – stack copies horizontally (connect buses 55-3,  59-4,  63-8,  60-5)
               'both' – random 2-D frontier growth using both connection sets

Options
-------
    --input       Input .m file          (default: ieee_118_bus_v10.m)
    --output      Output .m file         (default: scaled_<n>_<dir>.m)
    --bus-offset  Bus number increment per copy  (default: auto = ceil(max_bus/100)*100)
                  NOTE: offset 100 collides with buses 101-116 already in the 118-bus file.
                  The auto value is 200, which is always safe.
    --seed        Integer RNG seed for 'both' direction (enables reproducibility)

Bus-numbering scheme
--------------------
    Copy k  gets  bus_offset = k * bus_offset_step
    gencost rows are replicated once per copy (no bus numbers) to stay aligned with gen.
"""

import sys
import math
import random
import argparse
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Inter-copy connection topology  (original bus numbers, voltage in kV)
# ---------------------------------------------------------------------------

VERT_CONNECTIONS: List[Tuple[int, int, int]] = [
    ( 27,  15, 138),
    (115,  37, 138),
    ( 26,  38, 345),
    ( 25,  42, 138),
]

HORIZ_CONNECTIONS: List[Tuple[int, int, int]] = [
    ( 55,   3, 138),
    ( 59,   4, 138),
    ( 63,   8, 345),
    ( 60,   5, 138),
]

GENERIC_LINES: Dict[int, Dict] = {
    345: dict(r=0.003195, x=0.03710, b=0.61500, rate_a=1200.0, rate_bc=1500.0),
    138: dict(r=0.037650, x=0.12450, b=0.07030, rate_a=150.0,  rate_bc=180.0),
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_m_file(filepath: str) -> Dict:
    """
    Read a MATPOWER .m file and return its components.

    Returns a dict with keys:
        function_name : str            – the function/case name
        baseMVA       : str            – the baseMVA line (raw)
        version       : str            – mpc.version line (raw), may be empty
        bus           : list of lists  – each inner list is the whitespace-split tokens of one row
        gen           : list of lists
        gencost       : list of lists
        branch        : list of lists
        extra         : list of str    – any unrecognised mpc.X = [...] blocks, preserved verbatim
    """
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    result: Dict = {
        "function_name": "",
        "baseMVA": "mpc.baseMVA = 100.00;\n",
        "version": "",
        "bus": [],
        "gen": [],
        "gencost": [],
        "branch": [],
        "extra": [],
    }

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped.startswith("function "):
            # e.g. "function mpc = ieee_118_bus_v11"
            result["function_name"] = stripped.split("=")[-1].strip() if "=" in stripped else "mpc_case"
            i += 1
            continue

        if stripped.startswith("mpc.version"):
            result["version"] = raw
            i += 1
            continue

        if stripped.startswith("mpc.baseMVA"):
            result["baseMVA"] = raw
            i += 1
            continue

        # Detect start of a mpc.X = [ block
        if "= [" in stripped and stripped.startswith("mpc."):
            key_raw = stripped.split("=")[0].strip()   # e.g. "mpc.bus"
            key = key_raw.replace("mpc.", "")           # e.g. "bus"

            # Collect all data rows until the closing ];
            data_rows: List[List[str]] = []
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                i += 1
                if row.startswith("];"):
                    break
                if not row or row.startswith("%"):
                    continue
                # Strip inline comments
                row = row.split("%")[0].strip()
                if row:
                    data_rows.append(row.split())

            if key in ("bus", "gen", "gencost", "branch"):
                result[key] = data_rows
            else:
                # Preserve unrecognised sections verbatim
                result["extra"].append((key_raw, data_rows))
            continue

        i += 1

    return result


# ---------------------------------------------------------------------------
# Safe bus-offset calculation
# ---------------------------------------------------------------------------

def compute_safe_offset(bus_rows: List[List[str]]) -> int:
    buses = []
    for row in bus_rows:
        try:
            buses.append(int(row[0]))
        except (ValueError, IndexError):
            continue
    if not buses:
        return 100
    return math.ceil(max(buses) / 100) * 100


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def compute_layout(n_scale: int, direction: str,
                   bus_step: int) -> Dict[Tuple[int, int], int]:
    if n_scale < 1:
        raise ValueError("n_scale must be ≥ 1.")
    if direction == "v":
        return {(k, 0): k * bus_step for k in range(n_scale)}
    if direction == "h":
        return {(0, k): k * bus_step for k in range(n_scale)}
    if direction == "both":
        return _grow_both(n_scale, bus_step)
    raise ValueError(f"direction must be 'v', 'h', or 'both'; got {direction!r}")


def _grow_both(n_scale: int, bus_step: int) -> Dict[Tuple[int, int], int]:
    pos_map: Dict[Tuple[int, int], int] = {(0, 0): 0}
    for k in range(1, n_scale):
        frontier = {
            (nr, nc)
            for (r, c) in pos_map
            for (nr, nc) in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if (nr, nc) not in pos_map
        }
        chosen = random.choice(sorted(frontier))
        pos_map[chosen] = k * bus_step
    return pos_map


# ---------------------------------------------------------------------------
# Replication helpers
# ---------------------------------------------------------------------------

def _offset_col0(rows: List[List[str]], offset: int) -> List[List[str]]:
    """Return new rows with column 0 shifted by offset (bus number)."""
    out = []
    for row in rows:
        new_row = list(row)
        new_row[0] = str(int(new_row[0]) + offset)
        out.append(new_row)
    return out


def _offset_cols01(rows: List[List[str]], offset: int) -> List[List[str]]:
    """Return new rows with columns 0 and 1 shifted by offset (from/to bus)."""
    out = []
    for row in rows:
        new_row = list(row)
        new_row[0] = str(int(new_row[0]) + offset)
        new_row[1] = str(int(new_row[1]) + offset)
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# Inter-copy branch generation
# ---------------------------------------------------------------------------

def _make_branch_row(from_bus: int, to_bus: int, params: Dict) -> List[str]:
    """Return a branch data row (list of strings) for an inter-copy connection."""
    return [
        str(from_bus),
        str(to_bus),
        f"{params['r']:.6f}",
        f"{params['x']:.6f}",
        f"{params['b']:.6f}",
        f"{params['rate_a']:.1f}",
        f"{params['rate_bc']:.1f}",
        f"{params['rate_bc']:.1f}",
        "0",
        "0",
        "1",
        "-360",
        "360",
    ]


def build_inter_copy_branches(pos_map: Dict[Tuple[int, int], int]) -> List[List[str]]:
    records: List[List[str]] = []
    for (r, c) in pos_map:
        off_here = pos_map[(r, c)]

        if (r + 1, c) in pos_map:
            off_above = pos_map[(r + 1, c)]
            for bus_up, bus_dn, kv in VERT_CONNECTIONS:
                records.append(_make_branch_row(
                    bus_up + off_above,
                    bus_dn + off_here,
                    GENERIC_LINES[kv],
                ))

        if (r, c + 1) in pos_map:
            off_right = pos_map[(r, c + 1)]
            for bus_rt, bus_lt, kv in HORIZ_CONNECTIONS:
                records.append(_make_branch_row(
                    bus_rt + off_right,
                    bus_lt + off_here,
                    GENERIC_LINES[kv],
                ))

    return records


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def _rows_to_text(rows: List[List[str]], indent: str = "\t") -> str:
    lines = []
    for row in rows:
        lines.append(indent + "\t".join(row))
    return "\n".join(lines) + "\n" if lines else ""


def write_m_file(
    filepath: str,
    parsed: Dict,
    bus_rows: List[List[str]],
    gen_rows: List[List[str]],
    gencost_rows: List[List[str]],
    branch_rows: List[List[str]],
) -> None:
    func_name = parsed["function_name"] or "mpc_scaled"
    # Derive function name from output filename (without extension)
    import os
    base = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, "w") as fh:
        fh.write(f"function mpc = {base}\n")
        if parsed["version"]:
            fh.write(parsed["version"])
        fh.write(parsed["baseMVA"])
        fh.write("\n")

        fh.write("%% bus data\n")
        fh.write("mpc.bus = [\n")
        fh.write(_rows_to_text(bus_rows))
        fh.write("];\n\n")

        fh.write("%% gen data\n")
        fh.write("mpc.gen = [\n")
        fh.write(_rows_to_text(gen_rows))
        fh.write("];\n\n")

        fh.write("%% generator cost data\n")
        fh.write("mpc.gencost = [\n")
        fh.write(_rows_to_text(gencost_rows))
        fh.write("];\n\n")

        fh.write("%% branch data\n")
        fh.write("mpc.branch = [\n")
        fh.write(_rows_to_text(branch_rows))
        fh.write("];\n\n")

        for key_raw, extra_rows in parsed.get("extra", []):
            fh.write(f"%% {key_raw} data\n")
            fh.write(f"{key_raw} = [\n")
            fh.write(_rows_to_text(extra_rows))
            fh.write("];\n\n")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_m_file(filepath: str) -> None:
    """
    Re-parse the written file, fix PSS/E-style consistency issues, rewrite if needed.

    Checks:
      1. Multiple slack buses (type 3) → keep lowest-numbered, demote rest to type 2.
      2. Generator bus typed PQ (type 1) → promote to PV (type 2).
      3. PV bus (type 2) with no generator → demote to PQ (type 1).
      4. Isolated buses (BFS over branch edges).
    """
    parsed = parse_m_file(filepath)
    bus_rows = [list(r) for r in parsed["bus"]]
    gen_rows = parsed["gen"]

    bus_index: Dict[int, int] = {}
    bus_type:  Dict[int, int] = {}
    for idx, row in enumerate(bus_rows):
        num   = int(row[0])
        btype = int(row[1])
        bus_index[num] = idx
        bus_type[num]  = btype

    gen_buses: set = set()
    for row in gen_rows:
        gen_buses.add(int(row[0]))

    fixes: List[str] = []

    # 1. Multiple slacks
    slack_buses = sorted(b for b, t in bus_type.items() if t == 3)
    if len(slack_buses) > 1:
        keeper = slack_buses[0]
        for b in slack_buses[1:]:
            idx = bus_index[b]
            bus_rows[idx][1] = "2"
            bus_type[b] = 2
            fixes.append(f"  Slack→PV  bus {b:6d}  (slack kept at bus {keeper})")

    # 2. Generator bus typed PQ → promote to PV
    for b in sorted(gen_buses):
        if bus_type.get(b) == 1:
            idx = bus_index[b]
            bus_rows[idx][1] = "2"
            bus_type[b] = 2
            fixes.append(f"  PQ→PV     bus {b:6d}  (has generator, was typed PQ)")

    # 3. PV bus with no generator → demote to PQ
    for b, t in sorted(bus_type.items()):
        if t == 2 and b not in gen_buses:
            idx = bus_index[b]
            bus_rows[idx][1] = "1"
            bus_type[b] = 1
            fixes.append(f"  PV→PQ     bus {b:6d}  (no generator, was typed PV)")

    # 4. Connectivity check
    adj: Dict[int, List[int]] = {b: [] for b in bus_index}
    for row in parsed["branch"]:
        try:
            a, b2 = int(row[0]), int(row[1])
            if a in adj and b2 in adj:
                adj[a].append(b2)
                adj[b2].append(a)
        except (ValueError, IndexError):
            pass

    if bus_index:
        visited: set = set()
        queue = [min(bus_index)]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(nb for nb in adj.get(node, []) if nb not in visited)
        isolated = sorted(set(bus_index) - visited)
        if isolated:
            fixes.append(
                f"  ISOLATED  {len(isolated)} buses unreachable from bus {min(bus_index)}: "
                + ", ".join(str(b) for b in isolated[:10])
                + (" ..." if len(isolated) > 10 else "")
            )

    if fixes:
        print(f"\nPostprocessing {filepath}:")
        for msg in fixes:
            print(msg)
        write_m_file(
            filepath, parsed,
            bus_rows,
            parsed["gen"],
            parsed["gencost"],
            parsed["branch"],
        )
        print(f"  → {len(fixes)} issue(s) corrected, file rewritten.")
    else:
        print(f"\nPostprocessing {filepath}: no issues found.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    pos_map: Dict[Tuple[int, int], int],
    n_scale: int,
    direction: str,
    bus_step: int,
    output_path: str,
) -> None:
    n = len(pos_map)
    positions = list(pos_map)
    rmin = min(r for r, _ in positions)
    rmax = max(r for r, _ in positions)
    cmin = min(c for _, c in positions)
    cmax = max(c for _, c in positions)

    n_vert_edges  = sum(1 for (r, c) in pos_map if (r + 1, c) in pos_map)
    n_horiz_edges = sum(1 for (r, c) in pos_map if (r, c + 1) in pos_map)
    n_intercopy   = n_vert_edges * len(VERT_CONNECTIONS) \
                  + n_horiz_edges * len(HORIZ_CONNECTIONS)

    print(f"\n{'─'*55}")
    print(f"  Output file   : {output_path}")
    print(f"  Direction     : {direction}")
    print(f"  Copies        : {n}  (n_scale = {n_scale})")
    print(f"  Bus offset    : {bus_step} per copy")
    print(f"  ~Buses        : {n} × 118 ≈ {n * 118}")
    print(f"  ~Branches     : {n} × 186 + {n_intercopy} inter-copy = "
          f"~{n * 186 + n_intercopy}")
    print(f"  Grid span     : rows {rmin}..{rmax},  cols {cmin}..{cmax}")

    if direction == "both" or (rmax - rmin > 0 and cmax - cmin > 0):
        print("  Layout  (offset at each cell, blank = empty):")
        for r in range(rmax, rmin - 1, -1):
            row_str = "    "
            for c in range(cmin, cmax + 1):
                if (r, c) in pos_map:
                    row_str += f"[{pos_map[(r,c)]:5d}]"
                else:
                    row_str += "       "
            print(row_str)
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_network(
    filepath: str,
    n_scale: int,
    direction: str,
    output_path: str,
    bus_offset: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    parsed = parse_m_file(filepath)

    auto_step = compute_safe_offset(parsed["bus"])
    if bus_offset is None:
        bus_step = auto_step
    else:
        bus_step = bus_offset
        if bus_step < auto_step:
            print(
                f"WARNING: --bus-offset {bus_step} is smaller than the "
                f"computed safe minimum {auto_step}. "
                f"Bus number collisions will occur between copies."
            )

    pos_map = compute_layout(n_scale, direction, bus_step)

    all_bus:     List[List[str]] = []
    all_gen:     List[List[str]] = []
    all_gencost: List[List[str]] = []
    all_branch:  List[List[str]] = []

    for _pos, offset in pos_map.items():
        all_bus     += _offset_col0 (parsed["bus"],     offset)
        all_gen     += _offset_col0 (parsed["gen"],     offset)
        all_gencost += list(parsed["gencost"])           # no bus numbers, replicate verbatim
        all_branch  += _offset_cols01(parsed["branch"], offset)

    all_branch += build_inter_copy_branches(pos_map)

    write_m_file(output_path, parsed, all_bus, all_gen, all_gencost, all_branch)
    postprocess_m_file(output_path)
    _print_summary(pos_map, n_scale, direction, bus_step, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale a MATPOWER .m network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "n_scale", type=int,
        help="Total number of copies including the original (≥ 1).",
    )
    parser.add_argument(
        "direction", choices=["v", "h", "both"],
        help="'v' vertical, 'h' horizontal, 'both' random 2-D growth.",
    )
    parser.add_argument(
        "--input", default="ieee_118_bus_v10.m",
        help="Input .m file (default: ieee_118_bus_v10.m).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .m file (default: scaled_<n>_<dir>.m).",
    )
    parser.add_argument(
        "--bus-offset", type=int, default=None, dest="bus_offset",
        help=(
            "Bus number increment per copy. "
            "Default: auto = ceil(max_bus / 100) × 100 (200 for the 118-bus file). "
            "Values < 200 will collide with existing buses 101–116."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for 'both' direction (reproducible layouts).",
    )
    args = parser.parse_args()

    output = args.output or f"scaled_{args.n_scale}_{args.direction}.m"

    build_network(
        filepath=args.input,
        n_scale=args.n_scale,
        direction=args.direction,
        output_path=output,
        bus_offset=args.bus_offset,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
