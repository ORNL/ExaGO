"""MATPOWER .m file writer."""

from __future__ import annotations

import logging
from pathlib import Path

from agentigrid.parsers.matpower_model import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
)

logger = logging.getLogger("agentigrid.parsers.matpower")


def _fmt(value: float) -> str:
    """Format a numeric value, writing integers without decimals."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


def _write_bus_row(b: Bus) -> str:
    parts = [
        str(b.bus_i), str(b.type), _fmt(b.Pd), _fmt(b.Qd),
        _fmt(b.Gs), _fmt(b.Bs), str(b.area), _fmt(b.Vm), _fmt(b.Va),
        _fmt(b.baseKV), str(b.zone), _fmt(b.Vmax), _fmt(b.Vmin),
    ]
    for val in (b.lam_P, b.lam_Q, b.mu_Vmax, b.mu_Vmin):
        if val is not None:
            parts.append(f"{val:.4f}")
    return "\t" + "\t".join(parts) + ";"


def _write_gen_row(g: Generator) -> str:
    parts = [
        str(g.bus), _fmt(g.Pg), _fmt(g.Qg), _fmt(g.Qmax), _fmt(g.Qmin),
        _fmt(g.Vg), _fmt(g.mBase), str(g.status), _fmt(g.Pmax), _fmt(g.Pmin),
    ]
    for v in g.extra:
        parts.append(_fmt(v))
    return "\t" + "\t".join(parts) + ";"


def _write_branch_row(br: Branch) -> str:
    parts = [
        str(br.fbus), str(br.tbus), _fmt(br.r), _fmt(br.x), _fmt(br.b),
        _fmt(br.rateA), _fmt(br.rateB), _fmt(br.rateC),
        _fmt(br.ratio), _fmt(br.angle), str(br.status),
        _fmt(br.angmin), _fmt(br.angmax),
    ]
    for v in br.extra:
        parts.append(_fmt(v))
    return "\t" + "\t".join(parts) + ";"


def _write_gencost_row(gc: GenCost) -> str:
    parts = [str(gc.model), _fmt(gc.startup), _fmt(gc.shutdown), str(gc.ncost)]
    for v in gc.coeffs:
        parts.append(_fmt(v))
    return "\t" + "\t".join(parts) + ";"


def write_matpower(network: MATNetwork, path: Path) -> None:
    """Write a MATNetwork object to a MATPOWER .m file.

    The output file is readable by ExaGO. Formatting is close to the
    original; whitespace may differ but structure is identical.

    Args:
        network: The network data to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # --- Header ---
    lines.append(network.header_comments.rstrip("\n"))
    # Ensure function line is present (header_comments includes it)
    if f"function mpc = {network.casename}" not in network.header_comments:
        lines.append(f"function mpc = {network.casename}")

    # --- Version & baseMVA ---
    lines.append("")
    lines.append(f"%% MATPOWER Case Format : Version {network.version}")
    lines.append(f"mpc.version = '{network.version}';")
    lines.append("")
    lines.append("%%-----  Power Flow Data  -----%%")
    lines.append("%% system MVA base")
    lines.append(f"mpc.baseMVA = {_fmt(network.baseMVA)};")

    # Retrieve section comments if available
    sc: dict[str, str] = getattr(network, "_section_comments", {})

    # --- Bus data ---
    lines.append("")
    bus_comment = sc.get("bus", "%% bus data\n%\tbus_i\ttype\tPd\tQd\tGs\tBs\tarea\tVm\tVa\tbaseKV\tzone\tVmax\tVmin")
    lines.append(bus_comment.rstrip())
    lines.append("mpc.bus = [")
    for b in network.buses:
        lines.append(_write_bus_row(b))
    lines.append("];")

    # --- Generator data ---
    lines.append("")
    gen_comment = sc.get("gen", "%% generator data\n%\tbus\tPg\tQg\tQmax\tQmin\tVg\tmBase\tstatus\tPmax\tPmin")
    lines.append(gen_comment.rstrip())
    lines.append("mpc.gen = [")
    for g in network.generators:
        lines.append(_write_gen_row(g))
    lines.append("];")

    # --- Branch data ---
    lines.append("")
    branch_comment = sc.get("branch", "%% branch data\n%\tfbus\ttbus\tr\tx\tb\trateA\trateB\trateC\tratio\tangle\tstatus\tangmin\tangmax")
    lines.append(branch_comment.rstrip())
    lines.append("mpc.branch = [")
    for br in network.branches:
        lines.append(_write_branch_row(br))
    lines.append("];")

    # --- Gencost ---
    if network.gencost:
        lines.append("")
        gc_comment = sc.get("gencost", "%%-----  OPF Data  -----%%\n%% generator cost data")
        lines.append(gc_comment.rstrip())
        lines.append("mpc.gencost = [")
        for gc in network.gencost:
            lines.append(_write_gencost_row(gc))
        lines.append("];")

    # --- Extra sections (gentype, genfuel, bus_name, etc.) ---
    for section_name, raw_text in network.extra_sections.items():
        lines.append("")
        lines.append(raw_text.rstrip())

    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote MATPOWER file: %s", path)
