"""Network summary generator for LLM prompt context."""

from __future__ import annotations

from collections import Counter

from agentigrid.parsers.matpower_model import MATNetwork


_BUS_TYPE_NAMES = {1: "PQ", 2: "PV", 3: "Ref", 4: "Isolated"}


def network_summary(net: MATNetwork) -> str:
    """Generate a human-readable summary of the network for LLM context.

    Returns a compact string (typically 30-60 lines) describing the
    network topology, generator fleet, load distribution, and branch
    statistics.
    """
    lines: list[str] = []
    lines.append(f"=== Network Summary: {net.casename} ===")
    lines.append(f"Base MVA: {net.baseMVA}")
    lines.append(f"Buses: {len(net.buses)}  |  Generators: {len(net.generators)}  |  Branches: {len(net.branches)}")
    lines.append("")

    # --- Bus statistics ---
    type_counts = Counter(b.type for b in net.buses)
    kv_counts = Counter(b.baseKV for b in net.buses)
    area_counts = Counter(b.area for b in net.buses)

    lines.append("Bus types:")
    for t in sorted(type_counts):
        lines.append(f"  Type {t} ({_BUS_TYPE_NAMES.get(t, '?')}): {type_counts[t]}")

    lines.append("Voltage levels (kV):")
    for kv in sorted(kv_counts):
        lines.append(f"  {kv} kV: {kv_counts[kv]} buses")

    lines.append(f"Areas: {sorted(area_counts.keys())}")
    lines.append("")

    # --- Generator list ---
    # Try to infer fuel type from extra_sections
    fuel_types: list[str] = []
    if "genfuel" in net.extra_sections:
        raw = net.extra_sections["genfuel"]
        for line in raw.split("\n"):
            stripped = line.strip().strip("';")
            if stripped and not stripped.startswith("%") and not stripped.startswith("mpc.") and stripped not in ("{", "}"):
                fuel_types.append(stripped)

    lines.append("Generators:")
    lines.append(f"  {'Bus':>5}  {'Pmin':>8}  {'Pmax':>8}  {'Pg':>8}  {'Status':>6}  {'Fuel'}")
    lines.append(f"  {'---':>5}  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>6}  {'---'}")
    for i, g in enumerate(net.generators):
        fuel = fuel_types[i] if i < len(fuel_types) else "?"
        status_str = "ON" if g.status == 1 else "OFF"
        lines.append(f"  {g.bus:>5}  {g.Pmin:>8.2f}  {g.Pmax:>8.2f}  {g.Pg:>8.2f}  {status_str:>6}  {fuel}")

    # Generation totals
    total_pg = sum(g.Pg for g in net.generators)
    total_pmax = sum(g.Pmax for g in net.generators if g.status == 1)
    online = sum(1 for g in net.generators if g.status == 1)
    lines.append(f"  Total Pg: {total_pg:.2f} MW  |  Online capacity: {total_pmax:.2f} MW  |  Online: {online}/{len(net.generators)}")
    lines.append("")

    # --- Branch statistics ---
    n_lines = sum(1 for br in net.branches if br.ratio == 0)
    n_xfmr = sum(1 for br in net.branches if br.ratio != 0)
    lines.append("Branches:")
    lines.append(f"  Lines: {n_lines}  |  Transformers: {n_xfmr}")
    lines.append("")

    # --- Load summary by area ---
    area_pd: dict[int, float] = {}
    area_qd: dict[int, float] = {}
    for b in net.buses:
        area_pd[b.area] = area_pd.get(b.area, 0.0) + b.Pd
        area_qd[b.area] = area_qd.get(b.area, 0.0) + b.Qd

    total_pd = sum(area_pd.values())
    total_qd = sum(area_qd.values())
    lines.append("Load by area:")
    for area in sorted(area_pd):
        lines.append(f"  Area {area}: Pd={area_pd[area]:.2f} MW, Qd={area_qd[area]:.2f} MVAr")
    lines.append(f"  Total: Pd={total_pd:.2f} MW, Qd={total_qd:.2f} MVAr")

    return "\n".join(lines)
