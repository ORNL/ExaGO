"""Network summary generator for LLM prompt context."""

from __future__ import annotations

from collections import Counter

from agentigrid.parsers.matpower_model import MATNetwork


_BUS_TYPE_NAMES = {1: "PQ", 2: "PV", 3: "Ref", 4: "Isolated"}


def network_summary(net: MATNetwork, max_generators: int = 40) -> str:
    """Generate a human-readable summary of the network for LLM context.

    Returns a compact string (typically 30-60 lines) describing the
    network topology, generator fleet, load distribution, and branch
    statistics.

    The generator listing is bounded so the summary stays O(1) in network
    size (it is injected into the LLM system prompt). When the fleet has more
    than ``max_generators`` units, only the top ``max_generators`` by ``Pmax``
    are listed (with a fuel-mix histogram and an omitted-count marker); at or
    below the cap the full fleet is listed in original file order and the
    output is byte-identical to the un-capped version.
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

    def _fuel_for(idx: int) -> str:
        return fuel_types[idx] if idx < len(fuel_types) else "?"

    lines.append("Generators:")
    lines.append(f"  {'Bus':>5}  {'Pmin':>8}  {'Pmax':>8}  {'Pg':>8}  {'Status':>6}  {'Fuel'}")
    lines.append(f"  {'---':>5}  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>6}  {'---'}")

    n_gen = len(net.generators)
    if n_gen <= max_generators:
        # Small fleet: original file order, full list — byte-identical to before.
        listed = list(enumerate(net.generators))
        omitted = 0
    else:
        # Large fleet: top-N by Pmax descending (tie-break original index for
        # determinism). Keeps the system prompt O(1) in network size.
        indexed = sorted(
            enumerate(net.generators), key=lambda ig: (-ig[1].Pmax, ig[0])
        )
        listed = indexed[:max_generators]
        omitted = n_gen - max_generators

    for i, g in listed:
        status_str = "ON" if g.status == 1 else "OFF"
        lines.append(
            f"  {g.bus:>5}  {g.Pmin:>8.2f}  {g.Pmax:>8.2f}  {g.Pg:>8.2f}  "
            f"{status_str:>6}  {_fuel_for(i)}"
        )

    if omitted > 0:
        lines.append(
            f"  ... ({omitted} more generators omitted; "
            "full list in JSON journal / .m file)"
        )
        # Fuel-mix histogram over the WHOLE fleet, so the truncated view still
        # conveys the fleet composition. Only emitted when truncating, so
        # at-or-below-cap output stays unchanged.
        fuel_mix = Counter(_fuel_for(i) for i in range(n_gen))
        mix_str = ", ".join(
            f"{fuel}={count}"
            for fuel, count in sorted(fuel_mix.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        lines.append(f"  Fuel mix (all {n_gen} gens): {mix_str}")

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
