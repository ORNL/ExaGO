"""Deterministic graph-topology layer for power-grid networks.

Substations (buses joined by transformers) and the substation tiers around a
start bus, used by the contingency neighbor search; plus the branches incident
to a bus.

Pure module — no imports from agent_loop (avoids circular dependency).
All bus identifiers are external bus numbers (Bus.bus_i == Branch.fbus/tbus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from agentigrid.parsers.matpower_model import Branch, MATNetwork


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bus_set(net: MATNetwork) -> set[int]:
    return {b.bus_i for b in net.buses}


# ---------------------------------------------------------------------------
# Graph functions
# ---------------------------------------------------------------------------

def is_transformer(br: Branch) -> bool:
    """True for a transformer branch (MATPOWER TAP column != 0), False for a line.

    MATPOWER convention: ``ratio == 0`` marks a line; any other value (including
    1.0) marks a transformer. Phase shift and base-kV differences are not used.
    """
    return br.ratio != 0


def substation_map(
    net: MATNetwork, include_out_of_service: bool = False,
) -> dict[int, int]:
    """Group buses into substations: buses joined by transformers share one substation.

    Returns ``{bus_i: substation_id}`` for every bus, where the substation id is
    the smallest bus number in the group. A case with no transformers maps each
    bus to itself. Branches with status == 0 are ignored unless
    include_out_of_service=True.
    """
    parent: dict[int, int] = {b.bus_i: b.bus_i for b in net.buses}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for br in net.branches:
        if not include_out_of_service and br.status == 0:
            continue
        if not is_transformer(br) or br.fbus == br.tbus:
            continue
        if br.fbus not in parent or br.tbus not in parent:
            continue
        ra, rb = find(br.fbus), find(br.tbus)
        if ra != rb:
            # Keep the smaller bus number as the root → stable substation ids.
            lo, hi = min(ra, rb), max(ra, rb)
            parent[hi] = lo

    return {bus: find(bus) for bus in parent}


@dataclass
class SubstationScope:
    """Substations within a given number of line hops of the start substation(s).

    ``tiers[0]`` holds the start substation(s); ``tiers[d]`` the substations first
    reached across ``d`` lines (transformers add no hop). Substation ids are the
    smallest bus number in each substation.
    """

    start_buses: tuple[int, ...]
    depth: int
    tiers: list[list[int]]
    substation_of: dict[int, int]   # bus -> substation id, buses in scope only
    tier_of: dict[int, int]         # bus -> tier, buses in scope only

    @property
    def buses(self) -> list[int]:
        return sorted(self.substation_of)


def substations_within(
    net: MATNetwork,
    start_buses: list[int] | tuple[int, ...],
    depth: int,
    include_out_of_service: bool = False,
) -> SubstationScope:
    """All substations within ``depth`` line hops of the substation(s) of ``start_buses``.

    Buses joined by a transformer belong to the same substation (0 hops); each
    line between two substations is 1 hop. For a changed branch pass both end
    buses, so the tiers grow outward on both sides. There is no cap on the number
    of substations found.

    Raises ValueError for an unknown bus, an empty start list, or depth < 0.
    """
    if not start_buses:
        raise ValueError("start_buses must contain at least one bus.")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}.")
    bus_set = _build_bus_set(net)
    for b in start_buses:
        if b not in bus_set:
            raise ValueError(f"Bus {b} is not in the network.")

    sub_of = substation_map(net, include_out_of_service)

    # Substation graph: one edge per pair of substations joined by at least one line.
    sub_adj: dict[int, set[int]] = {s: set() for s in set(sub_of.values())}
    for br in net.branches:
        if not include_out_of_service and br.status == 0:
            continue
        if is_transformer(br) or br.fbus == br.tbus:
            continue
        if br.fbus not in sub_of or br.tbus not in sub_of:
            continue
        sa, sb = sub_of[br.fbus], sub_of[br.tbus]
        if sa != sb:
            sub_adj[sa].add(sb)
            sub_adj[sb].add(sa)

    tier0 = sorted({sub_of[b] for b in start_buses})
    tiers: list[list[int]] = [tier0]
    seen: set[int] = set(tier0)
    for _ in range(depth):
        nxt = sorted({nb for s in tiers[-1] for nb in sub_adj[s]} - seen)
        if not nxt:
            break
        tiers.append(nxt)
        seen.update(nxt)

    tier_of_sub = {s: d for d, tier in enumerate(tiers) for s in tier}
    substation_of = {bus: s for bus, s in sub_of.items() if s in tier_of_sub}
    tier_of = {bus: tier_of_sub[s] for bus, s in substation_of.items()}

    return SubstationScope(
        start_buses=tuple(start_buses),
        depth=depth,
        tiers=tiers,
        substation_of=substation_of,
        tier_of=tier_of,
    )


def incident_branches(
    net: MATNetwork,
    bus: int,
    include_out_of_service: bool = False,
) -> list[Branch]:
    """All Branch objects with fbus == bus or tbus == bus, in stable file order.

    Parallel circuits are returned as separate entries.
    Raises ValueError if bus is not in the network.
    """
    bus_set = _build_bus_set(net)
    if bus not in bus_set:
        raise ValueError(f"Bus {bus} is not in the network.")
    result = []
    for br in net.branches:
        if not include_out_of_service and br.status == 0:
            continue
        if br.fbus == bus or br.tbus == bus:
            result.append(br)
    return result


# ---------------------------------------------------------------------------
# Pure view formatters (token-bounded, LLM-facing)
# ---------------------------------------------------------------------------

def format_incident_branches_view(bus: int, branches: list[Branch]) -> str:
    """Compact, token-bounded text view of incident branches for the LLM."""
    header = f"Topology: branches incident to bus {bus} (in-service circuits).\n"
    if not branches:
        return header + f"[No in-service branches incident to bus {bus}.]\n"

    fb_w = max(4, max(len(str(br.fbus)) for br in branches))
    tb_w = max(4, max(len(str(br.tbus)) for br in branches))
    ra_w = max(5, max(len(f"{br.rateA:.0f}") for br in branches))

    header_row = f"{'fbus':>{fb_w}} | {'tbus':>{tb_w}} | {'rateA':>{ra_w}}"
    rows = [header_row]
    for br in branches:
        rows.append(f"{br.fbus:>{fb_w}} | {br.tbus:>{tb_w}} | {br.rateA:>{ra_w}.0f}")

    n = len(branches)
    note = (
        f"[{n} incident circuit(s). Parallel circuits are listed separately — each is an\n"
        f" independent branch-outage contingency.]"
    )
    return header + "\n".join(rows) + "\n" + note
