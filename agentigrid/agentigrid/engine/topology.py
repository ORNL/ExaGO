"""Deterministic graph-topology layer for power-grid networks.

Pure module — no imports from agent_loop (avoids circular dependency).
All bus identifiers are external bus numbers (Bus.bus_i == Branch.fbus/tbus).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from agentigrid.parsers.matpower_model import Branch, MATNetwork


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bus_set(net: MATNetwork) -> set[int]:
    return {b.bus_i for b in net.buses}


def _bfs_distances(adj: dict[int, set[int]], src: int) -> dict[int, int]:
    """BFS from src. Returns {bus: hop_distance} for all reachable buses, excluding src."""
    visited: dict[int, int] = {src: 0}
    q: deque[int] = deque([src])
    while q:
        node = q.popleft()
        for nb in sorted(adj[node]):
            if nb not in visited:
                visited[nb] = visited[node] + 1
                q.append(nb)
    del visited[src]
    return visited


# ---------------------------------------------------------------------------
# Graph functions
# ---------------------------------------------------------------------------

def build_adjacency(
    net: MATNetwork, include_out_of_service: bool = False,
) -> dict[int, set[int]]:
    """Undirected adjacency keyed by external bus number (bus_i).

    Every bus in net.buses appears as a key (isolated buses map to empty set).
    Branches with status == 0 are excluded unless include_out_of_service=True.
    Parallel branches collapse to a single edge (set semantics).
    Self-loops (fbus == tbus) are skipped.
    """
    adj: dict[int, set[int]] = {b.bus_i: set() for b in net.buses}
    for br in net.branches:
        if not include_out_of_service and br.status == 0:
            continue
        if br.fbus == br.tbus:
            continue
        adj.setdefault(br.fbus, set()).add(br.tbus)
        adj.setdefault(br.tbus, set()).add(br.fbus)
    return adj


def k_nearest_by_hops(
    net: MATNetwork,
    bus: int,
    k: int = 3,
    include_out_of_service: bool = False,
) -> list[tuple[int, int]]:
    """BFS from bus. Return up to k (neighbor_bus, hop_distance) pairs.

    Ordered by (hop_distance ASC, neighbor_bus ASC).
    Source bus is excluded. Raises ValueError for unknown bus or k < 1.
    """
    bus_set = _build_bus_set(net)
    if bus not in bus_set:
        raise ValueError(f"Bus {bus} is not in the network.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    adj = build_adjacency(net, include_out_of_service)
    distances = _bfs_distances(adj, bus)
    result = sorted(distances.items(), key=lambda x: (x[1], x[0]))
    return result[:k]


def hop_distance(
    net: MATNetwork,
    src: int,
    dst: int,
    include_out_of_service: bool = False,
) -> int | None:
    """Minimum hop distance src -> dst (0 if src == dst). None if dst unreachable.

    Raises ValueError if src or dst is not in the network.
    """
    bus_set = _build_bus_set(net)
    if src not in bus_set:
        raise ValueError(f"Bus {src} is not in the network.")
    if dst not in bus_set:
        raise ValueError(f"Bus {dst} is not in the network.")
    if src == dst:
        return 0
    adj = build_adjacency(net, include_out_of_service)
    distances = _bfs_distances(adj, src)
    return distances.get(dst)


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


def count_reachable(
    net: MATNetwork,
    bus: int,
    include_out_of_service: bool = False,
) -> int:
    """Total number of buses reachable from bus (excluding bus itself)."""
    bus_set = _build_bus_set(net)
    if bus not in bus_set:
        raise ValueError(f"Bus {bus} is not in the network.")
    adj = build_adjacency(net, include_out_of_service)
    return len(_bfs_distances(adj, bus))


# ---------------------------------------------------------------------------
# Pure view formatters (token-bounded, LLM-facing)
# ---------------------------------------------------------------------------

def format_nearest_neighbors_view(
    bus: int,
    k: int,
    neighbors: list[tuple[int, int]],
    total_reachable: int,
) -> str:
    """Compact, token-bounded text view of k-nearest neighbors for the LLM."""
    svc = "in-service branches"
    header = (
        f"Topology: {len(neighbors)} nearest neighbor bus(es) of bus {bus} "
        f"by hop count ({svc}).\n"
    )
    col_bus_w = max(3, max((len(str(nb)) for nb, _ in neighbors), default=3))
    col_hop_w = max(4, max((len(str(h)) for _, h in neighbors), default=4))
    rank_w = max(4, len(str(len(neighbors))))

    header_row = f"{'rank':>{rank_w}} | {'bus':>{col_bus_w}} | {'hops':>{col_hop_w}}"
    rows = [header_row]
    for i, (nb, hops) in enumerate(neighbors, 1):
        rows.append(f"{i:>{rank_w}} | {nb:>{col_bus_w}} | {hops:>{col_hop_w}}")

    note_lines = [
        f"[Equal-hop buses ordered by ascending bus number; source bus {bus} excluded.",
        f" {bus} can reach {total_reachable} buses in total. Adjacency uses {svc} only.",
    ]
    if len(neighbors) < k:
        note_lines.append(
            f" Only {len(neighbors)} neighbor(s) reachable (requested k={k})."
        )
    note_lines[-1] = note_lines[-1] + "]"
    # Close the bracket: remove from last line and add to end
    note = "\n".join(note_lines)

    return header + "\n".join(rows) + "\n" + note


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
