"""Bus topology of a GridKit case, from the element map.

Every Branch device (line or transformer) joins its two buses; "one bus
away" in PJM M-14B G.3.2 means one branch away.
"""

from __future__ import annotations

from collections import deque


def build_adjacency(element_map: dict) -> dict[int, set[int]]:
    """``{bus: neighbouring buses}`` over all Branch devices."""
    adj: dict[int, set[int]] = {b["bus"]: set() for b in element_map["buses"]}
    for d in element_map["devices"]:
        if d["class"] == "Branch" and len(d["buses"]) == 2:
            a, b = d["buses"]
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def buses_within(element_map: dict, bus: int, hops: int) -> dict[int, int]:
    """``{bus: branches away}`` for all buses up to *hops* away, including *bus* (0)."""
    adj = build_adjacency(element_map)
    if bus not in adj:
        raise ValueError(f"Bus {bus} is not in the case")
    dist = {bus: 0}
    queue = deque([bus])
    while queue:
        node = queue.popleft()
        if dist[node] == hops:
            continue
        for nb in sorted(adj[node]):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def fault_locations(element_map: dict, poi: int, hops: int = 1) -> list[int]:
    """PJM fault locations: the POI and every bus up to *hops* away, POI first."""
    dist = buses_within(element_map, poi, hops)
    return sorted(dist, key=lambda b: (dist[b], b))


def branches_at_bus(element_map: dict, bus: int) -> list[dict]:
    """Branch devices connected to *bus*."""
    return [d for d in element_map["devices"] if d["class"] == "Branch" and bus in d["buses"]]


def format_locations_view(element_map: dict, poi: int, hops: int = 1) -> str:
    """Short text for the LLM: the fault locations around the POI."""
    dist = buses_within(element_map, poi, hops)
    lines = [f"Fault locations around POI bus {poi} (up to {hops} bus(es) away): {len(dist)} buses"]
    for level in range(hops + 1):
        ring = sorted(b for b, d in dist.items() if d == level)
        if ring:
            label = "POI" if level == 0 else f"{level} away"
            lines.append(f"  {label}: {', '.join(str(b) for b in ring)}")
    return "\n".join(lines)
