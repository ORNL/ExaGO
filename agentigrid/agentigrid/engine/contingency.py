"""Pure N-1 / N-2 contingency enumeration (C.5).

Enumerates single outages (N-1) and unordered outage pairs (N-2) drawn from the
k nearest neighbor buses of a target bus. Deterministic: the outage pool has a
fixed total order, so contingency lists and N-2 pairings are reproducible.

Feasibility semantics (judged by the caller, not here): under OPFLOW the V-band
and Rate A are in-solve hard constraints, so a contingency PASSES iff the
post-contingency OPFLOW converges to a feasible re-dispatched operating point
and FAILS iff it does not — the OPF-redispatch post-contingency model.

This module is pure: it imports only topology, the MATPOWER model, and stdlib —
never agent_loop (avoids an import cycle). All bus ids are external ``bus_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from agentigrid.engine import topology
from agentigrid.parsers.matpower_model import Branch, MATNetwork

# Stable kind ordering for the pool total order.
_KIND_RANK = {"branch": 0, "gen": 1, "load": 2}


@dataclass(frozen=True)
class OutageElement:
    """A single element that can be taken out of service.

    ``neighbor_bus`` / ``hop`` attribute the element to the neighbor it was drawn
    from (for a branch touching two neighbors, the first in neighbor order),
    giving deterministic labeling.
    """

    kind: str                 # "branch" | "gen" | "load"
    neighbor_bus: int         # the neighbor this element is drawn from
    hop: int                  # hop distance of neighbor_bus from the target
    # branch fields:
    fbus: int | None = None
    tbus: int | None = None
    ckt: int | None = None    # circuit index (position in the net-wide endpoint group)
    # gen/load fields:
    bus: int | None = None
    gen_id: int | None = None

    def to_command(self) -> dict:
        """The outage command dict applied by ``apply_modifications``."""
        if self.kind == "branch":
            return {
                "action": "set_branch_status",
                "fbus": self.fbus,
                "tbus": self.tbus,
                "ckt": self.ckt,
                "status": 0,
            }
        if self.kind == "gen":
            return {
                "action": "set_gen_status",
                "bus": self.bus,
                "gen_id": self.gen_id,
                "status": 0,
            }
        if self.kind == "load":
            return {
                "action": "set_load",
                "bus": self.bus,
                "Pd": 0,
                "Qd": 0,
            }
        raise ValueError(f"Unknown outage kind: {self.kind!r}")

    def label(self) -> str:
        """Short human-readable label, e.g. 'branch 69-77', 'gen@76', 'load@75'."""
        if self.kind == "branch":
            ckt_suffix = f" (ckt {self.ckt})" if self.ckt else ""
            return f"branch {self.fbus}-{self.tbus}{ckt_suffix}"
        if self.kind == "gen":
            return f"gen@{self.bus}"
        if self.kind == "load":
            return f"load@{self.bus}"
        raise ValueError(f"Unknown outage kind: {self.kind!r}")

    def _sort_key(self) -> tuple:
        """Total-order key within the pool."""
        rank = _KIND_RANK[self.kind]
        if self.kind == "branch":
            lo = min(self.fbus, self.tbus)
            hi = max(self.fbus, self.tbus)
            return (rank, lo, hi, self.ckt or 0)
        if self.kind == "gen":
            return (rank, self.bus, self.gen_id or 0)
        # load
        return (rank, self.bus, 0)


@dataclass(frozen=True)
class Contingency:
    """One contingency: an ordered tuple of 1 (N-1) or 2 (N-2) outage elements."""

    elements: tuple[OutageElement, ...]

    @property
    def order(self) -> int:
        return len(self.elements)

    def commands(self) -> list[dict]:
        return [e.to_command() for e in self.elements]

    def label(self) -> str:
        return " + ".join(e.label() for e in self.elements) + " out"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _branch_ckt(net: MATNetwork, br: Branch) -> int:
    """Circuit index of ``br`` among all net.branches sharing its unordered endpoints.

    Matches ``modifier._branch_index_in_network`` semantics: the ckt is this
    branch's position in the net-wide group of branches with the same unordered
    (fbus, tbus). Non-parallel branches get ckt 0.
    """
    lo, hi = min(br.fbus, br.tbus), max(br.fbus, br.tbus)
    pos = 0
    for other in net.branches:
        if other is br:
            return pos
        if (min(other.fbus, other.tbus), max(other.fbus, other.tbus)) == (lo, hi):
            pos += 1
    return pos


def build_component_pool(
    net: MATNetwork,
    target_bus: int,
    neighbor_count: int,
    components: tuple[str, ...] = ("branch", "gen", "load"),
) -> list[OutageElement]:
    """Deterministic, deduped single-outage pool from the k nearest neighbors.

    - branch: union over neighbors of ``topology.incident_branches``, deduped by
      (min(fbus,tbus), max(fbus,tbus), ckt). A branch between two neighbors
      appears once; it is attributed to the FIRST neighbor (in neighbor order)
      that touches it.
    - gen: one element per generator physically at each neighbor bus
      (gen_id = 0..count-1, in net.generators order at that bus).
    - load: one element per neighbor bus with a non-zero (Pd or Qd).

    Restricted to the kinds present in ``components``. The returned list is in a
    fixed total order (kind rank, then branch by (min,max,ckt), gen by
    (bus,gen_id), load by (bus,)) so N-2 pairing is reproducible.
    """
    if neighbor_count < 1:
        raise ValueError(f"neighbor_count must be >= 1, got {neighbor_count}.")

    neighbors = topology.k_nearest_by_hops(net, target_bus, neighbor_count)
    hop_of = {nb: hop for nb, hop in neighbors}
    neighbor_order = [nb for nb, _ in neighbors]

    pool: list[OutageElement] = []

    if "branch" in components:
        seen_branches: dict[tuple[int, int, int], OutageElement] = {}
        for nb in neighbor_order:
            for br in topology.incident_branches(net, nb):
                ckt = _branch_ckt(net, br)
                key = (min(br.fbus, br.tbus), max(br.fbus, br.tbus), ckt)
                if key in seen_branches:
                    continue  # attributed to the first neighbor that touched it
                seen_branches[key] = OutageElement(
                    kind="branch", neighbor_bus=nb, hop=hop_of[nb],
                    fbus=br.fbus, tbus=br.tbus, ckt=ckt,
                )
        pool.extend(seen_branches.values())

    if "gen" in components:
        for nb in neighbor_order:
            gens_at_bus = [g for g in net.generators if g.bus == nb]
            for gen_id in range(len(gens_at_bus)):
                pool.append(OutageElement(
                    kind="gen", neighbor_bus=nb, hop=hop_of[nb],
                    bus=nb, gen_id=gen_id,
                ))

    if "load" in components:
        for nb in neighbor_order:
            bus_obj = next((b for b in net.buses if b.bus_i == nb), None)
            if bus_obj is not None and (bus_obj.Pd != 0 or bus_obj.Qd != 0):
                pool.append(OutageElement(
                    kind="load", neighbor_bus=nb, hop=hop_of[nb], bus=nb,
                ))

    pool.sort(key=lambda e: e._sort_key())
    return pool


def enumerate_contingencies(
    net: MATNetwork,
    target_bus: int,
    neighbor_count: int,
    order: int,
    components: tuple[str, ...] = ("branch", "gen", "load"),
) -> list[Contingency]:
    """Enumerate contingencies of the given order over the neighbor outage pool.

    order == 1 → one Contingency per pool element.
    order == 2 → all unordered pairs (i < j) over the ordered pool (C(m,2)).

    Raises ValueError for ``order`` not in (1, 2) or ``neighbor_count`` < 1.
    Deterministic (follows the pool total order).
    """
    if order not in (1, 2):
        raise ValueError(f"contingency order must be 1 or 2, got {order}.")
    pool = build_component_pool(net, target_bus, neighbor_count, components)

    if order == 1:
        return [Contingency(elements=(e,)) for e in pool]
    return [Contingency(elements=(a, b)) for a, b in combinations(pool, 2)]


def all_generator_contingencies(net: MATNetwork) -> list[Contingency]:
    """One N-1 contingency per in-service generator, system-wide (not neighbor-scoped).

    Used by the hot-reserve / N-1 generator security screen (C.8): each committed
    unit is tripped in turn and the OPF re-solved. Only in-service units
    (status == 1) are included.

    Deterministic: generators are grouped by bus and, within a bus, indexed in
    ``net.generators`` order (``gen_id`` matches the modifier's per-bus generator
    indexing, as in ``build_component_pool``). The returned list is sorted by
    ``(bus, gen_id)``.
    """
    contingencies: list[Contingency] = []
    buses = sorted({g.bus for g in net.generators})
    for bus in buses:
        gens_at_bus = [g for g in net.generators if g.bus == bus]
        for gen_id, gen in enumerate(gens_at_bus):
            if gen.status != 1:
                continue
            elem = OutageElement(
                kind="gen", neighbor_bus=bus, hop=0, bus=bus, gen_id=gen_id,
            )
            contingencies.append(Contingency(elements=(elem,)))
    return contingencies
