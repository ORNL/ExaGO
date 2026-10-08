"""Pure N-1 / N-2 contingency enumeration (C.5).

Two steps, kept separate so each can be checked on its own:

1. ``find_affected_elements`` — the neighbor search. From the element a mutation changed it
   walks the substations within a given number of line hops (buses joined by a
   transformer form one substation, 0 hops; each line between two substations is
   1 hop) and returns five lists: affected buses, branches, generators, loads and
   shunts, each entry tagged with its tier. The changed element is left out.
2. ``outages_from`` + ``enumerate_contingencies`` — join the lists into outage
   elements and build single outages (N-1) or unordered pairs (N-2).

Deterministic: every list and the outage pool have a fixed order, so contingency
lists and N-2 pairings are reproducible.

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

from agentigrid.exago_engine import topology
from agentigrid.exago_parsers.matpower_model import Branch, MATNetwork

# Stable kind ordering for the pool total order.
_KIND_RANK = {"branch": 0, "gen": 1, "load": 2, "shunt": 3}
OUTAGE_KINDS = tuple(_KIND_RANK)
# Line hops between substations searched when a request gives no substation_depth.
DEFAULT_SUBSTATION_DEPTH = 3


@dataclass(frozen=True)
class OutageElement:
    """A single element that can be taken out of service.

    ``tier`` is the element's line-hop distance from the changed element's substation
    (a branch takes the lower tier of its two ends).
    """

    kind: str                 # "branch" | "gen" | "load" | "shunt"
    tier: int
    # branch fields:
    fbus: int | None = None
    tbus: int | None = None
    ckt: int | None = None    # circuit index (position in the net-wide endpoint group)
    # gen/load/shunt fields:
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
        if self.kind == "shunt":
            return {
                "action": "set_shunt",
                "bus": self.bus,
                "Gs": 0,
                "Bs": 0,
            }
        raise ValueError(f"Unknown outage kind: {self.kind!r}")

    def label(self) -> str:
        """Short human-readable label, e.g. 'branch 69-77', 'gen@76', 'load@75', 'shunt@74'."""
        if self.kind == "branch":
            ckt_suffix = f" (ckt {self.ckt})" if self.ckt else ""
            return f"branch {self.fbus}-{self.tbus}{ckt_suffix}"
        if self.kind == "gen":
            id_suffix = f"#{self.gen_id}" if self.gen_id else ""
            return f"gen@{self.bus}{id_suffix}"
        if self.kind in ("load", "shunt"):
            return f"{self.kind}@{self.bus}"
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
        # load / shunt
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
# Mutation and the changed element
# ---------------------------------------------------------------------------

# Commands a contingency study may be centred on (the "mutation").
MUTATION_ACTIONS = (
    "add_load_at_bus", "add_generator_at_bus", "set_load", "set_gen_status", "set_branch_status",
)


@dataclass(frozen=True)
class ChangedElement:
    """The element the mutation adds or changes; it is never itself an outage candidate.

    load:   the (aggregate) load at ``bus``
    gen:    generator ``gen_id`` at ``bus`` (per-bus index, net.generators order)
    branch: circuit ``ckt`` between ``fbus`` and ``tbus``
    """

    kind: str
    bus: int | None = None
    gen_id: int | None = None
    fbus: int | None = None
    tbus: int | None = None
    ckt: int = 0

    @property
    def start_buses(self) -> tuple[int, ...]:
        """Buses whose substations form tier 0 (both ends for a branch)."""
        if self.kind == "branch":
            return (self.fbus, self.tbus)
        return (self.bus,)

    def matches_branch(self, entry: dict) -> bool:
        """True if ``entry`` ({"fbus", "tbus", "ckt"}) is this changed branch."""
        if self.kind != "branch":
            return False
        return (
            (min(entry["fbus"], entry["tbus"]), max(entry["fbus"], entry["tbus"]), entry["ckt"])
            == (min(self.fbus, self.tbus), max(self.fbus, self.tbus), self.ckt)
        )

    def label(self) -> str:
        if self.kind == "branch":
            ckt_suffix = f" (ckt {self.ckt})" if self.ckt else ""
            return f"branch {self.fbus}-{self.tbus}{ckt_suffix}"
        if self.kind == "gen":
            return f"gen@{self.bus}#{self.gen_id}"
        return f"load@{self.bus}"

    def to_dict(self) -> dict:
        if self.kind == "branch":
            return {"type": "branch", "fbus": self.fbus, "tbus": self.tbus, "ckt": self.ckt}
        if self.kind == "gen":
            return {"type": "gen", "bus": self.bus, "gen_id": self.gen_id}
        return {"type": "load", "bus": self.bus}


def apply_mutation(net: MATNetwork, raw: object) -> tuple[MATNetwork, ChangedElement]:
    """Apply a mutation command to a copy of ``net`` and name the element it changed.

    ``raw`` is one command from the command list, restricted to MUTATION_ACTIONS:
      add_load_at_bus / set_load  → the load at ``bus``
      add_generator_at_bus        → the new generator (last at ``bus``)
      set_gen_status              → generator ``gen_id`` (default 0) at ``bus``
      set_branch_status           → circuit ``ckt`` (default 0) of ``fbus``-``tbus``

    Returns ``(mutated_net, changed)``; ``net`` itself is not modified. Raises
    ValueError with an actionable message if the command is not allowed, cannot
    be parsed, or is rejected by validation.
    """
    # Imported here: modifier/commands are only needed for this step.
    from agentigrid.exago_engine.commands import (
        AddGeneratorAtBus, AddLoadAtBus, SetBranchStatus, SetGenStatus, SetLoad, parse_command,
    )
    from agentigrid.exago_engine.modifier import apply_modifications

    if not isinstance(raw, dict) or raw.get("action") not in MUTATION_ACTIONS:
        got = raw.get("action") if isinstance(raw, dict) else raw
        raise ValueError(
            f"'mutation' must be one command whose action is one of {list(MUTATION_ACTIONS)}, "
            f'e.g. {{"action": "add_load_at_bus", "bus": 77, "Pd": 50.0}}; got {got!r}.'
        )
    cmd = parse_command(raw)
    mutated, report = apply_modifications(net, [cmd], application="opflow")
    if report.skipped:
        reasons = "; ".join(r for _, rs in report.skipped for r in rs)
        raise ValueError(f"mutation could not be applied: {reasons}")

    if isinstance(cmd, (AddLoadAtBus, SetLoad)):
        return mutated, ChangedElement(kind="load", bus=cmd.bus)
    if isinstance(cmd, AddGeneratorAtBus):
        n_at_bus = sum(1 for g in mutated.generators if g.bus == cmd.bus)
        return mutated, ChangedElement(kind="gen", bus=cmd.bus, gen_id=n_at_bus - 1)
    if isinstance(cmd, SetGenStatus):
        return mutated, ChangedElement(kind="gen", bus=cmd.bus, gen_id=cmd.gen_id or 0)
    assert isinstance(cmd, SetBranchStatus)
    return mutated, ChangedElement(kind="branch", fbus=cmd.fbus, tbus=cmd.tbus, ckt=cmd.ckt or 0)


def added_load(before: MATNetwork, after: MATNetwork, changed: ChangedElement) -> dict[int, tuple[float, float]]:
    """The load the mutation added, as {bus: (Pd, Qd)}; empty unless a load was changed.

    This is the new development under study, so relief must never shed it. Only the
    increase is returned: load already at the bus before the mutation stays sheddable.
    """
    if changed.kind != "load":
        return {}
    pd0, qd0 = next((b.Pd, b.Qd) for b in before.buses if b.bus_i == changed.bus)
    pd1, qd1 = next((b.Pd, b.Qd) for b in after.buses if b.bus_i == changed.bus)
    dp, dq = max(pd1 - pd0, 0.0), max(qd1 - qd0, 0.0)
    return {changed.bus: (dp, dq)} if (dp or dq) else {}


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


@dataclass
class AffectedElements:
    """Result of the neighbor search: five lists, each entry tagged with its tier.

    buses:    {"bus", "substation", "tier"} for every bus in scope (the changed
              bus included — buses are not outaged themselves)
    branches: {"fbus", "tbus", "ckt", "tier"} in-service, both ends in scope
    gens:     {"bus", "gen_id", "tier"} in-service
    loads:    {"bus", "tier"} Pd or Qd != 0
    shunts:   {"bus", "tier"} Gs or Bs != 0

    The changed element is excluded from branches/gens/loads.
    """

    changed: ChangedElement
    depth: int
    buses: list[dict]
    branches: list[dict]
    gens: list[dict]
    loads: list[dict]
    shunts: list[dict]

    @property
    def substations(self) -> list[tuple[int, int]]:
        """(substation_id, tier) pairs, by tier then id."""
        pairs = {(b["substation"], b["tier"]) for b in self.buses}
        return sorted(pairs, key=lambda p: (p[1], p[0]))

    @property
    def tiers(self) -> list[list[int]]:
        """Substation ids grouped by tier."""
        out: list[list[int]] = []
        for sub, tier in self.substations:
            while len(out) <= tier:
                out.append([])
            out[tier].append(sub)
        return out

    def to_dict(self) -> dict:
        return {
            "buses": self.buses,
            "branches": self.branches,
            "gens": self.gens,
            "loads": self.loads,
            "shunts": self.shunts,
        }


def find_affected_elements(
    net: MATNetwork,
    changed: ChangedElement,
    depth: int,
) -> AffectedElements:
    """Neighbor search: all elements in the substations within ``depth`` of ``changed``.

    Scope: ``topology.substations_within`` from the changed element's bus (both
    end buses for a branch), so tier 0 is the changed substation(s). Branches
    leaving the last tier are excluded; parallel circuits are separate entries.
    Every list is in a fixed order (file order for branches and generators, bus
    order otherwise).
    """
    scope = topology.substations_within(net, changed.start_buses, depth)
    tier_of = scope.tier_of

    buses = [
        {"bus": b, "substation": scope.substation_of[b], "tier": tier_of[b]}
        for b in scope.buses
    ]

    branches = []
    for br in net.branches:
        if br.status == 0 or br.fbus == br.tbus:
            continue
        if br.fbus not in tier_of or br.tbus not in tier_of:
            continue
        entry = {
            "fbus": br.fbus, "tbus": br.tbus, "ckt": _branch_ckt(net, br),
            "tier": min(tier_of[br.fbus], tier_of[br.tbus]),
        }
        if not changed.matches_branch(entry):
            branches.append(entry)

    gens = []
    per_bus_index: dict[int, int] = {}
    for g in net.generators:
        gen_id = per_bus_index.get(g.bus, 0)
        per_bus_index[g.bus] = gen_id + 1
        if g.bus not in tier_of or g.status != 1:
            continue
        if changed.kind == "gen" and (g.bus, gen_id) == (changed.bus, changed.gen_id):
            continue
        gens.append({"bus": g.bus, "gen_id": gen_id, "tier": tier_of[g.bus]})

    loads, shunts = [], []
    for b in sorted(net.buses, key=lambda b: b.bus_i):
        if b.bus_i not in tier_of:
            continue
        tier = tier_of[b.bus_i]
        if (b.Pd != 0 or b.Qd != 0) and not (
            changed.kind == "load" and b.bus_i == changed.bus
        ):
            loads.append({"bus": b.bus_i, "tier": tier})
        if b.Gs != 0 or b.Bs != 0:
            shunts.append({"bus": b.bus_i, "tier": tier})

    return AffectedElements(
        changed=changed, depth=depth,
        buses=buses, branches=branches, gens=gens, loads=loads, shunts=shunts,
    )


def format_affected_elements_view(affected: AffectedElements, max_per_list: int = 40) -> str:
    """Compact, token-bounded text view of the neighbor search for the LLM.

    Shows the substations by tier and each of the five lists, capped at
    ``max_per_list`` entries per list with a count of the rest.
    """
    def _cap(items: list[str]) -> str:
        shown = ", ".join(items[:max_per_list])
        more = f", ... +{len(items) - max_per_list} more" if len(items) > max_per_list else ""
        return f"[{shown}{more}]" if items else "[]"

    def _branch(e: dict) -> str:
        ckt = f" ckt{e['ckt']}" if e["ckt"] else ""
        return f"{e['fbus']}-{e['tbus']}{ckt} (t{e['tier']})"

    lines = [
        f"Affected elements around changed element {affected.changed.label()} within substation "
        f"depth {affected.depth} (transformers add no hop; tN = tier N; the changed "
        "element is not listed):",
        "Substations by tier (id = lowest bus number): " + "; ".join(
            f"tier {d}: {_cap([str(s) for s in tier])}" for d, tier in enumerate(affected.tiers)
        ),
        f"buses ({len(affected.buses)}): "
        + _cap([f"{e['bus']} (t{e['tier']})" for e in affected.buses]),
        f"branches ({len(affected.branches)}): " + _cap([_branch(e) for e in affected.branches]),
        f"gens ({len(affected.gens)}): "
        + _cap([f"{e['bus']}#{e['gen_id']} (t{e['tier']})" for e in affected.gens]),
        f"loads ({len(affected.loads)}): "
        + _cap([f"{e['bus']} (t{e['tier']})" for e in affected.loads]),
        f"shunts ({len(affected.shunts)}): "
        + _cap([f"{e['bus']} (t{e['tier']})" for e in affected.shunts]),
        "[A contingency sweep with the same mutation and substation_depth tests "
        "exactly these elements; it does not need this lookup first.]",
    ]
    return "\n".join(lines)


def outages_from(
    affected: AffectedElements,
    components: tuple[str, ...] = OUTAGE_KINDS,
) -> list[OutageElement]:
    """Join the affected-element lists into one ordered outage pool.

    Restricted to the kinds in ``components``. Fixed total order (kind rank, then
    branch by (min,max,ckt), gen by (bus,gen_id), load/shunt by bus) so N-2
    pairing is reproducible.
    """
    pool: list[OutageElement] = []
    if "branch" in components:
        pool += [
            OutageElement(kind="branch", tier=e["tier"], fbus=e["fbus"], tbus=e["tbus"], ckt=e["ckt"])
            for e in affected.branches
        ]
    if "gen" in components:
        pool += [
            OutageElement(kind="gen", tier=e["tier"], bus=e["bus"], gen_id=e["gen_id"])
            for e in affected.gens
        ]
    if "load" in components:
        pool += [OutageElement(kind="load", tier=e["tier"], bus=e["bus"]) for e in affected.loads]
    if "shunt" in components:
        pool += [OutageElement(kind="shunt", tier=e["tier"], bus=e["bus"]) for e in affected.shunts]
    pool.sort(key=lambda e: e._sort_key())
    return pool


def enumerate_contingencies(pool: list[OutageElement], order: int) -> list[Contingency]:
    """Enumerate contingencies of the given order over an outage pool.

    order == 1 → one Contingency per pool element.
    order == 2 → all unordered pairs (i < j) over the ordered pool (C(m,2)).

    Raises ValueError for ``order`` not in (1, 2). Deterministic (follows the
    pool order).
    """
    if order not in (1, 2):
        raise ValueError(f"contingency order must be 1 or 2, got {order}.")
    if order == 1:
        return [Contingency(elements=(e,)) for e in pool]
    return [Contingency(elements=(a, b)) for a, b in combinations(pool, 2)]


def screen_verdict(
    changed: ChangedElement,
    order: int,
    n_total: int,
    failed_count: int,
    ref_passed: bool,
    ref_reason: str = "",
) -> tuple[bool, str]:
    """Pass/fail verdict for the whole screen: the change passes N-k only if every contingency passes.

    Returns ``(passes, line)``. A single failure, or an infeasible pre-contingency
    reference (the change alone already breaks the band), means it does not pass.
    """
    if not ref_passed:
        return False, (
            f"VERDICT: {changed.label()} does NOT pass N-{order}: the operating point with the change is "
            f"already infeasible before any outage ({ref_reason or 'did not converge'})."
        )
    if failed_count == 0:
        return True, f"VERDICT: {changed.label()} passes N-{order}: all {n_total} contingencies passed."
    return False, (
        f"VERDICT: {changed.label()} does NOT pass N-{order}: "
        f"{failed_count} of {n_total} contingencies failed."
    )


def failed_element_counts(
    contingencies: list[Contingency],
    passed: list[bool],
) -> list[tuple[OutageElement, int, int]]:
    """For each outage element, how many failed contingencies it appears in.

    ``passed`` is aligned with ``contingencies``. Returns ``(element, failed_in,
    appears_in)`` for elements in at least one failure, most failures first, then
    pool order. Used to summarise large N-2 screens by element.
    """
    failed_in: dict[OutageElement, int] = {}
    appears_in: dict[OutageElement, int] = {}
    for ctg, ok in zip(contingencies, passed):
        for e in ctg.elements:
            appears_in[e] = appears_in.get(e, 0) + 1
            if not ok:
                failed_in[e] = failed_in.get(e, 0) + 1
    rows = [(e, n, appears_in[e]) for e, n in failed_in.items()]
    rows.sort(key=lambda r: (-r[1], r[0]._sort_key()))
    return rows


def all_generator_contingencies(net: MATNetwork) -> list[Contingency]:
    """One N-1 contingency per in-service generator, system-wide (not scoped).

    Used by the hot-reserve / N-1 generator security screen (C.8): each committed
    unit is tripped in turn and the OPF re-solved. Only in-service units
    (status == 1) are included.

    Deterministic: generators are grouped by bus and, within a bus, indexed in
    ``net.generators`` order (``gen_id`` matches the modifier's per-bus generator
    indexing, as in ``find_affected_elements``). The returned list is sorted by
    ``(bus, gen_id)``.
    """
    contingencies: list[Contingency] = []
    buses = sorted({g.bus for g in net.generators})
    for bus in buses:
        gens_at_bus = [g for g in net.generators if g.bus == bus]
        for gen_id, gen in enumerate(gens_at_bus):
            if gen.status != 1:
                continue
            elem = OutageElement(kind="gen", tier=0, bus=bus, gen_id=gen_id)
            contingencies.append(Contingency(elements=(elem,)))
    return contingencies
