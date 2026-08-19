"""Minimum feasible hot reserve via greedy security-constrained de-commitment (C.8 Path A).

Prompt 4 defines hot reserve as ``R = Σ_on (Pmax − Pg)`` and asks for its MINIMUM
feasible value that remains N-1 secure. De-committing (turning off) an on-unit ``k``
removes its headroom from the sum, so minimizing ``R`` is equivalent to maximizing the
de-committed capacity subject to the operating point staying feasible AND every remaining
single-unit N-1 outage staying feasible.

Method (heuristic → the result is an UPPER BOUND on the true minimum): a single-pass,
largest-Pmax-first, skip-and-continue greedy. Starting from full commitment, each on-unit
is tentatively turned off; the de-commitment is accepted iff the base OPF still converges
AND every remaining single-unit N-1 outage is feasible; otherwise it is reverted and the
pass continues to the next unit. The reading-2 lower bound (largest remaining committed
Pg at the final commitment) brackets the true minimum from below.

This module is pure: it performs no simulation itself. The base-solve and N-1-screen
functions are injected by the caller (the closures build the net variant and drive the
executor). The module only decides the de-commitment order and accept/revert logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class _OPFLOWLike(Protocol):
    """Minimal shape the module reads from an injected base-solve result."""
    generators: list  # each g exposes .status (int), .bus (int), .Pg, .Pmax (float)


# base_solve_fn: (off_units) -> OPFLOWResult | None   (None = did not converge)
BaseSolveFn = Callable[[frozenset], Optional[_OPFLOWLike]]
# n1_screen_fn: (off_units) -> (all_feasible, n_passed, n_total)
N1ScreenFn = Callable[[frozenset], "tuple[bool, int, int]"]


@dataclass(frozen=True)
class ReserveMinResult:
    reserve_full: float                     # hot reserve at full commitment (starting point)
    min_reserve: float                      # hot reserve at the final (minimal) commitment
    decommitted: list[tuple[int, int]]      # (bus, gen_id) turned off, in de-commit order
    final_on_count: int
    lower_bound_pg: float                   # largest remaining on-unit Pg at final commitment
    lower_bound_bus: int
    n1_secure: bool                         # True by construction at the final commitment
    solves_used: int
    hit_budget: bool
    trajectory: list = field(default_factory=list)
    # trajectory entries: {"step": int, "reserve": float, "on_count": int,
    #   "decommitted_bus": int|None, "decommitted_gen_id": int|None}
    # Step 0 = full commitment; subsequent steps on each accepted de-commitment.


def _reserve_from(opflow: _OPFLOWLike) -> float:
    """Σ over on-units (status==1) of (Pmax − Pg) from a solved dispatch."""
    return sum(g.Pmax - g.Pg for g in opflow.generators if g.status == 1)


def _largest_pg(opflow: _OPFLOWLike) -> tuple[float, int]:
    """Largest committed-unit Pg (reading-2 lower bound) and its bus; (0.0, -1) if none."""
    on = [g for g in opflow.generators if g.status == 1]
    if not on:
        return 0.0, -1
    g = max(on, key=lambda g: g.Pg)
    return g.Pg, g.bus


def _in_service_units(net) -> list[tuple[int, int, float]]:
    """(bus, gen_id, Pmax) for every in-service generator.

    ``gen_id`` is the position among generators at that bus (modifier /
    ``all_generator_contingencies`` semantics), so the tuples line up with the
    outage elements the injected screen enumerates.
    """
    units: list[tuple[int, int, float]] = []
    buses = sorted({g.bus for g in net.generators})
    for bus in buses:
        gens_at_bus = [g for g in net.generators if g.bus == bus]
        for gen_id, g in enumerate(gens_at_bus):
            if g.status != 1:
                continue
            units.append((bus, gen_id, g.Pmax))
    return units


def minimize_hot_reserve(
    net,
    vmin: float,
    vmax: float,
    *,
    base_solve_fn: BaseSolveFn,
    n1_screen_fn: N1ScreenFn,
    max_solves: int,
) -> ReserveMinResult:
    """Largest-Pmax-first greedy de-commitment for minimum N-1-secure hot reserve.

    Orders the in-service generators by descending ``Pmax`` (tie-break ascending
    ``bus`` then ``gen_id``). Starting from full commitment, tentatively turns each
    off (via the ``off_units`` set passed to the injected closures); accepts the
    de-commitment iff ``base_solve_fn`` converges AND ``n1_screen_fn`` reports every
    remaining single-unit outage feasible; otherwise reverts. Reserve is recomputed
    from each accepted base solve. Stops when the order is exhausted or the solve
    budget (``max_solves``) is reached.
    """
    order = sorted(_in_service_units(net), key=lambda u: (-u[2], u[0], u[1]))

    off: set[tuple[int, int]] = set()
    decommitted: list[tuple[int, int]] = []
    solves = 0
    hit_budget = False
    trajectory: list[dict] = []

    # --- establish the full-commitment starting point ---
    base = base_solve_fn(frozenset(off))
    solves += 1
    if base is None:
        # Full commitment is itself infeasible — nothing to minimize.
        return ReserveMinResult(
            reserve_full=0.0, min_reserve=0.0, decommitted=[],
            final_on_count=len(order), lower_bound_pg=0.0, lower_bound_bus=-1,
            n1_secure=False, solves_used=solves, hit_budget=False,
        )

    reserve_full = _reserve_from(base)
    current_reserve = reserve_full
    current_base = base
    # Record step 0: full commitment.
    trajectory.append({
        "step": 0,
        "reserve": reserve_full,
        "on_count": len(order),
        "decommitted_bus": None,
        "decommitted_gen_id": None,
    })
    # N-1 security of the *full* commitment (screen over all on-units).
    full_feasible, _, _ = n1_screen_fn(frozenset(off))
    solves += len(order) + 1
    current_secure = full_feasible

    for (bus, gen_id, _pmax) in order:
        # Budget guard: need at least a base solve + a screen to evaluate a trial.
        if solves >= max_solves:
            hit_budget = True
            break

        trial_off = off | {(bus, gen_id)}

        sol = base_solve_fn(frozenset(trial_off))
        solves += 1
        if sol is None:
            continue  # base infeasible with this unit off → revert, keep going

        if solves >= max_solves:
            hit_budget = True
            break  # can't afford the confirming screen → stop before accepting

        all_feasible, _n_passed, n_total = n1_screen_fn(frozenset(trial_off))
        solves += n_total + 1
        if not all_feasible:
            continue  # de-committing this unit breaks N-1 → revert, keep going

        # Accept the de-commitment.
        off = trial_off
        decommitted.append((bus, gen_id))
        current_base = sol
        current_reserve = _reserve_from(sol)
        current_secure = True
        trajectory.append({
            "step": len(decommitted),
            "reserve": current_reserve,
            "on_count": len(order) - len(off),
            "decommitted_bus": bus,
            "decommitted_gen_id": gen_id,
        })

    lower_bound_pg, lower_bound_bus = _largest_pg(current_base)
    final_on_count = len(order) - len(decommitted)

    return ReserveMinResult(
        reserve_full=reserve_full,
        min_reserve=current_reserve,
        decommitted=decommitted,
        final_on_count=final_on_count,
        lower_bound_pg=lower_bound_pg,
        lower_bound_bus=lower_bound_bus,
        n1_secure=current_secure,
        solves_used=solves,
        hit_budget=hit_budget,
        trajectory=trajectory,
    )
