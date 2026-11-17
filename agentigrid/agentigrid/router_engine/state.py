"""Steady states shared by the tools of a router session, and how "latest" and "best" are chosen.

Every converged ExaGO solve adds a steady state: its network file (what was solved),
its solved file (what ExaGO saved), the state it was built on (parent), the ExaGO
commands that changed it, and metrics computed from the solved file. Every other
tool works from one of these: the most recent ("latest") or the best one for the
goal's objectives ("best"), with the objectives extracted from the goal the same
way the ExaGO agent does it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentigrid.exago_parsers.matpower_parser import parse_matpower

logger = logging.getLogger("agentigrid.router_engine.state")

START_FROM = ("latest", "best")

# Metrics computed from a solved MATPOWER file (names as in the ExaGO agent).
METRICS = ("total_generation_mw", "total_load_mw", "active_losses_mw", "voltage_min", "voltage_max",
           "voltage_deviation", "voltage_range", "total_reactive_gen_mvar", "phase_angle_range",
           "generation_cost")


def solved_metrics(solved: Path) -> dict[str, float]:
    """Metrics of a solved case. Load includes loads modelled as shunts (Gs x Vm^2, MW at 1 pu)."""
    case = parse_matpower(solved)
    gens = [g for g in case.generators if g.status]
    pg = sum(g.Pg for g in gens)
    load = sum(b.Pd + b.Gs * b.Vm ** 2 for b in case.buses)
    vm = [b.Vm for b in case.buses]
    va = [b.Va for b in case.buses]
    cost = 0.0
    for g, c in zip(case.generators, case.gencost or []):
        if not g.status:
            continue
        model = int(getattr(c, "model", 2))
        coeffs = list(getattr(c, "coeffs", []) or [])
        if model == 2 and coeffs:
            n = len(coeffs)
            cost += sum(a * g.Pg ** (n - 1 - i) for i, a in enumerate(coeffs))
    return {"total_generation_mw": pg, "total_load_mw": load, "active_losses_mw": pg - load,
            "voltage_min": min(vm), "voltage_max": max(vm),
            "voltage_deviation": max(abs(1.0 - v) for v in vm), "voltage_range": max(vm) - min(vm),
            "total_reactive_gen_mvar": sum(g.Qg for g in gens), "phase_angle_range": max(va) - min(va),
            "generation_cost": cost}


@dataclass
class SteadyState:
    sid: int                                  # S0, S1, ... in the order they were made
    tool: str                                 # the tool that made it ("exago")
    application: str                          # pflow | opflow
    label: str
    network_file: Path                        # the case that was solved
    solved_file: Path                         # what the solver saved
    run_dir: Path
    parent: Optional[int]                     # the state it was built on (None: the input case)
    commands: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    iterations: Optional[int] = None

    @property
    def name(self) -> str:
        return f"S{self.sid}"


@dataclass
class Objective:
    name: str
    direction: str                            # minimize | maximize | constraint
    threshold: Optional[float] = None
    priority: str = "primary"


class StateStore:
    def __init__(self) -> None:
        self.states: list[SteadyState] = []
        self.objectives: list[Objective] = []

    def add(self, state: SteadyState) -> SteadyState:
        self.states.append(state)
        return state

    def latest(self) -> Optional[SteadyState]:
        return self.states[-1] if self.states else None

    def _meets(self, s: SteadyState, o: Objective) -> bool:
        v = s.metrics.get(o.name)
        if v is None or o.threshold is None:
            return True
        upper = o.name not in ("voltage_min",)          # "voltage_min >= x"; the rest are "<= x"
        return v <= o.threshold if upper else v >= o.threshold

    def best(self) -> tuple[Optional[SteadyState], str]:
        """(state, why) or (None, reason): the state that best meets the goal's objectives."""
        if not self.states:
            return None, "there is no steady state yet"
        goals = [o for o in self.objectives if o.direction in ("minimize", "maximize") and o.name in METRICS]
        if not goals:
            return None, ("the goal states no objective AgentiGrid can rank steady states by "
                          f"(metrics: {', '.join(METRICS)}); use \"latest\"")
        cons = [o for o in self.objectives if o.direction == "constraint" and o.name in METRICS]
        pool = [s for s in self.states if all(self._meets(s, o) for o in cons)]
        if not pool:
            return None, "no steady state meets the goal's limits"
        order = sorted(goals, key=lambda o: {"primary": 0, "secondary": 1}.get(o.priority, 2))

        def key(s: SteadyState):
            return tuple((s.metrics[o.name] if o.direction == "minimize" else -s.metrics[o.name])
                         for o in order)
        best = min(pool, key=key)
        why = ", ".join(f"{o.direction} {o.name} = {best.metrics[o.name]:.6g}" for o in order)
        return best, why

    def choose(self, start_from: Optional[str]) -> tuple[Optional[SteadyState], str]:
        """The state an action asked for, or (None, reason)."""
        start_from = (start_from or "latest").lower()
        if start_from not in START_FROM:
            return None, f'start_from must be "latest" or "best", not {start_from!r}'
        if start_from == "latest":
            s = self.latest()
            return (s, "the most recent steady state") if s else (None, "there is no steady state yet")
        return self.best()

    def describe(self) -> str:
        if not self.states:
            return "none yet"
        rows = []
        for s in self.states:
            m = s.metrics
            rows.append(f"{s.name}: {s.label} (from {('S' + str(s.parent)) if s.parent is not None else 'the input case'}; "
                        f"{s.application.upper()}, {s.iterations} iterations); "
                        f"V {m.get('voltage_min', float('nan')):.6f}-{m.get('voltage_max', float('nan')):.6f} pu, "
                        f"generation {m.get('total_generation_mw', float('nan')):.4f} MW, "
                        f"load {m.get('total_load_mw', float('nan')):.4f} MW, "
                        f"losses {m.get('active_losses_mw', float('nan')):.4f} MW")
        return "\n".join(rows)
