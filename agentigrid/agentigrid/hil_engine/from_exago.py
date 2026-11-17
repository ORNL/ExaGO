"""The ExaGO steady state a HIL study is compared with.

"Most recent steady state" is found exactly as for GridKit
(``gridkit_engine.from_exago.find_latest_steady_state``): the last simulation of
the newest ExaGO journal, a single converged PFLOW or OPFLOW with a saved
solution, no fallback to an older one.

The HIL can only reproduce what its switches can do. The ExaGO network must
therefore be the HIL network with some ties closed: every line that is not a tie
in service, the same buses. Anything else (a line out, a missing bus) means the
HIL cannot build that network, and the comparison is not made.

The radial reference (the state before the tie is closed) is the most recent
earlier simulation in the same journal whose network is radial (all ties open).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.gridkit_engine.from_exago import (
    STEADY_STATE_APPS, SteadyStateError, SteadyStateSource, find_latest_steady_state,
)
from agentigrid.hil_engine.network import HilNetwork

logger = logging.getLogger("agentigrid.hil_engine.from_exago")

__all__ = ["ExagoState", "SteadyStateError", "load_state", "latest_state", "radial_reference"]


@dataclass
class ExagoState:
    label: str                                  # for verdicts and prompts
    solved_file: Path
    closed_ties: set[tuple[int, int]]           # ties in service
    volts: dict[int, float]                     # bus -> V line-to-neutral (pu x V_ln)
    degrees: dict[int, float]
    vm_pu: dict[int, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)   # why the HIL cannot build this network

    @property
    def usable(self) -> bool:
        return not self.problems


def load_state(solved_file: Path, net: HilNetwork, label: str) -> ExagoState:
    """Read a solved MATPOWER case and check it against the HIL network."""
    case = parse_matpower(solved_file)
    problems: list[str] = []
    bus_ids = {b.bus_i for b in case.buses}
    if bus_ids != set(net.buses):
        problems.append(f"ExaGO has buses {sorted(bus_ids - set(net.buses)) or '-'} the HIL lacks "
                        f"and lacks {sorted(set(net.buses) - bus_ids) or '-'}")
    tie_pairs = {t.buses for t in net.ties}
    closed: set[tuple[int, int]] = set()
    seen_lines = set()
    for br in case.branches:
        pair = tuple(sorted((br.fbus, br.tbus)))    # either order: saved files can swap ends
        if pair in tie_pairs:
            if br.status:
                closed.add(pair)
        else:
            seen_lines.add(pair)
            if not br.status:
                problems.append(f"line {pair[0]}-{pair[1]} is out of service in ExaGO; "
                                "the HIL has no switch on it")
    missing = set(net.lines.values()) - tie_pairs - seen_lines
    if missing:
        problems.append("lines missing in ExaGO: " + ", ".join(f"{a}-{b}" for a, b in sorted(missing)))
    vm = {b.bus_i: b.Vm for b in case.buses}
    return ExagoState(label=label, solved_file=solved_file, closed_ties=closed,
                      volts={b: v * net.v_ln for b, v in vm.items()},
                      degrees={b.bus_i: b.Va for b in case.buses}, vm_pu=vm, problems=problems)


def _label(source: SteadyStateSource) -> str:
    return (f"ExaGO {source.application.upper()} iteration {source.iteration} "
            f"(\"{source.description}\", {source.journal.name})")


def latest_state(exago_workdir: Path, net: HilNetwork) -> tuple[SteadyStateSource, ExagoState]:
    """The most recent ExaGO steady state. Raises SteadyStateError like GridKit does."""
    source = find_latest_steady_state(exago_workdir)
    return source, load_state(source.solved_file, net, _label(source))


def radial_reference(source: SteadyStateSource, net: HilNetwork) -> Optional[ExagoState]:
    """The most recent earlier converged steady state of the same journal with all ties open."""
    entries = json.loads(source.journal.read_text()).get("entries", [])
    for e in reversed(entries):
        cmd = e.get("exago_command") or {}
        it = e.get("iteration")
        if it is None or int(it) >= source.iteration or cmd.get("mode") != "single":
            continue
        app = cmd.get("application", "")
        if app not in STEADY_STATE_APPS:
            continue
        if not str(e.get("convergence_status", "")).upper().startswith("CONVERGED"):
            continue
        argv = cmd.get("argv") or []
        netfile = next((Path(v) for f, v in zip(argv, argv[1:]) if f == "-netfile"), None)
        solved = netfile.parent / STEADY_STATE_APPS[app] if netfile else None
        if solved is None or not solved.exists():
            continue
        label = f"ExaGO {app.upper()} iteration {it} (\"{e.get('description', '')}\", {source.journal.name})"
        try:
            state = load_state(solved, net, label)
        except Exception as exc:          # noqa: BLE001 - an unreadable old file is skipped
            logger.warning("Skipping %s: %s", solved, exc)
            continue
        if state.usable and not state.closed_ties:
            return state
    return None
