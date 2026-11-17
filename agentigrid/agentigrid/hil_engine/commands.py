"""The HIL study the agent asks for, as a dataclass, its parser and its checks.

One action does the switching: ``close_tie``. The engine starts the simulation
radial (all ties open, set before the start), reads every bus, closes the tie
live, reads again, and compares with ExaGO when asked.

Live opening is not offered: on the virtual HIL it leaves a wrong state near the
opened tie (study plan, step 4). A request to open is refused with that reason,
and the agent reports it as SKIPPED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agentigrid.hil_engine.network import HilNetwork

START_FROM = ("radial", "latest_steady_state")
REFUSED_ACTIONS = {"open_tie", "open", "open_switch", "open_line", "trip"}

OPEN_REFUSED = ("Opening a tie while the simulation runs is not available on this HIL: on the "
                "virtual HIL it leaves a wrong steady state near the opened tie. Do not imitate it "
                "with another action; report the request as SKIPPED with this reason.")


@dataclass(frozen=True)
class CloseTie:
    tie: tuple[int, int]                 # (low, high) IEEE bus numbers
    start_from: str = "radial"           # radial | latest_steady_state
    description: str = ""


def _pair(value) -> tuple[int, int]:
    """[12, 22], ["12", "22"], "12-22", "12 22", "line 12-22" -> (12, 22)."""
    if isinstance(value, (list, tuple)):
        nums = [int(v) for v in value]
    elif isinstance(value, str):
        nums = [int(n) for n in re.findall(r"\d+", value)]
    else:
        raise ValueError(f"tie must be two bus numbers, e.g. [12, 22], not {value!r}")
    if len(nums) != 2:
        raise ValueError(f"tie must be exactly two bus numbers, e.g. [12, 22], not {value!r}")
    return tuple(sorted(nums))


def parse_close_tie(raw: dict) -> CloseTie:
    fields_ = {"action", "tie", "start_from", "description", "reasoning"}
    unknown = set(raw) - fields_
    if unknown:
        raise ValueError(f"Unknown close_tie fields: {sorted(unknown)}. Valid: {sorted(fields_ - {'action'})}")
    if "tie" not in raw:
        raise ValueError('close_tie needs "tie": [bus, bus], e.g. {"action": "close_tie", "tie": [12, 22]}')
    start = raw.get("start_from") or "radial"
    if start not in START_FROM:
        raise ValueError(f"start_from must be one of {list(START_FROM)}, not {start!r}")
    return CloseTie(tie=_pair(raw["tie"]), start_from=start,
                    description=str(raw.get("description") or ""))


def validate_close_tie(study: CloseTie, net: HilNetwork) -> Optional[str]:
    """None if the HIL can do it, else the reason (shown to the agent)."""
    a, b = study.tie
    if a not in net.buses or b not in net.buses:
        return f"bus {a if a not in net.buses else b} does not exist (buses {net.buses[0]}-{net.buses[-1]})"
    if net.tie(a, b) is None:
        line = next((n for n, p in net.lines.items() if p == (a, b)), None)
        what = f"{a}-{b} is an ordinary line ({line}) with no switch" if line else \
            f"there is no line between buses {a} and {b}"
        return f"{what}; the switchable ties are {net.tie_names()}"
    return None
