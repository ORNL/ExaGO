"""What the HIL model offers the agent: buses, tie switches, measurements.

Read from the Typhoon schematic (.tse) itself, so the agent and the engine always
use the model's own names:

  buses     every three-phase meter ``3ph_Meter<NNN>``; IEEE bus n = meter NNN - offset
            (offset 700 in the IEEE 33 example: bus 1 = Meter701)
  ties      every contactor ``Tie_sw_<k>`` driven by a SCADA input
            (``connect <input>.out Tie_sw_<k>.ctrl_in``); its two buses are traced
            through the wiring, crossing TLM couplings and other tie switches
  feeder    the contactor ``Feeder sw``
  voltage   ``V_ll`` from the model's ``model_init`` code

Signals read on the HIL: ``Meter<NNN>.VAn_rms`` (V, line-to-neutral, phase a)
and ``Meter<NNN>.Va_ph`` (degrees). The model is balanced three-phase, so phase a
stands for all three.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ModelError(ValueError):
    """The Typhoon model does not have what the HIL engine needs."""


@dataclass(frozen=True)
class Tie:
    buses: tuple[int, int]        # (low, high) IEEE bus numbers
    switch: str                   # contactor, e.g. Tie_sw_5
    scada_input: str              # SCADA input that drives it, e.g. Tie_line_5
    line: Optional[str] = None    # line section in series, e.g. "Line 735"

    @property
    def name(self) -> str:
        return f"{self.buses[0]}-{self.buses[1]}"


@dataclass
class HilNetwork:
    model: Path
    buses: list[int]                          # IEEE bus numbers, ascending
    meter_offset: int
    ties: list[Tie]
    feeder_switch: Optional[str]
    v_ll_kv: float
    lines: dict[str, tuple[int, int]] = field(default_factory=dict)   # every RL section

    @property
    def v_ln(self) -> float:
        """Nominal line-to-neutral voltage, V (1.0 pu on the HIL)."""
        return self.v_ll_kv * 1e3 / math.sqrt(3)

    def meter(self, bus: int) -> str:
        return f"Meter{bus + self.meter_offset}"

    def signals(self) -> tuple[list[str], list[str]]:
        """(magnitude signals, angle signals), in bus order."""
        return ([f"{self.meter(b)}.VAn_rms" for b in self.buses],
                [f"{self.meter(b)}.Va_ph" for b in self.buses])

    def tie(self, a: int, b: int) -> Optional[Tie]:
        pair = tuple(sorted((int(a), int(b))))
        return next((t for t in self.ties if t.buses == pair), None)

    def tie_names(self) -> str:
        return ", ".join(t.name for t in self.ties)


# ---------------------------------------------------------------------------
# Reading the schematic
# ---------------------------------------------------------------------------

_CONNECT_RE = re.compile(r'^ {8}connect ("[^"]+"|\S+) ("[^"]+"|\S+) as', re.M)
_METER_RE = re.compile(r"^3ph_Meter(\d+)$")


def _nets(text: str):
    """Union of every top-level connection: terminal -> net id, net id -> terminals."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in _CONNECT_RE.findall(text):
        parent[find(a.strip('"'))] = find(b.strip('"'))
    nets = defaultdict(set)
    for t in list(parent):
        nets[find(t)].add(t)
    return find, nets, parent


def load_network(model: Path | str) -> HilNetwork:
    model = Path(model)
    if not model.exists():
        raise ModelError(f"Typhoon model not found: {model}")
    text = model.read_text(errors="replace")
    find, nets, parent = _nets(text)
    comp = lambda t: t.rsplit(".", 1)[0]  # noqa: E731

    meters = sorted({int(m.group(1)) for t in parent if (m := _METER_RE.match(comp(t)))})
    if not meters:
        raise ModelError(f"{model.name}: no three-phase meters named 3ph_Meter<NNN>")
    offset = meters[0] - 1
    buses = [m - offset for m in meters]

    # Elements that are a plain connection in steady state: TLM couplings and tie
    # switches (closed). Each side is the group of its three phase ports, because
    # the model does not keep phase labels along a path.
    through: dict[str, list[str]] = {}
    for t in parent:
        c, _, port = t.rpartition(".")
        if (c.startswith("CC") or c.startswith("Tie_sw_")) and len(port) > 2 and port[0] in "abc" \
                and port[1:] in ("_in", "_out"):
            other = "out" if port.endswith("in") else "in"
            through[t] = [f"{c}.{ph}_{other}" for ph in "abc"]

    def reach(terminals, stop_at: Optional[str] = None) -> tuple[Optional[int], set[str]]:
        """(bus reached, line sections touched) from *terminals*, crossing couplings and
        tie switches (not *stop_at*), never crossing a line section."""
        seen, todo, touched = set(), [find(t) for t in terminals if t in parent], set()
        while todo:
            net = todo.pop()
            if net in seen:
                continue
            seen.add(net)
            touched |= {comp(t) for t in nets[net] if comp(t).startswith("Line ")}
            for t in nets[net]:
                m = _METER_RE.match(comp(t))
                if m:
                    return int(m.group(1)) - offset, touched
            for t in nets[net]:
                if t in through and comp(t) != stop_at:
                    todo.extend(find(o) for o in through[t] if o in parent)
        return None, touched

    lines = {}
    for m in re.finditer(r'^ {8}component "core/RL Section" "(Line [^"]+)"', text, re.M):
        name = m.group(1)
        ends = (reach([f"{name}.P{k}" for k in (1, 3, 5)])[0],
                reach([f"{name}.P{k}" for k in (2, 4, 6)])[0])
        if None not in ends and ends[0] != ends[1]:
            lines[name] = tuple(sorted(ends))

    ties = []
    for m in re.finditer(r'^ {8}component "core/Triple Pole Single Throw Contactor" (Tie_sw_\d+)', text, re.M):
        sw = m.group(1)
        drivers = [comp(t) for t in nets[find(f"{sw}.ctrl_in")] if t.endswith(".out")] \
            if f"{sw}.ctrl_in" in parent else []
        if len(drivers) != 1:
            raise ModelError(f"{model.name}: {sw} is not driven by exactly one SCADA input")
        sides = [reach([f"{sw}.{p}_{s}" for p in "abc"], stop_at=sw) for s in ("in", "out")]
        line = None
        if all(bus is not None for bus, _ in sides):
            pair = tuple(sorted(bus for bus, _ in sides))      # switch directly between buses
        else:
            # One side leads into the tie line: the tie connects that line's two ends.
            open_side = [touched for bus, touched in sides if bus is None]
            cands = open_side[0] if len(open_side) == 1 else set()
            if len(cands) != 1 or next(iter(cands)) not in lines:
                raise ModelError(f"{model.name}: cannot trace the line in series with {sw}")
            line = next(iter(cands))
            pair = lines[line]
        if pair[0] == pair[1]:
            raise ModelError(f"{model.name}: {sw} traces to one bus only ({pair[0]})")
        ties.append(Tie(pair, sw, drivers[0], line))
    ties.sort(key=lambda t: t.buses)

    feeder = "Feeder sw" if re.search(r'component "core/[^"]*Contactor" "Feeder sw"', text) else None
    v_ll = re.search(r"^\s*V_ll\s*=\s*([0-9.eE+-]+)", text, re.M)
    if not v_ll:
        raise ModelError(f"{model.name}: no V_ll in the model_init code")
    return HilNetwork(model=model, buses=buses, meter_offset=offset, ties=ties,
                      feeder_switch=feeder, v_ll_kv=float(v_ll.group(1)) / 1e3, lines=lines)


def describe(net: HilNetwork) -> str:
    """Network text for the system prompt."""
    rows = "\n".join(
        f"  {t.name:<7}  switch {t.switch:<9}  (line {t.line or '-'})" for t in net.ties)
    return f"""\
Model: {net.model.name} (Typhoon HIL, virtual HIL device)
Buses: {len(net.buses)}, numbered {net.buses[0]}-{net.buses[-1]} (IEEE numbering; on the HIL bus n is
  {net.meter(net.buses[0])} ... {net.meter(net.buses[-1])}).
Balanced three-phase, {net.v_ll_kv:g} kV line-to-line. Voltages are reported line-to-neutral, phase a:
  1.0 pu = {net.v_ln:.2f} V. Angles in degrees, bus {net.buses[0]} (the source) at 0.
Tie switches (the only switches the agent may operate; every other line has no switch):
{rows}
Normal state: radial, all ties open (standard IEEE 33). The feeder breaker is closed."""
