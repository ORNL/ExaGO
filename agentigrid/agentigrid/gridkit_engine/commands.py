"""GridKit case commands and the fault-study request, as dataclasses and parsers.

Case commands change the COPY of the case in the run folder (never the
original file). The fault study says which buses to fault and how; the
engine turns it into GridKit fault events.

Only what GridKit can do natively is offered. Not available (skipped, see
criteria.py): line/generator trips, reclosing, connecting a new generator or
load with its dynamic models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from agentigrid.gridkit_engine import criteria

logger = logging.getLogger("agentigrid.gridkit_engine.commands")


@dataclass
class AddBusFault:
    """Make sure a bus has a fault device (added to the case copy if missing)."""

    bus: int
    R: float = criteria.FAULT_R_PU
    X: float = criteria.FAULT_X_PU


@dataclass
class RecordVariables:
    """Variables to record for every machine and every bus."""

    machines: list[str] = field(default_factory=lambda: ["delta", "omega", "p", "q"])
    buses: list[str] = field(default_factory=lambda: ["Vm"])


CaseCommand = Union[AddBusFault, RecordVariables]

_COMMAND_MAP: dict[str, tuple[type, set[str]]] = {
    "add_bus_fault": (AddBusFault, {"bus"}),
    "record_variables": (RecordVariables, set()),
}


def parse_command(raw: dict) -> CaseCommand:
    """Parse a raw JSON dict with an ``"action"`` key into a case command.

    Raises:
        ValueError: unknown action, missing or invalid fields.
    """
    action = raw.get("action")
    if action is None:
        raise ValueError("Command dict missing 'action' key")
    entry = _COMMAND_MAP.get(action)
    if entry is None:
        raise ValueError(f"Unknown action '{action}'. Valid: {sorted(_COMMAND_MAP)}")
    cls, required = entry
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Action '{action}' missing required fields: {missing}")
    try:
        return cls(**{k: v for k, v in raw.items() if k != "action"})
    except TypeError as exc:
        raise ValueError(f"Invalid fields for '{action}': {exc}") from exc


# ---------------------------------------------------------------------------
# Fault study request
# ---------------------------------------------------------------------------

@dataclass
class FaultStudy:
    """One bus short circuit to ground per listed bus, each in its own run.

    Locations, exactly one of:
      buses        explicit bus numbers
      all_buses    every bus in the case
      poi + hops   the POI bus and every bus up to *hops* branches away
                   (PJM M-14B G.3.2: at least the POI and one bus away)
    """

    buses: Optional[list[int]] = None
    all_buses: bool = False
    poi: Optional[int] = None
    hops: int = 1
    start_s: float = criteria.FAULT_START_S
    clearing_cycles: float = criteria.NORMAL_CLEARING_CYCLES
    margin_cycles: float = criteria.MARGIN_PRIMARY_CYCLES
    tmax_s: float = criteria.SIM_LENGTH_S
    dt_monitor_s: float = 0.01
    R: float = criteria.FAULT_R_PU
    X: float = criteria.FAULT_X_PU
    checks: list[str] = field(default_factory=lambda: [
        "angle_stability", "voltage_recovery", "damping", "final_voltage",
    ])

    def duration_s(self, freq_hz: float = 60.0) -> float:
        return criteria.clearing_time_s(self.clearing_cycles, self.margin_cycles, freq_hz)

    def clear_time_s(self, freq_hz: float = 60.0) -> float:
        return self.start_s + self.duration_s(freq_hz)


_STUDY_FIELDS = set(FaultStudy.__dataclass_fields__)


def parse_fault_study(raw: dict) -> FaultStudy:
    """Parse the LLM's fault-study request (the action dict minus bookkeeping keys)."""
    ignored = {"action", "description", "reasoning"}
    unknown = set(raw) - _STUDY_FIELDS - ignored
    if unknown:
        raise ValueError(f"Unknown fault-study fields: {sorted(unknown)}. Valid: {sorted(_STUDY_FIELDS)}")
    kwargs = {k: v for k, v in raw.items() if k in _STUDY_FIELDS}
    try:
        return FaultStudy(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid fault-study fields: {exc}") from exc
