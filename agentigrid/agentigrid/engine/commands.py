"""Modification command dataclasses and JSON parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger("agentigrid.engine.commands")


@dataclass
class SetLoad:
    """Set active/reactive load at a specific bus."""

    bus: int
    Pd: Optional[float] = None
    Qd: Optional[float] = None


@dataclass
class ScaleLoad:
    """Scale load at a bus, area, or zone by a factor."""

    factor: float
    bus: Optional[int] = None
    area: Optional[int] = None
    zone: Optional[int] = None


@dataclass
class ScaleAllLoads:
    """Scale all loads in the network uniformly."""

    factor: float


@dataclass
class SetGenStatus:
    """Turn a generator on or off."""

    bus: int
    status: int
    gen_id: Optional[int] = None


@dataclass
class SetGenDispatch:
    """Set generator active power output."""

    bus: int
    Pg: float
    gen_id: Optional[int] = None


@dataclass
class SetGenVoltage:
    """Set generator voltage setpoint."""

    bus: int
    Vg: float
    gen_id: Optional[int] = None


@dataclass
class SetBranchStatus:
    """Enable or disable a branch."""

    fbus: int
    tbus: int
    status: int
    ckt: Optional[int] = None


@dataclass
class SetBranchRate:
    """Modify branch thermal rating."""

    fbus: int
    tbus: int
    rateA: float
    ckt: Optional[int] = None


@dataclass
class SetCostCoeffs:
    """Modify generator cost curve coefficients."""

    bus: int
    coeffs: list[float] = field(default_factory=list)
    gen_id: Optional[int] = None


@dataclass
class SetBusVLimits:
    """Set bus voltage limits."""

    bus: int
    Vmin: Optional[float] = None
    Vmax: Optional[float] = None


@dataclass
class SetAllBusVLimits:
    """Set voltage limits on all buses in the network."""

    Vmin: Optional[float] = None
    Vmax: Optional[float] = None


@dataclass
class ScaleLoadProfile:
    """Scale per-period load profile values by a factor (TCOPFLOW only).

    Multiplies all numeric values in both P and Q load profile CSV files
    by the given factor. The Timestamp column is preserved. This is the
    correct mechanism for adjusting demand in TCOPFLOW, since TCOPFLOW
    reads per-period loads from profile files, not the .m case file.
    """

    factor: float


@dataclass
class ScaleWindScenario:
    """Scale wind generation values in a scenario CSV by a factor (SOPFLOW only).

    Multiplies all numeric wind generation columns in the scenario CSV file
    by the given factor, preserving non-numeric columns (scenario_nr,
    sim_timestamp, weight). This is the correct mechanism for adjusting
    wind generation variability in SOPFLOW.
    """

    factor: float


@dataclass
class SetTapRatio:
    """Set transformer tap ratio for a branch.

    Only applies to branches that are transformers (ratio != 0 in the base
    network). Setting the tap ratio on a non-transformer line (ratio == 0)
    is rejected as invalid.
    """

    fbus: int
    tbus: int
    ratio: float
    ckt: Optional[int] = None


@dataclass
class SetShuntSusceptance:
    """Set shunt susceptance at a bus.

    Modifies the Bs (shunt susceptance) field of the specified bus.
    Positive Bs adds capacitive susceptance; negative Bs adds inductive.
    """

    bus: int
    Bs: float


@dataclass
class SetPhaseShiftAngle:
    """Set phase shifter angle for a branch.

    Only applies to branches that are phase shifters (angle != 0 in the
    base network). Setting the angle on a non-phase-shifter branch (angle == 0)
    is rejected as invalid.
    """

    fbus: int
    tbus: int
    angle: float
    ckt: Optional[int] = None


@dataclass
class AddLoadAtBus:
    """Add (additively) active/reactive load to an existing bus."""

    bus: int
    Pd: float = 0.0
    Qd: float = 0.0


@dataclass
class AddGeneratorAtBus:
    """Append a NEW generator to a bus.

    Default is a forced-injection unit (Pmin=Pmax=Pg=capacity_mw), used for
    hosting-capacity / feasibility tests. Set dispatchable=True for an
    economic-dispatch unit (Pmin=0, Pmax=capacity_mw).
    """

    bus: int
    capacity_mw: float
    Qmax: Optional[float] = None
    Qmin: Optional[float] = None
    Vg: float = 1.0
    dispatchable: bool = False
    fuel: Optional[str] = None


ModCommand = Union[
    SetLoad, ScaleLoad, ScaleAllLoads, SetGenStatus, SetGenDispatch,
    SetGenVoltage, SetBranchStatus, SetBranchRate, SetCostCoeffs,
    SetBusVLimits, SetAllBusVLimits, ScaleLoadProfile, ScaleWindScenario,
    SetTapRatio, SetShuntSusceptance, SetPhaseShiftAngle,
    AddLoadAtBus, AddGeneratorAtBus,
]

# Map action names to command classes and their required fields
_COMMAND_MAP: dict[str, tuple[type, set[str]]] = {
    "set_load": (SetLoad, {"bus"}),
    "scale_load": (ScaleLoad, {"factor"}),
    "scale_all_loads": (ScaleAllLoads, {"factor"}),
    "set_gen_status": (SetGenStatus, {"bus", "status"}),
    "set_gen_dispatch": (SetGenDispatch, {"bus", "Pg"}),
    "set_gen_voltage": (SetGenVoltage, {"bus", "Vg"}),
    "set_branch_status": (SetBranchStatus, {"fbus", "tbus", "status"}),
    "set_branch_rate": (SetBranchRate, {"fbus", "tbus", "rateA"}),
    "set_cost_coeffs": (SetCostCoeffs, {"bus", "coeffs"}),
    "set_bus_vlimits": (SetBusVLimits, {"bus"}),
    "set_all_bus_vlimits": (SetAllBusVLimits, set()),
    "scale_load_profile": (ScaleLoadProfile, {"factor"}),
    "scale_wind_scenario": (ScaleWindScenario, {"factor"}),
    "set_tap_ratio": (SetTapRatio, {"fbus", "tbus", "ratio"}),
    "set_shunt_susceptance": (SetShuntSusceptance, {"bus", "Bs"}),
    "set_phase_shift_angle": (SetPhaseShiftAngle, {"fbus", "tbus", "angle"}),
    "add_load_at_bus": (AddLoadAtBus, {"bus"}),
    "add_generator_at_bus": (AddGeneratorAtBus, {"bus", "capacity_mw"}),
}


def parse_command(raw: dict) -> ModCommand:
    """Parse a raw JSON dict into a typed ModCommand.

    The dict must have an ``"action"`` key matching one of the command
    names (e.g., ``"set_load"``, ``"scale_load"``).

    Raises:
        ValueError: If the action is unknown or required fields are missing.
    """
    action = raw.get("action")
    if action is None:
        raise ValueError("Command dict missing 'action' key")

    entry = _COMMAND_MAP.get(action)
    if entry is None:
        raise ValueError(f"Unknown action '{action}'. Valid: {sorted(_COMMAND_MAP)}")

    cls, required = entry
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Action '{action}' missing required fields: {missing}")

    # Build kwargs from raw, excluding 'action'
    kwargs = {k: v for k, v in raw.items() if k != "action"}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid fields for '{action}': {exc}") from exc
