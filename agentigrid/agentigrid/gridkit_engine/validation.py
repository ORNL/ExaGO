"""Checks of GridKit commands and fault-study requests against the element map."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.commands import AddBusFault, CaseCommand, FaultStudy, RecordVariables
from agentigrid.gridkit_engine.sweep_metrics import CHECKS
from agentigrid.gridkit_engine.topology import fault_locations

logger = logging.getLogger("agentigrid.gridkit_engine.validation")

# Same limit as the ExaGO contingency screen (contingency_max_count).
MAX_FAULTS = 500

# Variables every GridKit machine model (GENROU, GENSAL, GenClassical) and bus can record.
MACHINE_VARIABLES = frozenset({"ir", "ii", "p", "q", "delta", "omega", "speed"})
BUS_VARIABLES = frozenset({"Vr", "Vi", "Vm", "Va"})


@dataclass
class ValidationResult:
    valid: bool
    item: Any
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _bus_numbers(element_map: dict) -> set[int]:
    return {b["bus"] for b in element_map["buses"]}


def _check_impedance(R: float, X: float, errors: list[str]) -> None:
    if R < 0 or X < 0:
        errors.append(f"Fault impedance must not be negative (R={R}, X={X})")
    elif R == 0 and X == 0:
        errors.append("Fault impedance R and X cannot both be 0")


def validate_command(cmd: CaseCommand, element_map: dict) -> ValidationResult:
    """Check a case command: bus exists, impedance valid, variables recordable."""
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(cmd, AddBusFault):
        if cmd.bus not in _bus_numbers(element_map):
            errors.append(f"Bus {cmd.bus} does not exist in the case")
        _check_impedance(cmd.R, cmd.X, errors)
        if any(f["bus"] == cmd.bus for f in element_map["bus_faults"]):
            warnings.append(f"Bus {cmd.bus} already has a fault device; its R and X are set on the copy")
    elif isinstance(cmd, RecordVariables):
        bad_m = sorted(set(cmd.machines) - MACHINE_VARIABLES)
        bad_b = sorted(set(cmd.buses) - BUS_VARIABLES)
        if bad_m:
            errors.append(f"Machines cannot record {bad_m}; available: {sorted(MACHINE_VARIABLES)}")
        if bad_b:
            errors.append(f"Buses cannot record {bad_b}; available: {sorted(BUS_VARIABLES)}")
    else:
        errors.append(f"Unknown command type {type(cmd).__name__}")
    return ValidationResult(not errors, cmd, warnings, errors)


def resolve_fault_buses(study: FaultStudy, element_map: dict) -> list[int]:
    """Bus numbers the study faults, in study order (POI first for poi mode)."""
    if study.all_buses:
        return sorted(_bus_numbers(element_map))
    if study.poi is not None:
        return fault_locations(element_map, study.poi, study.hops)
    return list(dict.fromkeys(study.buses or []))


def validate_fault_study(
    study: FaultStudy, element_map: dict, max_faults: int = MAX_FAULTS, freq_hz: float = 60.0,
) -> ValidationResult:
    """Check a fault-study request before anything is solved."""
    errors: list[str] = []
    warnings: list[str] = []
    buses = _bus_numbers(element_map)

    modes = sum([bool(study.buses), study.all_buses, study.poi is not None])
    if modes != 1:
        errors.append("Give exactly one of: buses, all_buses, poi")
    if study.buses:
        missing = sorted(set(study.buses) - buses)
        if missing:
            errors.append(f"Buses not in the case: {missing[:20]}")
    if study.poi is not None:
        if study.poi not in buses:
            errors.append(f"POI bus {study.poi} is not in the case")
        if study.hops < 0:
            errors.append("hops must be 0 or more")

    _check_impedance(study.R, study.X, errors)
    duration = study.duration_s(freq_hz)
    clear = study.clear_time_s(freq_hz)
    if study.start_s <= 0:
        errors.append("start_s must be after 0 s so the pre-fault state is recorded")
    if study.clearing_cycles <= 0 or study.margin_cycles < 0:
        errors.append("clearing_cycles must be positive and margin_cycles not negative")
    if clear >= study.tmax_s:
        errors.append(f"Fault clears at {clear:.3f} s, not before tmax_s = {study.tmax_s} s")
    if study.dt_monitor_s <= 0:
        errors.append("dt_monitor_s must be positive")
    elif study.dt_monitor_s > duration / 2:
        warnings.append(
            f"dt_monitor_s = {study.dt_monitor_s} s records fewer than two points during the "
            f"{duration:.3f} s fault"
        )

    unknown = sorted(set(study.checks) - set(CHECKS))
    if unknown:
        errors.append(f"Unknown checks {unknown}; available: {sorted(CHECKS)}")
    if not study.checks:
        errors.append("At least one check is needed")
    if "damping" in study.checks and study.tmax_s < criteria.DAMPING_MIN_RUN_S:
        warnings.append(
            f"tmax_s = {study.tmax_s} s is shorter than {criteria.DAMPING_MIN_RUN_S} s; "
            "damping will be reported as not tested"
        )
    if "voltage_recovery" in study.checks and study.tmax_s < clear + criteria.V_RECOVERY_AFTER_S:
        warnings.append("Run ends before 2.5 s after clearing; voltage recovery will be not tested")

    if not errors:
        n = len(resolve_fault_buses(study, element_map))
        if n == 0:
            errors.append("The study resolves to no buses")
        elif n > max_faults:
            errors.append(f"The number of bus faults ({n}) exceeds {max_faults}")
    return ValidationResult(not errors, study, warnings, errors)
