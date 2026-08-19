"""Command validation against a MATNetwork."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentigrid.engine.commands import (
    AddGeneratorAtBus,
    AddLoadAtBus,
    ModCommand,
    ScaleAllLoads,
    ScaleLoad,
    SetAllBusVLimits,
    SetBranchRate,
    SetBranchStatus,
    SetBusVLimits,
    SetCostCoeffs,
    SetGenDispatch,
    SetGenStatus,
    SetGenVoltage,
    SetLoad,
    SetPhaseShiftAngle,
    SetShuntSusceptance,
    SetTapRatio,
)
from agentigrid.parsers.matpower_model import MATNetwork

logger = logging.getLogger("agentigrid.engine.validation")


@dataclass
class ValidationResult:
    """Result of validating a single command against a network."""

    valid: bool
    command: ModCommand
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _find_bus(net: MATNetwork, bus_id: int, index_maps=None):
    """Return the bus or None (O(1) via *index_maps* when supplied)."""
    if index_maps is not None:
        idx = index_maps.bus_by_id.get(bus_id)
        return net.buses[idx] if idx is not None else None
    for b in net.buses:
        if b.bus_i == bus_id:
            return b
    return None


def _find_gens_at_bus(net: MATNetwork, bus_id: int, index_maps=None):
    """Return list of generators at the given bus (order preserved)."""
    if index_maps is not None:
        return [net.generators[i] for i in index_maps.gens_by_bus.get(bus_id, ())]
    return [g for g in net.generators if g.bus == bus_id]


def _find_branches(net: MATNetwork, fbus: int, tbus: int, index_maps=None):
    """Return list of branches matching (fbus, tbus) in either direction."""
    if index_maps is not None:
        key = (min(fbus, tbus), max(fbus, tbus))
        return [net.branches[i] for i in index_maps.branch_by_key.get(key, ())]
    return [
        br for br in net.branches
        if (br.fbus == fbus and br.tbus == tbus) or (br.fbus == tbus and br.tbus == fbus)
    ]


def _validate_bus_exists(net: MATNetwork, bus_id: int, errors: list[str], index_maps=None) -> bool:
    if _find_bus(net, bus_id, index_maps) is None:
        errors.append(f"Bus {bus_id} does not exist in the network")
        return False
    return True


def _validate_gen_at_bus(
    net: MATNetwork, bus_id: int, gen_id: int | None, errors: list[str], index_maps=None,
):
    """Validate generator exists and return it, or None."""
    gens = _find_gens_at_bus(net, bus_id, index_maps)
    if not gens:
        errors.append(f"No generator at bus {bus_id}")
        return None
    idx = gen_id if gen_id is not None else 0
    if idx < 0 or idx >= len(gens):
        errors.append(f"gen_id={idx} out of range (bus {bus_id} has {len(gens)} generator(s))")
        return None
    return gens[idx]


def _validate_branch(
    net: MATNetwork, fbus: int, tbus: int, ckt: int | None, errors: list[str], index_maps=None,
):
    """Validate branch exists and return it, or None."""
    branches = _find_branches(net, fbus, tbus, index_maps)
    if not branches:
        errors.append(f"No branch between bus {fbus} and bus {tbus}")
        return None
    idx = ckt if ckt is not None else 0
    if idx < 0 or idx >= len(branches):
        errors.append(f"ckt={idx} out of range ({len(branches)} branch(es) between bus {fbus} and {tbus})")
        return None
    return branches[idx]


def validate_command(cmd: ModCommand, net: MATNetwork, index_maps=None) -> ValidationResult:
    """Validate a command against the network.

    Checks bus/branch/generator existence, numerical bounds, and
    flags warnings for unusual but non-fatal values.

    Args:
        cmd: The command to validate.
        net: The network to validate against.
        index_maps: Optional prebuilt O(1) lookup maps (duck-typed: exposes
            ``bus_by_id``, ``gens_by_bus``, ``branch_by_key``). When supplied,
            existence/index checks use the maps instead of linear scans; the
            returned (valid, errors, warnings) are identical either way.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if isinstance(cmd, SetLoad):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        if cmd.Pd is not None and cmd.Pd < 0:
            warnings.append(f"Negative Pd={cmd.Pd} at bus {cmd.bus} (generation as negative load)")
        if cmd.Qd is not None and cmd.Qd < 0:
            warnings.append(f"Negative Qd={cmd.Qd} at bus {cmd.bus}")

    elif isinstance(cmd, ScaleLoad):
        if cmd.factor <= 0:
            errors.append(f"Scale factor must be > 0, got {cmd.factor}")
        if cmd.factor > 3.0:
            warnings.append(f"Very large scale factor: {cmd.factor}")
        if cmd.bus is not None:
            _validate_bus_exists(net, cmd.bus, errors, index_maps)
        if cmd.area is not None:
            areas = {b.area for b in net.buses}
            if cmd.area not in areas:
                errors.append(f"Area {cmd.area} does not exist (valid: {sorted(areas)})")
        if cmd.zone is not None:
            zones = {b.zone for b in net.buses}
            if cmd.zone not in zones:
                errors.append(f"Zone {cmd.zone} does not exist (valid: {sorted(zones)})")

    elif isinstance(cmd, ScaleAllLoads):
        if cmd.factor <= 0:
            errors.append(f"Scale factor must be > 0, got {cmd.factor}")
        if cmd.factor > 3.0:
            warnings.append(f"Very large scale factor: {cmd.factor}")

    elif isinstance(cmd, SetGenStatus):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        if cmd.status not in (0, 1):
            errors.append(f"Status must be 0 or 1, got {cmd.status}")
        gen = _validate_gen_at_bus(net, cmd.bus, cmd.gen_id, errors, index_maps)
        if gen is not None and gen.status == 0 and cmd.status == 0:
            warnings.append(f"Generator at bus {cmd.bus} is already offline")

    elif isinstance(cmd, SetGenDispatch):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        bus = _find_bus(net, cmd.bus, index_maps)
        if bus is not None and bus.type == 3:
            errors.append(
                f"Bus {cmd.bus} is the reference/slack bus (type=3). "
                f"Its generator output is determined by the power balance — "
                f"set_gen_dispatch has no effect on the slack bus."
            )
        gen = _validate_gen_at_bus(net, cmd.bus, cmd.gen_id, errors, index_maps)
        if gen is not None:
            if cmd.Pg < gen.Pmin or cmd.Pg > gen.Pmax:
                errors.append(
                    f"Pg={cmd.Pg} outside bounds [{gen.Pmin}, {gen.Pmax}] "
                    f"for generator at bus {cmd.bus}"
                )

    elif isinstance(cmd, SetGenVoltage):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        _validate_gen_at_bus(net, cmd.bus, cmd.gen_id, errors, index_maps)
        if cmd.Vg < 0.8 or cmd.Vg > 1.2:
            errors.append(f"Vg={cmd.Vg} outside reasonable range [0.8, 1.2]")

    elif isinstance(cmd, SetBranchStatus):
        if cmd.status not in (0, 1):
            errors.append(f"Status must be 0 or 1, got {cmd.status}")
        _validate_branch(net, cmd.fbus, cmd.tbus, cmd.ckt, errors, index_maps)

    elif isinstance(cmd, SetBranchRate):
        if cmd.rateA < 0:
            errors.append(f"rateA must be >= 0, got {cmd.rateA}")
        _validate_branch(net, cmd.fbus, cmd.tbus, cmd.ckt, errors, index_maps)

    elif isinstance(cmd, SetCostCoeffs):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        _validate_gen_at_bus(net, cmd.bus, cmd.gen_id, errors, index_maps)

    elif isinstance(cmd, SetBusVLimits):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        if cmd.Vmin is not None and cmd.Vmax is not None and cmd.Vmin >= cmd.Vmax:
            errors.append(f"Vmin={cmd.Vmin} must be < Vmax={cmd.Vmax}")

    elif isinstance(cmd, SetAllBusVLimits):
        if cmd.Vmin is None and cmd.Vmax is None:
            warnings.append("set_all_bus_vlimits has no effect: neither Vmin nor Vmax was provided")
        if cmd.Vmin is not None and cmd.Vmax is not None and cmd.Vmin >= cmd.Vmax:
            errors.append(f"Vmin={cmd.Vmin} must be < Vmax={cmd.Vmax}")
        for v, name in ((cmd.Vmin, "Vmin"), (cmd.Vmax, "Vmax")):
            if v is not None and (v < 0.5 or v > 1.5):
                warnings.append(f"{name}={v} is outside the reasonable range [0.5, 1.5] pu")

    elif isinstance(cmd, SetTapRatio):
        branch = _validate_branch(net, cmd.fbus, cmd.tbus, cmd.ckt, errors, index_maps)
        if branch is not None:
            if branch.ratio == 0:
                errors.append(
                    f"Branch {cmd.fbus}-{cmd.tbus} is not a transformer "
                    f"(tap ratio is 0). set_tap_ratio only applies to transformers."
                )
            if cmd.ratio <= 0:
                errors.append(f"Tap ratio must be > 0, got {cmd.ratio}")
            if cmd.ratio < 0.8 or cmd.ratio > 1.2:
                warnings.append(
                    f"Tap ratio {cmd.ratio} is outside the typical range [0.8, 1.2]"
                )

    elif isinstance(cmd, SetShuntSusceptance):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)

    elif isinstance(cmd, SetPhaseShiftAngle):
        branch = _validate_branch(net, cmd.fbus, cmd.tbus, cmd.ckt, errors, index_maps)
        if branch is not None:
            if branch.angle == 0:
                errors.append(
                    f"Branch {cmd.fbus}-{cmd.tbus} is not a phase shifter "
                    f"(angle is 0). set_phase_shift_angle only applies to phase shifters."
                )
            if abs(cmd.angle) > 90:
                warnings.append(
                    f"Phase shift angle {cmd.angle} degrees is very large (typical range: -60 to 60)"
                )

    elif isinstance(cmd, AddLoadAtBus):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)

    elif isinstance(cmd, AddGeneratorAtBus):
        _validate_bus_exists(net, cmd.bus, errors, index_maps)
        if cmd.capacity_mw <= 0:
            errors.append(f"capacity_mw must be > 0, got {cmd.capacity_mw}")
        if cmd.dispatchable and cmd.Qmax is not None and cmd.Qmin is not None:
            if cmd.Qmax < cmd.Qmin:
                errors.append(
                    f"Qmax={cmd.Qmax} must be >= Qmin={cmd.Qmin} for dispatchable generator"
                )

    return ValidationResult(
        valid=len(errors) == 0,
        command=cmd,
        warnings=warnings,
        errors=errors,
    )
