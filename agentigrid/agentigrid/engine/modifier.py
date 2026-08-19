"""Modification applicator — applies validated commands to a MATNetwork."""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from agentigrid.engine.commands import (
    AddGeneratorAtBus,
    AddLoadAtBus,
    ModCommand,
    ScaleAllLoads,
    ScaleLoad,
    ScaleLoadProfile,
    ScaleWindScenario,
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
from agentigrid.engine.validation import validate_command
from agentigrid.parsers.matpower_model import GenCost, Generator, MATNetwork

_DEFAULT_PF_TAN = 0.3286  # tan(acos(0.95)) — reactive limit as fraction of MW capacity
# Fallback mid-merit linear cost ($/MWh) when the case has no usable polynomial curves.
_FALLBACK_GEN_COST = [0.0, 40.0, 0.0]

logger = logging.getLogger("agentigrid.engine.modifier")


def _median_existing_cost_coeffs(net: MATNetwork) -> list[float]:
    """Median model-2 polynomial cost curve [c2, c1, c0] across existing generators.

    Used as the default cost curve for a dispatchable added unit so it is
    economically mid-merit and actually partially dispatches, revealing
    location effects. Piecewise-linear (model 1) rows are skipped. Returns a
    fallback linear curve if no usable polynomial rows exist.
    """
    import statistics

    c2s: list[float] = []
    c1s: list[float] = []
    c0s: list[float] = []
    for gc in net.gencost:
        if gc.model != 2:  # skip piecewise-linear cost models
            continue
        co = list(gc.coeffs)
        if len(co) >= 3:
            c2, c1, c0 = co[-3], co[-2], co[-1]
        elif len(co) == 2:
            c2, c1, c0 = 0.0, co[0], co[1]
        elif len(co) == 1:
            c2, c1, c0 = 0.0, 0.0, co[0]
        else:
            continue
        c2s.append(c2)
        c1s.append(c1)
        c0s.append(c0)

    if not c1s:
        return list(_FALLBACK_GEN_COST)
    return [statistics.median(c2s), statistics.median(c1s), statistics.median(c0s)]


@dataclass
class ModificationReport:
    """Summary of all modifications applied in one iteration."""

    applied: list[tuple[ModCommand, str]] = field(default_factory=list)
    skipped: list[tuple[ModCommand, list[str]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    profile_paths: dict[str, Path] = field(default_factory=dict)
    scenario_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class _IndexMaps:
    """O(1) element-lookup maps built once from a network's current lists.

    Valid only while list lengths are stable. ``AddGeneratorAtBus`` grows the
    generators/gencost lists; after such a command the gen maps are stale and
    lookups must fall back to a linear scan (see ``_WorkCtx.gens_dirty``).
    """

    bus_by_id: dict[int, int]              # bus_i        -> index in net.buses
    gens_by_bus: dict[int, list[int]]      # bus          -> [indices in net.generators]
    branch_by_key: dict[tuple[int, int], list[int]]  # (min,max) fbus/tbus -> [indices]
    # Source list lengths at build time — every index the maps reference is < these,
    # so a caller-supplied map can be sanity-checked against a working net in O(1).
    n_buses: int = 0
    n_generators: int = 0
    n_branches: int = 0


def _build_index_maps(net: MATNetwork) -> _IndexMaps:
    """Build the O(1) lookup maps from *net*'s current lists (first-match order)."""
    bus_by_id: dict[int, int] = {}
    for i, b in enumerate(net.buses):
        if b.bus_i not in bus_by_id:  # preserve first-match semantics of a linear scan
            bus_by_id[b.bus_i] = i
    gens_by_bus: dict[int, list[int]] = {}
    for i, g in enumerate(net.generators):
        gens_by_bus.setdefault(g.bus, []).append(i)
    branch_by_key: dict[tuple[int, int], list[int]] = {}
    for i, br in enumerate(net.branches):
        key = (min(br.fbus, br.tbus), max(br.fbus, br.tbus))
        branch_by_key.setdefault(key, []).append(i)
    return _IndexMaps(
        bus_by_id, gens_by_bus, branch_by_key,
        n_buses=len(net.buses),
        n_generators=len(net.generators),
        n_branches=len(net.branches),
    )


def build_index_maps(net: MATNetwork) -> _IndexMaps:
    """Public wrapper: build reusable O(1) lookup maps for *net*.

    Callers that apply many modifications to the SAME base network (e.g. a sweep
    over candidate buses) can build the maps once here and pass them to each
    ``apply_modifications(..., index_maps=maps)`` call, avoiding an O(N) rebuild
    per call. The caller must guarantee the working net has the same bus/gen/branch
    ORDER and COUNT as *net*, except for pure appends (``AddGeneratorAtBus``, which
    is handled internally via ``ctx.gens_dirty``).
    """
    return _build_index_maps(net)


@dataclass
class _WorkCtx:
    """Per-call working state for ``_apply_one``.

    Holds the working network, its index maps, the copy mode, and (for COW) the
    set of already-cloned elements so repeat writes reuse the same private clone.
    ``gens_dirty`` flips once a generator is appended, after which gen index
    lookups fall back to a linear scan (the maps no longer cover the new tail).
    """

    net: MATNetwork
    maps: _IndexMaps
    copy_mode: str
    cloned: set = field(default_factory=set)
    gens_dirty: bool = False


def _find_bus(net: MATNetwork, bus_id: int, maps: "_IndexMaps | None" = None):
    """Return the first bus with ``bus_i == bus_id`` (O(1) via *maps* if given)."""
    if maps is not None:
        idx = maps.bus_by_id.get(bus_id)
        return net.buses[idx] if idx is not None else None
    for b in net.buses:
        if b.bus_i == bus_id:
            return b
    return None


def _find_gens_at_bus(net: MATNetwork, bus_id: int):
    return [g for g in net.generators if g.bus == bus_id]


def _find_branches(net: MATNetwork, fbus: int, tbus: int):
    return [
        br for br in net.branches
        if (br.fbus == fbus and br.tbus == tbus) or (br.fbus == tbus and br.tbus == fbus)
    ]


def _gen_index_in_network(
    net: MATNetwork, bus_id: int, gen_id: int | None, maps: "_IndexMaps | None" = None,
) -> int:
    """Return the index into net.generators for the specified gen at bus."""
    idx = gen_id if gen_id is not None else 0
    if maps is not None:
        return maps.gens_by_bus[bus_id][idx]
    gens_at_bus = []
    for i, g in enumerate(net.generators):
        if g.bus == bus_id:
            gens_at_bus.append(i)
    return gens_at_bus[idx]


def _branch_index_in_network(
    net: MATNetwork, fbus: int, tbus: int, ckt: int | None, maps: "_IndexMaps | None" = None,
) -> int:
    """Return the index into net.branches for the specified branch."""
    idx = ckt if ckt is not None else 0
    if maps is not None:
        return maps.branch_by_key[(min(fbus, tbus), max(fbus, tbus))][idx]
    matching = []
    for i, br in enumerate(net.branches):
        if (br.fbus == fbus and br.tbus == tbus) or (br.fbus == tbus and br.tbus == fbus):
            matching.append(i)
    return matching[idx]


# ── Copy-on-write element access ─────────────────────────────────────────────
# In "deep" mode the working network is already a private deepcopy, so these
# return the element as-is. In "cow" mode the working lists are shallow copies
# whose elements are still shared with the source network; the first write to a
# given (kind, idx) replaces that element with a shallow clone so the source is
# never mutated. All in-place mutations in _apply_one MUST go through these.

def _writable_bus(ctx: _WorkCtx, idx: int):
    if ctx.copy_mode == "cow":
        key = ("bus", idx)
        if key not in ctx.cloned:
            ctx.net.buses[idx] = replace(ctx.net.buses[idx])
            ctx.cloned.add(key)
    return ctx.net.buses[idx]


def _writable_gen(ctx: _WorkCtx, idx: int):
    if ctx.copy_mode == "cow":
        key = ("gen", idx)
        if key not in ctx.cloned:
            ctx.net.generators[idx] = replace(ctx.net.generators[idx])
            ctx.cloned.add(key)
    return ctx.net.generators[idx]


def _writable_branch(ctx: _WorkCtx, idx: int):
    if ctx.copy_mode == "cow":
        key = ("branch", idx)
        if key not in ctx.cloned:
            ctx.net.branches[idx] = replace(ctx.net.branches[idx])
            ctx.cloned.add(key)
    return ctx.net.branches[idx]


def _writable_gencost(ctx: _WorkCtx, idx: int):
    if ctx.copy_mode == "cow":
        key = ("gencost", idx)
        if key not in ctx.cloned:
            ctx.net.gencost[idx] = replace(ctx.net.gencost[idx])
            ctx.cloned.add(key)
    return ctx.net.gencost[idx]


def _ctx_gen_index(ctx: _WorkCtx, bus_id: int, gen_id: int | None) -> int:
    """Gen index via the maps, or a linear scan once the gen list has grown."""
    maps = None if ctx.gens_dirty else ctx.maps
    return _gen_index_in_network(ctx.net, bus_id, gen_id, maps)


def _apply_one(cmd: ModCommand, ctx: _WorkCtx, application: str | None = None) -> str:
    """Apply a single command to *ctx* (mutating the working net) and describe it."""

    net = ctx.net

    if isinstance(cmd, SetLoad):
        idx = ctx.maps.bus_by_id[cmd.bus]
        bus = _writable_bus(ctx, idx)
        parts = []
        if cmd.Pd is not None:
            bus.Pd = cmd.Pd
            parts.append(f"Pd={cmd.Pd} MW")
        if cmd.Qd is not None:
            bus.Qd = cmd.Qd
            parts.append(f"Qd={cmd.Qd} MVAr")
        return f"Set load at bus {cmd.bus}: {', '.join(parts)}"

    if isinstance(cmd, ScaleLoad):
        count = 0
        for i, bus in enumerate(net.buses):
            match = False
            if cmd.bus is not None and bus.bus_i == cmd.bus:
                match = True
            elif cmd.area is not None and bus.area == cmd.area:
                match = True
            elif cmd.zone is not None and bus.zone == cmd.zone:
                match = True
            if match:
                bus = _writable_bus(ctx, i)
                bus.Pd *= cmd.factor
                bus.Qd *= cmd.factor
                count += 1
        scope = f"bus {cmd.bus}" if cmd.bus else f"area {cmd.area}" if cmd.area else f"zone {cmd.zone}"
        return f"Scaled load by factor {cmd.factor} at {scope} ({count} bus(es))"

    if isinstance(cmd, ScaleAllLoads):
        for i in range(len(net.buses)):
            bus = _writable_bus(ctx, i)
            bus.Pd *= cmd.factor
            bus.Qd *= cmd.factor
        return f"Scaled all loads by factor {cmd.factor}"

    if isinstance(cmd, SetGenStatus):
        gi = _ctx_gen_index(ctx, cmd.bus, cmd.gen_id)
        _writable_gen(ctx, gi).status = cmd.status
        status_str = "ON" if cmd.status == 1 else "OFF"
        return f"Set generator at bus {cmd.bus} status to {status_str}"

    if isinstance(cmd, SetGenDispatch):
        gi = _ctx_gen_index(ctx, cmd.bus, cmd.gen_id)
        _writable_gen(ctx, gi).Pg = cmd.Pg
        return f"Set generator at bus {cmd.bus} Pg={cmd.Pg} MW"

    if isinstance(cmd, SetGenVoltage):
        gi = _ctx_gen_index(ctx, cmd.bus, cmd.gen_id)
        _writable_gen(ctx, gi).Vg = cmd.Vg
        if application == "pflow":
            return f"Set generator at bus {cmd.bus} Vg={cmd.Vg} pu (constrains bus voltage)"
        return f"Set generator at bus {cmd.bus} Vg={cmd.Vg} pu (initial guess only)"

    if isinstance(cmd, SetBranchStatus):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt, ctx.maps)
        _writable_branch(ctx, bi).status = cmd.status
        status_str = "in-service" if cmd.status == 1 else "out-of-service"
        return f"Set branch {cmd.fbus}-{cmd.tbus} status to {status_str}"

    if isinstance(cmd, SetBranchRate):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt, ctx.maps)
        _writable_branch(ctx, bi).rateA = cmd.rateA
        return f"Set branch {cmd.fbus}-{cmd.tbus} rateA={cmd.rateA} MVA"

    if isinstance(cmd, SetCostCoeffs):
        gi = _ctx_gen_index(ctx, cmd.bus, cmd.gen_id)
        if gi < len(net.gencost):
            gc = _writable_gencost(ctx, gi)
            gc.coeffs = list(cmd.coeffs)
            gc.ncost = len(cmd.coeffs)
        return f"Set cost coefficients for generator at bus {cmd.bus}: {cmd.coeffs}"

    if isinstance(cmd, SetTapRatio):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt, ctx.maps)
        br = _writable_branch(ctx, bi)
        old_ratio = br.ratio
        br.ratio = cmd.ratio
        return f"Set tap ratio for branch {cmd.fbus}-{cmd.tbus}: {old_ratio} -> {cmd.ratio}"

    if isinstance(cmd, SetShuntSusceptance):
        idx = ctx.maps.bus_by_id[cmd.bus]
        bus = _writable_bus(ctx, idx)
        old_bs = bus.Bs
        bus.Bs = cmd.Bs
        return f"Set shunt susceptance at bus {cmd.bus}: {old_bs} -> {cmd.Bs}"

    if isinstance(cmd, SetPhaseShiftAngle):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt, ctx.maps)
        br = _writable_branch(ctx, bi)
        old_angle = br.angle
        br.angle = cmd.angle
        return f"Set phase shift angle for branch {cmd.fbus}-{cmd.tbus}: {old_angle} -> {cmd.angle} deg"

    if isinstance(cmd, SetBusVLimits):
        idx = ctx.maps.bus_by_id[cmd.bus]
        bus = _writable_bus(ctx, idx)
        parts = []
        if cmd.Vmin is not None:
            bus.Vmin = cmd.Vmin
            parts.append(f"Vmin={cmd.Vmin}")
        if cmd.Vmax is not None:
            bus.Vmax = cmd.Vmax
            parts.append(f"Vmax={cmd.Vmax}")
        return f"Set bus {cmd.bus} voltage limits: {', '.join(parts)}"

    if isinstance(cmd, SetAllBusVLimits):
        count = 0
        for i in range(len(net.buses)):
            bus = _writable_bus(ctx, i)
            if cmd.Vmin is not None:
                bus.Vmin = cmd.Vmin
            if cmd.Vmax is not None:
                bus.Vmax = cmd.Vmax
            count += 1
        parts = []
        if cmd.Vmin is not None:
            parts.append(f"Vmin={cmd.Vmin}")
        if cmd.Vmax is not None:
            parts.append(f"Vmax={cmd.Vmax}")
        return f"Set voltage limits on all {count} buses: {', '.join(parts)}"

    if isinstance(cmd, AddLoadAtBus):
        idx = ctx.maps.bus_by_id[cmd.bus]
        bus = _writable_bus(ctx, idx)
        bus.Pd += cmd.Pd
        bus.Qd += cmd.Qd
        return f"Added load at bus {cmd.bus}: +{cmd.Pd} MW, +{cmd.Qd} MVAr (now Pd={bus.Pd}, Qd={bus.Qd})"

    if isinstance(cmd, AddGeneratorAtBus):
        cap = cmd.capacity_mw
        if cmd.dispatchable:
            pmin, pmax, pg = 0.0, cap, cap
        else:
            pmin, pmax, pg = cap, cap, cap  # forced injection
        qmax = cmd.Qmax if cmd.Qmax is not None else _DEFAULT_PF_TAN * cap
        qmin = cmd.Qmin if cmd.Qmin is not None else -_DEFAULT_PF_TAN * cap
        new_gen = Generator(
            bus=cmd.bus, Pg=pg, Qg=0.0, Qmax=qmax, Qmin=qmin, Vg=cmd.Vg,
            mBase=net.baseMVA, status=1, Pmax=pmax, Pmin=pmin, extra=[],
        )
        # Appending is COW-safe: the working lists are already private (shallow
        # copies in cow mode, deepcopy in deep mode), so the source is untouched.
        net.generators.append(new_gen)
        # MANDATORY: keep gencost positionally aligned with generators list.
        # Cost curve selection:
        #   - explicit cost_coeffs            → use as given
        #   - dispatchable + no coeffs        → case median (mid-merit), so the
        #                                       OPF partially dispatches it and the
        #                                       siting comparison is meaningful
        #   - fixed injection + no coeffs     → zero cost (irrelevant; Pg is pinned)
        if cmd.cost_coeffs is not None:
            coeffs = [float(c) for c in cmd.cost_coeffs]
        elif cmd.dispatchable:
            coeffs = _median_existing_cost_coeffs(net)
        else:
            coeffs = [0.0, 0.0]
        net.gencost.append(GenCost(
            model=2, startup=0.0, shutdown=0.0, ncost=len(coeffs), coeffs=coeffs,
        ))
        # The generators/gencost lists have grown; the prebuilt maps no longer
        # cover the new tail, so subsequent gen lookups must scan.
        ctx.gens_dirty = True
        kind = "dispatchable" if cmd.dispatchable else "forced-injection"
        return f"Added {kind} generator at bus {cmd.bus}: {cap} MW (Pmin={pmin}, Pmax={pmax})"

    return f"Unknown command type: {type(cmd).__name__}"


_SET_GEN_VOLTAGE_OPF_WARNING = (
    "set_gen_voltage only sets the initial guess; OPFLOW will override it with "
    "the optimal voltage. Use set_bus_vlimits to enforce voltage constraints in OPF."
)

_SET_GEN_VOLTAGE_PFLOW_NOTE = (
    "In PFLOW, set_gen_voltage directly constrains the bus voltage — the solver "
    "uses this as a fixed setpoint, not an initial guess. This is the primary "
    "mechanism for voltage control in power flow analysis."
)

_DCOPFLOW_VOLTAGE_CMD_TYPES = (SetGenVoltage, SetBusVLimits, SetAllBusVLimits)

def _dcopflow_voltage_skip_warning(cmd: ModCommand) -> str:
    """Return the skip warning message for a voltage command in DCOPFLOW context."""
    action_name = type(cmd).__name__
    # Convert class name to action name for the message
    _name_map = {
        "SetGenVoltage": "set_gen_voltage",
        "SetBusVLimits": "set_bus_vlimits",
        "SetAllBusVLimits": "set_all_bus_vlimits",
    }
    action = _name_map.get(action_name, action_name)
    return (
        f"Command '{action}' skipped: has no effect in DCOPFLOW "
        "(DC approximation ignores voltage magnitude)."
    )


_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING = (
    "scale_load_profile skipped: only applies to TCOPFLOW. "
    "Use scale_all_loads or scale_load for other applications."
)

_SCALE_WIND_SCENARIO_NON_SOPFLOW_WARNING = (
    "scale_wind_scenario skipped: only applies to SOPFLOW. "
    "Use scale_all_loads or scale_load for other applications."
)

# Column names that should be preserved (not scaled) in scenario CSV files.
# Single-period format: scenario_nr, <wind_cols>, weight
# Multi-period format: sim_timestamp, scenario_nr, <wind_cols>
_SCENARIO_NON_NUMERIC_COLUMNS = {"scenario_nr", "sim_timestamp", "weight"}


def scale_load_profile_csv(
    csv_path: Path,
    factor: float,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Scale numeric values in a TCOPFLOW load profile CSV by a factor.

    The first column (Timestamp) is preserved. All other columns are
    multiplied by *factor*. The result is written to *output_dir* with
    the same filename.

    Args:
        csv_path: Path to the original profile CSV.
        factor: Scaling factor (e.g., 1.2 for +20%).
        output_dir: Directory to write the scaled CSV.
        suffix: Optional suffix appended to the filename (e.g., "_scaled").

    Returns:
        Path to the written scaled CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem + suffix
    out_path = output_dir / f"{stem}.csv"

    rows_out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                rows_out.append(row)
                continue
            scaled = [row[0]]  # Timestamp column
            for val in row[1:]:
                try:
                    scaled.append(str(float(val) * factor))
                except ValueError:
                    scaled.append(val)
            rows_out.append(scaled)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    return out_path


def scale_wind_scenario_csv(
    csv_path: Path,
    factor: float,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Scale wind generation values in a SOPFLOW scenario CSV by a factor.

    Handles both scenario file formats:
    - Single-period: scenario_nr, <wind_cols>, weight
    - Multi-period: sim_timestamp, scenario_nr, <wind_cols>

    Non-numeric columns (scenario_nr, sim_timestamp, weight) are preserved.
    All other columns (wind generation values) are multiplied by *factor*.

    Args:
        csv_path: Path to the original scenario CSV.
        factor: Scaling factor (e.g., 1.5 for +50%).
        output_dir: Directory to write the scaled CSV.
        suffix: Optional suffix appended to the filename.

    Returns:
        Path to the written scaled CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem + suffix
    out_path = output_dir / f"{stem}.csv"

    rows_out: list[list[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_out.append(header)

        # Identify which columns are non-numeric (preserve them)
        non_numeric_cols = set()
        for i, col_name in enumerate(header):
            if col_name.strip().lower() in _SCENARIO_NON_NUMERIC_COLUMNS:
                non_numeric_cols.add(i)

        for row in reader:
            scaled = []
            for i, val in enumerate(row):
                if i in non_numeric_cols:
                    scaled.append(val)
                else:
                    try:
                        scaled.append(str(float(val) * factor))
                    except ValueError:
                        scaled.append(val)
            rows_out.append(scaled)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    return out_path


def _cow_working_copy(net: MATNetwork) -> MATNetwork:
    """Shallow-copy the container and each list attribute (elements stay shared).

    The returned network has fresh ``buses``/``generators``/``branches``/``gencost``
    lists and a fresh ``extra_sections`` dict, so appends and index replacement do
    not touch *net*; individual elements are still shared until cloned on write.
    """
    work = copy.copy(net)
    work.buses = list(net.buses)
    work.generators = list(net.generators)
    work.branches = list(net.branches)
    work.gencost = list(net.gencost)
    work.extra_sections = dict(net.extra_sections)
    return work


def apply_modifications(
    net: MATNetwork,
    commands: list[ModCommand],
    application: str | None = None,
    pload_profile: Path | None = None,
    qload_profile: Path | None = None,
    profile_output_dir: Path | None = None,
    scenario_file: Path | None = None,
    scenario_output_dir: Path | None = None,
    copy_mode: str = "deep",
    index_maps: "_IndexMaps | None" = None,
) -> tuple[MATNetwork, ModificationReport]:
    """Apply a list of modification commands to a network.

    Creates a working copy of the network before applying changes; *net* is
    never mutated. Each command is validated first; invalid commands are skipped.

    Args:
        net: The network to modify (not mutated).
        commands: List of commands to apply.
        application: ExaGO application name (e.g. "opflow"). When provided and
            equal to "opflow", a warning is added for any SetGenVoltage commands
            because OPFLOW treats Vg as an initial guess, not a constraint.
        pload_profile: Path to active load profile CSV (required for ScaleLoadProfile).
        qload_profile: Path to reactive load profile CSV (required for ScaleLoadProfile).
        profile_output_dir: Directory for scaled profile output (required for ScaleLoadProfile).
        scenario_file: Path to wind scenario CSV (required for ScaleWindScenario, SOPFLOW).
        scenario_output_dir: Directory for scaled scenario output (required for ScaleWindScenario).
        copy_mode: "deep" (default) deep-copies the whole network before mutating —
            fully private, behavior identical to the original. "cow" shallow-copies
            the container and its lists and clones individual elements only on first
            write (copy-on-write), which is ~O(1) per single-element mutation instead
            of O(N) — used by the sweep hot path. Both modes leave *net* unmutated
            and produce identical output for the same commands.
        index_maps: Prebuilt O(1) lookup maps (from ``build_index_maps``) to reuse
            instead of rebuilding them from the working net on every call. This is
            the sweep optimization: build once for the invariant base network and
            pass here per candidate, so the O(N) map build does not repeat N times.
            When None (default), the maps are built from the working net as before.
            PRECONDITION when supplied: *net* must have the same bus/gen/branch ORDER
            and COUNT as the network the maps were built from (pure appends via
            AddGeneratorAtBus are fine — handled via ctx.gens_dirty). A cheap length
            sanity check is asserted; the maps are never rebuilt here.

    Returns:
        Tuple of (modified_network, report). The report.profile_paths dict
        contains "pload_profile" and "qload_profile" keys with paths to
        scaled profile files when ScaleLoadProfile was applied.
    """
    if copy_mode == "cow":
        modified = _cow_working_copy(net)
    else:
        modified = copy.deepcopy(net)
    report = ModificationReport()

    # Index maps: reuse the caller's prebuilt maps (sweep hot path) or build once
    # from the WORKING net. Every index a map references is < its build-time source
    # count, so a cheap O(1) length check confirms the working net is compatible.
    if index_maps is not None:
        assert len(index_maps.bus_by_id) <= len(modified.buses), (
            "index_maps has more bus ids than the working net has buses"
        )
        assert index_maps.n_generators <= len(modified.generators), (
            "index_maps references generator indices beyond the working net"
        )
        assert index_maps.n_branches <= len(modified.branches), (
            "index_maps references branch indices beyond the working net"
        )
        maps = index_maps
    else:
        maps = _build_index_maps(modified)

    # O(1) mutation context bound to the working net.
    ctx = _WorkCtx(
        net=modified,
        maps=maps,
        copy_mode=copy_mode,
    )

    for cmd in commands:
        # Handle ScaleLoadProfile separately — it modifies CSV files, not the network
        if isinstance(cmd, ScaleLoadProfile):
            if application != "tcopflow":
                report.warnings.append(_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING)
                logger.warning(_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING)
                report.skipped.append((cmd, ["Only applicable for TCOPFLOW"]))
                continue
            if pload_profile is None or qload_profile is None:
                err = "ScaleLoadProfile requires pload_profile and qload_profile paths"
                report.skipped.append((cmd, [err]))
                logger.warning(err)
                continue
            out_dir = profile_output_dir or pload_profile.parent
            effective_factor = cmd.factor
            # If profiles were already scaled, apply factor cumulatively
            if report.profile_paths.get("pload_profile"):
                base_p = report.profile_paths["pload_profile"]
                base_q = report.profile_paths["qload_profile"]
            else:
                base_p = pload_profile
                base_q = qload_profile
            scaled_p = scale_load_profile_csv(base_p, effective_factor, out_dir)
            scaled_q = scale_load_profile_csv(base_q, effective_factor, out_dir)
            report.profile_paths["pload_profile"] = scaled_p
            report.profile_paths["qload_profile"] = scaled_q
            desc = f"Scaled load profiles by factor {cmd.factor} (P: {scaled_p.name}, Q: {scaled_q.name})"
            report.applied.append((cmd, desc))
            logger.info("Applied: %s", desc)
            continue

        # Handle ScaleWindScenario separately — it modifies CSV files, not the network
        if isinstance(cmd, ScaleWindScenario):
            if application != "sopflow":
                report.warnings.append(_SCALE_WIND_SCENARIO_NON_SOPFLOW_WARNING)
                logger.warning(_SCALE_WIND_SCENARIO_NON_SOPFLOW_WARNING)
                report.skipped.append((cmd, ["Only applicable for SOPFLOW"]))
                continue
            if scenario_file is None:
                err = "ScaleWindScenario requires a scenario_file path"
                report.skipped.append((cmd, [err]))
                logger.warning(err)
                continue
            out_dir = scenario_output_dir or scenario_file.parent
            scaled_scenario = scale_wind_scenario_csv(
                scenario_file, cmd.factor, out_dir,
            )
            report.scenario_paths["scenario_file"] = scaled_scenario
            desc = f"Scaled wind scenario by factor {cmd.factor} ({scaled_scenario.name})"
            report.applied.append((cmd, desc))
            logger.info("Applied: %s", desc)
            continue

        # Skip voltage commands entirely for DCOPFLOW (they have no effect)
        if application == "dcopflow" and isinstance(cmd, _DCOPFLOW_VOLTAGE_CMD_TYPES):
            warning = _dcopflow_voltage_skip_warning(cmd)
            report.warnings.append(warning)
            logger.warning(warning)
            continue

        # Forward the maps so validation is O(1) too when they were supplied.
        # Once a generator has been appended the maps no longer cover the tail, so
        # fall back to the linear-scan validation path for subsequent commands.
        _val_maps = None if ctx.gens_dirty else ctx.maps
        result = validate_command(cmd, modified, index_maps=_val_maps)
        report.warnings.extend(result.warnings)

        if not result.valid:
            report.skipped.append((cmd, result.errors))
            logger.warning("Skipped invalid command %s: %s", type(cmd).__name__, result.errors)
            continue

        # Emit OPF-specific warning for set_gen_voltage
        if isinstance(cmd, SetGenVoltage) and application == "opflow":
            report.warnings.append(_SET_GEN_VOLTAGE_OPF_WARNING)
            logger.warning(_SET_GEN_VOLTAGE_OPF_WARNING)

        if isinstance(cmd, SetGenVoltage) and application == "pflow":
            report.warnings.append(_SET_GEN_VOLTAGE_PFLOW_NOTE)
            logger.info(_SET_GEN_VOLTAGE_PFLOW_NOTE)

        desc = _apply_one(cmd, ctx, application=application)
        report.applied.append((cmd, desc))
        logger.info("Applied: %s", desc)

    return modified, report
