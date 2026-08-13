"""Second-stage wind absorption metric for SOPFLOW.

SOPFLOW models scenario wind as a zero-cost, curtailable upper bound: the
solver dispatches wind up to whatever the network can absorb (a capacity P*)
and curtails the surplus. The well-posed signal is therefore ABSORBED wind
(dispatched) vs OFFERED wind (available) and the resulting CURTAILMENT.

This module reads the per-scenario second-stage solution files that SOPFLOW
writes under ``sopflowout/`` with ``-save_output`` and aggregates that signal.
"""

from __future__ import annotations

import csv
import logging
import re
import statistics
from pathlib import Path

from agentigrid.parsers.matpower_parser import parse_matpower

logger = logging.getLogger("agentigrid.parsers.sopflow_dispatch")

# Scenario CSV columns that are not wind generators.
_NON_WIND_COLUMNS = {"scenario_nr", "sim_timestamp", "weight"}

# ``mpc.converged = <0|1>`` written into each second-stage scenario solution file.
_SCEN_CONVERGED_RE = re.compile(r"mpc\.converged\s*=\s*(\d+)")


def all_scenarios_converged(workdir: Path) -> bool | None:
    """Whether every SOPFLOW second-stage scenario actually converged.

    EMPAR's SOPFLOW header can report CONVERGED even when individual scenario
    subproblems failed (their ``sopflowout/scen_*.m`` carry ``mpc.converged = 0``
    and ``mpc.obj = 0``). This reads the same scen files as
    :func:`compute_wind_absorption` and returns:
      - ``True``  if every scenario file has ``mpc.converged = 1``
      - ``False`` if any scenario file reports non-converged (or is unreadable)
      - ``None``  if no ``sopflowout/scen_*.m`` files exist (nothing to check)

    File/parse errors are wrapped so a bad workdir never raises. A file whose
    converged flag cannot be read is treated as NOT converged (fail-safe: we
    must never certify a solve we could not verify).
    """
    try:
        workdir = Path(workdir)
        sopflowout = workdir / "sopflowout"
        if not sopflowout.is_dir():
            return None
        scen_files = sorted(sopflowout.glob("scen_*.m"))
        if not scen_files:
            return None
        for sol in scen_files:
            try:
                text = sol.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.debug("all_scenarios_converged: cannot read %s: %s", sol, exc)
                return False  # unreadable → cannot certify
            m = _SCEN_CONVERGED_RE.search(text)
            if m is None or m.group(1) != "1":
                return False
        return True
    except Exception as exc:  # noqa: BLE001 — never let a bad workdir raise
        logger.debug("all_scenarios_converged: unexpected error: %s", exc)
        return False


def _parse_wind_columns(csv_path: Path) -> tuple[list[str], dict[str, int]]:
    """Return (wind_cols, {column -> bus}) derived from the CSV header."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    wind_cols = [c for c in header if c.strip().lower() not in _NON_WIND_COLUMNS]
    bus_map: dict[str, int] = {}
    for col in wind_cols:
        try:
            bus_map[col] = int(col.split("_")[0])
        except (ValueError, IndexError):
            pass
    return wind_cols, bus_map


def _load_offered_wind(csv_path: Path, wind_cols: list[str]) -> dict[int, dict[str, float]]:
    """Return {scenario_nr(int): {col: float}} of offered wind per scenario."""
    offered: dict[int, dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scen = int(float(row["scenario_nr"]))
            except (KeyError, ValueError):
                continue
            offered[scen] = {}
            for col in wind_cols:
                try:
                    offered[scen][col] = float(row[col])
                except (KeyError, ValueError):
                    offered[scen][col] = 0.0
    return offered


def compute_wind_absorption(workdir: Path, scenario_csv: Path) -> dict | None:
    """Aggregate second-stage wind dispatch vs availability across scenarios.

    Reads ``sopflowout/scen_*.m`` under *workdir* (flat files, ``scen_<N>.m``,
    N 0-based) and the scenario CSV (offered wind per scenario). Returns
    aggregate absorption figures, or None if no per-scenario files are found /
    parsing fails.
    """
    workdir = Path(workdir)
    scenario_csv = Path(scenario_csv)

    if not scenario_csv.exists():
        logger.debug("compute_wind_absorption: scenario CSV not found: %s", scenario_csv)
        return None

    sopflowout = workdir / "sopflowout"
    if not sopflowout.is_dir():
        logger.debug("compute_wind_absorption: sopflowout/ not found in %s", workdir)
        return None

    scen_files = sorted(sopflowout.glob("scen_*.m"))
    if not scen_files:
        logger.debug("compute_wind_absorption: no scen_*.m files in %s", sopflowout)
        return None

    wind_cols, bus_map = _parse_wind_columns(scenario_csv)
    wind_buses = set(bus_map.values())
    offered = _load_offered_wind(scenario_csv, wind_cols)

    num_parsed = 0
    total_available = 0.0
    total_dispatched = 0.0

    for sol in scen_files:
        try:
            scen_nr = int(sol.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        try:
            net = parse_matpower(sol)
        except Exception as exc:  # noqa: BLE001 — skip unparseable solution files
            logger.debug("compute_wind_absorption: failed to parse %s: %s", sol, exc)
            continue

        # Identify wind generators: prefer fuel match, fall back to bus set.
        wind_gens = [
            g for g in net.generators
            if "wind" in str(getattr(g, "fuel", "")).lower()
        ]
        if not wind_gens:
            wind_gens = [g for g in net.generators if g.bus in wind_buses]

        dispatched = sum(g.Pg for g in wind_gens)

        # Saved file index is 0-based; CSV scenario_nr is 1-based — fall back.
        scen_offered = offered.get(scen_nr)
        if scen_offered is None:
            scen_offered = offered.get(scen_nr + 1, {})
        available = sum(scen_offered.values())

        total_available += available
        total_dispatched += dispatched
        num_parsed += 1

    if num_parsed == 0:
        return None

    total_curtailment = total_available - total_dispatched
    curtailment_pct = (
        total_curtailment / total_available * 100.0 if total_available else 0.0
    )

    return {
        "num_scenarios": num_parsed,
        "total_available_mw": total_available,
        "total_dispatched_mw": total_dispatched,
        "total_curtailment_mw": total_curtailment,
        "curtailment_pct": curtailment_pct,
        "avg_dispatched_mw": total_dispatched / num_parsed,
    }


def compute_scenario_voltage_spread(workdir: Path) -> list[dict] | None:
    """Per-bus voltage spread across SOPFLOW second-stage scenarios.

    The bus whose voltage swings most across the wind scenarios is the most
    wind-affected. Reads every ``sopflowout/scen_*.m`` under *workdir* (the same
    discovery/parse path as :func:`compute_wind_absorption`) and, for each bus,
    collects ``Vm`` across all scenarios.

    Returns a list of per-bus dicts
    ``{bus, v_min, v_max, v_range, v_std, v_mean, n_scenarios}`` sorted by
    ``v_range`` descending, or ``None`` if fewer than 2 scenario files are found
    or nothing could be parsed. File/parse errors are wrapped so a bad workdir
    yields ``None`` rather than raising.
    """
    try:
        workdir = Path(workdir)
        sopflowout = workdir / "sopflowout"
        if not sopflowout.is_dir():
            logger.debug(
                "compute_scenario_voltage_spread: sopflowout/ not found in %s", workdir
            )
            return None

        scen_files = sorted(sopflowout.glob("scen_*.m"))
        if len(scen_files) < 2:
            logger.debug(
                "compute_scenario_voltage_spread: need >=2 scen_*.m files in %s (found %d)",
                sopflowout, len(scen_files),
            )
            return None

        # bus -> list of Vm observed across scenarios (in file-discovery order).
        vm_by_bus: dict[int, list[float]] = {}
        num_parsed = 0
        for sol in scen_files:
            try:
                net = parse_matpower(sol)
            except Exception as exc:  # noqa: BLE001 — skip unparseable solution files
                logger.debug(
                    "compute_scenario_voltage_spread: failed to parse %s: %s", sol, exc
                )
                continue
            for b in net.buses:
                vm_by_bus.setdefault(b.bus_i, []).append(float(b.Vm))
            num_parsed += 1

        if num_parsed < 2 or not vm_by_bus:
            return None

        rows: list[dict] = []
        for bus, vms in vm_by_bus.items():
            if not vms:
                continue
            v_min = min(vms)
            v_max = max(vms)
            v_mean = statistics.fmean(vms)
            v_std = statistics.pstdev(vms) if len(vms) > 1 else 0.0
            rows.append({
                "bus": bus,
                "v_min": v_min,
                "v_max": v_max,
                "v_range": v_max - v_min,
                "v_std": v_std,
                "v_mean": v_mean,
                "n_scenarios": len(vms),
            })

        if not rows:
            return None

        # Descending by spread; tie-break by bus id for deterministic output.
        rows.sort(key=lambda r: (-r["v_range"], r["bus"]))
        return rows
    except Exception as exc:  # noqa: BLE001 — never let a bad workdir raise
        logger.debug("compute_scenario_voltage_spread: unexpected error: %s", exc)
        return None
