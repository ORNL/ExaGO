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
from pathlib import Path

from agentigrid.parsers.matpower_parser import parse_matpower

logger = logging.getLogger("agentigrid.parsers.sopflow_dispatch")

# Scenario CSV columns that are not wind generators.
_NON_WIND_COLUMNS = {"scenario_nr", "sim_timestamp", "weight"}


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
