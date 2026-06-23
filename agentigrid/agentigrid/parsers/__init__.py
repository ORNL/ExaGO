"""File parsers for MATPOWER and simulation results."""

from __future__ import annotations

import logging

from agentigrid.parsers.matpower_model import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
)
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.matpower_writer import write_matpower
from agentigrid.parsers.network_metadata import network_metadata
from agentigrid.parsers.network_summary import network_summary
from agentigrid.parsers.opflow_parser import parse_opflow_output, parse_simulation_result
from agentigrid.parsers.opflow_results import (
    BranchResult,
    BusResult,
    GenResult,
    OPFLOWResult,
)
from agentigrid.parsers.results_summary import results_summary
from agentigrid.parsers.dcopflow_summary import dcopflow_results_summary
from agentigrid.parsers.scopflow_parser import (
    parse_scopflow_output,
    parse_scopflow_simulation_result,
)
from agentigrid.parsers.scopflow_summary import scopflow_results_summary
from agentigrid.parsers.tcopflow_parser import (
    parse_tcopflow_output,
    parse_tcopflow_simulation_result,
    parse_tcopflow_period_files,
    parse_tcopflow_metadata,
)
from agentigrid.parsers.tcopflow_summary import tcopflow_results_summary
from agentigrid.parsers.sopflow_parser import (
    parse_sopflow_output,
    parse_sopflow_simulation_result,
)
from agentigrid.parsers.sopflow_summary import sopflow_results_summary
from agentigrid.parsers.sopflow_dispatch import compute_wind_absorption
from agentigrid.parsers.pflow_parser import (
    parse_pflow_output,
    parse_pflow_simulation_result,
)
from agentigrid.parsers.pflow_summary import pflow_results_summary

_logger = logging.getLogger("agentigrid.parsers")

# Applications that use the OPFLOW parser directly (identical output format)
_OPFLOW_COMPATIBLE_APPS = {"opflow", "dcopflow"}


def parse_simulation_result_for_app(sim_result, application: str, bus_limits=None):
    """Dispatch to the correct parser based on application name.

    For 'opflow' and 'dcopflow': use parse_simulation_result (same output format).
    For 'scopflow': use parse_scopflow_simulation_result (same tables, different header).
      Returns the OPFLOWResult portion of the tuple (metadata available separately).
    For 'tcopflow': use parse_tcopflow_simulation_result (same tables, different header).
      Returns the OPFLOWResult portion of the tuple (metadata available separately).
    Falls back to OPFLOW parser for unknown applications with a warning.
    """
    if application in _OPFLOW_COMPATIBLE_APPS:
        return parse_simulation_result(sim_result, bus_limits=bus_limits)
    if application == "scopflow":
        parsed = parse_scopflow_simulation_result(sim_result, bus_limits=bus_limits)
        if parsed is None:
            return None
        opflow_result, _metadata = parsed
        return opflow_result
    if application == "tcopflow":
        parsed = parse_tcopflow_simulation_result(sim_result, bus_limits=bus_limits)
        if parsed is None:
            return None
        opflow_result, _metadata = parsed
        return opflow_result
    if application == "sopflow":
        parsed = parse_sopflow_simulation_result(sim_result, bus_limits=bus_limits)
        if parsed is None:
            return None
        opflow_result, _metadata = parsed
        return opflow_result
    if application == "pflow":
        parsed = parse_pflow_simulation_result(sim_result, bus_limits=bus_limits)
        if parsed is None:
            return None
        opflow_result, _metadata = parsed
        return opflow_result
    _logger.warning(
        "Unknown application '%s' for parse_simulation_result_for_app — "
        "falling back to OPFLOW parser.",
        application,
    )
    return parse_simulation_result(sim_result, bus_limits=bus_limits)


def parse_scopflow_metadata(sim_result) -> dict | None:
    """Extract SCOPFLOW-specific metadata (num_contingencies, multi_period).

    Returns a dict or None if parsing fails or the simulation did not succeed.
    """
    parsed = parse_scopflow_simulation_result(sim_result)
    if parsed is None:
        return None
    _opflow_result, metadata = parsed
    return metadata


def parse_tcopflow_metadata(sim_result) -> dict | None:
    """Extract TCOPFLOW-specific metadata (num_steps, duration, etc.).

    Returns a dict or None if parsing fails or the simulation did not succeed.
    """
    parsed = parse_tcopflow_simulation_result(sim_result)
    if parsed is None:
        return None
    _opflow_result, metadata = parsed
    return metadata


def parse_sopflow_metadata(sim_result) -> dict | None:
    """Extract SOPFLOW-specific metadata (num_scenarios, is_coupling, etc.).

    Returns a dict or None if parsing fails or the simulation did not succeed.
    """
    parsed = parse_sopflow_simulation_result(sim_result)
    if parsed is None:
        return None
    _opflow_result, metadata = parsed
    return metadata


def parse_pflow_metadata(sim_result) -> dict | None:
    """Extract PFLOW-specific metadata (solver, convergence_status).

    Returns a dict or None if parsing fails or the simulation did not succeed.
    """
    parsed = parse_pflow_simulation_result(sim_result)
    if parsed is None:
        return None
    _opflow_result, metadata = parsed
    return metadata


def results_summary_for_app(result: OPFLOWResult, application: str, **kwargs) -> str:
    """Dispatch to the correct results summary generator based on application.

    For 'opflow': use existing results_summary().
    For 'dcopflow': use dcopflow_results_summary().
    For 'scopflow': use scopflow_results_summary() with optional num_contingencies.
    For 'tcopflow': use tcopflow_results_summary() with optional TCOPFLOW kwargs.
    For 'sopflow': use sopflow_results_summary() with optional num_scenarios.
    For 'pflow': use pflow_results_summary() with optional gencost.
    For unknown applications: fall back to results_summary() with a warning.

    kwargs:
        num_contingencies (int): Number of contingencies enforced (SCOPFLOW only).
        num_steps (int): Number of time steps (TCOPFLOW only).
        duration_min (float): Duration in minutes (TCOPFLOW only).
        dT_min (float): Time-step size in minutes (TCOPFLOW only).
        is_coupling (bool): Ramp constraints enabled (TCOPFLOW only).
        period_data (list[dict]): Per-period parsed data (TCOPFLOW only).
        gencost (list[GenCost]): Generator cost curves for computing cost (PFLOW only).
    """
    if application == "dcopflow":
        return dcopflow_results_summary(result)
    if application == "opflow":
        return results_summary(result)
    if application == "scopflow":
        return scopflow_results_summary(
            result, num_contingencies=kwargs.get("num_contingencies", 0)
        )
    if application == "tcopflow":
        return tcopflow_results_summary(
            result,
            num_steps=kwargs.get("num_steps", 0),
            duration_min=kwargs.get("duration_min", 0.0),
            dT_min=kwargs.get("dT_min", 0.0),
            is_coupling=kwargs.get("is_coupling", True),
            period_data=kwargs.get("period_data"),
        )
    if application == "sopflow":
        return sopflow_results_summary(
            result,
            num_scenarios=kwargs.get("num_scenarios", 0),
            wind_absorption=kwargs.get("wind_absorption"),
        )
    if application == "pflow":
        return pflow_results_summary(
            result, gencost=kwargs.get("gencost"),
        )
    _logger.warning(
        "Unknown application '%s' for results_summary_for_app — "
        "falling back to OPFLOW summary.",
        application,
    )
    return results_summary(result)


__all__ = [
    "MATNetwork",
    "Bus",
    "Generator",
    "Branch",
    "GenCost",
    "parse_matpower",
    "write_matpower",
    "network_metadata",
    "network_summary",
    "OPFLOWResult",
    "BusResult",
    "BranchResult",
    "GenResult",
    "parse_opflow_output",
    "parse_simulation_result",
    "parse_simulation_result_for_app",
    "parse_scopflow_output",
    "parse_scopflow_simulation_result",
    "parse_scopflow_metadata",
    "parse_tcopflow_output",
    "parse_tcopflow_simulation_result",
    "parse_tcopflow_period_files",
    "parse_tcopflow_metadata",
    "parse_sopflow_output",
    "parse_sopflow_simulation_result",
    "parse_sopflow_metadata",
    "compute_wind_absorption",
    "parse_pflow_output",
    "parse_pflow_simulation_result",
    "parse_pflow_metadata",
    "results_summary",
    "results_summary_for_app",
    "dcopflow_results_summary",
    "scopflow_results_summary",
    "tcopflow_results_summary",
    "sopflow_results_summary",
    "pflow_results_summary",
]