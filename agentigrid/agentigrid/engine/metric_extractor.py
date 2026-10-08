"""Metric extraction for multi-objective tracking.

Provides deterministic extractors for standard OPFLOW metrics. Each extractor
takes an OPFLOWResult and returns a float value. Unknown metric names return None.
"""

from __future__ import annotations

import logging
from typing import Optional

from agentigrid.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("agentigrid.engine.metric_extractor")


def extract_metric(result: OPFLOWResult, metric_name: str) -> Optional[float]:
    """Extract a named metric value from an OPFLOW result.

    Args:
        result: Parsed OPFLOW simulation result.
        metric_name: Standardized metric name.

    Returns:
        Extracted float value, or None if the metric is unknown or cannot be computed.
    """
    extractor = _EXTRACTORS.get(metric_name)
    if extractor is not None:
        try:
            return extractor(result)
        except Exception as exc:
            logger.warning("Failed to extract metric '%s': %s", metric_name, exc)
            return None
    logger.debug("No extractor for metric '%s'", metric_name)
    return None


def extract_all_metrics(
    result: OPFLOWResult,
    metric_names: list[str],
) -> dict[str, float]:
    """Extract multiple metrics from an OPFLOW result.

    Returns a dict mapping metric name to value. Metrics that cannot be
    extracted are omitted.
    """
    metrics = {}
    for name in metric_names:
        value = extract_metric(result, name)
        if value is not None:
            metrics[name] = value
    return metrics


def available_metrics() -> list[str]:
    """Return the list of all metric names with known extractors."""
    return list(_EXTRACTORS.keys())


# Metrics that are NOT meaningful for DCOPFLOW
_DC_EXCLUDED_METRICS = {
    "voltage_min", "voltage_max", "voltage_deviation", "voltage_range",
    "total_reactive_gen_mvar",
}


def available_metrics_for_app(application: str) -> list[str]:
    """Return metrics relevant for the given application.

    For PFLOW, 'generation_cost' is always 0.0 because PFLOW has no
    optimization objective. The computed generation cost (dispatch x cost curves)
    is shown in the results summary instead.
    """
    all_metrics = list(_EXTRACTORS.keys())
    if application == "dcopflow":
        return [m for m in all_metrics if m not in _DC_EXCLUDED_METRICS]
    if application == "pflow":
        _PFLOW_EXCLUDED = {"generation_cost"}
        return [m for m in all_metrics if m not in _PFLOW_EXCLUDED]
    if application in ("tcopflow", "sopflow"):
        return all_metrics
    return all_metrics


# ── Extractor functions ──────────────────────────────────────────────────────

def _generation_cost(r: OPFLOWResult) -> float:
    return r.objective_value

def _total_generation_mw(r: OPFLOWResult) -> float:
    return r.total_gen_mw

def _total_load_mw(r: OPFLOWResult) -> float:
    return r.total_load_mw

def _active_losses_mw(r: OPFLOWResult) -> float:
    return r.total_gen_mw - r.total_load_mw

def _voltage_min(r: OPFLOWResult) -> float:
    return r.voltage_min

def _voltage_max(r: OPFLOWResult) -> float:
    return r.voltage_max

def _voltage_deviation(r: OPFLOWResult) -> float:
    """Max absolute deviation from 1.0 pu across all buses."""
    if not r.buses:
        return 0.0
    return max(abs(b.Vm - 1.0) for b in r.buses)

def _voltage_range(r: OPFLOWResult) -> float:
    """Spread between highest and lowest bus voltage."""
    return r.voltage_max - r.voltage_min

def _max_line_loading_pct(r: OPFLOWResult) -> float:
    return r.max_line_loading_pct

def _mean_line_loading_pct(r: OPFLOWResult) -> float:
    """Average loading across all lines with nonzero limits."""
    loaded = []
    for br in r.branches:
        if br.Slim > 0:
            pct = max(br.Sf, br.St) / br.Slim * 100
            loaded.append(pct)
    return sum(loaded) / len(loaded) if loaded else 0.0

def _violation_count(r: OPFLOWResult) -> float:
    return float(r.num_violations)

def _total_reactive_gen_mvar(r: OPFLOWResult) -> float:
    return r.total_gen_mvar

def _online_generator_count(r: OPFLOWResult) -> float:
    return float(sum(1 for g in r.generators if g.status == 1))

def _generation_reserve_mw(r: OPFLOWResult) -> float:
    """Total unused generation capacity (Pmax - Pg) for online generators."""
    return sum(g.Pmax - g.Pg for g in r.generators if g.status == 1 and g.Pg < g.Pmax)

def _phase_angle_range(r: OPFLOWResult) -> float:
    """Spread between max and min bus voltage angle (degrees)."""
    if not r.buses:
        return 0.0
    angles = [b.Va for b in r.buses]
    return max(angles) - min(angles)


_EXTRACTORS: dict[str, callable] = {
    "generation_cost": _generation_cost,
    "total_generation_mw": _total_generation_mw,
    "total_load_mw": _total_load_mw,
    "active_losses_mw": _active_losses_mw,
    "voltage_min": _voltage_min,
    "voltage_max": _voltage_max,
    "voltage_deviation": _voltage_deviation,
    "voltage_range": _voltage_range,
    "max_line_loading_pct": _max_line_loading_pct,
    "mean_line_loading_pct": _mean_line_loading_pct,
    "violation_count": _violation_count,
    "total_reactive_gen_mvar": _total_reactive_gen_mvar,
    "online_generator_count": _online_generator_count,
    "generation_reserve_mw": _generation_reserve_mw,
    "phase_angle_range": _phase_angle_range,
}
