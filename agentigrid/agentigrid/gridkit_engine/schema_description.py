"""Description of the GridKit actions, checks and skipped tests for LLM prompts."""

from __future__ import annotations

from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.sweep_metrics import CHECKS
from agentigrid.gridkit_engine.validation import MAX_FAULTS


def command_schema_text() -> str:
    """Text for the GridKit system prompt: what the LLM may ask for."""
    checks = "\n".join(f"     - {name}: {desc}" for name, (_, _, desc) in CHECKS.items())
    skipped = "\n".join(f"   - {tid} {name}: {why}" for tid, name, why in criteria.SKIPPED_TESTS)
    default_checks = ", ".join(f'"{c}"' for c in ("angle_stability", "voltage_recovery",
                                                   "damping", "final_voltage"))
    return f"""\
Available GridKit actions (JSON format). GridKit runs phasor-dynamics (transient)
simulations. The only disturbance it has is a bus short circuit to ground
(balanced fault), applied and then cleared.

1. fault_screen — apply one bus fault per run and judge each run with named checks
   Locations, exactly one of:
     buses (list of int)     explicit bus numbers
     all_buses (true)        every bus in the case
     poi (int), hops (int)   POI bus and every bus up to hops branches away
                             (PJM: at least the POI and one bus away -> hops = 1)
   Optional (defaults are PJM-based; only change them if the goal says so):
     start_s ({criteria.FAULT_START_S})            fault start time
     clearing_cycles ({criteria.NORMAL_CLEARING_CYCLES})       nominal clearing time, cycles
     margin_cycles ({criteria.MARGIN_PRIMARY_CYCLES})        PJM clearing margin, cycles
     tmax_s ({criteria.SIM_LENGTH_S})            simulation length, s (damping needs >= {criteria.DAMPING_MIN_RUN_S})
     R, X ({criteria.FAULT_R_PU}, {criteria.FAULT_X_PU})         fault impedance, pu
     checks (default [{default_checks}]):
{checks}
   At most {MAX_FAULTS} bus faults per screen.
   Example: {{"action": "fault_screen", "poi": 29, "hops": 1}}
   Example: {{"action": "fault_screen", "all_buses": true}}

2. complete — finish with the answer
   "summary" must be the VERDICT line of the screen; "details" must list every
   failed fault and every skipped PJM test.

PJM tests that CANNOT be run (GridKit lacks the element). Never claim them as
passed; list them as SKIPPED in the answer:
{skipped}
"""
