"""Command schema description for inclusion in LLM prompts."""

from __future__ import annotations


def command_schema_text() -> str:
    """Generate a text description of available modification commands.

    This text is included in the LLM system prompt so it knows what
    commands it can issue.

    Returns:
        Multi-line string describing all available commands.
    """
    return """\
Available modification commands (JSON format):

1. set_load — Set active/reactive load at a bus
   Required: bus (int)
   Optional: Pd (float, MW), Qd (float, MVAr)
   Example: {"action": "set_load", "bus": 10, "Pd": 50.0, "Qd": 14.0}

2. scale_load — Scale load at a bus, area, or zone
   Required: factor (float, e.g. 1.2 for +20%)
   Optional: bus (int), area (int), zone (int) — specify exactly one scope
   Example: {"action": "scale_load", "area": 4, "factor": 1.2}

3. scale_all_loads — Scale all loads in the network uniformly
   Required: factor (float)
   Example: {"action": "scale_all_loads", "factor": 1.1}

4. set_gen_status — Turn a generator on or off
   Required: bus (int), status (int: 1=on, 0=off)
   Optional: gen_id (int, 0-based index if multiple generators at bus)
   Example: {"action": "set_gen_status", "bus": 189, "status": 0}

5. set_gen_dispatch — Set generator active power output
   Required: bus (int), Pg (float, MW — must be within [Pmin, Pmax])
   Optional: gen_id (int)
   Example: {"action": "set_gen_dispatch", "bus": 189, "Pg": 400.0}

6. set_gen_voltage — Set generator voltage setpoint
    Required: bus (int), Vg (float, pu — reasonable range 0.8–1.2)
    Optional: gen_id (int)
    Example: {"action": "set_gen_voltage", "bus": 189, "Vg": 1.05}
    WARNING: In OPFLOW/SCOPFLOW/TCOPFLOW/SOPFLOW, bus voltages are optimization
    decision variables — the solver will override this setpoint with its own optimal
    value. To actually constrain bus voltages in OPF, use set_bus_vlimits (command 10)
    to set Vmin/Vmax on the bus. In PFLOW, this command directly constrains bus
    voltage — the solver uses Vg as a fixed setpoint, making it the primary voltage
    control mechanism.

7. set_branch_status — Enable or disable a branch
   Required: fbus (int), tbus (int), status (int: 1=in-service, 0=out-of-service)
   Optional: ckt (int, 0-based circuit index for parallel lines)
   Example: {"action": "set_branch_status", "fbus": 2, "tbus": 1, "status": 0}

8. set_branch_rate — Modify branch thermal rating
   Required: fbus (int), tbus (int), rateA (float, MVA, >= 0)
   Optional: ckt (int)
   Example: {"action": "set_branch_rate", "fbus": 2, "tbus": 1, "rateA": 200.0}

9. set_cost_coeffs — Modify generator cost curve coefficients
   Required: bus (int), coeffs (list of float)
   Optional: gen_id (int)
   Example: {"action": "set_cost_coeffs", "bus": 189, "coeffs": [0.003, 20.0, 500.0]}

10. set_bus_vlimits — Set voltage limits on a single bus
    Required: bus (int)
    Optional: Vmin (float, pu), Vmax (float, pu)
    Example: {"action": "set_bus_vlimits", "bus": 10, "Vmin": 0.95, "Vmax": 1.05}

    11. set_all_bus_vlimits — Set voltage limits on ALL buses at once
     Optional: Vmin (float, pu), Vmax (float, pu) — at least one required
     Example: {"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05}
     This is the preferred way to enforce system-wide voltage constraints in
     OPFLOW. Much more efficient than issuing set_bus_vlimits for each bus.

  12. scale_load_profile — Scale per-period load profile values (TCOPFLOW only)
      Required: factor (float, e.g. 1.2 for +20%)
      Example: {"action": "scale_load_profile", "factor": 1.2}
      Multiplies all numeric values in both P and Q load profile CSV files
      by the given factor. This is the correct way to adjust demand for
      TCOPFLOW, since TCOPFLOW reads per-period loads from profile files,
      not the .m case file. Standard load commands (scale_all_loads, set_load)
      modify the .m file but TCOPFLOW overrides these with profile data.

  13. scale_wind_scenario — Scale wind generation values in scenario CSV (SOPFLOW only)
      Required: factor (float, e.g. 1.5 for +50%)
      Example: {"action": "scale_wind_scenario", "factor": 1.5}
      Multiplies all numeric wind generation columns in the scenario CSV file
      by the given factor, preserving non-numeric columns (scenario_nr,
      sim_timestamp, weight). This is the correct way to adjust wind generation
      variability in SOPFLOW, since SOPFLOW reads wind scenarios from the CSV
      file, not the .m case file.

  14. set_tap_ratio — Set transformer tap ratio
      Required: fbus (int), tbus (int), ratio (float, typical range 0.9–1.1)
      Optional: ckt (int, 0-based circuit index for parallel branches)
      Example: {"action": "set_tap_ratio", "fbus": 45, "tbus": 55, "ratio": 1.05}
      Only applies to branches that are transformers (ratio ≠ 0 in the base
      network). Setting ratio on a non-transformer line (ratio=0) is rejected.

  15. set_shunt_susceptance — Modify shunt susceptance at a bus
      Required: bus (int), Bs (float, MVA at rated voltage — positive = capacitive)
      Example: {"action": "set_shunt_susceptance", "bus": 10, "Bs": 0.5}
      Positive Bs adds capacitive susceptance (raises voltage), negative Bs adds
      inductive susceptance (lowers voltage). Particularly useful for PFLOW voltage
      control.

  16. set_phase_shift_angle — Set phase shifter angle
      Required: fbus (int), tbus (int), angle (float, degrees — typical range -60 to 60)
      Optional: ckt (int, 0-based circuit index for parallel branches)
      Example: {"action": "set_phase_shift_angle", "fbus": 121, "tbus": 187, "angle": 5.0}
      Only applies to branches that are phase shifters (angle ≠ 0 in the base
      network). Setting angle on a non-phase-shifter branch (angle=0) is rejected.

Return your commands as a JSON object with a "commands" key containing a list:
{"commands": [{"action": "...", ...}, {"action": "...", ...}]}
"""
