"""Description of the HIL actions for the LLM prompt."""

from __future__ import annotations


def command_schema_text(tol_pct: float, tol_deg: float) -> str:
    return f"""\
Available HIL actions (JSON format). The HIL runs the network in real time; the
only operation is closing a tie switch while the simulation runs.

1. close_tie — start the simulation radial (all ties open), read every bus,
   close one tie live, read every bus again, and compare
   Required:
     tie (two bus numbers)   the tie to close, e.g. [12, 22] (order does not matter)
   Optional:
     start_from ("radial")   what the HIL readings are compared with:
       "radial"                no ExaGO comparison; report the HIL readings and changes
       "latest_steady_state"   the most recent ExaGO steady-state solution (last
                               simulation of the newest ExaGO journal). Its network must
                               be the radial network with this tie closed; the radial
                               start is compared with ExaGO's radial solution of the same
                               session. Agreement limit: {tol_pct} % on magnitude,
                               {tol_deg} deg on angle, at every bus.
     description (str)       short label
   Example: {{"action": "close_tie", "tie": [12, 22], "start_from": "latest_steady_state"}}

2. complete — finish with the answer
   "summary" must be the VERDICT line of the study; "details" must give the tie, the
   largest voltage changes, the largest ExaGO-HIL differences (when compared) and
   anything SKIPPED with its reason.

NOT available (refused if asked): opening a tie or any other switch while the
simulation runs, switching the feeder breaker, switching a line that is not a tie.
"""
