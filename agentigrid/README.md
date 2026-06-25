# AgentiGrid

LLM-driven iterative simulation and analysis tool for the [ExaGO](https://github.com/ornl/ExaGO) power grid optimization toolkit.

AgentiGrid uses large language models to iteratively modify power grid simulation inputs, run ExaGO solvers, interpret results, and search for configurations that satisfy user-defined goals expressed in natural language.

## Quick Start

```bash
# Install
pip install -e .

# Run with a simple goal
agentigrid ./data/case_ACTIVSg200.m \
  "Find the maximum uniform load scaling factor before the system becomes infeasible"

# Dry run (validate config without executing)
agentigrid ./data/case_ACTIVSg200.m "test" --dry-run
```

## How It Works

AgentiGrid runs an iterative agent loop:

1. **Parse** the MATPOWER base case network (.m file)
2. **Run** a baseline simulation with ExaGO (OPFLOW, DCOPFLOW, or other supported application)
3. **Prompt** the LLM with the goal, network summary, and simulation results
4. **LLM decides** an action:
   - **modify** — apply network changes (load scaling, generator dispatch, branch status, etc.) and run a new simulation
   - **sweep** — test the SAME mutation at every candidate bus (or a subset) in ONE action, solved in parallel
   - **analyze** — request specific data (voltage profiles, line loading, etc.)
   - **complete** — report findings and terminate
5. **Repeat** steps 3-4 until the goal is achieved, determined infeasible, or max iterations reached

The search journal tracks every iteration, providing the LLM with a history of what has been tried and the results observed.

### Boundary (Hosting-Capacity) Sweep

For "how much load/generation can each bus host?" or "what is the maximum MW at bus X?" goals, the `sweep` action supports a **boundary mode** (`"mode": "boundary"`). Instead of testing one fixed injection per bus, the engine runs a **per-bus bisection on injection magnitude** (in parallel) and returns the maximum feasible MW per bus together with the binding constraint (thermal / voltage, with the specific line or bus when available). This is **one LLM turn** performing N internal bisections — it does not consume N iterations of the LLM budget.

Two entity modes:

- **`entity: "load"`** — adds active load and scales reactive load along a **constant-power-factor ray** (`ΔQ = ΔP · tan φ`). `power_factor` is `"system_average"` (default; `ΣQd/ΣPd` of the base case), `"unity"`, or a number in `0..1`. `Pd` and `Qd` always move together.
- **`entity: "generator"`** — adds a generator in **fixed-injection** mode (`Pmin = Pmax = ΔP`); reactive output is left free within `±boundary_gen_q_frac · ΔP`. Fixed injection is mandatory — a dispatchable unit is zeroed by the OPF, making the hosting test vacuous.

The boundary located is the **OPFLOW convergence boundary** (the V-band and Rate A are in-solve hard constraints). Non-convergence is treated as the infeasible signal that caps the bisection; a probe that fails for numerical rather than physical reasons would understate capacity.

Relevant config (`search.*`): `boundary_initial_mw` (50), `boundary_max_mw` (2000, the bracketing cap), `boundary_tol_mw` (1.0), `boundary_max_probes` (24), `boundary_gen_q_frac` (0.4), `boundary_power_factor_default` (`system_average`). The per-candidate result table is journaled in full and rendered in the PDF/Streamlit hosting-capacity table; the LLM-facing text is token-bounded (top-N by capacity) above `search.sweep_full_table_threshold`.

### Generator Mode: Fixed Injection vs Dispatchable

Adding a generator supports two modes, chosen from the goal's wording (and named back in the answer — never defaulted silently, since the wrong mode flips the result):

- **Fixed injection** (`Pmin = Pmax = cap`) — the unit's output is pinned. Use for "hosting capacity / how much can connect / forced output" questions. Cost is irrelevant (the output cannot move).
- **Dispatchable** (`Pmin = 0, Pmax = cap`, `entity_dispatchable: true`) — the OPF *chooses* the unit's output under economic dispatch, so the unit needs a realistic **cost curve**. Use for "minimum-cost location for a generator" questions. The cost curve defaults to the **case median** (mid-merit, so the unit partially dispatches and locations differentiate) per `search.added_gen_cost_strategy`; pass `entity_cost_coeffs: [c2, c1, c0]` to override. Each location's dispatched Pg is recorded (a unit dispatching ~0 MW is not helping there).

### Custom Sweep Metric / Predicate (named registry)

Beyond the default cost metric and standard V-band/loading feasibility, a sweep can select a **named, verified primitive** (`agentigrid/engine/sweep_metrics.py`) — not free-form code. Selection is by name; the registry is extensible (`register_metric` / `register_predicate`) for later capabilities.

- **`metric: "max_delta_v"`** — ranks buses by the worst *system-wide* voltage step `max_b |V_candidate[b] − V_base[b]|` caused by switching in a load block (`search.switched_load_mw`) at the candidate bus. A power-quality flag; the largest step need not be at the switched bus. Requires (and triggers) a one-time base-case solve for the reference voltages.
- **`feasibility_predicate: "reactive_adequacy"`** — replaces standard feasibility with a reactive-headroom test: a feasible OPF must exist with the added unit forced to `(P = Pmax, Q = Qmax)` (the sweep pins `Qmin = Qmax`). Returns the limiting quantity when it fails.

## Search Modes

- **Boundary finding** — "Find the maximum load scaling factor before infeasibility"
- **Scenario exploration** — "What happens if generator at bus 189 trips offline?"
- **Optimization** — "Minimize generation cost while keeping all voltages above 0.95 pu"
- **Multi-objective** — "Minimize cost while keeping voltages above 0.95 pu and line loadings below 85%"
- **Stress testing** — "Find the most critical N-1 contingencies by systematically testing line outages"
- **Analysis** — "Report the top 5 most congested transmission lines"

## Supported Applications

| Application | Description | Status |
|-------------|-------------|--------|
| **OPFLOW** | AC Optimal Power Flow — full nonlinear OPF with voltage magnitudes, reactive power, and cost optimization | ✅ Fully supported |
| **DCOPFLOW** | DC Optimal Power Flow — linearized approximation using phase angles and active power only. Faster than OPFLOW, useful for screening and contingency ranking | ✅ Fully supported |
| **SCOPFLOW** | Security-Constrained OPF — finds a preventive dispatch that survives all contingencies in a `.cont` file. Requires a contingency file | ✅ Fully supported |
| **TCOPFLOW** | Multi-Period OPF — time-coupled optimization with generator ramp constraints and load profiles. Requires P and Q load profile CSV files | ✅ Fully supported |
| **SOPFLOW** | Stochastic OPF — two-stage optimization with wind generation scenarios. Requires a scenario CSV file and a network file with wind generators | ✅ Fully supported |
| **PFLOW** | Power Flow — analysis, not optimization. The LLM performs the search directly using voltage setpoints, tap ratios, shunts, and dispatch adjustments | ✅ Fully supported |

### DCOPFLOW vs OPFLOW

DCOPFLOW uses the DC power flow approximation:
- All bus voltages are fixed at 1.0 pu — voltage magnitude is not an optimization variable
- Reactive power (Q) is ignored — only active power (P) is optimized
- Simulations run significantly faster (typically 10-50x) than full AC OPF
- Voltage-related commands (`set_gen_voltage`, `set_bus_vlimits`, `set_all_bus_vlimits`) are automatically skipped with a warning
- Best suited for: fast screening, load scaling studies, contingency ranking, active power market analysis

Select the application via CLI (`--app dcopflow`) or in the launcher GUI dropdown.

### SCOPFLOW (Security-Constrained OPF)

SCOPFLOW optimizes the base case dispatch so that the network remains feasible even if any contingency in the contingency file occurs:
- Requires a `.cont` contingency file listing branch and generator outages
- The cost is typically higher than unconstrained OPFLOW — this "security premium" is the price of reliability
- Results show the **base case** operating point (the preventive dispatch), not individual contingency outcomes
- All OPFLOW commands work with SCOPFLOW (voltage control, load scaling, generator dispatch, etc.)
- Branch status commands (`set_branch_status`) permanently modify the topology — they do NOT simulate contingencies (the `.cont` file handles that)

Select via CLI (`--app scopflow --ctgc data/case_ACTIVSg200.cont`) or in the launcher GUI (application dropdown + contingency file selector).

### TCOPFLOW (Multi-Period OPF)

TCOPFLOW solves a multi-period AC optimal power flow over a time horizon with generator ramp constraints between successive periods:
- Requires **load profile CSV files** — active power (`*_load_P.csv`) and reactive power (`*_load_Q.csv`) — that define per-bus per-period demand
- The objective is **total cost across all time periods**, not a single snapshot
- Generator ramp coupling (`--tcopflow-iscoupling`) enforces that output changes between periods stay within ramp limits
- Standard load commands (`scale_all_loads`, `set_load`) modify the `.m` file but TCOPFLOW reads per-period loads from CSV profiles — use `scale_load_profile` to adjust demand instead
- Network topology commands (`set_gen_status`, `set_branch_status`, `set_all_bus_vlimits`, etc.) apply across all periods
- Results show **aggregated metrics** across all periods (worst voltage, peak load, worst line loading) plus a per-period summary table
- Only the IPOPT solver is supported
- The launcher auto-selects profile files matching the base case name (e.g., `case9mod.m` → `case9_load_P.csv`)

Load profile files follow the naming convention `<casename>_load_P.csv` / `<casename>_load_Q.csv` (see `data/README.md`). Select via CLI (`--app tcopflow --pload-profile data/case9_load_P.csv --qload-profile data/case9_load_Q.csv`) or in the launcher GUI (application dropdown + auto-matched profile selectors + temporal parameters).

### SOPFLOW (Stochastic OPF)

SOPFLOW solves a two-stage stochastic optimization: a first-stage (here-and-now) dispatch that must satisfy constraints across all wind scenarios simultaneously, plus per-scenario second-stage corrections:
- Requires a **wind scenario CSV file** (via `--scenario-file`) with columns for each wind generator — standard load commands modify the `.m` file but do NOT change wind scenario data
- Use `scale_wind_scenario` (command 13) to adjust wind penetration in the scenario CSV — e.g., `{"action": "scale_wind_scenario", "factor": 0.8}` reduces wind output by 20% across all scenarios
- The network file must have wind generators defined (`gentype='W2'`, `genfuel='wind'`) — use a case like `case9mod_gen3_wind.m`
- Two scenario file formats are supported: single-period (`scenario_nr, <wind_cols>, weight`) and multi-period (`sim_timestamp, scenario_nr, <wind_cols>`)
- Results show the **first-stage base-case dispatch** — the operating point that the system commits to before knowing which wind scenario materialises
- SOPFLOW supports both **IPOPT** (single-core, default) and **EMPAR** (multi-core via MPI) solvers
- The launcher auto-selects scenario files matching the base case name (e.g., `case9mod_gen3_wind.m` → `case9_scenarios.csv`)

Scenario files follow the naming convention `<casename>_scenarios.csv` / `<casename>_10_scenarios.csv` (see `data/README.md`). Select via CLI (`--app sopflow --scenario-file data/case9_10_scenarios.csv`) or in the launcher GUI (application dropdown + auto-matched scenario selector + solver/coupling options).

### PFLOW (Power Flow — Analysis, Not Optimization)

PFLOW solves the nonlinear power flow equations for a given network state — it does **not** optimize. The LLM performs the search directly: proposing dispatch changes, voltage setpoints, tap positions, and shunt adjustments, then evaluating feasibility from PFLOW's results.

Key differences from OPFLOW and other optimization applications:
- **No objective function** — `objective_value` is always 0.0. Generation cost is computed from the dispatch × cost curves and shown as "Computed generation cost" in results, but the solver does not minimize it.
- **`set_gen_voltage` directly constrains bus voltage** — in OPFLOW, Vg is an initial guess that the solver overrides; in PFLOW, the solver enforces the setpoint as a hard constraint. This is the primary voltage control tool.
- **`set_gen_dispatch` directly sets generator output** — no re-dispatch by the solver.
- **Three new commands** — `set_tap_ratio` (transformer tap positions), `set_shunt_susceptance` (reactive support at buses), `set_phase_shift_angle` (power flow control through phase shifters).
- **Newton-Rhapson solver** — convergence reported as `CONVERGED` / `DID NOT CONVERGE` (not IPOPT).
- **Search heuristics in the system prompt** — binary search for feasibility boundaries, gradient-like dispatch adjustment for cost reduction, iterative voltage tuning.

PFLOW is available from the CLI (`--app pflow`) and in the launcher GUI application dropdown. No additional files (contingency, profile, or scenario) are required — only the base case `.m` file.

### Concurrent PFLOW (Explore/Select)

When `--concurrent-pflow` is enabled, the LLM can propose multiple simulation variants per iteration and run them concurrently. This replaces sequential binary search with parallel coordinate search, reducing the number of LLM round-trips needed to converge.

**Why it matters:** In sequential search, each iteration takes one LLM round-trip (~20-30s) but only ~0.02s of simulation time. The LLM is the bottleneck, not the simulation. Concurrent explore lets the LLM evaluate 3-8 configurations per round-trip, converging on solutions in half the wall-clock time.

**How it works:**
1. The LLM proposes an **explore** action with 2–8 variant command sets (e.g., different load scaling factors)
2. The system runs all simulations concurrently via ThreadPoolExecutor
3. Results are presented with Pareto front analysis (★ marks non-dominated variants)
4. The LLM **selects** one variant as the new current point
5. Repeat: explore → select → explore → ...

**CLI usage:**
```bash
agentigrid ./data/case_ACTIVSg200.m \
  "Find the maximum load scaling factor" \
  --app pflow --concurrent-pflow --max-variants 5
```

**Launcher GUI:** Enable the "Concurrent explore/select" checkbox in the sidebar (PFLOW only) and set "Max variants per explore" (2-16, default 8).

The system prompt dynamically restructures when concurrent mode is on: explore is presented as action #1 (primary), with sequential search heuristics replaced by parallel search guidance. This ensures the LLM uses explore as its default search mechanism rather than falling back to sequential modify actions.

**Pre-execution rejection:** A variant whose every command would be a no-op against the base network (for example, `set_gen_dispatch` against the slack bus) is marked `rejected` and not simulated. The variant still appears in the explore table — labelled `REJECTED` — so the LLM sees that the proposal was wasted, but no PFLOW subprocess is launched and rejected variants are excluded from the Pareto front.

**Network metadata in the system prompt (Section G):** At session start, structural facts about the case are computed once and injected into the LLM's system prompt: slack/reference bus(es), must-run generators (`Pmin == Pmax`), offline generators, and a summary of cost-curve diversity. If all online generators share identical quadratic cost coefficients, an explicit warning is included so the LLM does not waste iterations attempting redispatch-based cost reduction. This section is computed once from the base network and remains stable across iterations.

**Cost reporting on select (PFLOW):** Because PFLOW does not produce an objective value, generation cost for a selected variant is computed from the dispatch and `mpc.gencost` polynomial coefficients. The journal entry's `objective_value` and `tracked_metrics["generation_cost"]` reflect that computed value rather than the placeholder `0.0`.

**Informative variant descriptions:** Each variant's `description` field is auto-generated from its command list when the LLM does not provide one. The description uses compact abbreviations (e.g. `"scale×1.23, vlim[0.95-1.05], dispatch bus135→250MW"`) and marks skipped commands inline with `[SKIP]` plus a parenthetical skip count at the end. This replaces the blank single-letter labels that previously appeared in the explore results table and the journal.

**Identical-cost sibling detection:** After each explore batch, feasible variants sharing the same rounded cost are identified. The more-complex variants (more commands) get an annotation appended to their description (`"← same cost as A; extra commands had no effect"`) and their `cost_equivalent_to` field is set. A batch-level warning is also injected into the next LLM prompt, so the agent knows to avoid repeating the no-op commands.

**Session-best tracking:** The journal maintains a `session_best` record (cost, iteration, variant label, commands) across all variants ever run — not just the selected ones. This is surfaced in the user prompt before every explore call so the LLM can detect regression: if its cheapest batch variant is more expensive than the session best, it knows its current direction is going backward. The record is persisted as a top-level `session_best` key in the journal JSON. The launcher GUI shows the session-best cost in the PFLOW metrics panel.

## Multi-Objective Tracking

AgentiGrid can track multiple objectives simultaneously and reason about tradeoffs between them. Objectives can be introduced in three ways:

- **From the initial goal** — the LLM extracts objectives automatically (e.g., "minimize cost while keeping voltages above 0.95" registers cost as primary and voltage as a constraint)
- **Via steering** — inject a directive mid-search like "also track line loading" to add a secondary objective
- **LLM-proposed** — the agent itself can propose tracking a new metric when it notices a tension (e.g., cost decreasing but voltage stability degrading)

Tracked objectives are shown in a multi-objective trend chart in the GUI and included in PDF reports. The LLM receives a structured summary of how all tracked metrics evolve across iterations, enabling it to articulate tradeoffs and make informed decisions. At the end of a search, the post-search analysis identifies the key tradeoffs and can recommend multiple solutions representing different points on the tradeoff space.

The system includes 14 built-in metric extractors (generation cost, voltage deviation, line loading, active losses, generation reserve, and more). For simple single-objective goals, this infrastructure is transparent — everything works exactly as before.

## Stress Test Mode

AgentiGrid includes a dedicated stress test mode for adversarial contingency exploration. When activated, the LLM acts as a security analyst, systematically disabling network components to identify critical vulnerabilities.

```bash
# CLI
agentigrid ./data/case_ACTIVSg200.m \
  "Find the most critical N-1 contingencies" \
  --search-mode stress_test
```

In stress test mode, the LLM always uses fresh mode (each contingency tested independently from the base case), starts with the most loaded lines, and can escalate to N-2 combinations. The post-search report ranks contingencies by severity: infeasibility > voltage violations > high line loading > cost increase.

## Session Save/Resume

Searches can be saved to disk and resumed later — useful for long runs, interrupted sessions, or exploring different strategies from the same checkpoint.

### CLI

```bash
# During a running search, type 'save' in the terminal:
save
# Output: [Steering] Session saved to: workdir/saved_session_20260414_150000

# Resume later:
agentigrid --resume workdir/saved_session_20260414_150000 --config configs/local_config.yaml
```

When resuming, the goal and journal are loaded from the saved session, but the LLM backend and config settings come from the current config/CLI arguments — so you can resume with a different model or temperature.

### GUI

The launcher sidebar includes a "Session Save/Resume" section with a Save button (available after search completes or while running) and a dropdown to resume from previously saved sessions.

## Interactive Steering

While a search is running, you can inject steering directives from the terminal (CLI) or the GUI — without stopping and restarting the search.

### CLI steering commands

When running interactively (stdin is a TTY), a background listener accepts these commands:

| Input | Action |
|-------|--------|
| `<text>` | Inject an **augment** directive — the LLM considers it alongside the original goal |
| `replace: <text>` | Inject a **replace** directive — the LLM treats it as a new primary goal |
| `pause` | Pause the search at the next iteration boundary |
| `resume` | Resume a paused search |
| `stop` | Request graceful termination |
| `status` | Print current pause state and the last 3 injected directives |
| `save` | Save the current session state to disk for later resumption |

**Augment vs. replace semantics:**
- **Augment** — adds a constraint or preference to the current goal without discarding it. Example: `Focus on buses in area 3`.
- **Replace** — supersedes the current goal entirely. Example: `replace: Minimize voltage violations, ignore cost`.

Entering a directive while paused automatically resumes the search.

### GUI steering panel

The Streamlit launcher exposes the same capabilities via a steering panel in the live monitor. See [launcher/README.md](launcher/README.md) for details.

---

## Usage

### Shell script (recommended for interactive use)

`run_agentigrid.sh` is the easiest way to start a session. It prompts you to type
the goal interactively, prints a confirmation header, and then launches the
simulation.

```bash
# Use all defaults (configs/local_config.yaml, case_ACTIVSg200.m, 20 iterations)
./run_agentigrid.sh

# Override the config file only
./run_agentigrid.sh configs/my_config.yaml

# Override config and case file
./run_agentigrid.sh configs/my_config.yaml ./data/case_RTS.m

# Override all three (config, case file, max iterations)
./run_agentigrid.sh configs/my_config.yaml ./data/case_RTS.m 10
```

When run, the script will ask:
```
Enter simulation prompt: Find the maximum load scaling factor before infeasibility
```

Then print a summary before executing:
```
============================================================
  AgentiGrid Run
============================================================
  Config:    configs/local_config.yaml
  Case file: ./data/case_ACTIVSg200.m
  Max iter:  20
  Prompt:    Find the maximum load scaling factor before infeasibility
============================================================
```

The three positional arguments correspond to the three most commonly varied
settings. Everything else (backend, model, application, verbosity) is
controlled by the config file.

### Direct CLI

```bash
# Basic run
agentigrid ./data/case_ACTIVSg200.m "Find the maximum load scaling factor"

# With options
agentigrid ./data/case_ACTIVSg200.m "Minimize generation cost" \
  --backend anthropic --model claude-sonnet-4-20250514 \
  --app opflow --max-iter 30 --verbose

# Quiet mode (only show final summary)
agentigrid ./data/case_ACTIVSg200.m "Analyze voltage profile" --quiet

# Stress test mode (adversarial contingency exploration)
agentigrid ./data/case_ACTIVSg200.m \
  "Find critical N-1 contingencies" --search-mode stress_test

# Resume a saved session
agentigrid --resume workdir/saved_session_20260414_150000

# DC Optimal Power Flow (fast screening)
agentigrid ./data/case_ACTIVSg200.m \
  "Find the maximum load scaling factor before infeasibility" \
  --app dcopflow --max-iter 10 --mode fresh

# Security-Constrained OPF (requires contingency file)
agentigrid ./data/case_ACTIVSg200.m \
  "Find the minimum cost dispatch that survives all N-1 contingencies" \
  --app scopflow --ctgc data/case_ACTIVSg200.cont --max-iter 10

# Multi-Period OPF (requires load profile files)
agentigrid ./data/case9mod.m \
  "Find the load scaling factor that causes infeasibility over the time horizon" \
  --app tcopflow --pload-profile data/case9_load_P.csv \
  --qload-profile data/case9_load_Q.csv --tcopflow-duration 1.0 --tcopflow-dt 15

# Stochastic OPF (requires wind scenario file and wind-enabled network)
agentigrid ./data/case9mod_gen3_wind.m \
  "Find the maximum wind penetration level before the system becomes infeasible" \
  --app sopflow --scenario-file data/case9_10_scenarios.csv

# Stochastic OPF with EMPAR solver (multi-core)
agentigrid ./data/case9mod_gen3_wind.m \
  "Find the maximum wind penetration level" \
  --app sopflow --scenario-file data/case9_10_scenarios.csv \
  --sopflow-solver EMPAR --np 4

# Power Flow (LLM-driven search, no optimization)
agentigrid ./data/case_ACTIVSg200.m \
  "Find the maximum load scaling factor before the power flow fails to converge" \
  --app pflow --max-iter 15 --mode fresh

# Dry run (validate config without executing)
python -m agentigrid ./data/case_ACTIVSg200.m "test goal" --dry-run
```

## Example Output

```
============================================================
  AgentiGrid — LLM-driven iterative simulation for ExaGO
  Version 0.1.0
============================================================
  Backend:        anthropic
  Model:          claude-sonnet-4-20250514
  Application:    opflow
  Base case:      data/case_ACTIVSg200.m
  Goal:           Find the maximum load scaling factor
  Max iterations: 20
  Mode:           accumulative
============================================================

[Iter 0] Running base case simulation...
[Iter 0] Base case: CONVERGED, cost=$27,557.57

[Iter 1] Sending prompt to anthropic (claude-sonnet-4-20250514)...
[Iter 1] LLM action: modify — "Scale all loads +20%"
[Iter 1] Applied 1 command(s), 0 skipped
[Iter 1] Simulation completed in 0.04s — CONVERGED, cost=$33,019.55

...

[Iter 5] LLM action: complete
[Iter 5] Search completed: "Maximum feasible load increase is ~27%."

============================================================
  AgentiGrid Search Complete
============================================================
  Goal:           Find the maximum load scaling factor
  Application:    opflow
  Backend:        anthropic (claude-sonnet-4-20250514)
  Iterations:     6 (of max 20)
  Duration:       18.3 seconds
  Tokens used:    ~12,450 (prompt: 9,200, completion: 3,250)
  Termination:    completed
  Best objective: $27,557.57 (iteration 1)

  Findings: Maximum feasible uniform load increase is approximately 27%.
============================================================
```

## Installation

```bash
cd /path/to/agentigrid
pip install -e .
```

Copy or symlink ExaGO binaries into `applications/` (see [applications/README.md](applications/README.md)) and place network data files in `data/` (see [data/README.md](data/README.md)).

## Configuration

Edit `configs/default_config.yaml` or pass `--config path/to/config.yaml`. CLI arguments override config file values.

Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

## Testing

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run multi-objective tracking tests
python -m pytest tests/test_multi_objective.py -v

# Run session save/resume tests
python -m pytest tests/test_session_io.py -v

# Run DCOPFLOW-specific tests
python -m pytest tests/test_dcopflow.py -v

# Run SCOPFLOW-specific tests
python -m pytest tests/test_scopflow.py -v

# Run TCOPFLOW-specific tests
python -m pytest tests/test_tcopflow.py -v

# Run SOPFLOW-specific tests
python -m pytest tests/test_sopflow.py -v

# Run PFLOW-specific tests
python -m pytest tests/test_pflow.py -v

# Run end-to-end tests (requires opflow binary)
python -m pytest tests/test_e2e.py -v -m "not slow"

# Run real LLM integration tests (requires API key + opflow)
python -m pytest tests/test_e2e.py -v
```

## Architecture

See [agentigrid_architecture.md](agentigrid_architecture.md) for the full design document.
