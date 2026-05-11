# AgentiGrid — Architecture Document

**LLM-Driven Iterative Simulation and Analysis for ExaGO**

| Field | Value |
|-------|-------|
| Project | AgentiGrid (in process of integration into ExaGO) |
| Version | 0.3 — PFLOW Direct Optimization |
| Date | May 2026 |
| Author | Samim K., University of Sarajevo |
| Supervisor | Slaven P., Oak Ridge National Laboratory |
| Target Platform | Linux (OpenSUSE Leap 15.6), Python 3.10+ |

---

## 1. Introduction and Motivation

AgentiGrid is a project that brings large language model (LLM) intelligence into the simulation loop of power grid analysis. Rather than using an LLM merely as a conversational interface for setting up a single simulation run, AgentiGrid positions the LLM as an iterative decision-making agent that proposes parameter modifications, triggers simulations via ExaGO, interprets results, and decides subsequent actions — repeating this cycle until a user-defined goal is achieved.

This approach is motivated by a class of power system analysis tasks that are difficult to express as formal optimization problems but natural to describe in plain language: "find the load level at which the network becomes infeasible," "identify the weakest corridor under stress," or "find a dispatch that balances cost and voltage quality." A human expert would approach these tasks through iterative exploration — running a simulation, examining results, forming hypotheses, adjusting parameters, and repeating. AgentiGrid automates this expert workflow.

AgentiGrid was initially developed as a self-contained codebase and is currently in the process of being integrated into the ExaGO project, a large-scale power grid optimization toolkit developed at Oak Ridge National Laboratory and designed for execution on exascale computers such as Frontier. At this stage the AgentiGrid sources reside as a sibling subproject inside the ExaGO source tree; deeper integration with the ExaGO build system and runtime is ongoing work.

---

## 2. Design Goals and Principles

- **LLM-in-the-loop (Option B architecture):** The LLM is the decision-maker in the search loop — it proposes parameter changes, interprets simulation results, and decides the next action. No external optimization algorithm is interposed.
- **Application-agnostic design:** While the PoC targets OPFLOW, the architecture treats the inner ExaGO application as a pluggable component. Supporting additional applications (SCOPFLOW, TCOPFLOW, SOPFLOW, DCOPFLOW, PFLOW) requires only defining their parameter schemas and output parsers.
- **Base case + deltas model:** A user-provided base case file (.m and optionally .gic) is immutable throughout the search. Each iteration creates a working copy and applies the LLM's modifications as structured JSON commands.
- **Multi-backend LLM support:** The system supports OpenAI API, Anthropic API, and Ollama (including Ollama Cloud) through a unified abstraction layer.
- **Iterative and bounded search:** Every search session has a configurable maximum iteration count (default: 20). The LLM can terminate early by declaring the goal achieved.
- **Interactive steering:** Users can inject directives into a running search (augment or replace mode) and pause/resume at iteration boundaries without restarting.
- **Goal-type-aware reporting:** Post-search analysis classifies the search goal (cost minimization, feasibility boundary, constraint satisfaction, parameter exploration) and selects the best iteration accordingly, rather than always defaulting to lowest cost.
- **Application-aware commands:** The system warns when commands are ineffective for the current application (e.g., `set_gen_voltage` under OPFLOW) and guides the LLM toward effective alternatives.
- **Designed for future multi-objective reasoning:** The system anticipates qualitative multi-objective tasks where the LLM balances competing goals.

---

## 3. System Architecture Overview

The system is organized into three layers:

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| Outer | Agent Loop Controller | Manages the iterative cycle: assembles LLM prompts, sends to LLM backend, parses responses, decides continue/stop |
| Middle | Simulation Orchestrator | Applies modification commands to working copy, invokes ExaGO, captures output, parses results |
| Inner | ExaGO Application | The actual simulation engine (OPFLOW, SCOPFLOW, etc.) |

### Data flow for one iteration:

1. Agent Loop Controller assembles prompt (goal + search journal + latest results + steering directives)
2. LLM backend generates response with structured action (modify / complete / analyze)
3. Controller parses response; if action is "modify", extracts JSON commands
4. Modification Engine validates and applies commands to a working copy of the base case; application-specific warnings are generated (e.g., `set_gen_voltage` under OPFLOW) and fed back to the LLM
5. Simulation Executor invokes ExaGO with the modified input files
6. Results Parser extracts structured data from ExaGO output and checks violations against actual bus voltage limits from the modified network
7. Controller updates the Search Journal with iteration summary
8. Loop returns to step 1 (or terminates if action is "complete" or max iterations reached)

### Post-search finalization:

9. A final LLM call classifies the search goal type and identifies the best iteration
10. Summary statistics, charts, and PDF report are generated with goal-type-aware framing

---

## 4. Component Design

### 4.1 LLM Backend Abstraction Layer

```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.3) -> str: ...
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def supports_json_mode(self) -> bool: ...
```

| Provider | Config Key | Recommended Models | Notes |
|----------|------------|-------------------|-------|
| OpenAI | openai | GPT-4o, GPT-4o-mini | JSON mode supported |
| Anthropic | anthropic | Claude Sonnet 4.6 | JSON via system prompt; excellent JSON reliability |
| Ollama (local) | ollama | Qwen 2.5 7B/14B, Llama 3.x | JSON mode varies |
| Ollama Cloud | ollama-cloud | GLM-5 | ~60% JSON reliability; GLM-5.1 incompatible |

Configuration via YAML file or environment variables. Temperature kept low (0.2–0.4).

**Model compatibility notes:** GLM-5 via Ollama Cloud works but has ~40% JSON parse failure rate, wasting iterations. GLM-5.1 is currently incompatible (fails to produce valid JSON actions). Claude Sonnet 4.6 via Anthropic API offers near-perfect JSON reliability at ~$0.70–0.80 per run on the ACTIVSg200 network.

### 4.2 Simulation Executor

Two execution modes:

| Mode | Description | Advantage |
|------|-------------|-----------|
| CLI subprocess | Invokes ExaGO binaries as subprocesses | Simpler setup; works without Python bindings |
| Python API | Uses ExaGO Python bindings | Direct access to solution objects |

The PoC implements CLI subprocess mode first. Each ExaGO application is registered with:

| Application | Binary | Inputs | Key Outputs |
|-------------|--------|--------|-------------|
| OPFLOW | opflow | .m [.gic] | Bus voltages, gen dispatch, line flows, cost, convergence |
| DCOPFLOW | dcopflow | .m | DC angles, active power flows, cost (same output format as OPFLOW; voltages fixed at 1.0 pu) |
| SCOPFLOW | scopflow | .m + .cont [-ctgcfile] | Base case preventive dispatch; per-contingency results via save (same OPFLOW table format for base case; contingency count in header) |
| TCOPFLOW | tcopflow | .m + load profiles [-save_output] | Multi-period dispatch with ramp coupling; per-period results in tcopflowout/; IPOPT only |
| SOPFLOW | sopflow | .m + scenarios [-windgen] [-sopflow_Ns] [-sopflow_solver IPOPT/EMPAR] [-sopflow_iscoupling] | Stochastic base-case dispatch with wind scenarios; convergence override from header metadata; two-stage feasibility classification |
| PFLOW | pflow | .m | Power flow solution (no optimization); Newton-Rhapson solver; LLM drives the search directly |

### 4.3 Results Parser

Produces two outputs:
- **Full structured result:** Python dictionary with all extracted data. Stored for detailed queries.
- **Compact summary:** 15–30 lines of text for LLM prompt inclusion. Contains: objective value, feasibility status, violation count/severity, voltage range, top-5 loaded lines, total generation vs. load.

**Application-aware dispatch:** The parser layer includes dispatch functions (`parse_simulation_result_for_app`, `results_summary_for_app`) that route to the correct parser and summary generator based on the application name. DCOPFLOW reuses the OPFLOW parser (identical output format) but has its own summary generator (`dcopflow_results_summary`) that shows phase angle profiles instead of voltage magnitudes and omits reactive power data. Unknown applications fall back to the OPFLOW parser with a logged warning.

SCOPFLOW has its own parser (`scopflow_parser.py`) because its output header differs from OPFLOW: it prints "Security-Constrained Optimal Power Flow" instead of "Optimal Power Flow", includes contingency metadata (number of contingencies, multi-period flag), and uses "Objective value (base)" instead of "Objective value". The SCOPFLOW parser extracts this metadata, then delegates bus/branch/generator table parsing to the OPFLOW parser (identical table format). The OPFLOW parser's objective value regex was updated to handle the optional "(base)" suffix, ensuring both formats are parsed by a single regex.

TCOPFLOW has its own parser (`tcopflow_parser.py`) because its output header differs from OPFLOW: it prints "Multi-Period Optimal Power Flow" and includes temporal metadata (duration, time-step size, number of steps, load profile paths, coupling constraint count). The TCOPFLOW parser extracts this metadata, delegates period-0 table parsing to the OPFLOW parser (identical table format), and handles TCOPFLOW-specific feasibility logic (IPOPT-only, with power balance checking). A second function (`parse_tcopflow_period_files()`) parses the per-period `.m` files generated by `-save_output` in the `tcopflowout/` directory, extracting per-period voltage, load, generation, and line loading data using the MATPOWER parser. The TCOPFLOW summary generator (`tcopflow_summary.py`) produces aggregated metrics across all periods plus a compact per-period mini-table, enabling the LLM to reason about temporal aspects of the solution.

SOPFLOW has its own parser (`sopflow_parser.py`) because its output header differs from OPFLOW: it prints "Stochastic Optimal Power Flow" and includes stochastic metadata (number of scenarios, multi-contingency flag, coupling constraint count, solver type). The SOPFLOW parser extracts this metadata, delegates bus/branch/generator table parsing to the OPFLOW parser (identical table format), and handles SOPFLOW-specific feasibility logic. Key differences: (1) SOPFLOW may not produce an IPOPT EXIT message in the output, so the parser overrides `converged` based on the header's `convergence_status` field; (2) the "Objective value (base)" regex handles the optional `(base)` suffix; (3) wind generator rows have `fuel="WIND"` in the generator table; (4) both IPOPT (single-core) and EMPAR (multi-core via MPI) solvers are supported. The SOPFLOW summary generator (`sopflow_summary.py`) produces stochastic-specific context (number of wind scenarios, first-stage dispatch note, wind generator output) alongside standard OPFLOW metrics.

PFLOW has its own parser (`pflow_parser.py`) because its output header and convergence indicators differ fundamentally from OPFLOW: it prints "AC Power Flow" instead of "Optimal Power Flow", uses Newton-Rhapson instead of IPOPT, reports "Number of iterations" (lowercase) and "Solve Time (sec)" instead of IPOPT-specific counters, has no `EXIT:` line, and most importantly has no "Objective value" line (power flow does not optimize). The PFLOW parser sets `objective_value=0.0`, `model="AC"`, `objective_type="PowerFlow"`, and `ipopt_exit_status=""`. It derives convergence from the `Convergence status` header line (`CONVERGED` or `DID NOT CONVERGE`) and classifies feasibility accordingly, with near-boundary detection for marginal cases. The PFLOW summary generator (`pflow_summary.py`) emphasizes "Power Flow Results — Analysis, Not Optimization", notes that `set_gen_voltage` directly constrains bus voltage (not an initial guess), and computes generation cost from dispatch × cost curves via `OPFLOWResult.compute_generation_cost()` rather than showing an optimised objective value.

**Violation checking** uses actual bus voltage limits (`Vmin`/`Vmax`) from the input MATPOWER network rather than hardcoded thresholds. This ensures that when the LLM tightens voltage limits via `set_bus_vlimits` or `set_all_bus_vlimits`, violations are reported accurately against the enforced limits. If bus limits are not available (backward compatibility), falls back to 0.9/1.1 p.u.

### 4.4 Search Journal

In-memory data structure — one entry per iteration:

| Field | Type | Description |
|-------|------|-------------|
| iteration | int | Sequential iteration number |
| description | str | LLM-provided description of changes |
| commands | list[dict] | JSON modification commands applied |
| objective_value | float\|None | ExaGO objective function value (for PFLOW: computed from `compute_generation_cost(gencost)`, since PFLOW reports `0.0`) |
| feasible | bool | Whether simulation converged without violations |
| violations_count | int | Number of constraint violations |
| voltage_range | str | Min–max bus voltage magnitudes |
| max_line_loading | float | Highest line loading % |
| total_gen_mw | float | Total generation MW |
| total_load_mw | float | Total load MW |
| llm_reasoning | str | LLM's explanation of its choices |
| mode | str | "accumulative" or "fresh" |
| steering_directive | str\|None | Active steering directive at this iteration |
| num_steps | int | Number of time periods (TCOPFLOW); 0 for other applications |
| num_scenarios | int | Number of wind scenarios (SOPFLOW); 0 for other applications |

The `summary_stats()` method accepts optional `best_iteration_override` and `goal_type` parameters, allowing the post-search goal classification to select the correct best iteration rather than always defaulting to lowest cost.

`add_from_results` accepts an optional `gencost` parameter. PFLOW does not produce an objective value (the parser reports `0.0`); when `gencost` is supplied, the entry's `objective_value` and `tracked_metrics["generation_cost"]` are derived from `OPFLOWResult.compute_generation_cost(gencost)` instead. If the computed cost is itself `0.0` (no priced generators or all generators offline), the entry stores `None` rather than the misleading sentinel. This applies on both the modify path and the select path of concurrent PFLOW.

**`session_best`** — A top-level field on `SearchJournal` (not a per-entry field). Tracks the globally cheapest feasible cost found across *all variants ever run*, including non-selected variants from explore batches. Structure: `{cost: float, iteration: int, variant_label: str, commands: list[dict]}`. Updated via `update_session_best(label, iteration, cost, commands)` which is called after each explore batch. Persisted as a top-level key in the journal JSON alongside `entries`, `objective_registry`, and `preference_history`. Zero costs are rejected as sentinels.

### 4.5 Agent Loop Controller

Responsibilities:
- **Prompt assembly:** Combines system prompt + goal + journal + latest results + steering directives + error/warning feedback
- **Response parsing:** Extracts JSON from LLM output (handles markdown fences, extra text)
- **Action dispatch:** Routes "modify" → Modification Engine + Executor; "analyze" → detailed query; "complete" → terminate with report
- **Warning feedback:** Modifier warnings (e.g., `set_gen_voltage` ineffective under OPFLOW) are fed back to the LLM in the next iteration's prompt, allowing self-correction
- **Termination logic:** Stops on LLM completion, max iterations, or critical error (3 consecutive JSON parse failures)
- **Modification mode management:** Supports "accumulative" and "fresh" modes
- **Post-search analysis:** Makes a final LLM call to classify the search goal type and identify the best iteration; stores classification and analysis text on `SearchSession`

### 4.6 Goal Classification

After the search loop completes, a shared utility module (`agentigrid/engine/goal_classifier.py`) builds a classification prompt and parses the LLM's structured JSON response. Both the CLI path (`AgentLoopController._finalize`) and the GUI path (`SessionManager.get_summary_analysis`) use this shared logic.

Goal types:

| Goal Type | Description | Best Iteration Selection |
|-----------|-------------|------------------------|
| `cost_minimization` | Minimize generation cost | Lowest cost among feasible |
| `feasibility_boundary` | Find parameter limit before infeasibility | Feasible iteration closest to boundary |
| `constraint_satisfaction` | Satisfy specific constraints | Iteration best satisfying constraints |
| `parameter_exploration` | Explore what-if scenarios | Most informative feasible iteration |

If classification fails, falls back to `cost_minimization` (lowest cost heuristic).

### 4.7 Modification Engine

Deterministic Python module that applies JSON commands to MATPOWER .m files:

| Command | Parameters | Description |
|---------|-----------|-------------|
| `set_load` | {bus, Pd, Qd} | Set load at a specific bus |
| `scale_load` | {area\|zone\|bus, factor} | Scale load by factor |
| `scale_all_loads` | {factor} | Scale all loads uniformly |
| `set_gen_status` | {bus, id?, status} | Turn generator on/off |
| `set_gen_dispatch` | {bus, id?, Pg} | Set generator power output |
| `set_gen_voltage` | {bus, id?, Vg} | Set generator voltage setpoint (initial guess only — see note below) |
| `set_branch_status` | {fbus, tbus, ckt?, status} | Enable/disable branch |
| `set_branch_rate` | {fbus, tbus, ckt?, rateA} | Modify thermal rating |
| `set_cost_coeffs` | {bus, id?, coeffs} | Modify cost curve |
| `set_bus_vlimits` | {bus, Vmin, Vmax} | Set voltage bounds on a single bus |
| `set_all_bus_vlimits` | {Vmin, Vmax} | Set voltage bounds on ALL buses at once |
| `set_ground_resistance` | {substation, R_ground} | Modify grounding (GIC) |
| `scale_load_profile` | {factor} | Scale per-period load profile CSV files by factor (TCOPFLOW only) |
| `scale_wind_scenario` | {factor} | Scale wind generation columns in scenario CSV by factor (SOPFLOW only) |
| `set_tap_ratio` | {fbus, tbus, ratio, ckt?} | Set transformer tap ratio; rejected on non-transformer branches (ratio=0) |
| `set_shunt_susceptance` | {bus, Bs} | Modify bus shunt susceptance; positive = capacitive (raises voltage), negative = inductive |
| `set_phase_shift_angle` | {fbus, tbus, angle, ckt?} | Set phase shifter angle (degrees); rejected on non-phase-shifter branches (angle=0) |

All commands validated before application. Invalid commands reported to LLM as errors. Application-specific warnings are generated and fed back to the LLM.

**Concurrent explore — pre-execution rejection of all-no-op variants:** During an `explore` action, the agent loop calls `apply_modifications` on each variant before submitting it to the executor. If `report.applied` is empty for a variant — every command was rejected by validation, e.g. dispatching the slack bus or modifying an offline component — the variant is marked `rejected` on its `VariantResult`, its description is prefixed with `[REJECTED — all commands no-op]`, and **no PFLOW subprocess is launched**. Rejected variants still appear in the explore results table (rendered as `REJECTED` with all numeric columns elided), but `compute_pareto_labels` excludes them from the front and `_handle_select` refuses to adopt them.

**`VariantResult` fields (relevant to prompt B changes):**

| Field | Default | Description |
|-------|---------|-------------|
| `description` | auto-generated | Human-readable command summary. Auto-generated by `build_variant_description(commands, skipped_cmds)` from the command abbreviations unless the LLM supplies an explicit non-trivial description. Includes `[SKIP]` markers inline and a parenthetical skip count suffix. |
| `cost_equivalent_to` | `""` | Set to the label of a simpler variant that achieved the same rounded cost. Empty when no cost-equivalence is detected. Populated by `annotate_cost_equivalent_siblings` after each explore batch. |
| `rejected` | `False` | True when all commands were no-ops and no PFLOW subprocess was launched. |

**Identical-cost sibling annotation:** `annotate_cost_equivalent_siblings(variants, gencost)` groups feasible variants by their rounded cost (2 dp) after all simulations complete. Within each group, the variant with the fewest commands is the reference; all others get `" ← same cost as {ref}; extra commands had no effect"` appended to their description and `cost_equivalent_to` set. The function returns a batch-level warning string injected into the LLM's next error-feedback section. Infeasible and rejected variants are excluded from grouping.

**OPF voltage control note:** In OPFLOW, bus voltages are optimization decision variables — the solver picks optimal voltages within `Vmin`/`Vmax` bounds. The `set_gen_voltage` command only sets the initial guess (`Vg`), which OPFLOW overrides. To actually constrain voltages in OPF, use `set_all_bus_vlimits` (system-wide) or `set_bus_vlimits` (per-bus). The system prompt and command schema include this guidance, and a runtime warning is emitted when `set_gen_voltage` is used under OPFLOW.

**TCOPFLOW load profile note:** In TCOPFLOW, per-bus per-period loads are read from CSV profile files, NOT from the `.m` case file. Standard load commands (`scale_all_loads`, `set_load`) modify the `.m` file but TCOPFLOW overrides these with profile data — they have limited effect. The `scale_load_profile` command multiplies all numeric values in both P and Q profile CSVs by a factor, which is the correct mechanism for adjusting demand in TCOPFLOW. The system prompt includes explicit guidance about this, and `scale_load_profile` is skipped with a warning for non-TCOPFLOW applications. The modifier writes scaled profiles to per-iteration directories, and the agent loop tracks profile path overrides so that subsequent iterations and the executor use the scaled profiles.

**SOPFLOW wind scenario note:** In SOPFLOW, wind generation scenarios are read from a CSV file via the `-windgen` flag, NOT from the `.m` case file. Standard load commands modify the `.m` file but do not change wind scenario data. The `scale_wind_scenario` command multiplies all wind generation columns in the scenario CSV by a factor, preserving non-numeric columns (scenario_nr, sim_timestamp, weight). This is the correct mechanism for adjusting wind penetration in SOPFLOW. The scenario file supports two formats: single-period (`scenario_nr, <bus>_Wind_<id>..., weight`) and multi-period (`sim_timestamp, scenario_nr, <bus>_Wind_<id>...`). The launcher auto-selects scenario files matching the base case name using layered fallback matching (exact prefix → stripped suffix → all scenarios). SOPFLOW supports both IPOPT (single-core) and EMPAR (multi-core via MPI) solvers.

**PFLOW voltage control note:** In PFLOW, `set_gen_voltage` directly constrains the bus voltage — the solver uses the generator's voltage setpoint as a hard constraint, not an initial guess. This is the opposite of OPFLOW where the solver overrides Vg. The application-aware modifier emits a different message for PFLOW ("constrains bus voltage") vs OPFLOW ("initial guess only"), and the system prompt includes explicit guidance. The three new PFLOW-specific commands (`set_tap_ratio`, `set_shunt_susceptance`, `set_phase_shift_angle`) provide additional voltage and power flow control mechanisms. Tap ratios adjust transformer voltage transformation; shunt susceptance adds or removes reactive support at a bus; phase shift angles control power flow direction through phase-shifting transformers. PFLOW uses the Newton-Rhapson solver (single-core), not IPOPT. Convergence is indicated by `Convergence status: CONVERGED` / `DID NOT CONVERGE`. Since PFLOW does not optimize, `objective_value` is always 0.0; generation cost is computed from the dispatch using `OPFLOWResult.compute_generation_cost()`. The `generation_cost` metric is excluded from PFLOW's available metrics; the cost is shown in the results summary as a computed value.

### 4.8 User Interfaces

**CLI:** User provides base case path, application, search goal, optional config. Progress displayed live; post-search analysis with goal classification; summary report on completion.

**Streamlit GUI Launcher (`launcher/`):** Three-state application:
- **State A (Welcome):** Network summary, configuration sidebar, search goal input with preset examples
- **State B (Live Monitor):** Real-time iteration timeline with expandable details, convergence and voltage range charts, steering panel (augment/replace/pause/resume)
- **State C (Results):** Three tabs — Overview (summary metrics, comparison table, convergence charts), Detailed Results (voltage profile, generator dispatch, line loading charts, iteration history), Analysis & Report (LLM analysis with goal classification, search narrative, PDF report generation)

The `SessionManager` bridges the Streamlit UI with `AgentLoopController` via a background thread with callback-based updates through a thread-safe queue.

### 4.9 PDF Report Generator

Generates comprehensive PDF reports using ReportLab with DejaVu Sans font for diacritics support. Report sections:

1. **Title page** — search type, goal, metadata
2. **Executive summary** — goal-type-aware framing of results (cost reduction for cost minimization, neutral framing for feasibility boundary searches), LLM analysis with markdown table rendering
3. **Convergence analysis** — objective value and voltage range charts
4. **Results comparison** — base case vs best solution table and charts (voltage profile, generator dispatch)
5. **Iteration log** — complete search history
6. **Steering history** — user directives injected during the search

---

## 5. Data Flow and Iteration Lifecycle

1. **Prompt Construction** — Controller reads journal, formats compact table, combines with system prompt (command schema + network metadata + OPF-specific guidance), goal statement, steering directives, and previous results summary. Warning/error feedback from prior iteration is included.
2. **LLM Inference** — Prompt sent to selected backend. Expected response: JSON with "action" (modify|complete|analyze), "reasoning", and action-specific payload.
3. **Response Parsing** — JSON extracted, handling markdown fences and extra text. Parse failure → error logged, error feedback sent to LLM, loop continues (up to 3 consecutive failures).
4. **Command Validation** — Each command validated against network data. Invalid → error message fed back to LLM. Application-specific warnings generated (e.g., `set_gen_voltage` under OPFLOW) and also fed back.
5. **File Modification** — Valid commands applied to working copy. Written to timestamped directory.
6. **Simulation Execution** — ExaGO invoked. Timeout guard (default 120s).
7. **Results Parsing** — Full structured result + compact summary produced. Violations checked against actual bus Vmin/Vmax from the modified network.
8. **Journal Update** — New entry appended with iteration metadata and results.
9. **Loop Decision** — Continue unless LLM declared complete or max iterations reached.
10. **Post-Search Finalization** — Final LLM call for goal classification and analytical summary. Results stored on `SearchSession` for both CLI display and GUI reuse.

---

## 6. LLM Prompt Architecture

Four sections assembled per iteration:

**Section A — System Prompt (static):** Role definition, command schema with parameter types and validation rules (including OPF voltage control guidance), network metadata (bus count, generators with fuel types and capacities, areas, voltage levels), response format specification.

**Section B — Goal Statement (static):** User's original natural language goal, preserved exactly.

**Section C — Search Journal (grows):** Compact table, e.g.:
```
Iter | Description                      | Cost($)   | Feasible | V_range(pu)  | Max_load(%)
-----|----------------------------------|-----------|----------|--------------|------------
  1  | Base case (no modifications)     | 27,361.42 | Yes      | 0.95 – 1.05  | 78.3%
  2  | Scale all loads +10%             | 30,154.88 | Yes      | 0.94 – 1.06  | 86.1%
  3  | Scale all loads +20%             | 33,019.55 | Yes      | 0.92 – 1.07  | 93.7%
```

**Section D — Latest Results (refreshed):** Compact summary from most recent simulation.

**Section E — Steering Directives (when active):** Operator directives injected via augment/replace mode.

**Section F — Error/Warning Feedback (when present):** Command errors, parse failures, and application-specific warnings from the previous iteration.

**Session best line (PFLOW concurrent only):** Injected between the multi-objective tracking section and the explore table, when `journal.session_best` is set:
```
Session best (feasible): $29,924.90  [iter 5, variant A]
  Commands: vlim[0.95-1.05], scale×1.23, dispatch bus135→250MW
```
When no feasible variant has been found yet (first explore), this line is omitted. The session_best includes non-selected variants, so the LLM can detect regression even when selected iterations show higher costs.

**PFLOW "Reading Variant Results" paragraph:** Added to the PFLOW system prompt (both concurrent and sequential) via `_PFLOW_VARIANT_READING_GUIDANCE`. Contains three bullets: (1) identical cost → skipped command diagnosis, (2) regression detection via Session best, (3) session best as primary cost reference not journal selected-iteration costs. Located at `agentigrid/prompts/system_prompt.py` lines 354–367.

**Section G — Network Metadata (static, computed once):** Structural facts about the base case that constrain which actions can have an effect. Computed once at session start by `network_metadata(MATNetwork)` and injected into the system prompt:

- **Slack/reference bus(es):** which buses have `type == 3`, with an explicit reminder that `set_gen_dispatch` against the slack bus is a no-op (its Pg is determined by power balance).
- **Must-run generators:** online generators with `Pmin == Pmax` (dispatch fixed). The note flags that `set_gen_dispatch` on these generators will not change Pg.
- **Offline generators:** generators with `status == 0`, listed with their `Pmax` so the LLM can identify candidates for `set_gen_status: 1` if more capacity is needed.
- **Cost-curve diversity:** a summary of unique `(c2, c1, c0)` polynomial cost tuples among priced online generators. If exactly one unique tuple exists, an explicit warning is emitted that redispatch among these generators cannot reduce cost (since total cost is essentially fixed by the load level under PFLOW). Two-to-four unique tuples are listed inline with their member buses; five or more are summarised as "N distinct cost curves" with a marginal-cost breakdown.

The section aims for under 25 lines on a typical 200-bus network. Surfacing these facts up-front prevents the LLM from wasting iterations on commands that have no effect by construction.

### Expected LLM response formats:

Modify action:
```json
{
  "action": "modify",
  "reasoning": "Loads at +20% pushed voltages near limits. Trying +25% to find boundary.",
  "mode": "fresh",
  "commands": [
    {"action": "scale_all_loads", "factor": 1.25}
  ]
}
```

Complete action:
```json
{
  "action": "complete",
  "reasoning": "Network becomes infeasible between +25% and +30% load scaling.",
  "findings": {
    "summary": "The maximum feasible uniform load increase is approximately 27%.",
    "critical_buses": [42, 87, 153],
    "critical_lines": [{"from": 42, "to": 87, "loading_pct": 99.1}],
    "recommendation": "Reinforce the 230kV corridor between buses 42-87."
  }
}
```

---

## 7. Search Modes and Use Cases

| Mode | Name | Example Goal | Search Pattern | Phase |
|------|------|-------------|---------------|-------|
| A | Scenario Exploration | What happens if load +10% while largest gen trips? | Sequential, few iterations | 1 |
| B | Sensitivity / Boundary Finding | At what load level does network become infeasible? | Binary search / bisection | 1 |
| C | Qualitative Multi-Objective | Find cheap dispatch with healthy voltages | Iterative refinement with tradeoffs | 2 |
| D | Adversarial / Stress Testing | Find worst-case N-2 contingency | Combinatorial exploration | 3 |

**Accumulative mode:** Each iteration builds on previous modified state. Natural for incremental tuning.
**Fresh-start mode:** Each iteration starts from base case. Natural for comparative exploration.

---

## 8. Interactive Steering (Implemented)

Users can inject steering directives into a running search without stopping it.

### Mechanism

- **Steering queue** (`queue.Queue`): Any thread (CLI listener, GUI) calls `controller.inject_steering(directive, mode)`. Items are drained at the start of each iteration boundary.
- **Directive modes:**
  - `augment` — the directive is added alongside the current goal. Multiple augment directives accumulate.
  - `replace` — the directive replaces all previously active directives (clears the accumulation list), then becomes the sole active directive.
- **Active directives** are injected into the user prompt as an `=== Operator Directives ===` section. The LLM is instructed on how each mode affects priority.
- **Steering history** is recorded per iteration in `JournalEntry.steering_directive` and in `AgentLoopController._steering_history`. It is included in the PDF report.

### Pause / Resume

- `controller.pause()` — clears `threading.Event._pause_event`; the search thread calls `_pause_event.wait()` at the next iteration boundary and blocks without busy-waiting.
- `controller.resume()` — sets the event, unblocking the thread.
- An optional `on_pause_state(paused: bool)` callback notifies the GUI.

### CLI entry point

`agentigrid/cli.py` starts a daemon thread (`_start_stdin_listener`) when stdin is a TTY. Accepted commands: `pause`, `resume`, `stop`, `status`, `replace: <text>`, or bare text (augment).

### GUI entry point

The Streamlit launcher's live monitor exposes a steering text input plus Augment, Replace, and Pause/Resume buttons that forward to `SessionManager`, which delegates to `AgentLoopController`.

### Branching (Future)

Saving a state snapshot and exploring alternative search paths from a prior iteration is not yet implemented.

---

## 9. Tested Configurations and Results

### ACTIVSg200 — Maximum Loadability Search

| Backend | Model | Result | JSON Failures | Duration | Notes |
|---------|-------|--------|---------------|----------|-------|
| ollama-cloud | GLM-5 | λ=1.462× (90.0% loading) | ~40% | 158s (17 iter) | Best result with `set_all_bus_vlimits` |
| ollama-cloud | GLM-5 | Stuck at voltage issue | ~40% | 947s (15 iter) | Before `set_all_bus_vlimits` — LLM issued 10/200 bus limits |
| ollama-cloud | GLM-5.1 | No iterations executed | 100% | 1090s | Incompatible JSON output format |
| anthropic | Claude Sonnet 4.6 | Not yet tested | Expected ~0% | Expected ~$0.70 | Planned |

Key finding: the `set_all_bus_vlimits` command was essential for this test case — without it, the LLM could only set voltage limits on a handful of buses per iteration due to JSON output size constraints, leaving the majority of buses unconstrained.

---

## 10. Multi-Objective Decision Making (Implemented)

- **Tradeoff articulation:** LLM explains cost of moving between solutions in natural language.
- **Pareto approximation:** Systematic exploration presenting non-dominated solutions.
- **Preference elicitation:** Clarifying questions about user priorities via interactive steering.
- **Constraint softening:** Distinguishing hard constraints from soft preferences.

---

## 11. Phased Implementation Plan

### Phase 1 — Foundation + OPFLOW PoC ✅

| Step | Task | Details | Status |
|------|------|---------|--------|
| 1.1 | Project scaffolding | Directory structure, config (YAML), CLI args, logging | ✅ |
| 1.2 | LLM backend abstraction | Base class + OpenAI, Anthropic, Ollama adapters | ✅ |
| 1.3 | MATPOWER file parser/writer | Read .m → Python structures; write back preserving format | ✅ |
| 1.4 | Modification engine | Apply JSON commands; validate; report errors; application-specific warnings | ✅ |
| 1.5 | Simulation executor (CLI) | Invoke opflow binary, capture output, handle timeouts | ✅ |
| 1.6 | OPFLOW results parser | Extract objective, voltages, dispatch, flows, convergence; violation checking against actual bus limits | ✅ |
| 1.7 | Search journal | In-memory structure with append, format, export; goal-type-aware summary stats | ✅ |
| 1.8 | Agent loop controller | Prompt assembly, response parsing, action dispatch, termination, post-search goal classification | ✅ |
| 1.9 | System prompt templates | OPFLOW prompts with command schema, network metadata, OPF voltage control guidance | ✅ |
| 1.10 | CLI interface | Goal input, live progress, goal-type-aware final report | ✅ |
| 1.11 | End-to-end testing | ACTIVSg200 case, Ollama Cloud (GLM-5), feasibility boundary + steering | ✅ |

### Phase 1+ — Launcher + Reporting ✅

| Step | Task | Details | Status |
|------|------|---------|--------|
| 1+.1 | Streamlit GUI launcher | Config builder, session manager, live monitor, results view | ✅ |
| 1+.2 | PDF report generator | ReportLab-based, DejaVu Sans, markdown table rendering | ✅ |
| 1+.3 | Goal classification | Shared post-search LLM analysis for CLI and GUI paths | ✅ |
| 1+.4 | Goal-type-aware reporting | Correct framing for feasibility boundary, constraint satisfaction, etc. | ✅ |
| 1+.5 | Interactive steering (GUI) | Augment/replace directives, pause/resume, steering history | ✅ |
| 1+.6 | Bulk voltage commands | `set_all_bus_vlimits` for system-wide voltage constraint enforcement | ✅ |

### Phase 2 — Enrichment + Multi-Objective

| Step | Task | Details |
|------|------|---------|
| 2.1 | Interactive steering | ✅ Implemented |
| 2.2 | Analyze action | ✅ Implemented (pattern-matched + LLM fallback) |
| 2.3 | Multi-objective tracking | ✅ Implemented (objective registry, 14 metric extractors, LLM-driven tradeoff reasoning) |
| 2.4 | Session save/resume | ✅ Implemented (CLI --resume, GUI save/load, full state serialization) |
| 2.5 | Report generation | ✅ Implemented |
| 2.6 | Stress test mode (D) | ✅ Implemented (adversarial prompt template, --search-mode stress_test) |
| 2.7 | Replay mode | Replay completed sessions in the GUI |

### Phase 3 — Additional ExaGO Applications

| Step | Task | Details | Status |
|------|------|---------|--------|
| 3.1 | DCOPFLOW support | DC approximation; fast screening proxy; reuses OPFLOW parser with DC-aware summary, prompt, and metric filtering | ✅ |
| 3.2 | SCOPFLOW support | Contingency file via -ctgcfile; SCOPFLOW parser with metadata extraction; security-constrained summary; AC voltage + SCOPFLOW prompt sections; launcher contingency file selector | ✅ |
| 3.3 | TCOPFLOW support | Multi-period OPF with load profiles; TCOPFLOW parser with temporal metadata extraction; per-period file parsing via `-save_output`; aggregated + per-period mini-table summary; `scale_load_profile` command for CSV profile scaling; TCOPFLOW system prompt section (ramp coupling, load profile guidance, IPOPT-only); profile auto-matching in launcher (layered fallback convention); temporal parameters (duration, dT, iscoupling) in CLI and GUI; `num_steps` journal field; application-aware goal classifier context; 69 dedicated tests | ✅ |
| 3.4 | SOPFLOW support | Stochastic OPF with wind scenarios; SOPFLOW parser with metadata extraction and convergence override; two-stage summary with wind generator info; `scale_wind_scenario` command for CSV scenario scaling; SOPFLOW system prompt section (two-stage optimization, wind scenario guidance, solver choices); scenario file auto-matching in launcher (layered fallback); SOPFLOW solver (IPOPT/EMPAR) and coupling parameters in CLI and GUI; `num_scenarios` journal field; MPI guard for EMPAR; application-aware goal classifier context; 76 dedicated tests | ✅ |

### Phase 4 — PFLOW-Based Direct Optimization ✅

| Step | Task | Details | Status |
|------|------|---------|--------|
| 4.1 | PFLOW executor + parser | Power flow results extraction; PFLOW parser recognizes "AC Power Flow" header, Newton-Rhapson solver, convergence status; computes generation cost from dispatch × cost curves; no objective value (set to 0.0); feasibility classification with near-boundary detection | ✅ |
| 4.2 | Extended command set | `set_tap_ratio` (transformer taps), `set_shunt_susceptance` (bus shunt Bs), `set_phase_shift_angle` (phase shifters); validation rejects non-transformer/non-shifter targets; `set_gen_voltage` behavior differs for PFLOW (constrains voltage directly, not initial guess) | ✅ |
| 4.3 | Optimization prompt templates | Dedicated `_PFLOW_SECTION` explaining PFLOW is analysis not optimization; LLM is the optimizer; search heuristics (binary search, gradient-like, iterative adjustment); new commands 14–16 documented | ✅ |
| 4.4 | Benchmark vs. OPFLOW | Compare LLM-driven PFLOW vs. OPFLOW optimal solutions | Planned |
| 4.5 | Concurrent PFLOW (explore/select) | LLM proposes multiple variant configurations per iteration; system runs them concurrently via ThreadPoolExecutor; Pareto front identifies non-dominated variants; LLM selects best variant as new current point; replaces sequential binary search with parallel coordinate search | ✅ |

### Phase 4.5 — Concurrent PFLOW Explore/Select Design

#### Concept

Traditional PFLOW search uses one `modify` action per iteration, testing a single parameter change. This makes binary search for feasibility boundaries inherently sequential: each iteration takes one LLM round-trip (~20-30s) but only ~0.02s of simulation time. The LLM round-trip is the bottleneck.

Concurrent PFLOW introduces two new actions:

| Action | Description |
|--------|-------------|
| `explore` | LLM proposes 2–8 variant command sets (each an independent set of modifications). System runs all simulations concurrently via ThreadPoolExecutor. Results are presented with Pareto front analysis. |
| `select` | LLM selects one variant from the explore results as the new current point. The selected variant's modified network becomes `_current_network` and a journal entry is recorded. |

Each explore+select cycle costs **one iteration** against `max_iterations`, but evaluates N configurations. The LLM roundtrip cost is amortized across N simulations.

#### Architecture

| Component | File | Description |
|-----------|------|-------------|
| `ParetoCandidate` + `pareto_filter()` | `agentigrid/engine/pareto.py` | Non-dominated sorting over variant results. Feasible candidates dominate infeasible ones. Default objective: lowest generation cost. Multi-objective: uses registered objective directions. |
| `VariantResult` + `ExploreCache` | `agentigrid/engine/explore.py` | Data structures for explore results. `ExploreCache` holds all variant results between explore and select actions. |
| `format_variant_results()` | `agentigrid/engine/explore.py` | Formats variant comparison table for LLM prompt with Pareto ★ markers. |
| `_handle_explore()` | `agentigrid/engine/agent_loop.py` | Parses variant command lists, applies each to a deep copy of the base/current network, runs all simulations concurrently via `run_parallel()`, computes Pareto front, stores results in `_explore_cache`. |
| `_handle_select()` | `agentigrid/engine/agent_loop.py` | Validates selection against cache, updates `_current_network` to selected variant's network, creates journal entry with `explored_variants` metadata, clears cache. |
| `SimulationExecutor.run_parallel()` | `agentigrid/engine/executor.py` | Thin ThreadPoolExecutor wrapper that runs N simulations concurrently. Each simulation is an independent subprocess. |
| System prompt | `agentigrid/prompts/system_prompt.py` | Dynamic action schema (5 actions for concurrent PFLOW vs 3 for standard). `explore` is action #1 (primary) when concurrent mode is on. Search heuristics are rewritten to recommend parallel search patterns. |

#### Search Strategy

When concurrent PFLOW is enabled, the system prompt is restructured:

- **Standard mode (concurrent off):** "Choose one of three actions: modify, complete, analyze" + sequential search heuristics (binary search with modify)
- **Concurrent mode (on):** "Choose one of five actions: explore, select, modify, complete, analyze" + parallel search heuristics (explore with multiple factors, select best, explore narrower range)

This ensures the LLM uses `explore` as its primary search mechanism rather than falling back to sequential `modify` actions.

#### Config and CLI

| Parameter | Default | CLI Flag | Purpose |
|-----------|---------|----------|---------|
| `search.concurrent_pflow` | `False` | `--concurrent-pflow` | Enable explore/select for PFLOW |
| `search.max_variants` | `8` | `--max-variants N` | Maximum variants per explore action (2–16) |

#### Launcher UI

- **"Concurrent explore/select" checkbox** in sidebar (PFLOW only)
- **"Max variants per explore" number input** (2–16, default 8, enabled only when concurrent is checked)
- **Explore status panel** in live monitor showing variant labels, feasibility, voltage ranges, Pareto ★ markers
- **Phase indicator** shows "Running 5 variants..." during concurrent simulation
- **Iteration log** shows explored variants summary for `select` entries

#### Data Model

| Field | Type | Description |
|-------|------|-------------|
| `JournalEntry.explored_variants` | `Optional[list[dict]]` | For `select` entries: `[{label, feasible, cost}]` for all variants explored, enabling CSV/JSON traceability |
| `ExploreCache` | dataclass | Temporary in-memory cache between explore and select; not persisted to session (variant results too large). Metadata saved as `explore_cache_info` in session JSON. |

#### Session Persistence

- `explore_cache_info` dict saved in session JSON with variant labels, description, and iteration number
- On resume: explore cache is cleared (variant results are too large to persist). The LLM must re-explore.
- Session format version bumped to `1.1` for backward compatibility with `1.0`
- Journal CSV includes `explored_variants` column

#### Pareto Front Computation

The `pareto_filter()` function uses standard Pareto dominance:

- A feasible candidate dominates an infeasible candidate
- Among feasible candidates, A dominates B iff A is ≥ B on all objectives and > B on at least one
- Direction-aware: `"minimize"` → lower is better; `"maximize"` → higher is better
- Default (no objectives registered): use `generation_cost` with direction `"minimize"`

For PFLOW, when objectives are registered (e.g., `load_scaling_factor` maximize + `max_line_loading_pct` constraint), the Pareto front identifies variants that trade off load increase against thermal loading.

#### Test Coverage

| Test file | Classes | Count |
|-----------|---------|-------|
| `tests/test_pareto.py` | `TestDominates`, `TestParetoFilter` | 21 |
| `tests/test_explore.py` | `TestFormatVariantResults`, `TestComputeParetoLabels`, `TestExploreCache`, `TestConfigConcurrentPflow`, `TestSystemPromptConcurrent`, `TestSystemPromptConcurrentMode`, `TestUserPromptExploreText`, `TestJournalExploredVariants`, `TestSessionExplorePersistence`, `TestExploreHandlerValidation`, `TestRunParallel`, `TestConcurrentPflowCLI` | 23 |

### Phase 5 — ExaGO Integration

| Step | Task | Details |
|------|------|---------|
| 5.1 | Python API executor | ExaGO Python bindings; eliminate file I/O overhead |
| 5.2 | Large network support | Scalable metadata summarization; tool-use for 10k+ buses |
| 5.3 | Launcher integration | Add to Streamlit launcher |
| 5.4 | Visualization integration | Connect to Flask visualization system |

---

## 12. Integration with ExaGO Project

- **Shared file formats:** Same MATPOWER .m and .gic files.
- **ExaGO as external dependency:** Black-box binary during Phases 1–4.
- **Configuration portability:** Paths externalized to YAML.
- **Python API readiness:** Executor interface supports mode swap.
- **Conda compatibility:** Runs in exago312 environment (ExaGO) and agentigrid environment (AgentiGrid, Python 3.11).

---

## 13. Claude Code Workflow

Each implementation step follows:
1. **Architecture briefing:** This document provided as context.
2. **Task specification:** Specific prompt saved to `prompts/` folder with I/O expectations, interfaces, test criteria.
3. **Implementation:** Claude Code reads the prompt from file and writes code following project structure.
4. **Review and testing:** Verified by Claude (brainstorming instance) via Filesystem MCP inspection; issues result in follow-up prompts.
5. **Integration check:** New component verified against existing components.

This document serves as shared context between the human developer, Claude (brainstorming), Gemini (brainstorming), and Claude Code (implementation).

---

## 14. Key Lessons Learned

- **OPF voltage control:** In OPFLOW, `set_gen_voltage` only sets an initial guess — the solver overrides it. Voltage must be controlled via bus limits (`Vmin`/`Vmax`). This required adding explicit guidance in the system prompt, command schema warnings, and runtime feedback to the LLM.
- **Bulk commands matter:** Without `set_all_bus_vlimits`, the LLM could only constrain ~10 of 200 buses per iteration due to JSON output size limits. The bulk command was essential for the feasibility boundary search to succeed.
- **Violation reporting must use actual limits:** Hardcoded violation thresholds (0.9/1.1) masked constraint violations when the user or LLM set tighter limits. Using the actual bus `Vmin`/`Vmax` from the modified network fixed this.
- **Goal-type-aware reporting:** Always framing results as cost reduction is incorrect for feasibility boundary searches where cost increase is expected and desirable.
- **Model selection matters significantly:** GLM-5 achieves ~60% JSON reliability (wasting ~40% of iterations), while Claude Sonnet 4.6 offers near-perfect reliability at ~$0.70 per run. GLM-5.1 is currently incompatible.
- **LLM analysis vs deterministic code:** The LLM's free-text analysis often correctly identifies the best iteration even when the deterministic Python code selects the wrong one. The adopted principle: let the LLM determine what "best" means, with Python fallback if classification fails.
- **TCOPFLOW load profiles vs .m file loads:** In TCOPFLOW, per-period loads come from CSV profile files, not the .m case file. Standard load commands (scale_all_loads, set_load) modify the .m file but have limited effect since TCOPFLOW overrides them. A dedicated `scale_load_profile` command that modifies the CSV files directly was essential. The system prompt must explain this clearly or the LLM will waste iterations modifying .m file loads that TCOPFLOW ignores.
- **Multi-period result aggregation:** TCOPFLOW's `-print_output` only shows period-0 in stdout; the full per-period data is in the `tcopflowout/` directory (requires `-save_output`). For effective LLM decision-making, results must be aggregated across all periods (worst voltage, peak load, worst line loading) with a compact per-period mini-table, rather than dumping all period data into the prompt.
- **SOPFLOW wind scenarios vs .m file loads:** In SOPFLOW, wind generation scenarios are read from a CSV file via `-windgen`, not from the .m case file. Standard load commands modify the .m file but do not change wind scenario data. A dedicated `scale_wind_scenario` command that modifies the scenario CSV directly was essential, analogous to `scale_load_profile` for TCOPFLOW. The system prompt must explain this clearly or the LLM will attempt invalid modifications.
- **SOPFLOW convergence override:** SOPFLOW output may not include an IPOPT EXIT message, causing the OPFLOW parser (which checks for exit status) to set `converged=False` even when the header says "CONVERGED". The SOPFLOW parser must override this based on the `convergence_status` header field, similar to SCOPFLOW.
- **Near-boundary feasibility classification:** Both TCOPFLOW and SOPFLOW can produce non-converged results where the solver is very close to feasibility (e.g., voltage within 0.01 pu of a limit). Classifying these as straightforward "infeasible" would miss boundary markers that are critical for feasibility boundary searches. The `_is_near_boundary()` heuristic checks voltage proximity (within 0.01 pu of limits) and line loading (within 5% of 100%) to classify such results as "marginal" rather than "infeasible", enabling binary search convergence guidance in the system prompt.
- **PFLOW is analysis, not optimization:** PFLOW does not minimise cost — it solves the power flow equations for a given network state. This means `set_gen_voltage` directly constrains bus voltage (the opposite of OPFLOW where it is just an initial guess). The LLM must serve as the optimizer: proposing dispatch changes, evaluating feasibility, and deciding next steps. The system prompt includes search heuristics (binary search for boundaries, gradient-like adjustments for cost reduction, iterative tuning for voltage improvement). Generation cost is computed from dispatch × cost curves via `compute_generation_cost()`, not from an objective value. The `generation_cost` metric is excluded from PFLOW's available metrics to avoid confusion with the always-zero `objective_value`. The launcher UI handles PFLOW's zero objective value by showing feasibility-based summaries instead of cost-based metrics.
- **Concurrent PFLOW prompt structure matters:** When `concurrent_pflow` is enabled, the system prompt must present `explore` as the primary search action (action #1), not as an optional add-on after `modify`. If `modify` is listed first and described as the default, the LLM will use sequential binary search instead of parallel explore even though explore is available. The prompt must also replace sequential search heuristics ("binary search with modify") with parallel search heuristics ("use explore to test multiple factors in one round"). Without these changes, the LLM ignores the explore action despite it being available and more efficient.
