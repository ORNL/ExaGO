# Claude Code Changes Log

This document records significant changes made by Claude Code, grouped by the prompt or analysis that motivated them.

---

## C.2/C.3 corrections — metric gating, summary aggregation, predicate audit, sweep dedup (2026-06-23)

Four contained fixes from the prompt-13/15/17 runs. No capability behavior changed (capacity numbers, dispatch results, metric/predicate definitions, and the C.1 binding identifier are untouched) — only how results are gated, summarized, audited, and deduplicated.

### Fix 1 — Custom metrics are certified-gated (prompt 15)

`max_delta_v` was computed for non-converged candidates from uncertified iterates, so prompt 15 reported 0.09 p.u. at bus 77 (which did not converge; iterate showed 293.8% loading) instead of the feasible max 0.032 at bus 139.

- `agentigrid/engine/agent_loop.py`: new shared `_is_certified(opflow)` gate (status starts with `CONVERGED`, the same gate the C.1 binding identifier relies on). In the sweep loop, `metric_value` is computed only when `_is_certified(opflow)`, else `None`. The LLM full-table view then shows `N/A` for non-converged candidates and the reduction is over feasible buses only.
- `agentigrid/engine/sweep_metrics.py`: docstring makes explicit that `max_delta_v` is over ALL buses (including the connection bus) and is a steady-state sensitivity between two cost-optimal profiles under OPFLOW, not a physical transient (documentation only).

### Fix 2 — Summary aggregation descends into sweep variants (prompt 13)

The header reported `Lowest-cost feasible: 27557.57 (iteration 0)` (base case) instead of the sweep optimum $27,367.73 at bus 181, because sweep iterations have `objective_value = None`.

- `agentigrid/engine/journal.py`: `summary_stats` now descends into cost sweeps' `explored_variants` and takes the global minimum feasible per-candidate cost; returns a new `best_bus`. Goal-aware: boundary sweeps (`max_feasible_mw`) and custom-metric sweeps (`metric_name`) are skipped, and descent only runs for `goal_type in (None, "cost_minimization")`. New helper `_best_cost_in_sweeps`.
- `agentigrid/engine/goal_classifier.py`: the "Lowest-cost feasible" line shows the bus when `best_bus` is present.

### Fix 3 — `reactive_adequacy` forces Q and records it (prompt 17)

Confirmed `_augment_generator_mutation` already pins `Qmin == Qmax` (forced, not bounded). Added the audit trail:
- `agentigrid/engine/agent_loop.py`: records `dispatched_q` (added unit's solved `Qg`) for `reactive_adequacy` gen sweeps, gated on `_is_certified`.
- `launcher/report_generator.py`, `launcher/app.py`: "Dispatched Q (MVAr)" column when present + caption noting it should equal the Qmax target at adequate buses.

### Fix 4 — Redundant sweeps suppressed (prompts 13 & 17)

Prompt 13 re-ran an identical sweep 3× (50,504 tokens), prompt 17 2×.
- `agentigrid/prompts/system_prompt.py`: explicit rule "DO NOT RE-RUN AN IDENTICAL SWEEP".
- `agentigrid/engine/agent_loop.py` (deterministic backstop): session-level `_sweep_signature_cache` keyed by `_sweep_cache_key(data)` (mode + entity + mutation + candidate_set + feasibility + metric + predicate + dispatchable/cost params; excludes description/reasoning). An identical re-request is served via `_serve_cached_sweep` (journaled "cached — not re-solved", no re-solve); changed parameters → different signature → still executes. Soft guard, never blocks a changed sweep. **Decision flagged for Slaven:** nudge-only vs. nudge + cache (both implemented here).

### Tests — `tests/test_sweep_corrections.py` (15)

Certified-gate helper; metric None for non-converged + max-over-feasible reduction + all-converged regression; summary descends into cost sweeps (bus 181/$27,367.73), ignores metric/boundary sweeps, gated off for non-cost goals; predicate forces Qmin==Qmax + dispatched_q == target; system-prompt rule present; dedup serves cache (no new solves) and re-executes on changed params. Full suite: **892 passed, 3 skipped**.

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/agent_loop.py` | `_is_certified`; metric certified-gate; `dispatched_q`; sweep dedup cache (`_sweep_cache_key`/`_store_sweep_cache`/`_serve_cached_sweep`) |
| `agentigrid/engine/journal.py` | `summary_stats` descends into sweep variant costs; `best_bus`; `_best_cost_in_sweeps` |
| `agentigrid/engine/goal_classifier.py` | header shows best bus |
| `agentigrid/engine/sweep_metrics.py` | `max_delta_v` scope/semantics docstring |
| `agentigrid/prompts/system_prompt.py` | no-redundant-sweep rule |
| `launcher/report_generator.py`, `launcher/app.py` | "Dispatched Q (MVAr)" column + caption |
| `tests/test_sweep_corrections.py` | new test file (15 tests) |
| `agentigrid_architecture.md`, `launcher/README.md`, `launcher/ARCHITECTURE.md` | the four corrections |

---

## C.1 correction — Binding-constraint identification (2026-06-23)

**Scope:** ONLY the routine that writes the `binding_constraint` diagnostic label for each boundary candidate. The bisection, bracketing, feasibility oracle, and every `max_feasible_mw` value are untouched — **no capacity number moved**.

### Task 0 — duals check

ExaGO OPFLOW output exposes **line-flow Lagrange multipliers** (`mult_Sf`/`mult_St` in the branch table; zero in the base sample, nonzero exactly when a line limit is active) but **not** voltage-bound shadow prices (the bus table has only power-balance multipliers `mult_Pmis`/`mult_Qmis`). → **Branch A (duals)** for thermal, **Branch B (margin-vs-base)** for voltage. Hybrid.

### Two bugs fixed

1. **Thermal misattribution** (41/129 load, 7/31 gen): the old identifier computed loading from `Sf` only and named a local line at 94–97% while the system's most-loaded line was at 100.0%. Now: prefer the active line with the largest |multiplier|; else the *globally* most-loaded line(s) within `binding_eps` (0.5%) of the system max, using both ends `max(|Sf|,|St|)/Slim`. Never restricted to lines incident to the injection bus; ties are listed.
2. **Pinned-voltage misattribution** (35 load, 40 gen): the old value-based rule (`metric == limit`) always blamed a PV bus regulating to Vmax=1.10 in *every* solve (including base). Now a voltage bound binds only if its margin **collapsed** vs the base solve (clearly positive in base → ~0 at the boundary); buses already at the bound in base are excluded. When both a thermal line and a voltage bound are newly active, **both** are reported.

### Changes

- `agentigrid/engine/agent_loop.py`: rewrote `_identify_binding` (now activity-based, signature `(opflow, vmin_lim, vmax_lim, base_opflow=None, binding_eps=0.5)` — backward-compatible) with helpers `_line_loading_pct` (both ends), `_line_dual`, `_thermal_binding` (duals → global-max), `_voltage_binding` (margin-vs-base, pinned-excluded), `_fallback_binding` (legacy normalized-slack, last resort). The base-case solved result is threaded through `_bisect_candidate` (new `base_opflow` param) and `_make_fn` so the margin comparison uses the unmodified base.
- No parser changes — `mult_Sf`/`mult_St` and per-bus `Vm` were already parsed.

### Capacity-invariance guarantee

The binding label is consumed only when building the result dict; `_bisect_boundary` reads only `_ProbeOutcome.feasible`. A test asserts two oracles with identical feasibility but different binding strings yield the same `max_feasible_mw`. Existing journals that predate this fix cannot be retroactively corrected (they lack the per-line/per-bus arrays needed to recompute) — only re-runs get the corrected label.

### Tests — `tests/test_boundary_sweep.py` (+9)

Thermal global-max (remote 100% line named over local 96%; both-end loading; no spurious thermal below limit); duals (largest-|multiplier| line chosen, zero-multiplier slack line ignored); voltage margin-vs-base (pinned Vmax excluded, collapsed Vmin reported, both-active reports both); capacity invariance to the binding label. Full suite: **877 passed, 3 skipped**.

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/agent_loop.py` | Rewrote `_identify_binding` (+ helpers); threaded `base_opflow` into `_bisect_candidate` |
| `tests/test_boundary_sweep.py` | +9 binding-correction tests |
| `agentigrid_architecture.md` | Activity-based binding identification (duals + margin-vs-base) |

---

## Slice 3 — C2 + C3: Dispatchable/Cost-Curve Generators & Custom Sweep Metric/Predicate (2026-06-23)

Two independent phases extending the post-C.1 sweep's per-candidate contract.

### C2 — Cost-curve & dispatchable generators (prompt 13)

Min-cost siting of a **dispatchable** generator under economic dispatch — the deliberate exception to forced-injection hosting. The OPF must choose the unit's output, so the unit needs a load-bearing cost curve (zero cost → dispatches to max everywhere; too-high → never dispatches).

- `agentigrid/engine/commands.py`: `AddGeneratorAtBus.cost_coeffs` (MATPOWER model-2 polynomial).
- `agentigrid/engine/modifier.py`: `_median_existing_cost_coeffs(net)` (case-median mid-merit curve, fallback `[0, 40, 0]`); the add-generator primitive writes explicit coeffs, else the case median for dispatchable units, else zero for fixed injection. `dispatchable` (Pmin=0/Pmax=cap vs Pmin=Pmax=cap) already existed.
- `agentigrid/engine/agent_loop.py`: `_augment_generator_mutation` resolves dispatchable mode (mutation > `entity_dispatchable` > config default) and the cost curve (`added_gen_cost_strategy`); the candidate loop records the added unit's **dispatched Pg** at the solution.
- System prompt: **rule X.1** — explicit fixed-injection vs dispatchable mapping (chosen from wording, named back, never defaulted) + an economic-siting sweep example.
- Report/Streamlit: "Dispatched Pg (MW)" column + dispatchable caption; routed through the existing cost-min ranking (inherits the prompt-14 near-optimal caveat).

### C3 — Custom per-candidate metric & feasibility predicate (prompts 15, 17)

A **named registry of verified primitives** (Phase 1; not free-form LLM code), extensible for later capabilities.

- `agentigrid/engine/sweep_metrics.py` (new): `METRICS` / `PREDICATES` dicts, `register_metric` / `register_predicate`, `metric_needs_base`, `metric_direction`.
  - metric `cost` (default), `max_delta_v` = `max_b |V_cand[b] − V_base[b]|` (worst system-wide voltage step; needs a base solve).
  - predicate `standard` (default; mirrors `_infeasible_reason`), `reactive_adequacy` (feasible OPF at forced P=Pmax, Q=Qmax; names the limiting quantity on failure).
- `agentigrid/engine/agent_loop.py`: the sweep selects `metric` / `feasibility_predicate` by name (unknown → clean error, no silent fallback); solves the base once when the metric needs it; the predicate decides feasibility; `reactive_adequacy` pins `Qmin = Qmax` in the mutation. New per-candidate keys (`metric_value`, `metric_name`, `predicate_name`, `dispatched_pg`) are added only when relevant, so the default path journals an **unchanged** payload.
- `_build_sweep_llm_view` gained a `rank_key` param (cost vs metric_value); the cost path is byte-identical (formatting preserved).
- Parser: per-bus `Vm` and per-gen `Pg/Qg` were already exposed — no parser changes needed.
- Report/Streamlit: adaptive feasible-table columns ("Max ΔV (pu)" / "Reactive-adequate buses"); **DejaVu Sans** preserved.

### Config (`search.*`)

`added_gen_cost_strategy` (`median_existing`), `added_gen_dispatchable_default` (`false`), `switched_load_mw` (`100.0`).

### Tests — `tests/test_sweep_metrics_dispatch.py` (32 tests)

C2 generator primitive (dispatchable bounds + non-zero cost, fixed-injection regression, explicit coeffs, median mid-merit, fallback); registry (defaults, needs-base, direction, unknown-name raises, extensibility); `max_delta_v` (true max abs diff, max bus ≠ candidate bus, None without base); `reactive_adequacy` (adequate, inadequate-with-reason, non-convergence); standard predicate unchanged; `_augment_generator_mutation` (dispatchable resolution, explicit-strategy error, Q-pin, non-generator untouched); sweep-handler integration (unknown metric/predicate clean error, metric recorded, **default payload unchanged**, dispatched-Pg recorded). Full suite: **868 passed, 3 skipped**.

### Decisions flagged for Slaven (non-blocking)

1. Added-generator cost curve: case-median (default) vs an explicit typical mid-merit curve — drives the prompt-13 ranking entirely.
2. Prompt-15 switched-in load size (`switched_load_mw`) — the voltage step scales with it.
3. Prompt-17 Pmax/Qmax source — taken from action params; confirm whether Qmax should default to a nameplate ratio (e.g. 0.33·Pmax) or always be explicit.

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/sweep_metrics.py` | New registry of named metrics/predicates |
| `agentigrid/engine/commands.py` | `AddGeneratorAtBus.cost_coeffs` |
| `agentigrid/engine/modifier.py` | Median-existing cost curve + cost-curve selection |
| `agentigrid/engine/agent_loop.py` | `_augment_generator_mutation`, metric/predicate selection, base solve, dispatched-Pg, `rank_key` |
| `agentigrid/config.py` + `configs/default_config.yaml` | Three `search.*` fields |
| `agentigrid/prompts/system_prompt.py` | X.1 mapping rule + economic-siting & metric/predicate examples |
| `launcher/report_generator.py`, `launcher/app.py` | Adaptive sweep columns (dispatched Pg, custom metric, predicate) |
| `tests/test_sweep_metrics_dispatch.py` | New test file (32 tests) |
| `README.md`, `agentigrid_architecture.md`, `launcher/README.md`, `launcher/ARCHITECTURE.md` | C2/C3 capability docs |

---

## Track A — C.1: Per-Candidate Boundary Search Inside the Sweep (2026-06-23)

**Capability C1** in `AgentiGrid_modification_plan.md`. Unlocks benchmark prompts 1 (max load hosting capacity per bus, at system-average PF) and 2 (max generator MW per bus). Generalizes the existing fixed-injection sweep: the per-candidate result becomes a **boundary** (max feasible MW + binding constraint) instead of a single feasible/infeasible verdict.

### Design

For each candidate bus, a bisection on injection magnitude finds the largest injection that still yields a feasible OPFLOW solve (the convergence boundary, since the V-band and Rate A are in-solve hard constraints). The outer loop over buses is parallel; the inner bisection is sequential. The whole boundary sweep is ONE LLM turn performing N bisections — it does not consume N iterations of the LLM budget.

- **Load entity:** adds `ΔP` and scales `ΔQ = ΔP · tan φ` on a constant-power-factor ray. `power_factor` is `system_average` (default, `ΣQd/ΣPd` of the base case), `unity`, or a number `0..1`.
- **Generator entity:** fixed-injection unit (`Pmin = Pmax = ΔP`); reactive output free within `±boundary_gen_q_frac · ΔP`. (Dispatchable would be zeroed by the OPF → vacuous test.)
- **Non-convergence** caps the bisection (standard for OPF hosting capacity); reported caveat that a numerically-failing probe understates capacity.
- **OPFLOW only.**

### Changes

- `agentigrid/config.py` + `configs/default_config.yaml`: new `search.boundary_initial_mw` (50.0), `boundary_max_mw` (2000.0), `boundary_tol_mw` (1.0), `boundary_max_probes` (24), `boundary_gen_q_frac` (0.4), `boundary_power_factor_default` ("system_average").
- `agentigrid/engine/agent_loop.py`:
  - Module-level pure helpers: `_system_average_tan_phi`, `_tan_phi_from_pf_spec`, `_mutate_candidate_network` (reuses `add_load_at_bus` / `add_generator_at_bus` via parse_command + apply_modifications), `_identify_binding` (thermal vs voltage by smallest normalized slack, with the specific line/bus), `_ProbeOutcome`, `_bisect_boundary` (exponential bracket then bisect), `_build_boundary_llm_view` (token-bounded, mirrors B.1 gating).
  - `AgentLoopController._resolve_candidate_buses` (refactored out of `_handle_sweep`, shared), `_bisect_candidate` (builds the solve-probe closure), `_handle_boundary_sweep` (base-feasibility check, parallel bisection map, journal, LLM view). `_handle_sweep` dispatches to it on `data["mode"] == "boundary"`.
- `agentigrid/engine/executor.py`: `map_callables(fns, max_workers, on_progress)` — runs arbitrary candidate callables in the thread pool (thread-pinning stays the callable's responsibility via `run(thread_limit=…)`); captures per-task exceptions by index.
- `agentigrid/prompts/system_prompt.py`: boundary mode documented as sweep example #5 + a rules line for max-capacity goals.
- `launcher/report_generator.py`: `_build_boundary_table` (hosting-capacity table, teal header, convergence-boundary footnote, DejaVu Sans preserved); `_build_sweep_results_section` and the executive summary detect the `max_feasible_mw` key.
- `launcher/app.py`: boundary detection in `_render_overview_tab` → hosting-capacity dataframe + header metrics (highest capacity / determining bus); normal feasible/infeasible sweep tables moved under an `else` so both paths reach the shared trailing sections.

### Tests — `tests/test_boundary_sweep.py` (34 tests)

System PF + PF-spec; candidate mutation on real ACTIVSg200 (PF ray, generator fixed injection, base untouched); binding identification (thermal/voltage with element, type-only fallback); bisection algorithm via synthetic oracle (boundary within tol, probe bound, cap-not-reached, first-probe-infeasible, convergence-as-infeasible); token-bounded boundary view (full vs summary, top-N bound, descending rank, 2000-bus size bound, grouped undetermined, threshold 0); `map_callables` (index order, exception capture, progress count); `_handle_boundary_sweep` orchestration (journals `max_feasible_mw`, one-LLM-turn invariant, base-infeasible abort, invalid-entity reject). Full suite: 836 passed, 3 skipped.

### Decisions flagged for Slaven (non-blocking)

1. Generator reactive handling: `Qg` free within `±0.4·ΔP` (current default) vs fixed at 0 — affects voltage-limited hosting numbers.
2. PF source for the load prompt: system-average (default) vs each bus's own PF vs a stated PF.
3. `boundary_max_mw` cap: absolute 2000 MW vs a multiple of total system load, for portability across network sizes.

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/agent_loop.py` | Boundary helpers + `_bisect_candidate` + `_handle_boundary_sweep` + sweep dispatch + `_resolve_candidate_buses` refactor |
| `agentigrid/engine/executor.py` | `map_callables` for parallel candidate callables |
| `agentigrid/config.py` + `configs/default_config.yaml` | Six `search.boundary_*` fields |
| `agentigrid/prompts/system_prompt.py` | Boundary sweep example + rules line |
| `launcher/report_generator.py` | Hosting-capacity table + boundary detection |
| `launcher/app.py` | Streamlit hosting-capacity table + header metrics |
| `tests/test_boundary_sweep.py` | New test file (34 tests) |
| `README.md`, `agentigrid_architecture.md`, `launcher/README.md`, `launcher/ARCHITECTURE.md` | Boundary-search capability, PF-ray/fixed-injection conventions, new config |

---

## Prompt #14 — Reporting Honesty Fixes (2026-06-23)

**Context:** A 100 MW load placement run on ACTIVSg200 (cost minimization goal) surfaced three reporting correctness issues. 55 of 56 infeasible sweep candidates had `status = "DID NOT CONVERGE"` but were being labelled "Line overload" by a heuristic that read iterate-level metrics. Bus 189 ($28,228.57) was declared the unique optimum over bus 187 ($28,230.05) despite a $1.48 gap — below IPOPT's noise floor. The report also implied voltage band was independently tested, which is not true for OPFLOW.

### Fix 1 — Solver certification semantics (infeasibility reason)

**Problem:** `_infeasible_reason` in `engine/agent_loop.py` accepted six scalar parameters (voltage_min, loading_pct, etc.) and applied heuristic thresholds to label buses "Line overload", "Voltage violation", etc. These labels were derived from the solver's last uncertified iterate and had no certified meaning.

**Principle:** OPFLOW reports `DID NOT CONVERGE` for all infeasible candidates. The iterate at which the solver stopped is not a certified operating point. The only certified information is that no feasible solution was found.

**Changes:**
- `agentigrid/engine/agent_loop.py`: simplified `_infeasible_reason(convergence_status: str) -> str` — returns `"constraint violation"` only if status starts with `"CONVERGED"` (PFLOW-style post-solve check), `"did not converge"` for everything else. No scalar metrics used.
- `launcher/report_generator.py`: added `_certified_reason(v: dict) -> str` static method that derives displayed reason from `status` field at render time, not from stored `reason` string. This correctly re-renders historical journals that contain old heuristic labels.
- `launcher/app.py`: same `_certified_reason` helper for Streamlit infeasible table.
- `launcher/report_generator.py`: two-row ReportLab table header with SPAN commands — "Bus" and "Status" span both header rows; "Last iterate — uncertified" spans the four metric columns in row 0; row 1 has per-column names. `repeatRows=2`. Footnote updated to explain the certification distinction.
- `launcher/app.py`: infeasible table column names updated to indicate uncertified iterate data ("V_min pu (last iter.)" etc.). Caption updated to state status-as-source-of-truth.

### Fix 2 — Near-optimal cluster caveat for cost minimization

**Problem:** A $1.48 cost difference between two buses (bus 189 vs bus 187) was reported as a decisive ranking without any indication that this gap is within IPOPT's noise floor (~$5/h).

**Changes:**
- `agentigrid/config.py`: added `ReportConfig` frozen dataclass with `cost_min_top_k: int = 10` and `near_optimal_abs_tol: float = 5.0`. Added to `AppConfig` with `field(default_factory=ReportConfig)`.
- `configs/default_config.yaml`: added `report.cost_min_top_k` and `report.near_optimal_abs_tol`.
- `launcher/app.py`: after the feasible bus table, when `goal_type in (None, "cost_minimization")`, shows a ranked table of the top-K cheapest buses with Δ-from-best column. Fires a `st.caption` caveat when rank-1 to rank-2 gap < `near_optimal_abs_tol`.
- `launcher/report_generator.py`: same top-K table and caveat in the PDF, using a blue-header ReportLab Table.

### Fix 3 — Voltage criterion honesty note

**Problem:** Report and UI implied that the voltage band was independently tested as a filter, when in fact OPFLOW enforces it as an in-solve hard constraint.

**Changes:**
- `launcher/app.py`: one-sentence `st.caption` note explaining that under OPFLOW, voltage band and Rate A are in-solve hard constraints satisfied by construction for any converged candidate.
- `launcher/report_generator.py`: same note added as a Paragraph after the sweep summary.

### Sweep concurrency (prompt #14 precursor, same session)

**Problem:** All OPFLOW sweep candidates were solved sequentially. No config knob for parallelism. No BLAS oversubscription protection.

**Changes:**
- `agentigrid/config.py`: `SearchConfig.sweep_max_workers: int = 0` (0 = auto = `min(cpu_count, 16)`).
- `configs/default_config.yaml`: `search.sweep_max_workers: 0`.
- `agentigrid/engine/executor.py`: `_THREAD_ENV_VARS` tuple; `run()` gains `thread_limit: int | None = None` — sets BLAS thread env vars to `thread_limit` when not None; `run_parallel()` gains `max_workers`, `thread_limit`, `on_progress` parameters.
- `agentigrid/engine/agent_loop.py`: `_resolve_sweep_workers(n_tasks)` method; passes `workers` and `thread_limit=1 if workers > 1 else None` to `run_parallel`; `_sweep_progress` callback for phase indicator.
- `launcher/config_builder.py`: `sweep_max_workers` parameter forwarded to config overrides.
- `launcher/app.py`: "Run sweep in parallel" checkbox + "Concurrent solves" number input in sidebar (OPFLOW only).

### Tests

New file: `tests/test_sweep_reporting.py` (288 lines)

- `TestInfeasibleReason` (7 tests): DID NOT CONVERGE, FAILED, empty, None, CONVERGED, case-insensitive, CONVERGED-prefix partial match
- `TestCertifiedReasonRendering` (5 tests): all DNC candidates render "Did not converge" regardless of stored reason; bus 77 (293.8% loading, DNC) correctly labelled; CONVERGED+infeasible → "Constraint violation"; BUILD_ERROR → "Did not converge"
- `TestNearOptimalCaveat` (4 tests): fires at $1.48 gap (< $5), doesn't fire at $10, doesn't fire at exactly $5, single candidate has no gap
- `TestTopKRanking` (4 tests): K-selection, delta computation, ascending order, None-cost excluded
- `TestPDFFontRegression` (2 tests): DejaVuSans or Helvetica used; `_certified_reason` static method present and correct

---

## Track B — B.1: Token-Bounded LLM-Facing Sweep Results View (2026-06-23)

**Context:** ACTIVSg200 sweeps cost ~18,658 tokens for the per-candidate collect-all table. ACTIVSg2000 is ~100,903 tokens. The table scales O(n_candidates) — at 3000+ buses it risks exhausting the context window. The LLM's job in a sweep is reduction, not reading N rows; the full data lives in the journal.

### Design

Threshold-gated view, controlled by two new config fields:
- `search.sweep_full_table_threshold: 250` — candidates ≤ threshold → full table (unchanged output, preserves ACTIVSg200 baselines); candidates > threshold → summarized view. Setting to 0 forces summary at every size.
- `search.sweep_llm_top_n: 25` — size of the ranked block in the summary.

Summarized view contains:
1. Header: candidate count, feasible count, infeasible count
2. Complete feasible bus list (integers, cheap even at 2000 buses)
3. Infeasible buses grouped by reason (O(n_reasons) not O(n_infeasible))
4. Top-N feasible ranked by primary objective (name + direction from the objective registry; fallback: cost minimize)
5. Aggregate stats: min/median/max over the feasible set
6. Journal pointer line stating that the full table is in the journal/report

### Changes

- `agentigrid/config.py`: `sweep_llm_top_n: int = 25` and `sweep_full_table_threshold: int = 250` added to `DEFAULTS["search"]` and `SearchConfig`.
- `configs/default_config.yaml`: same two fields added.
- `agentigrid/engine/agent_loop.py`:
  - Added `import statistics` and `from collections import defaultdict`.
  - New module-level `_build_sweep_llm_view(candidate_summaries, feasible_buses, mut_desc, objective_name, objective_direction, top_n, threshold) -> str` — full-table branch is byte-identical to pre-B.1 for candidate_count ≤ threshold; summary branch for > threshold.
  - Replaced inline section "# 5. Build compact results table" with primary-objective lookup from `self._journal.objective_registry.get_primary()` and a call to `_build_sweep_llm_view`. Journal path (`add_sweep`) untouched.
- `tests/test_sweep_llm_view.py` (new, 19 tests):
  - `TestSmallSweepFullTable`: byte-identical regression, threshold-boundary, infeasible truncation
  - `TestLargeSweepSummary`: header counts, complete feasible list, grouped infeasible, top-N count, aggregate stats, journal pointer, objective label, maximize direction
  - `TestSizeBound`: ranked block bounded at top_n for 2000-bus sweep; line count doesn't scale with n_infeasible
  - `TestCompleteness`: all buses in feasible + infeasible lists
  - `TestObjectiveFallback`: cost/minimize fallback label and ascending rank

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/agent_loop.py` | `_build_sweep_llm_view` helper; wired into `_handle_sweep` section 5 |
| `agentigrid/config.py` | `sweep_llm_top_n`, `sweep_full_table_threshold` in DEFAULTS and SearchConfig |
| `configs/default_config.yaml` | same two config fields |
| `tests/test_sweep_llm_view.py` | new test file (19 tests) |
| `agentigrid_architecture.md` | new Key Lesson on token-bounded sweep text |
| `claude_code_changes.md` | this entry |

---

## Prompt #14 — Reporting Honesty Fixes (2026-06-23)

### Files modified

| File | Change |
|---|---|
| `agentigrid/engine/agent_loop.py` | Simplified `_infeasible_reason`; added `_resolve_sweep_workers`; sweep progress callback |
| `agentigrid/engine/executor.py` | Thread pinning via `_THREAD_ENV_VARS`; `run_parallel` concurrency + progress |
| `agentigrid/config.py` | `ReportConfig` dataclass; `AppConfig.report`; `SearchConfig.sweep_max_workers` |
| `configs/default_config.yaml` | `report.*` and `search.sweep_max_workers` fields |
| `launcher/config_builder.py` | `sweep_max_workers` parameter |
| `launcher/app.py` | Sweep sidebar controls; `_certified_reason`; top-K ranking; voltage note; column renames |
| `launcher/report_generator.py` | `_certified_reason`; two-row infeasible header with SPAN; top-K PDF table; voltage note |
| `tests/test_sweep_reporting.py` | New test file (all of the above) |
| `launcher/README.md` | Sweep concurrency controls; reporting semantics; config table |
| `launcher/ARCHITECTURE.md` | Section 13: certification semantics, near-optimal, voltage note, concurrency |
| `agentigrid_architecture.md` | Three new Key Lessons Learned entries |
| `claude_code_changes.md` | This file |
