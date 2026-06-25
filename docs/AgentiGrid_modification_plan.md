# AgentiGrid — Modification Plan

**Purpose:** A single living roadmap so nothing falls out of view. Two parallel tracks:
**Track A** — capability additions so AgentiGrid can resolve all 20 benchmark prompts.
**Track B** — scaling (network size) and token-economy improvements.

**Last updated:** 2026-06-19
**Benchmark network for validation:** `case_ACTIVSg200` (primary), `case_ACTIVSg2000` (scaling)

**Status legend:** ✅ done · 🚧 doing (in progress) · ⛔ planned · ❓ needs decision

---

## 0. Baseline — what already works

- ✅ **Sweep action** (OPFLOW only): one fixed mutation applied per candidate bus, full ACOPF per bus, binary feasibility classification, per-bus cost recorded. Candidate sets: `all_buses`, `load_buses`, `bus_list`.
- ✅ **Add-entity primitives**: add generator (forced-injection default, `Pmin=Pmax=capacity`) and add load (carries both `Pd` and `Qd` — confirmed on prompt 3 with 100 MW / 10 MVar).
- ✅ **Per-candidate infeasibility reason labelling** (`voltage low` / `voltage high` / `line overload` / `did not converge`), validated on 2000-bus data.
- ✅ **Two-table sweep report** (feasible table + infeasible table with reason column + uncertified-iterate footnote).
- ✅ **Parallel execution plumbing** exists (`executor.run_parallel`, thread pool over subprocess solves; distinct negative workdir per candidate).
- 🚧 doing **Parallelization upgrade** (see Track B-2): decoupled worker count, thread pinning, progress, UI controls — *prompt drafted, pending Claude Code run.*

Validated end-to-end so far: **prompts 3, 4** (200-bus and 2000-bus).

---

## Track A — Cover all 20 prompts

### A.1 Coverage map

| # | Lvl | Prompt (short) | Needs | Status |
|---|-----|----------------|-------|--------|
| 3 | 2 | All buses hosting 100 MW/10 MVar load | C0 | ✅ validated |
| 4 | 2 | All buses connecting 100 MW gen | C0 | ✅ validated |
| 14 | 3 | Min-cost **location** for 100 MW load | C0 (sweep + cost reduction) | ✅ should work — unverified |
| 19 | 1 | Reserve margin (ΣPmax all − ΣPmax on) | C8-lite (analyze + arithmetic) | ✅ likely works — unverified |
| 1 | 3 | Max load hosting capacity per bus (hold avg PF) | **C1** | ⛔ |
| 2 | 3 | Max generator MW per bus | **C1** | ⛔ |
| 13 | 3 | Min-cost gen location under economic dispatch | **C2** | ⛔ |
| 15 | 3 | Max voltage step on switching per bus | **C3** | ⛔ |
| 17 | 3 | Reactive adequacy per bus (Qmax at Pmax) | **C3** | ⛔ |
| 16 | 2 | Best AVR regulation bus for 50 MVar gen | **C4** | ⛔ |
| 5 | 4 | N-1 on 3 nearest neighbors after +load | **C5** | ⛔ |
| 6 | 4 | N-1 on 3 nearest neighbors after +gen | **C5** | ⛔ |
| 7 | 4 | …+ ordered relief measures | **C5 + C7** | ⛔ |
| 8 | 4 | …+ ordered relief measures (gen) | **C5 + C7** | ⛔ |
| 9 | 6 | N-2 on 3 nearest neighbors after +load | **C5 + C6** | ⛔ |
| 10 | 6 | N-2 on 3 nearest neighbors after +gen | **C5 + C6** | ⛔ |
| 11 | 6 | N-2 + relief (load) | **C5 + C6 + C7** | ⛔ |
| 12 | 6 | N-2 + relief (gen) | **C5 + C6 + C7** | ⛔ |
| 18 | 5 | Min feasible hot reserve under N-1 | **C8** | ⛔ |
| 20 | 2 | Percent load per transmission corridor | **C8** | ⛔ |

**Fully solvable today:** 3, 4, 14, 19 (4 of 20).
**Remaining 16** collapse into 8 reusable capabilities (C1–C8) below.

### A.2 Capabilities (C1–C8)

#### C1 — Per-candidate boundary search inside the sweep  → unlocks **1, 2**
Nest a bisection *inside each sweep candidate*: instead of testing one fixed size, find the **maximum** injection each bus can host before a limit binds.
- Reuse existing sweep + parallelism; the bisection is the per-candidate inner loop, still embarrassingly parallel across buses.
- **Wrinkle (prompt 1):** hold power factor constant at the **system-average PF**. Compute avg PF once, then scale `Pd` and `Qd` together along that ray during bisection (not P alone).
- Output per bus: max feasible MW (and the binding constraint at that maximum).
- **Leverage:** highest ROI — smallest step, two Level-3 prompts, closest to current code.

#### C2 — Generator cost curve + dispatchable economic mode  → unlocks **13**
Hosting prompts use forced injection; **13 is the deliberate exception** — the OPF must *choose* the new unit's output, so the added generator needs a realistic cost curve (else free energy distorts the dispatch and the location ranking).
- Add a cost curve to the add-generator primitive; allow `dispatchable=true` for this prompt family only.
- Then it's a sweep + cost-minimization reduction over locations.
- Keep the explicit `fixed-injection` vs `dispatchable` wording rule (see X.1) so the mode never flips silently.

#### C3 — Custom per-candidate metric + custom feasibility predicate  → unlocks **15, 17** (supports 13)
Generalize the sweep's "what is measured / what counts as feasible" beyond the fixed V-band + loading check.
- **15:** metric = **max |ΔV| across all buses** when the load is switched in at the candidate bus (candidate-vs-base, system-wide), reduced to the largest step; connection bus held constant.
- **17:** predicate = **reactive adequacy** — a generator at the bus can supply `Qmax` while at `Pmax` (a headroom/capability test, not the standard V/loading feasibility).
- Design as a small, declared metric/predicate interface the sweep can call per candidate.

#### C4 — Remote voltage-regulation (AVR) control + voltage-deviation objective  → unlocks **16**
- New control: set which bus a generator regulates (remote AVR target).
- Search over candidate **regulation buses** for a 50 MVar generator at a given bus, minimizing **Σ |V − 1.0|** over the whole system.
- Self-contained; a sweep over regulation-bus assignment rather than over injection buses.

#### C5 — Topology + N-1 contingency sweep  → unlocks **5, 6** (prereq for 7–12)
The largest single capability; deserves its own design pass.
- **Hop-distance neighbors:** BFS on the network graph to get the "3 nearest neighbor buses by hop count."
- **Contingency generation:** auto-build N-1 sets — branch outage, generator outage, load outage — on those neighbors.
- **Execution:** add the 50 MW load/gen at the target bus, then run each contingency (SCOPFLOW with a generated contingency file, or repeated OPFLOW with the element disabled), classify pass/fail per contingency.
- Output: passed/failed contingency list.

#### C6 — N-2 combinatorial contingencies  → unlocks **9, 10** (prereq for 11, 12)
- Extend C5 to **pairs** of outages over the neighbor set's incident elements.
- Watch combinatorial blow-up; lean on the parallel sweep for the candidate pairs.

#### C7 — Ordered relief-measure loop  → unlocks **7, 8** (with C5), **11, 12** (with C6)
- When a contingency fails, apply remedial actions in **decreasing priority**: (1) transformer ratio change → (2) generator redispatch → (3) line switching → (4) load curtailment.
- Apply iteratively until the contingency passes or measures are exhausted; report which measure resolved it.

#### C8 — System-metric `analyze` extensions  → unlocks **18, 20** (19 already)
- **19 (done-ish):** reserve margin = ΣPmax(all units) − ΣPmax(on units) — arithmetic over generators via `analyze`; *verify on the network.*
- **18:** minimum feasible **hot reserve** under N-1. Hot reserve = Σ(Pmax − Pg) over committed units; minimize subject to all N-1 contingencies feasible → SCOPFLOW + custom system objective. Heaviest of the three.
- **20:** percent load per **transmission corridor**. Define corridors as cut-sets (~4–5 lines linking strongly-connected areas); detect them (graph partitioning / cut-set detection), then report max % loading across each corridor's lines.

### A.3 Dependency notes
- 7, 8 = C5 + C7 · 9, 10 = C5 + C6 · 11, 12 = C5 + C6 + C7. **C5 is the gateway** to the whole 5–12 contingency bloc (8 prompts).
- C2 and C3 both extend the sweep's per-candidate contract (cost curve / custom metric+predicate) — natural to design together.

---

## Track B — Scaling & token economy

#### B.1 — Token-bounded LLM-facing results view  → **HIGH priority (blocks larger networks)** ⛔
- At 2000 buses the collect-all table fed to the LLM hit **~100,903 tokens** (vs ~18k at 200 buses). The next scaling steps (3000+ buses) will approach the context window.
- The LLM's job is **reduction**, not reading N rows. Feed it a summarized view: feasible count, infeasible list with reasons, and a sampled/aggregated table. Keep the **full table in the report/journal only**.
- This is the item that actually gates the larger networks the paper is built around.

#### B.2 — Sweep parallelization upgrade  🚧 doing (prompt drafted)
- Decouple `sweep_max_workers` from `max_variants`; thread-pin each solve (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`) to stop BLAS oversubscription; add a progress callback; UI checkbox + worker-count control.
- **Recommended worker count on the i7-13650HX (14 physical cores / 20 threads): start at 12**, then benchmark 8 / 12 / 14 watching for thermal throttling and memory-bandwidth plateau.
- **Before-baseline to beat:** 2000-bus prompt-3 run = **1593 s** (~1.8× over serial → consistent with oversubscription on the pre-upgrade run).

#### B.3 — Δ-from-base / local loading metric in sweep reporting  ⛔
- On the 2000-bus run, **88% of feasible buses (1682/1921) reported max loading in [92.5, 93.0]** — i.e., pinned at the pre-existing base-case bottleneck (92.7%), which is a spectator unrelated to the injection at most buses.
- The "Max Line Loading" column as shown is misleading. Report **local impact** instead (Δ-loading vs base, or loading on lines incident to the candidate bus). Overlaps with C3 (custom metric).

#### B.4 — Marginal-boundary flag  ⛔ (reproducibility)
- Degenerate boundary points are being split feasible/infeasible only by IPOPT's convergence flag. Example from the 2000-bus run: buses **8042, 6328, 3125, 1089** are labelled *feasible* at `V_min = 0.900` **and** `loading = 100.0` — physically indistinguishable from the "voltage low" *infeasible* buses (also `0.900` / `100.0`); 53 feasible buses sit at ≥99.95% loading, 28 of them with elevated cost (up to bus 8110 at $1,255,333).
- Add a **"marginal"** tag (within ε of any limit) so boundary cases are reported honestly rather than as crisp pass/fail. Strengthens reproducibility claims for the paper.

#### B.5 — Slow infeasible-tail handling  ⛔
- Non-converging 2000-bus solves run to IPOPT's iteration limit and inflate wall time independent of worker count (79 infeasible + 2 hard failures at buses 1010, 1055).
- Consider a **tighter per-sweep solve timeout / iteration cap** distinct from the 600 s global timeout, so a few bad buses don't serialize the sweep.

#### B.6 — Determinism validation  ⛔ (paper hygiene)
- Confirm the feasible/infeasible split and per-bus numbers are **identical across repeated runs and independent of worker count** — the question a reviewer will ask. Run after B.2 lands.

---

## X. Cross-cutting / smaller items

- **X.1 — Explicit mode wording + system-prompt mapping rule** ❓: keep `fixed-injection` (Pmin=Pmax=cap) vs `dispatchable` (Pmin=0, Pmax=cap) explicit in prompt wording so the LLM never silently flips the answer (the 160/40 ↔ 200/200 confound). Planned system-prompt rule to map phrasing → mode.
- **X.2 — Verify prompts 14 & 19** actually resolve with current capabilities (both expected-solvable but unrun).
- **X.3 — `env_setup.sh` secrets** ❓: live Anthropic/OpenAI keys are in the file; confirm git history is clean (`git log -p -S 'sk-ant' --all`) and keep them out of any tracked/shared/student-image location.

---

## Suggested sequencing (slices)

Slice 1 (✅ done): add-entity primitives + binary-feasibility sweep + sweep report.

| Slice | Content | Unlocks | Notes |
|-------|---------|---------|-------|
| **B-early** | B.1 token-bounded view + finish B.2 parallelization | — | Do first; gates scaling, which is half the paper |
| **2** | **C1** per-candidate boundary search | 1, 2 | Smallest step, highest ROI; reuses sweep + parallelism |
| **3** | **C2** cost curve + **C3** custom metric/predicate | 13, 15, 17 | Both extend the sweep's per-candidate contract |
| **4** | **C4** AVR regulation control | 16 | Self-contained |
| **5** | **C5** topology + N-1 contingency sweep | 5, 6 | Largest; own design pass |
| **6** | **C6** N-2 contingencies | 9, 10 | Builds on C5 |
| **7** | **C7** ordered relief loop | 7, 8, 11, 12 | Builds on C5/C6 |
| **8** | **C8** hot reserve (18) + corridor detection (20) | 18, 20 | 19 already |
| **ongoing** | B.3 local metric, B.4 marginal flag, B.5 tail timeout, B.6 determinism | — | Reporting honesty + paper hygiene |

**Coverage trajectory:** 4 → (Slice 2) 6 → (Slice 3) 9 → (Slice 4) 10 → (Slice 5) 12 → (Slice 6) 14 → (Slice 7) 18 → (Slice 8) 20 of 20.

---

## Open decisions for you / Slaven

1. Confirm slice ordering — in particular whether B.1 (token view) jumps ahead of C1, given it gates the scaling half of the paper.
2. Prompt 18 (min hot reserve under N-1) is the heaviest single item (SCOPFLOW + custom minimization) — is it in-scope for this paper, or deferred?
3. Corridor definition for prompt 20 — confirm the cut-set definition and whether an automated detector is acceptable vs. a supplied corridor list.
