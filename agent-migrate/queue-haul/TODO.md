# Queue-Haul Dispatch — TODO

Power-first job dispatch: hit a grid shed target $S^\star$ by moving jobs, at least disruption.
Static snapshot, one source pool → one destination pool, absolute watts via parameter sweep.
(Phase 3 restores multi-destination: one source → K destinations.)

**The math is in `formulation.md`** — tasks below implement it; values come from `assumptions.md`.
**Order:** T1 and T2 are independent and first. T3 needs T1+T2. T4 needs T3. T5–T9 need T4.

Each task lists its **success criterion** — the one check that says it's done.

---

## Tasks

- [x] **T1 — Pool & power model** (`power.py`)
  Compute the scalar prices the dispatch consumes: `p̄ = P_busy/ρ*` (amortized), `s_plat = p̄ / bracket_ratio` (guaranteed plateau slope), two-price split `(p̄_pre, p̄_dec)` with `p̄_dec ≈ 5·p̄_pre` per busy-second, memory price **`μ = P_idle/S_node`** (ρ_low eliminated — a memory-bound node sits at idle by definition), and the regime test `N = max(L/ρ*, S_held/S_node)`.
  *The ramp-plateau node curve and `power knee` are for the validation plot only — the solver never evaluates the curve, just the prices above. Don't build a piecewise curve the dispatch won't call.*
  *Needs: §2 Node power, §4 Capacity.*
  **Success:** at center values, `p̄ ≈ 10.5 kW/node-unit`, `s_plat ≈ 350 W`, `μ ≈ 208 W/held-session` (BF16); ranking by `p̄·ℓ` is unchanged when `p̄` is scaled by any constant.

- [ ] **T2 — Job generator** (`instance.py`)
  Sample `T_j` (swept context mixture, BF16/FP8 toggle), state ∈ {active, idle-warm, cold}, turn rate, tokens/turn (geometric `Y`, log-normal `Δ`, sampled independently). Emit **the two load components separately**: `(ℓ_pre_j = f_j/F, ℓ_dec_j = g_j/G)`, with `ℓ_j` their sum; plus `m_j = η·T_j`.
  *Enforce the invariant: active jobs have `ℓ_j > 0` and sit in resident HBM; idle/cold jobs have `ℓ_j ≈ 0`; cold jobs sit in the paged tier (γ) and never carry a nonzero rate.*
  *Needs: §1 Model constants, §3 Rate distributions.*
  **Success:** generated population reproduces the target `E[T]`, state mix, and agentic:chat split; no cold job has `ℓ_j > 0`; `(ℓ_pre, ℓ_dec)` returned separately, not pre-summed.

- [x] **T3 — Per-job impact & move costs** (`impact.py`)
  ΔP_j bracket `[s_plat·ℓ_j, p̄·ℓ_j]` in **two-price form** (`p̄_pre·ℓ_pre + p̄_dec·ℓ_dec`), or `μ·m_j/m̄ = μ·T_j/E[T]` in the memory regime (μ stays W/session — m_j normalized to session-equivalents). Disruption costs `c_j(R)`, `c_j(S)` with the queue wait as a **split congestion multiplier** `(1+φ_pre)`/`(1+φ_in)` on the rebuild — replay against destination prefill load, transfer against ingest (φ_in≈0). **Replay rebuild cost uses `ρ_dest(T_j)`** — the prefill roofline *as a function of context length* (flat below `T*≈29k`, decaying `1/T` above), not a constant rate. T3 is a pure per-job calculator + the one pool-level `regime` flag (T4 picks the action); `b_j` egress bytes also produced here.
  *Needs: **T1 prices** (`p̄, s_plat, μ, p̄_pre, p̄_dec`) + T2 loads (`mfu` stored on the population) + §6 Movement.*
  **Success:** replay cost is flat for short-context jobs and rises ~`1/ρ_dest(T)` for long ones (not constant); two-price ΔP_j **deviates from single-price with opposite sign by phase skew** — prefill-skewed (non-reasoning agentic) below single, decode-skewed (chat, reasoning) above — and equals single only for a phase-balanced job (closure of T1's `(p̄_pre+p̄_dec)/2=p̄` split).

- [ ] **T4 — Dispatch solver** (`dispatch.py`)
  **Two solves, not a branch.** Primary: LP over `y ∈ [0,1]` minimizing `Σ y_j·c_j` s.t. `Σ y_j·ΔP_j ≥ S*` + source egress + destination rebuild by `D` (prefill via `ρ_dest(T)`, ingest via `μ_in`) + destination headroom (load `≤ L̄_dest`, held sessions `≤ S̄_dest`). If infeasible: **re-solve** maximizing `Σ y_j·ΔP_j` with the `≥ S*` constraint dropped, report shortfall. Plus the bang-per-buck greedy (sort by `c_j/ΔP_j`).
  *Needs: T3 + §5 Pools & event.*
  **Success:** LP output satisfies every constraint; on a feasible `S*` it sheds exactly `S*` (no over-shed); on an infeasible `S*` it returns the max-shed plan and a correct shortfall.

- [x] **T5 — Experiment: random vs greedy vs LP**
  Three policies, all respecting the same movement budgets (egress/rebuild/headroom drawn down over the deadline window): **random** (shuffle movable jobs, coin-flip action), **greedy** (decentralized first-fit — each job self-selects its cheaper action, best-deal jobs first), **LP/MILP** (coordinated optimum). The greedy is no longer resource-blind; it cannot ship more than the links carry.
  **Success:** where every policy is feasible, greedy lies exactly on the LP cost frontier (same shed, same downtime) and random sits above it (more downtime per watt). The policies separate by their **shed ceiling** — random < greedy < LP — because selection (greedy over random) and then global repacking transfers→replays (LP over greedy) each fit more shed under the same links.
  *Criterion revised: the greedy↔LP gap is the **value of central coordination** (≈2× shed at center), **not** "bounded by the one boundary-crossing job." That earlier bound assumed a single-constraint fractional knapsack; once the baseline respects all five resource budgets, the gap is a reach/ceiling gap that can exceed one job. LP max-shed bounds every policy.*

- [x] **T6 — Experiment: certify low, report high**
  Feasibility under guaranteed prices (`s_plat`) vs expected shed under amortized prices (`p̄`), swept over `S*`.
  **Success:** every `S*` certified feasible under `s_plat` is also met under `p̄`; the reported gap between guaranteed and expected shed tracks the bracket ratio (≈30× dense center).

- [x] **T7 — Experiment: §6.2 sensitivity sweeps**
  Sweep `ρ*`, MFU (drives `ρ_dest`), bracket ratio `p̄/s_plat`.
  **Success:** the *job ranking* is flat across all three sweeps; the *feasibility margin* moves monotonically with them. (Selection robust, absolute shed sensitive.)

- [x] **T8 — Experiment: load vs memory regime**
  Walk the regime boundary two ways: (a) idle/cold fraction × γ, and **(b) the context-length mixture short→long** (long `T` is what makes `m_j` bind — the KV-size approach to the same transition).
  **Success:** below the crossover, `ℓ`-ranking governs and `μ·m` is slack; above it, `μ·m`-ranking takes over; the switch occurs exactly at the `N = max(L/ρ*, S_held/S_node)` crossover, and approaches (a) and (b) agree on where. **Also certify ranking *stability* across the boundary** (not just crossover location): report the Spearman of the load-ranking (ΔP_expected) vs the memory-ranking (ΔP_memory) — at center it is ≈0 (T2 draws T independent of Δ/Y), so the regime flag genuinely reorders which jobs are shed, a substantive result.

- [x] **T9 — Tests**
  **Success (all pass):** (1) ranking invariant under `p̄` scaling; (2) regime switch lands at the `N = max(·,·)` crossover; (3) greedy = LP away from boundaries; (4) every solver output satisfies all constraints; (5) no cold job carries load; (6) BF16↔FP8 toggle shifts `S_node` ~2× and the memory threshold, leaving load-regime results unchanged.

---

## Phase 2 — Reconstruction DES (§10.2 validation)

The LP enforces the five movement constraints as volume budgets; it never models
execution order, pipeline fill, or the largest-job effect. These tasks replay a
solved `Plan` through a deterministic 2-machine flow shop (one shared egress link →
W rebuild servers). In this parameterization the precedence/pipeline-fill gap is
near-null (ingest non-binding, replay stage-1 tiny); the gaps that exist are the
link-discipline realized-shed gap and the W-bound prefill packing gap.

- [x] **T10 — Reconstruction DES** (`simulate.py`)
  A deterministic event-loop **checker** (no SimPy, no job selection): consume a solved `Plan` and replay it as a 2-machine flow shop — Stage 1 one shared egress link at `λ_src` serializing every moving job, Stage 2 `W` parallel prefill servers (replay, held `T_j/ρ_dest(T_j)`) and `W` ingest channels at `μ_in` (transfer). τ offsets enter as one-time per-stage availability (`link@τ_src`, `prefill@τ_pre`, `ingest@τ_in`) so a single-isolated-resource plan reproduces the LP budget at equality. Two pipeline modes (store-and-forward default, cut-through knob); four link disciplines — FIFO, LPT, Johnson, and **power-density-descending** (`bind_dp·y/p1`, a value-density heuristic — strong but not optimal for realized shed, which is a knapsack-prefix problem). Emit per-job egress/rebuild times, **two metrics** — realized shed (egress done ≤ D) and reconstruction success (rebuild done ≤ D) — makespan, and the analytic envelope `[lb, ub]`.
  *Stage-2 service uses the **bare** rates `T/ρ_dest` and `η·T/μ_in`, NOT the `(1+φ)`-inflated `c_replay`/`c_transfer` — the DES models contention explicitly via finite servers, so reusing `c_*` double-counts the queue wait.*
  *Needs: T4 `Plan` + §6 Movement + §5 Pools & event.*
  **Success:** on a single-isolated-resource plan the DES completion matches the LP budget at equality to `rel=1e-9`; on a `W=1` single-action path the makespan equals Johnson's-rule makespan; `lb ≤ makespan ≤ ub` always; realized shed ≤ certified `shed_guaranteed`.

- [x] **T11 — Experiment: realized vs certified shed** (`plot_simulate_validation.py`)
  Solve a feasible LP plan, replay it, and point the sweeps at the gaps that exist. **Primary (grid relief):** realized shed vs discipline {FIFO, LPT, Johnson, PD} as transfer-load on the shared link rises (transfer-fraction / `η·T` CoV) — show PD banks more watts by D because it refuses to let big transfers starve high-density replays. **Secondary (service continuity):** reconstruction shed and makespan vs **W** and **prefill-time CoV** (`T/ρ_dest(T)` among replays) — the `P||Cmax` packing gap, largest at small W and high CoV. **Companion:** makespan inside `[lb, ub]`, Johnson exact on the `W=1` single-action sub-sweep. Report the **nulls** honestly: S&F vs cut-through spread ≈0, and the reconstruction gap *shrinking* with transfer-fraction — each explained by the envelope.
  *Needs: T10 + T4.*
  **Success:** discipline swings realized shed when the link binds (LPT can collapse to ~0) and agrees when it's slack, quantifying the watts the LP's order-blindness leaves on the table — no single discipline universally dominates (realized shed is a knapsack-prefix problem); the reconstruction gap grows as W↑/CoV↑ (the perfect-packing budget gets optimistic) and is bracketed by `[lb, ub]`; the pipeline-fill spread is ≈0 with the envelope showing why. The result either cashes §10.2's exactness claim (gap null where stages don't overlap) or justifies widening `T_lat`/packing slack on the prefill row.

- [x] **T12 — DES tests** (`tests/test_simulate.py`)
  **Success (all pass):** (1) single-isolated-resource equality — egress-only, prefill-only (`W=1`), ingest-only plans each match the LP budget at equality to `rel=1e-9`; (2) 2–3-job hand-computed precedence cases (store-and-forward and cut-through); (3) conservation — `rebuild_done ≥ egress_done`, serial link never overlaps, `lb ≤ makespan ≤ ub` (corrected `ub=τ_src+Σp1+Σp2`), realized shed ≤ certified; (4) `Johnson makespan == DES makespan` on `W=1` single-action paths only; (5) cut-through ≤ store-and-forward completion per job; (6) realized shed is discipline-invariant when the link is slack and discipline-sensitive when it binds (no discipline universally dominates); (7) a split job (y_R,y_S>0) rebuilds both fractions — neither stage-2 piece is dropped.

---

## Phase 3 — Multi-destination routing (§4 ℓ-index restored)

The single-pool dispatch collapsed §4's destination index. These tasks restore ℓ, making the
source uplink a SHARED constraint across destinations — the one row that couples K otherwise-
independent dispatches — and verify the LP's routing holds under execution. Drop that single
egress row and the program separates into K independent single-destination dispatches; that row
is the entire multi-destination contribution (a transportation LP with one global uplink knapsack).
Two forks locked: **(a)** 2-stage DES, `Λ_ℓ ≥ λ_src` so the per-destination downlink never binds
and pipeline-fill stays the T11 near-null (the 3-stage downlink variant is deferred); **(b)**
fractional `y[j,ℓ]` reads as the routing *distribution* of a class-representative's sessions, so
verification is per-destination volume conservation.

- [x] **T13 — Multi-destination dispatch** (`dispatch.py`, `impact.py`)
  Re-index `y_R`, `y_S` by destination ℓ (`2·n·K` vars; uniform 2D `Plan`, `(n,K)`). Pairing
  blocks over ℓ (`Σ_ℓ (y_R+y_S)[j,ℓ] ≤ 1`); shed stays ℓ-free (`dp` broadcast); **ONE shared
  egress row** `Σ_{j,ℓ}(b_R·y_R+b_S·y_S)[j,ℓ] ≤ λ_src(D−τ_src)`; prefill/ingest/load/held block
  per-ℓ with ℓ-dependent `ρ_ℓ` and `φ_pre,ℓ`/`φ_in,ℓ`. New `DestFleet(W_ℓ, spare_ℓ, mfu_ℓ,
  dest_prefill_util_ℓ)` (length-K); `solve(..., fleet=None)` → K=1 from `Event`/`Movement`, so
  existing callers and Phase-1/2 tests stay green. `λ_src`/`μ_in`/`τ` shared. Costs in a new
  `impact.move_costs(pop, fleet, move)` → n×K `c_R`/`c_S`/`reb`; `impact.compute()` untouched
  (it emits only the ℓ-free `dp`/`regime`/`b_*`). Rebuild `ρ_ℓ` decouples from `pop.mfu` (source
  builds `ℓ_pre` with `pop.mfu`; destination rebuilds with `ρ_ℓ`). Switch the LP path to
  `scipy.optimize.linprog(method='highs')` (assemble `A_ub`/`b_ub`); report `θ_egress` (shared-
  egress dual) and `θ_admit,ℓ` (per-ℓ held/load duals) from `result.ineqlin.marginals`, taken on
  the **max-shed LP** (maximize `Σ dp·y`, the existing re-solve path — the S*-independent capacity
  price). `milp` stays for the integer path.
  *Keep egress a single row — that IS the multi-destination structure; everything else is K
  independent blocks. `dp` and `bind_dp` are unchanged (shed is destination-independent).*
  *Needs: T4 + §4 destination index + the `DestFleet` descriptor.*
  **Success:** with K=1 the plan equals the existing single-dest solve to `rel=1e-9`; with K>1 and
  homogeneous destinations any split is optimal (cost-tied); with heterogeneous `spare_ℓ` the LP
  concentrates on cheapest-reachable until `θ_admit,ℓ>0`, then spreads; `θ_egress>0` exactly when
  `Σ_ℓ spare_ℓ` exceeds what the uplink can feed.

- [ ] **T14 — Figure 1: θ_egress K-sweep (model validation, no DES — the gate)**
  (`plot_multidest_dual.py`)
  Cheap — no DES, no MILP. K-sweep at fixed total shed-demand: as K grows, per-destination
  problems slacken but the shared uplink tightens. Show `θ_egress` (max-shed dual) rising and
  routing concentrating onto cheapest-reachable sites; the binding constraint visibly migrating
  admission→egress is the single clearest statement that the one shared row is the whole story.
  **Saturation-band sizing is MANDATORY** (the most common way this experiment silently produces
  a non-result): size `Σ_ℓ spare_ℓ ∈ [1.1, 1.5]×` shed-demand with `max_ℓ spare_ℓ < demand`, and
  sweep through the band — don't pick a point. Produce this first; it gates everything.
  *Needs: T13.*
  **Success:** `θ_egress=0` across the slack region and rises monotonically as the uplink binds
  within the saturation band, with routing concentration tracking the `θ_admit,ℓ` ordering. If
  `θ_egress` never goes positive across the band the coupling isn't biting — resize the fleet
  before spending any DES effort.

- [ ] **T15 — Per-ℓ DES extension** (`simulate.py`)
  Extend the 2-stage flow shop: one shared egress link serializing **all** moving `(j,ℓ)`
  shipments regardless of destination (the coupling made executable), then K parallel rebuild
  clusters each running the existing `W_ℓ`-server prefill/ingest machinery with per-ℓ `ρ_ℓ`.
  Consume the uniform 2D `Plan`. 2-stage only (`Λ_ℓ ≥ λ_src`; no downlink stage — pipeline-fill
  stays the T11 near-null, don't reopen). Two new invariants: per-ℓ realized load `≤ L̄_dest,ℓ`
  (the dynamic §6.2 admission check) and realized routing distribution matches certified among
  by-D jobs.
  *Stage-2 still uses the bare rates (`T/ρ_ℓ`, `η·T/μ_in`), not the `(1+φ)`-inflated costs.*
  *Needs: T13 + T10.*
  **Success:** per-ℓ realized load never exceeds `L̄_dest,ℓ`; on a slack plan realized routing ==
  certified within by-D jobs; K=1 reproduces T10's single-dest DES to `rel=1e-9`; `lb ≤ makespan ≤ ub`.

- [ ] **T16 — Figure 2: realized vs certified routing under uplink contention** (the headline)
  (`plot_multidest_validation.py`)
  Replay each plan through the per-ℓ DES across the saturation band. Sweep: (1) K at fixed total
  demand (binding constraint migrates admission→egress, `θ_egress` rises, routing concentrates);
  (2) `spare_ℓ` heterogeneity; (3) `ρ_ℓ` heterogeneity (Hopper/Blackwell mix). Report realized vs
  certified routing distribution and per-ℓ realized load vs `L̄_dest,ℓ`.
  *Needs: T15 + T13 (+ T14 gate passed).*
  **Success:** realized routing matches certified within jobs clearing by D; no destination
  over-admitted under execution; where the uplink binds, some ℓ-bound transfers miss D → that ℓ
  under-fills → realized shed < certified — the multi-destination "certify low, report high,"
  attributable to uplink contention. Null gap → certification robust, §4 routing exact; non-null
  under tight uplink → the contention result. Either outcome publishes.

- [ ] **T17 — Multi-destination tests** (`tests/test_multidest.py`)
  **Success (all pass):** (1) K=1 reduces to single-dest solve at `rel=1e-9`; (2) homogeneous-
  destination split is cost-indifferent (any feasible routing optimal); (3) shed is invariant to
  routing (`Σ dp·y` unchanged across destination permutations); (4) `θ_egress=0` when uplink
  slack, `>0` when binding (max-shed dual); (5) per-ℓ DES never exceeds `L̄_dest,ℓ`; (6) realized
  routing == certified within by-D jobs on a slack plan.

---

## Corrections applied (vs. prior TODO)

1. **ρ_low eliminated** — T1 memory price is `μ = P_idle/S_node`, no `π(ρ_low)`.
2. **Power knee is plot-only** — T1 computes prices, not a piecewise node curve the solver won't call.
3. **ρ_dest(T) is a function** — T3 replay cost uses the per-session prefill roofline, not a constant.
4. **T2 emits `(ℓ_pre, ℓ_dec)` separately** — the two-price form in T3 needs the split, not the sum.
5. **T3 depends on T1's prices** — annotation corrected from "§6 only" to "T1 prices + §6".
6. **T4 fallback is a second solve** — not an exception handler on the first.

---

## Deferred (out of scope for v1)

- Re-solve dynamics over the hold window (cadence, sessions going cold mid-hold, autoscaler lag).
- Multi-pool fleet ranking by `p̄·ℓ`.
- Prefill/decode disaggregation.
- Real-trace replay validation.
- Per-destination downlink (`Λ_ℓ < λ_src`): the 3-stage pipeline that reopens pipeline-fill;
  named extension — Phase 3's headline assumes `Λ_ℓ ≥ λ_src` (uplink binds).
- Routing-integer MILP (one destination per job): the crisp per-job "this job went to ℓ" claim
  with `realized destination == certified destination` success; Phase 3 reads fractional
  `y[j,ℓ]` as a routing distribution verified by per-destination volume conservation.