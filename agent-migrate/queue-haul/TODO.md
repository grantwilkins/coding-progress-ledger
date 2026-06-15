# Queue-Haul Dispatch — TODO

Power-first job dispatch: hit a grid shed target $S^\star$ by moving jobs, at least disruption.
Static snapshot, one source pool → one destination pool, absolute watts via parameter sweep.

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