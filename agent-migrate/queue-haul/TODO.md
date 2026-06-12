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

- [ ] **T3 — Per-job impact & move costs** (`impact.py`)
  ΔP_j bracket `[s_plat·ℓ_j, p̄·ℓ_j]` in **two-price form** (`p̄_pre·ℓ_pre + p̄_dec·ℓ_dec`), or `μ·m_j` in the memory regime; disruption costs `c_j(R)`, `c_j(S)`. **Replay rebuild cost uses `ρ_dest(T_j)`** — the prefill roofline *as a function of context length* (flat below `T*≈29k`, decaying `1/T` above), not a constant rate.
  *Needs: **T1 prices** (`p̄, s_plat, μ, p̄_pre, p̄_dec`) + T2 loads + §6 Movement.*
  **Success:** replay cost is flat for short-context jobs and rises ~`1/ρ_dest(T)` for long ones (not constant); two-price ΔP_j differs from single-price for phase-skewed (agentic) jobs and matches for chat.

- [ ] **T4 — Dispatch solver** (`dispatch.py`)
  **Two solves, not a branch.** Primary: LP over `y ∈ [0,1]` minimizing `Σ y_j·c_j` s.t. `Σ y_j·ΔP_j ≥ S*` + source egress + destination rebuild by `D` (prefill via `ρ_dest(T)`, ingest via `μ_in`) + destination headroom (load `≤ L̄_dest`, held sessions `≤ S̄_dest`). If infeasible: **re-solve** maximizing `Σ y_j·ΔP_j` with the `≥ S*` constraint dropped, report shortfall. Plus the bang-per-buck greedy (sort by `c_j/ΔP_j`).
  *Needs: T3 + §5 Pools & event.*
  **Success:** LP output satisfies every constraint; on a feasible `S*` it sheds exactly `S*` (no over-shed); on an infeasible `S*` it returns the max-shed plan and a correct shortfall.

- [ ] **T5 — Experiment: greedy vs LP**
  **Success:** greedy and LP coincide (same job set, same shed) when no resource constraint binds; the gap appears only at/after the constraint boundary, and is bounded by the one boundary-crossing job.

- [ ] **T6 — Experiment: certify low, report high**
  Feasibility under guaranteed prices (`s_plat`) vs expected shed under amortized prices (`p̄`), swept over `S*`.
  **Success:** every `S*` certified feasible under `s_plat` is also met under `p̄`; the reported gap between guaranteed and expected shed tracks the bracket ratio (≈30× dense center).

- [ ] **T7 — Experiment: §6.2 sensitivity sweeps**
  Sweep `ρ*`, MFU (drives `ρ_dest`), bracket ratio `p̄/s_plat`.
  **Success:** the *job ranking* is flat across all three sweeps; the *feasibility margin* moves monotonically with them. (Selection robust, absolute shed sensitive.)

- [ ] **T8 — Experiment: load vs memory regime**
  Walk the regime boundary two ways: (a) idle/cold fraction × γ, and **(b) the context-length mixture short→long** (long `T` is what makes `m_j` bind — the KV-size approach to the same transition).
  **Success:** below the crossover, `ℓ`-ranking governs and `μ·m` is slack; above it, `μ·m`-ranking takes over; the switch occurs exactly at the `N = max(L/ρ*, S_held/S_node)` crossover, and approaches (a) and (b) agree on where.

- [ ] **T9 — Tests**
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