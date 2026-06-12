# Queue-Haul Dispatch — TODO

Power-first job dispatch: hit a grid shed target $S^\star$ by moving jobs, at least disruption.
Static snapshot, one source pool → one destination pool, absolute watts via parameter sweep.

**Order:** T0 first. T1 and T2 are independent. T3 needs T1+T2. T4 needs T3. T5–T8 need T4.

## Tasks

- [ ] **T0 — Fill assumptions.md** (Grant). Every `??` must have a value and a source before T2+ runs.

- [ ] **T1 — Pool & power model** (`power.py`)
  Ramp-plateau node curve from (P_idle, P_busy, power knee); pool price p̄ = P_busy/ρ*; plateau slope s_plat; two-price split (p̄_pre, p̄_dec); memory price μ = π(ρ_low)/S_node; regime test N = max(L/ρ*, S_held/S_node).
  *Needs: §2 Node power, §4 Capacity.*

- [ ] **T2 — Job generator** (`instance.py`)
  Sample T_j (context mixture), state ∈ {active, idle, cold}, turn rate, tokens/turn → f_j, g_j, ℓ_j = f/F + g/G, m_j = ηT_j.
  *Needs: §1 Model constants, §3 Rate distributions.*

- [ ] **T3 — Per-job impact & move costs** (`impact.py`)
  ΔP_j bracket [s_plat·ℓ_j, p̄·ℓ_j] (two-price form), μ·m_j in memory regime; disruption c_j(R), c_j(S).
  *Needs: §6 Movement.*

- [ ] **T4 — Dispatch solver** (`dispatch.py`)
  LP over y ∈ [0,1] + bang-per-buck greedy (sort by c_j/ΔP_j); constraints: source egress, destination rebuild by D (prefill + ingest), destination headroom (load + held sessions); infeasible → maximize shed, report shortfall.
  *Needs: §5 Pools & event.*

- [ ] **T5 — Experiment: greedy vs LP** — where they coincide, gap at constraint boundaries.
- [ ] **T6 — Experiment: certify low, report high** — feasibility under guaranteed prices vs expected shed under amortized prices, across S*.
- [ ] **T7 — Experiment: §6.2 sweeps** — ρ*, F/G, bracket ratio p̄/s_plat; show ranking invariance vs feasibility-margin dependence.
- [ ] **T8 — Experiment: load vs memory regime** — sweep idle/cold fraction × γ; where μ·m ranking takes over from ℓ ranking.

- [ ] **T9 — Tests**
  Ranking invariant under p̄ scaling; regime switch at the N = max(·,·) crossover; greedy = LP away from boundaries; solver output satisfies all constraints.

## Deferred (out of scope for v1)

- Re-solve dynamics over the hold window (cadence, sessions going cold mid-hold, autoscaler lag).
- Multi-pool fleet ranking by p̄·ℓ.
- Prefill/decode disaggregation.
- Real-trace replay validation.
