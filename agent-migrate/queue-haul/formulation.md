# Queue-Haul Dispatch — Formulation

Stripped to what the code implements. Scope per TODO.md: one source pool → one destination,
static snapshot, two-price form, power parameters swept in absolute watts (assumptions.md).

## Assumptions

- **A1** All power quantities are few-second averages.
- **A2** One pool of identical nodes; an autoscaler holds active nodes near setpoint ρ*. We model its consequence (pool power follows load), not node on/off.
- **A3** A job is prefilling, decoding, or idle. Tool calls, think-time, dormancy = idle: off-GPU, drawing nothing, holding only KV.
- **A4** Per-job loads add (first-order; the headroom 1−ρ* absorbs the error).
- **A5** Held KV draws no power; it is a capacity constraint only. Cold (paged-out) sessions count toward held capacity at a discount via uplift γ.
- **A6** Two move primitives: replay (ship context tokens, re-prefill) or KV transfer (ship KV bytes, skip prefill).

## Job model

Two numbers per job, measured over the hold window conditioned on current state:

$$\ell_j = \frac{f_j}{F} + \frac{g_j}{G} \in [0,1], \qquad m_j = \eta\, T_j$$

- f_j = expected prefill tok/s (turn rate × input tok/turn), g_j = expected decode tok/s (turn rate × output tok/turn, incl. reasoning).
- F, G = node's sustained per-phase tok/s **at the latency knee**, so ℓ = 1 is the operational ceiling.
- Idle jobs: ℓ_j → 0, m_j unchanged. Cold sessions need no separate flag — their measured rate is ~0, so ℓ_j → 0 automatically; they are simultaneously cheap to keep and cheap to move.

## Pool power model

**Node curve: ramp-then-plateau.** Power climbs from P_idle to P_busy as ℓ rises to the **power knee**, then is flat. (Not linear-in-load: dense large models are near-step.) Latency departs at a separate, later **latency knee**; ρ* sits just below the latency knee.

**Two prices for removed load:**

- s_plat — plateau slope, marginal W per unit load on a node that stays on (**guaranteed**: realized even if no node drains);
- p̄ = P_busy/ρ* — amortized W per unit load when the autoscaler converts removed load into drained nodes (**expected**).

The ratio p̄/s_plat is a per-node-type parameter (swept: ~3–5× MoE … ~58× dense 405B).

**Pool average:** P(L) = P_floor + p̄·L while the autoscaler holds ρ*.

**Two-price split** (operative here; workload is phase-skewed agentic): prices p̄_pre, p̄_dec per unit of f/F and g/G respectively, with per-token prefill/decode cost ratio c₁/c₂ ≈ 0.10.

**Capacity / memory regime.** Per-node held sessions:

$$S_{\text{node}} = (1+\gamma)\, S_{\text{node}}^{\text{resident}}, \qquad S_{\text{node}}^{\text{resident}} = \mathrm{Cap}/\bar m, \qquad \bar m = \eta\,\bar T$$

Resident bounds what can run; (1+γ)× resident bounds what can be held. Node count:

$$N = \max\big(L/\rho^\star,\; S_{\text{held}}/S_{\text{node}}\big)$$

Load sets N when busy; held sessions set it when idle. When memory binds, the marginal power of holding a session is μ = π(ρ_low)/S_node W/session. The pool is in exactly one regime at a time (a single comparison).

## Per-job power impact

$$\Delta P_j \in \big[\, s_{\text{plat}}\,\ell_j,\;\; \bar p_{\text{pre}} f_j/F + \bar p_{\text{dec}} g_j/G \,\big] \;\text{(load-bound)}, \qquad \Delta P_j = \mu\, m_j \;\text{(memory-bound)}$$

Low end guaranteed, high end expected (autoscaler drains within the hold). Both ends additive over a removed set. Ranking by ℓ is identical at either end — *which* jobs to move doesn't depend on autoscaler timing, only *how many*. Within one node type, ℓ ranks; across node types, p̄·ℓ (watts) ranks.

## Dispatch program

Grid command: shed S* by deadline D, held over [D, D+H]. Decisions: y_j ∈ [0,1], action a_j ∈ {R (replay), S (transfer)}.

$$\min_{y,a} \sum_j y_j\, c_j(a_j) \quad \text{s.t.} \quad \sum_j y_j\, \Delta P_j \ge S^\star$$

**Disruption cost (downtime = ship + rebuild + destination queueing wait):**

$$c_j(R) = \frac{\beta T_j}{\Lambda} + \frac{T_j}{\rho_{\text{dest}}} + w_{\text{pre}}, \qquad c_j(S) = \frac{\eta T_j}{\Lambda} + \frac{\eta T_j}{\mu_{\text{in}}} + w_{\text{in}}$$

**Movement constraints:**

1. Source egress: $\sum_j y_j\, b_j(a_j) \le \Lambda_{\text{src}} (D - \tau_{\text{src}})$, with $b_j(R) = \beta T_j$, $b_j(S) = \eta T_j$.
2. Destination rebuild by D: $\sum_{a_j=R} y_j T_j / \rho_{\text{dest}} \le W (D - \tau_{\text{pre}})$ and $\sum_{a_j=S} y_j\, \eta T_j \le W \mu_{\text{in}} (D - \tau_{\text{in}})$.
3. Destination headroom: $\sum_j y_j \ell_j \le \bar L_{\text{dest}}$ (load below its knee) and $\sum_j y_j \le \bar S_{\text{dest}}$ (held capacity, incl. (1+γ) uplift).
4. Optional service floor: pinned classes get y_j = 0.

If S* is unreachable, re-solve maximizing $\sum_j y_j \Delta P_j$ and report the shortfall.

**Certify low, report high.** Feasibility of S* is certified against guaranteed prices (s_plat·ℓ_j, plus μ·m_j where memory binds); expected shed is reported against amortized prices (p̄·ℓ_j).

**Solution structure.** Linear objective and target ⇒ bang-per-buck: take jobs in order of c_j/ΔP_j (seconds of downtime per watt shed) until S* is met, deviating only where a movement constraint binds. LP relaxation and greedy coincide except at resource boundaries. A sort plus a small LP.

## Sensitivity claims the experiments test

- Within-pool job ranking is invariant to ρ*, F, G, p̄ (common scaling cancels); only the cross-pool comparison and absolute feasibility margin depend on their values.
- The bracket width p̄/s_plat measures how much shed value depends on nodes actually draining.
- Regime switch: idle/cold-heavy populations move from ℓ-ranking to m-ranking; γ lowers μ and delays the switch.

## Deliberately cut (reattachable without changing the above)

Prefill/decode disaggregation, instance integrality, session classes / Markov chains, lexicographic objectives, re-solve dynamics.
