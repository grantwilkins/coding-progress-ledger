# Queue-Haul Dispatch — Formulation

The grid (or a power cap) asks an inference cluster for a **demand-response event**: shed `S*` watts
by deadline `D` and hold the reduction over `[D, D+H]`. We hit it by **migrating live sessions** off
the source pool to other sites — replaying their context or shipping their KV cache — choosing *which*
jobs and *how* to move them at **least disruption**. Static snapshot, one source pool → `K`
destinations, coupled only by the source uplink. Power parameters swept in absolute watts
(`assumptions.md`). This file is exactly what `power.py` / `instance.py` / `impact.py` / `dispatch.py`
implement; the DES in `simulate.py` (§10.2) replays a solved plan to check it under execution.

## Assumptions

- **A1** All power quantities are few-second averages.
- **A2** One pool of identical nodes; an autoscaler holds active nodes near setpoint ρ*. We model its consequence (pool power follows load), not node on/off.
- **A3** A job is prefilling, decoding, or idle. Tool calls, think-time, dormancy = idle: off-GPU, drawing nothing, holding only KV.
- **A4** Per-job loads add (first-order; the headroom 1−ρ* absorbs the error).
- **A5** Held KV draws no power; it is a capacity constraint only. Cold (paged-out) sessions count toward held capacity at a discount via uplift γ.
- **A6** Two move primitives: **replay** (ship context token-IDs, re-prefill at the destination) or **KV transfer** (ship KV bytes, skip prefill).

## Job model (`instance.py`)

Each session carries **two numbers**, measured over the hold window in its current state:

$$\ell_j = \underbrace{\frac{f_j}{\rho_{\text{dest}}(T_j)}}_{\ell^{\text{pre}}_j} + \underbrace{\frac{g_j}{G}}_{\ell^{\text{dec}}_j}\ \in[0,1], \qquad m_j = \eta\, T_j$$

- **ℓ_j — compute load**, the fraction of one node a job would draw if running (`pop.ell`). Kept split into a prefill and a decode share, because they cost different watts (below).
  - `f_j` = expected **prefill** tok/s = turn rate × input tok/turn.
  - `g_j` = expected **decode** tok/s = turn rate × output tok/turn (reasoning inflates output via a fatter `Y`).
  - `ρ_dest(T_j)` = the **prefill roofline as a function of context length** (`rho_dest`): flat ≈63k tok/s below a knee `T*≈29k`, then decays `~1/T` as attention FLOPs dominate. Prefill is normalized by this *T-dependent* rate, **not** a constant — the same roofline the rebuild cost pays.
  - `G` = the node's sustained **decode** ceiling, a precision-keyed constant (BF16 4600, FP8 9200 tok/s). So `ℓ=1` is the operational knee.
- **m_j — KV footprint**, bytes of cache the session pins (`pop.m`). `η` = KV bytes/token (188 KiB/tok, exact from the attention config); `T_j` = context length.
- **Idle / cold jobs** measure rate ≈0, so `ℓ_j→0` automatically (no separate flag): simultaneously cheap to keep *and* cheap to move. Cold sessions sit in the paged tier (γ) and never carry a nonzero rate.

## Pool & power prices (`power.py`)

**Node curve: ramp-then-plateau.** Power climbs from `P_idle` to `P_busy` as `ℓ` rises to the **power knee**, then is flat (dense models are near-step, not linear-in-load). The **latency knee** sits later; ρ* sits just below it. *The solver never evaluates this curve — only the scalar prices below.*

**Two prices for shed load** — what a removed unit of load is worth in watts:

| symbol | code | watts per node-unit of load | meaning |
|---|---|---|---|
| p̄ | `p_bar` | `P_busy/ρ*` | **amortized**: load the autoscaler eventually turns into drained nodes (*expected*) |
| s_plat | `s_plat` | `p̄ / bracket_ratio` | **plateau slope**: the marginal watts on a node that stays on (*guaranteed*, realized even if no node drains) |

The **bracket** `p̄/s_plat` (swept ~3–5× MoE … ~58× dense 405B) is how much shed value *depends on nodes actually draining* — the central honesty knob of the whole result.

**Two-price split** (the workload is phase-skewed agentic, so prefill and decode watts differ):

$$\bar p_{\text{pre}} = \frac{2\,\bar p}{1+r},\qquad \bar p_{\text{dec}} = r\,\bar p_{\text{pre}},\qquad r=\frac{\bar p_{\text{dec}}}{\bar p_{\text{pre}}}=5$$

A decode busy-second costs `r=5×` a prefill one; the split closes back to the single price, `(p̄_pre+p̄_dec)/2 = p̄`.

**Memory / capacity regime.** Per-node held sessions and the marginal price of holding one:

$$S_{\text{node}} = (1+\gamma)\,\frac{\mathrm{Cap}}{\bar m},\qquad \bar m=\eta\,E[T],\qquad \mu = \frac{P_{\text{idle}}}{S_{\text{node}}}\ \text{(W/held-session)}$$

`Cap` = KV bytes/node after weights; `(1+γ)×` resident gives the **oversubscribed** held ceiling. A memory-bound node sits at idle by construction, so its hold price is `P_idle/S_node` (the old `π(ρ_low)` collapses to `P_idle`). The pool is in exactly one regime — a single comparison:

$$N=\max\!\Big(\tfrac{L}{\rho^\star},\ \tfrac{S_{\text{held}}}{S_{\text{node}}}\Big),\qquad \text{memory-bound} \iff \tfrac{S_{\text{held}}}{S_{\text{node}}} > \tfrac{L}{\rho^\star}$$

Load sets node count `N` when busy; held sessions set it when idle.

## Per-job impact & move cost (`impact.py`)

**Watts freed by shedding job `j`** — three columns the solver reads:

$$\Delta P_j^{\text{guar}} = s_{\text{plat}}\,\ell_j,\qquad \Delta P_j^{\text{exp}} = \bar p_{\text{pre}}\,\ell^{\text{pre}}_j + \bar p_{\text{dec}}\,\ell^{\text{dec}}_j,\qquad \Delta P_j^{\text{mem}} = \mu\,\frac{T_j}{E[T]}$$

Low end **guaranteed**, high end **expected** (autoscaler drains within the hold); both additive over a shed set. The two-price `ΔP^exp` deviates from the single-price `p̄·ℓ_j` *by phase skew* — below it for prefill-skewed jobs, above for decode-skewed (reasoning, chat), equal only when balanced. In the **memory** regime, `m_j` is normalized to session-equivalents (`T_j/E[T]`) so `μ` stays W/session; a job at `E[T]` sheds exactly `μ`, a 2× context job sheds 2μ.

**Downtime of each move** (seconds = *ship* + *rebuild* × destination queue congestion):

$$c_j(R)=\underbrace{\frac{\beta T_j}{\lambda_{\text{src}}}}_{\text{ship IDs}}+\underbrace{(1+\varphi_{\text{pre}})\,\frac{T_j}{\rho_{\text{dest}}(T_j)}}_{\text{re-prefill}},\qquad c_j(S)=\underbrace{\frac{\eta T_j}{\lambda_{\text{src}}}}_{\text{ship KV}}+\underbrace{(1+\varphi_{\text{in}})\,\frac{\eta T_j}{\mu_{\text{in}}}}_{\text{ingest}}$$

- `λ_src` = source WAN **egress** bandwidth (B/s); `μ_in` = destination host-staged **ingest** (B/s).
- `β` = 4 B/tok (replay ships uint32 token IDs); `η` = 188 KiB/tok (transfer ships the whole KV cache). Replay sends ~`η/β ≈ 48k×` fewer bytes but pays re-prefill.
- The **queue wait** enters only as a congestion multiplier `(1+φ)` on the *rebuild* term, with `φ = u/(1−u)` the **M/M/1** wait factor. Replay and transfer hit *different* destination resources, so they carry different utilizations: `φ_pre` at the destination prefill load, `φ_in` at ingest (≈0 — ingest provisioned non-binding). The *ship* terms carry no multiplier; aggregate uplink contention is the egress *constraint* below, not a per-job penalty.

**Egress bytes** a move puts on the uplink: `b_j(R)=β T_j` (IDs), `b_j(S)=η T_j` (KV).

## Dispatch program (`dispatch.py`)

Two decision variables **per (job, destination ℓ)**: replay fraction `y^R_{jℓ}` and transfer fraction `y^S_{jℓ}` (`Y_R, Y_S ∈ [0,1]^{n×K}`), so the action choice needs no separate indicator. Let `y_{jℓ}=y^R+y^S`.

$$\min_{Y_R,Y_S}\ \sum_{j,\ell} y^R_{j\ell}\,c_j(R) + y^S_{j\ell}\,c_j(S)\quad\text{(total downtime, s)}$$

subject to:

| # | constraint | code | reads as |
|---|---|---|---|
| shed | $\sum_{j,\ell} y_{j\ell}\,\Delta P^{\text{bind}}_j \ge S^\star$ | `dp @ total ≥ s_star` | meet the grid ask |
| pairing | $\sum_\ell y_{j\ell}\le 1$ | `sum(Y,axis=1)≤1` | each job moves **at most once**, across all sites |
| **egress** | $\sum_{j,\ell} b_j(R)y^R + b_j(S)y^S \le \lambda_{\text{src}}(D-\tau_{\text{src}})$ | `egress` | **ONE shared uplink** — the sole multi-destination coupling |
| prefill | $\sum_j \tfrac{T_j}{\rho_\ell(T_j)}y^R_{j\ell}\le W_\ell(D-\tau_{\text{pre}})$ | per-ℓ | rebuild replays on `W_ℓ` prefill servers by `D` |
| ingest | $\sum_j \eta T_j\,y^S_{j\ell}\le W_\ell\,\mu_{\text{in}}(D-\tau_{\text{in}})$ | per-ℓ | land KV on `W_ℓ` ingest channels by `D` |
| load | $\sum_j \ell_j\,y_{j\ell}\le \bar L_\ell = \text{spare}_\ell\,\rho^\star$ | `load` | destination stays below its knee |
| held | $\sum_j y_{j\ell}\le \bar S_\ell = \text{spare}_\ell\,S_{\text{node}}$ | `held` | destination KV capacity (incl. (1+γ) uplift) |
| floor | `pinned` classes get `y=0` | — | optional service-level floor |

`τ_*` are one-time ramps (egress connection setup, prefill batch-form, ingest pipeline-fill). Drop the single **egress** row and the program separates into `K` independent single-destination dispatches — it is a **transportation LP with one global uplink knapsack**.

**Certify low, report high (`bind_dp`).** The `≥S*` floor binds against the **guaranteed** column — `s_plat·ℓ_j` in the load regime, `μ·T_j/E[T]` in the memory regime — *never* the optimistic `ΔP^exp`, or we'd promise the grid watts contingent on the autoscaler draining. The expected shed `Σ y·ΔP^exp` (p̄ prices) rides along on the plan as reported upside, not a commitment.

**Two solves, not a branch.** Primary = min-downtime above. If infeasible, **re-solve** maximizing `Σ y·ΔP^bind` (drop the `≥S*` row) and report `shortfall = S* − shed`. Solved both as a fractional **LP** (`y∈[0,1]`, the achievable target) and an integer **MILP** (`y∈{0,1}`, granularity cost); HiGHS via CVXPY.

**Duals (shadow prices).** The LP returns `θ_egress` (watts per uplink byte — the value of the shared bottleneck) and `θ_admit,ℓ` = load-dual + held-dual per site (value of destination admission headroom) — the routing prices that say where the next watt of shed should go.

**Baselines (`greedy`, `random_dispatch`).** Both are myopic single-pass first-fit drawing down the *same* five movement budgets:
- **greedy** — decentralized **bang-per-buck**: each job self-selects its cheaper action, best-deal-first by `min(c_R,c_S)/ΔP` (seconds of downtime per watt). Equals the LP away from resource boundaries; the gap is the **value of central coordination** (repacking transfers→replays to fit more shed under the same links).
- **random** — shuffle movable jobs, coin-flip the action: the floor any real policy should beat.

## Solution structure & sensitivity

- **Bang-per-buck.** Linear objective + target ⇒ take jobs by `c_j/ΔP_j` until `S*` is met, deviating only where a movement constraint binds. LP relaxation and greedy coincide except at resource boundaries — a sort plus a small LP.
- **Ranking is robust, the margin is not.** Within-pool job order is invariant to ρ*, F/G, p̄ (common scaling cancels — verified by scaling `p̄`); only the cross-pool comparison and the absolute feasibility margin move with their values.
- **Regime switch.** Idle/cold-heavy or long-context populations cross from ℓ-ranking to `m`-ranking exactly at the `N=max(·,·)` boundary; γ lowers μ and delays the switch. The two columns are ~uncorrelated at center, so the flip genuinely reorders *which* jobs are shed.

## Deliberately cut (reattachable without changing the above)

Prefill/decode disaggregation, session classes / Markov chains, lexicographic objectives, receding-horizon re-solve as the pool drains (the static snapshot freezes the pre-move regime), 3-stage per-destination downlink contention.
