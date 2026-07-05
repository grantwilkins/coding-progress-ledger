# Queue-Haul Dispatch — Formulation

The grid (or a power cap) asks an inference cluster for a **demand-response event**: shed $S^\star$ watts
by deadline $D$ and hold the reduction over $[D, D+H]$. We hit it by **migrating live sessions** off
the source pool to other sites — replaying their context or shipping their KV cache — choosing *which*
jobs and *how* to move them at **least disruption**. Static snapshot, one source pool → $K$
destinations, coupled only by the source uplink. Power parameters swept in absolute watts
(`assumptions.md`). This file is exactly what `power.py` / `instance.py` / `impact.py` / `dispatch.py`
implement; the DES in `simulate.py` (§10.2) replays a solved plan to check it under execution.

## Assumptions

- **A1** All power quantities are few-second averages.
- **A2** One pool of identical nodes; an autoscaler holds active nodes near setpoint $\rho^\star$. We model its consequence (pool power follows load), not node on/off.
- **A3** A job is prefilling, decoding, or idle. Tool calls, think-time, dormancy = idle: off-GPU, drawing nothing, holding only KV.
- **A4** Per-job loads add (first-order; the headroom $1-\rho^\star$ absorbs the error).
- **A5** Held KV draws no power; it is a capacity constraint only. Cold (paged-out) sessions count toward held capacity at a discount via uplift $\gamma$.
- **A6** Two move primitives: **replay** (ship context token-IDs, re-prefill at the destination) or **KV transfer** (ship KV bytes, skip prefill).

## Job model (`instance.py`)

Two numbers per session, both **time-averages over the hold window** in the session's current state — a **compute load** $\ell_j$ (one axis) and a **KV footprint** $m_j$ (the other axis).

**Compute load.** The node is **colocated** (no prefill/decode disaggregation, `assumptions.md §2`), so prefill and decode *time-share one wall-clock budget*: $\ell^{\text{pre}}_j$ and $\ell^{\text{dec}}_j$ are both **busy-fractions of the same node**, which is why they **add** into one utilization under one ceiling:

$$\ell_j = \underbrace{\frac{f_j}{\rho(T_j)}}_{\text{prefill busy-frac}} + \underbrace{\frac{g_j}{G}}_{\text{decode busy-frac}},\qquad m_j = \eta\, T_j$$

- $f_j$ = prefill tok/s demanded. With a retained KV cache, $f_j=\text{turn rate}\times\Delta_j$, where $\Delta_j$ is new appended input since the cached prefix. On a cache miss, $f_j=\text{turn rate}\times T_j$. $g_j=\text{turn rate}\times Y_j$, where $Y_j$ includes visible answer tokens, hidden reasoning, and tool-call text.
- Sessions are drawn from explicit classes: ordinary chat, long chat/code, reasoning chat, and agentic tool loop. Each class has its own turn rate, $\Delta$, $Y$, context distribution, and cache-hit rate; idle/cold states set current $f_j,g_j,\ell_j$ to zero.
- $\rho(T_j)$ = the node's **prefill roofline** at context $T_j$ (`rho_dest`, here at the **source** MFU): flat $\approx 63\text{k}$ tok/s below $T^\star\approx 29\text{k}$, decaying $\sim 1/T$ above as attention FLOPs dominate. Prefill is **compute-bound**, so longer context costs throughput. *(Same function, at a destination's MFU, is the rebuild rate $\rho_\ell$ — hence the `dest` in the code name; in the source load it is the source's own rate.)*
- $G$ = the **decode** ceiling, a precision-keyed constant (BF16 4600, FP8 9200 tok/s). This is a first-order approximation: long context taxes decode mainly through memory capacity here, and any $G(T)$ slowdown is a sensitivity knob, not in the base model.

**KV footprint.** $m_j = \eta\,T_j$ is the cache the session pins in HBM ($\eta$ = 188 KiB/tok, exact from the attention config) — the *capacity* axis, independent of whether the session computes.

**How much fits on a node (two knees, one setpoint).** Power saturates at the **power knee** $\ell\approx 0.10$ (the node draws $\approx P_{\text{busy}}$ for any larger $\ell$ — why $s_{\text{plat}}$ is small); latency departs at the **latency knee** $\ell\approx 0.85$ ($\ell=1$ normalization). The autoscaler holds each node at the setpoint $\rho^\star\approx 0.80$ between them. So a node **runs** $\sum_{\text{on node}}\ell_j\le\rho^\star$ — at center $\approx 9$ active-agentic sessions ($\ell\approx 0.087$ each) or $\sim 240$ chat ($\ell\approx 0.003$) — and **holds** up to $S_{\text{node}}\approx 15$ sessions in memory. The binding axis sets the node count (the $N=\max(\cdot,\cdot)$ test below).

A single job also has to fit under the same setpoint. If $\ell_j>\rho^\star$, this one-node placement model rejects the job instead of averaging it into the pool or clipping it.

**Idle / cold** sessions measure rate $\approx 0$, so $\ell_j\to 0$ automatically (no flag): cheap to keep *and* cheap to move. They still pin $m_j$ (cold at the $\gamma$ discount).

## Pool & power prices (`power.py`)

**Node curve: ramp-then-plateau.** Power climbs from $P_{\text{idle}}$ to $P_{\text{busy}}$ as $\ell$ rises to the **power knee**, then is flat (dense models are near-step, not linear-in-load). The **latency knee** sits later; $\rho^\star$ sits just below it. The canonical dispatch solver never evaluates this curve — only the scalar prices below. `node_knee.py` is a separate exploration path that evaluates the curve when source-node placement is explicit.

**Two prices for shed load** — what a removed unit of load is worth in watts:

| symbol            | code     | watts per node-unit of load        | meaning                                                                                                       |
| ----------------- | -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| $\bar p$          | `p_bar`  | $P_{\text{busy}}/\rho^\star$       | **amortized**: load the autoscaler eventually turns into drained nodes (*expected*)                           |
| $s_{\text{plat}}$ | `s_plat` | $\bar p\,/\,\text{bracket\_ratio}$ | **plateau slope**: the marginal watts on a node that stays on (*guaranteed*, realized even if no node drains) |

The **bracket** $\bar p/s_{\text{plat}}$ (swept $\sim 3\text{–}5\times$ MoE … $\sim 58\times$ dense 405B) is how much shed value *depends on nodes actually draining* — the central honesty knob of the whole result.

**Average work power vs. future node impact.** Measured per-token energies estimate the job's **work power**:

$$P_j^{\text{work}} = c_1\, f_j + c_2\, g_j,\qquad \frac{c_2}{c_1}\approx 5\text{–}14\ \text{(measured)}$$

The token-energy coefficients are calibrated averages, not fixed facts over time. They should be refit or swept when the model, hardware, precision, serving stack, batching policy, context mix, or traffic mix changes.

The job's **future node-power impact** also includes the share of static node power attached to the node time it occupies:

$$\Delta P_j^{\text{future}} = \frac{P_{\text{idle}}}{\rho^\star}\ell_j + c_1 f_j + c_2 g_j$$

The old single-price proxy $\bar p\ell_j$ is kept only as a comparison column.

Decode is higher-energy per token, not higher instant power. Prefill can draw higher instant power for a short burst; decode draws lower power for longer.

**Empirical grounding (powertrace-sim — measured A100/H100 vLLM traces, [Wilkins et al. 2026](https://arxiv.org/abs/2603.18383)).** The ramp-then-plateau holds: a saturating node fit scores $R^2$ 0.91–0.99 vs 0.25 for a linear fit on dense 70B+, knee at $\ell\approx 0.09$ (dense; later, $\ell\approx 0.3$–$0.6$, for MoE). The bracket ratio $\bar p/s_{\text{plat}}$ is *read off the same traces* — **58× (405B), 30× (70B-A100), 17× (70B-H100), ~5× (MoE)** — and $c_1,c_2$ above are the fitted coefficients.

**Memory / capacity regime.** Per-node held sessions and the marginal price of holding one:

$$S_{\text{node}} = (1+\gamma)\,\frac{\mathrm{Cap}}{\bar m},\qquad \bar m=\eta\,E[T],\qquad \mu = \frac{P_{\text{idle}}}{S_{\text{node}}}\ \text{(W/held-session)}$$

$\mathrm{Cap}$ = KV bytes/node after weights; $(1+\gamma)\times$ resident gives the **oversubscribed** held ceiling. A memory-bound node sits at idle by construction, so its hold price is $P_{\text{idle}}/S_{\text{node}}$ (the old $\pi(\rho_{\text{low}})$ collapses to $P_{\text{idle}}$). The pool is in exactly one regime — a single comparison:

$$N=\max\!\Big(\tfrac{L}{\rho^\star},\ \tfrac{S_{\text{held}}}{S_{\text{node}}}\Big),\qquad \text{memory-bound} \iff \tfrac{S_{\text{held}}}{S_{\text{node}}} > \tfrac{L}{\rho^\star}$$

Load sets node count $N$ when busy; held sessions set it when idle.

## Per-job impact & move cost (`impact.py`)

**Watts freed by shedding job $j$** — columns the solver/reports read:

$$\Delta P_j^{\text{floor}} = s_{\text{plat}}\,\ell_j,\qquad
\Delta P_j^{\text{cert}} = s_{\text{plat}}\,\ell_j + c_1 f_j + c_2 g_j,\qquad
\Delta P_j^{\text{future}} = \frac{P_{\text{idle}}}{\rho^\star}\ell_j + c_1 f_j + c_2 g_j,\qquad
\Delta P_j^{\text{mem}} = \mu\,w_j\,\frac{T_j}{E[T]}$$

The dispatch certificate uses $\Delta P_j^{\text{cert}}$: fixed-node load slope plus measured
average token work. It deliberately gives idle/cold sessions zero certified watts. The high end
**future** estimate adds the static node share that is only justified once removed load lets nodes
shut off. $\Delta P_j^{\text{mem}}$ is a memory-pressure diagnostic only: it normalizes held KV to
session-equivalents ($T_j/E[T]$), with $w_j=1/(1+\gamma)$ for cold sessions and $w_j=1$ otherwise.

**Downtime of each move** (seconds = *ship* + *rebuild* × destination queue congestion):

$$c_j(R)=\mathbf{1}_{\text{active}}\Big[\underbrace{\frac{\beta T_j}{\lambda_{\text{src}}}}_{\text{ship IDs}}+\underbrace{(1+\varphi_{\text{pre}})\,\frac{T_j}{\rho_{\text{dest}}(T_j/2)}}_{\text{full replay}}\Big],\qquad c_j(S)=\mathbf{1}_{\text{active}}\Big[\underbrace{\frac{\eta T_j}{\lambda_{\text{src}}}}_{\text{ship KV}}+\underbrace{(1+\varphi_{\text{in}})\,\frac{\eta T_j}{\mu_{\text{in}}}}_{\text{ingest}}\Big]$$

- $\lambda_{\text{src}}$ = source WAN **egress** bandwidth (B/s); $\mu_{\text{in}}$ = destination host-staged **ingest** (B/s).
- $\beta$ = 4 B/tok (replay ships uint32 token IDs); $\eta$ = 188 KiB/tok (transfer ships the whole KV cache). Replay sends $\sim\eta/\beta\approx 48\text{k}\times$ fewer bytes but pays re-prefill.
- The **queue wait** enters only as a congestion multiplier $(1+\varphi)$ on the *rebuild* term, with $\varphi = u/(1-u)$ the **M/M/1** wait factor. Replay and transfer hit *different* destination resources, so they carry different utilizations: $\varphi_{\text{pre}}$ at the destination prefill load, $\varphi_{\text{in}}$ at ingest ($\approx 0$ — ingest provisioned non-binding). The *ship* terms carry no multiplier; aggregate uplink contention is the egress *constraint* below, not a per-job penalty.

**Egress bytes** a move puts on the uplink: $b_j(R)=\beta T_j$ (IDs), $b_j(S)=\eta T_j$ (KV).

## Dispatch program (`dispatch.py`)

Two decision variables **per (job, destination $\ell$)**: replay fraction $y^R_{j\ell}$ and transfer fraction $y^S_{j\ell}$ ($Y_R, Y_S \in [0,1]^{n\times K}$), so the action choice needs no separate indicator. Let $y_{j\ell}=y^R_{j\ell}+y^S_{j\ell}$.

$$\min_{Y_R,Y_S}\ \sum_{j,\ell} y^R_{j\ell}\,c_j(R) + y^S_{j\ell}\,c_j(S)\quad\text{(total downtime, s)}$$

subject to:

| #          | constraint                                                                                              | code                  | reads as                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------- |
| shed       | $\sum_{j,\ell} y_{j\ell}\,\Delta P^{\text{bind}}_j \ge S^\star$                                         | `dp @ total ≥ s_star` | meet the grid ask                                                     |
| pairing    | $\sum_\ell y_{j\ell}\le 1$                                                                              | `sum(Y,axis=1)≤1`     | each job moves **at most once**, across all sites                     |
| **egress** | $\sum_{j,\ell} b_j(R)\,y^R_{j\ell} + b_j(S)\,y^S_{j\ell} \le \lambda_{\text{src}}(D-\tau_{\text{src}})$ | `egress`              | **ONE shared uplink** — the sole multi-destination coupling           |
| prefill    | $\sum_j \tfrac{T_j}{\rho_\ell(T_j/2)}\,y^R_{j\ell}\le \lfloor\text{spare}_\ell\rfloor(D-\tau_{\text{pre}})$ | per-$\ell$        | rebuild replays on the $\lfloor\text{spare}_\ell\rfloor$ whole spare nodes by $D$ |
| ingest     | $\sum_j \eta T_j\,y^S_{j\ell}\le \lfloor\text{spare}_\ell\rfloor\,\mu_{\text{in}}(D-\tau_{\text{in}})$  | per-$\ell$            | land KV on the same $\lfloor\text{spare}_\ell\rfloor$ spare nodes by $D$ |
| load       | $\sum_j \ell_j\,y_{j\ell}\le \bar L_\ell = \text{spare}_\ell\,\rho^\star$                               | `load`                | destination stays below its knee                                      |
| held       | $\sum_j w_j\,\tfrac{T_j}{E[T]}\,y_{j\ell}\le \bar S_\ell = \text{spare}_\ell\,S_{\text{node}}$          | `held`                | destination KV capacity (incl. cold discount and $(1+\gamma)$ uplift) |
| floor      | pinned classes get $y=0$                                                                                | `pinned`              | optional service-level floor                                          |
| deadline   | $y^a_{j\ell}=0$ if the session's *no-wait* completion misses $D$                                        | `deadline_infeasible` | sole link access + a free rebuild server, per the DES timeline (sf: $\max(ed,\tau)+\text{reb}$; ct: $\max(ed,\max(\tau_{\text{src}},\tau)+\text{reb})$); whole-job basis, so it also bans fractional moves the DES could split — deliberate tightening of the LP (affects the LP-vs-MILP granularity reading). Applied to **all** policies (baselines see banned actions as infinitely priced), never to feasibility audits. |

$\tau_*$ are one-time ramps (egress connection setup, prefill batch-form, ingest pipeline-fill). Drop the single **egress** row and the program separates into $K$ independent single-destination dispatches — it is a **transportation LP with one global uplink knapsack**.

**No dedicated rebuild hardware.** Rebuild runs on the destination's spare pool ($\lfloor\text{spare}_\ell\rfloor$ whole nodes, the same pool that backs the load/held rows) — that is what the testbed physically is. Two acknowledged approximations, both pending Track 1 calibration: (a) prefill and ingest on a shared node are budgeted as overlapping (compute-bound vs copy-engine-bound; partially relaxed by $\alpha_{\text{in}}$ below); (b) sessions that finish rebuilding inside $[0,D]$ start consuming spare serving capacity, and neither the rows above nor the DES debit rebuild capacity for it. Note the removal of the dedicated pool *raises* rebuild capacity at the center parameters (8 dedicated servers → $\lfloor 0.4\cdot 32\rfloor = 12$ shared nodes); the planner-side cushion $\kappa$ (a `solve(..., kappa=)` derate on the prefill/ingest RHS only — never applied to the DES, diagnostics, or baselines) is the lever that covers both approximations.

**Ingest–prefill interference (DES only, `Movement.alpha_in`, default 0.10).** Prefill work assigned at start time $t$ is inflated by $1/(1-\alpha_{\text{in}} u)$, $u$ = the cluster's ingest-channel busy fraction at $t$ — first-order coupling, sampled at start only (later-arriving ingest is invisible to a running prefill, and a split shipment's own ingest piece never drags its own prefill). Evidence for the center value: pure-DMA copy-engine loading overlaps prefill with near-zero coupling, while the SM-based copy path costs ~6% end-to-end throughput (vLLM KV-offloading connector blog, 2026-01) — 0.10 is a conservative center between those and worse-than-6% synchronous paths; LMCache loads KV layer-wise on a side stream with a per-layer sync (LMCache tech report, arXiv:2510.09665). This models **async loading of uncompressed KV over DMA** (vLLM ≥0.9 connector async / LMCache `async_loading`); with LMCache's *default synchronous* retrieval, or GPU-side decompression (≥30% mutual slowdown, ShadowServe arXiv:2509.16857; CacheGen SIGCOMM'24), the overlap assumption fails and $\alpha_{\text{in}}$ would be far larger. The planner never sees $\alpha_{\text{in}}$ ($\kappa$ covers it); Track 1 measures it directly. Note cut-through ≤ store-and-forward is guaranteed only at $\alpha_{\text{in}}=0$: an earlier prefill start can land in a busier ingest window.

**Certify active work, report high (`bind_dp`).** The $\ge S^\star$ floor binds against
`dp_certified = s_plat·ℓ_j + c_1 f_j + c_2 g_j` in every regime. Memory remains a capacity
constraint (`held`), but held KV does not create certified watts unless a later node-drain model
proves whole source nodes can turn off by the deadline. The future-impact proxy is reported, not
bound, because it depends on autoscaler/node-drain behavior.

**Two solves, not a branch.** Primary = min-downtime above. If infeasible, **re-solve** maximizing $\sum y\,\Delta P^{\text{bind}}$ (drop the $\ge S^\star$ row) and report $\text{shortfall} = S^\star - \text{shed}$. Solved both as a fractional **LP** ($y\in[0,1]$, the achievable target) and an integer **MILP** ($y\in\{0,1\}$, granularity cost); HiGHS via CVXPY.

**Duals (shadow prices).** The LP returns $\theta_{\text{egress}}$ (watts per uplink byte — the value of the shared bottleneck) and $\theta_{\text{admit},\ell}$ = load-dual + held-dual per site (value of destination admission headroom) — the routing prices that say where the next watt of shed should go.

**Reported disruption.** The solver minimizes aggregate session downtime. Plots report `cost / S*` in **seconds per certified kW** so comparisons are not driven by a larger requested shed or a larger synthetic population.

**Baselines (`greedy`, `random_dispatch`).** Both are myopic integer first-fit policies drawing down the *same* five movement budgets:
- **greedy** — decentralized **bang-per-buck**: jobs are considered best-deal-first by $\min(c_R,c_S)/\Delta P$ (seconds of downtime per watt). Each job tries its cheaper action, then the alternate, and moves only if the whole job fits. It may overshoot $S^\star$ by one job.
- **random** — shuffle movable jobs, coin-flip the action, and move only whole jobs: the floor any real policy should beat.

## Solution structure & sensitivity

- **Bang-per-buck.** Linear objective + target ⇒ the fractional LP takes jobs by $c_j/\Delta P_j$ until $S^\star$ is met, deviating only where a movement constraint binds. Integer greedy and MILP carry job-granularity overshoot; at boundaries, compare MILP and LP separately so relaxation value is not mistaken for deployable policy value.
- **Ranking is robust, the margin is not.** Within-pool job order is invariant to $\rho^\star$, $F/G$, $\bar p$ (common scaling cancels — verified by scaling $\bar p$); only the cross-pool comparison and the absolute feasibility margin move with their values.
- **Regime switch.** Idle/cold-heavy or long-context populations cross from $\ell$-ranking to $m$-ranking exactly at the $N=\max(\cdot,\cdot)$ boundary; $\gamma$ lowers $\mu$ and delays the switch. The two columns are $\approx$ uncorrelated at center, so the flip genuinely reorders *which* jobs are shed.

## Deliberately cut (reattachable without changing the above)

Prefill/decode disaggregation, session classes / Markov chains, lexicographic objectives, receding-horizon re-solve as the pool drains (the static snapshot freezes the pre-move regime), 3-stage per-destination downlink contention.

## Node-knee exploration (`node_knee.py`)

The additive dispatch path values moved jobs independently. The node-knee exploration instead requires an explicit `source_node` placement and evaluates modeled expected source shed by node:

$$r_i=\sum_{j\in i}\ell_j y_j,\qquad F_i(r_i)=P(L_i)-P(L_i-r_i).$$

For the ramp-then-plateau curve, $F_i$ is convex in removed load: concentrating removals can become more valuable once a node crosses the power knee. The exact target $\sum_iF_i(r_i)\ge S^\star$ is nonconvex, so `node_knee.py` keeps it out of the canonical solver and provides compact exploration methods:

- sequential tangent LPs using global lower bounds of $F_i$,
- active-knee LP relaxation and MILP candidates that force selected nodes below the power knee, keep unselected above-knee nodes in the plateau region, and use the exact affine power expression inside each fixed region,
- live and node-drain greedy baselines,
- a tiny exact enumeration oracle for hand-checkable cases.

Reported names are `node_expected_w` for modeled node-curve power and `active_floor_w` for the conservative active-work floor. Node-knee expected watts are model evidence, not a hard grid guarantee.
