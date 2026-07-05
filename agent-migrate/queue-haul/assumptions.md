# Queue-Haul Assumptions (filled)

**Setup fixed for these experiments:** Qwen3-235B-A22B served on a single colocated pool
of 8×H100 SXM nodes, TP=8, BF16, **no prefill/decode disaggregation**. One pool, one model.

**Conventions.** *Hard* = grounded in a measurement or exact arithmetic (cited). *Derived* =
computed from other rows. *Sweep* = bespoke or deployment-specific; swept over the stated
band, which comes from the formulation's measured ranges, not invented. Every *Sweep* row
has a **center** (the default single value for non-sweep runs) and **bounds**.

---

## 1. Model & workload constants

| Parameter | Value | Type | Source |
|---|---:|---|---|
| Model | Qwen3-235B-A22B | hard | fixed |
| η (KV bytes/tok) | 188 KiB | hard | exact from attention config (evac) |
| β (context bytes/tok) | 4 B | hard | uint32 token IDs |
| **Weights** | **BF16 470 GB / FP8 235 GB — run both** | sweep | precision is a primary axis; FP8 is the production-recommended serving mode (SGLang/vLLM), BF16 the full-precision baseline |
| **Context length T** | mixture below, **parameters swept** | sweep | evac snapshot is the center; the shape is an axis |
| → center mixture | 0.70·LogN(10.07, 1.0) + 0.30·LogN(11.45, 0.8), clip [1e3, 1e6] | center | E[T] ≈ 65,800, median ≈ 36,100 |
| → sweep: short-context | 0.85·LogN(9.0, 0.9) + 0.15·LogN(10.5, 0.8) | sweep | E[T] ≈ 18k; tests the load-bound (not memory-bound) regime |
| → sweep: long-context | 0.50·LogN(10.5, 1.0) + 0.50·LogN(12.0, 0.9) | sweep | E[T] ≈ 130k; pushes into memory-bound, big-KV migration |
| → sweep: tail weight | vary the long-component mass 0.15 → 0.50 | sweep | the single knob that moves E[T] and the memory/load regime boundary |
| Seed | 42 | hard | reproducibility |

**Why T is swept, not fixed.** ServeGen's central finding is that input/output and context length
distributions are well-described by mixtures of standard distributions, but *the parameters shift
significantly over time* — there is no single canonical distribution. So T's mixture parameters are
an experimental axis: the short/center/long settings above bracket the regimes (load-bound →
memory-bound), and m̄, S_node, and the memory/load boundary all move with them.

**What the workload is.** Queue-Haul samples a one-time evacuation snapshot, not a full traffic
trace. Each row is one held session on the source pool. A session may be active now, warm but
between turns, or cold/paged out. Active sessions consume current serving time; warm and cold
sessions keep memory but have zero current load. The default center is an agentic long-context
stress point. It is not a generic chat mix.

| setup | what it represents | why it exists |
|---|---|---|
| Default `Workload()` | 30% active, 25% idle-warm, 45% cold; all agentic tool-loop sessions | memory-heavy evacuation center |
| Class-isolated plots | one class at a time, active-only, cache-hit | shows whether ordinary chat, long chat/code, reasoning, or agentic loops create different movement bottlenecks |
| Short-context load-bound plots | short contexts with enough active work for load to bind | isolates load-price behavior without memory dominating |
| Multi-destination plots | constructed pure-transfer or heterogeneous-destination cases | validates routing constraints; not meant to be a representative workload |

**How population size is determined.** The generator does not choose a request arrival rate and run
a queue. It chooses the number of held sessions from memory occupancy:

```text
n_jobs = round(occupancy * source_nodes * S_node)
```

where `S_node` is the held-session capacity of one node at the current context scale and precision.
Then it draws each session's state, class, context, turn rate, appended input, output length, and
cache-hit flag.

---

## 2. Node power — absolute watts, 8×H100 SXM node

GPU-only draw is ~5.6 kW (8 × 700 W); a loaded HGX-H100 node is ~10–11 kW including host,
fans, and PSU losses; idle GPUs draw <100 W each. Serving rarely hits TDP — POLCA reports
inference clusters use ≤80% of provisioned power — so the **serving plateau** `P_busy` is set
at 0.8× the TDP ceiling, not the ceiling itself.

| Parameter | Value | Type | Source / derivation |
|---|---:|---|---|
| TDP ceiling (node) | 10.5 kW | hard | 8×700 W GPU + ~4.9 kW host overhead |
| **P_idle (W/node)** | **3,200 W** (center); sweep [2,500, 4,000] | sweep | 8×90 W idle GPU + ~half host overhead; the idle fraction is the swept quantity |
| **P_busy (W/node)** | **8,400 W** (center); sweep [7,500, 10,000] | sweep | 0.8× TDP ceiling (POLCA serving headroom) |
| P_idle / P_busy | ≈ 0.38 | derived | the static fraction; this dense node is consolidation-dominated |
| **Power knee ℓ** | **0.10** (center); sweep [0.03, 0.15] | sweep | dense 70B+ band; power saturates early |
| **Latency knee ℓ** | **0.85** (center); sweep [0.73, 1.0] | sweep | sets ρ* ceiling and ℓ=1 normalization |
| **ρ\* setpoint** | **0.80** (center); sweep [0.55, 0.90] | sweep | between the two knees; main operating-band knob |
| **p̄ = P_busy/ρ\*** | ≈ 10,500 W/node-unit | derived | amortized price (at center) |
| **Bracket ratio p̄/s_plat** | **30×** (center); sweep [17, 58] | sweep | dense-70B-to-405B band (235B sits in it) |
| **s_plat = p̄/ratio** | ≈ 350 W/node-unit | derived | fixed-node plateau slope (the guaranteed price) |
| Token-energy c1, c2 | prefill 0.148 J/tok, decode 1.76 J/tok | sweep | H100 dense analog; calibrated trace averages, not constants of nature |
| Certified power | `s_plat·ℓ_j + c1·f_j + c2·g_j` | implementation | active serving work only; held KV is capacity, not certified watts |
| Future node-drain proxy | `P_idle/ρ* · ℓ_j + c1·f_j + c2·g_j` | implementation | `p̄·ℓ_j` is kept only as the single-price comparison column |
| **G (decode tok/s ceiling)** | **BF16 4,600 / FP8 9,200** | sweep | the `ℓ_dec = g/G` normalizer, precision-keyed: Baseten 4×H100-FP8 ~4,600 tok/s anchor scaled to the 8×H100 node |
| **F (prefill normalizer)** | **per-job `ρ_dest(T_j)`** | derived-fn | `ℓ_pre,j = f_j/ρ_dest(T_j)` uses the §6 roofline at each job's own context — **not a constant** (retires the median-vs-mean choice for E[T]) |

**Token-energy coefficients:** work power is `c1·f_j + c2·g_j`. These
coefficients are calibrated averages for a measured model, hardware, precision,
serving stack, batching policy, and workload mix; refit or sweep them when those
change. Future impact is the static node share plus token work:
`ΔP_j = P_idle/ρ* · ℓ_j + c1·f_j + c2·g_j`.
The dispatch certificate excludes held-KV memory credit and uses the active-work
column `s_plat·ℓ_j + c1·f_j + c2·g_j`.

---

## 3. Per-job rate distributions

These define the synthetic job population. **Invariant to enforce in the generator:**
active jobs have `ℓ_j > 0` and sit in resident HBM; cold jobs have `ℓ_j ≈ 0` and sit in the
paged tier counted by γ (§4). Never assign a cold job a nonzero rate — that double-counts.

| Parameter | Value | Type | Source / rationale |
|---|---:|---|---|
| **State mix (active / idle-warm / cold)** | **0.30 / 0.25 / 0.45** (center) | sweep | sweep active∈[0.15,0.50]; cold is the migration-cheap majority |
| **Session classes** | ordinary chat / long chat-code / reasoning chat / agentic tool loop | sweep | default center is agentic; class-isolated plots use one class at a time |
| **Turn rate, active jobs (turns/s)** | **0.01 chat/reasoning, 0.15 agentic** (per-class mean) | sweep | agentic ≈ tight loop (1 per ~7 s); chat/reasoning ≈ 1 per ~100 s |
| **Turn-rate within-class spread σ** | **0.3** | sweep | per job `rate ~ LogN(log(mean)−σ²/2, σ)`, mean-preserving; gives within-class ℓ heterogeneity so greedy≠LP at boundaries |
| **Per-session occupation cap** | **ℓ ≤ 0.50** | hard/sweep | a session cannot start turns faster than its own prefill+decode service time; long generated outputs reduce effective turn rate |
| **Input tokens/turn Δ** | median: ordinary 150, long/reasoning 500; agentic Δ/Y=3.0 | sweep | Δ is appended input since cached prefix, not full context |
| **Output tokens/turn Y (incl. reasoning)** | geometric mean: ordinary 300, long 1000, reasoning 3000, agentic 600 | sweep | decode length is geometric (memoryless EOS); reasoning fattens Y |
| **Context T by class** | mean ≈ 3.4K / 17K / 22K / 66K | sweep | ordinary chat and agents no longer share one T distribution |
| **Cache hit by class** | 0.99 / 0.95 / 0.90 / 0.95 | sweep | hit: prefill Δ; miss: prefill full T |
| Idle-warm job rate | ~0 (resident, between turns) | hard | warm but not generating |
| Cold job rate | 0 (paged out) | hard | formulation §2: rate carries liveness |
| Class mix | **agentic center** by default | sweep | mixed populations set `class_mix`; stateless API is out of scope for this session model |

**Grounding for Δ and Y (from ServeGen + workload-profile characterizations).** Three empirical
facts shape these: (i) **decode length follows a geometric distribution** — at each step the model
emits EOS with roughly constant probability, so NumPy's `geometric(p)` draw has mean `1/p`;
this is the memoryless tail, not a log-normal. (ii) **Input and output lengths are weakly correlated** — so Δ
and Y are sampled independently per turn, not jointly. (iii) **The parameters shift over time** —
hence the wide sweep bands rather than point values. Concrete per-class anchors: persona/roleplay
and agentic system prompts run 2k–10k input tokens; code-completion and chat inputs are shorter
(hundreds); reasoning models inflate Y by 5–20× through hidden tokens (the 4000-mean reasoning
case). Use the geometric form for Y so the synthetic decode tail matches production; use log-normal
for Δ (bounded, mean is what matters for prefill load).

The per-job load is `ℓ_j = f_j/ρ(T_j) + g_j/G`, with `f_j = turn_rate·Δ` on a
cache hit and `f_j = turn_rate·T_j` on a cache miss. `g_j = turn_rate·Y`. Idle
and cold jobs set current `turn_rate`, `f`, `g`, and `ℓ` to zero. For active jobs,
the sampled turn rate is capped by the per-turn service time so one session cannot
occupy more than its configured share of a node.

---

## 4. Capacity & memory regime

| Parameter | Value | Type | Source / derivation |
|---|---:|---|---|
| **Cap (KV bytes/node after weights)** | **BF16: 130 GB; FP8: 365 GB** — run both | sweep | BF16: 640−470−40. FP8: 640−235−40. Precision is a primary axis, not a footnote. |
| m̄ = η·E[T] | ≈ 12.7 GB/session (at center T) | derived | 188 KiB × E[T]; **moves with the T sweep (§1)** |
| S_node^resident = Cap/m̄ | BF16 ≈ 10.3, FP8 ≈ 29 sessions | derived | what a node can actively serve |
| **γ (paged-out uplift)** | **0.5** (center); sweep [0.5, 1.0] | sweep | cold/resident ratio; offload-tier size × cold fraction |
| **S_node = (1+γ)·S_node^res** | BF16 ≈ 15.4, FP8 ≈ 43 held (center) | derived | total holdable incl. paged |
| **μ = P_idle / S_node** | BF16 ≈ 208, FP8 ≈ 74 W/held-session | derived | memory-pressure diagnostic; not a dispatch certificate |

**Note on the memory regime (ρ_low eliminated).** A node is in the memory regime precisely when
KV fills before compute does — which means the sessions it holds are mostly idle or cold (that is
*why* memory binds rather than load). Its compute utilization is therefore ~0 and its power is
just `P_idle`. So the marginal power of a held session is `μ = P_idle / S_node` — there is no
separate "utilization when memory binds" parameter to set. (Any residual active load on that node
is already counted in the load-regime term; the `max()` over the two regimes in `N` picks whichever
binds, so the two never double-count.) This is diagnostic accounting for future node-drain work,
not a certifiable watt value for the current dispatch objective. **ρ_low is removed from the model.**

---

## 5. Pools & event

| Parameter | Value | Type | Source / rationale |
|---|---:|---|---|
| **Source pool size (nodes)** | **32** (center); sweep {16, 32, 64} | sweep | 32 nodes ≈ 0.27 MW full-serving; large enough that integer node-drain is fine-grained |
| **Occupancy (population size)** | **1.2** (center); sweep [0.8, 1.5] | sweep | sessions held ÷ node memory capacity: `n_jobs = occupancy·N_nodes·S_node` held sessions; >1 = oversubscribed source (the shed trigger), <1 = slack. Sets the memory side of the regime test (`S_held/S_node = occupancy·N`); the load side `L/ρ*` is the measured output. |
| Pool full-serving power | ≈ 0.27 MW (at 32 nodes, P_busy) | derived | the **load-bound** shed ceiling (reached only when load, not memory, binds) |
| **ρ\* setpoint band** | [0.55, 0.90], center 0.80 | sweep | = §2 row; between power and latency knees |
| **Destination spare load L̄_dest** | **0.40·(dest nodes)·ρ\*** (center) | sweep | sweep spare∈[0.2, 0.6] of a dest node; headroom below its knee |
| **Destination spare held S̄_dest** | (1+γ)·Cap/m̄ · (spare frac) | derived | counts the paged uplift |
| **Shed target S\*** | sweep [0.05, 0.25] MW (≈ 20–90% of pool) | sweep | the grid command; primary x-axis of the dispatch results |
| **Deadline D** | sweep **15–600 s**, center **300 s** | sweep | evac band; the migration-time pressure knob |
| **Hold H** | **1200 s** (20 min); sweep [600, 3600] | sweep | must be ≫ node drain time (minutes) for the autoscaler model to hold; short-H is the throttle regime |
| Pinned job classes (y_j=0) | **none** (center) | sweep | optionally pin a chat class to test service-floor behavior |

---

## 6. Movement

| Parameter | Value | Type | Source |
|---|---:|---|---|
| Λ_src (egress link) | **1 GB/s** (center); sweep [0.5, 10] GB/s | sweep | WAN-class; the drain-rate knob, swept since inter-site BW varies widely |
| **Rebuild nodes (per dest)** | **⌊spare⌋ = ⌊0.4·32⌋ = 12** (center) | derived | no dedicated pool: rebuild runs on the whole spare nodes that also absorb migrated serving; swept only via spare_frac/dest_nodes |
| **ρ_dest(T) (prefill tok/s, a function)** | **see below** | derived-function | FLOP roofline of one 8×H100 node, not a constant |
| **μ_in (ingest, per node)** | **512 GB/s** (center); sweep [256, 512] GB/s | sweep | host-staged PCIe Gen5 ×8; sweep covers staging overhead / contention |
| **τ_src, τ_pre, τ_in (startup s)** | **2 / 5 / 3 s** (center) | sweep | conn ramp / batch-form / pipeline-fill; sweep each [0, 2×center] |
| **φ_pre, φ_in (dest queue congestion)** | `(1+φ)` on rebuild; φ=u/(1−u) at `dest_prefill_util` **0.6** / `dest_ingest_util` **0** | derived | Replay queues against destination *prefill*, transfer against *ingest* — different resources, different utilizations (one shared φ would erase the action distinction). Both destination-side knobs, swept in T7. `dest_prefill_util`=0.6 from §5's ~0.4 destination spare; `dest_ingest_util`=0 (ingest non-binding by construction → φ_in≈0; network is the binding transfer resource). Ship term uncontended (φ_src=0); aggregate Λ_src contention is the T4 egress constraint, not a per-job downtime. |

**ρ_dest is a function of context length, not a number.** The destination re-prefills the full
context on replay, and the prefill rate slips as context grows because attention is quadratic. From
the FLOP roofline of one 8×H100 node (MFU 0.35):

```
ρ_dest(T) = EFF / (2·N_act + C·T)          [tokens/s]
  EFF   = 8 · 989.5 TFLOP/s · MFU(=0.35)   sustained node compute
  N_act = 22e9                              active params (A22B)
  C     = 2 · L_attn · H_q · d_head         per-token attention coefficient
         (Qwen3-235B: L_attn=94, H_q=64, d_head=128 → C ≈ 1.54e6)
```

**Use QUERY heads H_q (64), not KV heads H_kv (4), in C.** Attention *compute* scales with the
query heads; only KV-cache *size* (η, §1) scales with KV heads. With H_q, T\* = 2·N_act/C ≈ **29k**
and ρ_dest(100k) ≈ **14k tok/s** — the stated landmarks; the H_kv form would mislocate T\* to ~457k.

This is **FFN-bound and flat at EFF/(2·N_act) ≈ 63k tok/s below a crossover T\* ≈ 29k tokens, then
attention-bound and decaying as 1/T above it.** At the workload's long contexts (E[T] ≈ 66k, deep in
the 1/T regime) it lands near the ~14k tok/s the paper reports at 100k. ρ_dest serves **two** roles,
both per-session from a job's own T_j (never a fixed rate): the replay rebuild cost `T_j/ρ_dest(T_j)`
(§6, T3), and the **prefill load normalizer** `ℓ_pre,j = f_j/ρ_dest(T_j)` (the job generator, T2 —
this is what retires the constant F). MFU is the one swept input (sweep [0.3, 0.5]); everything else is
architecture. This makes the replay/transfer tradeoff context-dependent exactly as the single-session
boundary (Fig. 1/2 of the EE364b draft) already showed.

---

## 7. What's hard vs. swept — the one-glance summary

**Hard (cited or exact), do not sweep:** η, β, the TDP ceiling, μ_in nominal, seed.

**Derived, computed once (and re-derived per sweep point where they depend on a swept input):**
E[T], m̄, S_node^resident, S_node, p̄, s_plat, **μ = P_idle/S_node**, pool power, S̄_dest, w_pre/w_in,
and **ρ_dest(T)** (a function, computed per session — also the prefill load normalizer, retiring F).

**Swept (bespoke/uncertain) — these are the experiment's axes:**
- *Precision:* **weights BF16 / FP8** (primary — run both end to end; shifts Cap, S_node, and **G** ~2×).
- *Power model:* P_idle, P_busy, power knee, latency knee, ρ*, bracket ratio, **G (decode ceiling)**.
- *Workload:* session class mix, class-specific context distribution, state mix, turn rates + **σ spread**, occupation cap, Δ, Y, and cache-hit rate.
- *Capacity:* γ.
- *Event:* pool size, **occupancy**, S*, D, H, destination spare (rebuild runs on its ⌊spare⌋ whole nodes).
- *Movement:* Λ_src, μ_in, MFU (drives ρ_dest), startup latencies.

**Eliminated:** ρ_low (the memory-regime node sits at P_idle by definition; μ = P_idle/S_node);
constant F (prefill load normalized per-job by ρ_dest(T_j)).

**Primary results to produce** (each a sweep over one axis with the rest at center):
1. **Shed vs. S\*** — does dispatch hit the target; where does it become infeasible (vs. D), for BF16 and FP8.
2. **Certify-vs-expect gap** — active-work certificate vs. future node-drain proxy, with the old `p̄` bracket as reference.
3. **Ranking invariance** — job order vs. {ρ*, MFU, γ} (should be flat); feasibility margin vs. same (should move).
4. **Greedy vs. LP** — bang-per-buck sort vs. the LP; agreement except at constraint boundaries.
5. **Class mix** — class-isolated and mixed-population sweeps showing when ordinary chat, long chat/code, reasoning, or agentic loops set the bottleneck.
6. **Context-length regime** — short→long T mixture sweeping the load-bound → memory-bound transition; shows when the memory term (and γ) starts to matter.

---

## Open items flagged for confirmation

- **Precision is now a primary axis, run end to end** (not a footnote): BF16 (Cap 130 GB, S_node ≈ 15)
  and FP8 (Cap 365 GB, S_node ≈ 43). FP8 is the production-recommended serving mode; BF16 is the
  full-precision baseline. The two shift the memory-regime threshold ~2× but leave the load-regime
  results unchanged — which is itself a result worth showing.
- **P_busy serving plateau:** set at 0.8× TDP from POLCA. If you have the actual measured plateau
  for a 235B-class node from your traces, use it directly and collapse that sweep to a point.
- **ρ_dest(T) is a function, not a constant** — computed per session from its context length via the
  prefill roofline. MFU (∈[0.3,0.5]) is its one swept input. This makes the replay-vs-transfer choice
  context-dependent, matching the single-session boundary already in the draft.
- **Context-length mixture is swept** because ServeGen shows these parameters drift over time; the
  short/center/long settings bracket the load-bound → memory-bound transition.
- **Destination in the no-disagg setting:** rebuild nodes are full serving nodes that rebuild *and*
  serve — implemented: rebuild capacity is ⌊spare⌋ of the same pool that gates load/held admission
  (no dedicated pool). Rebuild-vs-post-rebuild-serving overlap inside [0, D] remains an acknowledged
  approximation (see formulation.md), to be covered by the planner cushion κ pending Track 1 calibration.
