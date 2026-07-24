# Destination landing architecture

Status: Queue-Haul implements the v1 admission model in `destination.py`,
`pool_planner.py`, and `destination_evaluation.py`. The 2026-07-23 run does
not support an accepted destination profile: its compact records cannot
identify a service frontier or migration interference. They do support
exploratory component timing. Recover the archived request and engine records
before collecting more GPU data; see `FINDINGS.md`.

Queue-Haul asks one question: **can a set of active sessions land on warm
destination capacity before a source-power deadline?** It does not simulate a
destination scheduler or predict latency after admission. The first case is the
current mirror: one destination datacenter with the same warm GPT-OSS-20B/A100
serving configuration as the source. Every extension keeps the same sparse
resource-allocation problem.

## Why these resources are sufficient

Modern serving systems expose many mechanisms, but only five quantities decide
whether an already-warm serving pool can land a session:

1. ongoing prefill and decode service work;
2. live KV residency;
3. replay reconstruction work during migration;
4. KV ingestion or promotion work during migration; and
5. bytes on every transport edge used by the migration.

Compatibility and warm model availability are eligibility predicates, not
consumable rows. Source power is the objective and target, not destination
capacity. GPU count, FLOPs, SM occupancy, HBM bandwidth, batch size, and
scheduler policy are mechanisms whose effects must already be represented by
measured service or migration rates.

This boundary follows the systems evidence. DistServe and Splitwise show that
prefill and decode consume different serving resources and can favor different
hardware. PagedAttention makes KV capacity a block-residency constraint.
Mooncake and LMCache make the storage hierarchy and promotion path explicit,
while active decoding still requires device-resident state. ServerlessLLM and
AlpaServe show that loading or reallocating models is a separate placement
problem. Sarathi shows why scheduler details should be absorbed into measured
envelopes rather than modeled as portable constants.

Primary sources: [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin),
[Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/),
[PagedAttention](https://arxiv.org/abs/2309.06180),
[Mooncake](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf),
[LMCache](https://arxiv.org/abs/2510.09665),
[ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu),
[AlpaServe](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan), and
[Sarathi-Serve](https://arxiv.org/abs/2403.02310).

## Objects and units

- A **replica** is one `ServingInstance`; it may use multiple GPUs.
- A **pool** is an ordered set of replicas with one type and route.
- A **pool type** fixes model, tokenizer, durable-log contract, KV ABI, hardware,
  precision, parallel layout, engine configuration, measured rates, envelopes,
  and valid domains.
- A **site** contains pools. Multiple pools may share route links, so adding a
  pool never creates network capacity implicitly.
- A **candidate** is `c = (session s, method a, pool p)`, where v1 methods are
  `replay` and `kv_transfer` and `s` is active.

Service work is dimensionless. KV stock is tokens in the current fixed
model/ABI implementation and should become bytes or blocks when mixed KV
layouts are admitted. Migration and route rows use seconds and bytes. Every
capacity has a provenance and measured validity range; unsupported context,
workload direction, load, bandwidth, or compatibility hard-fails.

Replay requires equal model, tokenizer, and durable-log execution contract.
KV transfer additionally requires an exact KV ABI/layout match. A pool may
disable either method. V1 assumes the model is already warm; weights consume
baseline memory but cold loading and model reallocation are a later facility-
opening problem.

## Mirrored destination

Let `x_s^R` and `x_s^K` select replay or KV transfer for source session `s`, and
let `y_s = x_s^R + x_s^K <= 1`. The source-power target is

\[
\sum_s w_s y_s \ge \Delta P,
\]

where `w_s` is the conservative removable source power of the whole session.

For destination type `q`, convert expected token rates at current context `T_s`
to portable measured work:

\[
d_{s,q}=\left(f_s/F_q(T_s),\;g_s/G_q(T_s)\right).
\]

`F_q` and `G_q` are context-conditioned prefill and decode rates measured for
the complete serving configuration, not theoretical hardware throughput.
Workload direction is the fraction of normalized work due to prefill, not the
raw fraction of tokens.
Common nonnegative facet normals define nested policy envelopes:

\[
\mathcal C_q^m=\{d\ge0:N_qd\le h_q^m\},\qquad
h_q^{normal}\le h_q^{emergency}\le h_q^{stable}.
\]

The smallest useful resource model has one facet, `f/F + g/G <= h`. More
facets are allowed only when valid held-out mixed-load data reject it. Before
fitting any facet, every failed point must be checked for componentwise
dominance by a successful point using realized per-request work. A dominance
violation is evidence for a missing eligibility, request-shape, or burst
variable, not another resource facet. For a pool with baseline work `b_p`,
admission requires

\[
N_q\left(b_p+\sum_s d_{s,q}y_s\right)\le |p|h_q^m.
\]

Normal and emergency are independently solved operator policies. Stable is the
outer hard-safety ceiling used only by the execution validator. A baseline
already outside the selected envelope makes that pool unavailable. Until a
service envelope is identified, `h` remains an explicit sensitivity input:
placements can be robust across the conservative range, possible in only some
cases, or unsupported outside the measured domain.

Let `H_m = deadline - controller_delay - power_window` be the migration
horizon and `H_r` the latest claimed residency horizon. They are deliberately
separate. Live-state admission is

\[
K_p^0+\sum_s k_s(H_r)y_s\le |p|K_q.
\]

Replay contributes reconstructed context work and durable-log bytes. KV
transfer contributes sealed-state bytes and destination ingestion/promotion
work. Exact transport time must not be scaled by a runtime calibration. The
desired component models are

\[
\tau_{s,R,q} =
\frac{B_s^{log}}{b_{route}}+
\alpha_{R,q}\tau_{s,R}^{compute,old}(T_s)+c_{R,q}
\]

and, for the measured KV primitive,

\[
\tau_{s,K,q} =
\frac{B_s^{sealed}}{b_{route}}+c_{K,q}(T_s,u),
\]

plus separately measured catch-up and route-switch terms when they are not
already included in `c`. If network and ingestion overlap, the transfer term
becomes the slower supported stage, never less than
`sealed_bytes / route_bytes_per_s`. A load term may be added only after
migration-interval work identifies one. A candidate whose predicted duration
exceeds `H_m` is invalid.

The v7 evidence supports only exploratory low-work coefficients: replay keeps
the old context curve with a compute/completion calibration, while KV uses
`sealed_bytes / route_bytes_per_s + c`. The current scalar
`LoadedCoefficients` multiplies the complete duration and cannot encode these
physical semantics safely, so no v7 profile is emitted.

For every exact route edge `e`,

\[
\sum_c b_{c,e}x_c\le B_eH_m.
\]

This retains shared source/WAN cuts and prevents capacity borrowing across
routes. Migration concurrency remains one per replica in v1. Network transfer
and destination ingestion may overlap when the measured primitive does so; the
duration model uses the measured setup plus the slower stage rather than
adding both blindly.

The older `../evacuation` formulation already separated network, replay
prefill, state ingestion, and KV residency. Queue-Haul adds steady service,
whole-session selection, source-power gain, destination baselines, exact
routes, concrete replica packing, and independent execution validation.

## Sparse general form

All solvers consume the same candidate table. Its session-incidence matrix
`A` and resource matrix `U` express the complete relaxation:

\[
Ax\le\mathbf1,\qquad Ux\le\mathbf1,\qquad 0\le x\le1.
\]

Each column contains session and pool identity, method, source-power gain,
migration work, pool service work, residency, method occupancy, and exact
route-link bytes. The resource rows are only:

| Row | Capacity after baseline | Horizon |
|---|---:|---|
| pool service facet | `|p| h - N b_p` | steady state |
| pool live KV | `|p| K - K_p^0` | `H_r` |
| replica migration occupancy | replicas × `H_m` | `H_m` |
| source stream | streams × `H_m` | `H_m` |
| exact route edge | bytes/s × `H_m` | `H_m` |

Per-replica baseline work and KV are preserved. Supplying aggregate baseline
fields and destination `SimSession` backgrounds simultaneously is rejected to
prevent double counting.

Optimization is lexicographic: meet the conservative power target, then
minimize migration work. If no valid plan meets the target, maximize valid
power shed and then minimize work. The result is `target_unmet` with an
explicit watt shortfall; it is never described as safe completion.

## Packing and execution

Aggregate pool feasibility is not enough because sessions are indivisible.
Selected sessions are packed only inside their chosen pool, ordered
deterministically by worst service-facet, KV, and migration pressure. A failed
assignment adds a cut and re-solves. The plan reports repair count and repair
time separately, and small instances are checked against exact replica
assignment.

The execution validator independently checks each concrete replica against the
stable envelope and KV capacity. It validates admission and migration timing;
it does not claim to predict continuous-batching latency.

Each outcome independently reports `admission_mode`, `feasible`,
`power_shortfall_w`, `failure_reason`, packing repairs, and predicted migration
makespan. Normal is attempted before emergency.

## Memory tiers without scope creep

A general tiered pool can expose stock `M_{p,t}` and promotion-edge work for
HBM, DRAM, and SSD. In v1, an active landed session must fit the live HBM KV
row. DRAM or SSD may stage a transfer but cannot substitute for HBM unless a
future lazy-retrieval mode has its own measured latency and service envelope.
Public hardware specifications provide sensitivity capacities, not claims
about operator headroom.

## Deliberate extension path

1. **Empty mirror:** one warm pool, current source-equivalent type and route.
2. **Loaded mirror:** exact baseline service and KV on that pool.
3. **Tier-aware staging:** explicit lower-tier stock and promotion work, while
   HBM remains mandatory for active landing.
4. **Multiple pools or sites:** duplicate candidate columns and add their pool
   and exact route rows; no new solver abstraction.
5. **Heterogeneous hardware:** add measured or explicitly synthetic pool types;
   the variables remain service work, KV, ingestion, and bytes.

The evaluation begins with the mirror and then varies initial destination load
`rho`, effective normal headroom `H`, and pool count `P`. Until measured
headroom and migration interference are identified, `rho` and `H` are
sensitivity variables rather than calibrated probabilities. Pool panels
isolate fragmentation/fungibility at fixed total resources; little difference
under a homogeneous layout is a valid result. Multiple sites are represented
by routes, not by another capacity abstraction.

GPT-OSS-20B/A100 has measured anchors, KV capacity, correctness, and migration
components, but no accepted service envelope or loaded-migration curve.
Non-A100 profiles are synthetic sensitivity cases. Continuous destination
load, continuous-batching simulation, replanning, cold sessions, model loading,
concurrency above one, and predictive latency claims are out of scope.

The service facet is a fluid admission model. It is valid only after every
workload cell has a feasible/infeasible bracket and held-out cells show no
false-feasible placement. A censored lower point is not capacity. If
request-shape or arrival-burst effects still separate cells with the same
normalized direction, add a measured arrival-envelope row rather than facets
fitted to failed or under-sampled runs.

`DATA_TO_COLLECT.md` is the evidence contract for every coefficient and claim.
Absence of `DestinationArchitecture` invokes the exact legacy adapter and must
preserve prior results.
