# Destination landing architecture

Status: Queue-Haul implements an aggregate sensitivity prototype in
`destination.py`, `pool_planner.py`, and `destination_evaluation.py`; it is not
an operational admission certificate. The 2026-07-23 run does not support an
accepted destination profile. The recovered raw records expose six invalid
forced-token signatures, widespread append-hot service cells, one excluded
under-hit, and five descriptive private-prefix-consistent cells without strict
completion evidence or an admissible service frontier.
They do support component timing for the recorded concurrency-one v7 request
schedules in the measured 16K/10-Gbps and 24K/5-Gbps cells, plus a provisional
live-traffic migration ranking. See `FINDINGS.md`.

Queue-Haul asks one question: **can a set of active sessions land on warm
destination capacity before a source-power deadline?** It does not simulate a
destination scheduler or predict latency after admission. The first case is the
current mirror: one destination datacenter with the same warm GPT-OSS-20B/A100
serving configuration as the source. Every extension keeps the same sparse
resource-allocation problem.

## Why these resources are sufficient

For a pinned serving class and a workload/concurrency domain already represented
by its measured envelope, five consumable quantities decide whether an
already-warm serving pool can land a session:

1. private-prefix prefill work and decode work;
2. block-rounded private live KV;
3. replay reconstruction work during migration;
4. KV ingestion or promotion work during migration; and
5. bytes on every transport edge used by the migration.

Compatibility, workload affinity, pinned hardware/runtime identity, and warm
model availability are eligibility predicates, not consumable rows. Source
power is the objective and target, not destination capacity. GPU count, FLOPs,
SM occupancy, HBM bandwidth, batch size, and scheduler policy affect measured
service or migration rates; they are not portable capacity constants.

This boundary follows the systems evidence. DistServe and Splitwise show that
prefill and decode consume different serving resources and can favor different
hardware. PagedAttention makes KV capacity a block-residency constraint.
Mooncake and LMCache make the storage hierarchy and promotion path explicit,
while active decoding still requires device-resident state. ServerlessLLM and
AlpaServe show that loading or reallocating models is a separate placement
problem. Sarathi shows why scheduler details should be absorbed into measured
envelopes rather than modeled as portable constants. Orca shows why
iteration-level batching means static request counts alone do not predict
service capacity, and
Llumnix treats migration as a separate dynamic operation rather than permanent
serving load.

Primary sources: [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin),
[Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/),
[PagedAttention](https://arxiv.org/abs/2309.06180),
[vLLM prefix caching](https://docs.vllm.ai/en/v0.22.1/features/automatic_prefix_caching/),
[Orca](https://www.usenix.org/conference/osdi22/presentation/yu),
[SGLang](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf),
[Llumnix](https://www.usenix.org/system/files/osdi24-sun-biao.pdf),
[Mooncake](https://madsys.cs.tsinghua.edu.cn/publication/mooncake-a-kvcache-centric-disaggregated-architecture-for-llm-serving/ToS2025-Qin.pdf),
[LMCache](https://arxiv.org/abs/2510.09665),
[ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu),
[AlpaServe](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan), and
[Sarathi-Serve](https://arxiv.org/abs/2403.02310).

## Objects and units

- A **replica** is one `ServingInstance`; it may use multiple GPUs.
- A **pool** is an ordered set of homogeneous replicas. Routes belong to
  source-method-replica candidates, not to the site or pool in the general
  model; v1 stores one route on a pool because it has one source site.
- A **pool type** fixes model and revision, tokenizer, durable-log contract, KV
  ABI and dtype, hardware, precision, parallel layout, engine version and
  scheduler configuration, measured rates, envelopes, and valid domains.
- A **site** contains pools. Multiple pools may share route links, so adding a
  pool never creates network capacity implicitly.
- A general **candidate** is `c = (session s, method a, replica r)` with derived
  `pool(r)` and `path(s,a,r)`. V1 first uses the aggregate
  `(session, method, pool)` relaxation and packs replicas afterward.

Service work is dimensionless. KV stock is tokens in the current fixed
model/ABI implementation and should become bytes or blocks when mixed KV
layouts are admitted. Migration and route rows use seconds and bytes. Every
capacity has a provenance and measured validity range; unsupported context,
workload direction, load, bandwidth, or compatibility hard-fails.

Replay requires equal model, tokenizer, and durable-log execution contract.
KV transfer additionally requires an exact KV ABI/layout match. A pool may
disable either method. V1 assumes the model is already warm; weights consume
baseline memory but cold loading and model reallocation are a later facility-
opening problem. Destination hardware need not match the source: heterogeneous
replay is eligible only when that destination class has its own measured
profile; KV also requires the exact transferable state contract.

## Quantity and evidence map

The constraint uses direct inventory, measured profiles, live destination
state, and explicit policy. These sources are not interchangeable.

| Quantity | Admission role | Current GPT-OSS-20B/A100 evidence |
|---|---|---|
| pinned replica class \(q\) | compatibility and profile key | model, A100 80 GB, BF16, TP=1, image/runtime campaign record; the current type does not enforce the full tuple |
| \(F_q(T),G_q(T)\) | cold-prefill and decode work coordinates | measured 256–31,562-token curves with declared 25% relative error; cache-conditioned prefill is unprofiled |
| \(\mathcal C_q^m\) | joint TTFT/TPOT/stability service blob | five legacy cache-geometry-consistent sensitivity anchors across three affinities; common point 0.096953; no admissible point or boundary |
| \(b_r\) | current per-replica traffic point | must use profile-compatible prompt/decode telemetry; v7 `achieved_rho` is invalid |
| \(K_q\) | allocatable live-KV stock after fixed runtime memory | 963,152-token vLLM 0.22.0 readback at 0.75 GPU-memory utilization |
| private KV blocks | block-rounded per-session residency | target v1 accounting; current code sums unrounded tokens and gives no sharing credit |
| replica inventory | assignment and fragmentation | direct site input; pool multiplication is only a relaxation |
| migration components | deadline and temporary endpoint occupancy | empirical-max replay/KV envelopes for the recorded concurrency-one v7 schedules; no statistical coverage guarantee |
| route edges and rates | transport feasibility | scenario topology plus shaped link rates; geography/fleet QoS remain inputs |
| workload scenarios | horizon-specific arrivals, contexts, and prefix identity | modeled input; not a measurement of fleet behavior |
| policy | requested SLO, headroom, and confidence rule | explicit operator input |
| guaranteed cache state | session-history compute reuse | replay/KV installs private history; append-hot repeated prompts are excluded |

Fixed model weights, activations, graph captures, and engine workspace do not
need separate optimization rows for an already-warm pinned replica. They are
already removed from measured \(K_q\) and embodied in its measured service
blob. A different engine flag, accelerator layout, model, or memory partition
defines another replica class and requires its own evidence.

## Private-prefix v1 and contested situations

The target v1 contract credits only the active session's migrated history; its
next append remains new work. That history must be installed and protected
until the credited request begins. V7 observes request-time reuse but does not
prove protected residency under arbitrary pressure. Without a runtime
reservation, prospective prefix-compute credit is sensitivity-only. Per replica
and uncertainty case, target private-KV accounting is

\[
B_{r,\omega}^{0,blocks}+
\sum_{s,a}
\left\lceil\frac{\widehat T_{s,\omega}(H_r)}
{L_{q(r)}^{block}}\right\rceil z_{s,a,r}
\le K_{q(r)}^{blocks}\qquad\forall r,\omega.
\]

This intentionally overstates physical memory when unrelated sessions happen
to share exact prefixes. The result is uncredited sharing headroom and possible
false-negative admission, not a prediction that those prefixes will miss.
Cross-session block unions remain a later optimization.

A separate destination is described by which constraint is contested:

| Situation | Binding state or resource | Admission consequence |
|---|---|---|
| incompatible or cold | pinned model/runtime/hardware profile or warmness | no candidate |
| service-contested | baseline plus private-prefix prefill/decode reaches an observed or assumed envelope | reduce admitted work or label sensitivity |
| KV-contested | block-rounded private histories and growth reach allocatable HBM KV | reject or choose another replica |
| packing-contested | aggregate pool stock fits but indivisible sessions do not fit replicas | repair assignment or reject |
| route-contested | source-method-replica paths share residual edge bandwidth/latency | schedule later, select another route, or reject deadline |
| endpoint-contested | replay compute, KV ingest/copy, source stream, or replica migration slot serializes work | schedule explicitly and check makespan |
| foreground-impact-contested | migration overlaps latency-sensitive serving without an accepted impact bound | require idle/drained state or report possible |
| stale or unreserved | replica, KV, endpoint, or residual-route state changed after planning | reacquire a fresh atomic lease or reject |

These situations are intersections, not alternative destination types. A site
may be service-, KV-, and route-contested simultaneously; the planner reports
the largest modeled relaxation pressure. Concrete packing and deterministic
execution validation cover only the resources explicitly represented.

Operational `feasible` requires more than the current prototype enforces: an
accepted evidence status, full pinned runtime and warm/healthy attestation,
fresh baseline and residual-route state held by a lease through commit,
per-session KV block rounding, a supported service horizon, and either an
idle/drained migration window or an accepted foreground-impact bound. Until
those gates exist, architecture results are descriptive or
`sensitivity/possible` even if the optimizer returns a placement. The simulator
may model destination action power, but destination facility power and site
caps are not admission constraints; either would require a separate measured
resource row.

## Mirrored destination

Let \(z_{s,a,r}\) select migration method \(a\) and concrete destination replica
\(r\) for source session \(s\), with
\(y_s=\sum_{a,r}z_{s,a,r}\le1\). The conservative source-power target is

\[
\sum_s w_s y_s \ge \Delta P,
\]

where \(w_s\) is the removable source-power lower bound for the whole session.
Exact source power is reevaluated after integer selection.

For destination type `q` and concrete replica `r`, convert horizon-specific
request forecasts to portable measured work:

\[
d_{s,q,r,\omega}=\left(
p_{s,q,r,\omega},\;
\delta_{s,q,\omega}
\right).
\]

\(p_{s,q,r,\omega}\) is pinned-replica service-seconds per wall-second from a
measured cache-conditioned function
\(\tau_q^P(T^{full},T^{hit},T^{miss})\), and
\(\delta_{s,q,\omega}\) is horizon-average decode work:

\[
p_{s,q,r,\omega}=
\frac{1}{H_s}\sum_{j\in requests(H_s)}
\tau_{q,\omega}^P(T_j^{full},T_j^{hit},T_j^{miss}),\qquad
\delta_{s,q,\omega}=
\frac{1}{H_s}\int_0^{H_s}
\frac{g_{s,\omega}(t)}{G_{q,\omega}(T_s(t))}\,dt.
\]

A cache hit cannot be modeled by merely subtracting hit tokens from a
cold-prefill curve: the suffix still attends to the cached prefix. Credit
requires that the same exact pinned-engine blocks be available before prefill
and protected in the KV constraint; if migration supplies them, its schedule
must enforce install before use. The current
\(p=f_s/F_q(T_s)\) is an append-token/cold-rate normalization coordinate, not
a conservative physical cached-prefill bound. It describes five legacy
cache-geometry-consistent cells in their affinity, context, and concurrency
domains; because their stream completion is unobservable, even use at those
cells is sensitivity analysis until strict evidence and a service envelope or
cache-conditioned work model are accepted. Prefix caching does not reduce
generation-phase work for the same context and output.

\(G_q\) is measured for the complete serving configuration and keyed by live
sequence length. The current use of fixed rates and one \(T_s\) is a measured
simplification. A physical guarantee integrates the supported context
trajectory as above or uses the worst supported rate over the projected range.
General profiles must state their cache/full/miss/live-context keys, horizon,
arrival/length distribution, and concurrency domain.
Workload direction is the fraction of normalized work due to prefill, not raw
input-token fraction.
Each \(\omega\in\Omega\) jointly selects a demand forecast and empirical
profile case, including \(F_{q,\omega}\), \(G_{q,\omega}\),
\(N_{q,\omega}\), and \(h_{q,\omega}^m\). Common nonnegative facet normals
define nested policy envelopes:

\[
\mathcal C_{q,\omega}^m=
\{d\ge0:N_{q,\omega}d\le h_{q,\omega}^m\},\qquad
h_{q,\omega}^{normal}\le h_{q,\omega}^{emergency}
\le h_{q,\omega}^{stable}.
\]

V1 has one central case and common normals; multi-case uncertainty is target
semantics.

The smallest useful resource model has one facet, `f/F + g/G <= h`. More
facets are allowed only when valid held-out mixed-load data reject it. A run
with missing token work or usage is measurement-invalid, not a capacity point.
After excluding such runs, the v7 data contains only successful inner
observations and cannot select another facet. For a pool with scenario-indexed
baseline work `b_{p,\omega}`, the necessary aggregate pruning relaxation is

\[
N_{q,\omega}\left(
b_{p,\omega}+\sum_{s,a,r\in p}d_{s,q(r),r,\omega}z_{s,a,r}
\right)\le |p|h_{q,\omega}^m
\qquad \forall p,\omega.
\]

Normal and emergency are independently solved operator policies. Stable is the
outer hard-safety ceiling used only by the execution validator. A baseline
already outside the selected envelope makes that pool unavailable. Until a
service envelope is identified, `h` remains an explicit sensitivity input.
V7 supplies five descriptive private-prefix-consistent anchors, not admissible
service points: coding has one, the other affinities have two, and none is a
failure. The colder coding under-hit is excluded rather than promoted through
an untested cache-monotonicity assumption. A placement that depends on these
legacy anchors, synthetic `h`, interpolation between affinity rays, or the
assumed one-facet shape is **sensitivity/possible** even if it survives every
chosen value. Anything outside the measured domain is unsupported. V7 does not
identify a global conservative `h` or a mixed-affinity convex blob.

The simple destination constraint is a nonanticipative existence statement.
Let \(\omega\in\Omega\) jointly index demand forecasts and empirical profile
uncertainty. For binary \(z_{s,a,r}\), one assignment and one scheduling policy
\(\pi\) must work for every declared case:

\[
\exists z,\pi:\quad
\begin{cases}
\sum_{a,r}z_{s,a,r}=y_s,\quad y_s\in\{0,1\} & \forall s,\\
z_{s,a,r}\le E_{s,a,r} & \forall s,a,r,\\
\sum_s w_sy_s\ge\Delta P,\\
b_{r,\omega}+\sum_{s,a}d_{s,q(r),r,\omega}z_{s,a,r}
\in\mathcal C_{q(r),\omega}^m & \forall r,\omega,\\
\operatorname{KVPrivate}_{r,\omega}(z)\le K_{q(r)} & \forall r,\omega,\\
\operatorname{makespan}(\operatorname{Schedule}(\pi,z,\omega))
\le H_m & \forall\omega,\\
\operatorname{ImpactOK}_{\omega}(z,\pi) & \forall\omega.
\end{cases}
\]

\(\pi\) is a history-dependent policy selected before uncertainty is known. It
maps observed completions to actions but cannot read future case information.
\(w_s\) is a simultaneous lower bound over declared source-power cases, and
\(E_{s,a,r}=\min_\omega E_{s,a,r,\omega}\) requires eligibility in every case.

The source sessions and destinations are therefore chosen jointly. A session
that looks attractive for source-power relief but has no feasible
`(method, replica)` assignment is not selectable. Multiple \(\omega\) cases and
the complete pinned compatibility predicate are target semantics; v1 accepts
only its central case and a smaller fingerprint.

Let `H_m = deadline - controller_delay - power_window` be the migration
horizon and `H_r` the latest claimed residency horizon. They are deliberately
separate. Live-state admission is

\[
B_{r,\omega}^0+
\sum_{s,a}
\left\lceil
\frac{\widehat T_{s,\omega}(H_r)}{L_{q(r)}^{block}}
\right\rceil z_{s,a,r}
\le K_{q(r)}^{blocks}.
\]

Target v1 charges each session's history and projected growth independently. It does
not require block identities and gives no cross-session prefix-sharing credit.
The current implementation still sums unrounded token equivalents, so
block-rounding is target semantics rather than a present physical guarantee.
A later shared-KV extension may replace the sum with an exact protected-block
union without changing the other constraints.

Replay contributes reconstructed context work and durable-log bytes. Every
log or sealed-KV byte quantity is evaluated at the snapshot selected by
\(\pi\); append and catch-up bytes remain separate phases. KV
transfer contributes sealed-state bytes and destination ingestion/promotion
work. Exact transport time must not be scaled by a runtime calibration. The
desired component models are

\[
\tau_{s,R,q} =
\tau_{route}(B_s^{log},path(s,R,r))+
\alpha_{R,q}\tau_{s,R}^{compute+completion,old}(T_s)+\tau_q^{switch}
\]

and, for the measured KV primitive,

\[
\tau_{s,K,q} =
\max\left(
\tau_{route}(B_s^{sealed},path(s,K,r)),
\frac{B_s^{sealed}}{C_{ingest,q}}
\right)+c_{K,q}(T_s),
\]

plus separately measured catch-up and route-switch terms when they are not
already included in `c`. The schedule assigns residual bandwidth after
background reservations, and
\(\tau_{route}(B,path)\ge B/\min_{e\in path}B_e^{alloc}\). A portable WAN model
must additionally measure fixed or per-round latency; v7 validates only the
shaped local route and cannot supply a geographic-WAN term. A load term may be
added only after migration-interval work identifies one. A candidate whose
predicted duration exceeds `H_m` is invalid.

The fitted v7 KV residual is `observed - route_floor`; it can be reused in the
explicit `max(route, ingest)` form only where route time is the measured
bottleneck. Otherwise the residual must be redefined or refit.

The v7 evidence supports empirical envelopes only for its recorded
concurrency-one request schedules in the measured 16K/10-Gbps and
24K/5-Gbps cells; “one overlapping request” alone is not a workload bound.
Replay keeps the old context curve with a compute/completion calibration, while
KV uses `sealed_bytes / route_bytes_per_s + c`. The current scalar
`LoadedCoefficients` multiplies the complete duration and cannot encode these
physical semantics safely, so no v7 profile is emitted.

Migration method also carries a provisional foreground-impact policy.
Twelve v7 treatments overlap foreground work. The one request arriving during
replay incurred 1.084 s additional TTFT, versus 4.7 ms for the matched KV
request. Until a larger sample establishes percentiles, a latency-sensitive
busy pool may rank KV ahead of replay, but that observation does not prove a
tail-SLO bound. A robust result currently requires idle/drained foreground or
an explicit impact bound within the exact measured overlap domain. This policy
is separate from steady service capacity and is not encoded by current types.

For every exact route edge `e`,

\[
\sum_c b_{c,e}x_c\le B_e^{alloc}H_m.
\]

Here \(x_c\) is the candidate view of \(z_{s,a,r}\), and the path is
source-method-replica specific. This byte row is a necessary fluid relaxation,
not an exact concurrent schedule. The target schedule reserves per-time
residual capacity on every edge, source stream, destination ingest/copy engine,
and replica migration slot. Any temporary staging allocation must become an
explicit measured row; current code has none. Migration concurrency remains
one per replica in v1. Network transfer and destination ingestion may overlap
only when the measured primitive supports it.

The older `../evacuation` formulation already separated network, replay
prefill, state ingestion, and KV residency. Queue-Haul adds steady service,
whole-session selection, source-power gain, destination baselines, exact
routes, concrete replica packing, and independent execution validation.

## Sparse general form

All solvers consume the same candidate table. Its session-incidence matrix
`A` and resource matrix `U` express the current linear pruning relaxation:

\[
Ax\le\mathbf1,\qquad Ux\le\mathbf1,\qquad 0\le x\le1.
\]

Each column contains session and pool identity, method, source-power gain,
migration work, pool service work, residency, method occupancy, and exact
route-link bytes. The resource rows are only:

| Row | Capacity after baseline | Horizon |
|---|---:|---|
| pool service facet | `|p| h - N b_p` | steady state |
| additive pool live KV | `|p| K - K_p^0` | `H_r` |
| replica migration occupancy | replicas × `H_m` | `H_m` |
| source stream | streams × `H_m` | `H_m` |
| route-edge fluid bound | allocatable bytes/s × `H_m` | `H_m` |

Per-replica baseline work and KV are preserved. Supplying aggregate baseline
fields and destination `SimSession` backgrounds simultaneously is rejected to
prevent double counting. Target v1 is block-rounded and additive; current code
uses an unrounded token approximation. A later
shared-prefix extension would require nonadditive prefix-group or block-union
accounting in concrete packing.

Optimization is lexicographic: meet the conservative power target, then
minimize migration work. If no valid plan meets the target, maximize valid
power shed and then minimize work. The result is `target_unmet` with an
explicit watt shortfall; it is never described as safe completion.

## Packing and execution

Aggregate pool feasibility is not enough because sessions are indivisible.
Selected sessions are packed only inside their chosen pool, ordered
deterministically by worst service-facet, KV, and migration pressure. A failed
assignment adds a cut and re-solves. The plan reports repair count and repair
time separately.

V1 checks each returned additive assignment but uses a deterministic greedy
packer plus selection cuts, so it may reject a pool-feasible set that another
replica assignment could place. The small exact assignment routine is a test
oracle, not the production fallback.

The destination execution validator independently checks each concrete replica
against the stable envelope and additive KV capacity. Migration timing is
validated separately by prediction and discrete-event execution simulation.
The simulator is authoritative only for its deterministic schedule under fixed
healthy modeled resources; it has no failure, lease-expiry, or rollback model
and does not predict continuous-batching latency. Execution must revalidate the
lease and retain source ownership until commit.

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

GPT-OSS-20B/A100 has measured anchors, KV capacity, migration correctness, and
migration components, but no accepted service envelope or continuous
loaded-migration curve.
Non-A100 profiles are synthetic sensitivity cases. Continuous destination
load, continuous-batching simulation, replanning, cold sessions, model loading,
concurrency above one, and predictive latency claims are out of scope.

The service facet is a fluid admission model. It is valid only after evidence
validity passes, every workload cell has a feasible/infeasible bracket, and
held-out cells show no false-feasible placement. A successful censored point
is only an inner observation. If valid request-shape or arrival-burst evidence
still separates cells with the same normalized direction, add a measured
affinity or arrival-envelope row rather than facets fitted to invalid runs.

`DATA_TO_COLLECT.md` is the evidence contract for every coefficient and claim.
Absence of `DestinationArchitecture` invokes the exact legacy adapter and must
preserve prior results.
