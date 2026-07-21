# Proposed destination architecture

## Goal

Replace the empty destination with measured, pool-level admission constraints
without modeling batching, request scheduling, autoscaling, or a provider's
cluster router. Queue-Haul targets a logical pool; the provider router chooses a
replica. The existing simulator remains authoritative for migration timing,
network contention, endpoint queues, commits, and source power.

## Abstraction

A **pool type** is a serving-equivalence class: model/version, hardware,
precision, parallel layout, engine configuration, KV representation, and
measured service envelope.

A **destination pool** is a site-scoped set of warm, interchangeable replicas
of one pool type. It has a route, existing non-movable sessions, resident KV,
and method-specific eligibility. Pools with different routes, capacity, or
compatibility are distinct. Replay may target a broader pool type than KV
transfer, which requires matching model and KV representation.

Existing destination demand is represented by active `movable=False`
`SimSession`s. This reuses current compute, KV, and power accounting instead of
adding reserved-load fields.

## Measured service envelope

Session demand is

\[
d_j=(f_j,g_j),
\]

where `f` and `g` are expected prefill and decode token rates. For pool type
`q` and operating envelope `m`, measured capacity is a conservative polytope:

\[
\mathcal C_q^m=\{d\ge0:A_q^m d\le\mathbf 1\}.
\]

The current scalar load `f/F + g/G` is a one-facet special case. Start with one
facet and add at most two or three only if held-out mixed workloads show
systematic error.

Store separate normal, emergency, and stable destination envelopes. They must
not reuse or extend `max_ell`, which is bounded by the source power calibration.
Normal and emergency are two SLO-selected regions measured with one fixed
serving configuration; stable is the no-runaway-queue/failure boundary.

## Optimization

Create one candidate `c = (session j, method a, pool p)` with variable
`x_c in [0, 1]`. Each session receives at most one candidate:

\[
\sum_{c:j(c)=j}x_c\le1.
\]

Retain the source-power requirement:

\[
\sum_c g_{j(c)}x_c\ge\Delta P.
\]

Each candidate contributes one sparse column to the existing resource matrix
`Ux <= 1`. Pool `p`, with `n_p` replicas and baseline demand `b_p`, adds:

\[
A_q^m\left(b_p+\sum_{c:p(c)=p}d_{j(c)}x_c\right)
\le n_p\mathbf1,
\]

\[
K_p^0+\sum_{c:p(c)=p}k_{j(c)}(H)x_c\le n_pK_q,
\]

plus pool-specific replay-time and KV-ingestion-time rows using loaded endpoint
measurements. Source-stream, link, deadline, and power rows are unchanged.

Solve normal and emergency as separate ordinary LPs. Prefer a normal-feasible
plan; use emergency only when normal cannot meet the source target. Report
`normal`, `degraded`, or `unsafe`; do not add an LP mode-selection variable.

LP, rounding, and greedy policies consume the same candidate columns. Greedy
selects the feasible method/pool candidate with the best source-power gain
relative to remaining resource pressure. Rounding chooses at most one candidate
per session.

After selection, group sessions by pool and use balanced whole-session packing
within that pool. Emit both the pool target and the concrete simulation
instance. Packing cannot silently use another pool. The simulator validates
final service and KV capacity but does not simulate continuous batching or
predict destination latency.

## Two-A100 measurements

Use GPU 0 as migration source and GPU 1 as a loaded destination. Drive
destination foreground traffic with open-loop arrivals.

### Service envelope

Measure prefill-heavy, decode-heavy, interactive-coding, coding, and agentic
workloads at approximately six offered-load points spanning saturation, with
three repeats. Fit repeats 0-1 and hold out repeat 2 plus one mixed workload.
Record prompt/output throughput, completion rate, queue slope, queue time,
TTFT, TPOT/ITL, goodput, running/waiting requests, KV usage, preemption,
rejections, OOMs, restarts, and GPU telemetry.

Derive conservative normal, emergency, and stable capacity facets. Stability
requires no failures and no positive sustained queue slope. Preserve the full
frontier so SLO choices can change without rerunning hardware.

### Migration under load

Run replay and KV transfer at destination loads `0`, `0.7`, and `0.9` of normal
capacity; contexts near 4K, 16K, and 32K; 1 and 10 Gbps; and three repeats.
Reuse paired no-migration controls. Record migration readiness, work/bytes,
ingestion rate, foreground goodput and latency change, queue slope, KV pressure,
continuation, power, and failures.

Reduce these observations to load-bucketed replay/KV duration and safe
concurrency. Normal planning uses coefficients measured near the normal
boundary; emergency planning uses coefficients near its boundary. Hard-fail
outside measured context, load, bandwidth, or concurrency ranges.

Hold out an approximately 24K-context, 0.8-load mixed-method case and a
different foreground mix for end-to-end validation.

## Evaluation axes

The primary axes are:

1. Compatible warm capacity ratio
   `R in {0.5, 1, 2, 4}` relative to displaced source demand.
2. Baseline normalized serving load
   `rho in {0, 0.5, 0.7, 0.85, 0.95}`.
3. Measured emergency squeeze, the radial emergency/normal capacity ratio along
   the workload mix, reported with uncertainty.

Logical pool count is a topology ablation at fixed total capacity, for example
`P in {1, 2, 4, 8}`. Secondary overlays are KV occupancy, equal versus
concentrated capacity, workload-mix mismatch, correlated site load, and
faster/central/slower ingestion profiles. Do not take their full Cartesian
product.

## Reporting

Report:

- measured normal/emergency/stable prefill-decode frontiers with absolute
  TTFT, TPOT, and goodput;
- migration completion and foreground impact versus destination load;
- `R x rho` feasibility heatmaps for normal and emergency envelopes;
- source power shed and requested evacuation achieved;
- service, KV, replay, ingestion, WAN, and source-stream bottleneck shares;
- LP-versus-greedy quality, runtime, and memory;
- predicted-versus-measured held-out feasibility and migration time.

Claims are limited to measured two-A100 behavior and explicit sensitivity
regimes. Public data motivate localized, heterogeneous, bursty capacity but do
not establish hyperscaler utilization, spare-GPU distributions, router
behavior, or multi-replica linear scaling.

## Implementation order

1. Add explicit pool identity and destination service-envelope profile data.
2. Generate non-movable destination background sessions.
3. Preserve exact single-empty-pool behavior.
4. Generalize options to `(session, method, pool)` candidates shared by LP and
   greedy solvers.
5. Restrict whole-session packing to the selected pool.
6. Add final pool service-envelope validation.
7. Add measurement reduction and held-out validation.

Required regressions include single-pool equivalence, no cross-pool borrowing,
alternate-pool selection, one candidate per session, background compute/KV
accounting, pool-preserving packing, normal/emergency row isolation, hard
failure on unpackable plans, and exact-simulator rejection of temporally
unschedulable aggregate plans.
