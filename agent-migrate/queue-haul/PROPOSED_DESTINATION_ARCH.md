# Proposed destination architecture

Status: proposal. None of this exists yet. The current simulator has one
empty destination site mirroring the source, gated by scalar `max_ell` and
KV tokens. Values marked `??` are assumptions Grant must confirm in
`assumptions.md` before experiments run.

## Physical setup

A source site serving live LLM sessions receives a curtailment request: shed
`ΔP` watts of local power by deadline `D`. The planner snapshots the fleet at
`t = 0`, solves one static plan choosing which whole sessions migrate over a
shared WAN to destination pools, and the discrete-event simulator executes
that plan; there is no replanning. Sessions not selected stay at the source
and keep running — the source remains partially powered, and source sleep or
shutdown stays opt-in per plan. A session is a live chat or agent
conversation: GPU-resident KV cache in sealed immutable blocks, a durable
log at the source DC for replay, and expected prefill/decode token rates.

One operator runs both source and destinations, so the planner sees
destination baseline demand, resident KV, and non-movable sessions directly;
this is a private-fleet admission problem, not an external service.

Containment hierarchy and glossary, used consistently below:

- **site** — a datacenter; contains pools. All destination pools sit behind
  the one shared source-egress/WAN/ingress cut unless the topology ablation
  says otherwise. Adding pools does not add WAN capacity.
- **pool** — a site-scoped set of warm replicas of one pool type.
- **replica** — one serving engine on one GPU; identical to the simulator's
  `ServingInstance`; "endpoint" is not used below.
- **warm** — engine process running, weights loaded, spare KV headroom;
  admission needs no model load or cold start.
- **route** — the ordered tuple of network links from source to a pool.

Testbed mapping: GPU 0 stands in for the entire source site; GPU 1 for one
destination replica, whose measured envelope is scaled to `n_p` replicas
under a stated assumption (see Replica scaling). The 1/10 Gbps "WAN" between
them is a rate-shaped emulation inside one node, not a physical WAN.

## Abstraction

A **pool type** is a serving-equivalence class: model/version, hardware,
precision, parallel layout, engine configuration, KV representation, and
measured service envelope.

A **destination pool** is a site-scoped set of warm, interchangeable replicas
of one pool type. It has a route, existing non-movable sessions, resident KV,
and method-specific eligibility. Pools with different routes, capacity, or
compatibility are distinct.

Method eligibility predicate: KV transfer requires every pool-type field to
match the session's source configuration; replay requires only matching
model/version; `replay_on_request` follows the replay predicate. Precision
or engine-config differences under replay are allowed and flagged in output.

Existing destination demand is represented by active `movable=False`
`SimSession`s. This reuses current compute, KV, and power accounting instead
of adding reserved-load fields.

## Measured service envelope

Session demand is `d_j = (f_j, g_j) := (expected_f, expected_g)` — the
existing steady-state prefill and decode token rates on `SimSession`,
unchanged and not re-estimated from request traces. Growth enters only the
KV row via `_resident_tokens`. The source-power coefficient is renamed
`w_j = P(L_s) - P(L_s - ell_j)` (watts) to end the symbol collision with
decode rate `g_j`; `formulation.md` is updated to match.

For pool type `q` and envelope `m in {normal, emergency, stable}`, measured
capacity per replica is a polytope

\[
\mathcal C_q^m=\{d\ge0:A_q^m d\le\mathbf 1\}.
\]

The current scalar load `f/F + g/G` is the one-facet special case. Normal
and emergency are SLO-selected regions; stable is the
no-runaway-queue/no-failure boundary. The stable envelope is never an LP
mode: it is the conservatism floor for the emergency fit and the surface the
simulator validates final placements against. Envelopes are per pool type,
measured once, and must not reuse or extend `max_ell`, which is bounded by
the source power calibration and survives only on the source side.

Preregistered fitting procedure:

1. SLOs (`??` until confirmed): normal = p90 TTFT ≤ 2 s and p90 TPOT ≤
   100 ms; emergency = p90 TTFT ≤ 10 s and p90 TPOT ≤ 250 ms. These follow
   published serving practice (chat 500 ms–2 s TTFT; 50–100 ms TPOT).
2. Per workload ray, the boundary point is the highest measured load meeting
   the SLO on both fitting repeats, minus one within-cell repeat spread
   (conservative = inner approximation; no measured SLO-violating point may
   be classified feasible).
3. Fit one facet through the per-ray boundary points by least squares,
   shrunk until every violating point is excluded.
4. Add a second or third facet only if the tuning mix (below) shows
   capacity-prediction error above 15%; facet count is selected on the
   tuning mix, never on the final validation set.

## Optimization

Create one candidate `c = (session j, method a, pool p)` over all three
methods, with `x_c in [0, 1]` and at most one candidate per session:

\[
\sum_{c:j(c)=j}x_c\le1.
\]

Source-power requirement, in watts:

\[
\sum_c w_{j(c)}x_c\ge\Delta P.
\]

Each candidate contributes one sparse column to the existing resource matrix
`Ux <= 1`. Pool `p` with `n_p` replicas adds:

\[
A_q^m\left(b_p+\sum_{c:p(c)=p}d_{j(c)}x_c\right)\le n_p\mathbf1,
\qquad
K_p^0+\sum_{c:p(c)=p}k_{j(c)}(H)x_c\le n_pK_q,
\]

with every symbol defined: `b_p = sum of (expected_f, expected_g)` over the
pool's `movable=False` sessions — the session list is the single source of
truth, and the `rho` axis is realized by generating background sessions
until `max(A_q^normal b_p)/n_p` reaches the target, reporting achieved
`rho`. `k_j(H) = _resident_tokens(j, H)` in tokens with `H` the existing
usable window `D - controller delay - power window`; `K_q` is the
engine-reported per-replica KV-token capacity (`kv_capacity_tokens`),
cross-checked against measured usage at saturation; `K_p^0` is the sum of
`_resident_tokens` over the pool's non-movable sessions.

These envelope rows **replace** the existing destination `max_ell` service
row and the packer's `max_load`; they do not coexist with them. Migration
ingest work (replay prefill, KV loading) appears only in the dedicated
pool-specific replay-time and KV-ingestion-time rows, never in the service
row — double-counting is avoided because the time-row coefficients are
measured under foreground load. Safe migration concurrency enters as a
time-averaged row per pool, `sum_{c:p(c)=p} (duration_c/H) x_c <= c_max`,
with the integer limit enforced exactly at ordering time in the simulator.
Source-stream (source-instance move time), link, deadline, and power rows
are unchanged from `formulation.md`.

Mode decision rule: solve normal and emergency as separate ordinary LPs.
The plan is `normal` iff the normal LP's exact post-rounding shortfall
(`lp_power_shortfall_w`) is 0; else `degraded`, shipping the best
emergency-feasible plan; `unsafe` iff the emergency shortfall is also
positive, still shipping the shortfall-minimizing emergency plan. No LP
mode-selection variable. Rounding fills remaining `ΔP` in LP-fraction order
as today; report the LP-versus-rounded objective gap and the distribution of
post-rounding `ΔP` shortfall, since missed power during curtailment is the
dangerous failure.

Greedy retains static one-shot pricing (as in the current implementation)
extended over the enlarged column space. Since `w_j` is identical across
pools for a session, pool choice is decided entirely by the priced resource
pressure term; ties break by lowest pool index for determinism.

### Replica scaling

Multiplying the one-replica envelope by `n_p` is an explicit modeling
assumption, not a measured fact. It is made tenable by (a) per-replica
whole-session packing (below), so no plan is accepted whose aggregate
feasibility hides an infeasible replica; (b) a measured two-replica check
(two memory-capped vLLM processes on GPU 1) reporting pool-aggregate error;
and (c) a worst-case-replica sensitivity variant that replaces `n_p 1` with
per-replica rows for the most loaded replica. Results report which variant
they use.

## Packing

After selection, group sessions by pool. `n_p` is fixed — packing never
opens replicas and cannot silently use another pool. Per-replica feasibility
is all envelope facets plus KV: `A_q^m d_replica <= 1` and resident tokens
`<= K_q`. Placement is the existing worst-fit heap keyed on
`max_i (A_q^m d_replica)_i` (generalizing the current
`max(load/max_load, tokens/max_tokens)` score). Hard-fail on unpackable
plans. Emit both the pool target and the concrete replica assignment. The
simulator validates final per-replica service and KV capacity against the
stable envelope but does not simulate continuous batching or predict
destination latency; claims are scoped to admission feasibility, not
SLO attainment (see Reporting).

## Two-A100 measurements

Use GPU 0 as migration source and GPU 1 as a loaded destination.

Fixed serving configuration for every probe: one serve command with pinned
image digest (reuse the bounded-campaign LMCache image pinning), identical
flags across all probes — chunked prefill enabled everywhere, ending the
current runner's prefill-staircase exception — and vLLM prefix caching
disabled, since trace workloads share prefixes and a cache-inflated envelope
is non-conservative for fresh-session admission. Record clocks and GPU
temperature as controls. Restart the server between repeats, discard the
first 60 s of every level as warmup, and interleave repeat order across
workloads so the held-out repeat does not share thermal/cache state
back-to-back with the fitting repeats.

### Service envelope

Workloads: prefill-heavy, decode-heavy, interactive-coding, coding, and
agentic, each from a named trace-backed manifest with pinned session IDs
(the coding manifest `outputs/coding-manifest.json` is the template; the
interactive and agentic manifests are prerequisites, per
`DATA_TO_COLLECT.md`, `??` sources).

Load placement: a coarse pilot doubling sweep brackets saturation per
workload (first level with sustained positive queue slope); then six points
at {0.5, 0.7, 0.85, 0.95, 1.0, 1.1} of the bracketed saturation. Arrivals
are open-loop Poisson; one burstiness probe (gamma interarrivals, CV = 2) at
the two highest loads for one workload bounds envelope sensitivity to
burstiness. Hold each level ≥ 8 min at the top three loads and ≥ 3 min
below (replacing the runner's 45 s default); "sustained positive queue
slope" means the least-squares slope of queue depth over the final 2/3 of
the hold is positive with its 95% CI excluding zero.

Three repeats per cell. Record prompt/output throughput, completion rate,
queue depth and slope, queue time, TTFT, TPOT/ITL, goodput, running/waiting
requests, KV usage, preemption, rejections, OOMs, restarts, and GPU
telemetry. Report per-cell coefficient of variation; if CoV > 10% in
boundary cells, add repeats there before fitting.

Validation is distributional, not single-point: leave-one-workload-out
cross-validation across the five workload rays, plus disjoint tuning and
final-validation mixed workloads (tuning selects facet count; validation is
touched once). Report P50/P95 relative error of predicted versus measured
boundary load, and false-feasible rate (target 0). SLO re-derivation from
the preserved frontier is claimed only within the measured rays and load
resolution.

### Migration under load

Destination load points {0, 0.5, 0.7, 0.9} of normal capacity plus one
emergency point at ~1.05 of normal (inside emergency, outside normal), so
the emergency LP's coefficients have producing runs and the load grid covers
the `rho` evaluation axis under the bucket rule below. Contexts pinned at
the manifest turns near 4K, 16K, and 32K; bandwidths 1 and 10 Gbps; three
repeats. Migration concurrency is 1 except a dedicated concurrency surface:
{1, 2, 4} concurrent ingests at 0.7 load, 16K context, 10 Gbps. Safe
concurrency `c_max` per load bucket is the largest level with no foreground
SLO breach and no ingestion-rate collapse (aggregate ingest rate still
within 20% of linear).

Controls: paired no-migration windows share fixed arrival seeds with their
treatment runs; report control-versus-control variance as the noise floor
before claiming any migration-induced delta. Record queue depth and
running/waiting requests at migration start in every row; gate migration
start on a preregistered queue-depth band and report t0 state as a
covariate.

Record migration readiness, work/bytes, ingestion rate, foreground goodput
and latency change, queue slope, KV pressure, continuation, power, and
failures.

Reduction: duration coefficients are multiplicative slowdown factors on the
existing `_duration` model per (method, load bucket), keyed on the pool's
baseline `rho` rounded **up** to the nearest measured bucket; `rho` above
0.9 uses the emergency coefficients. Bandwidth is handled analytically as
today — `max(bytes/effective_bw, bytes/ingest_rate)` with the ingest rate
load-bucketed — so contention-produced intermediate bandwidths need no
lookup. Hard-fail applies to the context, load, and concurrency dimensions
only.

Held-out validation: the pinned 24,292-token coding turn at loads {0.35,
0.8, 0.9} with mixed methods and a different foreground mix. Pre-commit to
comparing the load-bucketed model against a flat-coefficient null on these
points, so the campaign's central question — does destination load actually
change migration cost — gets a directional answer rather than one interior
point. Report the P50/P95 relative error of predicted migration time.

Emergency squeeze (evaluation axis 3) is defined per workload ray and for
the evaluation mix as the radial emergency/normal boundary ratio;
uncertainty comes from bootstrapping over requests within runs, not from
the three repeats.

## Baselines

Self-comparison among LP, rounding, and greedy is not sufficient. Evaluate:

1. **Shed-by-preemption**: no migration; pause/kill lowest-`w_j` sessions
   until `ΔP` is met. The operator's default alternative.
2. **Oracle LP**: realized (not expected) rates and durations; bounds the
   value of better prediction.
3. **Envelope-unaware planner**: the current scalar `f/F + g/G` planner run
   against the measured-envelope simulator; demonstrates whether the
   polytope machinery changes any decision.
4. **Least-loaded pool**: greedy session order, pools chosen by lowest
   normalized load.

## Evaluation axes

1. Compatible warm capacity ratio `R in {0.5, 1, 2, 4}`: total destination
   normal-envelope capacity along the evacuating workload mix
   (`sum_p n_p ×` radial per-replica capacity) divided by the displaced
   source demand along the same mix. `R = 2, rho = 0.85` means twice the
   evacuating token-rate demand exists as warm replicas, each already 85%
   full.
2. Baseline load `rho in {0, 0.5, 0.7, 0.85, 0.95}`, normalized against the
   normal envelope, realized via background sessions as defined above.
   Buckets map to measured migration loads by the round-up rule.
3. Measured emergency squeeze, with bootstrap uncertainty.

Each `R × rho` cell runs ≥ 10 seeded session populations; heatmaps show
fraction feasible with dispersion, never a single draw. Logical pool count
`P in {1, 2, 4, 8}` is a topology ablation at fixed total capacity with all
pools behind the same WAN cut — it isolates logical fragmentation, not
network path diversity. Secondary overlays are KV occupancy, equal versus
concentrated capacity, workload-mix mismatch, correlated site load, and
faster/central/slower ingestion profiles. Do not take their full Cartesian
product.

## Reporting

Report:

- measured normal/emergency/stable prefill-decode frontiers with absolute
  TTFT, TPOT, and goodput, and per-cell CoV;
- P50/P95 held-out relative error for boundary load, feasibility, and
  migration time (cross-validated, not single-point);
- migration completion and foreground SLO impact versus destination load
  and concurrency;
- `R × rho` feasibility heatmaps (fraction over seeds) for normal and
  emergency envelopes, for both the `n_p`-scaled and worst-case-replica
  constraint variants;
- source power shed, requested evacuation achieved, and the post-rounding
  `ΔP` shortfall distribution with the LP-versus-rounded gap;
- service, KV, replay, ingestion, concurrency, WAN, and source-stream
  bottleneck shares;
- planner-versus-baseline quality (all four baselines), runtime, memory.

Preemption counts, restarts, rejections, and GPU telemetry are
diagnostics-and-controls only; no claim consumes them. Claims are limited to
measured two-A100 behavior, admission feasibility (not destination SLO
attainment), and explicit sensitivity regimes. Public data motivate
localized, heterogeneous, bursty capacity but do not establish hyperscaler
utilization, spare-GPU distributions, router behavior, or multi-replica
linear scaling — which is why the scaling assumption carries its own
measured check and sensitivity variant.

## Implementation order

1. Rename the power coefficient to `w_j` in `formulation.md` and planner
   docstrings.
2. Add explicit pool identity and destination service-envelope profile data.
3. Generate non-movable destination background sessions (defining `b_p`,
   `K_p^0`, and achieved `rho`).
4. Preserve exact single-empty-pool behavior.
5. Generalize options to `(session, method, pool)` candidates over all three
   methods, shared by LP and greedy solvers; replace the destination
   `max_ell` row with envelope rows.
6. Add the concurrency row and pool-specific replay/ingestion-time rows.
7. Restrict whole-session packing to the selected pool with the per-replica
   facet score.
8. Add final per-replica stable-envelope validation.
9. Add measurement reduction (facet fit, slowdown factors, `c_max`) and
   cross-validated held-out validation.
10. Add the four baselines.

Required regressions include single-pool equivalence, no cross-pool
borrowing, alternate-pool selection, one candidate per session, background
compute/KV accounting, pool-preserving packing, normal/emergency row
isolation, degraded/unsafe labeling by the shortfall rule, hard failure on
unpackable plans and on load/context/concurrency outside the measured grid,
round-up bucket mapping, `rho`-axis/measured-grid alignment, and
exact-simulator rejection of temporally unschedulable aggregate plans
(link/stream scheduling and ingestion-rate timelines; per-replica final
capacity is validated, destination latency is not simulated).
