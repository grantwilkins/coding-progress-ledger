# Destination evidence ledger

Status: the existing corpus supplies most v1 coefficients. The old 72-hour
destination grid is superseded. Launch one mandatory 12-hour two-A100 job and
hold one targeted 12-hour reserve; expected use is 12 hours and the hard maximum
is 24 hours.

The rule is simple: use existing measurements or deterministic derivations
unless they cannot answer the admission question. Public data can establish a
mechanism or provide a labeled sensitivity value, but it cannot establish this
GPT-OSS-20B/A100 serving envelope, its loaded migration interference, or the
correctness of this exact LMCache/vLLM contract.

## What each task consumes

| Planner or validation task | Required datum | Existing evidence | New measurement | Use |
|---|---|---|---|---|
| context-conditioned service work | isolated `F(T)` and `G(T)` | stage-1 prefill/decode curves, 256–31,562 tokens | 4K/16K/24K drift anchors only | convert each session to `(f/F, g/G)` |
| normal/emergency admission | mixed prefill/decode boundary under open-loop arrivals | none; isolated staircases cannot establish joint interference or queue stability | mandatory mixed-ray frontier | fit common normals and policy bounds |
| stable execution ceiling | nonpositive backlog drift, complete drain, no OOM/restart/rejection | none under controlled mixed destination load | same frontier runs | reject final placements outside hard safety |
| workload-direction domain | interactive-coding, coding, and agentic boundary behavior | content-free trace shapes exist; no serving boundary comparison | same frontier runs | bound supported prefill fractions and test one-facet portability |
| live KV residency | KV capacity and per-session resident state | exact 1,214,544-token vLLM capacity plus block/token accounting | preflight readback only | HBM stock row through residency horizon |
| replay migration | context reconstruction and log-transfer time | coding and serial migration corpora | no unloaded reprofiling; loaded probes below | base replay work and deadline |
| KV migration | sealed bytes, copy/ingest time, append catch-up | coding, serial, parallel-gate, append, and bounded campaigns | no unloaded reprofiling; loaded probes below | base KV work, bytes, and deadline |
| loaded migration | replay/KV slowdown and foreground impact versus baseline `rho` | only `rho=0`; prior destination requests are continuation checks, not sustained background | mandatory paired probes at `rho=0.8` and near the measured boundary | conservative slowdown from initial load to mode boundary |
| method eligibility | replay and KV compatibility | current pinned model/image logs and continuation checks | exact preflight fingerprint | candidate mask; KV additionally requires exact ABI |
| migration correctness | cache hits, exact blocks/bytes, no WAN GET, valid continuation | parallel gate, serial, append, bounded campaigns | preflight plus every treatment run | reject invalid evidence and plans |
| source-power target | marginal GPU power curve | stage-1 power curve and serial power windows | none | candidate source-power gain and shortfall |
| source sleep/shutdown | transition time and whole-node power | level-1 sleep released memory but stayed near 84.9 W/GPU; no whole-node shutdown evidence | none for destination v1 | sensitivity only; no whole-node claim |
| exact route rows | route and usable bandwidth | scenario topology and shaped 1/10-Gbps tests | none | bytes on every traversed link |
| WAN geography and fabric | usable route capacity | public/administrator inputs only | none on these two GPUs | labeled scenario sensitivity, never a measured fleet claim |
| DRAM/SSD tiers | tier capacity and promotion rate | public systems establish the mechanism, not local headroom | none for v1 | sensitivity/staging only; cannot satisfy active HBM row |
| model weights/cold load | warmness, footprint, load time | fixed warm deployment and public hardware/model metadata | none for v1 | eligibility/baseline; cold placement is out of scope |
| architecture sweeps | `rho`, `H`, pools, routes, replicas | generated architecture instances | none | offline LP/greedy evaluation |
| packing correctness | per-replica assignment feasibility | hand-worked tests and exact small oracle | none | validate aggregate relaxation and repair |

## Existing evidence retained

All retained rows must preserve scenario ID, run/repeat, exact configuration,
units, completion state, and provenance. Partial or dirty scenarios are never
fit.

- `outputs/coding-run/scenarios.csv`: 648 complete replay/KV scenarios across
  no activity and one-turn activity, 250/1,000/10,000 Mbps, concurrency
  1/2/4, and three repeats.
- `outputs/serial-power-run-2`: 24 migrations and six matched controls at
  1/10 Gbps, concurrency one, and three repeats. It supplies exact prompt/KV
  work, timing, continuation, and GPU—not whole-node—power.
- `outputs/parallel-kv-gate-run-2`: 12 KV scenarios at 1 Gbps, concurrency
  1/2, and three repeats. It proves two connections overlap, exact 48/5/5/6
  new-block transfers, full destination watermark, no inference-time WAN GET,
  and valid continuation. V1 still admits concurrency one only.
- `outputs/append-catch-up-run-2`: 24 one-turn KV scenarios at 1/10 Gbps with
  32/128/512/2,048-token appends and two repeats. It supplies incremental
  growth, bytes, catch-up pause, and continuation.
- `outputs/bounded-hardware-campaign-run`: 105 complete deadline-meeting
  scenarios—63 parallel-surface and 42 staged-append—and 462 successful
  service requests across about 4K/16K/32K context, 1/10 Gbps, concurrency
  1/2/4, and three repeats. The 63 parallel-surface request schedules are
  empty, so this corpus does **not** establish migration under controlled
  destination background load.
- `profiles/gpt_oss_20b_a100_tp1.json`: current profile remains `estimated`.
  Its service reference is measured over 256–31,562 tokens with 25% relative
  error; replay and KV references are measured; KV capacity is an exact vLLM
  log readback; transitions remain assumed.
- The content-free workload manifest contains 72 pinned GPT-OSS-tokenized
  shapes: 24 each from Trace Commons coding, WildChat interactive coding, and
  NVIDIA SWE-Hero agentic data, split 12/6/6 for fit/tune/validation and
  spanning 74–32,757 context tokens. It supplies request shapes, not arrival
  timing. Arrival processes are generated and recorded explicitly.

The original 72-hour plan and its dense `rho × context × bandwidth × method`
grid must not be submitted. Existing unloaded work curves make most of that
grid redundant.

## Mandatory 12-hour A100-pair job

GPU 0 is the source and GPU 1 is one loaded destination replica. Pin the image
digest, model revision, tokenizer, engine flags, durable-log contract, KV ABI,
clocks, and rate shaper. Keep migration concurrency at one. Randomize scenario
order, checkpoint each row, and use matched arrival seeds for treatment and
no-migration controls.

### 1. Integrity preflight

This is a gate, not a new scientific result. Record the exact fingerprints and
prove retained same-session KV hits, zero unintended cross-session hits, exact
block counts and sizes, no inference-time WAN GET after warm prefetch, and
valid continuation for replay and KV transfer. Replay the frozen 4K, 16K, and
24K `F(T)`/`G(T)` rates with uniform arrivals. More than 15% underdelivery
is recorded and recalibrates the mixed-load normalization; incomplete anchor
data stops the job.

The historical corpus cannot prove that the newly launched container and
runtime still implement the same contract, which is the only reason this
preflight is repeated.

### 2. Mixed service envelope

The hypothesis is that context-normalized prefill and decode work compose into
one portable facet across interactive-coding, coding, and agentic mixtures. No
existing run used controlled open-loop mixed load long enough to classify
normal, emergency, and stable boundaries, so this cannot be inferred from the
isolated `F/G` corpus.

Use the three trace-shape families with unique prefixes and retained
per-session KV. Generate open-loop arrivals and adaptively bracket each realized
work direction. Refine boundary radius to at most 5%. Use at least three
independent runs around each boundary and increase only cells whose run-level
classifications disagree.

Freeze the primary policies before fitting:

- normal: p90 TTFT at most 2 s and p90 per-request mean TPOT at most 100 ms;
- emergency: p90 TTFT at most 10 s and p90 per-request mean TPOT at most 250 ms;
- stable: one-sided evidence of nonpositive backlog drift, no OOM, restart, or
  rejection, and complete drain after arrivals stop.

Revise policy thresholds only if the pilot cannot bracket a boundary,
emergency lies outside stability, or normal and emergency are empirically
indistinguishable. Freeze any revision before profile fitting.

Each service row must contain scheduled and actual arrival time, request and
session IDs, shape family and split, context, input/output tokens, offered and
completed input/output token rates, TTFT, per-request mean TPOT, completion,
queue time, queue-depth time series and fitted drift, running/waiting requests,
KV/HBM use, preemptions, rejections, OOMs, restarts, and configuration digest.

Fit one common-normal facet first. Add at most two facets only if tuning data
show more than 15% radial error. The final validation split is touched once.

### 3. Migration under destination load

The hypothesis is that the existing unloaded replay and KV timing
factorization remains conservative after applying one uncertainty multiplier
for foreground load. Existing migration corpora are the `rho=0` baseline; they
cannot reveal scheduler, HBM, or ingestion interference from a busy
destination.

For replay and KV transfer, measure:

- 16K context and 10 Gbps at `rho=0.8` and just inside the measured admission
  boundary; and
- held-out 24K context and 5 Gbps at the boundary/high-load condition.

Use three independent repeats with paired no-migration controls and identical
arrival seeds. Record target and achieved `rho`, service-work direction,
queue/running/waiting state at migration start, reconstruction or sealed
bytes, measured work, readiness and completion times, achieved path and ingest
rates, foreground TTFT/TPOT/goodput deltas, KV pressure, cache hits,
continuation, and failures.

Reuse the existing context and bandwidth work curves. If held-out median
migration-time error is at most 15%, no valid case is predicted feasible when
it is not, and residuals have no systematic interaction, retain the
factorization and fit only an upper-confidence slowdown. Do not create a
load-indexed grid without evidence that it is needed.

## Reduction and acceptance

Runs—not requests—are the independent experimental units. Use run-level
inference with block resampling inside a run. Never bootstrap individual
requests as independent experiments.

Produce central and conservative profiles. Admission consumes capacity lower
bounds and migration-duration upper bounds. The measured profile is accepted
only when:

1. no final-validation direction is falsely classified feasible;
2. median radial boundary error is at most 15%;
3. held-out migration-time median error is at most 15%;
4. every migration preserves exact state and continuation; and
5. all required contexts, workload directions, loads, and bandwidths have an
   explicit valid domain.

Expected campaign outputs are a configuration/preflight record, service-cell
rows, loaded-migration rows, reduction/validation summaries, central and
conservative versioned destination profiles, and checksums for every input and
output. These map directly to `DestinationType.prefill`, `decode`, `normals`,
`bounds`, `kv_capacity_tokens`, `loaded`, compatibility fingerprints, valid
ranges, and provenance.

## Conditional 12-hour reserve

Use the reserve only when the mandatory job exposes one of these preregistered
failures:

- boundary classifications disagree after three runs;
- the one-facet model exceeds 15% held-out radial error or produces a
  false-feasible result;
- loaded-migration residuals exceed 15% or show a systematic context,
  bandwidth, method, or load interaction; or
- state correctness, continuation, or deadline validation fails.

Measure only the failing direction or interaction. Do not repeat passing cells
or expand to a Cartesian grid.

## Explicitly not collected on the two A100s

- geographic WAN latency, datacenter egress, NIC rails, or operator QoS;
- DRAM/SSD fleet headroom or storage hierarchy capacity;
- published hardware specifications, model size, or KV bytes per token when
  derivable from the pinned model and exact ABI;
- cold model loading or hardware/model reallocation;
- multi-pool topology, `rho × H × P` sweeps, replica allocation, packing, or
  million-session scaling, all of which are offline experiments;
- non-A100 destination claims, tensor parallelism above the measured layout,
  migration concurrency above one, or continuous destination scheduling;
- a full replay/KV context-bandwidth grid already covered by the existing
  corpus; or
- whole-node shutdown power. The present evidence supports GPU power only.

Public measurements or operator telemetry may parameterize clearly labeled
sensitivity profiles for these items. They must never be presented as results
from this testbed.
