# Destination evidence ledger

Status: the archived request and engine records are recovered and verified.
Do not submit the current reserve bundle: its runner ignores the task list and
would repeat the full campaign. Migration needs no new measurement for the
measured low-concurrency domain. Service profiling remains conditional on
replacing six invalid forced-token signatures; the 66 valid runs are all
feasible and therefore supply inner observations, not a capacity boundary.

The rule is simple: use existing measurements or deterministic derivations
unless they cannot answer the admission question. Public data can establish a
mechanism or provide a labeled sensitivity value, but it cannot establish this
GPT-OSS-20B/A100 serving envelope, its loaded migration interference, or the
correctness of this exact LMCache/vLLM contract.

## What each task consumes

| Planner or validation task | Required datum | Existing evidence | New measurement | Use |
|---|---|---|---|---|
| pinned warm replica class | model/revision, tokenizer, weight/KV dtype, engine/version/flags, accelerator layout, TP/PP, scheduler, block layout | GPT-OSS-20B, A100 80 GB, BF16, TP=1 and campaign configuration records | exact preflight fingerprint only | select the profile; changing the tuple creates another class |
| context-conditioned service work | cold `F(T)`, decode `G(T)`, and any cache-conditioned prefill work function over full/hit/miss lengths | stage-1 cold curves over 256–31,562 tokens; request rows record planned and cached tokens | none unless quantitative prefix-compute credit is required | use `(f/F, g/G)` today; a guaranteed hit alone does not calibrate reduced prefill work |
| normal/emergency admission | mixed prefill/decode boundary under open-loop arrivals | all 66 complete-work v7 runs pass both policies; the frontier is right-censored | only after selecting safe forced tokens, probe the required unresolved boundary | fit capacity only from valid realized work |
| stable execution ceiling | nonpositive backlog drift, complete drain, no OOM/restart/rejection | all 66 complete-work v7 runs pass stability | same targeted service follow-up if an upper boundary is required | reject final placements outside hard safety |
| workload-direction domain | interactive-coding, coding, and agentic boundary behavior | conditional inner successes exist for all three affinities; no valid failure selects another facet | none until valid held-out data reject the one-facet model | keep workload affinity as eligibility/domain |
| baseline service state | per-replica profile-compatible prefill/decode work, context, queue, and forecast window | raw request/engine telemetry exists; v7 `achieved_rho` is not a valid baseline | live telemetry at planning time | locate each replica inside its measured affinity blob |
| allocatable KV supply | physical block capacity after weights, activations, graph captures, and runtime workspace | exact 1,214,544-token vLLM capacity for the pinned homogeneous ABI | preflight readback only | per-replica HBM stock through the residency horizon |
| incremental KV demand | protected baseline block IDs, required session blocks, private growth, and exact pinned-engine shared-prefix keys | within-session reuse and exact migration blocks are proven; cross-session sharing and idle-prefix residency are not | state readback only if sharing credit is claimed | enforce the physical block-set union; otherwise sum block-rounded private demands |
| instance inventory and packing | healthy warm replica count, exact pinned class, pool/type membership, and per-replica baselines | architecture/scenario input plus packing tests | live health, configuration, and baseline readback at planning time | test existence of a concrete assignment; never infer it from GPU count alone |
| replay migration | context reconstruction and log-transfer time | coding and serial migration corpora | none within the measured domain; reuse v7 loaded evidence | base replay work and deadline |
| KV migration | sealed bytes, copy/ingest time, append catch-up | coding, serial, parallel-gate, append, and bounded campaigns | none within the measured domain; reuse v7 loaded evidence | base KV work, bytes, and deadline |
| loaded migration | runtime calibration, overlap topology, and foreground impact | recorded concurrency-one v7 request schedules in the measured 16K/10-Gbps and 24K/5-Gbps cells; exact interval work is not identifiable | none for component timing/ranking inside those recorded schedules | component timing; foreground observations rank methods but do not prove a tail-SLO bound |
| method eligibility | replay and KV compatibility | current pinned model/image logs and continuation checks | exact preflight fingerprint | candidate mask; KV additionally requires exact ABI |
| migration correctness | cache hits, exact blocks/bytes, no WAN GET, valid continuation | parallel gate, serial, append, bounded campaigns | preflight plus every treatment run | reject invalid evidence and plans |
| source-power target | marginal GPU power curve | stage-1 power curve and serial power windows | none | candidate source-power gain and shortfall |
| source sleep/shutdown | transition time and whole-node power | level-1 sleep released memory but stayed near 84.9 W/GPU; no whole-node shutdown evidence | none for destination v1 | sensitivity only; no whole-node claim |
| exact route rows | source-method-replica path, residual bandwidth, latency, and reservation state | shaped 1/10-Gbps local tests validate component behavior only | live path and allocatable-capacity inventory at planning time | bytes and time reservations on every traversed link |
| WAN geography and fabric | usable route capacity and fixed/per-round latency | public/administrator inputs only | none on these two GPUs | labeled scenario sensitivity, never a measured fleet claim |
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
- `outputs/destination-v7-20260722` plus its verified ignored raw archive: 18
  correct migration treatments and 113 service executions. Six forced-token
  signatures produce 50 empty responses and invalidate 47 runs; all 66
  complete-work runs pass every policy. Twelve migrations overlap a foreground
  request. The rows support
  low-concurrency replay calibration, additive KV timing, and a live-traffic
  method-affinity rule; see `FINDINGS.md`.

The original 72-hour plan and its dense `rho × context × bandwidth × method`
grid must not be submitted. Existing unloaded work curves make most of that
grid redundant.

## Completed 12-hour A100-pair job

This section records the completed campaign contract. It is not a launch
instruction. Reuse its valid evidence and replace the invalid forced-token
signatures before planning another service measurement.

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

The hypothesis was that context-normalized prefill and decode work compose into
one portable facet across interactive-coding, coding, and agentic mixtures.
The valid executions establish only conditional inner points because none
fails an SLO or stability check.

Use the three trace-shape families with unique prefixes and retained
per-session KV. Generate open-loop arrivals and adaptively bracket each realized
work direction. Refine boundary radius to at most 5%. Use at least three
independent runs around each boundary and increase only cells whose run-level
classifications disagree. Stop after five runs, retain the vote counts, and use
the majority without aborting the remaining campaign.

Retry transient stack, health, preflight, and phase failures in the same job;
reuse complete cells and archive invalid checkpoints. Record target-load misses
and censor unbracketed boundaries instead of terminating the campaign. Only
changed immutable inputs or persistently invalid measurements stop collection.

Freeze the primary policies before fitting:

- normal: p90 TTFT at most 2 s and p90 per-request mean TPOT at most 100 ms;
- emergency: p90 TTFT at most 10 s and p90 per-request mean TPOT at most 250 ms;
- stable: one-sided evidence of nonpositive backlog drift, no OOM, restart, or
  rejection, and complete drain after arrivals stop.

The first destination pilot could not bracket exact-zero drift at any positive
load. Before fitting, the frozen measurable threshold was revised to a
block-bootstrap upper bound of one queued request per measurement window;
scheduled client requests waiting to enter vLLM count toward that backlog.

Revise policy thresholds only if the pilot cannot bracket a boundary,
emergency lies outside stability, or normal and emergency are empirically
indistinguishable. Freeze any revision before profile fitting.

Each service row must contain scheduled and actual arrival time, request and
session IDs, shape family and split, context, input/output tokens, offered and
completed input/output token rates, TTFT, per-request mean TPOT, completion,
queue time, queue-depth time series and fitted drift, running/waiting requests,
KV/HBM use, preemptions, rejections, OOMs, restarts, and configuration digest.

Fit one common-normal facet only after valid data bracket it. Add at most two
facets only if tuning data show more than 15% radial error. The current invalid
runs do not justify another facet.

### 3. Migration under destination load

The hypothesis is that the existing unloaded replay and KV timing
factorization remains conservative under foreground load. The raw records
identify observed busy/idle overlap and component timing, but not a continuous
load curve.

The completed job attempted:

- 16K context and 10 Gbps at `rho=0` on the same pinned runtime;
- 16K context and 10 Gbps at `rho=0.8` and just inside the measured admission
  boundary; and
- held-out 24K context and 5 Gbps at the boundary/high-load condition.

The recovered request intervals show 12/18 treatments overlap foreground work.
The achieved `rho` remains a preceding 30-second average, not
migration-interval intensity, and it counts cached prompt tokens as compute.
The exact request records support categorical idle/busy evidence and paired
TTFT/TPOT effects; engine counters cannot isolate foreground tokens from replay
or KV migration work.

Reuse the existing replay context curve with the conservative runtime
calibration. Use sealed bytes divided by route rate plus the conservative KV
residual. Both have held-out median error below 10% and no conservative
underprediction. Do not create a load-indexed grid without evidence that it is
needed.

## Reduction and acceptance

Runs—not requests—are the independent experimental units. Use run-level
inference with block resampling inside a run. Never bootstrap individual
requests as independent experiments.

Produce central and conservative profiles only from independent bracketed
service runs and component timing that preserves exact route lower bounds.
Admission consumes capacity lower bounds and migration-duration upper bounds.
The measured profile is accepted only when:

1. no final-validation direction is falsely classified feasible;
2. median radial boundary error is at most 15%;
3. held-out migration-time median error is at most 15%;
4. every migration preserves exact state and continuation; and
5. all required contexts, workload directions, loads, and bandwidths have an
   explicit valid domain.

Expected campaign outputs are a configuration/preflight record, service-cell
rows, loaded-migration rows, reduction/validation summaries, central and
conservative versioned destination profiles, and checksums for every input and
output. Existing fields cover `DestinationType.prefill`, `decode`, `normals`,
`bounds`, additive `kv_capacity_tokens`, loaded coefficients, compatibility,
and provenance. The full pinned replica tuple, component migration envelopes,
cache-miss prefill semantics, and physical block unions are documented target
semantics but are not represented by the current schema.

## Conditional targeted follow-up

The generated reserve is not executable as a targeted job:
`destination_runner.py` does not consume `reserve_tasks`. Do not submit it.

First select forced tokens below the observed invalid range and make missing
completion, usage, or forced tokens a hard request failure. Reuse the 66 valid
service runs. If an accepted service boundary is still required, run three
independent corrected probes per workload affinity at nominal radius 0.5 and
adapt only the affinity that fails. Compute normal, emergency, and stability
from every physical run instead of rerunning a cell by policy. No migration
follow-up is required for component timing or method ranking within the
recorded concurrency-one v7 schedules and measured 16K/10-Gbps and
24K/5-Gbps cells. A robust foreground tail-SLO claim still requires enforced
idle/drained migration or additional impact evidence.

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
