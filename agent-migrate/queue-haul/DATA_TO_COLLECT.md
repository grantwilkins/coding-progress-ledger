# Destination evidence ledger

Status: the archived request and engine records are recovered and verified.
Do not submit the current reserve bundle: its runner ignores the task list and
would repeat the full campaign. Migration needs no new measurement for the
measured low-concurrency domain. Service profiling remains conditional on
an accepted boundary. Of 66 complete-work runs, 60 are append-hot because
repeated prompts persist in vLLM APC. Five executions are forensically
private-prefix-consistent but lack strict stream-completion evidence, so they
supply descriptive sensitivity anchors only; one under-hit is excluded after a
likely silent prewarm failure.

The rule is simple: use existing measurements or deterministic derivations
unless they cannot answer the admission question. Public data can establish a
mechanism or provide a labeled sensitivity value, but it cannot establish this
GPT-OSS-20B/A100 serving envelope, its loaded migration interference, or the
correctness of this exact LMCache/vLLM contract.

## What each task consumes

| Planner or validation task | Required datum | Existing evidence | New measurement | Use |
|---|---|---|---|---|
| pinned warm replica class | model/revision, tokenizer, weight/KV dtype, engine/version/flags, accelerator layout, TP/PP, scheduler, block layout | GPT-OSS-20B, A100 80 GB, BF16, TP=1 and campaign configuration records | exact preflight fingerprint only | select the profile; changing the tuple creates another class |
| context-conditioned service work | private-prefix prefill and decode work over the demand horizon | stage-1 cold curves plus five legacy cache-geometry-consistent cells | none for descriptive sensitivity; targeted service follow-up for admissible evidence | use the affinity-level legacy anchor only as sensitivity; do not infer work from append-hot requests |
| normal/emergency admission | mixed prefill/decode boundary under open-loop arrivals | five legacy summaries pass at a common radius 0.096953, but stream completion is unobservable | remeasure an inner point and probe the unresolved boundary with the corrected cache/work gate | fit capacity only from strict complete private-prefix executions |
| stable execution ceiling | nonpositive backlog drift, complete drain, no OOM/restart/rejection | the same five legacy summaries pass but are not strict evidence | same targeted follow-up | reject final placements outside hard safety |
| workload-direction domain | interactive-coding, coding, and agentic behavior | one legacy coding anchor and two for each other affinity; no admissible failure or mixed blob | none for sensitivity; targeted follow-up for evidence-robust admission | keep workload affinity as an evidence domain |
| baseline service state | per-replica profile-compatible prefill/decode work, context, queue, and forecast window | raw request/engine telemetry exists; v7 `achieved_rho` is not a valid baseline | live telemetry at planning time | locate each replica inside its measured affinity blob |
| live-state lease | freshness generation plus reservations for service, KV, endpoint, and route headroom | not implemented | transactional destination readback and lease at planning/commit time | fail closed if state changes or the lease expires |
| allocatable KV supply | physical block capacity after weights, activations, graph captures, and runtime workspace | 963,152-token readback for vLLM 0.22.0 at 0.75 GPU-memory utilization; the older 1,214,544 value is another class | exact live preflight readback | per-replica HBM stock through the residency horizon |
| incremental KV demand | block-rounded private history and projected growth | within-session reuse and exact migration blocks are proven; cross-session sharing is deliberately uncredited | none for v1 | sum private blocks per replica; shared-prefix unions are a later optimization |
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
  signatures produce 50 empty responses and invalidate 47 runs. Sixty of 66
  complete-work runs are append-hot; five private-prefix-consistent executions
  have passing legacy summaries but lack strict completion evidence. One
  under-hit is excluded after a likely failed prewarm. Twelve migrations overlap a
  foreground request. The rows support low-concurrency replay calibration,
  additive KV timing, and a live-traffic method-affinity rule; see
  `FINDINGS.md`.

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
The five legacy private-prefix-consistent executions are descriptive sensitivity
anchors because strict stream completion is unobservable. The under-hit and
complete-work append-hot executions are excluded from service fitting.

Use the three trace-shape families with unique prefixes and retained
per-session KV. Clear APC between cells or make every appended prompt unique.
Hard-fail a request whose cached blocks extend beyond the block-rounded warmed
history prefix. Generate open-loop arrivals and adaptively bracket each
realized work direction. Refine boundary radius to at most 5%. Use at least
three independent runs around each boundary and increase only cells whose
run-level classifications disagree. Stop after five runs, retain the vote
counts, and use the majority without aborting the remaining campaign.

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
KV/HBM use, cached tokens, intended warmed-prefix tokens, cache classification,
preemptions, rejections, OOMs, restarts, and configuration digest.

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
private-prefix service semantics, and block-rounded private KV are documented
target semantics but are not represented by the current schema.

## Conditional targeted follow-up

The generated reserve is not executable as a targeted job:
`destination_runner.py` does not consume `reserve_tasks`. Do not submit it.

Retain the five private-prefix-consistent executions as descriptive sensitivity
anchors and exclude the under-hit and append-hot cells. The corrected runner
now caps forced tokens below 200000, pins the 16-token block size, resets local
APC, validates prewarm work and stream completion, and hard-fails cache-state
violations before reduction. To obtain admissible service evidence, first
remeasure near 0.096953, then expand until failure, and
collect three independent probes only around each affinity boundary. Compute
normal, emergency, and stability from every physical run. No migration
follow-up is required for component timing or method ranking within the
recorded concurrency-one v7 schedules and measured 16K/10-Gbps and
24K/5-Gbps cells. A robust foreground tail-SLO claim still requires enforced
idle/drained migration or additional impact evidence.

## Prefill/decode holding follow-up

Do not add a decode-hold planner field before this test. The minimum necessity
campaign is 24 physical runs, with at most nine additional runs if the knee is
not bracketed.

### Instrumentation gate

Pin one production serving class; prefer the H100 migration stack at
`gpu_memory_utilization=0.75`. Fingerprint model/tokenizer/image, GPU, TP/PP,
vLLM and CUDA, scheduler policy, `max_num_seqs`, `max_num_batched_tokens`,
chunked prefill, APC, block size, KV capacity, eager/graph mode, and cache
policy. Restart the process and clear cache between physical repeats.

Record scheduled and actual arrivals, offered prompt/output tokens, admission,
first generated token, every output-token timestamp, finish reason, `[DONE]`,
running/waiting/preemption/KV time series, and intended versus cached prefix.
Preserve Prometheus histogram labels and buckets; the current label-stripping
reduction destroys ITL quantiles. Hard-fail configuration drift, a cache
contract violation, incomplete output, or offered-arrival error above 1%.
Failures remain SLO misses rather than disappearing from the denominator.

### Paired necessity design

Use one middle context family at one clearly inside load and one adaptively
located SLO knee. Each 60-second matched block has identical total normalized
prefill work `P`, decode work `D`, context histogram, and physical KV stock
`K`. Run three shapes:

1. smooth prefills and staggered many-short decodes;
2. the same request/output multiset with prefills clustered at two declared
   short windows; and
3. smooth prefills with the same decode-token total concentrated into
   few-long, synchronized decodes.

Park neutral sessions when necessary to equalize KV and session inventory.
Use three independent process restarts for `3 shapes x 2 loads`, giving 18
runs. Hold out a second context family at the selected knee and run the
baseline plus whichever shaped treatment differs, again with three restarts,
giving six final-validation runs. If the initial loads do not bracket pass and
fail, add one outside load for all three shapes and repeats, giving nine more.

Split by whole process/cache-reset execution. Keep paired nodes and phases in
the same fold. Hold out complete contexts, workload shapes, and a temporal
campaign block. Never use realized overlap, completed throughput, queue depth,
admitted starts, or first/end times as admission features. Candidate features
must be scheduled offered `P,D`, a multiscale prefill-burst envelope, and a
context-conditioned forecast of active decode derived from enforced request
rate and output caps.

Predeclare these nested models:

- baseline: horizon `P,D` plus block-rounded `K`;
- burst: baseline plus maximum short-window prefill service debt;
- hold: burst plus one context-conditioned active-sequence or
  sequence-second row; and
- interaction: only if the factorial has independent support.

Promote the burst row only if it improves held-out classification. Promote the
decode-hold row only if equal-`P,D,K` shapes move the safe boundary by more than
15% at both tuning loads, the direction repeats on the held-out context, and
one monotone occupancy term reduces held-out radial error below 10% with no
false-feasible validation cell. Otherwise retain `P,D,K` only. Every promoted
model hard-fails outside its measured context, load, burst, output-length,
scheduler, KV, and power hull.

Keep the existing run-level policies: an inside point passes at least two of
three independent runs, an outside point fails at least two of three, and a
disagreement grows to five. Require a radial bracket at most 5%, complete
drain, the frozen queue-drift tolerance, and no OOM, restart, or rejection.
Report false-infeasible cells and interval coverage as secondary costs.

Only after the base service test passes should migration be introduced. If a
percentile interference bound is required, run replay, KV, and no-migration on
the identical inside and knee traces with three repeats: 18 additional runs.
The existing data already support qualitative KV-over-replay method ordering,
so omit this stage otherwise.

If “hold” instead means physical residency over horizon `H`, run a separate
stock experiment at 25/50/75/90% physical KV fill, churn/wait for `H`, and
resume random sentinels near the decode knee with three restarts. Measure
eviction, recompute, cache hits, TTFT, true ITL, and preemption. A synthetic KV
reservation is not residency evidence.

Before promotion, extend the checksum catalog to raw request arrays, engine
telemetry, full histogram buckets, H100 saturation JSON, compressed episode
records, and every regional sink stream. Retaining only reduced queue/drain
summaries makes later SLO fitting impossible.

## Three-region Azure follow-up

The three-region campaign is implemented but is not yet evidence. Sweden
Central is the source/power-down node; East US 2 and West Europe are
destinations.
The formal design is the seven one-factor conditions and 126 policy scenarios
documented in `README.md`, not the 648-cell Cartesian matrix. Each matched
condition/repeat uses identical sessions for all six policies, and every policy
uses both destinations.

Freeze each Spot allocation only after the following all pass: exact private
IP/region/VM/GPU/runtime/commit identity, clean tracked worktree, mounted
`/datadrive`, Azure PTP-backed chrony uncertainty at most 2 ms, zero-loss
200-sample ping, and three 60-second isolated and simultaneous iperf repeats.
Compute 40% and 80% route and aggregate caps from median simultaneous receiver
goodput. After reallocation, append a new allocation check and continue only if
every route RTT and route/aggregate goodput remains within 10% of the original
contract.

Retain, without aggregation or deletion:

- calibration host reports, every RTT sample, and raw client/server iperf JSON;
- 250-ms bytes by route, direction, connection, and billed status;
- per-connection start/end, directional bytes, TCP RTT/variance, congestion
  window, and total retransmissions;
- per-KV GET request/response/payload bytes and start/end;
- streaming request timestamps/chunks, TTFT, token use, cache hits, method,
  destination, commit, deadline, and state-code validation;
- source and both destination GPU power/utilization/memory with monotonic and
  wall timestamps, plus source sleep/wake timestamps;
- foreground-load requests, all service logs, every changed or active Azure
  Scheduled Events record, every failed/retried attempt, run metadata, and
  checksums.

Reject a physical attempt on host/runtime drift, clock or calibration gate
failure, Spot notice, service exit, invalid continuation, missing KV bytes,
zero cached tokens for a KV move, or telemetry failure. A deadline miss is an
experimental outcome, not a mechanism failure, and remains in the result.
`summary.json` is formally valid only when the latest attempt for every one of
the 126 scenarios is complete. Power conclusions remain GPU-scoped; no
whole-node or facility-power claim is permitted.

## Explicitly not collected on the prior two A100s

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
