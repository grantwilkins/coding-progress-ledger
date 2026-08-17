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
| Queue-Haul performance effect | incumbent p90 TTFT/mean TPOT, joint attainment, and stability versus measured normalized load under phase-separated added session work | migration transients and external phase-sensitivity anchors only; no clean steady-state placement curve | 54-cell discovery plus 18-cell held-out confirmation per hardware | hardware-specific service reserve and raw paper curve; do not fit a serving simulator |
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

## Queue-Haul performance evaluation

This is a measured calibration of the existing service constraint, not a new
serving simulator. Hold a deterministic incumbent stream fixed at
`rho_b=0.25`, preload the same fixed, predeclared pre-arrival parked stock in
every main cell, and add Queue-Haul work until total scheduled load reaches
`rho={0.25,0.50,0.70,0.85,0.95,1.10}`. This scalar is offered normalized phase
work, not measured GPU utilization. The x coordinate is recomputed from all
offered work during the frozen measurement window:

```text
rho = sum_r(append_r / R_P(4096) + output_r / R_D(4096)) / window_s.
```

Preserve the two components before summation:
`rho_f=sum_r append_r/R_P/window_s` and
`rho_d=sum_r output_r/R_D/window_s`. A scalar may be promoted only as a
conservative envelope across tested mixtures; composition-dependent latency or
stability must remain visible in `(rho_f,rho_d)` space.

No migration bytes move during this curve: it isolates the steady-state
serving effect after sessions are placed. Existing replay/KV experiments remain
the evidence for the distinct transfer transient.

Plot only incumbent request performance. The top panel is p90 TTFT; the bottom
is p90 per-request mean TPOT. Queue drain/stability and service failures mark a
point infeasible rather than adding a third line. Preserve p50/p90/p95/p99,
every returned token ID and SSE timestamp, and full Prometheus histograms. A
multi-token SSE event is an exact completion but does not expose literal token
gaps, so it invalidates TPOT/ITL measurement rather than counting as a service
failure. P99 is reportable only with at least 1,000 incumbent requests in a
cell; otherwise it is null.
Horizontal latency targets are evaluation inputs, not values inferred from the
curve.

Use one 4K controlled continuation pack so phase attribution is unambiguous.
Incumbent requests use 3,840 cached +256 appended prompt tokens and 128 output
tokens. Added prefill-heavy requests use 2,048+2,048/32; added decode-heavy
requests use 4,032+64/512. Recalibrated phase shares must be at least 0.7 and at
most 0.2 respectively. Eight sessions of every direction, including the parked
direction, are prewarmed in every main cell. A matched baseline with only the
incumbent sessions resident checks that parked KV alone is harmless.

Run A100 and H100 independently and never pool their boundaries. Before either
headroom sweep, measure `R_P(4096)` at concurrency 1/4/16 and `R_D(4096)` at
16/64/128 on the exact TP1, vLLM 0.22, eager, chunked-prefill, APC, block-16,
8,192-batched-token, 256-sequence, `gpu_memory_utilization=0.75` Queue-Haul
destination stack. Take the maximum stable throughput inside each fresh-process
block and the median of three block maxima. The inherited H100 context table is
not admissible normalization for this experiment. This is explicitly a
synchronized-burst throughput normalizer, not an arrival-rate SLO boundary; an
optimum at the largest tested concurrency is retained as right-censored.

The frozen discovery core shares one direction-free baseline per block:
`(1 baseline + 2 directions x 5 added-load points) x 3 restart blocks = 33`
valid cells per hardware, plus 18 normalizer cells and three no-resident
controls, or 54 cells per
hardware. Follow the checksum-frozen randomized order in the plan. Run all
calibration cells first; one headroom block produces an internal preview, and
all three produce the scout curve.
After thresholds are frozen, select the last passing and first failing points
without changing the core data. Confirm one shared baseline and each direction's
last-pass/first-fail in three unseen blocks (15 cells), then run a balanced
check at the selected boundary in three blocks. It uses 128 output tokens and a
block-rounded append length derived from that hardware's measured `R_P/R_D`,
and must place 40--60% of normalized work in prefill. Thus the promotion stage
is 18 cells per hardware and 72 including discovery. Matched-rho long/few versus
short/many decode and smooth versus microburst prefill are optional domain
checks; disagreement restricts the scalar contract rather than adding a planner
dimension automatically.

Every cell uses a fresh process, cache reset, private-prefix prewarm, 60-second
warmup, 240-second measurement window, absolute 180-second request deadline,
180-second drain limit, and a smooth uniform open-loop schedule. One
predeclared asynchronous client task is available per offered request under a
frozen 4,096-task ceiling. One unbounded aiohttp connector drives the exact
open-loop schedule in each of 32 fixed event-loop shards, so queued responses
neither consume one OS thread apiece nor starve token-stream parsing. The plan
pins the client runtime and shard count and disables cyclic GC only while the
trace is active so response-object collection cannot pause request release.
Reference counting remains active, and cyclic GC is restored after the trace.
Stability uses
running plus waiting plus client-pending requests only inside the measurement
window; it cannot be rescued by warmup or post-window drain. Fit those requests
over the final two thirds of the observed window and report both the slope and
its fitted accumulated-request change. At most one fitted accumulated request
is no material growth. Preserve the legacy 30-second block-bootstrap upper
bound as diagnostic-only metadata. The measurement
hard-fails a scrape gap over one second. A send slip over 50 ms,
parser/configuration/cache-proof error, exact token-timing coverage below 99%, GPU
preemption/Xid, or missing live-engine sampler invalidates the measurement.
Timeout, rejection, incomplete output, OOM, or load-induced engine exit is a
service miss and remains in the offered denominator. Partial request and metric
evidence is written before failure classification. Mean-TPOT quantiles use only
the exact-timing subset and carry its coverage; every ambiguous stream remains a
miss in joint TTFT/TPOT attainment. Invalid measurements may be
rerun only immediately in place, before any later cell in the frozen order;
otherwise stop the stage. Complete service failures may not be rerun.

Every calibration, discovery, and confirmation result carries the exact model
revision, image-byte hash, vLLM/LMCache versions, semantic scheduler command,
GPU SKU/UUID/memory/power/application clocks, commit, plan hash, and
normalization hash. The runtime identity includes and rechecks the canonical
vLLM, LMCache, and Redis launch commands, including environment-dependent GPU
and KV roles. The image is hashed once per unchanged stage artifact, and each
reducer reconstructs the observed wall-clock cell order and requires it to equal
the frozen randomized order. Any cross-cell service-identity mismatch hard-fails
reduction. An analysis-only collector change between discovery and confirmation
is allowed only when the model, hardware, runtime versions, scheduler, and
semantic launch commands are unchanged; record both full identities and commit
SHAs. Successful
measurement requests must report exactly the block-rounded private prefix as
cached; under-hit and append-hot cells are invalid because their actual prefill
work no longer equals the x coordinate. Fixed prefix conditioning is checked
from exact successful uncached prewarm prompt tokens: resident minus control
must equal the planned block-rounded prefix tokens exactly, and confirmation
must reproduce the resident prewarm count. The engine-reported live KV-token
capacity remains bound into the normalization. The immediate active-KV gauge
is diagnostic only because it does not measure reclaimable APC cache blocks.
Measurement-start KV includes load-dependent active decodes, so it is retained
as an outcome and is not used to match cells.

For supplied targets `(tau_F,tau_D)`, the scout reports the largest contiguous
load for which every restart drains, has bounded total in-system drift and
exact-length completion, and satisfies incumbent p90 TTFT/TPOT. It also reports
the target-independent physical-stability bracket, first failing point,
censoring, and joint request attainment over all offered incumbents. The scout
is never planner-usable. P1/P2 receive the minimum boundary across directions
only when all unseen baseline/last-pass cells pass, all unseen first-fail cells
fail, the balanced check passes, and the parked-KV control is within the frozen
15% measurement tolerance. Every control and resident baseline must also be
stable with 100% exact completion. Final reduction re-supplies the source core
plan and scout and reproduces the checksum-bound confirmation plan; P1/P2
accept the result only through `supported_bound()`. Report the raw curves and
target sensitivity beside that scalar. The result supports a statement about
the tested smooth 4K serving class and its fixed pre-arrival parked stock; it is
not a maximum hardware-occupancy claim, universal latency equation, or fleet
reliability guarantee. The confirmed value is a total-load cap:
`b_f + b_g + sum_i(w_i,f + w_i,g) <= rho_safe`. Available added headroom is
`rho_safe - (b_f + b_g)`, never `rho_safe` itself.

The 2026-08-15 A100 execution completed 54/54 discovery and 18/18 held-out cells
without a collection retry, but did not confirm the initially selected scalar
cap. At the selected
`rho=0.70`, prefill-heavy, balanced, and decode-heavy each passed only two of
three unseen blocks. Prefill-heavy `rho=0.85` failed the 100-ms P90 mean-TPOT
target in all three blocks, while decode-heavy `rho=0.85` was queue-stable in
one block and unstable in two despite repeatable latency. Consequently the
frozen reducer reports no planner-usable value. Retain these curves as
composition-sensitivity evidence.

The minimal fallback does not require a two-dimensional lattice. The frozen
transition follow-up tested total work `W=0.50` at prefill-heavy, balanced, and
decode-heavy recipes across three fresh restart blocks each. All 9/9 cells
met both cohorts' P90 TTFT/TPOT targets, exact completion/cache checks, drain,
and strict stability. This makes `W=0.50` a conservative tested floor for the
exact 4K A100 normalization and stack, not a production cap. The held-out
`W=0.70` cells met both latency SLOs but missed the every-repeat stability rule,
so the higher stable bound is bracketed in `[0.50, 0.70)` and remains
unidentified. The transition artifact does not automatically promote a
planner profile. If more headroom is operationally necessary, confirm one
preregistered intermediate point, preferably `W=0.60`, rather than assuming
interpolation. Collect a two-dimensional lattice only if a future claim
specifically requires interpolation across unseen compositions. No such
campaign is required for the scalar tested-work certificate.

## Prefill/decode holding follow-up (optional model promotion)

This deeper campaign is not required for the Queue-Haul performance result
above. Run it only if a future planner revision needs a portable decode-hold or
burst dimension rather than a measured policy reserve.

Do not add a decode-hold planner field before this test. Use a staged campaign:
the quick necessity screen is 24 confirmatory runs plus 2--4 bracket scouts,
or 26--28 planned valid runs. Reserve at most four measurement-invalid attempts,
for a hard screen cap of 32 attempts. A promotion-ready held-context result
requires 24 more confirmatory runs and
2--4 more scouts, for 52--56 planned valid runs; reserve at most eight invalid
attempts across both stages, for a hard base-promotion cap of 64 attempts. The
first stage may reject the new dimension; it may not promote or certify one.
At the 720 s run cap, the screen and base promotion attempt budgets consume at
most 6.4 and 12.8 single-GPU hours respectively.

Freeze one primary joint run rule before the scouts. Either retain the legacy
rule--exact completion of every request, marginal p90 TTFT <=2 s, and marginal
p90 per-request mean TPOT <=100 ms--or predeclare a product request-good rule
with an exact decode statistic, thresholds, offered-request denominator, and
attainment objective. Marginal p90 rules are not 90% joint request goodput.

### Instrumentation gate

Pin one production serving class; prefer the H100 migration stack at
`gpu_memory_utilization=0.75`. Fingerprint model/tokenizer/image, GPU, TP/PP,
vLLM and CUDA, scheduler policy, `max_num_seqs`, `max_num_batched_tokens`,
chunked prefill, APC, block size, KV capacity, eager/graph mode, and cache
policy. Restart the process and clear cache between physical repeats.

Record scheduled and actual arrivals, offered prompt/output caps, admission,
first generated token, every output-token timestamp, finish reason, `[DONE]`,
running/waiting/preemption/KV time series, and intended versus cached prefix.
Preserve Prometheus histogram labels and buckets; the current label-stripping
reduction destroys ITL quantiles. Configuration, telemetry, cache-contract, or
offered-arrival error above 1% makes a measurement invalid and eligible for one
rerun only while the stage's global invalid-attempt reserve remains. A service
timeout, rejection, OOM, crash, or incomplete output under a valid offered trace
is instead an SLO miss; do not rerun it away.

Use the existing open-loop driver with 60 s warmup, a 180--480 s measurement
window, a 720 s run cap, and a predeclared drain cap. Preflight every trace for
at least 200 offered requests and 10,000 planned token gaps. A shortfall in
finite emitted gaps caused by service misses is a valid infeasible outcome, not
an invalid measurement. Record any unmet runtime sample target rather than
extending only unfavorable cells. A valid run must achieve the scheduled load
within 1% without client-side completion backpressure.

### Non-gating single-A100 capacity discovery

Before treating a model/context width as a comparable architecture point, run
`single_gpu_capacity_campaign.py` on one visible A100. Use fresh processes for
the Cartesian product of the three pinned checkpoints and five contexts, BF16
KV, TP1, `max_model_len=32768`, `max_num_seqs=256`, and synchronized widths
`1,2,4,8,16,32,64,128,256`. Qwen prompt contexts must be exact multiples of
its measured 784-token unified attention block. Preserve the model-specific
LMCache chunking and separate object groups. For vLLM 0.22's Qwen K/V-major
attention tensor, require the zero-copy 784-token logical-page view, contiguous
per-K and per-V slices, and the live group-edit log; a tensor-layout mismatch is
a runtime-contract failure, never a launch-capacity outcome.

This stage has no performance gate. Report launchability, launch/OOM/service
failure, the largest completely served burst, maximum observed running and
waiting requests, first saturated and failed width, and whether the sweep is
right-censored. vLLM may eventually complete a burst much larger than its
physical simultaneous-running capacity, so those two quantities must never be
collapsed. Retry only instrumentation, host, or runtime-contract failures;
valid OOM, context rejection, timeout, incomplete output, or engine exit is the
limit being discovered.

### Paired necessity design

At one middle context, construct the full `2 x 2` scheduled-traffic factorial:

1. smooth prefill, low planned decode hold;
2. burst prefill, low planned decode hold;
3. smooth prefill, high planned decode hold; and
4. burst prefill, high planned decode hold.

Use independent prompt-heavy/short-output and short-prompt/decode-heavy
substreams, with at least two exact output tokens in every request. Change only
the prompt-heavy arrival permutation for the burst factor and only the
decode-heavy arrival permutation for the hold factor. Within each burst level,
`B` must match across low/high hold; within each hold level, `N` must match
across smooth/burst. Freeze the matching tolerances before the scouts.

All four cells must use the identical per-request prompt/output multiset,
including the output-length histogram and context-conditioned decode work, and
must match horizon, request count, normalized offered `P,D`, and initial
block-rounded physical `K`. Only the schedule permutation may change. Keep
memory away from its eviction knee; any parked equalization session issues no
requests. Enforce exact output lengths with audited forced tokens.

Freeze the two admission-time treatments before mixed-load collection. For
isolated prefill service time `f0_i`, define
`B_w = max_t(max(0, sum_{a_i in [t,t+w)} f0_i - w) / w)` and
`B = max(B_tau_D, B_(tau_F/2))`. Define planned decode hold `N` as the peak
overlap of intervals
`[a_i + f0_i, a_i + f0_i + d0_i]`, where `a_i` is scheduled arrival and
`f0_i,d0_i` are isolated per-context prefill/decode durations measured on this
serving class and frozen before the factorial. This forecast uses only
scheduled arrivals and output caps. Realized first/end times, active overlap,
achieved throughput, queue depth, and admitted starts are outcomes, never
admission features. Require `B_high >= 2 B_low > 0`, `N_high >= 2 N_low`, at
most 1% relative error from each `B` target, and exact integer `N` targets.

Use 2--4 baseline-only single-run scouts to locate adjacent radial loads around
the joint run boundary, with safe/unsafe radius ratio at most 1.05. Scouts tune
the design and are not confirmation evidence. If four scouts cannot bracket,
censor the context and stop rather than widening the campaign silently.

Run all four shapes at both radial loads in three randomized restart blocks:
`4 shapes x 2 loads x 3 blocks = 24` confirmatory runs. Counterbalance shape
order, restart/clear/prewarm for every cell, reuse the exact paired trace
multiset and seed within a block, record the block identifier, and keep the
whole block in one inference fold. This screen estimates both main effects and
their interaction at one context. It decides only whether temporal shape is
material; three repeats do not estimate a production violation probability.

If neither treatment has a material paired effect, stop and retain `P,D,K`. If
one does, repeat the entire 24-run factorial at a second context whose treatment
labels were not inspected during fitting. Use 2--4 baseline scouts to normalize
that context's joint boundary. This is the minimum promotion-ready design:
48 confirmatory plus 4--8 scout runs. If a shape is not bracketed, one extra
radial level costs a complete `4 shapes x 3 blocks = 12` run block; allow at
most one such block per context. This raises the planned-valid-run cap to 80;
the eight invalid-attempt reserve makes the absolute cap 88 attempts or 17.6
single-GPU hours. Otherwise leave the model unpromoted.

### Labels and promotion gates

For each run, report the frozen primary rule, request attainment
`sum(good_i)/N_offered`, request goodput
`sum(good_i)/measurement_seconds`, service-failure rate
`N_service_failures/N_offered`, completed-only latency distributions,
offered-versus-actual arrivals, and queue/KV/preemption traces. A bracket cell
must have three concordant independent run labels. Any mixed cell is unresolved
and blocks promotion; do not spend uncounted cell-only repeats or break the
paired restart blocks.

Require complete drain within the fixed cap and a 95% block-bootstrap upper
bound on waiting-plus-client-backlog slope no greater than one request per
measurement window. A valid service failure is always a request miss but makes
the run infeasible only when the frozen availability/attainment rule is
violated. It does not make the measurement invalid.

For every shape, let `[r_safe, r_fail]` be the adjacent concordantly labeled
radial bracket and use its geometric midpoint as the reported boundary coordinate.
Index boundaries as `r_bn` for burst `b` and planned hold `n` in `{0,1}`. Burst
contrasts are `r_10/r_00` and `r_11/r_01`; hold contrasts are `r_01/r_00` and
`r_11/r_10`. A contrast is greater-than-15% adverse only when
`r_fail_treatment < 0.85 * r_safe_control`, and a promoted main effect must have
that direction in both strata and repeat in the held context. Treat 15% as the
repository's existing model-error tolerance, not a tunable fitted coefficient.

Replace `B,N` below by `(B-B_low)/(B_high-B_low)` and
`(N-N_low)/(N_high-N_low)`. Because the `P,D` radial direction and `K` are fixed,
fit the predeclared boundary family
`log(r_hat_bn) = beta_0 - beta_B B - beta_N N - beta_BN B*N` by unweighted
least squares on tuning-context geometric boundary midpoints, with `beta_0`
unconstrained and the three burden coefficients constrained nonnegative.
`M0`, `M_B`, `M_N`, and `M_B+N` fix unused
coefficients to zero; fit `beta_BN` only after the two-main-effect model. Do not
impute or fit an unbracketed cell.

The interaction contrast is
`Delta_BN = log(r_11)-log(r_10)-log(r_01)+log(r_00)`. Propagate each radial
bracket through that formula; call the interaction material only when the
resulting interval lies wholly below `-log(1.15)` and its negative sign repeats
in the held context. A wholly positive interval is reported as subadditivity
and leaves the combined region unpromoted. Weight each held-context shape
equally. Radial error is
`abs(predicted_boundary / observed_boundary - 1)`. Promotion requires zero
predicted-safe/observed-unsafe held-context cells and median held-context radial
error at most 15%; report false-infeasible cells and bracket coverage. Zero
false-feasible cells here is a structural validation result, not a reliability
confidence guarantee.

Hard-fail admission outside the measured context, load, burst, planned-hold,
output-length, KV, model, hardware, and scheduler hull. Gate power separately.
Migration interference and physical KV residency are separate campaigns and
must not be added to this go/no-go test.

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
