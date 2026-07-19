# Data to collect

This is the single list of missing live measurements. Keep the checked profile
at concurrency one until the matching concurrency tests pass.

## North star and next campaign

The target is a source datacenter shedding power across 1,000,000 or more
sessions before a deadline. The simulator must choose whole sessions, replay or
KV transfer, transfer timing and rate, and final quiescence while respecting
per-source capacity and one shared source-site/WAN cut. The destination is an
aggregate admission, replay, KV-ingest, and residency pool; its server scheduler
is out of scope.

The next campaign is one resumable two-GPU job with a 12-hour limit. Start the
stack once, randomize scenarios, checkpoint each result, reuse compatible
controls, and exclude incomplete or dirty scenarios from fitting.

- [ ] **Parallel KV surface:** approximately 4k/16k/32k context, 1/10 Gbps,
  logical concurrency 1/2/4, and three repeats. Rerun every cell with the
  corrected shaper and exact key attribution; use the old gate only as a
  regression smoke. Fit repeats 0-1 and hold out repeat 2.
- [ ] **Multi-stage append:** one approximately 30k context session, steady and
  bursty growth, 1/10 Gbps, three repeats, and four ordered background writes
  before final quiescence.
- [ ] **Workload validation:** interactive coding, coding, and agentic
  tool-loop traces at low/high context and serving concurrency 1/4, with
  paired no-migration and migration windows. Use traces, not GPU runs, to fit
  request gaps and tool delays.
- [ ] **Held-out drain:** eight mixed sessions in an unseen approximately
  10-minute/10-Gbps case and one approximately one-hour/1-Gbps drip. Do not use
  either case for fitting; evaluate a six-hour horizon offline.

This campaign does not require a new transport library, TCP simulation,
content-addressed KV, compression, target relays, failure replication, NIXL,
or a 16-GPU drain. Add lower-level source/transport/destination probes only if
the aggregate model retains systematic held-out error.

### Required evidence and gates

For every migration record session, phase or append-stage index, logical and
wire bytes, start, destination-ready, pause, final catch-up, route-switch and
commit times, achieved rate, cache hits, continuation, source/destination GPU
power, and errors. For parallel runs also record exact per-session connection
attribution and aggregate goodput.

Each multi-stage row must additionally record newly created tokens, cumulative
sealed and copied blocks, stage wire bytes, duplicate bytes, and the final
residual. Hard-fail on missing stages, ambiguous session attribution, byte
nonconservation, lost appended tokens, invalid continuation, or an incomplete
scenario.

Stop after this campaign if held-out transfer and drain errors are within
15-20% without systematic residuals. Add pipeline-stage instrumentation only
if that test fails; add connection pooling only if measured setup or connection
churn materially limits representative transfers.

## Completed: serial crossover and GPU power states

- [x] Reuse `outputs/coding-manifest.json` and the existing two-model stack.
  Pin `codex:e381cc89-38ef-e67e-79b9-4b800369b4f5` at trace turns 0 and 60,
  which measured 11,047 and 30,474 prompt tokens.
- [x] Run raw KV and replay at 1 and 10 Gbps, concurrency one, no activity, and
  three repeats. Share one control across the four method/rate migrations for
  each turn and repeat: 24 migrations plus 6 controls.
- [x] Collect two paired 60-second empty-awake and level-1 sleep GPU windows,
  source memory, verified transitions, and post-wake requests. Sleep released
  memory but source power remained about 84.9 W.
- [ ] Obtain exclusive whole-node energy separately. `serial-power-run-2`
  records `node_power: false`; do not label its GPU measurements as
  whole-server power.
- [x] Record exact measured prompt and KV work, transfer and ingest timing,
  cache hits, continuation, shaped and achieved rate, and source/destination
  power for every migration.

## Completed GPU job: parallel KV gate

- [x] Script `stage1d_parallel_gate.sbatch` with the reviewed 12-scenario
  `outputs/parallel-kv-gate-plan.json`.
- [x] Run the fixed two-session, 1 Gbps, no-activity smoke at concurrency 1
  and 2 with matched controls.
- [x] Require two distinct connection IDs and overlapping KV-byte windows at
  concurrency two. Also require exact aggregate wire bytes, correct cache hits
  and continuation, and no errors. `check-parallel` hard-fails otherwise.
- [ ] Complete the missing cells in the bounded parallel KV surface above.
- [ ] Record logical and wire bytes, initial start, destination ready, route
  switch, completion, per-session and aggregate rate, cache hits, concurrent
  action power, and errors.
- [ ] Keep the checked profile at KV concurrency one until the smoke and
  targeted run show correct overlap, accounting, and completion.

## Urgency and append-only catch-up

- [x] Script `stage1e_catch_up.sbatch` with the reviewed 24-scenario
  `outputs/append-catch-up-plan.json`: 32/128/512/2,048-token controlled
  appends, 1/10 Gbps, two repeats, and matched controls.
- [x] Run the scripted two-stage initial/final catch-up job and determine from
  connection-attributed wire bytes whether the implementation is incremental.
- [x] Measure initial KV backlog, new KV bytes per generated token, path
  goodput, end-to-end copy service, and fixed quiesce, synchronization, and
  route-switch time. Do not label the aggregate copy rate as source extraction
  or destination ingestion without stage-specific evidence.
- [x] Add controlled turns of 32, 128, 512, and 2,048 measured prompt tokens.
  Separate appended prompt from decoded output tokens; record processed and new
  tokens, KV bytes, cache hits, response timing, service pause, and continuation
  correctness for replay and KV transfer.
- [ ] Measure multi-stage append-only KV copy while generation continues,
  including bytes created and copied in every stage and the final paused
  residual. Test whether copy service remains faster than KV growth and record
  duplicate destination-residency byte-seconds.
- [ ] Pair every method on the same session state and report the measured
  crossover among replay, full KV transfer, and incremental KV transfer.
- [ ] Evaluate 10-minute, 1-hour, and 6-hour horizons offline from the measured
  primitives. For each session report required rate, achieved rate, predicted
  completion quantile, remaining slack, final pause, and power reduction over
  time; do not run wall-clock six-hour GPU cases.
- [ ] Use slack and required-rate pressure, not a fixed wall-clock label, to
  identify background, expedited, and critical operating regions.

## Network topology and outflow

- [ ] Use published A100 sensitivity bounds instead of a multi-node GPU
  campaign: 24 GB/s effective per 200 Gbps RDMA rail, full and half-bisection
  intra-DC fabrics, 5/10/20% external cuts, and 1:4/1:8/1:16
  between-building-to-within-building ratios.
- [x] Keep 1 and 10 Gbps as explicitly shaped usable-WAN allocations, not
  publication-derived facts. Apply each as one shared site/WAN cut while
  keeping the published endpoint and fabric tiers nonbinding.
- [ ] Obtain administrator telemetry or run one small aggregate check only
  before making claims about this cluster's NIC rails, QoS share, background
  load, routing, or physical site egress.
- [x] Fix durable logs at the source DC and route replay through the same
  shared site egress and WAN cut as KV transfer.

## Workloads and state transitions

- [ ] Run the bounded workload-validation windows above; fit each job class
  separately before sharing a curve.
- [ ] Validate an eight-session awake drain on the two-GPU testbed. This checks
  transfer, catch-up, request delay, and power accounting, not physical
  eight-GPU node shutdown.
- [ ] Profile exclusive whole-node sleep or shutdown separately only before
  making claims about those final states.
- [ ] Pair the same session, turn, and repeat across methods and bandwidths when
  estimating a method or bandwidth effect.

## Workloads and capacity

- [ ] Collect held-out complete session traces with context, prompt and output
  tokens, request gaps, tool time, service time, KV growth, and first-request
  timing for each job type.
- [ ] Extend request, replay, and KV measurements beyond 31.6k context tokens
  only as required by those traces.
- [ ] Measure shared-prefix blocks when traces contain shared prefixes.
- [ ] Validate the compute admission limit with held-out multi-session latency
  runs and reserve KV capacity for growth over the simulated time window.

## Power, network, and hardware

- [ ] Measure request and migration power at every supported concurrency.
- [x] Fit empty-awake and sleep power and transition time from the paired job;
  the measured level-1 sleep point is about 84.9 W/GPU and did not reduce board
  power, but only two transitions were observed.
- [ ] Measure shutdown power and transition time separately; a running Slurm
  allocation cannot measure physical node power-off and reboot.
- [ ] Measure and validate tensor-parallel layouts before simulating tensor
  parallelism greater than one.
- [ ] Use cited public benchmark data only for clearly labeled sensitivity
  profiles when the testbed cannot measure a model or hardware configuration.

## Final validation

- [ ] Hold out complete drains and compare predicted versus measured queue
  wait, transfer time, route-switch time, request delay, and power over time.
- [ ] Validate the shared-cut simulator against the shaped proxy before making
  claims about unmeasured physical topology.
- [ ] Test background, expedited, and critical behavior from remaining slack
  and required rate on the same held-out drains at 10-minute, 1-hour, and
  offline 6-hour horizons.
- [ ] Compare node-aware greedy with exact enumeration on tiny cases and the LP
  on tractable cases before using greedy for the million-session sweep.
- [ ] Require the summary-mode planner and simulator to complete a
  million-session case within an explicit runtime and memory budget; do not
  emit per-block audit events at that scale.
- [ ] Record failures and incomplete runs; never fit a profile from partial
  scenarios.
