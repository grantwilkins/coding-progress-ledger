# Data to collect

This is the single list of missing live measurements. Keep the checked profile
at concurrency one until the matching concurrency tests pass.

## Next GPU job: serial crossover and power states

- [ ] Reuse `outputs/coding-manifest.json` and the existing two-model stack.
  Pin `codex:e381cc89-38ef-e67e-79b9-4b800369b4f5` at trace turns 0 and 60,
  expected to measure about 11k and 31k prompt tokens.
- [ ] Run raw KV and replay at 1 and 10 Gbps, concurrency one, no activity, and
  three repeats. Share one control across the four method/rate migrations for
  each turn and repeat: 24 migrations plus 6 controls.
- [ ] Before migrations, collect two paired 60-second empty-awake and vLLM
  level-1 sleep windows on the already-running stack. Record both GPUs, source
  memory, verified sleep/wake transitions, a post-wake request, and exclusive
  whole-node Slurm energy.
- [ ] Hard-fail before model startup if Slurm reports missing or zero node
  energy. Do not label GPU-only measurements as whole-server power.
- [ ] Record exact measured prompt and KV work, transfer and ingest timing,
  cache hits, continuation, shaped and achieved rate, and source/destination
  power for every migration.

## Later GPU job: parallel KV gate

- [ ] Give each simultaneous KV lookup an independent LMCache connection. First
  run a fixed two-session, 1 Gbps, no-activity smoke at concurrency 1 and 2,
  with matched controls.
- [ ] Require two distinct connection IDs and overlapping KV-byte windows at
  concurrency two. Also require exact aggregate wire bytes, correct cache hits
  and continuation, no errors, and complete timing, power, and reduction data.
- [ ] After that gate passes, run KV transfer for the same fixed four sessions
  and turns at concurrency 1, 2, and 4, shaped aggregate rates of 1 and 10 Gbps,
  no activity, and two repeats.
- [ ] Record connection, payload and wire bytes, dispatch, queue arrival, copy
  start, first byte, last byte, destination ready, route switch, completion,
  queue depth, aggregate rate, cache hits, power, and errors.
- [ ] Keep the checked profile at KV concurrency one until the smoke and
  targeted run show correct overlap, accounting, and completion.

## Urgency and append-only catch-up

- [ ] Measure initial KV backlog, new KV bytes per generated token, source
  extraction rate, path goodput, destination ingest rate, and fixed quiesce,
  synchronization, and route-switch time.
- [ ] Add controlled turns of 32, 128, 512, and 2,048 measured prompt tokens.
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
- [ ] Keep 1 and 10 Gbps as explicitly shaped usable-WAN allocations, not
  publication-derived facts. Apply each as one shared site/WAN cut while
  keeping the published endpoint and fabric tiers nonbinding.
- [ ] Obtain administrator telemetry or run one small aggregate check only
  before making claims about this cluster's NIC rails, QoS share, background
  load, routing, or physical site egress.
- [ ] Measure external durable-log paths separately; do not infer them from the
  final destination link of the KV path.

## Workloads and state transitions

- [ ] Run serial interactive coding, coding, and agentic tool-loop sessions;
  fit and validate each separately before sharing a curve.
- [ ] Full drain: move all eight sessions and test awake, sleep, and shutdown
  separately. Record the final route switch, transition start and end, and the
  complete GPU and whole-server power traces.
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
- [ ] Fit empty-awake and sleep power and transition time from the next job;
  the current 67.12 W/GPU value is idle power, not measured sleep power.
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
- [ ] Test background, expedited, and critical policies on the same held-out
  drains at 10-minute, 1-hour, and 6-hour horizons.
- [ ] Record failures and incomplete runs; never fit a profile from partial
  scenarios.
