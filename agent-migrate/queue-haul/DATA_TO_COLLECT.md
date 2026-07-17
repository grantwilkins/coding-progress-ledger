# Data to collect

This is the single list of missing live measurements. Keep the checked profile
at concurrency one until the matching concurrency tests pass.

## Next GPU job: parallel KV gate

- [ ] Give each simultaneous KV lookup an independent LMCache connection. First
  run one fixed two-session, 8k-token, 1 Gbps, no-activity smoke at concurrency
  1 and 2, with matched controls.
- [ ] Require two distinct connection IDs and overlapping KV-byte windows at
  concurrency two. Also require exact aggregate wire bytes, correct cache hits
  and continuation, no errors, and complete timing, power, and reduction data.
- [ ] After that gate passes, run only KV transfer for the same fixed sessions
  and turns at concurrency 1, 2, and 4, shaped aggregate rates of 1 and 10 Gbps,
  no activity, and two repeats. Include matched controls: 24 scenarios total.
- [ ] As a separate paired group, run raw KV and replay at approximately 8k and
  31.6k measured context tokens, 1 and 10 Gbps, concurrency one, no activity,
  and three repeats. Reuse each session and turn across methods and rates:
  24 migrations plus 12 shared controls.
- [ ] Record per transfer: source, destination, connection, exact payload and
  wire bytes, dispatch, queue arrival, copy start, first byte, last byte,
  destination ready, route switch, completion, and error. Record aggregate
  proxy rate, queue depth and bytes, cache hits, and source/destination power.
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

- [ ] Characterize topology separately from the two-GPU job. Test same-node,
  same-DC cross-node, and cross-DC paths where available; otherwise label
  shaped-proxy results as sensitivity data.
- [ ] At concurrency 1, 2, 4, and 8, run one source to one destination, one to
  many, many to one, and disjoint many-to-many transfers. Include staggered
  completions to test whether freed capacity is redistributed.
- [ ] Record source/destination node, rack, site, and direction; payload and
  wire bytes; per-flow and aggregate goodput; host NIC and shared-boundary
  counters; destination ingest/queue rate; background traffic; and failures.
- [ ] Infer each sharing domain and its residual application goodput from the
  aggregate plateaus. Validate max-min sharing with per-flow rates and the rate
  jump after a competing transfer completes.
- [ ] Calibrate separate directional capacities for source node egress, any
  binding rack/fabric cut, source-DC egress, WAN pair, destination-DC ingress,
  destination node ingress, and destination KV ingest. Omit tiers shown to be
  nonbinding rather than assigning every tier the same guessed rate.
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
- [ ] Measure empty-awake, sleep, and shutdown power and transition time for
  both GPUs and the whole server; the current 67.12 W/GPU value is idle power,
  not measured sleep power.
- [ ] Measure and validate tensor-parallel layouts before simulating tensor
  parallelism greater than one.
- [ ] Use cited public benchmark data only for clearly labeled sensitivity
  profiles when the testbed cannot measure a model or hardware configuration.

## Final validation

- [ ] Hold out complete drains and compare predicted versus measured queue
  wait, transfer time, route-switch time, request delay, and power over time.
- [ ] Hold out concurrency shapes and path classes; compare predicted versus
  measured per-flow rates, aggregate goodput, completion order, and completion
  time under shared bottlenecks.
- [ ] Test background, expedited, and critical policies on the same held-out
  drains at 10-minute, 1-hour, and 6-hour horizons.
- [ ] Record failures and incomplete runs; never fit a profile from partial
  scenarios.
