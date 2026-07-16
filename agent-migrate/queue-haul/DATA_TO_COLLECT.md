# Data to collect

This is the single list of missing live measurements. Keep the checked profile
at concurrency one until the matching concurrency tests pass.

## Next GPU job

- [ ] KV concurrency: use one independent LMCache connection per transfer and
  test concurrency 1, 2, and 4 across the same and different source instances,
  destination instances, and destination nodes.
- [ ] Per KV transfer, record source, destination, connection, exact KV bytes,
  dispatch, queue arrival, copy start, first byte, last byte, destination ready,
  route switch, completion, and error.
- [ ] Record queue depth, queued bytes, per-transfer and aggregate rates, proxy
  bytes, cache hits, and source and destination power during KV concurrency.
- [ ] Catch-up: add turns of 32, 128, 512, and 2,048 measured prompt tokens.
  Record processed and new tokens, KV bytes, cache hits, response timing,
  service pause, and continuation correctness for replay and KV transfer.
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
- [ ] Measure source egress, destination ingress, external-log, rack, and shared
  site paths before making site-wide network claims.
- [ ] Measure and validate tensor-parallel layouts before simulating tensor
  parallelism greater than one.
- [ ] Use cited public benchmark data only for clearly labeled sensitivity
  profiles when the testbed cannot measure a model or hardware configuration.

## Final validation

- [ ] Hold out complete drains and compare predicted versus measured queue
  wait, transfer time, route-switch time, request delay, and power over time.
- [ ] Record failures and incomplete runs; never fit a profile from partial
  scenarios.
