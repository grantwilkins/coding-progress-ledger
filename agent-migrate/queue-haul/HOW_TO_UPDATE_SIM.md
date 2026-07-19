# How to update the simulator

Use new measurements to improve the million-session power-drain model, not to
reproduce a transport implementation. Prefer a profile update over a structural
simulator change whenever the measured aggregate model predicts held-out data.

## Model boundary

The simulator selects sessions, migration method, destination aggregate, start
time, transfer rate, quiescence, and final source state. It models:

```text
session append-only stream
  -> source endpoint
  -> shared source-site egress
  -> WAN
  -> destination-site ingress and aggregate endpoint
```

Connections, packets, TCP, destination-server scheduling, and LMCache internal
calls are out of scope. A connection is never a unit of simulated fairness.

## Evidence-to-model mapping

| Evidence | Update when supported | Do not infer |
|---|---|---|
| Serial size/rate trials | KV fixed time and aggregate endpoint rate | Separate source and destination rates |
| Concurrency 1/2/4 | Source/destination stream limits; a concurrency curve only if needed | Linear speedup from connection count |
| Four-stage append | Copied-block progress, update cost, final residual and pause | Token-level packet events |
| Workload traces | Request gaps, tool delay, first request, growth distribution | Workload timing from synthetic GPU loops |
| Migration under load | Service and action-power interference | A new scheduler from one workload |
| Held-out drains | Profile uncertainty and plan acceptance | Refit with held-out observations |
| Published network data | Named sensitivity profiles and intra-DC bounds | This cluster's physical site egress |
| Exclusive node telemetry | Sleep/off trace and whole-node power | Whole-node power from GPU telemetry |

Fit repeats 0-1 and reserve repeat 2 plus the mixed drains for evaluation.
Preserve raw observations and provenance. Hard-fail outside measured ranges.
Do not fit any new profile until `check-campaign` passes every planned hardware
scenario; do not substitute the coding calibration manifest for missing
workload-class or mixed held-out traces.

## Required updates after the campaign

1. Refit KV setup, endpoint throughput, initial completion, catch-up fixed time,
   tail rate, action power, and uncertainty from the complete clean runs.
2. Raise concurrency limits only through the highest concurrency that passes
   overlap, byte conservation, readiness, continuation, and held-out timing.
3. Represent every background update as newly sealed blocks since the prior
   copied-block watermark. Final catch-up sends only uncopied complete blocks
   and handles the measured partial tail.
4. Fit workload growth with an upper quantile appropriate for admission and
   retain the observed distribution for evaluation.
5. Accept a plan only when trailing-window power, migration commit, requested
   final state, and arrived-request service all meet the deadline.

The existing `destination_bytes_per_s` is an aggregate non-WAN pipeline rate
unless stage-specific evidence proves otherwise. Rename it to
`endpoint_pipeline_bytes_per_s` when updating the profile schema. Split source
and destination rates only if held-out residuals or heterogeneous endpoints
require that distinction.

## Policy updates

Derive urgency from remaining slack:

```text
required_rate =
remaining_bytes
/
(deadline - now - final_catchup - switch - transition - power_window)
```

Retain three policy comparisons:

- Maximum-rate greedy.
- Deadline-paced greedy.
- Node-drain greedy with the whole-node power bonus.

Use exact enumeration on tiny cases and the LP on tractable cases as quality
references. The node-aware greedy is the intended million-session solver and
should remain approximately `O(N log N)`.

Earlier quiescence is necessary when observed growth or contention consumes
slack. A plan is infeasible when its required rates exceed the measured
endpoint or shared-cut capacity after reserving final catch-up and the power
window. Reserve setup and initial-completion time inside preparation, pace
background copies only, and leave paused final catch-up uncapped.

## Large-scale execution

Keep two execution modes:

- Detailed mode for profiling and held-out drains, with session, request,
  network, queue, append-stage, and power evidence.
- Summary mode for 1,000,000 or more sessions, without per-block audit events.

Summary mode must conserve aggregate bytes and capacity and reproduce detailed
mode on small deterministic cases. Add a performance regression with explicit
runtime and memory limits before reporting million-session results.

## Conditional changes

Do not add lower-level transport stages if aggregate held-out error is within
15-20% and residuals are not systematic.

Add source extraction and destination ingestion resources only if errors vary
systematically with source fan-in, destination fan-in, or endpoint type. Add
connection pooling only if setup or churn materially limits representative
transfers. Add an explicit update journal only if staged writes duplicate,
lose, or ambiguously attribute KV. Consider compression only when required
rates frequently exceed usable WAN capacity.

NIXL integration, custom transport protocols, target relays, replication,
failure injection, content-addressed blocks, and TCP simulation are deferred
until a measured simulator error or an explicit research question requires
them.
