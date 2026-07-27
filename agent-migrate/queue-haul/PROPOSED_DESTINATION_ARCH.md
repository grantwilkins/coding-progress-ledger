# Destination pool architecture

Status: the code implements measured migration primitives, a requirement
frontier, aggregate pool planning, internal replica packing, and a deterministic
event simulator. The archived destination campaign does not provide an accepted
service boundary. Until the targeted rerun brackets a passing and failing point,
destination serving headroom is a sensitivity.

## Public contract

Queue-Haul plans against compatible serving pools, not datacentres or individual
GPUs. A pool contains warm serving capacity with one pinned model/runtime
contract. Its manager advertises resources it is willing and able to allocate to
the event.

The public decision is:

```text
(session, replay-or-KV, destination pool)
```

The destination manager chooses a replica. Queue-Haul does not model its local
load balancer, batching policy, reservations, or unrelated arrivals.

A pool contract contains:

- compatibility fingerprint and warm/healthy state;
- measured normal and stable serving capacity;
- current serving work;
- ongoing event admission limit;
- temporary queued-work budget;
- replay reconstruction and KV-ingest budgets;
- live-KV blocks available after fixed runtime memory;
- logical route bandwidth and queued-byte budget;
- allowed migration methods;
- provenance, uncertainty, and validity range; and
- a lease or snapshot generation for operational use.

Compatibility is Boolean. Replay requires the same model, tokenizer, and
durable-log execution contract. KV transfer additionally requires the exact KV
ABI, layout, block format, and dtype.

## Service flex

The destination contract separates three quantities:

1. **Normal operating point:** the load at which the pool normally meets its
   latency policy.
2. **Event admission limit:** the ongoing work the operator accepts after
   handoff. It may be above normal but cannot exceed measured stable capacity.
3. **Transition debt:** temporary work queued during replay, KV ingest,
   catch-up, and switching.

Service work is measured in replica-equivalent seconds. A pool with eight
independent TP=1 replicas supplies eight replica-seconds per second. Five
percent flex means five percentage points of stable pool capacity above the
normal point. It never means five percent more sessions.

The planner constrains ongoing service, maximum queued replica-seconds, and
live KV separately. It reports required recovery time. A pool with positive
debt and no post-migration spare service cannot recover and is infeasible.

Queue-Haul reports work and queue requirements. It does not translate flex into
predicted TTFT or TPOT. Testbed measurements use SLOs only to ground normal and
stable operating points.

## Resources and units

| Quantity | Unit | Meaning |
|---|---|---|
| ongoing prefill/decode work | replica-s/s | continuing landed demand |
| transition work | replica-s | replay or ingest work during handoff |
| service debt | replica-s | work not served during the migration window |
| recovery | s | debt divided by post-migration spare service |
| live KV | physical blocks | private session state accepted by the pool |
| route demand | bytes | bytes sent over one logical route |
| route supply | bytes/s | event bandwidth advertised for that route |
| route debt | bytes | bytes waiting on the shared route |

Memory is a stock, not a queue. V1 rounds each session independently and gives
no credit for shared prefixes. The pool advertises usable KV after accounting
for internal fragmentation.

An integrated pool exposes its measured prefill/decode sharing rule. A
prefill/decode-disaggregated pool exposes separate resource budgets and queues.
The integrated A100 TP=1 case is measured. Disaggregated pools remain an
explicit sensitivity until separately profiled.

## Current code mapping

- `DestinationType` holds the pinned serving type, measured work curves,
  service bounds, KV capacity, migration components, and evidence status.
- `DestinationPool` identifies a compatible pool and logical route.
- `DestinationArchitecture` groups types and pools.
- `pool_planner.py` builds pool candidates and checks aggregate feasibility.
- Each selected plan retains physical use, capacity, unit, and utilization for
  every advertised resource row, plus per-pool/per-facet debt and recovery.
- Internal replica packing remains a contract-validation tool, not a public
  Queue-Haul decision.
- `requirement_frontier.py` reports raw landing requirements without assuming a
  destination inventory.
- `simulate.py` independently schedules migrations, routes, requests, power,
  and queues.

The planner implements explicit event admission, an aggregate service-work debt
bound, and required recovery. That bound is not a time-scheduled queue. The
event simulator separately drives a fluid service queue from realized replay
start/finish and commit times and rejects debt or recovery violations. Existing
`normal/emergency/stable` bounds map to normal policy, operator event policy,
and hard stability respectively; they must not remain identical placeholder
numbers in accepted evidence.

## Evidence map

| Quantity | Current status |
|---|---|
| source power | measured; held-out group-removal gate remains |
| replay and KV correctness | measured on two A100s |
| replay/KV duration | conservatively fitted with held-out context/bandwidth |
| KV bytes and block size | exact for the pinned ABI |
| live-KV capacity | measured vLLM readback |
| loaded migration effect | descriptive low-concurrency observations |
| normal/stable service boundary | not accepted; targeted rerun required |
| route bandwidth/RTT | assumed sensitivity |
| pool flex and debt | operator sensitivity until testbed validation |
| multi-pool inventory | assumed sensitivity |
| H100 and A100 TP=2 service | required later measurement |
| disaggregated prefill/decode site | assumed sensitivity |

The invalid 2026-07-23 service cells remain excluded. The five
private-prefix-consistent points near 0.096953 are descriptive anchors, not
safe capacity.

## Targeted service measurement

Use only three resource mixes derived from coding, interactive coding, agentic,
and ShareGPT shapes:

- prefill-heavy;
- balanced; and
- decode-heavy.

For each mix:

1. use open-loop arrivals;
2. begin near the existing descriptive anchor;
3. bracket a passing and failing normal/stable point;
4. repeat only boundary points at least three times;
5. reset prefix cache or use unique appends;
6. require complete streams and exact private-prefix state; and
7. inject 0/5/10/20% bursts to measure queued work and recovery.

Any missing work, wrong cache state, restart, rejection, or unbracketed boundary
prevents an accepted service profile.

## Admission flow

1. Validate the pool fingerprint and evidence domain.
2. Read current pool work, KV stock, route state, and contract generation.
3. Build replay and KV candidates only for compatible pools.
4. Check ongoing event capacity, stable capacity, debt, KV, transition, route,
   source streams, and deadline.
5. Select at most one action and pool per session.
6. Validate the time schedule in the event simulator.
7. Retain source ownership until the destination produces the first
   post-switch token.
8. Revalidate the destination contract at commit in an operational deployment.

Current code does not implement a live lease. Results therefore remain
sensitivity or testbed validation rather than production admission
certificates.

## Scale experiments

The large simulator consumes advertised residual vectors; it does not generate
unrelated destination traffic. Primary capacity is relative to the source
relocation requirement. Pool-relative 0/5/10/20% flex is a sensitivity.

Use:

- 10K, 100K, and 1M source sessions;
- coding, interactive coding, agentic, and ShareGPT conversation shapes;
- 30/60/120/300-second deadlines;
- 1/5/10-Gbps routes;
- 1/2/4/8 pools;
- fixed-total and fixed-per-pool capacity;
- separate resource and compatibility diversity experiments; and
- integrated pools first, then disaggregated-pool sensitivity.

The 8+8 A100 experiment validates the contract at a larger hardware point. It
does not define the simulated site size.
