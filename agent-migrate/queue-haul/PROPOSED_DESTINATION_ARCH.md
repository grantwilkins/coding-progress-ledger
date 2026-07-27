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

The contract is a versioned snapshot with units, evidence status, provenance,
validity range, and replacement evidence for every value. Operational use also
requires a lease and commit-time revalidation; the current simulator has
neither and therefore does not produce production admission certificates.

## Contract-to-output mapping

The executable schedule is the primary output. The requirement frontier is a
summary of validated schedules across requested shed targets.

| Advertised contract field | Schedule use | Result-table columns |
|---|---|---|
| compatibility fingerprint and allowed actions | candidate eligibility and selected action/pool | `action`, `pool`, compatibility provenance |
| current and event prefill/decode capacity | ongoing landed work | ongoing prefill/decode use, capacity, normalized slack |
| stable capacity, debt budget | transition queue and recovery | service debt, recovery, capacity, normalized slack |
| replay reconstruction capacity | replay start/finish and endpoint occupancy | reconstruction work/use, capacity, normalized slack |
| KV-ingest capacity | KV transfer/ingest finish and occupancy | ingest work/use, capacity, normalized slack |
| usable live-KV blocks | post-commit KV placement | KV blocks used, capacity, normalized slack |
| route bandwidth and queued bytes | transfer start/finish and route occupancy | route bytes/time/debt, capacity, normalized slack |
| lease or snapshot generation | validation domain | contract generation, provenance, validity range |

Every selected-migration row records session ID, source, action, pool, start,
transfer/reconstruction finish, quiesce, commit, first-token completion, bytes,
transition work, ongoing work, KV blocks, and conservative source watts
credited. Every scenario row records requested, achieved, and unmet watts;
selected sessions; replay/KV counts and bytes; destination assignments; all
resource use and normalized slack; the complete binding-resource set;
predicted and realized makespan; debt and recovery; source shutdown; exposed
sessions, context tokens, and KV bytes; seed; workload; source
hardware/model/packing; deadline and measurement window; and evidence status,
validity range, and provenance for every input.

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

The simulator consumes the advertised residual vector at the contract
generation. It does not synthesize unrelated destination traffic, hidden
provider capacity, or future arrivals. Simulated scheduling and queue outcomes
are evidence about coordination under that contract, not direct evidence of
production destination behavior.

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

## Evaluation contracts

### Fixed contract

The many-session coordination experiment holds one compatible integrated pool
contract fixed while requested shed rises through
10/25/50/75/90/100% of maximum modeled shed. A single versioned canonical
record fixes workload; source hardware, model, and packing; deadline; route
bandwidth and RTT; event service flex; debt; usable KV; reconstruction and
ingest capacity; and seed. This experiment asks how the contract is spent and
whether joint replay/KV planning produces more executable shed. It is not a
multi-site heterogeneity experiment.

Plot normalized residual slack against requested shed for source preparation,
route, reconstruction, ingest, ongoing prefill, ongoing decode, debt, live KV,
and realized makespan. Zero is binding and negative values remain visible.
Compare achieved shed against requested shed for Queue-Haul, all replay, all
KV, the best simple greedy baseline, and an exact integer or LP reference where
tractable. Report unmet watts, the first infeasible target, and the complete
binding-resource set.

### Multiple pool contracts

Pool count is 1, 2, 4, or 8 under two explicitly defined regimes:

1. **Fixed total resources:** hold each summed physical budget constant and
   split it across the pools. Pool-specific routes and compatibility remain
   explicit. This isolates fragmentation, route diversity, and compatibility.
2. **Fixed resources per pool:** give each pool the same explicit budget, so
   every summed destination budget grows linearly with pool count. This
   measures added headroom until a source-side or other shared constraint
   binds.

Results must state the explicit route, reconstruction, ingest, ongoing service,
debt, and KV settings; opaque scenario names are not sufficient.

Resource diversity and compatibility diversity are separate experiments:

- **Resource diversity:** vary pool route, reconstruction, ingest, service,
  debt, or KV budgets while holding compatible action/pool choices fixed.
- **Compatibility diversity:** vary compatible action/pool choices while
  holding total physical resources fixed.

For each destination resource, plot maximum executable shed against its
capacity multiplier or advertised headroom with identical axes across panels.
Use separate panels for route bandwidth or queued bytes, replay
reconstruction, KV ingest, ongoing prefill, ongoing decode, event debt, and
live-KV blocks. Each curve must expose the knee where another resource joins
the binding set.

### Schedule morphing

Choose three or four explicit settings from the pool-count and bottleneck
experiments, such as route-constrained, replay-reconstruction-constrained,
service-constrained, and KV-memory/ingest-constrained. For each:

- x-axis: time;
- y-axis: destination pools;
- rectangle: one migration, with width equal to scheduled duration;
- fill or hatch: replay versus KV;
- markers: commit and first-token completion;
- overlays: shared route and transition-resource occupancy; and
- annotation: final achieved shed and complete binding-resource set.

The examples explain why the capacity curves bend or flatten; they are not
decorative Gantt charts.

### Scale and sensitivity

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

Coding, interactive coding, agentic, and ShareGPT-like conversation are
separate workload scenarios. Deadlines of 30/60/120/300 seconds, routes of
1/5/10 Gbps, 0/5/10/20% flex and debt, workload, source packing, and seed are
scenario axes, not error bars. Service flex, debt, multi-pool inventory,
disaggregated pools, unmeasured routes, and unmeasured hardware points remain
visibly `assumed/sensitivity` until their targeted measurements pass.

The 8+8 A100 experiment validates the contract at a larger hardware point. It
does not define the simulated site size.
