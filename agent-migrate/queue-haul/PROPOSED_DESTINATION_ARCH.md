# Destination and execution contract

Status: the code implements measured migration primitives, a requirement
frontier, aggregate pool planning, internal replica packing, and a deterministic
event simulator. The pinned two-A100 data shows working replay and KV handoff,
including request-boundary switching, continuation, exact KV accounting, and
phase-level timelines. The primary end-to-end contract is ordered eager-parallel
execution: order controls launch and queue ties, while moves overlap.
Destination serving headroom remains an assumed sensitivity because the
archived service campaign does not provide an accepted boundary.

## Public contract

Queue-Haul plans against compatible serving pools, not datacentres or individual
GPUs. A pool contains warm serving capacity with one pinned model/runtime
contract. Its manager advertises resources it is willing and able to allocate to
the event.

The public decision is:

```text
(session, replay-or-KV, destination pool)
```

Queue-Haul deterministically packs a selected pool action onto a replica for
validation. It does not model the pool's live load balancer, batching policy,
reservations, or unrelated arrivals.

A V1 pool contract contains:

- compatibility fingerprint;
- measured normal and stable serving capacity;
- current serving work;
- ongoing event admission limit;
- temporary queued-work budget;
- live-KV blocks available after fixed runtime memory;
- logical route identity and path, whose link rates come from the scenario;
- allowed migration methods;
- migration timing evidence; and
- provenance and evidence status.

Compatibility is Boolean. Replay requires the same model, tokenizer, and
durable-log execution contract. KV transfer additionally requires the exact KV
ABI, layout, block format, and dtype.

Warm/healthy attestations, uncertainty fields, route queued-byte admission,
separate replay/KV endpoint budgets, live leases, snapshot generations, and
commit-time revalidation are production extensions, not V1 inputs.

## Contract-to-output mapping

The executable schedule is the primary output. The requirement frontier is a
summary of validated schedules across requested shed targets.

| Advertised contract field | Schedule use | Result-table columns |
|---|---|---|
| compatibility fingerprint and allowed actions | candidate eligibility and selected action/pool | `action`, `pool`, compatibility provenance |
| current and event prefill/decode capacity | ongoing landed work | ongoing prefill/decode use, capacity, normalized slack |
| stable capacity, debt budget | transition queue and recovery | service debt, recovery, capacity, normalized slack |
| migration timing evidence | candidate duration and pool occupancy | migration replica-seconds, capacity, normalized slack |
| usable live-KV blocks | post-commit KV placement | KV blocks used, capacity, normalized slack |
| route path and scenario link rates | transfer start/finish and route occupancy | route bytes, capacity, normalized slack |

For the end-to-end evaluation, each selected migration needs session, action,
order, destination, start, migration-ready, commit, and first-token times.
Each scenario needs requested and achieved shed, replay/KV counts, last commit,
route bytes/queue, destination work, exposed sessions, deadline status, seed,
workload, and input provenance. More detailed resource rows remain available
for diagnosis but are not required in one universal table.

## Service flex

The destination contract separates three quantities:

1. **Normal operating point:** the load at which the pool normally meets its
   latency policy.
2. **Event admission limit:** the ongoing work the operator accepts after
   handoff. It may be above normal but cannot exceed measured stable capacity.
3. **Transition debt:** temporary serving work queued during replay.

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
| transition work | replica-s | replay work during handoff |
| service debt | replica-s | work not served during the migration window |
| recovery | s | debt divided by post-migration spare service |
| live KV | physical blocks | private session state accepted by the pool |
| route demand | bytes | bytes sent over one logical route |
| route supply | bytes/s | event bandwidth advertised for that route |
| migration occupancy | replica-s | conservative combined pool migration work |

Memory is a stock, not a queue. V1 rounds each session independently and gives
no credit for shared prefixes. The pool advertises usable KV after accounting
for internal fragmentation.

An integrated pool exposes a prefill/decode sharing rule. The two-A100
migration mechanisms are measured, but the destination service boundary is not;
simulated service headroom therefore remains assumed. Disaggregated pools are
an optional sensitivity.

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
- `migration.py` defines ordered eager-parallel hardware execution.

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
| source accelerator power curves | measured; used as model input |
| replay and KV correctness | measured on two A100s |
| migration completion | 24/24 serial and 90/90 bounded migrations completed by deadline |
| bounded campaign gates | 105/105 passed |
| replay/KV duration | conservatively fitted with held-out context/bandwidth |
| KV bytes and block size | exact for the pinned ABI |
| live-KV capacity | measured vLLM readback |
| loaded migration effect | 18 observations; 12 overlap foreground work |
| planner-driven width-eight policy execution | completed; ordered eager-parallel dedicated sink |
| normal/stable service boundary | not accepted; optional for a measured shared-serving claim |
| route bandwidth/RTT | assumed sensitivity |
| pool flex and debt | operator sensitivity until testbed validation |
| multi-pool inventory | assumed sensitivity |
| H100 and A100 TP=2 service | optional hardware-generality extension |
| disaggregated prefill/decode site | assumed sensitivity |

The invalid 2026-07-23 service cells remain excluded. The five
private-prefix-consistent points near 0.096953 are descriptive anchors, not
safe capacity.

## Optional targeted service measurement

This measurement is required only for a measured shared-destination admission
claim. It is not required to show that migration works on a dedicated
destination or to evaluate assumed service headroom in the simulator. If run,
use three resource mixes derived from coding, interactive coding, agentic, and
ShareGPT shapes:

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

## Evaluation flow

1. Validate compatibility and the stated input evidence domain.
2. Build replay and KV candidates.
3. Select at most one action and destination per session.
4. Launch the fixed plan eagerly in order, up to the selected concurrency.
5. Retain source ownership until route commit and verify the first destination
   token.
6. Validate the same schedule in the event simulator.

Live contract leasing is an optional production extension.

## Evaluation contracts

### Minimum simulator evaluation

Hold one compatible integrated-pool contract fixed while requested shed rises.
Compare Queue-Haul with replay-only, KV-only, static greedy, and Lagrangian
greedy. Report selected actions, achieved shed, last commit, route use,
destination work, exposed sessions, and deadline status. Show 10K, 100K, and
1M-session planning and execution behavior.

Plot achieved against requested shed and show the replay/KV mix. A compact
resource-slack view is useful but not required to duplicate every internal
resource in the main result.

### Optional pool and diversity extensions

Pool count may be varied under two explicitly defined regimes:

1. **Fixed total resources:** hold each summed physical budget constant and
   split it across the pools. Pool-specific routes and compatibility remain
   explicit. This isolates fragmentation, route diversity, and compatibility.
2. **Fixed resources per pool:** give each pool the same explicit budget, so
   every summed destination budget grows linearly with pool count. This
   measures added headroom until a source-side or other shared constraint
   binds.

Results must state the explicit route, migration occupancy, ongoing service,
debt, and KV settings; opaque scenario names are not sufficient.

Resource diversity and compatibility diversity are separate experiments:

- **Resource diversity:** vary pool route, migration occupancy, service, debt,
  or KV budgets while holding compatible action/pool choices fixed.
- **Compatibility diversity:** vary compatible action/pool choices while
  holding total physical resources fixed.

These experiments explain fragmentation and headroom, but they are not needed
to establish the primary ordered eager-parallel end-to-end result.

### Representative schedules

Show one representative measured mixed-action schedule after the
planner-driven two-A100 campaign and one representative simulator schedule.
For each:

- x-axis: time;
- y-axis: destination pools;
- rectangle: one migration, with width equal to scheduled duration;
- fill or hatch: replay versus KV;
- markers: commit and first-token completion;
- overlays: shared route and transition-resource occupancy; and
- annotation: final achieved shed and complete binding-resource set.

The schedules show where end-to-end time goes and when source ownership can
move.

### Scale and sensitivity

The large simulator consumes advertised residual vectors; it does not generate
unrelated destination traffic. Primary capacity is relative to the source
relocation requirement. Pool-relative 0/5/10/20% flex is a sensitivity.

Use 10K, 100K, and 1M source sessions for the main scale result. One main
workload and a compact workload robustness view are sufficient. Deadline,
route, pool-count, skew, and diversity grids are optional sensitivities.

Coding, interactive coding, agentic, and ShareGPT-like conversation are
separate workload scenarios. Deadlines of 30/60/120/300 seconds, routes of
1/5/10 Gbps, 0/5/10/20% flex and debt, workload, source packing, and seed are
scenario axes, not error bars. Service flex, debt, multi-pool inventory,
disaggregated pools, unmeasured routes, and unmeasured hardware points remain
visibly `assumed/sensitivity` until their targeted measurements pass.

An 8+8 A100 experiment would add hardware-scale validation but is not required
for the two-A100 ordered eager-parallel claim.
