# Queue-Haul evidence roadmap

This roadmap orders work by the paper's evidence flow, not by implementation
component or date. The canonical task checklist remains `TODO.md`.

The paper distinguishes four evidence classes:

- **Measured:** source power, two-A100 replay/KV correctness, exact KV bytes and
  blocks, and measured capacities.
- **Fitted:** conservative migration-time and throughput models within stated
  domains.
- **Simulated:** coordination, schedules, queues, and power outcomes driven by
  measured, fitted, and assumed inputs.
- **Assumed / sensitivity:** unmeasured routes, destination service flex and
  debt, multi-pool inventories, disaggregated pools, and unmeasured hardware.

The simulator evaluates planning and scale under advertised residual budgets.
It does not invent unrelated destination traffic, and its outputs are not
direct evidence of production destination behavior.

## Current position

Queue-Haul has measured A100 source power curves; working request-boundary
replay and KV handoff on two A100s; conservative timing fits; exact KV block
accounting; requirement-frontier, LP, integer, greedy, and pool-aware planners;
an event simulator; and 10K-1M-session scaling machinery.

The principal gaps are held-out source group-removal validation, a bracketed
destination service boundary, complete end-to-end execution validation,
provenance-complete tidy tables for the new figure structure, and measured
hardware diversity. The archived destination campaign supplies descriptive
anchors only. Until the targeted rerun brackets passing and failing points,
service flex and debt remain sensitivities.

## Universal result and figure gates

The executable schedule is the primary system output. The requirement frontier
summarizes validated schedules across targets.

Every result row records experiment/scenario ID, seed, workload, source
hardware/model/packing, deadline and measurement window, requested/achieved/
unmet watts, selected sessions, replay/KV counts and bytes, pool assignments,
all resource use and normalized slack, the complete binding-resource set,
predicted and realized makespan, source shutdown, exposed work, and evidence
status, units, provenance, validity range, and replacement evidence for every
input.

Every schedule row records session ID, source, action, pool, start,
transfer/reconstruction finish, quiesce, commit, first-token completion, bytes,
transition work, ongoing work, KV blocks, and conservative source watts
credited.

Every main-paper figure has the same pass condition:

1. its documented command runs from a clean checkout;
2. its tidy input table contains complete provenance and passes the execution
   validator; and
3. the figure is generated only from that table, never from ad hoc simulator
   objects.

Negative slack, failed validations, unmet watts, and exposed work remain in the
tables and figures. Workload, packing, deadline, bandwidth, flex/debt, hardware,
and seed sweeps are scenarios, not statistical error bars.

## 1. Make the assumed experiment runnable

Centralize one canonical fixed-contract operating point: workload; source
packing, hardware, and model; one compatible integrated destination pool;
deadline; route bandwidth and RTT; event service flex; debt budget; usable KV;
reconstruction and ingest capacity; and seed. Reuse existing central defaults
if complete; otherwise choose and document one mid-range point.

For each value, record units, measured/fitted/assumed status, provenance,
validity range, and replacement evidence. Make the fixed-contract, multi-pool,
planner-quality, and scale commands emit the universal result and schedule
tables.

Keep 30/60/120/300-second deadlines, 1/5/10-Gbps routes, 0/5/10/20% flex and
debt, alternate workloads and packings, and seed variation as separate
sensitivities.

Pass condition: the assumed canonical experiment and every planned figure
command run from a clean checkout, the tables pass schema/provenance checks and
the execution validator, and `uv run pytest` passes.

## 2. Validate source power with held-out group removals

Use complete-run fit, calibration, and untouched final splits. Measure
controlled whole-session group removals at several source loads, including
groups resembling planned selections.

Produce source power-model validation with predicted versus measured group
shed and averaging-window stability. Show every negative safety margin as a
failure.

Pass condition: every final held-out group sheds at least the conservative
watts credited. Measure shutdown delay separately and require the accelerator
off before the final power window.

## 3. Finish two-A100 replay/KV and execution validation

Complete missing context/bandwidth points in the valid two-A100 corpus. Validate
exact context, KV state, bytes, blocks, background preparation, overlap,
catch-up, request-boundary quiesce, route switch, first post-switch token,
timing jitter, realized source power change, and shutdown where applicable.

Produce:

- replay/KV single-session crossover and measured phase breakdown; and
- one end-to-end replay/KV timeline with time on the x-axis, source serving,
  bulk preparation, route transfer, reconstruction or KV ingest, catch-up,
  quiesce, route switch, first post-switch token, source power, and shutdown.

Overlay predicted and realized timing and report prediction error and safety
margin. A small companion scatter or table may cover measured migration time
and power shed across the corpus.

Pass condition: no destination WAN fetch occurs after commit, quiesce is
bounded, the first post-switch token is correct, state and byte accounting are
exact, and predicted handoff time is conservative in its stated domain.

Do not prioritize a three-A100 cross-region demonstration unless it exposes a
qualitatively new implementation constraint not covered by the two-A100
timeline. "It also works in three regions" is not a sufficient result.

## 4. Produce fixed-contract many-session figures

Hold the canonical one-pool contract fixed while requested shed rises through
10/25/50/75/90/100% of maximum modeled shed.

### B1. Resource slack versus requested shed

- x-axis: requested shed in watts and/or percent of maximum modeled shed;
- y-axis: normalized residual slack
  \((\mathrm{capacity}-\mathrm{use})/\mathrm{capacity}\);
- series or small multiples: source preparation, route bytes/time, replay
  reconstruction, KV ingest, ongoing prefill, ongoing decode, service debt,
  live-KV blocks, and deadline/realized makespan; and
- annotations: complete binding-resource set at each target.

Zero is binding and negative values are visible failures. The figure shows how
one fixed contract is spent.

### B2. Executable shed by planner

- x-axis: requested shed;
- y-axis: executable achieved shed;
- reference: \(y=x\);
- main series: Queue-Haul, all replay, all KV, best simple greedy, and exact
  integer or LP reference where tractable; and
- companion panel: selected replay/KV mix.

Report unmet watts and annotate the first infeasible target and complete
binding-resource set for each policy. Put the full baseline set in secondary
material.

Use the two-A100 timeline and predicted-versus-realized makespan, debt,
recovery, and power-shed columns to validate concrete scheduling; do not repeat
the Gantt chart here.

Pass condition: the universal figure gate passes and every planner point has a
concrete execution-validator result.

## 5. Complete destination service-boundary measurements

Run the corrected targeted campaign for prefill-heavy, balanced, and
decode-heavy mixes. Use open-loop arrivals, unique appends or a reset prefix
cache, exact private-prefix state, and 0/5/10/20% bursts.

Pass condition: complete streams; bracketed passing and failing normal/stable
points; at least three independent boundary runs; no restart, rejection,
missing work, wrong cache state, or false-safe final point; and measured queued
work and recovery. If the gate fails, service flex and debt remain
`assumed/sensitivity` in every affected row.

## 6. Produce multi-pool figures

This stage opens the fixed contract and varies pool count, composition,
compatibility, and headroom. Describe every scenario by explicit resource
settings, not an opaque name.

### C1. Maximum shed versus pool count

Use pool count 1/2/4/8 on the x-axis and maximum executable source
accelerator-power shed in watts and percent of maximum modeled shed on the
y-axis. Separate:

1. **Fixed total resources:** keep summed physical budgets constant and split
   them across pools.
2. **Fixed resources per pool:** give each pool the same budget so summed
   resources grow with pool count.

Attribute changes to added capacity, fragmentation, route diversity, or
compatibility diversity. Use representative workloads in the main paper and
the full sweep in secondary material.

### C2. Bottleneck-improvement small multiples

Use identical axes with resource capacity multiplier or advertised headroom on
the x-axis and maximum executable shed on the y-axis. Give route
bandwidth/queued bytes, replay reconstruction, KV ingest, ongoing prefill,
ongoing decode, event debt, and live-KV blocks separate panels. Each curve must
show the knee where another resource joins the binding set. Use 1/2/4/8-pool
lines only when readable.

### C3. Schedule morphing

Choose three or four explicit route-, reconstruction-, service-, and
KV-memory/ingest-constrained settings from C1-C2. Plot time on the x-axis and
destination pools on the y-axis. Draw one duration-width rectangle per
migration with replay/KV fill or hatch; mark commit and first-token completion;
show shared route and transition occupancy; and annotate final achieved shed
and the complete binding-resource set. Each example must explain a capacity
curve's knee or plateau.

### C4. Workload and diversity robustness

Keep resource diversity and compatibility diversity separate. Resource
diversity changes route, reconstruction, ingest, service, debt, or KV budgets
while compatibility is fixed. Compatibility diversity changes eligible
action/pool choices while total physical resources are fixed.

For coding, interactive coding, agentic, and ShareGPT-like conversation,
report maximum executable shed and complete binding-resource sets in a compact
binary or normalized-slack matrix. Do not assign one exclusive failure cause
when resources bind simultaneously.

Pass condition: the universal figure gate passes for C1-C4, and all simulated
points retain their measured, fitted, and assumed input provenance.

## 7. Validate the pool contract on 8+8 A100s

Run the same compatible integrated-pool contract on 8+8 A100s configured as
independent TP=1 replicas. Measure source and destination accelerator power,
route traffic and queued bytes, reconstruction and service queues, selected
actions, quiesce, route switch, first-token completion, shutdown, makespan,
debt, and recovery.

Pass condition: every accepted migration commits by the deadline, the source
stays below the accelerator-power limit for the final window, and realized
makespan, debt, recovery, and resource use stay within the advertised contract.
This validates the abstraction at a larger testbed point; it does not define a
production site or simulated inventory.

## 8. Run planner-quality and scale experiments

For planner quality, compare exact integer, LP bound, rounded/packed,
Queue-Haul greedy, and focused baselines on executable shed, resource debt,
optimality gap, planning time, and memory.

For scale, plot planning time and memory at 10K, 100K, and 1M sessions for ten
seeds. Every point retains provenance and an execution-validator result.

Pass condition: exact/relaxed gaps are reported wherever tractable, all large
points include validator status, and the universal figure gate passes.

## 9. Add measured hardware diversity

Only after the A100 TP=1 path passes, measure H100 TP=1 and A100 TP=2. Collect
the smallest profile that identifies source power, prefill, decode, KV
capacity, replay, and KV ingest throughout the required domain.

Pass condition: main hardware-robustness results use measured profiles. Any
missing dimension stays visibly assumed with units, validity range,
provenance, and required replacement evidence.

## Main-paper figure budget

1. Source power-model validation.
2. Replay/KV single-session crossover and measured breakdown.
3. Two-A100 end-to-end execution timeline.
4. Fixed-contract resource slack versus requested shed.
5. Queue-Haul versus coordinated-planning baselines.
6. Maximum shed versus 1/2/4/8 pools, fixed-total and fixed-per-pool.
7. Destination bottleneck-improvement small multiples.
8. Representative schedule morphing.
9. Planner quality and scale.
10. Compact workload/hardware robustness matrix.

Full deadline, route, workload, packing, seed, and hardware sweeps belong in
secondary material.

## Claim boundary

Queue-Haul may claim:

> Given measured handoff primitives and advertised compatible-pool budgets,
> Queue-Haul computes and validates the source accelerator-power shed achievable
> by a deadline.

It may not claim facility or grid power from accelerator measurements,
arbitrary mid-token migration, hidden provider capacity or unrelated
destination arrivals, safe service headroom before Gate 5 passes, long-term
destination equilibrium, production admission certificates without live
contract leasing and revalidation, or measured hardware generality for assumed
profiles. Infeasible targets report unmet watts and exposed work; partial
achievement is never successful curtailment.
