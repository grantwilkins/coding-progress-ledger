# Queue-Haul evidence roadmap

## Evaluation boundary

The main result has two parts:

1. **Measured two-A100 execution:** Queue-Haul can replay or transfer KV state,
   preserve the session, switch at a request boundary, and continue at the
   destination. The primary executor is sequential.
2. **Simulated coordination:** Given measured/fitted migration primitives and
   an explicitly assumed destination contract, Queue-Haul chooses actions and
   predicts deadline-feasible source accelerator-power shed at larger scale.

The completed experiments validate the mechanisms. The prepared policy
campaign will validate planner-driven sequential execution. The simulator
evaluates scale and resource coordination. Neither is a production admission
certificate or a facility-power measurement.

## What the retained data already shows

- `outputs/serial-power-run-2`: 24/24 replay or KV migrations completed and met
  the campaign deadline at 1 and 10 Gbps.
- `outputs/bounded-hardware-campaign-run`: 90/90 migrations completed and met
  the deadline; all 105 campaign gates passed across replay/KV, one-turn and
  idle activity, 1/10 Gbps, and migration concurrency 1/2/4.
- `outputs/parallel-kv-gate-run-2`: 6/6 KV concurrency gates passed.
- `outputs/mp-campaign-run-10-20260719/report.json`: parallel KV accounting,
  overlap, wire, token, and continuation gates passed.
- `outputs/mp-incremental-run-3-20260720/report.json`: exact incremental blocks,
  wire bytes, destination coverage, no duplicate prefix traffic, and real
  continuation passed.
- `outputs/mechanism-validation`: measured replay and KV Gantt charts and
  migration-to-first-token rows.
- `outputs/mechanism-validation/migration_ttft_cdf.csv`: 18 loaded replay/KV
  observations at 16K/10 Gbps and 24K/5 Gbps; 12 overlap foreground work.

This is sufficient to claim that both migration paths work on the pinned
two-A100 setup and to show where their time goes.

## Remaining measured result

The missing end-to-end link is not another mechanism campaign. It is execution
of a planner-generated mixed action list.

`policy_hardware_campaign.py` already:

- asks the planner for Queue-Haul replay/KV choices and order;
- executes all eight migrations eagerly and sequentially;
- compares matched Queue-Haul, greedy, per-session-fastest, random, KV-only,
  and replay-only episodes; and
- reduces completion, first-token, commit, migration-TTFT, and matched
  continuation-TTFT results.

Run that prepared campaign and add one mixed-action Gantt chart. This closes the
two-A100 end-to-end performance story.

## Remaining simulator result

Run `canonical_simulator_campaign.py`, which freezes one canonical sequential
contract, and show:

- selected replay/KV mix and order;
- requested versus achieved source accelerator-power shed;
- last commit by the deadline;
- route bytes/queue, destination work, and exposed sessions;
- comparison with replay-only, KV-only, isolated-fastest, and greedy; and
- planning and execution behavior at 10K, 100K, and 1M sessions.

One main workload plus a compact robustness view is sufficient. The full
deadline × route × flex × debt × pool × workload Cartesian product is not
required.

## Evidence labels

- **Measured:** two-A100 mechanism timing, continuation, exact KV
  blocks/bytes, and recorded foreground-overlap observations.
- **Fitted:** conservative replay/KV duration models inside their measured
  context and bandwidth ranges.
- **Simulated:** planner choice, schedules, queues, and power outcomes.
- **Assumed/sensitivity:** destination service headroom, route reservation,
  multi-pool inventory, and unmeasured hardware.

The archived destination service campaign does not establish a safe capacity
boundary. That does not invalidate the dedicated two-A100 migration result; it
only prevents describing assumed simulator headroom as measured admission
capacity.

## Optional extensions

Destination boundary measurement is needed only for a measured shared-serving
admission claim. Live leases are needed only for production admission.
Parallel scheduling, 8+8 A100, H100, A100 TP=2, disaggregated pools, and
exhaustive diversity sweeps improve generality but are not required for the
sequential end-to-end claim.

## Final claim

After the prepared policy campaign passes, Queue-Haul may claim:

> On the pinned two-A100 setup, replay and KV-transfer handoffs preserve
> session state and resume at the destination. Queue-Haul chooses and
> sequentially executes these actions, while its simulator evaluates
> deadline-constrained source accelerator-power shed at larger scale using
> measured, fitted, and explicitly assumed inputs.

It may not claim measured production destination headroom, facility/grid power,
arbitrary mid-token migration, or parallel planner execution without the
corresponding evidence.
