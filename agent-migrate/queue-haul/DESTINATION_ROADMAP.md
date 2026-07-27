# Queue-Haul evidence roadmap

This roadmap orders the work by dependency. It does not assign dates. The
canonical task checklist is `TODO.md`.

## Current position

Queue-Haul already has:

- measured A100 source power curves;
- working request-boundary replay and KV handoff on two A100s;
- conservative replay/KV timing fits;
- exact KV block accounting;
- a requirement-frontier solver;
- LP, exact integer, and greedy planning paths;
- aggregate destination-pool planning;
- an event simulator; and
- 10K–1M-session scaling machinery.

The main gaps are:

- source power is not validated on held-out group removals;
- destination service has no accepted pass/fail boundary;
- the canonical assumed inputs exist, but not every older experiment consumes
  them yet;
- pool debt and recovery now have a deterministic fluid trace but still need
  testbed validation;
- existing figure outputs do not cover the full Q1–Q9 evaluation;
- plan-versus-execution validation is too small; and
- H100, A100 TP=2, and disaggregated-pool results are incomplete or assumed.

## Gate 1: make the complete assumed experiment runnable

Before collecting more GPU data:

1. centralize every assumed operating point;
2. mark it `TODO: ASSUMED` in code and `assumed` in result records;
3. add units, validity range, and required replacement evidence;
4. implement pool event capacity, service debt, and recovery;
5. emit one tidy result table for each memo question; and
6. generate every planned figure from those tables.

The simulator must use advertised destination residuals. It must not invent
unrelated destination arrivals or claim to predict destination latency.

Pass condition: all Q1–Q9 commands run from a clean checkout, assumptions are
visible in every affected row, and `uv run pytest` passes.

## Gate 2: validate source power

Use complete-run fit, calibration, and untouched final splits. Measure controlled
session-group removals at several source loads.

Pass condition: every final measured group sheds at least the watts Queue-Haul
credited. Measure shutdown delay separately and require off before the final
five-second power window.

## Gate 3: finish single-session evidence

Reuse the valid two-A100 replay/KV corpus. Add only missing 4K–8K context and
5-Gbps points.

Pass condition:

- correct context and KV state;
- exact bytes and blocks;
- no destination WAN fetch after commit;
- bounded quiesce;
- correct route switch;
- valid first post-switch token; and
- conservative handoff-time prediction in the stated domain.

## Gate 4: measure destination flex

Run the corrected targeted campaign for prefill-heavy, balanced, and
decode-heavy mixes.

Pass condition:

- complete streams;
- exact private-prefix cache state;
- normal and stable pass/fail points bracketed;
- at least three independent boundary runs;
- no false-safe final point; and
- measured queued work and recovery for 0/5/10/20% bursts.

If this gate fails, keep all service results labelled sensitivity.

## Gate 5: validate execution

First run the full event on two A100s. Then run the same pool contract on 8+8
A100s configured as independent TP=1 replicas.

Measure:

- source and destination accelerator power;
- route traffic and queued bytes;
- reconstruction and service queues;
- selected replay/KV actions;
- quiesce and route-switch times;
- first-token completion;
- source shutdown; and
- predicted versus realized makespan, debt, and recovery.

Pass condition: every accepted migration commits by the deadline, the source is
under the accelerator-power limit for the final window, and realized makespan,
debt, and recovery do not exceed the advertised contract.

## Gate 6: scale and diversity

Run coding, interactive coding, agentic, and ShareGPT conversation workloads at
10K, 100K, and 1M sessions with ten seeds.

Use:

- measured-normal source packing;
- balanced, moderate-skew, and high-skew sensitivity;
- 30/60/120/300-second deadlines;
- 1/5/10-Gbps routes;
- 0/5/10/20% service flex and debt;
- 1/2/4/8 destination pools;
- fixed total resources split over pools;
- fixed resources added per pool;
- separate resource-diversity and compatibility-diversity cases; and
- integrated and assumed prefill/decode-disaggregated sites.

Pass condition: every reported point has a complete provenance record and a
separate execution-validator result.

## Gate 7: hardware generality

After the A100 TP=1 path passes, measure H100 TP=1 and A100 TP=2. Collect the
smallest profile that identifies source power, prefill, decode, KV capacity,
replay, and KV ingest in the required domain.

Pass condition: hardware sensitivity plots use measured profiles. Any missing
dimension remains visibly assumed.

## Evaluation questions and plots

Use the questions and plots from the design memo.

### Q1. Does the source model predict session-removal power?

- predicted versus measured held-out group shed;
- averaging-window stability;
- show every negative safety margin as a failure.

### Q2. Are replay and KV distinct actions?

- measured action phase diagram over bandwidth and context;
- end-to-end breakdown of route, reconstruction/ingest, catch-up, scheduling,
  and software time.

### Q3. Does joint planning respond to contention?

- replay/KV mix versus route and reconstruction pressure;
- achieved shed and complete binding-resource set.

### Q4. How much can one pool accept?

- achieved versus requested shed;
- required route, transition, service, debt, recovery, and KV versus watts;
- binding-resource map.

### Q5. Does the plan execute?

- predicted versus realized makespan;
- one timeline with power, migrations, route use, pool queues, commits, and
  shutdown.

### Q6. Is greedy good enough?

- exact integer, LP bound, rounded/packed, greedy, and focused policy quality;
- planning time and memory through 1M sessions.

### Q7. What do more pools buy?

- maximum shed versus 1/2/4/8 pools;
- fixed-total versus fixed-per-pool budgets;
- separate resource and compatibility diversity.

### Q8. Does the abstraction survive hardware change?

- achieved shed as route, KV, prefill, decode, and ingest improve;
- action and destination changes at each bottleneck transition.

### Q9. Does the workload determine the result?

- four-workload robustness matrix;
- trace, fitted synthetic, and shifted future-like workloads;
- Poisson central testbed arrivals and equal-mean burst sensitivity.

## Baselines

Use:

- all replay;
- all KV;
- isolated-fastest;
- network-greedy;
- service-greedy;
- power-first then place;
- exact integer and LP references;
- round-robin and least-loaded for multiple pools; and
- immediate session drop as the non-migration comparison.

When migration cannot meet the target, report unmet watts, sessions, context
tokens, and KV bytes still exposed. Do not invent the operator's drop or lower
tier policy.

## Final claim boundary

Queue-Haul may claim:

> Given measured handoff primitives and advertised compatible-pool budgets,
> Queue-Haul computes and validates the source accelerator-power shed achievable
> by a deadline.

It may not claim full-facility power, arbitrary live decode migration, hidden
provider capacity, long-term destination equilibrium, or a safe destination
service envelope until the matching evidence gate passes.
