# Queue-Haul State - 2026-07-05

## Status

Queue-Haul is now a static selection optimizer plus deterministic execution checker for moving stateful LLM sessions under a source-site power shed command.

The current result is clear:

- Additive session-power LP is the wrong main method for node-level grid relief.
- Source-node-aware active-knee selection fixes the main power modeling error.
- Whole-job active-knee MILP is the most credible method under DES validation.
- The remaining hard issue is not source selection alone. It is whether selected sessions actually egress and rebuild by the deadline.

## Why This Matters

AI datacenters are becoming power-dense grid actors. Flexibility contracts increasingly ask them to reduce load quickly, but the mechanism is under-specified: curtailment, geographic load balancing, demand response, generators, islanding, and migration all mean different things operationally.

LLM workloads make migration materially different from classical VM or web-service migration:

- Sessions are stateful: context, tool traces, retrieved files, user memory, and KV state matter.
- GPU availability is scarce and power-dense.
- KV cache can be much larger than text context.
- Rebuilding state at a destination creates prefill storms.
- A grid deadline turns "can this session resume elsewhere?" into a scheduling problem.

Queue-Haul studies the migration primitive for this setting: choose which source sessions to move, how to reconstruct them, and whether the move produces usable power relief by a deadline.

## Current Scope

Queue-Haul models a static shed event:

```text
source site receives a shed command
-> choose sessions to move
-> choose replay vs compatible state transfer
-> route to destination capacity
-> replay the chosen plan through finite resources
-> compare selected, egress-realized, and rebuild-realized power
```

This is not yet an online stochastic queueing model. The simulator is a deterministic finite-server replay checker for a solved plan.

## Session Model

Each session/job has:

- `T`: context tokens.
- `f`: expected future prefill tokens or future prompt work.
- `g`: expected future decode tokens.
- `ell = f / rho_dest(T) + g / G`: active load.
- `m = eta * T`: KV memory/state size.
- `source_node`: current source GPU node.
- active/idle/cold state, with current canonical plots focused on active-agentic stress cases when testing migration.

The default model center is Qwen3-235B-A22B-like:

- `eta = 188 KiB/token` KV scale.
- `beta = 4 B/token` context transfer scale.
- BF16, 8xH100-style node assumptions.
- Node power is a ramp-then-plateau curve, not a per-session additive constant.

## Actions

For each job `j` and destination `l`, the solver can choose:

- Replay: send compact context, then recompute KV by prefill.
- State transfer: send compatible KV/state, then ingest/admit it.

The single-request crossover is:

```text
beta*T/lambda + T/rho < eta*T/lambda
lambda* = rho * (eta - beta)
```

This is only a single-job guide. The fleet problem is harder because many jobs share one source egress link and finite destination rebuild/ingest capacity.

## Power Semantics

There are three different power quantities. Keeping them separate is essential.

```text
active certified watts
  additive lower-bound-like session certificate used by the original dispatch LP

node_expected watts
  source-node power reduction from P(L_i) - P(L_i - removed_i)

active floor watts
  conservative active-work floor, useful as a diagnostic but not the node-knee objective
```

The paper-facing result should primarily report node-expected shed for source-node relief and should explicitly state whether grid relief is counted at egress completion or rebuild completion.

## Formulations

### Additive Dispatch LP

Variables:

```text
Y_R[j,l], Y_S[j,l] in [0,1]
sum_l (Y_R[j,l] + Y_S[j,l]) <= 1
```

Objective:

```text
min total movement seconds
```

Target:

```text
sum_j dp_certified[j] * moved[j] >= requested shed
```

Constraints:

- one shared source egress budget;
- per-destination prefill work budget;
- per-destination ingest budget;
- destination active-load headroom;
- destination held-state capacity;
- pinned jobs cannot move.

This LP is useful as a baseline, but it optimizes the wrong source-power geometry. It spreads removals across source nodes because the target is additive by session. That can look cheap while barely crossing any node power knee.

### Active-Knee LP/MILP

The active-knee formulation changes the target from additive session watts to source-node power:

```text
removed_i = sum_{j on node i} ell_j * moved_j
node shed = sum_i P(L_i) - P(L_i - removed_i)
```

The implementation enumerates active source-node knee regions and solves fixed-region LP/MILP subproblems.

- LP relaxation: fractional sessions allowed.
- MILP: whole sessions only.

This keeps the dispatch resource constraints but makes source relief node-aware.

### Deterministic Replay Checker

The DES takes a solved plan and simulates execution:

```text
shared serial source egress
per-destination prefill servers
per-destination ingest servers
finite reconstruction release times
ordering policy
deadline cutoff
```

It reports:

```text
selected_node_expected_w
egress_realized_node_expected_w
rebuild_realized_node_expected_w
selected_s_per_kw
egress_realized_s_per_kw
rebuild_realized_s_per_kw
```

The key distinction:

- selected: optimizer claims the plan contains enough modeled node power.
- egress-realized: jobs whose source egress finishes by `D`.
- rebuild-realized: jobs fully reconstructed at destination by `D`.

## Ordering Policies

Current policies:

- FIFO.
- LPT.
- certified power density.
- Johnson-like diagnostic.
- live node-marginal power density.

The important new policy is live node-marginal PD:

```text
score_j = live marginal node watts from removing j / egress seconds_j
```

It updates residual source-node load as jobs are ordered, so the value of finishing the next job reflects the current node knee.

## Implementation Map

- `queue-haul/power.py`: node power, model constants, prefill/decode rates.
- `queue-haul/instance.py`: workload/session generation and source placement.
- `queue-haul/impact.py`: per-job active, expected, memory, replay, and transfer quantities.
- `queue-haul/dispatch.py`: additive LP/MILP baseline and shared movement columns.
- `queue-haul/node_knee.py`: node-knee evaluator, active-knee LP/MILP, greedy/oracle baselines, execution metrics.
- `queue-haul/simulate.py`: deterministic replay checker and ordering policies.
- `queue-haul/plot_node_knee_target_sweep.py`: requested shed target sweep.
- `queue-haul/plot_node_knee_deadline_sweep.py`: fixed target vs deadline.
- `queue-haul/plot_node_knee_execution_validation.py`: selected vs egress vs rebuild validation.
- `queue-haul/plot_node_knee_agentic_des_sweep.py`: agentic requested-shed DES disruption plot.
- `queue-haul/plot_node_knee_scale_workload_sweep.py`: workload and source-node scale sweep.

## Current Figures

Canonical outputs are in `queue-haul/outputs/`:

- `node_knee_target_sweep.*`
- `node_knee_deadline_sweep.*`
- `node_knee_execution_validation.*`
- `node_knee_fixed_plan_replay.*`
- `node_knee_agentic_des_sweep.*`
- `node_knee_scale_workload_sweep.*`

## Main Results

### Fixed Target Sweep

At 4 source nodes, `D = 300s`, and targets from 5% to 95% of full node-removable power:

```text
ordinary chat:
  active-knee LP/MILP hit 19/19, median 7.7 s/kW
  additive LP hit 13/19, median 21.9 s/kW

long chat/code:
  active-knee LP/MILP hit 19/19, median 6.4 s/kW
  additive LP hit 16/19, median 14.3 s/kW

reasoning chat:
  active-knee LP/MILP hit 19/19, median 10.0 s/kW
  additive LP hit 9/19, median 15.3 s/kW

agentic tool loop:
  active-knee LP hit 19/19, median 20.8 s/kW
  active-knee MILP hit 19/19, median 21.8 s/kW
  additive LP hit 4/19, median 87.5 s/kW
```

Interpretation: agentic workloads are the stress case. Additive LP can move active work without producing enough source-node power relief.

### Deadline Sweep

Agentic fixed-target recomputation:

```text
target = 6.253 kW
active-knee LP relaxation first hit: 10s, 24.5 s/kW
active-knee MILP first hit:          10s, 24.0 s/kW
live greedy first hit:               12s, 35.1 s/kW
random jobs first hit:               16s, 59.2 s/kW
additive LP:                         never hits
```

Interpretation: under tight deadlines, node-aware selection is the main differentiator. Additive selection never reaches the node target in this setup.

### Execution Validation

For active-knee plans replayed through DES:

```text
operational re-solve, 45% full-node target:
  active-knee MILP selected hit: 14s
  active-knee MILP egress hit:   14s
  active-knee MILP rebuild hit:  45s

  active-knee LP selected hit:   14s
  active-knee LP egress hit:     14s
  active-knee LP rebuild hit:    184.1s
```

Fixed-plan replay, solving once at `D_ref = 300s`, reaches egress and rebuild target at `184.1s`.

Interpretation: the LP relaxation can select fractional/spread movement that looks good as selected power but restores late. The MILP whole-job plan survives execution much better.

### Agentic DES Requested-Shed Sweep

This is the requested plot shape:

```text
x-axis: requested node-expected shed
y-axis: disruption intensity, s/kW
curves: selected, egress-realized, rebuild-realized
```

At `D = 300s` on the active-agentic workload:

```text
active-knee MILP:
  selected/egress/rebuild hit 19/19
  rebuild intensity range 7.4-33.1 s/kW

active-knee LP relaxation:
  selected/egress hit 19/19
  rebuild hit 18/19
  rebuild intensity range 7.1-29.0 s/kW

live greedy:
  hit 19/19
  intensity range 17.4-86.6 s/kW

random jobs:
  hit 19/19
  intensity range 80.4-574.4 s/kW

additive LP:
  hit 4/19
  first plotted successful target near 16.5 kW
  intensity range 30.5-103.0 s/kW
```

Why additive LP appears late: the plot only includes points where the method actually hits the requested node-expected shed. Additive LP optimizes an additive active-work certificate, not node-knee power, so it misses low and medium node-power requests until the requested target is high enough that its dispersed removals happen to cross enough node knees.

### Scale/Workload Sweep

Across 1/2/4 source nodes, 4 workloads, `D in {10,30,120}s`, and target fractions `{25%,45%,65%}`:

```text
additive LP:                 54/108 hits
active-knee LP relaxation:  104/108 hits
active-knee MILP:           101/108 hits
live greedy:                 99/108 hits
random jobs:                 86/108 hits
```

Agentic is again the stress case. Active-knee misses are concentrated at `D = 10s`.

## What We Can Claim Now

1. Source-node geometry matters. Session-additive shed is not enough.
2. Active-knee selection is the right current mainline.
3. Whole-job MILP is more execution-valid than the LP relaxation.
4. DES validation is necessary because aggregate rebuild budgets do not prove reconstruction by deadline.
5. Egress-realized and rebuild-realized power answer different operational questions.

## What We Cannot Claim Yet

1. This is not a stochastic queueing model.
2. This is not a full online controller.
3. This is not exact scheduling inside the optimization.
4. This does not yet model background serving contention.
5. This does not yet model prefix sharing, novel state reuse, LMCache/Mooncake/vLLM block behavior, or KV compatibility failures.
6. This does not prove that active in-flight decode migration is possible. Current semantics are future session reconstruction/resume.

## Critical Assumptions

- `W` is treated in code as dedicated reconstruction capacity. If `W` actually shares GPUs with destination serving, current feasibility is optimistic and may double-count capacity.
- Destination active-load and held-capacity constraints are aggregate headroom constraints, not detailed time-varying admission.
- LP/MILP prefill and ingest rows enforce total work budgets, not release-time-aware schedules.
- Egress is modeled as one serial source link, so aggregate egress budget aligns more directly with DES than rebuild rows do.
- Node power is modeled by a synthetic ramp-then-plateau curve.
- Current workload distributions and model parameters are synthetic/proxy, not measured from a production trace.
- The deadline sweep target basis currently differs from most node-knee sweeps: it uses a fraction of active certified power, while most newer plots use full node-expected removable power.
- Seeds differ across some plots, so kW values across figure families are not one-to-one comparable.

## Subagent Critique Distilled

The review agents found three issues that matter most:

1. Rebuild capacity semantics must be cleaned up. The implementation treats `W` as dedicated reconstruction servers, but some prose describes it like full serving nodes that also serve background work.
2. The LP/MILP and DES now share movement columns, but aggregate rebuild rows still do not imply rebuild-by-deadline completion.
3. Some documentation is stale: memory-regime ranking and dual-unit interpretations overclaim relative to the current implementation.

Secondary issues:

- `simulate(mode=...)` should hard fail on unknown modes.
- Active-knee candidate failures are currently too quiet.
- Plot scripts are more cwd-sensitive than ideal.
- Some plots emit CSVs while deadline sweep only emits figures.
- Tests cover semantics well, but artifact writing/visual validity is not deeply tested.

## On Deadline-Realized Formulations

A `z`-variable formulation is directionally right but not sufficient by itself:

```text
y = moved
z = counted by deadline
z <= y
target uses node_expected(z)
movement budgets use y
```

This fixes the accounting only if `z` is constrained by actual realizability. Cheap next constraints:

- no-wait egress + rebuild lower-bound filters;
- conservative rebuild capacity margin `kappa < 1`;
- optional fixed-order prefix constraints for a chosen ordering policy.

Exact rebuild-realized optimization is a scheduling MILP or time-indexed formulation. That is too much for the current stage. Receding horizon should come after the static accounting is fixed.

## Recommended Next Steps

Near term:

1. Make active-knee MILP plus DES the main result.
2. Keep additive LP as a baseline showing why additive power is insufficient.
3. Standardize target basis across figures.
4. Clarify `W`: dedicated rebuild pool vs shared destination serving capacity.
5. Add no-wait deadline filters before adding larger scheduling machinery.
6. Add rebuild-capacity safety margin and report sensitivity.
7. Make simulator modes hard fail on invalid input.

Then:

1. Add prefix/novel-state manifest abstraction.
2. Measure replay, state transfer, prefix reuse, and state materialization on a serving stack.
3. Add background destination serving contention.
4. Add chunked replay as a DES option.
5. Add receding-horizon control only after static execution-aware accounting is stable.

## Paper Story So Far

The clean layered story is:

```text
classical migration ideas do not directly apply to stateful LLM sessions
-> single-job replay vs KV transfer has a simple crossover
-> fleet shed is harder because many sessions share egress and rebuild capacity
-> additive session-power LP misses source-node power knees
-> active-knee LP/MILP selects the right source-node removals
-> DES separates selected, egress-realized, and rebuild-realized relief
-> whole-job active-knee MILP is the current execution-valid mainline
```

The paper should emphasize that Queue-Haul is not merely "move bytes elsewhere." The scarce resource is deadline-realized reconstruction of useful LLM state under a source-node power target.
