# Queue-Haul State - 2026-07-06

## Status

Queue-Haul is now a static selection optimizer plus deterministic execution checker for moving stateful LLM sessions under a source-site power shed command.

The current result is clear:

- Additive session-power LP is the wrong main method for node-level grid relief.
- Source-node-aware active-knee selection fixes the main power modeling error.
- Whole-job active-knee MILP is the most credible method under DES validation.
- Stage 1a gpt-oss-20b A100 TP1 traces show a stable concave/saturating power curve, but the absolute `ell` axis is a provisional colocated-load calibration.
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
- `ell = f / rho_dest(T) + g / G`: colocated service-time load.
- `m = eta * T`: KV memory/state size.
- `source_node`: current source GPU node.
- active/idle/cold state, with current canonical plots focused on active-agentic stress cases when testing migration.

The scalar `ell` assumes prefill and decode time-share one colocated GPU budget. It is not automatically the right axis for disaggregated inference. If the measured service surface does not collapse to one scalar, the state should stay vector-valued: `(u_pre, u_dec, u_ing, m)`.

The default model center is Qwen3-235B-A22B-like:

- `eta = 188 KiB/token` KV scale.
- `beta = 4 B/token` context transfer scale.
- BF16, 8xH100-style node assumptions.
- Node power is a ramp-then-plateau curve, not a per-session additive constant.

The Stage 1a measured calibration is separate: gpt-oss-20b on one A100 with TP1. Its current role is to test the load/power shape and the stability of `rho(T)` and `G(T)`, not to replace every Queue-Haul fleet parameter yet.

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

Stage 1a adds a fourth distinction: measured node power as a function of observed service rates. The 5s curve uses `ell = f/F + g/G` from raw rate windows and reaches `ell_max = 2.03`, beyond the simulator's per-node `rho*` placement ceiling. That does not break the formulation; it means the measurement x-axis is offered load under a probe, not a feasible admission state.

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

The implementation exhaustively enumerates active source-node knee regions for small source-node counts and solves fixed-region LP/MILP subproblems. It hard fails above the exhaustive cap rather than silently using a heuristic.

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
- `queue-haul/plot_node_knee_kappa_sweep.py`: planner rebuild-cushion sensitivity.
- `queue-haul/power_profile_reduce.py`: rebuild measured gpt-oss-20b A100 TP1 `ell` and power curves from raw windows.
- `queue-haul/power_window_sensitivity.py`: sweep rate/power window sizes and compare concave-fit stability.
- `queue-haul/service_surface_runner.py`: generate the A100 runbook for isolated `rho(T)`, `G(T)`, and mixed prefill/decode probes.
- `queue-haul/service_profile_reduce.py`: reduce completed service-surface probe bundles into CSV/PNG/PDF artifacts.

## Current Figures

Canonical outputs are in `queue-haul/outputs/`:

- `node_knee_target_sweep.*`
- `node_knee_deadline_sweep.*`
- `node_knee_execution_validation.*`
- `node_knee_fixed_plan_replay.*`
- `node_knee_agentic_des_sweep.*`
- `node_knee_scale_workload_sweep.*`
- `node_knee_kappa_sweep.*`
- `stage1_gpt_oss_20b_a100_tp1.{png,pdf}`
- `stage1_gpt_oss_20b_a100_tp1_curve.csv`
- `stage1_gpt_oss_20b_a100_tp1_constants.csv`
- `stage1_gpt_oss_20b_a100_tp1_power_curve.csv`
- `stage1_gpt_oss_20b_a100_tp1_window_sensitivity.*`
- `stage1_gpt_oss_20b_a100_tp1_window_sensitivity_{summary,binned}.csv`

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
target = 9.213 kW, 45% of full node-expected removable power
active-knee LP relaxation first hit: 45s, 27.9 s/kW requested
active-knee MILP first hit:          45s, 28.5 s/kW requested
live greedy first hit:               45s, 36.0 s/kW requested
random jobs first hit:               45s, 54.2 s/kW requested
additive LP:                         never hits
```

Interpretation: under tight deadlines, node-aware selection is the main differentiator. Additive selection never reaches the node target in this setup.

### Execution Validation

For active-knee plans replayed through DES:

```text
operational re-solve, 45% full-node target:
  active-knee MILP selected hit: 45s
  active-knee MILP egress hit:   45s
  active-knee MILP rebuild hit:  45s

  active-knee LP selected hit:   45s
  active-knee LP egress hit:     45s
  active-knee LP rebuild hit:    45s
```

Fixed-plan replay, solving once at `D_ref = 300s`, reaches egress and rebuild target at `184.1s` for MILP and `160.9s` for the LP relaxation.

Interpretation: after no-wait deadline filters, operational re-solves no longer certify plans that cannot rebuild by the active deadline. Fixed-plan replay still shows the gap between selected power and deadline-realized execution.

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
active-knee LP relaxation:   97/108 hits
active-knee MILP:            97/108 hits
live greedy:                 96/108 hits
random jobs:                 87/108 hits
```

Agentic is again the stress case. Active-knee misses are concentrated at `D = 10s`.

### Stage 1a Power Calibration

For gpt-oss-20b on one A100 with TP1, the measured `ell`-vs-power curve is distinctively concave/saturating. The separate saturating fit is the useful "power story"; the scalar x-axis is still a modeling hypothesis.

Window sensitivity:

```text
window  saturating R2  linear R2  ell knee  rho*  ell max
1s      0.975          0.792      1.17      0.44  2.44
2s      0.984          0.812      1.50      0.52  2.22
5s      0.990          0.846      2.05      0.53  2.03
10s     0.993          0.874      2.53      0.62  2.01
30s     0.997          0.903      3.12      0.59  1.90
```

Read: the concave shape is robust to windowing, but `F`, `G`, `rho*`, and the absolute `ell` knee are distributional/window-dependent normalizers. This supports careful positioning, not a universal utilization law.

The next hardware run is the service surface:

```text
uv run python queue-haul/service_surface_runner.py \
  --run-id gpt-oss-20b-a100-tp1-service-surface \
  -- --async-scheduling

uv run python queue-haul/service_profile_reduce.py \
  --run-dir queue-haul/runs/stage1_service_surface/gpt-oss-20b-a100-tp1-service-surface/bundles
```

It measures isolated prefill `rho(T)`, decode `G(T)` across context, and the mixed prefill/decode interaction. Those outputs decide whether the simulator keeps a scalar `ell` or moves to a vector service model.

## What We Can Claim Now

1. Source-node geometry matters. Session-additive shed is not enough.
2. Active-knee selection is the right current mainline.
3. Whole-job MILP is the deployable active-knee baseline; the LP relaxation is a lower bound.
4. DES validation is necessary because aggregate rebuild budgets do not prove reconstruction by deadline.
5. Egress-realized and rebuild-realized power answer different operational questions.
6. On the current gpt-oss-20b A100 TP1 trace, node power is better described by a concave/saturating curve than by a linear curve across 1/2/5/10/30s windows.

## What We Cannot Claim Yet

1. This is not a stochastic queueing model.
2. This is not a full online controller.
3. This is not exact scheduling inside the optimization.
4. This does not yet model detailed time-varying background serving contention.
5. This does not yet model prefix sharing, novel state reuse, LMCache/Mooncake/vLLM block behavior, or KV compatibility failures.
6. This does not prove that active in-flight decode migration is possible. Current semantics are future session reconstruction/resume.
7. This does not prove that scalar `ell` generalizes to disaggregated inference.
8. This does not yet provide measured `rho(T)`, context-dependent `G(T)`, or mixed-surface constants for the simulator; the runbook exists, but the A100 service-surface run still has to be executed.

## Critical Assumptions

- Rebuild capacity uses the destination spare pool, not a dedicated reconstruction pool. The remaining approximation is time-varying contention between rebuild work and post-rebuild serving.
- Destination active-load and held-capacity constraints are aggregate headroom constraints, not detailed time-varying admission.
- LP/MILP prefill and ingest rows enforce total work budgets, not release-time-aware schedules.
- Egress is modeled as one serial source link, so aggregate egress budget aligns more directly with DES than rebuild rows do.
- Node power is modeled by a synthetic ramp-then-plateau curve.
- Current workload distributions and model parameters are synthetic/proxy, not measured from a production trace.
- `ell` is valid as a scalar only under colocated prefill/decode sharing. The measured Stage 1a `F`/`G` constants are distributional normalizers, not fixed physical constants.
- Seeds differ across some plots, so kW values across figure families are not one-to-one comparable.

## Subagent Critique Distilled

The Stage 0 review found and fixed these issues:

1. `active_floor_w` now reports the true conservative floor `dp_guaranteed`, not the certified dispatch target.
2. Active-knee LP/MILP now exhaustively enumerates small source-node active regions and hard fails above the exhaustive cap.
3. Deadline resource windows clamp to zero before startup ramps.
4. Invalid movement bandwidth/utilization parameters hard fail.
5. Root docs and CSV metadata now reflect full-node target basis, shared spare-pool rebuild semantics, no-wait deadline filters, and `kappa` sensitivity.

Remaining issues: exact rebuild-realized optimization is still a scheduling problem, and background-serving contention still needs measured testbed calibration.

## On Deadline-Realized Formulations

A `z`-variable formulation is directionally right but not sufficient by itself:

```text
y = moved
z = counted by deadline
z <= y
target uses node_expected(z)
movement budgets use y
```

This fixes the accounting only if `z` is constrained by actual realizability. Current cheap constraints:

- no-wait egress + rebuild lower-bound filters;
- conservative rebuild capacity margin `kappa < 1`;
- optional fixed-order prefix constraints for a chosen ordering policy.

Exact rebuild-realized optimization is a scheduling MILP or time-indexed formulation. That is too much for the current stage. Receding horizon should come after the static accounting is fixed.

## Recommended Next Steps

Near term:

1. Make active-knee MILP plus DES the main result.
2. Keep additive LP as a baseline showing why additive power is insufficient.
3. Add prefix/novel-state manifest abstraction.
4. Run the Stage 1a service-surface probes on the A100 node and reduce `rho(T)`, `G(T)`, and mixed interaction plots.
5. Measure replay, state transfer, prefix reuse, and state materialization on a serving stack.
6. Add background destination serving contention.
7. Add chunked replay as a DES option.
8. Add receding-horizon control only after static execution-aware accounting is stable.

Then:

1. Re-parameterize the fleet simulator from measured curves.
2. Build the online router and compare against static active-knee MILP as an oracle baseline.

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
-> measured Stage 1a traces motivate the concave node-power curve, while service-surface probes decide whether scalar ell is sufficient
```

The paper should emphasize that Queue-Haul is not merely "move bytes elsewhere." The scarce resource is deadline-realized reconstruction of useful LLM state under a source-node power target.
