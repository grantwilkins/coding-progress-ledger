# Queue-Haul — Current Findings

Queue-Haul is currently focused on **node-aware power shedding**: move jobs off source nodes to hit a modeled power target while minimizing disruption. The canonical additive dispatch path is still available, but the current exploration asks whether source-node concentration can exploit the ramp-then-plateau node power curve.

The important modeled quantity is:

```text
node_expected_w = sum_i P(L_i) - P(L_i - removed_load_i)
```

This is modeled expected source-node power shed, not a hard grid guarantee. The conservative active-work floor is still reported separately as `active_floor_w`.

## Current Methods

The active plots now compare five methods:

- `additive LP`: old per-job additive LP, evaluated afterward under the node curve.
- `active-knee LP relaxation`: fractional node-aware LP over exhaustively enumerated small fixed active-knee regions, using the exact affine node-power expression inside each region.
- `active-knee MILP`: the same fixed-region active-knee model with whole-job movement variables; it hard fails above the exhaustive source-node cap.
- `live greedy`: whole-job greedy with live node marginal values.
- `random jobs`: whole-job random job order, budget-respecting.

`random nodes`, `node-drain greedy`, `tangent LP`, and the tiny exact oracle remain useful debugging tools in `node_knee.py`, but they are not part of the current plotted comparison.

## Main Result

The old additive LP often satisfies the active-work target while missing the node-expected target because it spreads removals across nodes. The node-aware methods improve by concentrating removals enough to cross node knees.

On the fixed 4-node, `D=300s` target sweep, active-knee LP relaxation removes the previous high-target overshoot cliff. For the agentic workload:

```text
target   achieved   active-knee cost   cost / achieved kW
10.3 kW  10.3 kW       214 s              20.8 s/kW
11.3 kW  11.3 kW       235 s              20.7 s/kW
15.4 kW  15.4 kW       400 s              25.9 s/kW
19.6 kW  19.6 kW       624 s              31.9 s/kW
```

Before fixed-region active-knee candidates, the high-target case jumped to nearly full drain (`20.6 kW`) at much higher cost (`1313 s`). The current formulation tracks the requested modeled shed instead.

## Stage 1a Calibration

The gpt-oss-20b A100 TP1 measurement pass now recomputes:

```text
ell = f/F + g/G
```

from raw prefill/decode window rates, rather than trusting a saved upstream `ell` vector. On the current 5s windows, the observed axis reaches `ell_max = 2.03`; this is above the simulator's colocated `rho*` placement ceiling, so the plot should be read as an offered-load probe axis, not a feasible per-node admission state.

The separate saturating power fit is the paper-interesting shape: power rises concavely and then flattens. Window sensitivity supports the shape but not a fixed absolute `ell` calibration:

| window | saturating R2 | linear R2 | ell knee | rho* | ell max |
|---:|---:|---:|---:|---:|---:|
| `1s` | `0.975` | `0.792` | `1.17` | `0.44` | `2.44` |
| `2s` | `0.984` | `0.812` | `1.50` | `0.52` | `2.22` |
| `5s` | `0.990` | `0.846` | `2.05` | `0.53` | `2.03` |
| `10s` | `0.993` | `0.874` | `2.53` | `0.62` | `2.01` |
| `30s` | `0.997` | `0.903` | `3.12` | `0.59` | `1.90` |

This makes `ell` useful as a colocated service-time load hypothesis, but not yet a universal x-axis. For disaggregated inference, the safer state is a vector such as `(u_pre, u_dec, u_ing, m)` unless the measured service surface collapses cleanly to one scalar.

The MVP service-surface runbook is set up but not yet executed on the A100 node. It will measure isolated prefill `rho(T)`, context-dependent decode `G(T)`, and the mixed prefill/decode interaction surface needed to decide whether `G` is stable and whether scalar `ell` remains defensible.

## Reproducible Plots

Run each script from `queue-haul/`.

| script | output | current read |
|---|---|---|
| `plot_node_knee_target_sweep.py` | `outputs/node_knee_target_sweep.{csv,pdf,png}` | Fixed 4-node, `D=300s`, target sweep from `5%` to `95%` of full modeled removable node power. Both active-knee methods hit every target in all four active cached workload classes. |
| `plot_node_knee_deadline_sweep.py` | `outputs/node_knee_deadline_sweep.{csv,pdf,png}` | Agentic fixed target at `9.213 kW` (`45%` of full node-expected removable power). Active-knee LP relaxation, active-knee MILP, live greedy, and random jobs first hit at `45s`; additive LP never hits the node target. |
| `plot_node_knee_execution_validation.py` | `outputs/node_knee_execution_validation.{csv,pdf,png}`, `outputs/node_knee_fixed_plan_replay.{csv,pdf,png}` | Replays active-knee plans through the deterministic reconstruction simulator against a `45%` full-node-model target. In the re-solved operational sweep, both active-knee variants reach selected/egress/rebuild target at `45s`. Fixed-plan replay solves once at `D=300s` and reaches egress/rebuild target at `184.1s` for MILP and `160.9s` for LP relaxation. |
| `plot_node_knee_agentic_des_sweep.py` | `outputs/node_knee_agentic_des_sweep.{csv,pdf,png}` | Agentic requested-shed sweep at `D=300s`: x-axis is requested node-expected shed and y-axis is disruption per requested kW for solver-selected, DES egress-realized, and DES rebuild-realized outcomes under node-marginal-PD replay. |
| `plot_node_knee_scale_workload_sweep.py` | `outputs/node_knee_scale_workload_sweep.{csv,pdf,png}` | Sweeps `1/2/4` source nodes, all four active cached classes, deadlines `10/30/120s`, and target fractions `25/45/65%`. Hit counts are additive LP `54/108`, active-knee LP `97/108`, active-knee MILP `97/108`, live greedy `96/108`, random jobs `87/108`. |
| `plot_node_knee_kappa_sweep.py` | `outputs/node_knee_kappa_sweep.{csv,pdf,png}` | Planner-side rebuild-cushion sensitivity. The CSV separates selected shortfall from deadline misses; for example `D=30s` has no deadline misses but still misses target by planner shortfall. |

Only these six node-knee plot scripts are canonical for the current dispatch result. Older validation and exploration plot scripts were removed from the active tree; their underlying model code remains covered by semantic tests.

Stage 1a calibration artifacts:

| script | output | current read |
|---|---|---|
| `stage1_profile.py` | `outputs/stage1_gpt_oss_20b_a100_tp1.{pdf,png}`, `outputs/stage1_gpt_oss_20b_a100_tp1_curve.csv`, `outputs/stage1_gpt_oss_20b_a100_tp1_constants.csv`, `outputs/stage1_gpt_oss_20b_a100_tp1_power_curve.csv` | Rebuilds `ell` from raw `f/g` rates and overlays the concave/saturating power curve. |
| `stage1_window_sensitivity.py` | `outputs/stage1_gpt_oss_20b_a100_tp1_window_sensitivity.{pdf,png}`, `outputs/stage1_gpt_oss_20b_a100_tp1_window_sensitivity_{summary,binned}.csv` | Checks that the concave power shape survives `1/2/5/10/30s` windows while absolute `ell` constants drift. |
| `stage1_service_surface.py` | `runs/stage1_service_surface/<run-id>/commands.sh` | Generates the single-node A100 probe runbook for `rho(T)`, `G(T)`, and mixed prefill/decode interaction. |
| `stage1_service_reduce.py` | `outputs/stage1_gpt_oss_20b_a100_tp1_{prefill_rho,decode_context,mixed_surface}.{csv,pdf,png}` | Reduces completed service-surface probe bundles; outputs appear after the hardware run. |

Latest median disruption intensities from the fixed 4-node target sweep:

| workload | additive LP | active-knee LP relaxation | active-knee MILP | live greedy | random jobs |
|---|---:|---:|---:|---:|---:|
| ordinary chat | `21.9` | `7.7` | `7.7` | `11.5` | `39.4` |
| long chat/code | `14.3` | `6.4` | `6.4` | `12.4` | `42.3` |
| reasoning chat | `15.3` | `10.0` | `10.0` | `15.4` | `44.8` |
| agentic tool loop | `87.5` | `20.8` | `21.8` | `63.6` | `119.0` |

Units are seconds of disruption per kW of modeled node-expected shed. Misses are excluded from median intensity.

## Validation

### Profile-driven simulator revision (2026-07-15)

The new path has versioned model/workload profiles, whole-session planning,
balanced placement, named shared links, deferred replay on observed requests,
commit-gated source power, explicit incomplete rows, and raw event, session,
network, power, and plan tables. It plans centrally once and replays that plan
under faster and slower cases.

Local 10,000-session agentic smoke on the M1 MacBook Pro:

| stage | time | peak process memory |
|---|---:|---:|
| build scenario | `0.066 s` | |
| load-only plan | `1.76 s` | |
| execute 75,850 events | `3.08 s` | `166 MB` |
| rounded LP | `1.72 s` | `271 MB` |

These are local smoke measurements, not paper performance claims.

Current semantic tests cover:

- explicit `source_node` requirement for node-knee evaluation,
- convex removed-load value under the ramp-then-plateau curve,
- active-knee concentration cases,
- exhaustive small active-knee region search and hard failure above the cap,
- active-knee MILP integrality and LP lower-bound relation,
- fixed-region active-knee constraints,
- true conservative active-floor accounting,
- zero-clamped stage windows before startup ramps,
- movement parameter domain validation,
- pinned-job handling in node-knee whole-job methods,
- actual movement-feasibility reporting,
- seeded/budget-respecting random baselines,
- fixed 4-node target-sweep semantics,
- deadline-sweep modeled target semantics,
- execution replay selected/egress/rebuild node-power semantics,
- agentic requested-shed DES disruption semantics,
- kappa sweep failure-cause semantics,
- scale/workload/deadline sweep semantics,
- Stage 1a `ell` recomputation and concave power-knee semantics,
- Stage 1a window-sensitivity semantics,
- Stage 1a service-surface runbook and reducer semantics.

Run:

```text
uv run pytest
```

Current queue-haul verification: `200 passed`.
