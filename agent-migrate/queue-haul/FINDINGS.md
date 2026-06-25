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
- `active-knee LP relaxation`: fractional node-aware LP over fixed active-knee regions, using the exact affine node-power expression inside each region.
- `active-knee MILP`: the same fixed-region active-knee model with whole-job movement variables.
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

## Reproducible Plots

Run each script from `queue-haul/`.

| script | output | current read |
|---|---|---|
| `plot_node_knee_target_sweep.py` | `outputs/node_knee_target_sweep.{csv,pdf,png}` | Fixed 4-node, `D=300s`, target sweep from `5%` to `95%` of full modeled removable node power. Both active-knee methods hit every target in all four active cached workload classes. |
| `plot_node_knee_deadline_sweep.py` | `outputs/node_knee_deadline_sweep.{pdf,png}` | Agentic fixed target. Active-knee LP relaxation and MILP first hit at `10s` with `24.5` and `24.0 s/kW`; live greedy at `12s` with `35.1 s/kW`; random jobs at `16s` with `59.2 s/kW`; additive LP never hits the node target. |
| `plot_node_knee_scale_workload_sweep.py` | `outputs/node_knee_scale_workload_sweep.{csv,pdf,png}` | Sweeps `1/2/4` source nodes, all four active cached classes, deadlines `10/30/120s`, and target fractions `25/45/65%`. Active-knee MILP exposes the whole-job gap; additive LP misses every agentic node-expected target. |

Only these three plot scripts are canonical. Older validation and exploration plot scripts were removed from the active tree; their underlying model code remains covered by semantic tests.

Latest median disruption intensities from the fixed 4-node target sweep:

| workload | additive LP | active-knee LP relaxation | active-knee MILP | live greedy | random jobs |
|---|---:|---:|---:|---:|---:|
| ordinary chat | `21.9` | `7.7` | `7.7` | `11.5` | `39.4` |
| long chat/code | `14.3` | `6.4` | `6.4` | `12.4` | `42.3` |
| reasoning chat | `15.3` | `10.0` | `10.0` | `15.4` | `44.8` |
| agentic tool loop | `87.5` | `20.8` | `21.8` | `63.6` | `119.0` |

Units are seconds of disruption per kW of modeled node-expected shed. Misses are excluded from median intensity.

## Validation

Current semantic tests cover:

- explicit `source_node` requirement for node-knee evaluation,
- convex removed-load value under the ramp-then-plateau curve,
- active-knee concentration cases,
- active-knee MILP integrality and LP lower-bound relation,
- fixed-region active-knee constraints,
- pinned-job handling in node-knee whole-job methods,
- actual movement-feasibility reporting,
- seeded/budget-respecting random baselines,
- fixed 4-node target-sweep semantics,
- deadline-sweep modeled target semantics,
- scale/workload/deadline sweep semantics.

Run:

```text
uv run pytest
```

Current verification after the active-knee MILP cleanup: `159 passed`, with the existing 5 CVXPY accuracy warnings in evacuation tests.
