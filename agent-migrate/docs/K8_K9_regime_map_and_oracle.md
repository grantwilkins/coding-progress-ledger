# K8/K9 — regime map and small-N oracle

**Date:** 2026-05-06  
**Artifacts:** `runs/k8_regime_map/`, `runs/k9_oracle/`  
**Runner:** `uv run python scripts/run_k8_k9.py`

## What landed

K8 now has a reusable regime-map harness in `src/agent_migrate_agent/k8_regime.py`.
It sweeps:

```text
N workflows:              10, 100, 1000, 10000
workspace/artifact scale: tiny, swe_bench, medium, monorepo, large_artifact
prefill capacity:         loose, moderate, tight
link bandwidth:           1, 5, 25, 100 Gbps
```

The fixed policy set is:

```text
strong_reuse              (`cache_reuse`)
replay_all
kv_all
workspace_sticky
random_diversification    (`random_mode`)
mixed_min_pressure
```

K9 now has a restricted exact simulator-backed oracle in
`src/agent_migrate_agent/k9_oracle.py`. It is intentionally exponential and
currently enumerates workflow-level destination, prompt-mode, and workspace-mode
choices for small episodes. It does not search per-state destination choices or
action ordering.

## K8 first-pass result

The emitted K8 sweep has:

```text
240 regime cells
1440 policy rows
24 heatmaps: best policy + dominant bottleneck for each prefill/link panel
```

The full emitted sweep uses K8's aggregate service-time estimator, not exact K4
event simulation, so that 1K/10K workflow cells are tractable. Exact K4 is
still available through `run_k8_cell(...)` and is used by the focused tests and
the calibration artifact below.

Summary over 240 cells:

| Category | Count |
| -------- | ----: |
| `mixed_min_pressure` best | 233 |
| `random_diversification` best | 7 |
| `network` dominant bottleneck | 119 |
| `workspace` dominant bottleneck | 72 |
| `prefill` dominant bottleneck | 49 |

By state scale:

| State scale | Best-policy result | Dominant bottlenecks |
| ----------- | ------------------ | -------------------- |
| tiny | mixed 48/48 | prefill 25, network 23 |
| swe_bench | mixed 48/48 | prefill 24, network 24 |
| medium | mixed 41/48, random 7/48 | workspace 24, network 24 |
| monorepo | mixed 48/48 | workspace 24, network 24 |
| large_artifact | mixed 48/48 | workspace 24, network 24 |

The map is directionally consistent with the project hypothesis:

- small-state cells are mostly prefill/network regimes;
- medium and larger state cells move into workspace/network regimes;
- richer planning matters most when it can split destination pressure or choose
  hydrate vs copy under finite resources.

Important caveat: because the full map is aggregate-estimated, it should be
read as the first regime map, not as final proof of exact episode timing at
10K workflows.

## K8 exact-vs-aggregate calibration

`runs/k8_regime_map/exact_vs_aggregate.csv` compares exact K4 and aggregate
estimates on sampled cells:

```text
N workflows:              10, 100
workspace/artifact scale: tiny, medium, monorepo
prefill capacity:         loose, tight
link bandwidth:           1, 25, 100 Gbps
```

Calibration summary:

| Metric | Result |
| ------ | -----: |
| sampled cells | 36 |
| policy rows | 216 |
| best-policy agreement | 24 / 36 cells |
| bottleneck-label agreement | 102 / 216 policy rows |
| median relative p50 error | 48.9% |
| max relative p50 error | 789.2% |

Interpretation:

> The aggregate map is useful for broad regime exploration, but it is not a
> calibrated substitute for exact K4 timing or bottleneck labels near policy
> boundaries.

This weakens any “mixed wins 233/240 cells” reading. The safer reading is that
the aggregate map identifies candidate regimes and pressure axes that need exact
K4 or workload-anchor validation.

## V1 exact claim-cell validation

`runs/k8_validation/` and `docs/K8_exact_validation.md` rerun seven named
claim cells through exact K4:

```text
swe_bench_reuse_scale
tiny_prefill_pressure
tiny_slow_link
medium_multi_resource
monorepo_workspace_pressure
large_artifact_slow_link
large_artifact_fast_link
```

All seven currently receive `needs_exact_k4`. Aggregate and exact often agree
on the best policy, but p50 timing error is large and bottleneck labels are not
stable enough for claims. This makes the K8 heatmaps useful for choosing cells
to inspect, not for quoting timing or bottleneck conclusions without exact K4.

## K9 oracle result

K9 was broadened to four 4-workflow, two-destination, single-source evacuation
diagnostic cells. Each row enumerates 4096 plans in candidate-space v1:
workflow-level destination, prompt mode, and workspace mode.

| Scenario | Oracle p50 | Strong reuse p50 | Mixed p50 | Oracle gap vs strong | Oracle gap vs mixed |
| -------- | ---------: | ---------------: | --------: | -------------------: | ------------------: |
| tiny_prefill_pressure | 0.061 s | 0.140 s | 0.094 s | 56.2% | 35.0% |
| medium_multi_resource | 0.735 s | 3.796 s | 1.469 s | 80.6% | 50.0% |
| monorepo_workspace_pressure | 9.975 s | 20.079 s | 10.044 s | 50.3% | 0.7% |
| slow_link_network_pressure | 0.533 s | 16.053 s | 1.062 s | 96.7% | 49.9% |

Interpretation:

> In these restricted small diagnostic cells, the ceiling above strong reuse is
> real. `mixed_min_pressure` leaves a meaningful oracle gap in the prefill and
> multi-resource cells, but is nearly oracle-level in the monorepo/workspace and
> slow-link/network cells.

That supports continuing with regime-map work and gives a concrete reason to
inspect planner gaps later. It does not yet justify tuning the heuristic before
adding workload anchors and more exact K4 validation.

## Verification

Focused semantic tests:

```text
uv run pytest tests/test_k8_regime.py tests/test_k9_oracle.py
8 passed
```

Artifact generation:

```text
uv run python scripts/run_k8_k9.py
```
