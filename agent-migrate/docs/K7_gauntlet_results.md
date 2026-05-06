# K7 — gauntlet results after critic fixes

**Date:** 2026-05-05  
**Corrected rerun:** 2026-05-05  
**Fixture parameters:** N=100 workflows; compact_kv model; K4 fluid simulator; K5 policies plan against the same `ResourceBudget` that K4 simulates.

**Verdict:** **GAUNTLET PASSES AFTER FIXES.** The original Week 1 K7 result was too pessimistic because `mixed_min_pressure` planned against `ProfileBundle` capacities while K4 simulated custom `ResourceBudget`s. The rerun also fixes workspace hydrate units, concurrent shared-state materialization, and T3 fixture coverage.

## What changed

The critic pass found four load-bearing issues:

1. K5 policies did not receive the K7 `ResourceBudget`; load-aware planning used stale YAML profile capacities.
2. K4 allowed many workflows to concurrently materialize the same cold `(state_id, dst_site)` and pay duplicate work before warmness was updated.
3. `workspace_hydrate_bps` was documented as bytes/s but K3/K4 treated workspace bytes as bits.
4. T3 was documented as single-source/multi-resource with finite KV, but the committed fixture was balanced distributed-origin with KV memory uncapped.

The corrected code now:

- passes `ResourceBudget` into K5 policies;
- coalesces concurrent in-flight materialization of the same `(state_id, dst_site)`;
- treats workspace hydrate capacity as bytes/s;
- makes T3 a single-source evacuation from phoenix to seattle/austin with finite destination KV memory.

## Summary of outcomes

| Test | Pass criterion | Corrected result | Verdict |
| ---- | -------------- | ---------------- | ------- |
| T1 (capacity-free collapse) | mixed ≈ cache_reuse within 1e-6 under math.inf capacity | all policies p50 = 0.0 | **PASS** |
| T2 (prefill stampede) | replay_all > kv_all and mixed < replay_all − 10% | replay 26.5s; mixed 0.0s | **PASS** |
| T3 (multi-resource bottleneck) | mixed < best fixed-mode − 10% | best fixed: workspace_sticky 48.2s; mixed 24.3s | **PASS** |

Per-policy p50 time-to-resume, seconds:

| Policy | T1 | T2 | T3 |
| ------ | -: | -: | -: |
| min_cost_independent | 0.0 | 0.0 | 67.73 |
| replay_all | 0.0 | 26.52 | 67.73 |
| kv_all | 0.0 | 0.0 | 177.93 |
| cache_reuse | 0.0 | 0.0 | 67.73 |
| workspace_sticky | 0.0 | 0.0 | 48.16 |
| mixed_min_pressure | 0.0 | 0.0 | **24.35** |
| random_mode | 0.0 | 0.0 | 50.47 |

For T3, `mixed_min_pressure` beats the best fixed-mode policy by about **49%** on p50 time-to-resume and beats `random_mode` by about **52%**.

## Interpretation

The corrected K7 result changes the gate decision:

> The mobility-episode substrate is worth carrying forward, but the correct next artifact is still a regime map, not a universal policy-win claim.

T3 now demonstrates that richer planning can matter in a single-source evacuation with finite network, prefill, workspace, and KV resources. T2 is weaker as a planner claim after budget-aware planning: several policies can choose/free-ride on KV when network is unconstrained, so T2 should be read as a prefill-stampede sanity check, not evidence that `mixed_min_pressure` is uniquely good.

The next work remains K8/R1 + O1:

- sweep N × state scale × prefill capacity × link bandwidth;
- include `random_mode` and strong per-site reuse in every cell;
- add the small-N oracle before tuning `mixed_min_pressure`.

## Verification

Focused rerun:

```text
uv run pytest tests/test_reconstitution_budget.py tests/test_resources.py tests/test_fluid_sim.py tests/test_k7_gauntlet.py
38 passed
```

Full Vagrant suite:

```text
uv run pytest
410 passed, 1 skipped
```

