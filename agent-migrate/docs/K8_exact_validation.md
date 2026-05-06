# V1 — exact validation of K8 regime cells

**Artifacts:** `runs/k8_validation/`  
**Runner:** `uv run python scripts/run_k8_validation.py`  
**Date:** 2026-05-06

## Purpose

K8 heatmaps are aggregate-estimated and should be read as candidate regime discovery. V1 reruns selected claim cells through exact K4 and records best-policy agreement, dominant-bottleneck agreement, and p50/p95 timing error.

## Summary

| Target | Cell | Exact best | Aggregate best | Exact bottleneck | Aggregate bottleneck | Exact best bottleneck | Median p50 err | Median p95 err | Trust |
| ------ | ---- | ---------- | -------------- | ---------------- | -------------------- | --------------------- | --------------: | --------------: | ----- |
| large_artifact_fast_link | `n10_large_artifact_loose_100g` | mixed_min_pressure | mixed_min_pressure | prefill | workspace | prefill | 50.0% | 5.0% | needs_exact_k4 |
| large_artifact_slow_link | `n10_large_artifact_loose_1g` | mixed_min_pressure | mixed_min_pressure | prefill | network | prefill | 50.0% | 5.0% | needs_exact_k4 |
| medium_multi_resource | `n100_medium_tight_5g` | mixed_min_pressure | mixed_min_pressure | prefill | network | prefill | 34.6% | 13.8% | needs_exact_k4 |
| monorepo_workspace_pressure | `n100_monorepo_loose_100g` | mixed_min_pressure | mixed_min_pressure | prefill | workspace | prefill | 50.0% | 5.1% | needs_exact_k4 |
| swe_bench_reuse_scale | `n100_swe_bench_moderate_25g` | random_diversification | mixed_min_pressure | prefill | prefill | network | 18.7% | 0.9% | needs_exact_k4 |
| tiny_prefill_pressure | `n100_tiny_tight_100g` | random_diversification | mixed_min_pressure | network | network | workspace | 20.3% | 0.9% | needs_exact_k4 |
| tiny_slow_link | `n100_tiny_loose_1g` | mixed_min_pressure | mixed_min_pressure | prefill | network | prefill | 40.3% | 7.8% | needs_exact_k4 |

## Reading

Only cells labeled `timing_reliable` should be used for aggregate timing claims. `label_reliable` cells can support qualitative regime labels but need exact K4 for wall-clock numbers. `policy_boundary` means the heatmap winner is unstable near a small exact margin. `needs_exact_k4` means aggregate K8 is useful only as a search hint.

Two caveats matter. First, aggregate p50/p95 are service-time approximations (`0.50 * makespan` and `0.95 * makespan`), not completion-CDF estimates. Second, K8's cell bottleneck heatmap uses the existing cell-level summary convention, while exact K4 also reports the bottleneck of the exact winning policy. Those definitions should not be mixed in paper claims.
