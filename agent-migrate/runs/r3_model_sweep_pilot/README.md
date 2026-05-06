# R3 — model-architecture axis for the K8 regime sweep

`r3_regime_by_model.csv` holds the per-cell best-policy and dominant-bottleneck label under each model profile. The `best_policy_flips` and `bottleneck_flips` columns mark cells where architecture flips the regime label, and `r3_flip_summary.json` gives the headline counts.

Each model profile is run with a model-aware budget: K8's prefill capacity (loose / moderate / tight) is rescaled by the model's per-stream prefill rate relative to compact_kv (the K8 baseline). Link bandwidth is unchanged.

Aggregate K8 estimator caveat: the same calibration that applies to `runs/k8_regime_map/` applies here. Cells where the architecture flips the label are candidates for exact K4 + V1 re-validation; do not quote timing claims off this artifact alone.
