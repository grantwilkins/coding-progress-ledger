# K8 regime-map artifacts

`regime_policy_metrics.csv` contains per-policy metrics. `regime_cell_summary.csv` / `.json` contain the best-policy and dominant-bottleneck map. The emitted full sweep uses K8's aggregate service-time estimator so 1K/10K workflow cells are tractable; `run_k8_cell(... )` remains the exact K4 simulator path for focused validation cells. `exact_vs_aggregate.csv`, when present, compares sampled exact K4 cells to the aggregate estimator and is the source of truth for how much confidence to put in aggregate heatmap labels.
