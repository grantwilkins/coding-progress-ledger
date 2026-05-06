# K8 exact-validation artifacts

`claim_cell_policy_validation.csv` compares exact K4 and aggregate K8 estimates for each selected claim cell and policy. `claim_cell_summary.csv` / `.json` collapse that into best-policy agreement, bottleneck agreement, p50/p95 timing error, and a trust label.

Trust labels:
- `timing_reliable`: aggregate best policy and bottleneck agree, with median p50 and p95 relative error <= 25%.
- `label_reliable`: labels agree, but timing should not be quoted.
- `policy_boundary`: aggregate/exact best differs, but the exact winner margin is <= 5%.
- `needs_exact_k4`: use exact K4 before making claims.

Validated cells: 7.
