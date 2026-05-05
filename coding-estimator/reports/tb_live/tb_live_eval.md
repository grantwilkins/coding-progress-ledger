# TB-live only — checkpoint evaluation (K1)

_Generated 2026-05-05T03:06:54+00:00._

K1 — Logistic G4 (ledger_basic) and G2 (time_only) on `tb_live` alone under LORO. Run-level bootstrap CIs (B=1000).

Per-target metrics on `tb_live` under LORO. ECE uses 3 equal-width bins (10-bin ECE is unestimable at N=12).

| target | model | n_runs | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | ECE_3bin | note |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| y_future_progress_drop_h5 | ledger_basic | 10 | 23 | 0.130 | 0.917 | 0.096 | [0.003, 0.157] | 0.080 |  |
| y_future_progress_drop_h5 | time_only | 10 | 23 | 0.130 | 0.867 | 0.133 | [0.004, 0.208] | 0.117 |  |
| y_submit_without_validation | ledger_basic | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | single-class y |
| y_submit_without_validation | time_only | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | single-class y |
| y_success_eventual | ledger_basic | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | single-class y |
| y_success_eventual | time_only | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | single-class y |
| y_validation_new_work_h5 | ledger_basic | 10 | 23 | 0.478 | 0.492 | 0.284 | [0.196, 0.386] | 0.090 |  |
| y_validation_new_work_h5 | time_only | 10 | 23 | 0.478 | 0.432 | 0.254 | [0.155, 0.387] | 0.354 |  |

## G4 vs G2 (Brier)

| target | G2 Brier | G4 Brier | Δ (G4 - G2) | G4 wins-or-ties |
|---|---:|---:|---:|:---:|
| y_future_progress_drop_h5 | 0.133 | 0.096 | -0.037 | yes |
| y_submit_without_validation | 0.000 | 0.000 | +0.000 | yes |
| y_success_eventual | 0.000 | 0.000 | +0.000 | yes |
| y_validation_new_work_h5 | 0.254 | 0.284 | +0.030 | no |
