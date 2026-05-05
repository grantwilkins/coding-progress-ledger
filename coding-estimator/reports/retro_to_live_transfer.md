# Retrospective → live transfer (L3)

_Generated 2026-05-05T02:55:11+00:00._

Train: `swe_agent_pilot ∪ hermes_pilot_h5_v2`. Test: `tb_live`. Feature-group ablation drops one of {closure, frontier, instability, discovery} at a time and retrains. Annotation-leakage caveat from § C1 applies to retrospective fits.

Train: `swe_agent_pilot ∪ hermes_pilot_h5_v2`. Test: `tb_live`. ECE uses 3 equal-width bins (10-bin ECE is unestimable at N=12).

## Per-target metrics

| target | config | ablated | n_train | n_test | n_ckpts | pos | AUROC | Brier | Brier 95% CI | ECE_3bin |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| y_future_progress_drop_h5 | g2_time_only | - | 20 | 10 | 23 | 0.130 | 0.917 | 0.112 | [0.011, 0.211] | 0.023 |
| y_future_progress_drop_h5 | g4_full | - | 20 | 10 | 23 | 0.130 | 1.000 | 0.002 | [0.000, 0.004] | 0.020 |
| y_future_progress_drop_h5 | g4_minus_closure | closure | 20 | 10 | 23 | 0.130 | 1.000 | 0.100 | [0.001, 0.192] | 0.084 |
| y_future_progress_drop_h5 | g4_minus_discovery | discovery | 20 | 10 | 23 | 0.130 | 1.000 | 0.001 | [0.000, 0.002] | 0.019 |
| y_future_progress_drop_h5 | g4_minus_frontier | frontier | 20 | 10 | 23 | 0.130 | 1.000 | 0.002 | [0.000, 0.003] | 0.021 |
| y_future_progress_drop_h5 | g4_minus_instability | instability | 20 | 10 | 23 | 0.130 | 1.000 | 0.002 | [0.000, 0.003] | 0.026 |
| y_submit_without_validation | g2_time_only | - | 20 | 12 | 83 | 0.000 | n/a | 0.037 | [0.034, 0.039] | 0.188 |
| y_submit_without_validation | g4_full | - | 20 | 12 | 83 | 0.000 | n/a | 0.209 | [0.102, 0.338] | 0.351 |
| y_submit_without_validation | g4_minus_closure | closure | 20 | 12 | 83 | 0.000 | n/a | 0.299 | [0.171, 0.434] | 0.439 |
| y_submit_without_validation | g4_minus_discovery | discovery | 20 | 12 | 83 | 0.000 | n/a | 0.197 | [0.094, 0.322] | 0.341 |
| y_submit_without_validation | g4_minus_frontier | frontier | 20 | 12 | 83 | 0.000 | n/a | 0.182 | [0.100, 0.277] | 0.337 |
| y_submit_without_validation | g4_minus_instability | instability | 20 | 12 | 83 | 0.000 | n/a | 0.100 | [0.049, 0.155] | 0.225 |
| y_success_eventual | g2_time_only | - | 20 | 12 | 83 | 1.000 | n/a | 0.198 | [0.196, 0.201] | 0.445 |
| y_success_eventual | g4_full | - | 20 | 12 | 83 | 1.000 | n/a | 0.145 | [0.123, 0.168] | 0.337 |
| y_success_eventual | g4_minus_closure | closure | 20 | 12 | 83 | 1.000 | n/a | 0.180 | [0.159, 0.199] | 0.414 |
| y_success_eventual | g4_minus_discovery | discovery | 20 | 12 | 83 | 1.000 | n/a | 0.175 | [0.151, 0.201] | 0.373 |
| y_success_eventual | g4_minus_frontier | frontier | 20 | 12 | 83 | 1.000 | n/a | 0.136 | [0.124, 0.148] | 0.327 |
| y_success_eventual | g4_minus_instability | instability | 20 | 12 | 83 | 1.000 | n/a | 0.169 | [0.116, 0.230] | 0.364 |
| y_validation_new_work_h5 | g2_time_only | - | 20 | 10 | 23 | 0.478 | 0.731 | 0.475 | [0.207, 0.704] | 0.475 |
| y_validation_new_work_h5 | g4_full | - | 20 | 10 | 23 | 0.478 | 0.735 | 0.422 | [0.182, 0.642] | 0.442 |
| y_validation_new_work_h5 | g4_minus_closure | closure | 20 | 10 | 23 | 0.478 | 0.697 | 0.396 | [0.164, 0.625] | 0.416 |
| y_validation_new_work_h5 | g4_minus_discovery | discovery | 20 | 10 | 23 | 0.478 | 0.644 | 0.462 | [0.203, 0.689] | 0.466 |
| y_validation_new_work_h5 | g4_minus_frontier | frontier | 20 | 10 | 23 | 0.478 | 0.735 | 0.422 | [0.182, 0.642] | 0.442 |
| y_validation_new_work_h5 | g4_minus_instability | instability | 20 | 10 | 23 | 0.478 | 0.735 | 0.413 | [0.176, 0.632] | 0.433 |

## Ablation Δ (Brier vs `g4_full`)

Positive Δ ⇒ removing the group **hurt** transfer ⇒ that group carried signal.

| target | full Brier | minus_closure | minus_frontier | minus_instability | minus_discovery |
|---|---:|---:|---:|---:|---:|
| y_future_progress_drop_h5 | 0.002 | +0.098 | -0.001 | -0.000 | -0.001 |
| y_submit_without_validation | 0.209 | +0.089 | -0.027 | -0.110 | -0.012 |
| y_success_eventual | 0.145 | +0.035 | -0.009 | +0.024 | +0.031 |
| y_validation_new_work_h5 | 0.422 | -0.026 | -0.000 | -0.009 | +0.041 |
