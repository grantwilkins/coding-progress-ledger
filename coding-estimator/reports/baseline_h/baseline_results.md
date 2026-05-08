# Baseline ladder — metrics

Per-cell results from `scripts/run_baselines.py`. Bootstrap CIs are
computed by resampling **test runs** with replacement (B=1000, seed=0).
Cells flagged not-feasible by the data-budget gate are emitted as
`n/a (insufficient data)` and never silently zeroed.

## Scheme: `loro`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | 20.000 | 20.000 | 499.000 | 0.170 | 0.323 | 0.143 | [0.112, 0.171] | 0.461 |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | 20.000 | 20.000 | 499.000 | 0.170 | 0.977 | 0.039 | [0.019, 0.062] | 0.138 |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | 20.000 | 20.000 | 499.000 | 0.170 | 0.626 | 0.142 | [0.110, 0.172] | 0.452 |
| swe_agent_pilot | y_submit_without_validation | constant | 20.000 | 20.000 | 599.000 | 0.088 | 0.000 | 0.086 | [0.009, 0.193] | 0.338 |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | 20.000 | 20.000 | 599.000 | 0.088 | 0.676 | 0.080 | [0.017, 0.167] | 0.270 |
| swe_agent_pilot | y_submit_without_validation | time_only | 20.000 | 20.000 | 599.000 | 0.088 | 0.573 | 0.085 | [0.013, 0.185] | 0.308 |
| swe_agent_pilot | y_success_eventual | constant | 20.000 | 20.000 | 599.000 | 0.467 | 0.000 | 0.280 | [0.265, 0.295] | 0.754 |
| swe_agent_pilot | y_success_eventual | ledger_basic | 20.000 | 20.000 | 599.000 | 0.467 | 0.410 | 0.291 | [0.243, 0.340] | 0.802 |
| swe_agent_pilot | y_success_eventual | time_only | 20.000 | 20.000 | 599.000 | 0.467 | 0.281 | 0.283 | [0.263, 0.305] | 0.761 |
| swe_agent_pilot | y_timeout | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | 10.000 | 10.000 | 23.000 | 0.478 | 0.110 | 0.287 | [0.273, 0.302] | 0.769 |
| tb_live | y_validation_new_work_h5 | ledger_basic | 10.000 | 10.000 | 23.000 | 0.478 | 0.492 | 0.284 | [0.196, 0.386] | 0.777 |
| tb_live | y_validation_new_work_h5 | time_only | 10.000 | 10.000 | 23.000 | 0.478 | 0.432 | 0.254 | [0.155, 0.387] | 0.719 |

## Scheme: `loso`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| loso->tb_live | y_future_progress_drop_h5 | constant | 20.000 | 10.000 | 23.000 | 0.130 | 0.500 | 0.115 | [0.029, 0.199] | 0.393 |
| loso->tb_live | y_future_progress_drop_h5 | ledger_basic | 20.000 | 10.000 | 23.000 | 0.130 | 1.000 | 0.002 | [0.000, 0.004] | 0.022 |
| loso->tb_live | y_future_progress_drop_h5 | time_only | 20.000 | 10.000 | 23.000 | 0.130 | 0.917 | 0.112 | [0.011, 0.211] | 0.382 |
| loso->tb_live | y_submit_without_validation | constant | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.008 | [0.008, 0.008] | 0.093 |
| loso->tb_live | y_submit_without_validation | ledger_basic | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.209 | [0.102, 0.338] | 0.734 |
| loso->tb_live | y_submit_without_validation | time_only | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.037 | [0.034, 0.039] | 0.210 |
| loso->tb_live | y_success_eventual | constant | 20.000 | 12.000 | 83.000 | 1.000 | n/a | 0.284 | [0.284, 0.284] | 0.760 |
| loso->tb_live | y_success_eventual | ledger_basic | 20.000 | 12.000 | 83.000 | 1.000 | n/a | 0.145 | [0.123, 0.168] | 0.447 |
| loso->tb_live | y_success_eventual | time_only | 20.000 | 12.000 | 83.000 | 1.000 | n/a | 0.198 | [0.196, 0.201] | 0.590 |
| loso->tb_live | y_timeout | constant | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 |
| loso->tb_live | y_timeout | ledger_basic | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 |
| loso->tb_live | y_timeout | time_only | 20.000 | 12.000 | 83.000 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 |
| loso->tb_live | y_validation_new_work_h5 | constant | 20.000 | 10.000 | 23.000 | 0.478 | 0.500 | 0.471 | [0.205, 0.697] | 2.312 |
| loso->tb_live | y_validation_new_work_h5 | ledger_basic | 20.000 | 10.000 | 23.000 | 0.478 | 0.735 | 0.422 | [0.182, 0.642] | 2.090 |
| loso->tb_live | y_validation_new_work_h5 | time_only | 20.000 | 10.000 | 23.000 | 0.478 | 0.731 | 0.475 | [0.207, 0.704] | 2.735 |

## Scheme: `ltfo`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

