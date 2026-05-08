# Baseline ladder — calibration

Expected Calibration Error (ECE, 10-bin equal-width) and predicted vs.
observed positive rates per cell. ECE > 0.10 indicates the model is
miscalibrated enough that downstream consumers should not trust raw
probabilities without per-source recalibration (Workstream J).

## Scheme: `loro`

| source_slice | target | model | pos_rate (data) | pos_rate (predicted) | ECE |
|---|---|---|---:|---:|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | 0.170 | 0.169 | 0.001 |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | 0.170 | 0.176 | 0.032 |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | 0.170 | 0.179 | 0.029 |
| swe_agent_pilot | y_submit_without_validation | constant | 0.088 | 0.091 | 0.003 |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | 0.088 | 0.091 | 0.078 |
| swe_agent_pilot | y_submit_without_validation | time_only | 0.088 | 0.089 | 0.053 |
| swe_agent_pilot | y_success_eventual | constant | 0.467 | 0.471 | 0.247 |
| swe_agent_pilot | y_success_eventual | ledger_basic | 0.467 | 0.459 | 0.196 |
| swe_agent_pilot | y_success_eventual | time_only | 0.467 | 0.479 | 0.188 |
| swe_agent_pilot | y_timeout | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | 0.478 | 0.470 | 0.420 |
| tb_live | y_validation_new_work_h5 | ledger_basic | 0.478 | 0.436 | 0.406 |
| tb_live | y_validation_new_work_h5 | time_only | 0.478 | 0.479 | 0.379 |

## Scheme: `loso`

| source_slice | target | model | pos_rate (data) | pos_rate (predicted) | ECE |
|---|---|---|---:|---:|---:|
| loso->tb_live | y_future_progress_drop_h5 | constant | 0.130 | 0.170 | 0.040 |
| loso->tb_live | y_future_progress_drop_h5 | ledger_basic | 0.130 | 0.123 | 0.020 |
| loso->tb_live | y_future_progress_drop_h5 | time_only | 0.130 | 0.107 | 0.023 |
| loso->tb_live | y_submit_without_validation | constant | 0.000 | 0.088 | 0.088 |
| loso->tb_live | y_submit_without_validation | ledger_basic | 0.000 | 0.351 | 0.351 |
| loso->tb_live | y_submit_without_validation | time_only | 0.000 | 0.188 | 0.188 |
| loso->tb_live | y_success_eventual | constant | 1.000 | 0.467 | 0.533 |
| loso->tb_live | y_success_eventual | ledger_basic | 1.000 | 0.663 | 0.337 |
| loso->tb_live | y_success_eventual | time_only | 1.000 | 0.555 | 0.445 |
| loso->tb_live | y_timeout | constant | 0.000 | 0.001 | 0.001 |
| loso->tb_live | y_timeout | ledger_basic | 0.000 | 0.001 | 0.001 |
| loso->tb_live | y_timeout | time_only | 0.000 | 0.001 | 0.001 |
| loso->tb_live | y_validation_new_work_h5 | constant | 0.478 | 0.008 | 0.470 |
| loso->tb_live | y_validation_new_work_h5 | ledger_basic | 0.478 | 0.037 | 0.442 |
| loso->tb_live | y_validation_new_work_h5 | time_only | 0.478 | 0.003 | 0.475 |

## Scheme: `ltfo`

| source_slice | target | model | pos_rate (data) | pos_rate (predicted) | ECE |
|---|---|---|---:|---:|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_timeout | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_timeout | time_only | n/a (insufficient data) | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | n/a (insufficient data) | n/a | n/a |
| tb_live | y_validation_new_work_h5 | ledger_basic | n/a (insufficient data) | n/a | n/a |
| tb_live | y_validation_new_work_h5 | time_only | n/a (insufficient data) | n/a | n/a |

