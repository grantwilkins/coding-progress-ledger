
# Baseline ladder — v0 eval report
_Generated 2026-05-05T01:35:42+00:00._

v0 baselines (G1 constant, G2 time-only, G4 ledger-basic) evaluated under loro/ltfo per source and loso to tb_live. Run-level bootstrap CIs (B=1000, seed=0). Slices flagged with < 5 positives or < 5 negatives are emitted as `n/a (insufficient data)`.

## Headline metrics

| scheme | source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | 20 | 10 | 23 | 0.130 | 0.500 | 0.115 | [0.029, 0.199] | 0.393 | 0.040 |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | 20 | 10 | 23 | 0.130 | 1.000 | 0.002 | [0.000, 0.004] | 0.022 | 0.020 |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | 20 | 10 | 23 | 0.130 | 0.917 | 0.112 | [0.011, 0.211] | 0.382 | 0.023 |
| loso | loso->tb_live | y_submit_without_validation | constant | 20 | 12 | 83 | 0.000 | n/a | 0.008 | [0.008, 0.008] | 0.093 | 0.088 |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | 20 | 12 | 83 | 0.000 | n/a | 0.209 | [0.102, 0.338] | 0.734 | 0.351 |
| loso | loso->tb_live | y_submit_without_validation | time_only | 20 | 12 | 83 | 0.000 | n/a | 0.037 | [0.034, 0.039] | 0.210 | 0.188 |
| loso | loso->tb_live | y_success_eventual | constant | 20 | 12 | 83 | 1.000 | n/a | 0.284 | [0.284, 0.284] | 0.760 | 0.533 |
| loso | loso->tb_live | y_success_eventual | ledger_basic | 20 | 12 | 83 | 1.000 | n/a | 0.145 | [0.123, 0.168] | 0.447 | 0.337 |
| loso | loso->tb_live | y_success_eventual | time_only | 20 | 12 | 83 | 1.000 | n/a | 0.198 | [0.196, 0.201] | 0.590 | 0.445 |
| loso | loso->tb_live | y_timeout | constant | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso | loso->tb_live | y_timeout | ledger_basic | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso | loso->tb_live | y_timeout | time_only | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | 20 | 10 | 23 | 0.478 | 0.500 | 0.471 | [0.205, 0.697] | 2.312 | 0.470 |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | 20 | 10 | 23 | 0.478 | 0.735 | 0.422 | [0.182, 0.642] | 2.090 | 0.442 |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | 20 | 10 | 23 | 0.478 | 0.731 | 0.475 | [0.207, 0.704] | 2.735 | 0.475 |



## Scheme: `loro`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | 20 | 20 | 499 | 0.170 | 0.323 | 0.143 | [0.112, 0.171] | 0.461 | 0.001 |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | 20 | 20 | 499 | 0.170 | 0.977 | 0.039 | [0.019, 0.062] | 0.138 | 0.032 |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | 20 | 20 | 499 | 0.170 | 0.626 | 0.142 | [0.110, 0.172] | 0.452 | 0.029 |
| swe_agent_pilot | y_submit_without_validation | constant | 20 | 20 | 599 | 0.088 | 0.000 | 0.086 | [0.009, 0.193] | 0.338 | 0.003 |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | 20 | 20 | 599 | 0.088 | 0.676 | 0.080 | [0.017, 0.167] | 0.270 | 0.078 |
| swe_agent_pilot | y_submit_without_validation | time_only | 20 | 20 | 599 | 0.088 | 0.573 | 0.085 | [0.013, 0.185] | 0.308 | 0.053 |
| swe_agent_pilot | y_success_eventual | constant | 20 | 20 | 599 | 0.467 | 0.000 | 0.280 | [0.265, 0.295] | 0.754 | 0.247 |
| swe_agent_pilot | y_success_eventual | ledger_basic | 20 | 20 | 599 | 0.467 | 0.410 | 0.291 | [0.243, 0.340] | 0.802 | 0.196 |
| swe_agent_pilot | y_success_eventual | time_only | 20 | 20 | 599 | 0.467 | 0.281 | 0.283 | [0.263, 0.305] | 0.761 | 0.188 |
| swe_agent_pilot | y_timeout | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | 10 | 10 | 23 | 0.478 | 0.110 | 0.287 | [0.273, 0.302] | 0.769 | 0.420 |
| tb_live | y_validation_new_work_h5 | ledger_basic | 10 | 10 | 23 | 0.478 | 0.492 | 0.284 | [0.196, 0.386] | 0.777 | 0.406 |
| tb_live | y_validation_new_work_h5 | time_only | 10 | 10 | 23 | 0.478 | 0.432 | 0.254 | [0.155, 0.387] | 0.719 | 0.379 |



## Scheme: `loso`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| loso->tb_live | y_future_progress_drop_h5 | constant | 20 | 10 | 23 | 0.130 | 0.500 | 0.115 | [0.029, 0.199] | 0.393 | 0.040 |
| loso->tb_live | y_future_progress_drop_h5 | ledger_basic | 20 | 10 | 23 | 0.130 | 1.000 | 0.002 | [0.000, 0.004] | 0.022 | 0.020 |
| loso->tb_live | y_future_progress_drop_h5 | time_only | 20 | 10 | 23 | 0.130 | 0.917 | 0.112 | [0.011, 0.211] | 0.382 | 0.023 |
| loso->tb_live | y_submit_without_validation | constant | 20 | 12 | 83 | 0.000 | n/a | 0.008 | [0.008, 0.008] | 0.093 | 0.088 |
| loso->tb_live | y_submit_without_validation | ledger_basic | 20 | 12 | 83 | 0.000 | n/a | 0.209 | [0.102, 0.338] | 0.734 | 0.351 |
| loso->tb_live | y_submit_without_validation | time_only | 20 | 12 | 83 | 0.000 | n/a | 0.037 | [0.034, 0.039] | 0.210 | 0.188 |
| loso->tb_live | y_success_eventual | constant | 20 | 12 | 83 | 1.000 | n/a | 0.284 | [0.284, 0.284] | 0.760 | 0.533 |
| loso->tb_live | y_success_eventual | ledger_basic | 20 | 12 | 83 | 1.000 | n/a | 0.145 | [0.123, 0.168] | 0.447 | 0.337 |
| loso->tb_live | y_success_eventual | time_only | 20 | 12 | 83 | 1.000 | n/a | 0.198 | [0.196, 0.201] | 0.590 | 0.445 |
| loso->tb_live | y_timeout | constant | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso->tb_live | y_timeout | ledger_basic | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso->tb_live | y_timeout | time_only | 20 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loso->tb_live | y_validation_new_work_h5 | constant | 20 | 10 | 23 | 0.478 | 0.500 | 0.471 | [0.205, 0.697] | 2.312 | 0.470 |
| loso->tb_live | y_validation_new_work_h5 | ledger_basic | 20 | 10 | 23 | 0.478 | 0.735 | 0.422 | [0.182, 0.642] | 2.090 | 0.442 |
| loso->tb_live | y_validation_new_work_h5 | time_only | 20 | 10 | 23 | 0.478 | 0.731 | 0.475 | [0.207, 0.704] | 2.735 | 0.475 |



## Scheme: `ltfo`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| swe_agent_pilot | y_future_progress_drop_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_submit_without_validation | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_success_eventual | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_timeout | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_future_progress_drop_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_submit_without_validation | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_success_eventual | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_timeout | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | constant | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| tb_live | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |



## Slice-specific metrics

Slices with `< 5` positives or `< 5` negatives are emitted as `n/a (insufficient data)`.


### Slice kind: `phase`

| scheme | source_slice | target | model | slice | n_runs | n_ckpts | pos | neg | AUROC | Brier | ECE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | early | 20 | 171 | 14 | 157 | 0.320 | 0.084 | 0.088 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | late | 20 | 166 | 41 | 125 | 0.331 | 0.193 | 0.078 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | middle | 20 | 162 | 30 | 132 | 0.297 | 0.153 | 0.016 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | early | 20 | 171 | 14 | 157 | 1.000 | 0.001 | 0.017 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | late | 20 | 166 | 41 | 125 | 0.936 | 0.080 | 0.058 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | middle | 20 | 162 | 30 | 132 | 0.973 | 0.038 | 0.036 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | early | 20 | 171 | 14 | 157 | 0.642 | 0.076 | 0.039 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | late | 20 | 166 | 41 | 125 | 0.457 | 0.202 | 0.096 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | middle | 20 | 162 | 30 | 132 | 0.571 | 0.149 | 0.021 |
| loro | swe_agent_pilot | y_submit_without_validation | constant | early | 20 | 206 | 19 | 187 | 0.000 | 0.089 | 0.001 |
| loro | swe_agent_pilot | y_submit_without_validation | constant | late | 20 | 202 | 18 | 184 | 0.000 | 0.087 | 0.002 |
| loro | swe_agent_pilot | y_submit_without_validation | constant | middle | 20 | 191 | 16 | 175 | 0.000 | 0.082 | 0.008 |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | early | 20 | 206 | 19 | 187 | 0.267 | 0.100 | 0.117 |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | late | 20 | 202 | 18 | 184 | 0.969 | 0.065 | 0.062 |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | middle | 20 | 191 | 16 | 175 | 0.781 | 0.074 | 0.071 |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | early | 20 | 206 | 19 | 187 | 0.291 | 0.097 | 0.073 |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | late | 20 | 202 | 18 | 184 | 0.836 | 0.081 | 0.059 |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | middle | 20 | 191 | 16 | 175 | 0.675 | 0.076 | 0.038 |
| loro | swe_agent_pilot | y_success_eventual | constant | early | 20 | 206 | 97 | 109 | 0.000 | 0.280 | 0.248 |
| loro | swe_agent_pilot | y_success_eventual | constant | late | 20 | 202 | 94 | 108 | 0.000 | 0.280 | 0.247 |
| loro | swe_agent_pilot | y_success_eventual | constant | middle | 20 | 191 | 89 | 102 | 0.000 | 0.280 | 0.247 |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | early | 20 | 206 | 97 | 109 | 0.263 | 0.289 | 0.143 |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | late | 20 | 202 | 94 | 108 | 0.642 | 0.242 | 0.137 |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | middle | 20 | 191 | 89 | 102 | 0.231 | 0.345 | 0.344 |
| loro | swe_agent_pilot | y_success_eventual | time_only | early | 20 | 206 | 97 | 109 | 0.102 | 0.279 | 0.177 |
| loro | swe_agent_pilot | y_success_eventual | time_only | late | 20 | 202 | 94 | 108 | 0.209 | 0.287 | 0.224 |
| loro | swe_agent_pilot | y_success_eventual | time_only | middle | 20 | 191 | 89 | 102 | 0.106 | 0.281 | 0.297 |
| loro | tb_live | y_validation_new_work_h5 | constant | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | early | 10 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | late | 7 | 8 | 2 | 6 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | middle | 3 | 3 | 1 | 2 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | early | 10 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | late | 7 | 8 | 2 | 6 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | middle | 3 | 3 | 1 | 2 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | early | 10 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | late | 7 | 8 | 2 | 6 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | middle | 3 | 3 | 1 | 2 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | early | 12 | 32 | 32 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | late | 12 | 27 | 27 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | middle | 12 | 24 | 24 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | early | 12 | 32 | 32 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | late | 12 | 27 | 27 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | middle | 12 | 24 | 24 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | early | 12 | 32 | 32 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | late | 12 | 27 | 27 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | middle | 12 | 24 | 24 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | early | 12 | 32 | 0 | 32 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | late | 12 | 27 | 0 | 27 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | middle | 12 | 24 | 0 | 24 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | early | 10 | 12 | 4 | 8 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | late | 7 | 8 | 5 | 3 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | middle | 3 | 3 | 2 | 1 | n/a (insufficient data) | n/a | n/a |



### Slice kind: `shape`

| scheme | source_slice | target | model | slice | n_runs | n_ckpts | pos | neg | AUROC | Brier | ECE |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | hidden_work_gap | 2 | 88 | 19 | 69 | 0.398 | 0.173 | 0.053 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | high_progress_failure | 3 | 124 | 26 | 98 | 0.400 | 0.169 | 0.045 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | low_progress_success | 1 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | no_validation_frontier | 3 | 55 | 6 | 49 | 0.406 | 0.101 | 0.064 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | nonmonotone_recovery | 2 | 60 | 14 | 46 | 0.475 | 0.183 | 0.067 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | scope_discovery_after_high_progress | 1 | 32 | 8 | 24 | 0.500 | 0.195 | 0.085 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | stuck_loop | 6 | 179 | 34 | 145 | 0.331 | 0.156 | 0.023 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | submit_without_validation | 4 | 74 | 11 | 63 | 0.361 | 0.127 | 0.022 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | constant | validation_induced_reopen | 3 | 96 | 21 | 75 | 0.463 | 0.174 | 0.052 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | hidden_work_gap | 2 | 88 | 19 | 69 | 0.966 | 0.063 | 0.102 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | high_progress_failure | 3 | 124 | 26 | 98 | 0.975 | 0.059 | 0.092 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | low_progress_success | 1 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | no_validation_frontier | 3 | 55 | 6 | 49 | 1.000 | 0.002 | 0.027 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | nonmonotone_recovery | 2 | 60 | 14 | 46 | 1.000 | 0.013 | 0.065 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | scope_discovery_after_high_progress | 1 | 32 | 8 | 24 | 1.000 | 0.021 | 0.091 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | stuck_loop | 6 | 179 | 34 | 145 | 0.977 | 0.052 | 0.063 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | submit_without_validation | 4 | 74 | 11 | 63 | 0.988 | 0.049 | 0.053 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | ledger_basic | validation_induced_reopen | 3 | 96 | 21 | 75 | 0.990 | 0.027 | 0.041 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | hidden_work_gap | 2 | 88 | 19 | 69 | 0.596 | 0.180 | 0.108 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | high_progress_failure | 3 | 124 | 26 | 98 | 0.622 | 0.172 | 0.095 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | low_progress_success | 1 | 12 | 0 | 12 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | no_validation_frontier | 3 | 55 | 6 | 49 | 0.609 | 0.097 | 0.033 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | nonmonotone_recovery | 2 | 60 | 14 | 46 | 0.783 | 0.172 | 0.070 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | scope_discovery_after_high_progress | 1 | 32 | 8 | 24 | 0.729 | 0.184 | 0.083 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | stuck_loop | 6 | 179 | 34 | 145 | 0.564 | 0.162 | 0.111 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | submit_without_validation | 4 | 74 | 11 | 63 | 0.808 | 0.119 | 0.056 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | time_only | validation_induced_reopen | 3 | 96 | 21 | 75 | 0.752 | 0.163 | 0.077 |
| loro | swe_agent_pilot | y_submit_without_validation | constant | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | nonmonotone_recovery | 2 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | scope_discovery_after_high_progress | 1 | 37 | 0 | 37 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | constant | submit_without_validation | 4 | 94 | 53 | 41 | 0.000 | 0.501 | 0.488 |
| loro | swe_agent_pilot | y_submit_without_validation | constant | validation_induced_reopen | 3 | 111 | 0 | 111 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | nonmonotone_recovery | 2 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | scope_discovery_after_high_progress | 1 | 37 | 0 | 37 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | submit_without_validation | 4 | 94 | 53 | 41 | 0.571 | 0.427 | 0.474 |
| loro | swe_agent_pilot | y_submit_without_validation | ledger_basic | validation_induced_reopen | 3 | 111 | 0 | 111 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | nonmonotone_recovery | 2 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | scope_discovery_after_high_progress | 1 | 37 | 0 | 37 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | submit_without_validation | 4 | 94 | 53 | 41 | 0.663 | 0.470 | 0.494 |
| loro | swe_agent_pilot | y_submit_without_validation | time_only | validation_induced_reopen | 3 | 111 | 0 | 111 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | nonmonotone_recovery | 2 | 70 | 70 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | scope_discovery_after_high_progress | 1 | 37 | 37 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | constant | submit_without_validation | 4 | 94 | 17 | 77 | 0.000 | 0.253 | 0.304 |
| loro | swe_agent_pilot | y_success_eventual | constant | validation_induced_reopen | 3 | 111 | 70 | 41 | 0.000 | 0.295 | 0.542 |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | nonmonotone_recovery | 2 | 70 | 70 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | scope_discovery_after_high_progress | 1 | 37 | 37 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | submit_without_validation | 4 | 94 | 17 | 77 | 0.247 | 0.280 | 0.398 |
| loro | swe_agent_pilot | y_success_eventual | ledger_basic | validation_induced_reopen | 3 | 111 | 70 | 41 | 0.415 | 0.280 | 0.261 |
| loro | swe_agent_pilot | y_success_eventual | time_only | hidden_work_gap | 2 | 98 | 0 | 98 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | high_progress_failure | 3 | 139 | 0 | 139 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | low_progress_success | 1 | 17 | 17 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | no_validation_frontier | 3 | 70 | 0 | 70 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | nonmonotone_recovery | 2 | 70 | 70 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | scope_discovery_after_high_progress | 1 | 37 | 37 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | stuck_loop | 6 | 209 | 0 | 209 | n/a (insufficient data) | n/a | n/a |
| loro | swe_agent_pilot | y_success_eventual | time_only | submit_without_validation | 4 | 94 | 17 | 77 | 0.414 | 0.261 | 0.327 |
| loro | swe_agent_pilot | y_success_eventual | time_only | validation_induced_reopen | 3 | 111 | 70 | 41 | 0.318 | 0.295 | 0.224 |
| loro | tb_live | y_validation_new_work_h5 | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loro | tb_live | y_validation_new_work_h5 | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_future_progress_drop_h5 | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_submit_without_validation | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_success_eventual | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_timeout | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | constant | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | ledger_basic | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | hidden_work_gap | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | high_progress_failure | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | low_progress_success | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | no_validation_frontier | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | nonmonotone_recovery | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | scope_discovery_after_high_progress | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | stuck_loop | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | submit_without_validation | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |
| loso | loso->tb_live | y_validation_new_work_h5 | time_only | validation_induced_reopen | 0 | 0 | 0 | 0 | n/a (insufficient data) | n/a | n/a |


