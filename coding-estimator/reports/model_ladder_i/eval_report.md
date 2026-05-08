
# Workstream I model ladder
_Generated 2026-05-05T02:39:58+00:00._

I0 empirical-bin and I1 logistic-regression models evaluated on the combined holdout split and per-source loro diagnostics. All models consume prefix-only checkpoint features and write bundles under `models/`.

## Headline metrics

| scheme | source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| holdout | holdout->all | y_future_progress_drop_h5 | empirical_bin_v0 | 24 | 6 | 92 | 0.228 | 0.887 | 0.105 | [0.041, 0.150] | 0.304 | 0.022 |
| holdout | holdout->all | y_future_progress_drop_h5 | logreg_v0 | 24 | 6 | 92 | 0.228 | 0.975 | 0.047 | [0.008, 0.129] | 0.163 | 0.060 |
| holdout | holdout->all | y_submit_without_validation | empirical_bin_v0 | 26 | 6 | 122 | 0.000 | n/a | 0.013 | [0.011, 0.016] | 0.098 | 0.090 |
| holdout | holdout->all | y_submit_without_validation | logreg_v0 | 26 | 6 | 122 | 0.000 | n/a | 0.013 | [0.012, 0.014] | 0.091 | 0.084 |
| holdout | holdout->all | y_success_eventual | empirical_bin_v0 | 26 | 6 | 122 | 0.811 | 0.575 | 0.248 | [0.183, 0.285] | 0.723 | 0.282 |
| holdout | holdout->all | y_success_eventual | logreg_v0 | 26 | 6 | 122 | 0.811 | 0.527 | 0.308 | [0.217, 0.362] | 0.826 | 0.357 |
| holdout | holdout->all | y_validation_new_work_h5 | empirical_bin_v0 | 24 | 6 | 92 | 0.000 | n/a | 0.002 | [0.002, 0.003] | 0.027 | 0.026 |
| holdout | holdout->all | y_validation_new_work_h5 | logreg_v0 | 24 | 6 | 92 | 0.000 | n/a | 0.002 | [0.001, 0.004] | 0.029 | 0.028 |
| loro | hermes_pilot_h5_v2 | y_future_progress_drop_h5 | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_future_progress_drop_h5 | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_submit_without_validation | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_submit_without_validation | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_success_eventual | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_success_eventual | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_validation_new_work_h5 | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | hermes_pilot_h5_v2 | y_validation_new_work_h5 | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | empirical_bin_v0 | 20 | 20 | 499 | 0.170 | 0.843 | 0.099 | [0.072, 0.127] | 0.298 | 0.097 |
| loro | swe_agent_pilot | y_future_progress_drop_h5 | logreg_v0 | 20 | 20 | 499 | 0.170 | 0.977 | 0.039 | [0.019, 0.062] | 0.138 | 0.032 |
| loro | swe_agent_pilot | y_submit_without_validation | empirical_bin_v0 | 20 | 20 | 599 | 0.088 | 0.666 | 0.085 | [0.020, 0.175] | 0.300 | 0.044 |
| loro | swe_agent_pilot | y_submit_without_validation | logreg_v0 | 20 | 20 | 599 | 0.088 | 0.676 | 0.080 | [0.017, 0.167] | 0.270 | 0.078 |
| loro | swe_agent_pilot | y_success_eventual | empirical_bin_v0 | 20 | 20 | 599 | 0.467 | 0.432 | 0.287 | [0.247, 0.330] | 0.806 | 0.242 |
| loro | swe_agent_pilot | y_success_eventual | logreg_v0 | 20 | 20 | 599 | 0.467 | 0.410 | 0.291 | [0.243, 0.340] | 0.802 | 0.196 |
| loro | swe_agent_pilot | y_validation_new_work_h5 | empirical_bin_v0 | 20 | 20 | 499 | 0.008 | 0.384 | 0.008 | [0.000, 0.024] | 0.065 | 0.001 |
| loro | swe_agent_pilot | y_validation_new_work_h5 | logreg_v0 | 20 | 20 | 499 | 0.008 | 0.158 | 0.008 | [0.000, 0.025] | 0.063 | 0.003 |
| loro | tb_live | y_future_progress_drop_h5 | empirical_bin_v0 | 10 | 10 | 23 | 0.130 | 0.867 | 0.097 | [0.000, 0.193] | 0.288 | 0.134 |
| loro | tb_live | y_future_progress_drop_h5 | logreg_v0 | 10 | 10 | 23 | 0.130 | 0.917 | 0.096 | [0.003, 0.157] | 0.291 | 0.140 |
| loro | tb_live | y_submit_without_validation | empirical_bin_v0 | 12 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loro | tb_live | y_submit_without_validation | logreg_v0 | 12 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loro | tb_live | y_success_eventual | empirical_bin_v0 | 12 | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loro | tb_live | y_success_eventual | logreg_v0 | 12 | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| loro | tb_live | y_validation_new_work_h5 | empirical_bin_v0 | 10 | 10 | 23 | 0.478 | 0.485 | 0.299 | [0.215, 0.407] | 1.008 | 0.222 |
| loro | tb_live | y_validation_new_work_h5 | logreg_v0 | 10 | 10 | 23 | 0.478 | 0.492 | 0.284 | [0.196, 0.386] | 0.777 | 0.406 |



## Scheme: `holdout`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| holdout->all | y_future_progress_drop_h5 | empirical_bin_v0 | 24 | 6 | 92 | 0.228 | 0.887 | 0.105 | [0.041, 0.150] | 0.304 | 0.022 |
| holdout->all | y_future_progress_drop_h5 | logreg_v0 | 24 | 6 | 92 | 0.228 | 0.975 | 0.047 | [0.008, 0.129] | 0.163 | 0.060 |
| holdout->all | y_submit_without_validation | empirical_bin_v0 | 26 | 6 | 122 | 0.000 | n/a | 0.013 | [0.011, 0.016] | 0.098 | 0.090 |
| holdout->all | y_submit_without_validation | logreg_v0 | 26 | 6 | 122 | 0.000 | n/a | 0.013 | [0.012, 0.014] | 0.091 | 0.084 |
| holdout->all | y_success_eventual | empirical_bin_v0 | 26 | 6 | 122 | 0.811 | 0.575 | 0.248 | [0.183, 0.285] | 0.723 | 0.282 |
| holdout->all | y_success_eventual | logreg_v0 | 26 | 6 | 122 | 0.811 | 0.527 | 0.308 | [0.217, 0.362] | 0.826 | 0.357 |
| holdout->all | y_validation_new_work_h5 | empirical_bin_v0 | 24 | 6 | 92 | 0.000 | n/a | 0.002 | [0.002, 0.003] | 0.027 | 0.026 |
| holdout->all | y_validation_new_work_h5 | logreg_v0 | 24 | 6 | 92 | 0.000 | n/a | 0.002 | [0.001, 0.004] | 0.029 | 0.028 |



## Scheme: `loro`

| source_slice | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | log_loss | ECE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| hermes_pilot_h5_v2 | y_future_progress_drop_h5 | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_future_progress_drop_h5 | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_submit_without_validation | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_submit_without_validation | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_success_eventual | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_success_eventual | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_validation_new_work_h5 | empirical_bin_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| hermes_pilot_h5_v2 | y_validation_new_work_h5 | logreg_v0 | n/a | n/a | n/a | n/a (insufficient data) | n/a | n/a | n/a | n/a | n/a |
| swe_agent_pilot | y_future_progress_drop_h5 | empirical_bin_v0 | 20 | 20 | 499 | 0.170 | 0.843 | 0.099 | [0.072, 0.127] | 0.298 | 0.097 |
| swe_agent_pilot | y_future_progress_drop_h5 | logreg_v0 | 20 | 20 | 499 | 0.170 | 0.977 | 0.039 | [0.019, 0.062] | 0.138 | 0.032 |
| swe_agent_pilot | y_submit_without_validation | empirical_bin_v0 | 20 | 20 | 599 | 0.088 | 0.666 | 0.085 | [0.020, 0.175] | 0.300 | 0.044 |
| swe_agent_pilot | y_submit_without_validation | logreg_v0 | 20 | 20 | 599 | 0.088 | 0.676 | 0.080 | [0.017, 0.167] | 0.270 | 0.078 |
| swe_agent_pilot | y_success_eventual | empirical_bin_v0 | 20 | 20 | 599 | 0.467 | 0.432 | 0.287 | [0.247, 0.330] | 0.806 | 0.242 |
| swe_agent_pilot | y_success_eventual | logreg_v0 | 20 | 20 | 599 | 0.467 | 0.410 | 0.291 | [0.243, 0.340] | 0.802 | 0.196 |
| swe_agent_pilot | y_validation_new_work_h5 | empirical_bin_v0 | 20 | 20 | 499 | 0.008 | 0.384 | 0.008 | [0.000, 0.024] | 0.065 | 0.001 |
| swe_agent_pilot | y_validation_new_work_h5 | logreg_v0 | 20 | 20 | 499 | 0.008 | 0.158 | 0.008 | [0.000, 0.025] | 0.063 | 0.003 |
| tb_live | y_future_progress_drop_h5 | empirical_bin_v0 | 10 | 10 | 23 | 0.130 | 0.867 | 0.097 | [0.000, 0.193] | 0.288 | 0.134 |
| tb_live | y_future_progress_drop_h5 | logreg_v0 | 10 | 10 | 23 | 0.130 | 0.917 | 0.096 | [0.003, 0.157] | 0.291 | 0.140 |
| tb_live | y_submit_without_validation | empirical_bin_v0 | 12 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| tb_live | y_submit_without_validation | logreg_v0 | 12 | 12 | 83 | 0.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| tb_live | y_success_eventual | empirical_bin_v0 | 12 | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| tb_live | y_success_eventual | logreg_v0 | 12 | 12 | 83 | 1.000 | n/a | 0.000 | [0.000, 0.000] | 0.001 | 0.001 |
| tb_live | y_validation_new_work_h5 | empirical_bin_v0 | 10 | 10 | 23 | 0.478 | 0.485 | 0.299 | [0.215, 0.407] | 1.008 | 0.222 |
| tb_live | y_validation_new_work_h5 | logreg_v0 | 10 | 10 | 23 | 0.478 | 0.492 | 0.284 | [0.196, 0.386] | 0.777 | 0.406 |


