# Calibration — v0 headline

_Generated 2026-05-05T03:06:52+00:00._

Headline rollup for the v0 calibration gate. Cross-validated isotonic recalibration uses K-fold over run_ids.

Gate: any (model, source, target) with `ECE > 0.1` after isotonic recalibration is **not_safe_for_control** and must carry that annotation in its model card.

| model | source | target | n | Brier (raw) | ECE (raw) | Brier (iso) | ECE (iso) | not_safe_for_control |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| ledger_basic | loso->tb_live | y_future_progress_drop_h5 | 23 | 0.002 | 0.020 | 0.000 | 0.003 | no |
| ledger_basic | loso->tb_live | y_submit_without_validation | 83 | 0.209 | 0.351 | n/a | n/a | **yes** |
| ledger_basic | loso->tb_live | y_success_eventual | 83 | 0.145 | 0.337 | n/a | n/a | **yes** |
| ledger_basic | loso->tb_live | y_validation_new_work_h5 | 23 | 0.422 | 0.442 | 0.312 | 0.277 | **yes** |
| ledger_basic | swe_agent_pilot | y_future_progress_drop_h5 | 499 | 0.039 | 0.032 | 0.047 | 0.042 | no |
| ledger_basic | swe_agent_pilot | y_submit_without_validation | 599 | 0.080 | 0.078 | 0.089 | 0.111 | **yes** |
| ledger_basic | swe_agent_pilot | y_success_eventual | 599 | 0.291 | 0.196 | 0.268 | 0.144 | **yes** |
| ledger_basic | swe_agent_pilot | y_validation_new_work_h5 | 499 | 0.008 | 0.003 | 0.008 | 0.001 | no |
| ledger_basic | tb_live | y_future_progress_drop_h5 | 23 | 0.096 | 0.140 | 0.142 | 0.174 | **yes** |
| ledger_basic | tb_live | y_submit_without_validation | 83 | 0.000 | 0.001 | n/a | n/a | no |
| ledger_basic | tb_live | y_success_eventual | 83 | 0.000 | 0.001 | n/a | n/a | no |
| ledger_basic | tb_live | y_validation_new_work_h5 | 23 | 0.284 | 0.406 | 0.349 | 0.283 | **yes** |
| time_only | loso->tb_live | y_future_progress_drop_h5 | 23 | 0.112 | 0.023 | 0.107 | 0.135 | **yes** |
| time_only | loso->tb_live | y_submit_without_validation | 83 | 0.037 | 0.188 | n/a | n/a | **yes** |
| time_only | loso->tb_live | y_success_eventual | 83 | 0.198 | 0.445 | n/a | n/a | **yes** |
| time_only | loso->tb_live | y_validation_new_work_h5 | 23 | 0.475 | 0.475 | 0.288 | 0.281 | **yes** |
| time_only | swe_agent_pilot | y_future_progress_drop_h5 | 499 | 0.142 | 0.029 | 0.143 | 0.052 | no |
| time_only | swe_agent_pilot | y_submit_without_validation | 599 | 0.085 | 0.053 | 0.089 | 0.098 | no |
| time_only | swe_agent_pilot | y_success_eventual | 599 | 0.283 | 0.188 | 0.269 | 0.222 | **yes** |
| time_only | swe_agent_pilot | y_validation_new_work_h5 | 499 | 0.016 | 0.018 | 0.008 | 0.001 | no |
| time_only | tb_live | y_future_progress_drop_h5 | 23 | 0.133 | 0.117 | 0.161 | 0.168 | **yes** |
| time_only | tb_live | y_submit_without_validation | 83 | 0.000 | 0.001 | n/a | n/a | no |
| time_only | tb_live | y_success_eventual | 83 | 0.000 | 0.001 | n/a | n/a | no |
| time_only | tb_live | y_validation_new_work_h5 | 23 | 0.254 | 0.379 | 0.296 | 0.340 | **yes** |

## Cells flagged not_safe_for_control

- `time_only` / `swe_agent_pilot` / `y_success_eventual` — ECE_after=0.222
- `ledger_basic` / `swe_agent_pilot` / `y_success_eventual` — ECE_after=0.144
- `ledger_basic` / `swe_agent_pilot` / `y_submit_without_validation` — ECE_after=0.111
- `time_only` / `tb_live` / `y_future_progress_drop_h5` — ECE_after=0.168
- `time_only` / `tb_live` / `y_validation_new_work_h5` — ECE_after=0.340
- `ledger_basic` / `tb_live` / `y_future_progress_drop_h5` — ECE_after=0.174
- `ledger_basic` / `tb_live` / `y_validation_new_work_h5` — ECE_after=0.283
- `time_only` / `loso->tb_live` / `y_success_eventual` — ECE_after=0.445
- `time_only` / `loso->tb_live` / `y_future_progress_drop_h5` — ECE_after=0.135
- `time_only` / `loso->tb_live` / `y_validation_new_work_h5` — ECE_after=0.281
- `time_only` / `loso->tb_live` / `y_submit_without_validation` — ECE_after=0.188
- `ledger_basic` / `loso->tb_live` / `y_success_eventual` — ECE_after=0.337
- `ledger_basic` / `loso->tb_live` / `y_validation_new_work_h5` — ECE_after=0.277
- `ledger_basic` / `loso->tb_live` / `y_submit_without_validation` — ECE_after=0.351

