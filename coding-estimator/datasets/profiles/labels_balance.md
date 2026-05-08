# Label-balance profile (F3)

Cross-source rollup of E8 per-source balance reports. Used by F11 to enforce the `>= 5 positives AND >= 5 negatives on >= 1 source for terminal labels` gate.

## Cross-source headline (binary targets only)

| target | source | positives | negatives | masked | n_unmasked | positive_rate | thin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y_future_progress_drop_h5 | swe_agent_pilot | 85 | 414 | 100 | 499 | 0.170 | no |
| y_future_progress_drop_h5 | tb_live | 3 | 20 | 60 | 23 | 0.130 | YES |
| y_submit_without_validation | swe_agent_pilot | 53 | 546 | 0 | 599 | 0.088 | no |
| y_submit_without_validation | tb_live | 0 | 83 | 0 | 83 | 0.000 | YES |
| y_success_eventual | swe_agent_pilot | 280 | 319 | 0 | 599 | 0.467 | no |
| y_success_eventual | tb_live | 83 | 0 | 0 | 83 | 1.000 | YES |
| y_timeout | swe_agent_pilot | 0 | 599 | 0 | 599 | 0.000 | YES |
| y_timeout | tb_live | 0 | 83 | 0 | 83 | 0.000 | YES |
| y_validation_new_work_h5 | swe_agent_pilot | 4 | 495 | 100 | 499 | 0.008 | YES |
| y_validation_new_work_h5 | tb_live | 11 | 12 | 60 | 23 | 0.478 | no |

---

## Source: swe_agent_pilot

# Label balance: swe_agent_pilot

_n_runs=20, n_checkpoints=599_

## Per target (binary targets only)

| target_name | positives | negatives | masked | n_unmasked | positive_rate | thin |
| --- | --- | --- | --- | --- | --- | --- |
| y_future_progress_drop_h5 | 85 | 414 | 100 | 499 | 0.170 | False |
| y_submit_without_validation | 53 | 546 | 0 | 599 | 0.088 | False |
| y_success_eventual | 280 | 319 | 0 | 599 | 0.467 | False |
| y_timeout | 0 | 599 | 0 | 599 | 0.000 | True |
| y_validation_new_work_h5 | 4 | 495 | 100 | 499 | 0.008 | True |

### Thin cells (positives<5 or negatives<5)

| target_name | positives | negatives |
| --- | --- | --- |
| y_timeout | 0 | 599 |
| y_validation_new_work_h5 | 4 | 495 |

---

## Source: tb_live

# Label balance: tb_live

_n_runs=12, n_checkpoints=83_

## Per target (binary targets only)

| target_name | positives | negatives | masked | n_unmasked | positive_rate | thin |
| --- | --- | --- | --- | --- | --- | --- |
| y_future_progress_drop_h5 | 3 | 20 | 60 | 23 | 0.130 | True |
| y_submit_without_validation | 0 | 83 | 0 | 83 | 0.000 | True |
| y_success_eventual | 83 | 0 | 0 | 83 | 1.000 | True |
| y_timeout | 0 | 83 | 0 | 83 | 0.000 | True |
| y_validation_new_work_h5 | 11 | 12 | 60 | 23 | 0.478 | False |

### Thin cells (positives<5 or negatives<5)

| target_name | positives | negatives |
| --- | --- | --- |
| y_future_progress_drop_h5 | 3 | 20 |
| y_submit_without_validation | 0 | 83 |
| y_success_eventual | 83 | 0 |
| y_timeout | 0 | 83 |

---

