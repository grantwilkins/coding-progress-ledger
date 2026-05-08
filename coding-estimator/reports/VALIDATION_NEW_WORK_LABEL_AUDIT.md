# Validation-New-Work Label Audit

Use the upstream snapshot logic as the source of truth for validation transitions and discovery/reopen events.

| slice_name | n_runs | n_checkpoints | n_runs_with_validation_transition | n_runs_with_discovery_or_reopen_after_validation_within_5 | n_runs_with_discovery_or_reopen_after_validation_any_later | n_unmasked_positive_checkpoints_current_label | recommendation |
|---|---|---|---|---|---|---|---|
| all_tb_live_v2_runs | 102 | 703 | 0 | 0 | 0 | 0 | defer_on_tb_live_v2 |
| validation_new_work_family_only | 21 | 147 | 0 | 0 | 0 | 0 | defer_on_tb_live_v2 |

- recommendation: `defer_on_tb_live_v2`

The current live substrate does not emit the upstream-recognized validation-transition pattern required by `y_validation_new_work_h5`.

