# Estimator checkpoint table — summary (W3)

Derived feature table for downstream estimators. Each row is a
checkpoint corresponding to one retained step from the step-level
observation table. **All feature columns are derivable from events at
or before the row's step**; nothing leaks future state.

- Source step CSV: `datasets/hermes_pilot_h5_observations_step.csv`
- Source runs dir: `runs/hermes_pilot_h5`
- Shape labels source: `datasets/hermes_pilot_h5_shape_labels.csv`
- Horizon for `label_success_by_horizon`: **30 steps**
- Total checkpoints: **370**
- Distinct runs: **30**

## Column groups

| Group | Columns |
|---|---|
| frontier | active_leaf_count, active_coding_leaf_count, active_validation_leaf_count |
| closure | completed_leaf_count, coding_progress, validation_progress |
| instability | num_reopens_so_far, num_invalidations_so_far, largest_progress_drop_so_far |
| discovery | num_splits_so_far, steps_since_new_subtask, denominator_growth_so_far |
| stalls | steps_since_completion, blocked_leaf_count, repeated_observation_loop_flag |
| validation | validation_started, validation_complete, validation_failed, submit_without_validation |
| evidence | strong_completion_count, manual_only_completion_count, weak_product_completion_count |
| **labels (never features)** | label_final_success, label_finish_step, label_success_by_horizon, label_shape_tags |

## Per-run checkpoint counts

| run_id | checkpoints |
|---|---:|
| `hermes_pilot_h5_001` | 3 |
| `hermes_pilot_h5_002` | 5 |
| `hermes_pilot_h5_003` | 17 |
| `hermes_pilot_h5_004` | 9 |
| `hermes_pilot_h5_005` | 11 |
| `hermes_pilot_h5_006` | 3 |
| `hermes_pilot_h5_007` | 11 |
| `hermes_pilot_h5_008` | 31 |
| `hermes_pilot_h5_009` | 15 |
| `hermes_pilot_h5_010` | 15 |
| `hermes_pilot_h5_011` | 5 |
| `hermes_pilot_h5_012` | 11 |
| `hermes_pilot_h5_013` | 25 |
| `hermes_pilot_h5_014` | 9 |
| `hermes_pilot_h5_015` | 23 |
| `hermes_pilot_h5_016` | 3 |
| `hermes_pilot_h5_017` | 11 |
| `hermes_pilot_h5_018` | 17 |
| `hermes_pilot_h5_019` | 17 |
| `hermes_pilot_h5_020` | 11 |
| `hermes_pilot_h5_021` | 3 |
| `hermes_pilot_h5_022` | 9 |
| `hermes_pilot_h5_023` | 23 |
| `hermes_pilot_h5_024` | 13 |
| `hermes_pilot_h5_025` | 13 |
| `hermes_pilot_h5_026` | 3 |
| `hermes_pilot_h5_027` | 9 |
| `hermes_pilot_h5_028` | 19 |
| `hermes_pilot_h5_029` | 11 |
| `hermes_pilot_h5_030` | 15 |

## Caveats

- `label_*` columns must not be used as features. Tests assert the
  prefix; training pipelines should drop them at the schema layer.
- `repeated_observation_loop_flag` keys on "loop"/"stuck" in the
  block-event reason text. Live ledgers without explicit blocked
  semantics will leave it false.
- `denominator_growth_so_far` measures total active coding-category
  weight added since the first non-empty checkpoint; SPLIT events
  preserve denominator and won't move it.
- Legacy retrospective rows (no timestamps) remain supported; the
  table consumes step indices, not wall-clock seconds.
