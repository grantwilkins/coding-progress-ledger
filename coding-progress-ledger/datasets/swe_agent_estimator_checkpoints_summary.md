# Estimator checkpoint table — summary (W3)

Derived feature table for downstream estimators. Each row is a
checkpoint corresponding to one retained step from the step-level
observation table. **All feature columns are derivable from events at
or before the row's step**; nothing leaks future state.

- Source step CSV: `datasets/swe_agent_pilot_observations_step.csv`
- Source runs dir: `runs/swe_agent_pilot`
- Shape labels source: `datasets/swe_agent_pilot_shape_labels.csv`
- Horizon for `label_success_by_horizon`: **30 steps**
- Total checkpoints: **191**
- Distinct runs: **20**

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
| `swe_agent_pilot_f_01` | 7 |
| `swe_agent_pilot_f_02` | 5 |
| `swe_agent_pilot_f_03` | 5 |
| `swe_agent_pilot_f_04` | 7 |
| `swe_agent_pilot_f_05` | 10 |
| `swe_agent_pilot_f_06` | 11 |
| `swe_agent_pilot_f_07` | 7 |
| `swe_agent_pilot_f_08` | 15 |
| `swe_agent_pilot_f_09` | 13 |
| `swe_agent_pilot_f_10` | 7 |
| `swe_agent_pilot_s_01` | 12 |
| `swe_agent_pilot_s_02` | 11 |
| `swe_agent_pilot_s_03` | 14 |
| `swe_agent_pilot_s_04` | 6 |
| `swe_agent_pilot_s_05` | 13 |
| `swe_agent_pilot_s_06` | 11 |
| `swe_agent_pilot_s_07` | 10 |
| `swe_agent_pilot_s_08` | 9 |
| `swe_agent_pilot_s_09` | 8 |
| `swe_agent_pilot_s_10` | 10 |

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
