# Feature leakage profile (F4)

Auto-generated. Per-feature audit of leakage indicators. A feature is **clean** when no FLAG appears in its row.

## Summary

- Total features: 60
- Features flagged FORBIDDEN_TOKEN: 0
- Features flagged NOT_PREFIX_ONLY: 0
- Features flagged NEAR_CONSTANT: 6
- Features flagged HIGH_CARDINALITY_ID: 0

## Group: frontier

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| active_leaf_count | frontier | 1.00 | — |
| active_coding_leaf_count | frontier | 1.00 | — |
| active_validation_leaf_count | frontier | 1.00 | — |

## Group: closure

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| completed_leaf_count | closure | 1.00 | — |
| coding_progress | closure | 1.00 | — |
| validation_progress | closure | 1.00 | — |
| product_progress | closure | 1.00 | — |
| investigation_progress | closure | 1.00 | — |

## Group: discovery

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| num_adds_so_far | discovery | 1.00 | — |
| num_splits_so_far | discovery | 1.00 | NEAR_CONSTANT |
| denominator_growth_so_far | discovery | 1.00 | — |
| steps_since_new_subtask | discovery | 1.00 | — |
| new_leaf_count_last_1_steps | discovery | 1.00 | — |
| new_leaf_count_last_3_steps | discovery | 1.00 | — |
| new_leaf_count_last_5_steps | discovery | 1.00 | — |

## Group: instability

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| num_reopens_so_far | instability | 1.00 | — |
| num_invalidations_so_far | instability | 1.00 | NEAR_CONSTANT |
| num_deletes_so_far | instability | 1.00 | NEAR_CONSTANT |
| largest_progress_drop_so_far | instability | 1.00 | — |
| num_progress_drops_so_far | instability | 1.00 | — |
| steps_since_last_drop | instability | 1.00 | — |

## Group: stalling

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| blocked_leaf_count | stalling | 1.00 | — |
| blocked_coding_leaf_count | stalling | 1.00 | — |
| blocked_validation_leaf_count | stalling | 1.00 | — |
| steps_since_completion | stalling | 1.00 | — |
| steps_since_progress_increase | stalling | 1.00 | — |
| steps_since_status_change | stalling | 1.00 | — |
| steps_since_evidence | stalling | 1.00 | — |
| repeated_observation_loop_flag | stalling | 1.00 | NEAR_CONSTANT |
| no_progress_window_5 | stalling | 1.00 | — |
| no_progress_window_10 | stalling | 1.00 | — |

## Group: validation

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| validation_leaf_exists | validation | 1.00 | — |
| validation_started | validation | 1.00 | — |
| validation_complete | validation | 1.00 | — |
| validation_failed | validation | 1.00 | NEAR_CONSTANT |
| validation_blocked | validation | 1.00 | — |
| validation_in_progress | validation | 1.00 | — |
| num_validation_attempts | validation | 1.00 | — |
| num_validation_failures | validation | 1.00 | NEAR_CONSTANT |
| num_validation_successes | validation | 1.00 | — |
| steps_since_last_validation | validation | 1.00 | — |
| submit_without_validation_so_far | validation | 1.00 | — |

## Group: evidence

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| strong_completion_count | evidence | 1.00 | — |
| manual_only_completion_count | evidence | 1.00 | — |
| weak_product_completion_count | evidence | 1.00 | — |
| strong_evidence_fraction | evidence | 1.00 | — |
| manual_only_evidence_fraction | evidence | 1.00 | — |
| latest_completion_evidence_type | evidence | 0.66 | — |

## Group: time_budget

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| elapsed_steps | time_budget | 1.00 | — |
| elapsed_wall_time | time_budget | 0.34 | — |
| fraction_timeout_consumed | time_budget | 0.00 | — |
| remaining_timeout_budget | time_budget | 0.00 | — |
| completion_rate_recent_steps | time_budget | 1.00 | — |

## Group: source_task

| feature | group | non_null_rate | flags |
| --- | --- | --- | --- |
| source | source_task | 1.00 | — |
| agent_scaffold | source_task | 0.00 | — |
| model_name | source_task | 0.00 | — |
| task_family_hash | source_task | 0.00 | — |
| repo_family_hash | source_task | 0.00 | — |
| initial_prompt_length | source_task | 0.00 | — |
| initial_files_count | source_task | 0.00 | — |

