# Estimator feature groups

Every feature column is **prefix-only at `t`**. The original v0 groups
were derivable from `ledger.jsonl` alone; the observation-upgrade phase
adds a separate transcript/verifier-derived group without changing
ledger semantics.

The machine-readable form lives in
`coding_estimator/checkpoints/features/registry.py` and is validated against
`schemas/feature_schema.json`.

## Groups

```
frontier      active_leaf_count, active_coding_leaf_count, active_validation_leaf_count
closure       completed_leaf_count, coding_progress, validation_progress,
              product_progress, investigation_progress
discovery     num_adds_so_far, num_splits_so_far, denominator_growth_so_far,
              steps_since_new_subtask, new_leaf_count_last_{1,3,5}_steps
instability   num_reopens_so_far, num_invalidations_so_far, num_deletes_so_far,
              largest_progress_drop_so_far, num_progress_drops_so_far,
              steps_since_last_drop
stalling      blocked_leaf_count, blocked_coding_leaf_count, blocked_validation_leaf_count,
              steps_since_completion, steps_since_progress_increase,
              steps_since_status_change, steps_since_evidence,
              repeated_observation_loop_flag, no_progress_window_{5,10}
validation    validation_leaf_exists, validation_started, validation_complete,
              validation_failed, validation_blocked, validation_in_progress,
              num_validation_attempts, num_validation_failures, num_validation_successes,
              steps_since_last_validation, submit_without_validation_so_far
evidence      strong_completion_count, manual_only_completion_count,
              weak_product_completion_count, strong_evidence_fraction,
              manual_only_evidence_fraction, latest_completion_evidence_type
observation   obs_num_validation_{attempts,successes,failures}_so_far,
              obs_num_error_observations_so_far, obs_num_repeated_errors_so_far,
              obs_num_environment_blocks_so_far,
              obs_solution_oracle_read_so_far, obs_agent_claims_done_so_far,
              obs_done_without_validation_so_far,
              obs_verifier_disagreement_proxy_so_far,
              obs_validation_after_oracle_read_so_far
time_budget   elapsed_steps (always); elapsed_wall_time (live wallclock sources only);
              fraction_timeout_consumed, remaining_timeout_budget (tb_live only);
              completion_rate_recent_steps
source_task   source, agent_scaffold, model_name, task_family_hash, repo_family_hash,
              initial_prompt_length, initial_files_count   (run-constant; reported separately)
```

## v0 exclusions

Before the observation upgrade, these transcript-derived signals were
out of scope:

```
elapsed_agent_turns, elapsed_tool_calls, elapsed_commands,
elapsed_tokens_if_available
repeated_command_count, repeated_observation_count,
same_error_loop_flag, two_command_oscillation_flag
average_wall_time_per_completion, tool_call_rate_recent
```

The new `observation` group is the structured replacement for that gap.
It is built from additive `observation_events.jsonl`, not by changing
ledger replay or progress scoring.

Stretch (deferred to Workstream Q):

```
semantic / wall-clock-stalling-cross-source / cross-validation-features
```

## Per-source availability

| group | swe_agent_* | hermes_pilot* | tb_live | tb_live_v2 |
|---|---|---|---|
| frontier, closure, discovery, instability, stalling, validation, evidence | populated | populated | populated | populated |
| observation | not populated | not populated | not populated | populated |
| `elapsed_wall_time` | only `*_wallclock` | not populated | populated | populated |
| `fraction_timeout_consumed`, `remaining_timeout_budget` | not populated | not populated | populated | not populated |

`source_task` features are run-constant; the audit in B4.5 will fail-loud
on any (run-constant feature, run-constant target) pair.
