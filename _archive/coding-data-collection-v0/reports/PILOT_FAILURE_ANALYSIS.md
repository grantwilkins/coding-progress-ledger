# Pilot Failure Analysis

Scale batch is blocked because at least one pilot gate failed.

## Failed Gates

- `real_agent_pilot_runs_present`
- `median_transcript_steps`
- `validation_attempt_coverage`
- `validation_fail_observed_coverage`
- `progress_drop_coverage`
- `terminal_failure_rate`
- `high_progress_failures_or_disagreements`
- `median_observation_events_per_run`
- `verifier_outcomes_reproducible`
- `artifact_hardening`
- `estimator_prefix_safety_and_alignment`

## Recommended Action

- Run typed `model_tool_loop` arms; protocol-smoke shell runs are excluded from L metrics.
- Resample toward longer tasks or improve live transcript capture before K.
- Prefer tasks with visible test/validation loops and richer verifier feedback.
- Add tasks likely to expose late failures after apparent progress.
- Rebuild estimator artifacts and require complete prefix provenance/alignment.
- Rerun verifier determinism on a broader sample before scaling.

Do not run Workstream M until all gates pass.
