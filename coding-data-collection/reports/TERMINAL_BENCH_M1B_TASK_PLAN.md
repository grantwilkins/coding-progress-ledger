# Terminal-Bench M1b Task Plan

## Intent

M1b should test whether the same OpenAI adaptive path can collect visible validation-loop trajectories. It should not scale the M1 task mix.

Model arms:

- `gpt-5.4`
- `gpt-5.3-codex`
- `gpt-5.4-mini`

## Small Preflight

Run 4 tasks x 3 arms = 12 attempted runs first. Continue only if `validation_fail_observed` coverage improves materially.

| task_id | visible_validation_loop_score | role | reason |
| --- | ---: | --- | --- |
| broken-python | 7 | m1b_preflight_target | target_visible_validation_loop_or_mixed_failure_signal |
| attention-mil | 2 | m1b_preflight_target | target_visible_validation_loop_or_mixed_failure_signal |
| grid-pattern-transform | 2 | m1b_preflight_target | target_visible_validation_loop_or_mixed_failure_signal |
| csv-to-parquet | 1 | m1b_preflight_target | target_visible_validation_loop_or_mixed_failure_signal |

## Controls / Reserves

| task_id | score | reason |
| --- | ---: | --- |
| fix-permissions | 5 | retain_as_clean_or_mixed_control_after_preflight_targets |
| extract-safely | 2 | retain_as_clean_or_mixed_control_after_preflight_targets |
| aimo-airline-departures | -4 | retain_as_clean_or_mixed_control_after_preflight_targets |

## Excluded

| task_id | reason |
| --- | --- |
| classifier-debug | exclude_no_eligible_m1_setup_failures |
| adaptive-rejection-sampler | exclude_no_eligible_m1_setup_failures |
| blind-maze-explorer-algorithm | exclude_no_eligible_m1_setup_failures |
| nginx-request-logging | exclude_no_eligible_m1_setup_failures |

## Gate Additions

Keep `validation_fail_observed_coverage`. Add a separate `validation_disagreement_coverage` metric instead of weakening the visible-failure gate.

Definition: a run has `validation_disagreement` when it has a visible validation attempt or `agent_claims_done`, and the terminal verifier fails.

## Cost Guard

- per-run warning at `$0.75`
- per-run hard stop at `$1.25` unless explicitly overridden
- batch warning at `$10`

## Do Not Run Yet

This plan is a preparation artifact. Before launching M1b, inspect `reports/M1_VALIDATION_SIGNAL_AUDIT.md` and confirm the 4-task preflight target list.
