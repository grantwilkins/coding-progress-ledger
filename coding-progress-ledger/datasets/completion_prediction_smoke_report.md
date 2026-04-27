# Completion Prediction Smoke Report

This smoke test verifies the completion-prediction plumbing on a tiny curated dataset. It is not evidence of general predictive performance. The next scientific test requires retrospective SWE-agent or Terminal-Bench trajectories with many more natural successes and failures.

## Dataset

- Number of runs: 18
- Number of checkpoint rows: 198
- Success runs: 15 (`control_coding_complete_artifacts_incomplete`, `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_monotonic_one_line`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`)
- Failure runs: 3 (`control_high_progress_wrong_solution`, `control_monotonic_incomplete_failure`, `negative_control_incomplete_budget_limited`)

## Evaluation

- Method: leave-one-run-out by run_id
- Train/test run_id overlap: none, validated before fitting
- Estimator: deterministic binned success-rate baseline

## Feature Sets Used

- `progress_only`: `coding_progress`
- `ledger_basic`: `coding_progress`, `overall_progress`, `active_coding_weight`, `completed_coding_weight`, `active_coding_leaves`, `completed_coding_leaves`, `num_splits_so_far`, `num_reopens_so_far`, `num_invalidations_so_far`, `delta_coding_progress`
- `elapsed_only`: `step`, `event_index`

## Leakage Exclusions

- `run_id` is used only for leave-one-run-out grouping, never as a model feature.
- `final_success` is used only as the label.
- `final_success_source`, `event_type`, `subtask_id`, all `native_*` fields, drop-source fields, test-result fields, final-row aggregates copied backward, and summary_by_category final metrics are excluded from model features.

## Metrics

| model | AUROC | Brier score | log loss |
| --- | ---: | ---: | ---: |
| progress_only | 0.393890 | 0.084065 | 0.331231 |
| ledger_basic | 0.321904 | 0.083494 | 0.326147 |
| elapsed_only | 0.544036 | 0.082207 | 0.296114 |

## Mean Predicted Probability

High-progress failures are failure rows with `coding_progress >= 0.8`.

| model | successes | failures | high-progress failures | monotonic incomplete failures |
| --- | ---: | ---: | ---: | ---: |
| progress_only | 0.908485 | 0.921863 | 0.999000 | 0.911023 |
| ledger_basic | 0.908102 | 0.921778 | 0.906667 | 0.914371 |
| elapsed_only | 0.912584 | 0.908737 | 0.923077 | 0.903358 |

## Case Notes

- `control_high_progress_wrong_solution`: final_success=false, checkpoint_rows=5, final coding_progress=0.857143, final overall_progress=0.857143, model mean predicted probabilities: progress_only=0.922944, ledger_basic=0.914371, elapsed_only=0.899767.
- `control_monotonic_incomplete_failure`: final_success=false, checkpoint_rows=5, final coding_progress=0.600000, final overall_progress=0.600000, model mean predicted probabilities: progress_only=0.911023, ledger_basic=0.914371, elapsed_only=0.903358.
- `control_coding_complete_artifacts_incomplete`: final_success=true, checkpoint_rows=4, final coding_progress=1.000000, final overall_progress=0.666667, model mean predicted probabilities: progress_only=0.894923, ledger_basic=0.878655, elapsed_only=0.833333.
