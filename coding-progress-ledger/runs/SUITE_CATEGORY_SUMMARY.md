# Suite Category Summary

Coding progress includes product, validation, and investigation leaves. Artifact, documentation, and environment leaves are excluded from coding progress. The underlying scoring rule is unchanged: progress is completed active leaf weight divided by active leaf weight after normal leaf reduction.

| Run | final_coding_progress | final_overall_progress | coding_complete_weight_final | coding_active_weight_final | overall_complete_weight_final | overall_active_weight_final | active_coding_leaves_final | completed_coding_leaves_final | active_overall_leaves_final | completed_overall_leaves_final | historical_subtasks_created | coding_largest_drop | overall_largest_drop | largest_coding_drop_source | largest_overall_drop_source | excluded_active_weight_final | excluded_completed_weight_final | coding_nonmonotonic | final_success | final_success_source | evidence_audit_status | weak_completion_evidence_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |
| `task_1_parser_timezone_offset` | 1.000 | 1.000 | 6.000 | 6.000 | 7.000 | 7.000 | 6 | 6 | 7 | 7 | 8 | 0.150 | 0.143 | validation | artifact | 1.000 | 1.000 | yes | yes | summary.test_status | weak | 3 |
| `task_2_cli_output_flag` | 1.000 | 1.000 | 6.000 | 6.000 | 7.000 | 7.000 | 6 | 6 | 7 | 7 | 9 | 0.167 | 0.143 | validation | product | 1.000 | 1.000 | yes | yes | summary.test_status | strong | 0 |
| `task_3_config_error_type` | 1.000 | 1.000 | 5.000 | 5.000 | 6.000 | 6.000 | 5 | 5 | 6 | 6 | 7 | 0.150 | 0.100 | product | product | 1.000 | 1.000 | yes | yes | summary.test_status | weak | 3 |
| `task_4_csv_messy_aggregation` | 1.000 | 1.000 | 9.000 | 9.000 | 9.000 | 9.000 | 9 | 9 | 9 | 9 | 10 | 0.321 | 0.321 | product | product | 0.000 | 0.000 | yes | yes | summary.test_status | weak | 1 |
| `task_5_reset_state_reducer` | 1.000 | 1.000 | 4.000 | 4.000 | 7.000 | 7.000 | 4 | 4 | 7 | 7 | 9 | 0.333 | 0.167 | product | product | 3.000 | 3.000 | yes | yes | summary.test_status | weak | 1 |
| `task_6_async_stale_result` | 1.000 | 1.000 | 6.000 | 6.000 | 8.000 | 8.000 | 6 | 6 | 8 | 8 | 9 | 0.300 | 0.300 | product | product | 2.000 | 2.000 | yes | yes | summary.test_status | weak | 1 |
| `task_7_refactor_validation_split` | 1.000 | 1.000 | 6.000 | 6.000 | 7.000 | 7.000 | 6 | 6 | 7 | 7 | 8 | 0.333 | 0.229 | validation | validation | 1.000 | 1.000 | yes | yes | summary.test_status | strong | 0 |
| `task_8_package_import_failure` | 1.000 | 1.000 | 6.000 | 6.000 | 7.000 | 7.000 | 6 | 6 | 7 | 7 | 9 | 0.167 | 0.143 | mixed | mixed | 1.000 | 1.000 | yes | yes | summary.test_status | strong | 0 |
| `control_coding_complete_artifacts_incomplete` | 1.000 | 0.667 | 2.000 | 2.000 | 2.000 | 3.000 | 2 | 2 | 3 | 2 | 3 | 0.000 | 0.000 | none | none | 1.000 | 0.000 | no | yes | summary.final_success | strong | 0 |
| `control_high_progress_wrong_solution` | 0.857 | 0.857 | 3.000 | 3.500 | 3.000 | 3.500 | 4 | 3 | 4 | 3 | 4 | 0.000 | 0.000 | none | none | 0.000 | 0.000 | no | no | summary.final_success | strong | 0 |
| `control_monotonic_incomplete_failure` | 0.600 | 0.600 | 3.000 | 5.000 | 3.000 | 5.000 | 5 | 3 | 5 | 3 | 5 | 0.000 | 0.000 | none | none | 0.000 | 0.000 | no | no | summary.final_success | strong | 0 |
| `negative_control_incomplete_budget_limited` | 0.667 | 0.667 | 4.000 | 6.000 | 4.000 | 6.000 | 6 | 4 | 6 | 4 | 6 | 0.250 | 0.250 | product | product | 0.000 | 0.000 | yes | no | summary.test_status | strong | 0 |
| `negative_control_monotonic_one_line` | 1.000 | 1.000 | 4.000 | 4.000 | 4.000 | 4.000 | 4 | 4 | 4 | 4 | 4 | 0.000 | 0.000 | none | none | 0.000 | 0.000 | no | yes | summary.test_status | strong | 0 |

## Non-monotonicity After Filtering

Coding progress remains non-monotonic for `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`, `negative_control_incomplete_budget_limited`.

## Bookkeeping-driven Largest Drops

The largest overall drop source is excluded artifact/documentation/environment work for `task_1_parser_timezone_offset`.

## Final Progress Divergence

Final overall progress and final coding progress diverge for `control_coding_complete_artifacts_incomplete`.

Runs with excluded active weight at the final step: `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`, `control_coding_complete_artifacts_incomplete`.

## Failure Modes

Monotonic incomplete failures `control_high_progress_wrong_solution`, `control_monotonic_incomplete_failure`.

High-progress failed runs `control_high_progress_wrong_solution`.

## Evidence Audit

Runs with weak product/validation completion evidence `task_1_parser_timezone_offset`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`.

## Audit Resolution

Category filtering separates coding progress from run-management work without changing ledger scoring semantics or rewriting historical ledgers. The added controls make the remaining distinctions explicit: progress is not success, coding progress can differ from overall progress, failures can be monotonic, and weak evidence remains an audit finding rather than a replay failure.
