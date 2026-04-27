# Ledger Observations v0 Summary

Event rows preserve replay fidelity with one row per LedgerEvent prefix. Step rows keep the final state for each (run_id, step) and are intended for plotting and later modeling-oriented analysis.

## Totals

- Total runs: 18
- Event rows: 277
- Step rows: 198
- Successful runs: 15
- Failed runs: 3
- Unknown success runs: 0

## Category Resolution

Event rows by category resolution mode:

- `legacy_inferred`: 213
- `mixed`: 53
- `native`: 11

Step rows by category resolution mode:

- `legacy_inferred`: 149
- `mixed`: 40
- `native`: 9

Runs with native/resolved metric mismatch: `control_coding_complete_artifacts_incomplete`, `live_validation/03_evidence_audit_by_category`, `live_validation/05_active_incomplete_coding_leaves`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`.

## Non-monotonic Coding Progress

Event-level: `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_incomplete_budget_limited`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`.

Step-level: `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_incomplete_budget_limited`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`.

## Largest Event-Level Coding Drops

- `live_validation/04_docs_progress_not_success`: 0.500000 (validation)
- `task_7_refactor_validation_split`: 0.500000 (validation)
- `live_validation/01_suite_summary_weight_source`: 0.333333 (validation)
- `live_validation/02_drop_category_contributions`: 0.333333 (validation)
- `live_validation/03_evidence_audit_by_category`: 0.333333 (validation)
- `live_validation/05_active_incomplete_coding_leaves`: 0.333333 (validation)
- `task_5_reset_state_reducer`: 0.333333 (product)
- `task_3_config_error_type`: 0.250000 (product)
- `task_6_async_stale_result`: 0.200000 (product)
- `task_2_cli_output_flag`: 0.166667 (product)

## Largest Step-Level Coding Drops

- `live_validation/04_docs_progress_not_success`: 0.500000 (validation)
- `live_validation/01_suite_summary_weight_source`: 0.333333 (validation)
- `live_validation/02_drop_category_contributions`: 0.333333 (validation)
- `live_validation/03_evidence_audit_by_category`: 0.333333 (validation)
- `live_validation/05_active_incomplete_coding_leaves`: 0.333333 (validation)
- `task_5_reset_state_reducer`: 0.333333 (product)
- `task_7_refactor_validation_split`: 0.333333 (validation)
- `task_4_csv_messy_aggregation`: 0.321429 (product)
- `task_6_async_stale_result`: 0.300000 (product)
- `negative_control_incomplete_budget_limited`: 0.250000 (product)

## Largest Event-Level Overall Drops

- `task_7_refactor_validation_split`: 0.371429 (validation)
- `live_validation/01_suite_summary_weight_source`: 0.333333 (validation)
- `live_validation/02_drop_category_contributions`: 0.333333 (validation)
- `live_validation/03_evidence_audit_by_category`: 0.333333 (validation)
- `live_validation/04_docs_progress_not_success`: 0.333333 (documentation)
- `live_validation/05_active_incomplete_coding_leaves`: 0.333333 (artifact)
- `task_6_async_stale_result`: 0.200000 (product)
- `task_3_config_error_type`: 0.200000 (product)
- `task_5_reset_state_reducer`: 0.166667 (product)
- `negative_control_incomplete_budget_limited`: 0.150000 (product)

## Largest Step-Level Overall Drops

- `live_validation/01_suite_summary_weight_source`: 0.333333 (validation)
- `live_validation/02_drop_category_contributions`: 0.333333 (validation)
- `live_validation/03_evidence_audit_by_category`: 0.333333 (validation)
- `live_validation/04_docs_progress_not_success`: 0.333333 (documentation)
- `live_validation/05_active_incomplete_coding_leaves`: 0.333333 (artifact)
- `task_4_csv_messy_aggregation`: 0.321429 (product)
- `task_6_async_stale_result`: 0.300000 (product)
- `negative_control_incomplete_budget_limited`: 0.250000 (product)
- `task_7_refactor_validation_split`: 0.228571 (validation)
- `task_5_reset_state_reducer`: 0.166667 (product)

## Event vs Step

Runs where event-level and step-level largest coding drops differ: `negative_control_incomplete_budget_limited`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_6_async_stale_result`, `task_7_refactor_validation_split`.

Runs with multiple events at the same step: `control_coding_complete_artifacts_incomplete`, `control_high_progress_wrong_solution`, `control_monotonic_incomplete_failure`, `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_incomplete_budget_limited`, `negative_control_monotonic_one_line`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_6_async_stale_result`, `task_7_refactor_validation_split`.

## Success / Progress Quadrants

- Success + high progress: `control_coding_complete_artifacts_incomplete`, `live_validation/01_suite_summary_weight_source`, `live_validation/02_drop_category_contributions`, `live_validation/03_evidence_audit_by_category`, `live_validation/04_docs_progress_not_success`, `live_validation/05_active_incomplete_coding_leaves`, `negative_control_monotonic_one_line`, `task_1_parser_timezone_offset`, `task_2_cli_output_flag`, `task_3_config_error_type`, `task_4_csv_messy_aggregation`, `task_5_reset_state_reducer`, `task_6_async_stale_result`, `task_7_refactor_validation_split`, `task_8_package_import_failure`.
- Success + low progress: none
- Failure + high progress: `control_high_progress_wrong_solution`.
- Failure + low progress: `control_monotonic_incomplete_failure`, `negative_control_incomplete_budget_limited`.
- Unknown success: none

## Sanity Check Warnings

- control_coding_complete_artifacts_incomplete: native/resolved metrics differ
- live_validation/03_evidence_audit_by_category: native/resolved metrics differ
- live_validation/05_active_incomplete_coding_leaves: native/resolved metrics differ
- task_1_parser_timezone_offset: native/resolved metrics differ
- task_2_cli_output_flag: native/resolved metrics differ
- task_3_config_error_type: native/resolved metrics differ
- task_5_reset_state_reducer: native/resolved metrics differ
- task_6_async_stale_result: native/resolved metrics differ
- task_7_refactor_validation_split: native/resolved metrics differ
- task_8_package_import_failure: native/resolved metrics differ
