# TB Live V2 Observation Loss Audit

This report quantifies signal present in `transcript.jsonl` / `verifier_output.txt` that the current ledger path does not preserve explicitly.

- runs audited: 102
- runs with verifier failure after a done claim: 20
- runs with final ledger progress = 1.0 but verifier failed: 0

## Validation-Like Shell Commands

| run_id | validation_like_shell_commands |
|---|---|
| high_progress_failure_02_partial_solution_passes_smoke__armC__f4f11fbd | 9 |
| high_progress_failure_01_subtasks_done_verifier_strict__armC__825e9208 | 8 |
| progress_drop_01_lint_then_runtime_failure__armC__3a328701 | 7 |
| progress_drop_03_lint_clean_logic_wrong__armC__bf4b2df7 | 7 |
| high_progress_failure_01_subtasks_done_verifier_strict__armC__9d5fecbd | 7 |
| validation_new_work_05_quoted_field_in_tsv__armC__3393383d | 6 |
| validation_new_work_04_tz_offset_in_log__armC__c4758717 | 6 |
| progress_drop_03_lint_clean_logic_wrong__armC__8e0cde09 | 6 |
| progress_drop_01_lint_then_runtime_failure__armC__a943cd95 | 5 |
| high_progress_failure_02_partial_solution_passes_smoke__armC__924a5795 | 5 |
| stuck_blocked_01_missing_dep_loop__armC__3898b400 | 5 |
| validation_new_work_01_test_reveals_edge_case__armB__034df633 | 4 |
| stuck_blocked_01_missing_dep_loop__armB__a0441ef5 | 4 |
| validation_new_work_02_silent_io_format_drift__armC__c8d6335f | 4 |
| validation_new_work_01_test_reveals_edge_case__armC__5847c15b | 4 |

## Nonzero Shell Commands

| run_id | nonzero_shell_commands |
|---|---|
| validation_new_work_01_test_reveals_edge_case__armB__034df633 | 1 |

## solution.sh Reads

| run_id | solution_sh_reads |
|---|---|
| high_progress_failure_01_subtasks_done_verifier_strict__armA__b235b364 | 1 |
| low_progress_success_04_missing_env_export__armB__457c3488 | 1 |
| validation_new_work_05_quoted_field_in_tsv__armA__047ccbd8 | 1 |
| validation_new_work_03_unicode_normalization__armC__564bb3c1 | 1 |
| validation_new_work_01_test_reveals_edge_case__armC__5847c15b | 1 |
| validation_new_work_01_test_reveals_edge_case__armA__aceba53f | 1 |
| stuck_blocked_01_missing_dep_loop__armC__1317d385 | 1 |
| stuck_blocked_01_missing_dep_loop__armA__94e2a40f | 1 |
| progress_drop_03_lint_clean_logic_wrong__armA__f3457d82 | 1 |
| high_progress_failure_01_subtasks_done_verifier_strict__armA__f29facec | 1 |
| progress_drop_01_lint_then_runtime_failure__armC__a943cd95 | 1 |
| progress_drop_01_lint_then_runtime_failure__armC__3a328701 | 1 |
| progress_drop_01_lint_then_runtime_failure__armA__c88194a9 | 1 |
| progress_drop_03_lint_clean_logic_wrong__armA__acedee30 | 1 |
| high_progress_failure_03_url_encode_strict__armC__b8f48a10 | 0 |

## Unexpected Product Writes

| run_id | unexpected_write_count | unexpected_paths |
|---|---|---|
| validation_new_work_05_quoted_field_in_tsv__armC__09b20117 | 3 | test.tsv, tsv_to_json.py |
| high_progress_failure_01_subtasks_done_verifier_strict__armB__9309afec | 2 | server.py |
| progress_drop_01_lint_then_runtime_failure__armC__a943cd95 | 2 | csv_summary.py, test.csv |
| high_progress_failure_01_subtasks_done_verifier_strict__armB__efb2347d | 2 | server.py |
| validation_new_work_05_quoted_field_in_tsv__armB__87f7ab5e | 1 | tsv_to_json.py |
| validation_new_work_05_quoted_field_in_tsv__armB__82ae8468 | 1 | tsv_to_json.py |
| validation_new_work_01_test_reveals_edge_case__armC__642b2075 | 1 | days_until.py |
| validation_new_work_01_test_reveals_edge_case__armB__034df633 | 1 | days_until.py |
| stuck_blocked_01_missing_dep_loop__armC__1317d385 | 1 | scrape.py |
| stuck_blocked_01_missing_dep_loop__armA__1934c642 | 1 | scrape.py |
| progress_drop_05_two_function_integration__armC__e4179be8 | 1 | orders.csv |
| progress_drop_03_lint_clean_logic_wrong__armC__bf4b2df7 | 1 | sliding_mean.py |
| progress_drop_03_lint_clean_logic_wrong__armC__8e0cde09 | 1 | sliding_mean.py |
| progress_drop_03_lint_clean_logic_wrong__armB__bfd317a6 | 1 | sliding_mean.py |
| progress_drop_03_lint_clean_logic_wrong__armB__6214dca2 | 1 | sliding_mean.py |

## Done Claims Before Verifier Failure

| run_id | termination_reason |
|---|---|
| high_progress_failure_01_subtasks_done_verifier_strict__armB__9309afec | verifier_fail |
| high_progress_failure_01_subtasks_done_verifier_strict__armB__efb2347d | verifier_fail |
| high_progress_failure_01_subtasks_done_verifier_strict__armC__825e9208 | verifier_fail |
| high_progress_failure_02_partial_solution_passes_smoke__armC__924a5795 | verifier_fail |
| progress_drop_01_lint_then_runtime_failure__armB__ad40a1a5 | verifier_fail |
| progress_drop_01_lint_then_runtime_failure__armC__3a328701 | verifier_fail |
| progress_drop_01_lint_then_runtime_failure__armC__a943cd95 | verifier_fail |
| progress_drop_03_lint_clean_logic_wrong__armB__6214dca2 | verifier_fail |
| progress_drop_03_lint_clean_logic_wrong__armB__bfd317a6 | verifier_fail |
| progress_drop_03_lint_clean_logic_wrong__armC__8e0cde09 | verifier_fail |
| progress_drop_03_lint_clean_logic_wrong__armC__bf4b2df7 | verifier_fail |
| validation_new_work_01_test_reveals_edge_case__armB__034df633 | verifier_fail |
| validation_new_work_01_test_reveals_edge_case__armB__17a7f8df | verifier_fail |
| validation_new_work_01_test_reveals_edge_case__armC__642b2075 | verifier_fail |
| validation_new_work_03_unicode_normalization__armA__4891ecb9 | verifier_fail |

## Verifier Failure Types

| failure_type | n_runs |
|---|---|
| assertion_failure | 21 |

## Final Progress = 1.0 But Verifier Failed

| none |
|---|

## Takeaways

- The current ledger preserves coarse work-frontier movement but throws away explicit validation, shell failure, oracle-read, and wrong-path signals.
- Those discarded signals are plausible drivers for high-progress failure detection and terminal success calibration.
- `observation_events.jsonl` can be backfilled from the frozen corpus, but some future improvements still require first-class live emission rather than post-hoc heuristics.
