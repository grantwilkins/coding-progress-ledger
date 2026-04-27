# Suite Summary

All eight benchmark runs produced the required artifact bundle and passed the
coordinator rerun of their toy repo tests. Two negative controls were added
after the audit, and three hardening controls now exercise final coding/overall
divergence, monotonic incomplete failure, and high-progress failed work.

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

## Run Notes

`task_1_parser_timezone_offset`: The curve dropped when compact offsets forced
validation beyond the first colon-format fix. The ledger was useful for making
old behavior preservation and negative compact offsets visible.

`task_2_cli_output_flag`: Progress dropped after file output was fixed but
`--output -` needed to preserve stdout behavior. The invalidated path captured
an abandoned too-simple output abstraction.

`task_3_config_error_type`: The first fixed `ValueError` site made progress
look high, then the second raise site reopened the exception-consistency work.
The ledger made the two-site nature of the bug inspectable.

`task_4_csv_messy_aggregation`: This had the largest drop. Clean aggregation
passed first, then messy rows added whitespace normalization, blank amount
handling, and deterministic ordering.

`task_5_reset_state_reducer`: Resetting the visible count was insufficient; the
submitted/derived state and validation error state had to be cleared too. The
ledger was useful for showing why reset behavior expanded.

`task_6_async_stale_result`: A loading-state check and a self-contained async
test-harness fix expanded the work after the initial stale-result framing. The
ledger was useful, though test-harness work felt slightly awkward as product
progress.

`task_7_refactor_validation_split`: The main non-monotonic event was the split
of vague validation into targeted unit tests, broader regression tests, and API
compatibility checks. This is the cleanest example of split-driven denominator
growth.

`task_8_package_import_failure`: Fixing package execution revealed a direct
script-style compatibility decision. The ledger captured the reopened import
work and the invalidated import-only approach.

## Coordinator Validation

Coordinator rerun results:

- Task 1: `9 passed`
- Task 2: `3 passed`
- Task 3: `3 passed`
- Task 4: `3 passed`
- Task 5: `4 pass`
- Task 6: `2 passed`
- Task 7: `9 passed`
- Task 8: module execution, direct tests, and package import command passed

The suite demonstrates non-monotonic progress over mutable discovered work
without changing ledger scoring semantics or adding LLM calls to the ledger
package.
