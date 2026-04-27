# Suite Summary

All eight benchmark runs produced the required artifact bundle and passed the
coordinator rerun of their toy repo tests. Two negative controls were added
after the audit: one monotonic passing run and one intentionally incomplete
failing run.

| Run | Final progress | Subtasks | Completed | Splits | Reopens | Invalidations | Largest drop | Non-monotonic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `task_1_parser_timezone_offset` | 1.00 | 8 | 8 | 1 | 1 | 0 | 0.143 | yes |
| `task_2_cli_output_flag` | 1.00 | 9 | 7 | 1 | 1 | 1 | 0.143 | yes |
| `task_3_config_error_type` | 1.00 | 7 | 6 | 1 | 1 | 0 | 0.100 | yes |
| `task_4_csv_messy_aggregation` | 1.00 | 10 | 9 | 1 | 0 | 0 | 0.321 | yes |
| `task_5_reset_state_reducer` | 1.00 | 9 | 7 | 1 | 1 | 1 | 0.167 | yes |
| `task_6_async_stale_result` | 1.00 | 9 | 8 | 1 | 1 | 0 | 0.300 | yes |
| `task_7_refactor_validation_split` | 1.00 | 8 | 8 | 1 | 0 | 0 | 0.229 | yes |
| `task_8_package_import_failure` | 1.00 | 9 | 7 | 1 | 1 | 1 | 0.143 | yes |
| `negative_control_monotonic_one_line` | 1.00 | 4 | 4 | 0 | 0 | 0 | 0.000 | no |
| `negative_control_incomplete_budget_limited` | 0.667 | 6 | 4 | 0 | 0 | 0 | 0.250 | yes |

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
