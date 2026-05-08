# tb_live_v2 shape profile

_Generated 2026-05-05T17:50:11+00:00._

`tb_live_v2` has 102 runs: 81 successes, 21 failures, 0 unresolved. The exact-task unit is 25 unique `task_id`s spread across 5 coarse shape families.

Exact-task replication histogram: 16 tasks x 3 runs, 9 tasks x 6 runs. This is why exact-task holdout is stricter than coarse shape holdout on this corpus.

## By arm

| arm | model | pass | total | pass_rate | fail |
|---|---|---:|---:|---:|---:|
| A | claude-opus-4-7 | 33 | 34 | 0.971 | 1 |
| B | claude-sonnet-4-6 | 24 | 34 | 0.706 | 10 |
| C | claude-haiku-4-5 | 24 | 34 | 0.706 | 10 |

## By coarse family

| family | pass | total | pass_rate | fail |
|---|---:|---:|---:|---:|
| validation_new_work | 12 | 21 | 0.571 | 9 |
| progress_drop | 14 | 21 | 0.667 | 7 |
| high_progress_failure | 16 | 21 | 0.762 | 5 |
| low_progress_success | 21 | 21 | 1.000 | 0 |
| stuck_blocked | 18 | 18 | 1.000 | 0 |

## Failure-concentrated exact tasks

| task_id | pass | total | pass_rate | fail |
|---|---:|---:|---:|---:|
| high_progress_failure_01_subtasks_done_verifier_strict | 2 | 6 | 0.333 | 4 |
| progress_drop_03_lint_clean_logic_wrong | 2 | 6 | 0.333 | 4 |
| validation_new_work_05_quoted_field_in_tsv | 2 | 6 | 0.333 | 4 |
| progress_drop_01_lint_then_runtime_failure | 3 | 6 | 0.500 | 3 |
| validation_new_work_01_test_reveals_edge_case | 3 | 6 | 0.500 | 3 |
| validation_new_work_03_unicode_normalization | 1 | 3 | 0.333 | 2 |
| high_progress_failure_02_partial_solution_passes_smoke | 5 | 6 | 0.833 | 1 |

## Caveat

The seeded workspace included `solution.sh` during collection. Any terminal-success analysis on this corpus should therefore be treated as optimistic rather than deployment-grade.

