# M1 Task Outcome Profile

| task_id | attempted | eligible | rejected | success | terminal_failure | validation_attempt_runs | validation_fail_runs | validation_pass_hidden_fail_runs | verifier_disagreement_runs | profile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| adaptive-rejection-sampler | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | setup_failure_exclude |
| aimo-airline-departures | 3 | 3 | 0 | 2 | 1 | 0 | 0 | 0 | 1 | hidden_or_terminal_failure_without_visible_check |
| attention-mil | 3 | 3 | 0 | 1 | 2 | 2 | 0 | 1 | 1 | visible_pass_then_hidden_fail |
| blind-maze-explorer-algorithm | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | setup_failure_exclude |
| broken-python | 3 | 2 | 1 | 2 | 0 | 2 | 1 | 0 | 0 | visible_validation_failure_present |
| classifier-debug | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | setup_failure_exclude |
| count-dataset-tokens | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 3 | hidden_or_terminal_failure_without_visible_check |
| csv-to-parquet | 3 | 3 | 0 | 0 | 3 | 1 | 0 | 1 | 1 | visible_pass_then_hidden_fail |
| extract-safely | 3 | 3 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | clean_success_control |
| fix-permissions | 3 | 3 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | clean_success_control |
| grid-pattern-transform | 3 | 3 | 0 | 2 | 1 | 3 | 0 | 1 | 1 | visible_pass_then_hidden_fail |
| nginx-request-logging | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | setup_failure_exclude |

## Interpretation

`classifier-debug`, `adaptive-rejection-sampler`, `blind-maze-explorer-algorithm`, and `nginx-request-logging` should be excluded from M1b unless their setup path is fixed.

`broken-python`, `grid-pattern-transform`, `attention-mil`, and `csv-to-parquet` are the best small M1b preflight targets because they combine visible-check potential with either observed validation activity or mixed/failed terminal outcomes.

`extract-safely`, `fix-permissions`, and `aimo-airline-departures` are useful controls or reserves, but should not dominate M1b because they are not the missing validation-loop signal.
