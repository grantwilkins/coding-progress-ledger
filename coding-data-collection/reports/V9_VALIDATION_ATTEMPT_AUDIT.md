# V9 Validation Attempt Audit

## Summary

L-eligible runs audited: 16
Runs with validation_attempt: 9
Runs with validation_fail_observed: 1

Finding: visible validation failure remains too sparse in V9. A small number of failed visible checks is now recognized, but most terminal failures are still hidden-verifier failures, blocked data/dependency cases, or semantic mismatches after visible smoke checks passed.

## Runs

| run_id | status | final_success | validation_attempts | validation_failures | verifier_disagreements | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| aimo-airline-departures__gpt54 | completed_success | True | 0 | 0 | 0 | no visible validation failure signal |
| aimo-airline-departures__gpt54mini | completed_success | True | 0 | 0 | 0 | no visible validation failure signal |
| attention-mil__gpt54 | completed_failure | False | 0 | 0 | 0 | no visible validation attempt; terminal failure came from verifier |
| attention-mil__gpt54mini | completed_success | True | 1 | 0 | 0 | visible validation attempt observed |
| broken-python__gpt54 | completed_success | True | 3 | 0 | 0 | visible validation attempt observed |
| broken-python__gpt54mini | completed_success | True | 3 | 2 | 0 | visible validation attempt observed |
| count-dataset-tokens__gpt54 | completed_failure | False | 1 | 0 | 1 | visible validation passed or was non-failing; hidden verifier failed |
| count-dataset-tokens__gpt54mini | completed_failure | False | 0 | 0 | 1 | no visible validation attempt; terminal failure came from verifier |
| csv-to-parquet__gpt54 | completed_failure | False | 0 | 0 | 0 | no visible validation attempt; terminal failure came from verifier |
| csv-to-parquet__gpt54mini | completed_failure | False | 0 | 0 | 0 | no visible validation attempt; terminal failure came from verifier |
| extract-safely__gpt54 | completed_success | True | 1 | 0 | 0 | visible validation attempt observed |
| extract-safely__gpt54mini | completed_success | True | 0 | 0 | 0 | no visible validation failure signal |
| fix-permissions__gpt54 | completed_success | True | 2 | 0 | 0 | visible validation attempt observed |
| fix-permissions__gpt54mini | completed_success | True | 3 | 0 | 0 | visible validation attempt observed |
| grid-pattern-transform__gpt54 | completed_success | True | 2 | 0 | 0 | visible validation attempt observed |
| grid-pattern-transform__gpt54mini | completed_failure | False | 2 | 0 | 1 | visible validation passed or was non-failing; hidden verifier failed |

## Implication For V10

V10 needs tasks with visible tests or smoke checks that can fail before the hidden verifier. If a task only allows hidden-verifier disagreement, it helps the high-progress/disagreement gate but will not satisfy `validation_fail_observed`.
