# M1 Validation Signal Audit

## Summary

Eligible runs audited: 23
Runs with validation attempts: 14
Runs with observed validation failures: 1
Terminal failures: 10
Verifier disagreements: 7
Runs with validation-like nonzero shells not classified as validation failure: 0

Finding: M1 mostly lacks visible validation-loop failures. The audit shows this is primarily a task/agent behavior issue rather than a broad extractor miss: terminal failures usually had no visible failed validation, or had visible checks that passed before hidden verifier failure.

## Run Audit

| run_id | task_id | arm | terminal_success | validation_attempt_count | validation_fail_observed_count | visible_validation_pass_count | agent_claims_done | verifier_disagreement | hidden_verifier_fail_type | did_visible_check_exist | did_agent_run_visible_check | did_visible_check_fail | if_no_validation_fail_why | missed_failure_like_shells |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| aimo-airline-departures__gpt53codex | aimo-airline-departures | gpt53codex | True | 0 | 0 | 0 | 1 | 0 | not_terminal_failure | true | false | false | terminal_passed_without_visible_check | 0 |
| aimo-airline-departures__gpt54 | aimo-airline-departures | gpt54 | True | 0 | 0 | 0 | 1 | 0 | not_terminal_failure | true | false | false | terminal_passed_without_visible_check | 0 |
| aimo-airline-departures__gpt54mini | aimo-airline-departures | gpt54mini | False | 0 | 0 | 0 | 1 | 1 | agent_claimed_done_then_hidden_fail | true | false | false | agent_did_not_run_check | 0 |
| attention-mil__gpt53codex | attention-mil | gpt53codex | False | 1 | 0 | 1 | 1 | 1 | visible_validation_pass_then_hidden_fail | true | true | false | visible_check_passed_hidden_failed | 0 |
| attention-mil__gpt54 | attention-mil | gpt54 | False | 0 | 0 | 0 | 0 | 0 | no_visible_validation_attempt | true | false | false | agent_did_not_run_check | 0 |
| attention-mil__gpt54mini | attention-mil | gpt54mini | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| broken-python__gpt53codex | broken-python | gpt53codex | True | 3 | 0 | 3 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| broken-python__gpt54mini | broken-python | gpt54mini | True | 3 | 1 | 2 | 1 | 0 | not_terminal_failure | true | true | true | visible_validation_failure | 0 |
| count-dataset-tokens__gpt53codex | count-dataset-tokens | gpt53codex | False | 0 | 0 | 0 | 1 | 1 | agent_claimed_done_then_hidden_fail | false | false | false | no_visible_validation_route | 0 |
| count-dataset-tokens__gpt54 | count-dataset-tokens | gpt54 | False | 0 | 0 | 0 | 1 | 1 | agent_claimed_done_then_hidden_fail | false | false | false | no_visible_validation_route | 0 |
| count-dataset-tokens__gpt54mini | count-dataset-tokens | gpt54mini | False | 0 | 0 | 0 | 1 | 1 | agent_claimed_done_then_hidden_fail | false | false | false | no_visible_validation_route | 0 |
| csv-to-parquet__gpt53codex | csv-to-parquet | gpt53codex | False | 1 | 0 | 1 | 1 | 1 | visible_validation_pass_then_hidden_fail | true | true | false | visible_check_passed_hidden_failed | 0 |
| csv-to-parquet__gpt54 | csv-to-parquet | gpt54 | False | 0 | 0 | 0 | 0 | 0 | no_visible_validation_attempt | true | false | false | agent_did_not_run_check | 0 |
| csv-to-parquet__gpt54mini | csv-to-parquet | gpt54mini | False | 0 | 0 | 0 | 0 | 0 | no_visible_validation_attempt | true | false | false | agent_did_not_run_check | 0 |
| extract-safely__gpt53codex | extract-safely | gpt53codex | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| extract-safely__gpt54 | extract-safely | gpt54 | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| extract-safely__gpt54mini | extract-safely | gpt54mini | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| fix-permissions__gpt53codex | fix-permissions | gpt53codex | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| fix-permissions__gpt54 | fix-permissions | gpt54 | True | 2 | 0 | 2 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| fix-permissions__gpt54mini | fix-permissions | gpt54mini | True | 2 | 0 | 2 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| grid-pattern-transform__gpt53codex | grid-pattern-transform | gpt53codex | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| grid-pattern-transform__gpt54 | grid-pattern-transform | gpt54 | True | 1 | 0 | 1 | 1 | 0 | not_terminal_failure | true | true | false | visible_check_passed_terminal_passed | 0 |
| grid-pattern-transform__gpt54mini | grid-pattern-transform | gpt54mini | False | 1 | 0 | 1 | 1 | 1 | visible_validation_pass_then_hidden_fail | true | true | false | visible_check_passed_hidden_failed | 0 |
