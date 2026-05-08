# Terminal-Bench V10 Task Selection

## Decision

Selected V10 targeted sample: 9 tasks x 2 arms.
Selected V10 pre-pilot: 4 tasks x 2 arms.

V10 keeps L gates unchanged and does not count environment setup failures as model failures.

## Pre-Pilot Tasks

| task_id | reason | env | process | visible_fail | hidden_disagreement |
| --- | --- | ---: | ---: | ---: | ---: |
| grid-pattern-transform | prepilot|process_rich|visible_validation_candidate|hidden_disagreement_candidate | 5 | 5 | 3 | 5 |
| broken-python | prepilot|process_rich|visible_validation_candidate | 5 | 5 | 5 | 1 |
| attention-mil | prepilot|process_rich|hidden_disagreement_candidate | 5 | 4 | 2 | 3 |
| csv-to-parquet | prepilot|process_rich|hidden_disagreement_candidate | 5 | 5 | 1 | 3 |

## Full Targeted Sample

| task_id | reason | env | process | visible_fail | hidden_disagreement |
| --- | --- | ---: | ---: | ---: | ---: |
| grid-pattern-transform | prepilot|process_rich|visible_validation_candidate|hidden_disagreement_candidate | 5 | 5 | 3 | 5 |
| broken-python | prepilot|process_rich|visible_validation_candidate | 5 | 5 | 5 | 1 |
| extract-safely | selected | 5 | 2 | 2 | 2 |
| attention-mil | prepilot|process_rich|hidden_disagreement_candidate | 5 | 4 | 2 | 3 |
| csv-to-parquet | prepilot|process_rich|hidden_disagreement_candidate | 5 | 5 | 1 | 3 |
| fix-permissions | process_rich | 5 | 5 | 2 | 1 |
| aimo-airline-departures | process_rich | 5 | 4 | 1 | 2 |
| create-bucket | selected | 4 | 2 | 1 | 2 |
| hello-world | selected | 4 | 2 | 1 | 1 |

## Excluded Setup-Incompatible Tasks

| task_id | setup_status | reason |
| --- | --- | --- |
| classifier-debug | exclude_prebuild_required | exclude_prebuild_required |
| adaptive-rejection-sampler | exclude_hidden_artifact_risk | exclude_hidden_artifact_risk |
| blind-maze-explorer-algorithm | exclude_hidden_artifact_risk | exclude_hidden_artifact_risk |
| nginx-request-logging | exclude_prebuild_required | exclude_prebuild_required |

## V10 Pre-Pilot Criteria

Run the 4-task pre-pilot first. Continue to 8-12 tasks only if it improves at least two of the four failed V9 gate signals: observation density, validation-fail coverage, progress-drop coverage, and high-progress/disagreement count. Do not run Workstream M.
