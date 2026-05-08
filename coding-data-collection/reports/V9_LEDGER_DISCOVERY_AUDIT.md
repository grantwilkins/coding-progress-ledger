# V9 Ledger Discovery Audit

## Summary

L-eligible runs audited: 16
progress_drop runs: 16
progress_drop_run_fraction: 1.0
largest max_coding_progress values: [0.5294117647058824, 0.5294117647058824, 0.5, 0.5, 0.5]

Finding: after the V10 ledger bridge hardening, the sidecar now observes progress drops. The bridge completes concrete successful tool rows, blocks failed tool/controller rows, and reopens prior visible work when the final verifier shows it was incomplete. This preserves ledger-side scoring semantics while exposing process dynamics that V9 previously hid.

## Evidence

| run_id | final_success | transcript_steps | max_coding_progress | progress_drop | observation_events |
| --- | --- | ---: | ---: | --- | ---: |
| aimo-airline-departures__gpt54 | True | 20 | 0.5000 | True | 8 |
| aimo-airline-departures__gpt54mini | True | 17 | 0.5000 | True | 5 |
| attention-mil__gpt54 | False | 9 | 0.5000 | True | 5 |
| attention-mil__gpt54mini | True | 17 | 0.5000 | True | 11 |
| broken-python__gpt54 | True | 18 | 0.5294 | True | 12 |
| broken-python__gpt54mini | True | 20 | 0.5000 | True | 16 |
| count-dataset-tokens__gpt54 | False | 41 | 0.5000 | True | 9 |
| count-dataset-tokens__gpt54mini | False | 32 | 0.5000 | True | 7 |
| csv-to-parquet__gpt54 | False | 41 | 0.5000 | True | 15 |
| csv-to-parquet__gpt54mini | False | 29 | 0.5000 | True | 10 |
| extract-safely__gpt54 | True | 16 | 0.5294 | True | 4 |
| extract-safely__gpt54mini | True | 16 | 0.5000 | True | 3 |
| fix-permissions__gpt54 | True | 16 | 0.5000 | True | 8 |
| fix-permissions__gpt54mini | True | 17 | 0.5000 | True | 12 |
| grid-pattern-transform__gpt54 | True | 18 | 0.5000 | True | 10 |
| grid-pattern-transform__gpt54mini | False | 26 | 0.5000 | True | 11 |

## V10 Requirement

Keep the explicit ledger bridge behavior. The remaining V10 selection issue is not progress-drop coverage; it is selecting enough tasks with visible validation failures without counting setup failures as model failures.
