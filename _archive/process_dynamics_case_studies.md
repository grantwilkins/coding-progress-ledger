# Process Dynamics Case Studies

Each case uses frozen exact-task OOF `LEDGER_BASIC` predictions.

## True Positive

- run_id: `stuck_blocked_01_missing_dep_loop__armC__1317d385`
- task_id: `stuck_blocked_01_missing_dep_loop`
- task_family: `stuck_blocked`
- arm: `C`
- checkpoint_step: 6
- predicted_probability: 0.977
- true_label: 1
- why_selected: highest-probability positive checkpoint
- interpretation: The model assigns high risk before a realized progress drop.
- figure: `process_dynamics_stuck_blocked_01_missing_dep_loop__armC__1317d385.png`

## Hardest Negative

- run_id: `high_progress_failure_01_subtasks_done_verifier_strict__armC__825e9208`
- task_id: `high_progress_failure_01_subtasks_done_verifier_strict`
- task_family: `high_progress_failure`
- arm: `C`
- checkpoint_step: 11
- predicted_probability: 0.146
- true_label: 0
- why_selected: highest-probability negative checkpoint
- interpretation: This is the hardest negative under the frozen exact-task OOF scores.
- figure: `process_dynamics_high_progress_failure_01_subtasks_done_verifier_strict__armC__825e9208.png`

## False Negative

- run_id: `progress_drop_01_lint_then_runtime_failure__armC__a943cd95`
- task_id: `progress_drop_01_lint_then_runtime_failure`
- task_family: `progress_drop`
- arm: `C`
- checkpoint_step: 6
- predicted_probability: 0.499
- true_label: 1
- why_selected: lowest-probability positive checkpoint
- interpretation: This is the hardest positive under the frozen exact-task OOF scores.
- figure: `process_dynamics_progress_drop_01_lint_then_runtime_failure__armC__a943cd95.png`

## True Negative Quiet Run

- run_id: `progress_drop_03_lint_clean_logic_wrong__armA__acedee30`
- task_id: `progress_drop_03_lint_clean_logic_wrong`
- task_family: `progress_drop`
- arm: `A`
- checkpoint_step: 1
- predicted_probability: 0.005
- true_label: 0
- why_selected: lowest-probability negative checkpoint from a run with no realized positives
- interpretation: The model stays quiet on a run with no realized progress-drop positives.
- figure: `process_dynamics_progress_drop_03_lint_clean_logic_wrong__armA__acedee30.png`

