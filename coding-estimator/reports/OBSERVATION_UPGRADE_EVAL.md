# Observation Upgrade Evaluation

Exact-task holdout on `tb_live_v2` comparing `time_only`, `ledger_basic`, and `observation_basic`.

Arm and `model_name` are descriptive metadata only. They are not part of the headline estimator feature set.

## Corpus

- runs: 102
- successes: 81
- failures: 21
- exact tasks: 25

## Per-Arm Outcome Breakdown

| arm | model | runs | successes | failures | success rate |
|---|---|---:|---:|---:|---:|
| A | claude-opus-4-7 | 34 | 33 | 1 | 0.971 |
| B | claude-sonnet-4-6 | 34 | 24 | 10 | 0.706 |
| C | claude-haiku-4-5 | 34 | 24 | 10 | 0.706 |

## Exact-Task Holdout

| target | model | feasible | AUROC | Brier | 95% CI | ECE | note |
|---|---|---:|---:|---:|---:|---:|---|
| y_future_progress_drop_h5 | ledger_basic | True | 1.000 | 0.004 | [0.002, 0.007] | 0.038 |  |
| y_future_progress_drop_h5 | observation_basic | True | 1.000 | 0.005 | [0.003, 0.008] | 0.040 |  |
| y_future_progress_drop_h5 | time_only | True | 0.832 | 0.125 | [0.082, 0.171] | 0.176 |  |
| y_success_eventual | ledger_basic | True | 0.501 | 0.178 | [0.132, 0.222] | 0.113 |  |
| y_success_eventual | observation_basic | True | 0.475 | 0.189 | [0.138, 0.238] | 0.107 |  |
| y_success_eventual | time_only | True | 0.427 | 0.196 | [0.146, 0.243] | 0.095 |  |
| y_validation_new_work_h5 | ledger_basic | False | nan | nan | n/a | nan | min_per_fold<5 (pos=0,neg=182) |
| y_validation_new_work_h5 | observation_basic | False | nan | nan | n/a | nan | min_per_fold<5 (pos=0,neg=182) |
| y_validation_new_work_h5 | time_only | False | nan | nan | n/a | nan | min_per_fold<5 (pos=0,neg=182) |

## Completion-Risk Retest

This is the X2-style re-test after adding transcript/verifier-derived observation features.

| model | AUROC | Brier | Δ Brier vs G2 | ECE |
|---|---:|---:|---:|---:|
| ledger_basic | 0.501 | 0.178 | +0.018 | 0.113 |
| observation_basic | 0.475 | 0.189 | +0.007 | 0.107 |
| time_only | 0.427 | 0.196 | +0.000 | 0.095 |

## Supplementary Success Diagnostics

Exact-task holdout on `y_success_eventual` including the existing G5 dynamics probes.

| model | AUROC | Brier | ECE | note |
|---|---:|---:|---:|---|
| g4_plus_g5 | 0.492 | 0.182 | 0.106 |  |
| g5_dynamics | 0.356 | 0.197 | 0.109 |  |
| ledger_basic | 0.501 | 0.178 | 0.113 |  |
| observation_basic | 0.475 | 0.189 | 0.107 |  |
| time_only | 0.427 | 0.196 | 0.095 |  |

## Success Slices

| slice | model | n | mean label | mean P(success) | Brier |
|---|---|---:|---:|---:|---:|
| high_progress_failure | ledger_basic | 69 | 0.000 | 0.739 | 0.580 |
| high_progress_failure | observation_basic | 69 | 0.000 | 0.746 | 0.601 |
| high_progress_failure | time_only | 69 | 0.000 | 0.759 | 0.582 |
| low_progress_success | ledger_basic | 240 | 1.000 | 0.808 | 0.037 |
| low_progress_success | observation_basic | 240 | 1.000 | 0.808 | 0.037 |
| low_progress_success | time_only | 240 | 1.000 | 0.799 | 0.042 |
| recovery_after_drop_success | ledger_basic | 93 | 1.000 | 0.582 | 0.201 |
| recovery_after_drop_success | observation_basic | 93 | 1.000 | 0.583 | 0.211 |
| recovery_after_drop_success | time_only | 93 | 1.000 | 0.653 | 0.130 |

## Interpretation

- `observation_basic` tests the instrumentation bottleneck directly by adding structured transcript-visible validation, error, and oracle-read signals on top of `ledger_basic`.
- Terminal success remains a bounded / negative headline on `tb_live_v2`: `observation_basic` does not beat `ledger_basic` on Brier (0.189 vs 0.178), and its AUROC is 0.475 vs 0.501.
- `y_validation_new_work_h5` may remain infeasible on `tb_live_v2`; that is a substrate fact, not a modeling failure.
- Verifier terminal events are emitted after the last transcript step, so they do not leak into earlier prefixes.
