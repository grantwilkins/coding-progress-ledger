# tb_live_v2 estimator evaluation — exact-task holdout

_Generated 2026-05-05T17:50:11+00:00._

Headline metrics are exact-task holdout (`task_id` held out across all arms). Overlap-heavy run-level splits are reported separately as easier auxiliary diagnostics.

## Headline: exact-task holdout process dynamics

`tb_live_v2` contains 102 runs across 25 exact tasks and 5 coarse shape families. Exact-task group histogram: 16 tasks x 3 runs, 9 tasks x 6 runs.
At the current base rates, `y_future_progress_drop_h5` is evaluable under exact-task holdout; `y_validation_new_work_h5` remains below the per-fold positive-count budget and is reported as `n/a`.

| split | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| exact-task-holdout | y_future_progress_drop_h5 | ledger_basic | 59 | 59 | 213 | 0.146 | 1.000 | 0.004 | [0.002, 0.007] |  |
| exact-task-holdout | y_future_progress_drop_h5 | time_only | 59 | 59 | 213 | 0.146 | 0.832 | 0.125 | [0.082, 0.171] |  |
| exact-task-holdout | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=182) |
| exact-task-holdout | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=182) |

## Auxiliary only: overlap-heavy splits

These splits can place task X / arm A in train and task X / arm B in test, so they are easier than the exact-task claim and are not the headline result.

| split | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| holdout-overlap | y_future_progress_drop_h5 | ledger_basic | 47 | 12 | 60 | 0.200 | 1.000 | 0.006 | [0.002, 0.009] |  |
| holdout-overlap | y_future_progress_drop_h5 | time_only | 47 | 12 | 60 | 0.200 | 0.794 | 0.197 | [0.073, 0.305] |  |
| holdout-overlap | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=201) |
| holdout-overlap | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=201) |
| loro-overlap | y_future_progress_drop_h5 | ledger_basic | 59 | 59 | 213 | 0.146 | 1.000 | 0.004 | [0.002, 0.006] |  |
| loro-overlap | y_future_progress_drop_h5 | time_only | 59 | 59 | 213 | 0.146 | 0.826 | 0.124 | [0.082, 0.170] |  |
| loro-overlap | y_validation_new_work_h5 | ledger_basic | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=201) |
| loro-overlap | y_validation_new_work_h5 | time_only | n/a | n/a | n/a | n/a | n/a | n/a | n/a | min_per_fold<5 (pos=0,neg=201) |

## Secondary only: terminal success

Read this section conservatively:
- Arm concentration is strong: A 33/34, B 24/34, C 24/34.
- The corpus is ceiling-limited in several shape families, so terminal success is not a balanced substrate.
- `solution.sh` was present in the seeded workspace during collection, so observed success is optimistic.

| split | target | model | n_train | n_test | n_ckpts | pos_rate | AUROC | Brier | Brier 95% CI | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| holdout-overlap | y_success_eventual | ledger_basic | 82 | 20 | 155 | 0.748 | 0.689 | 0.174 | [0.087, 0.262] |  |
| holdout-overlap | y_success_eventual | time_only | 82 | 20 | 155 | 0.748 | 0.516 | 0.204 | [0.118, 0.294] |  |
| loro-overlap | y_success_eventual | ledger_basic | 102 | 102 | 703 | 0.755 | 0.585 | 0.170 | [0.126, 0.214] |  |
| loro-overlap | y_success_eventual | time_only | 102 | 102 | 703 | 0.755 | 0.505 | 0.190 | [0.142, 0.235] |  |
| exact-task-holdout | y_success_eventual | ledger_basic | 102 | 102 | 703 | 0.755 | 0.501 | 0.178 | [0.132, 0.222] |  |
| exact-task-holdout | y_success_eventual | time_only | 102 | 102 | 703 | 0.755 | 0.427 | 0.196 | [0.146, 0.243] |  |

