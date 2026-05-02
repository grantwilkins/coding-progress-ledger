# Q3 / Q4 — Baseline evaluation summary

Leave-one-run-out by `run_id` over the W3 checkpoint table joined to
the Q1 channel-native targets. **No predictive performance claim is
made; see `datasets/RESULTS_DISCLAIMERS.md`.**

- Source W3 features: `datasets/swe_agent_estimator_checkpoints.csv`
- Source Q1 labels: `datasets/swe_agent_q_labels.csv`
- Estimator: binned base-rate baseline
- Splits: leave-one-run-out (N=20)

## Targets and label base rates

| target | n rows | positive rate |
|---|---:|---:|
| `future_progress_drop` | 191 | 0.298 |
| `product_reopened_after_completion` | 191 | 0.021 |
| `validation_exposes_new_work` | 191 | 0.021 |
| `stuck_loop_next_window` | 191 | 0.005 |
| `submit_without_validation_state` | 191 | 0.105 |

## LORO metrics by target × model

### `future_progress_drop`

| model | AUROC | Brier | log loss |
|---|---:|---:|---:|
| `always_mean` | 0.408 | 0.210 | 0.611 |
| `elapsed_only` | 0.400 | 0.217 | 0.648 |
| `progress_only` | 0.793 | 0.132 | 0.415 |
| `checkpoint_table` | 0.485 | 0.205 | 0.597 |

### `product_reopened_after_completion`

| model | AUROC | Brier | log loss |
|---|---:|---:|---:|
| `always_mean` | 0.042 | 0.021 | 0.123 |
| `elapsed_only` | 0.686 | 0.021 | 0.102 |
| `progress_only` | 0.388 | 0.021 | 0.121 |
| `checkpoint_table` | 0.618 | 0.021 | 0.105 |

### `validation_exposes_new_work`

| model | AUROC | Brier | log loss |
|---|---:|---:|---:|
| `always_mean` | 0.021 | 0.021 | 0.166 |
| `elapsed_only` | 0.249 | 0.022 | 0.167 |
| `progress_only` | 0.131 | 0.022 | 0.166 |
| `checkpoint_table` | 0.086 | 0.021 | 0.167 |

### `stuck_loop_next_window`

| model | AUROC | Brier | log loss |
|---|---:|---:|---:|
| `always_mean` | 0.011 | 0.005 | 0.042 |
| `elapsed_only` | 0.311 | 0.005 | 0.042 |
| `progress_only` | 0.447 | 0.005 | 0.042 |
| `checkpoint_table` | 0.289 | 0.005 | 0.042 |

### `submit_without_validation_state`

| model | AUROC | Brier | log loss |
|---|---:|---:|---:|
| `always_mean` | 0.000 | 0.101 | 0.381 |
| `elapsed_only` | 0.295 | 0.100 | 0.349 |
| `progress_only` | 0.504 | 0.086 | 0.325 |
| `checkpoint_table` | 0.170 | 0.100 | 0.363 |

## Reading these numbers

- AUROC "n/a" means the held-out fold had only one class; common at
  N=20 LORO for skewed targets.
- `always_mean` is the trivial base-rate baseline. A model below it on
  Brier or log loss is *worse* than predicting the train-set mean.
- The `submit_without_validation_state` target is constant per run, so
  any model that uses run-level state at step S will look near-perfect
  on rows where the agent has already committed to no-validation.
  This is a property of the data, not predictive skill.
- `progress_only` and `elapsed_only` mirror the `completion_prediction_smoke`
  feature sets; they are informational baselines, not target models.
