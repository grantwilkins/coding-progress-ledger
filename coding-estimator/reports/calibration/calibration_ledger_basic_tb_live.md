# Calibration — ledger_basic on tb_live (LORO)

_Generated 2026-05-05T02:50:49+00:00._

- model: `ledger_basic`
- source: `tb_live`
- bins: 10 equal-width on [0, 1]

## Target: `y_future_progress_drop_h5`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.096 | 0.170 | 0.142 |
| ECE | 0.140 | 0.153 | 0.174 |
| n | 23 | | |
| positive rate | 0.130 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 17 | 0.043 | 0.000 | 0.043 |
| 1 | [0.10, 0.20) | 1 | 0.108 | 0.000 | 0.108 |
| 2 | [0.20, 0.30) | 0 | n/a | n/a | n/a |
| 3 | [0.30, 0.40) | 2 | 0.352 | 1.000 | -0.648 |
| 4 | [0.40, 0.50) | 2 | 0.409 | 0.500 | -0.091 |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 1 | 0.902 | 0.000 | 0.902 |

## Target: `y_submit_without_validation`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.000 | n/a | n/a |
| ECE | 0.001 | n/a | n/a |
| n | 83 | | |
| positive rate | 0.000 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 83 | 0.001 | 0.000 | 0.001 |
| 1 | [0.10, 0.20) | 0 | n/a | n/a | n/a |
| 2 | [0.20, 0.30) | 0 | n/a | n/a | n/a |
| 3 | [0.30, 0.40) | 0 | n/a | n/a | n/a |
| 4 | [0.40, 0.50) | 0 | n/a | n/a | n/a |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

## Target: `y_success_eventual`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.000 | n/a | n/a |
| ECE | 0.001 | n/a | n/a |
| n | 83 | | |
| positive rate | 1.000 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 0 | n/a | n/a | n/a |
| 1 | [0.10, 0.20) | 0 | n/a | n/a | n/a |
| 2 | [0.20, 0.30) | 0 | n/a | n/a | n/a |
| 3 | [0.30, 0.40) | 0 | n/a | n/a | n/a |
| 4 | [0.40, 0.50) | 0 | n/a | n/a | n/a |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 83 | 0.999 | 1.000 | -0.001 |

## Target: `y_validation_new_work_h5`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.284 | 0.375 | 0.349 |
| ECE | 0.406 | 0.461 | 0.283 |
| n | 23 | | |
| positive rate | 0.478 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 0 | n/a | n/a | n/a |
| 1 | [0.10, 0.20) | 2 | 0.198 | 1.000 | -0.802 |
| 2 | [0.20, 0.30) | 1 | 0.209 | 1.000 | -0.791 |
| 3 | [0.30, 0.40) | 8 | 0.309 | 0.125 | 0.184 |
| 4 | [0.40, 0.50) | 4 | 0.408 | 1.000 | -0.592 |
| 5 | [0.50, 0.60) | 4 | 0.542 | 0.000 | 0.542 |
| 6 | [0.60, 0.70) | 1 | 0.606 | 1.000 | -0.394 |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 3 | 0.851 | 0.667 | 0.184 |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

