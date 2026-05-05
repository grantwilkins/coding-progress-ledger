# Calibration — time_only on tb_live (LOSO)

_Generated 2026-05-05T02:50:50+00:00._

- model: `time_only`
- source: `loso->tb_live`
- bins: 10 equal-width on [0, 1]

## Target: `y_future_progress_drop_h5`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.112 | 0.127 | 0.107 |
| ECE | 0.023 | 0.126 | 0.135 |
| n | 23 | | |
| positive rate | 0.130 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 0 | n/a | n/a | n/a |
| 1 | [0.10, 0.20) | 23 | 0.107 | 0.130 | -0.023 |
| 2 | [0.20, 0.30) | 0 | n/a | n/a | n/a |
| 3 | [0.30, 0.40) | 0 | n/a | n/a | n/a |
| 4 | [0.40, 0.50) | 0 | n/a | n/a | n/a |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

## Target: `y_submit_without_validation`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.037 | n/a | n/a |
| ECE | 0.188 | n/a | n/a |
| n | 83 | | |
| positive rate | 0.000 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 0 | n/a | n/a | n/a |
| 1 | [0.10, 0.20) | 47 | 0.165 | 0.000 | 0.165 |
| 2 | [0.20, 0.30) | 36 | 0.219 | 0.000 | 0.219 |
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
| Brier | 0.198 | n/a | n/a |
| ECE | 0.445 | n/a | n/a |
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
| 5 | [0.50, 0.60) | 83 | 0.555 | 1.000 | -0.445 |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

## Target: `y_validation_new_work_h5`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.475 | 0.335 | 0.288 |
| ECE | 0.475 | 0.435 | 0.281 |
| n | 23 | | |
| positive rate | 0.478 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 23 | 0.003 | 0.478 | -0.475 |
| 1 | [0.10, 0.20) | 0 | n/a | n/a | n/a |
| 2 | [0.20, 0.30) | 0 | n/a | n/a | n/a |
| 3 | [0.30, 0.40) | 0 | n/a | n/a | n/a |
| 4 | [0.40, 0.50) | 0 | n/a | n/a | n/a |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 0 | n/a | n/a | n/a |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

