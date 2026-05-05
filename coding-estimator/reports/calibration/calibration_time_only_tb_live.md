# Calibration — time_only on tb_live (LORO)

_Generated 2026-05-05T03:06:52+00:00._

- model: `time_only`
- source: `tb_live`
- bins: 10 equal-width on [0, 1]

## Target: `y_future_progress_drop_h5`

| metric | raw | platt | isotonic |
|---|---:|---:|---:|
| Brier | 0.133 | 0.202 | 0.161 |
| ECE | 0.117 | 0.156 | 0.168 |
| n | 23 | | |
| positive rate | 0.130 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 18 | 0.053 | 0.056 | -0.002 |
| 1 | [0.10, 0.20) | 2 | 0.144 | 0.500 | -0.356 |
| 2 | [0.20, 0.30) | 1 | 0.283 | 1.000 | -0.717 |
| 3 | [0.30, 0.40) | 1 | 0.393 | 0.000 | 0.393 |
| 4 | [0.40, 0.50) | 0 | n/a | n/a | n/a |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 0 | n/a | n/a | n/a |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 1 | 0.827 | 0.000 | 0.827 |
| 9 | [0.90, 1.00) | 0 | n/a | n/a | n/a |

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
| Brier | 0.254 | 0.321 | 0.296 |
| ECE | 0.379 | 0.262 | 0.340 |
| n | 23 | | |
| positive rate | 0.478 | | |

### Reliability table (raw)

| bin | range | count | avg_predicted | avg_observed | gap |
|---:|---|---:|---:|---:|---:|
| 0 | [0.00, 0.10) | 0 | n/a | n/a | n/a |
| 1 | [0.10, 0.20) | 0 | n/a | n/a | n/a |
| 2 | [0.20, 0.30) | 5 | 0.259 | 1.000 | -0.741 |
| 3 | [0.30, 0.40) | 9 | 0.358 | 0.111 | 0.247 |
| 4 | [0.40, 0.50) | 3 | 0.421 | 0.000 | 0.421 |
| 5 | [0.50, 0.60) | 0 | n/a | n/a | n/a |
| 6 | [0.60, 0.70) | 1 | 0.635 | 1.000 | -0.365 |
| 7 | [0.70, 0.80) | 0 | n/a | n/a | n/a |
| 8 | [0.80, 0.90) | 2 | 0.860 | 1.000 | -0.140 |
| 9 | [0.90, 1.00) | 3 | 0.962 | 0.667 | 0.295 |

