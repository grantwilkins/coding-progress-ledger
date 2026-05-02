# Shape labels report for `runs/hermes_pilot_h5`

Shape tags are **audit labels**, not final model targets. They are derived
from ledger structure (status, category, REOPEN events) plus evidence-text
citations. Anchors per W4: `high_progress_failure` fires at
coding_progress >= 0.7; `no_validation_frontier`
means no VALIDATION subtask was attempted.

Runs labeled: **30**

## Tag counts

| Tag | Count |
|---|---:|
| `high_progress_failure` | 0 |
| `low_progress_success` | 0 |
| `stuck_loop` | 22 |
| `submit_without_validation` | 0 |
| `no_validation_frontier` | 6 |
| `validation_induced_reopen` | 0 |
| `scope_discovery_after_high_progress` | 0 |
| `hidden_work_gap` | 0 |
| `nonmonotone_recovery` | 0 |
| `clean_success` | 0 |

## Per-run tags

| run_id | success | coding | clean | tags |
|---|:---:|---:|:---:|---|
| `hermes_pilot_h5_001` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_002` |  | 1.000 |  | — |
| `hermes_pilot_h5_003` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_004` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_005` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_006` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_007` |  | 1.000 |  | — |
| `hermes_pilot_h5_008` |  | 0.500 |  | stuck_loop |
| `hermes_pilot_h5_009` |  | 0.833 |  | stuck_loop |
| `hermes_pilot_h5_010` |  | 0.833 |  | stuck_loop |
| `hermes_pilot_h5_011` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_012` |  | 0.800 |  | stuck_loop |
| `hermes_pilot_h5_013` |  | 0.545 |  | stuck_loop |
| `hermes_pilot_h5_014` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_015` |  | 0.700 |  | stuck_loop |
| `hermes_pilot_h5_016` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_017` |  | 0.800 |  | stuck_loop |
| `hermes_pilot_h5_018` |  | 0.500 |  | stuck_loop |
| `hermes_pilot_h5_019` |  | 0.800 |  | stuck_loop |
| `hermes_pilot_h5_020` |  | 1.000 |  | stuck_loop |
| `hermes_pilot_h5_021` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_022` |  | 0.500 |  | stuck_loop |
| `hermes_pilot_h5_023` |  | 0.818 |  | stuck_loop |
| `hermes_pilot_h5_024` |  | 0.500 |  | stuck_loop |
| `hermes_pilot_h5_025` |  | 0.600 |  | stuck_loop |
| `hermes_pilot_h5_026` |  | 1.000 |  | no_validation_frontier |
| `hermes_pilot_h5_027` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_028` |  | 0.778 |  | stuck_loop |
| `hermes_pilot_h5_029` |  | 0.750 |  | stuck_loop |
| `hermes_pilot_h5_030` |  | 0.750 |  | stuck_loop |

## Caveats

- Labels are derived from ledger fields plus evidence text.
- `hidden_work_gap` requires explicit annotator-cited phrases (e.g. 
  "DID NOT trigger", "hidden-work"); silent gaps will not be flagged.
- `validation_induced_reopen` keys on reason strings naming repro / 
  pytest / Traceback / re-run / still emits.
- These tags are intended as the audit surface that distinguishes a
  progress=1.0 success from a progress=1.0 submit-without-test; they
  are not yet training labels for any predictive model.
