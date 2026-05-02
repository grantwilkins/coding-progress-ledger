# Shape labels report for `runs/hermes_pilot`

Shape tags are **audit labels**, not final model targets. They are derived
from ledger structure (status, category, REOPEN events) plus evidence-text
citations. Anchors per W4: `high_progress_failure` fires at
coding_progress >= 0.7; `no_validation_frontier`
means no VALIDATION subtask was attempted.

Runs labeled: **5**

## Tag counts

| Tag | Count |
|---|---:|
| `high_progress_failure` | 0 |
| `low_progress_success` | 0 |
| `stuck_loop` | 2 |
| `submit_without_validation` | 0 |
| `no_validation_frontier` | 0 |
| `validation_induced_reopen` | 0 |
| `scope_discovery_after_high_progress` | 0 |
| `hidden_work_gap` | 0 |
| `nonmonotone_recovery` | 0 |
| `clean_success` | 0 |

## Per-run tags

| run_id | success | coding | clean | tags |
|---|:---:|---:|:---:|---|
| `hermes_pilot_01` |  | 1.000 |  | — |
| `hermes_pilot_02` |  | 1.000 |  | — |
| `hermes_pilot_03` |  | 1.000 |  | stuck_loop |
| `hermes_pilot_04` |  | 1.000 |  | — |
| `hermes_pilot_05` |  | 1.000 |  | stuck_loop |

## Caveats

- Labels are derived from ledger fields plus evidence text.
- `hidden_work_gap` requires explicit annotator-cited phrases (e.g. 
  "DID NOT trigger", "hidden-work"); silent gaps will not be flagged.
- `validation_induced_reopen` keys on reason strings naming repro / 
  pytest / Traceback / re-run / still emits.
- These tags are intended as the audit surface that distinguishes a
  progress=1.0 success from a progress=1.0 submit-without-test; they
  are not yet training labels for any predictive model.
