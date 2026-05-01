# Shape labels report for `runs/swe_agent_pilot`

Shape tags are **audit labels**, not final model targets. They are derived
from ledger structure (status, category, REOPEN events) plus evidence-text
citations. Anchors per W4: `high_progress_failure` fires at
coding_progress >= 0.7; `no_validation_frontier`
means no VALIDATION subtask was attempted.

Runs labeled: **20**

## Tag counts

| Tag | Count |
|---|---:|
| `high_progress_failure` | 3 |
| `low_progress_success` | 1 |
| `stuck_loop` | 6 |
| `submit_without_validation` | 4 |
| `no_validation_frontier` | 3 |
| `validation_induced_reopen` | 3 |
| `scope_discovery_after_high_progress` | 1 |
| `hidden_work_gap` | 2 |
| `nonmonotone_recovery` | 2 |
| `clean_success` | 9 |

## Per-run tags

| run_id | success | coding | clean | tags |
|---|:---:|---:|:---:|---|
| `swe_agent_pilot_f_01` | ✗ | 0.667 |  | submit_without_validation |
| `swe_agent_pilot_f_02` | ✗ | 0.500 |  | no_validation_frontier, stuck_loop |
| `swe_agent_pilot_f_03` | ✗ | 0.500 |  | no_validation_frontier, stuck_loop |
| `swe_agent_pilot_f_04` | ✗ | 0.667 |  | submit_without_validation |
| `swe_agent_pilot_f_05` | ✗ | 0.600 |  | stuck_loop |
| `swe_agent_pilot_f_06` | ✗ | 1.000 |  | hidden_work_gap, high_progress_failure |
| `swe_agent_pilot_f_07` | ✗ | 0.667 |  | stuck_loop |
| `swe_agent_pilot_f_08` | ✗ | 0.714 |  | hidden_work_gap, high_progress_failure, stuck_loop |
| `swe_agent_pilot_f_09` | ✗ | 0.800 |  | high_progress_failure, submit_without_validation, validation_induced_reopen |
| `swe_agent_pilot_f_10` | ✗ | 0.667 |  | no_validation_frontier, stuck_loop |
| `swe_agent_pilot_s_01` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_02` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_03` | ✓ | 1.000 | ✓ | nonmonotone_recovery, scope_discovery_after_high_progress, validation_induced_reopen |
| `swe_agent_pilot_s_04` | ✓ | 0.667 |  | low_progress_success, submit_without_validation |
| `swe_agent_pilot_s_05` | ✓ | 1.000 | ✓ | nonmonotone_recovery, validation_induced_reopen |
| `swe_agent_pilot_s_06` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_07` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_08` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_09` | ✓ | 1.000 | ✓ | — |
| `swe_agent_pilot_s_10` | ✓ | 1.000 | ✓ | — |

## Caveats

- Labels are derived from ledger fields plus evidence text.
- `hidden_work_gap` requires explicit annotator-cited phrases (e.g. 
  "DID NOT trigger", "hidden-work"); silent gaps will not be flagged.
- `validation_induced_reopen` keys on reason strings naming repro / 
  pytest / Traceback / re-run / still emits.
- These tags are intended as the audit surface that distinguishes a
  progress=1.0 success from a progress=1.0 submit-without-test; they
  are not yet training labels for any predictive model.
