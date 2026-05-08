# Failure-mode tests (Workstream O)

_Generated 2026-05-05T17:45:33+00:00._

Source-only failure-mode checks for the tb_live_v2 exact-task evaluation pass. O5 is expected to be indeterminate on a single-source slice.

## Headline outcomes

| test | outcome | metric | value | threshold | note |
|---|---|---|---:|---:|---|
| O1 | **fail** | median_p_success_on_high_progress_failures | 0.806 | 0.700 |  |
| O5 | **indeterminate** | loso_brier_delta_with_source_task | n/a | 0.020 | LOSO target source `tb_live_v2` not present |
| O7 | **fail** | brier_g2_minus_brier_g4 | 0.020 | 0.020 |  |

## O1 — progress-overconfidence

Slice: rows where `final_success == 0` AND `coding_progress >= 0.8`. Gate: median `P(success)` on slice must be < 0.7.

- `high_progress_threshold`: 0.8
- `n_failed_runs`: 21
- `n_rows`: 68

## O5 — source-leakage

LOSO -> `tb_live` on `y_success_eventual`. Compare G4 vs G4+source_task; `|Brier_plus - Brier_g4| < 0.02` ⇒ pass.


## O7 — timeout-bias

Per source under LORO on `y_success_eventual`. Pass iff `Brier_G2 - Brier_G4 >= 0.02` (the ledger adds information beyond elapsed time). Indeterminate when y is single-class on the source.

| source | outcome | Brier G2 | Brier G4 | Δ (G2 - G4) | n |
|---|---|---:|---:|---:|---:|
| tb_live_v2 | fail | 0.190 | 0.170 | 0.020 | 703 |

