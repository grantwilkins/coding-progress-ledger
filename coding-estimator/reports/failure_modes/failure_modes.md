# Failure-mode tests (Workstream O)

_Generated 2026-05-05T04:21:10+00:00._

Adversarial tests against the v0 G4 (`ledger_basic`) estimator. O1, O5, O7 only — O2/O3/O4/O6 are deferred at current N (see TASKS.md § Workstream O).

## Headline outcomes

| test | outcome | metric | value | threshold | note |
|---|---|---|---:|---:|---|
| O1 | **pass** | median_p_success_on_high_progress_failures | 0.578 | 0.700 |  |
| O5 | **indeterminate** | loso_brier_delta_with_source_task | n/a | 0.020 | no source_task numeric columns present in checkpoints frame — comparison is identity, test is vacuous |
| O7 | **indeterminate** | brier_g2_minus_brier_g4 | n/a | 0.020 | hermes_pilot_h5_v2: no LORO predictions |
| O7 | **fail** | brier_g2_minus_brier_g4 | -0.009 | 0.020 |  |
| O7 | **indeterminate** | brier_g2_minus_brier_g4 | n/a | 0.020 | tb_live: single-class y on `y_success_eventual` |

## O1 — progress-overconfidence

Slice: rows where `final_success == 0` AND `coding_progress >= 0.8`. Gate: median `P(success)` on slice must be < 0.7.

- `high_progress_threshold`: 0.8
- `n_failed_runs`: 10
- `n_rows`: 45

## O5 — source-leakage

LOSO -> `tb_live` on `y_success_eventual`. Compare G4 vs G4+source_task; `|Brier_plus - Brier_g4| < 0.02` ⇒ pass.

- `source_task_columns_present`: 

## O7 — timeout-bias

Per source under LORO on `y_success_eventual`. Pass iff `Brier_G2 - Brier_G4 >= 0.02` (the ledger adds information beyond elapsed time). Indeterminate when y is single-class on the source.

| source | outcome | Brier G2 | Brier G4 | Δ (G2 - G4) | n |
|---|---|---:|---:|---:|---:|
| hermes_pilot_h5_v2 | indeterminate | n/a | n/a | n/a | ? |
| swe_agent_pilot | fail | 0.283 | 0.291 | -0.009 | 599 |
| tb_live | indeterminate | n/a | n/a | n/a | ? |

