# Sign-off — ledger_basic_v0.1

_Generated 2026-05-05T04:43:11+00:00._

## Headline verdict: ⚠️ INDETERMINATE

- gate report: `reports/ESTIMATOR_GO_NO_GO.md`
- model bundle: `models/ledger_basic_v0.1/`
- model_card.json: `models/ledger_basic_v0.1/model_card.json` (validates against `schemas/model_card_schema.json`)
- not_safe_for_control: **True**

## Required gate conditions

| id | outcome | summary |
|---|---|---|
| `P1.a` | ✅ pass | G4 wins or ties G2 on 6 of 8 (target, source) cells. |
| `P1.b` | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes) |
| `P1.c` | ⚠️ indeterminate | hermes_pilot_h5_v2 labels not built into `datasets/labels_all.parquet` — combined retrospective is not testable as the plan defines it |
| `P1.d` | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` |
| `P1.e` | ✅ pass | no forbidden columns in the checkpoints frame |
| `P1.f` | ✅ pass | audited 128 (source, target, fold) cells; 0 have run-constant pairs; skipped 120 cells (0 no labels, 120 empty join) |
| `P1.g` | ⚠️ indeterminate | D5 audit artifact not provided; Workstream M is deferred — re-evaluate this condition once D5 ships with required fields ['schema_version', 'n_runs_audited', 'n_checkpoints_audited', 'findings', 'clean'] |
| `P1.h` | ✅ pass | winning cells span multiple targets — caveat does not apply |

## Failure-mode tests (O1, O5, O7)

| test | outcome | metric | value | threshold |
|---|---|---|---:|---:|
| `O1` | ✅ pass | median_p_success_on_high_progress_failures | 0.578 | 0.700 |
| `O5` | ⚠️ indeterminate | loso_brier_delta_with_source_task | n/a | 0.020 |
| `O7 (hermes_pilot_h5_v2)` | ⚠️ indeterminate | brier_g2_minus_brier_g4 | n/a | 0.020 |
| `O7 (swe_agent_pilot)` | ❌ fail | brier_g2_minus_brier_g4 | -0.009 | 0.020 |
| `O7 (tb_live)` | ⚠️ indeterminate | brier_g2_minus_brier_g4 | n/a | 0.020 |

## Known limits

- not_safe_for_control = true: required gate `P1.b` is `indeterminate`; required gate `P1.c` is `indeterminate`; required gate `P1.d` is `indeterminate`; required gate `P1.g` is `indeterminate`; O5 outcome is `indeterminate`; O7 fails on source `swe_agent_pilot`
- O7 timeout-bias FAIL on ['swe_agent_pilot']: ledger does not add ≥ 0.02 Brier over time-only on these sources
- required gate conditions ['P1.b', 'P1.c', 'P1.d', 'P1.g'] are indeterminate at current N — see ESTIMATOR_GO_NO_GO.md for details
- raw probabilities are un-recalibrated unless the consumer applies isotonic recalibration from `calibration.json`
- retrospective sources carry outcome-aware annotation caveats
- `y_submit_without_validation` is run-constant within a run; any non-trivial AUROC at non-terminal t is a data property, not skill

## Recommendations for next collection

- **BLOCKING (O7)** the v0 ledger features do not carry decision-relevant signal beyond elapsed time on ['swe_agent_pilot']. Cheapest next experiment: add the deferred dynamics group (G5) and re-run O7.
- **DATA (P1.b)** tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- **DATA (P1.c)** build hermes_pilot_h5_v2 labels into `datasets/labels_all.parquet` so the combined retrospective (~50 runs) is testable as the plan intended.
- **DATA (P1.d)** tb_live cohort is 12/12 successes — collect at least 5 tb_live failures before this gate is even testable.
- **AUDIT (P1.g)** ship the D5 behavioral leakage audit artifact (Workstream M deferred → D5 substitute is still required).
