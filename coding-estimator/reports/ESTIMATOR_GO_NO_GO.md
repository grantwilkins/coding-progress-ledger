# Estimator go/no-go gate (Workstream P)

_Generated 2026-05-05T04:58:38+00:00._

## Overall verdict: ❌ FAIL

**Blocked by:** FAIL on `P1.g`; INDETERMINATE on `P1.b`, `P1.c`, `P1.d`.

Eight conditions from TASKS.md § Workstream P. The gate is intentionally a *no-regression* gate at v0 — see § P-future for the aspirational gate that requires CI exclusion on tb_live and ECE within plan.

Required conditions must all be `pass` for the overall verdict to be `pass`. Any `fail` on a required condition forces `fail`. Otherwise the verdict is `indeterminate`.

## Condition summary

| id | required | outcome | summary |
|---|:---:|---|---|
| `P1.a` | yes | ✅ pass | G4 wins or ties G2 on 6 of 8 (target, source) cells. |
| `P1.b` | yes | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes) |
| `P1.c` | yes | ⚠️ indeterminate | hermes_pilot_h5_v2 labels not built into `datasets/labels_all.parquet` — combined retrospective is not testable as the plan defines it |
| `P1.d` | yes | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` |
| `P1.e` | yes | ✅ pass | no forbidden columns in the checkpoints frame |
| `P1.f` | yes | ✅ pass | audited 128 (source, target, fold) cells; 0 have run-constant pairs; skipped 120 cells (0 no labels, 120 empty join) |
| `P1.g` | yes | ❌ fail | D5 audit reports 1 findings or `clean: false` |
| `P1.h` | yes | ✅ pass | winning cells span multiple targets — caveat does not apply |

## P1.a — G4 ties or beats G2 on at least one (target, source) under LORO

- outcome: ✅ pass
- required: yes
- summary: G4 wins or ties G2 on 6 of 8 (target, source) cells.

### Evidence

| source | target | Brier G2 | Brier G4 | wins or ties |
|---|---|---:|---:|:---:|
| swe_agent_pilot | y_future_progress_drop_h5 | 0.142 | 0.039 | ✅ |
| swe_agent_pilot | y_submit_without_validation | 0.085 | 0.080 | ✅ |
| swe_agent_pilot | y_success_eventual | 0.283 | 0.291 | ❌ |
| swe_agent_pilot | y_validation_new_work_h5 | 0.016 | 0.008 | ✅ |
| tb_live | y_future_progress_drop_h5 | 0.133 | 0.096 | ✅ |
| tb_live | y_submit_without_validation | 0.000 | 0.000 | ✅ |
| tb_live | y_success_eventual | 0.000 | 0.000 | ✅ |
| tb_live | y_validation_new_work_h5 | 0.254 | 0.284 | ❌ |

## P1.b — ECE_3bin (after isotonic) on tb_live LORO does not increase by > 0.05 from G2 to G4

- outcome: ⚠️ indeterminate
- required: yes
- summary: single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes)

### Evidence

- `g2_pos_rate`: 1.0
- `g4_pos_rate`: 1.0
- `target`: y_success_eventual

## P1.c — Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero

- outcome: ⚠️ indeterminate
- required: yes
- summary: hermes_pilot_h5_v2 labels not built into `datasets/labels_all.parquet` — combined retrospective is not testable as the plan defines it

### Evidence

- `missing_source_labels`: hermes_pilot_h5_v2
- `note`: swe_agent_pilot-only result is available in the Workstream H baselines; do NOT promote that to the combined-retrospective gate
- `target`: y_success_eventual

## P1.d — LOSO->tb_live Brier ≤ within-source LORO Brier + 0.05

- outcome: ⚠️ indeterminate
- required: yes
- summary: single-class y on tb_live for `y_success_eventual`

### Evidence

- `target`: y_success_eventual

## P1.e — Forbidden-column audit: zero hits

- outcome: ✅ pass
- required: yes
- summary: no forbidden columns in the checkpoints frame

### Evidence

- `forbidden_exact_count`: 10
- `forbidden_prefix_count`: 6
- `forbidden_suffix_count`: 2
- `hits`: []

## P1.f — G4 training-fold run-constancy: zero joint (feature, target) pairs

- outcome: ✅ pass
- required: yes
- summary: audited 128 (source, target, fold) cells; 0 have run-constant pairs; skipped 120 cells (0 no labels, 120 empty join)

### Evidence

- `audited_cells`: 128
- `audits`: []
- `skipped_empty_join`: 120
- `skipped_no_labels`: 0

## P1.g — D5 behavioral leakage audit (Workstream M deferred)

- outcome: ❌ fail
- required: yes
- summary: D5 audit reports 1 findings or `clean: false`

### Evidence

- `d5_audit_path`: reports/d5_audit.json
- `n_checkpoints_audited`: 1578
- `n_findings`: 1
- `n_runs_audited`: 62
- `schema_version`: 1.0.0

## P1.h — Submit-without-validation caveat

- outcome: ✅ pass
- required: yes
- summary: winning cells span multiple targets — caveat does not apply

### Evidence

- `only_swv`: False
- `winning_cells`:
  - brier_g2=0.14162896462928085, brier_g4=0.03933643964066982, source=swe_agent_pilot, target=y_future_progress_drop_h5, wins_or_ties=True
  - brier_g2=0.016317137880214595, brier_g4=0.008330856377028389, source=swe_agent_pilot, target=y_validation_new_work_h5, wins_or_ties=True
  - brier_g2=0.08478658305945336, brier_g4=0.08012378174179058, source=swe_agent_pilot, target=y_submit_without_validation, wins_or_ties=True
  - brier_g2=1.0000000000000019e-06, brier_g4=1.0000000000000019e-06, source=tb_live, target=y_success_eventual, wins_or_ties=True
  - brier_g2=0.13280946592119064, brier_g4=0.09614838939749977, source=tb_live, target=y_future_progress_drop_h5, wins_or_ties=True
  - brier_g2=1e-06, brier_g4=1e-06, source=tb_live, target=y_submit_without_validation, wins_or_ties=True

