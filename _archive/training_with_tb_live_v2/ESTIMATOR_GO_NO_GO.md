# Estimator go/no-go gate (Workstream P)

_Generated 2026-05-05T17:45:34+00:00._

## Overall verdict: ⚠️ INDETERMINATE

**Blocked by:** INDETERMINATE on `P1.b`, `P1.c`, `P1.d`.

**v0 framing (recentered).** The primary v0 headline is **process-dynamics prediction** (`y_future_progress_drop_h5`, `y_validation_new_work_h5`). Terminal success (`y_success_eventual`) is reported as a secondary / negative result: ledger features do not yet beat elapsed time on it at this N. See `reports/V0_FINDINGS.md` for the publishable story; this gate keeps its original P1.a–h structure so no-regression on success is still measured.

Eight conditions from TASKS.md § Workstream P. The gate is intentionally a *no-regression* gate at v0 — see § P-future for the aspirational gate that requires CI exclusion on tb_live and ECE within plan.

Required conditions must all be `pass` for the overall verdict to be `pass`. Any `fail` on a required condition forces `fail`. Otherwise the verdict is `indeterminate`.

## Condition summary

| id | required | outcome | summary |
|---|:---:|---|---|
| `P1.a` | yes | ✅ pass | G4 wins or ties G2 on 4 of 4 (target, source) cells. |
| `P1.b` | yes | ⚠️ indeterminate | no LORO predictions on tb_live for y_success_eventual |
| `P1.c` | yes | ⚠️ indeterminate | no retrospective sources present |
| `P1.d` | yes | ⚠️ indeterminate | missing within-source or LOSO predictions on tb_live |
| `P1.e` | yes | ✅ pass | no forbidden columns in the checkpoints frame |
| `P1.f` | yes | ✅ pass | audited 408 (source, target, fold) cells; 0 have run-constant pairs; skipped 0 cells (0 no labels, 0 empty join) |
| `P1.g` | yes | ✅ pass | D5 audit clean (102 runs, 703 checkpoints; 0 findings) |
| `P1.h` | yes | ✅ pass | winning cells span multiple targets — caveat does not apply |

## P1.a — G4 ties or beats G2 on at least one (target, source) under LORO

- outcome: ✅ pass
- required: yes
- summary: G4 wins or ties G2 on 4 of 4 (target, source) cells.

### Evidence

| source | target | Brier G2 | Brier G4 | wins or ties |
|---|---|---:|---:|:---:|
| tb_live_v2 | y_future_progress_drop_h5 | 0.124 | 0.004 | ✅ |
| tb_live_v2 | y_submit_without_validation | 0.000 | 0.000 | ✅ |
| tb_live_v2 | y_success_eventual | 0.190 | 0.170 | ✅ |
| tb_live_v2 | y_validation_new_work_h5 | 0.000 | 0.000 | ✅ |

## P1.b — ECE_3bin (after isotonic) on tb_live LORO does not increase by > 0.05 from G2 to G4

- outcome: ⚠️ indeterminate
- required: yes
- summary: no LORO predictions on tb_live for y_success_eventual

### Evidence

- `target`: y_success_eventual

## P1.c — Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero

- outcome: ⚠️ indeterminate
- required: yes
- summary: no retrospective sources present

### Evidence

- `target`: y_success_eventual

## P1.d — LOSO->tb_live Brier ≤ within-source LORO Brier + 0.05

- outcome: ⚠️ indeterminate
- required: yes
- summary: missing within-source or LOSO predictions on tb_live

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
- summary: audited 408 (source, target, fold) cells; 0 have run-constant pairs; skipped 0 cells (0 no labels, 0 empty join)

### Evidence

- `audited_cells`: 408
- `audits`: []
- `skipped_empty_join`: 0
- `skipped_no_labels`: 0

## P1.g — D5 behavioral leakage audit (Workstream M deferred)

- outcome: ✅ pass
- required: yes
- summary: D5 audit clean (102 runs, 703 checkpoints; 0 findings)

### Evidence

- `d5_audit_path`: reports/training_with_tb_live_v2/d5_audit.json
- `n_checkpoints_audited`: 703
- `n_findings`: 0
- `n_runs_audited`: 102
- `schema_version`: 1.1.0

## P1.h — Submit-without-validation caveat

- outcome: ✅ pass
- required: yes
- summary: winning cells span multiple targets — caveat does not apply

### Evidence

- `only_swv`: False
- `winning_cells`:
  - brier_g2=0.18952276448086244, brier_g4=0.16984129040618468, source=tb_live_v2, target=y_success_eventual, wins_or_ties=True
  - brier_g2=0.12434824102067528, brier_g4=0.0038012112810254417, source=tb_live_v2, target=y_future_progress_drop_h5, wins_or_ties=True
  - brier_g2=1.0000000000000002e-06, brier_g4=1.0000000000000002e-06, source=tb_live_v2, target=y_validation_new_work_h5, wins_or_ties=True
  - brier_g2=1.0000000000000002e-06, brier_g4=1.0000000000000002e-06, source=tb_live_v2, target=y_submit_without_validation, wins_or_ties=True

