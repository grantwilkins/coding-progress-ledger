# Estimator go/no-go gate (Workstream P)

_Generated 2026-05-05T04:27:29+00:00._

## Overall verdict: ❌ FAIL

Eight conditions from TASKS.md § Workstream P. The gate is intentionally a *no-regression* gate at v0 — see § P-future for the aspirational gate that requires CI exclusion on tb_live and ECE within plan.

Required conditions must all be `pass` for the overall verdict to be `pass`. Any `fail` on a required condition forces `fail`. Otherwise the verdict is `indeterminate`.

## Condition summary

| id | required | outcome | summary |
|---|:---:|---|---|
| `P1.a` | yes | ✅ pass | G4 wins or ties G2 on 6 of 8 (target, source) cells. |
| `P1.b` | yes | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes) |
| `P1.c` | yes | ❌ fail | Δ Brier (G2 − G4) = -0.009, 95% CI = [-0.050, +0.030]; CI INCLUDES zero |
| `P1.d` | yes | ⚠️ indeterminate | single-class y on tb_live for `y_success_eventual` |
| `P1.e` | yes | ✅ pass | no forbidden columns in the checkpoints frame |
| `P1.f` | yes | ✅ pass | audited 128 (source, target, fold) cells; 0 have run-constant pairs |
| `P1.g` | yes | ⚠️ indeterminate | D5 audit artifact not provided; Workstream M is deferred — re-evaluate this condition once D5 ships |
| `P1.h` | no | ✅ pass | winning cells span multiple targets — caveat optional |

## P1.a — G4 ties or beats G2 on at least one (target, source) under LORO

- outcome: ✅ pass
- required: yes
- summary: G4 wins or ties G2 on 6 of 8 (target, source) cells.

### Evidence

- `rows`:
  - {'source': 'swe_agent_pilot', 'target': 'y_success_eventual', 'brier_g2': 0.28256053563653377, 'brier_g4': 0.2911810432855222, 'wins_or_ties': False}
  - {'source': 'swe_agent_pilot', 'target': 'y_future_progress_drop_h5', 'brier_g2': 0.14162896462928085, 'brier_g4': 0.03933643964066982, 'wins_or_ties': True}
  - {'source': 'swe_agent_pilot', 'target': 'y_validation_new_work_h5', 'brier_g2': 0.016317137880214595, 'brier_g4': 0.008330856377028389, 'wins_or_ties': True}
  - {'source': 'swe_agent_pilot', 'target': 'y_submit_without_validation', 'brier_g2': 0.08478658305945336, 'brier_g4': 0.08012378174179058, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_success_eventual', 'brier_g2': 1.0000000000000019e-06, 'brier_g4': 1.0000000000000019e-06, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_future_progress_drop_h5', 'brier_g2': 0.13280946592119064, 'brier_g4': 0.09614838939749977, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_validation_new_work_h5', 'brier_g2': 0.25383582398789756, 'brier_g4': 0.283888109329645, 'wins_or_ties': False}
  - {'source': 'tb_live', 'target': 'y_submit_without_validation', 'brier_g2': 1e-06, 'brier_g4': 1e-06, 'wins_or_ties': True}

## P1.b — ECE_3bin (after isotonic) on tb_live LORO does not increase by > 0.05 from G2 to G4

- outcome: ⚠️ indeterminate
- required: yes
- summary: single-class y on tb_live for `y_success_eventual` (N=12 cohort is currently 12/12 successes)

### Evidence

- `g2_pos_rate`: 1.0
- `g4_pos_rate`: 1.0
- `target`: y_success_eventual

## P1.c — Combined-retrospective LORO: G4 beats G2 with 95% CI excluding zero

- outcome: ❌ fail
- required: yes
- summary: Δ Brier (G2 − G4) = -0.009, 95% CI = [-0.050, +0.030]; CI INCLUDES zero

### Evidence

- `delta_brier_ci_high`: 0.030304287809465282
- `delta_brier_ci_low`: -0.05002770406980162
- `delta_brier_point`: -0.008620507648988418
- `n_rows`: 599
- `n_runs`: 20
- `note`: hermes_pilot_h5_v2 labels not built; combined retrospective is currently swe_agent_pilot alone — plan assumed both
- `target`: y_success_eventual
- `train_sources`: ('swe_agent_pilot',)

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
- summary: audited 128 (source, target, fold) cells; 0 have run-constant pairs

### Evidence

- `audited_cells`: 128
- `audits`: []

## P1.g — D5 behavioral leakage audit (Workstream M deferred)

- outcome: ⚠️ indeterminate
- required: yes
- summary: D5 audit artifact not provided; Workstream M is deferred — re-evaluate this condition once D5 ships

### Evidence

- `d5_audit_path`: None

## P1.h — Submit-without-validation caveat

- outcome: ✅ pass
- required: no
- summary: winning cells span multiple targets — caveat optional

### Evidence

- `only_swv`: False
- `winning_cells`:
  - {'source': 'swe_agent_pilot', 'target': 'y_future_progress_drop_h5', 'brier_g2': 0.14162896462928085, 'brier_g4': 0.03933643964066982, 'wins_or_ties': True}
  - {'source': 'swe_agent_pilot', 'target': 'y_validation_new_work_h5', 'brier_g2': 0.016317137880214595, 'brier_g4': 0.008330856377028389, 'wins_or_ties': True}
  - {'source': 'swe_agent_pilot', 'target': 'y_submit_without_validation', 'brier_g2': 0.08478658305945336, 'brier_g4': 0.08012378174179058, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_success_eventual', 'brier_g2': 1.0000000000000019e-06, 'brier_g4': 1.0000000000000019e-06, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_future_progress_drop_h5', 'brier_g2': 0.13280946592119064, 'brier_g4': 0.09614838939749977, 'wins_or_ties': True}
  - {'source': 'tb_live', 'target': 'y_submit_without_validation', 'brier_g2': 1e-06, 'brier_g4': 1e-06, 'wins_or_ties': True}

