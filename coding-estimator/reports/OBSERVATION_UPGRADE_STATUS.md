# Observation Upgrade Status

Date: 2026-05-05

## Scope completed

This pass implemented the remaining non-human-baseline work centered on
Workstreams `Z1-Z5` plus the `X2` completion-risk re-test on the frozen
`tb_live_v2` corpus.

Shipped:

- `scripts/audit_tb_live_v2_observation_loss.py`
- `reports/TB_LIVE_V2_OBSERVATION_LOSS.md`
- `docs/TB_LIVE_V3_OBSERVATION_SCHEMA.md`
- `reports/TB_LIVE_V3_INSTRUMENTATION_PLAN.md`
- `coding_estimator/runner/observation_events.py`
- `scripts/backfill_observation_events.py`
- `coding_estimator/checkpoints/features/observation.py`
- `coding_estimator/baselines/observation_basic.py`
- `coding_estimator/eval/observation_upgrade.py`
- `scripts/run_observation_eval.py`
- `reports/OBSERVATION_UPGRADE_EVAL.md`
- `reports/observation_upgrade_metrics.csv`

Supporting updates:

- `driver.py` now emits additive `observation_events.jsonl` during run finalization.
- `build.py` / feature registry now attach prefix-safe observation features to checkpoints.
- `docs/ESTIMATOR_FEATURE_GROUPS.md` now documents the observation feature family.
- Frozen `tb_live_v2` runs were backfilled with `observation_events.jsonl`.

## Where we are

The observation-channel upgrade is now implemented enough to test the
instrumentation hypothesis on `tb_live_v2` without changing ledger
semantics.

Important boundary:

- Hidden verifier expectations are used only for the post-hoc audit.
- `OBSERVATION_BASIC` itself now uses transcript-visible signals only
  (validation, errors, oracle reads, done-related fields where visible),
  not verifier-derived path matches.

Current exact-task-holdout results:

- `y_future_progress_drop_h5`
  - `ledger_basic`: AUROC `1.000`, Brier `0.004`
  - `observation_basic`: AUROC `1.000`, Brier `0.005`
  - `time_only`: AUROC `0.832`, Brier `0.125`
- `y_success_eventual`
  - `ledger_basic`: AUROC `0.501`, Brier `0.178`
  - `observation_basic`: AUROC `0.475`, Brier `0.189`
  - `time_only`: AUROC `0.427`, Brier `0.196`

Supplementary exact-task terminal-success diagnostics:

- `g5_dynamics`: AUROC `0.356`, Brier `0.197`
- `g4_plus_g5`: AUROC `0.492`, Brier `0.182`

Interpretation:

- The positive process-dynamics result remains intact.
- The observation backfill adds some ranking signal for terminal
  success over `time_only`, but after removing hidden-verifier path
  leakage it does **not** beat `ledger_basic` on AUROC or Brier.
- `y_validation_new_work_h5` remains unevaluable on `tb_live_v2`
  because the live substrate still does not emit the required
  validation-transition pattern.

## How we are doing

Good:

- The repo now has an additive observation layer rather than only a
  roadmap entry for it.
- The new layer is prefix-safe at the checkpoint level.
- The frozen live corpus can now be audited and re-evaluated with
  structured transcript/verifier signals.
- The core result is scientifically clearer: the bottleneck is still
  mostly observation quality, not model family.

Not good enough yet:

- Backfilled transcript heuristics are still too weak/noisy to turn
  `tb_live_v2` into a strong completion-risk dataset.
- The strongest arm remains near ceiling, and failure diversity is still
  concentrated in weaker arms / a small subset of tasks.
- `tb_live_v2` still cannot support `validation_new_work` as a live
  headline target.

## Recommended next step

Move to real `tb_live_v3` collection using the shipped
`observation_events.jsonl` schema rather than collecting more
same-logging `tb_live_v2` runs.

Concretely:

1. Emit `observation_events.jsonl` natively during collection.
2. Prefer tasks where validation attempts, path mistakes, and verifier
   disagreement happen naturally and visibly.
3. Re-run the same `time_only` vs `ledger_basic` vs `observation_basic`
   exact-task comparison on that new substrate before adding richer
   model classes.
