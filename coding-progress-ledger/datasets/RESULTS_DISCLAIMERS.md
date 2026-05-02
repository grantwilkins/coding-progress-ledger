# RESULTS_DISCLAIMERS — Q baseline evaluation

This document states what we **can** and **cannot** claim from the
Workstream Q baseline evaluation in
`datasets/swe_agent_q_baselines_summary.md`. Read this before quoting
any AUROC / Brier / log-loss number from that file.

## Scope of the evaluation

- **Dataset**: 191 checkpoints from 20 SWE-agent retrospective pilot
  runs (`runs/swe_agent_pilot/`). 10 success / 10 failure, balanced by
  design.
- **Targets**: five channel-native binary labels defined in
  `docs/Q_TARGETS.md`. The horizon-dependent four use a 5-step look-
  ahead; the fifth (`submit_without_validation_state`) is terminal.
- **Splits**: leave-one-run-out (LORO) by `run_id`. 20 folds.
- **Estimator**: scikit-learn LogisticRegression if importable, else a
  deterministic 5-bin base-rate baseline. The repo has zero ML deps in
  `pyproject.toml`; default execution path is the binned baseline.
- **Models**: `always_mean`, `elapsed_only`, `progress_only`,
  `checkpoint_table` — mirroring the prior smoke test's feature sets.

## What we can claim

- **The plumbing is sound.** Labels are built without future-event
  leakage from the checkpoint row's perspective; tests in
  `tests/test_q_no_leakage.py` lock this.
- **The label generator is reproducible.** Same input runs + same
  horizon → byte-identical output CSV.
- **Channel features are computable.** The five Q1 targets can be
  derived from the existing W3 checkpoint table joined to per-run
  ledgers; no schema change to `LedgerEvent` was required.
- **Class imbalance is real and quantified.** Three of the five
  targets have positive rates ≤ 2.1% on this corpus. Reports
  surface this in the "label base rates" table.

## What we cannot claim

- **Predictive performance.** The `§ 0` rules forbid this claim and
  the data does not support it. AUROC numbers in the summary are
  diagnostic, not evidence of generalization.
- **Significance**. N=20 LORO with binary targets at positive rates
  of 0.5%–10% gives held-out folds with 0–2 positives. Confidence
  intervals on AUROC at this scale span essentially the entire unit
  interval. Do not compare two AUROC values and conclude one model is
  better.
- **Cross-source generalization.** All 20 runs come from one
  retrospective sample of nebius/SWE-agent-trajectories. Results say
  nothing about how the channel behaves on other agents, scaffolds,
  models, or repos.
- **Live-instrumentation transfer.** The retrospective pilot uses the
  retrospective frontier policy; the live N=20 batch in
  `runs/swe_agent_live/` uses a different one (see
  `runs/swe_agent_live/PARITY_REPORT.md`). Targets 3
  (`validation_exposes_new_work`) and 5 (`submit_without_validation_state`)
  in particular may not transfer until the policies are reconciled.
- **Final-success prediction.** Q6 is explicitly deferred. Per the
  decoupling memory, the ledger measures process shape, not outcome.

## Reading specific numbers carefully

- **`always_mean` AUROC ≈ 0.0 on `submit_without_validation_state`.**
  This target is constant per run. Each LORO fold trains on a base
  rate computed *without* the held-out run, then predicts that base
  rate for the held-out rows. When the held-out run's label is `true`,
  the train-set rate is *lower* than the population rate; when the
  held-out run is `false`, the train-set rate is *higher*. The
  predictions are therefore systematically anti-correlated with held-
  out labels. AUROC near 0 is the expected pathology of LORO on
  per-run-constant targets, not a real predictor.
- **`progress_only` AUROC ≈ 0.79 on `future_progress_drop`.** Not
  surprising: high `coding_progress` is a structural pre-requisite
  for "drop in next 5 steps" (you can't drop from zero). Treat this
  as a *feasibility check* of the channel feature, not as a
  predictive headline.
- **`product_reopened_after_completion` and `validation_exposes_new_work`**
  each have only 4 positives across the corpus. Any AUROC reading is
  driven by 1–2 fold-level swings. Do not draw conclusions.

## When this disclaimer needs revising

Re-evaluate every line of this document if any of the following land:

- A live-N=20 checkpoint table is built (the
  `submit_without_validation` and `validation_*` semantics will shift).
- The corpus grows past N=50 and class balance is preserved.
- Final-success prediction (Q6) is attempted; that requires a fresh
  disclaimer pass on outcome-vs-channel decoupling.
- A new prediction target is added; its base rate and degeneracy
  pathologies need to be checked first.

## Files referenced

- `docs/Q_TARGETS.md` — formal target definitions.
- `scripts/build_q_labels.py` — label generation.
- `scripts/q_baselines.py` — LORO evaluation.
- `datasets/swe_agent_q_labels.csv` — per-checkpoint labels.
- `datasets/swe_agent_q_baselines.csv` — per-row predictions.
- `datasets/swe_agent_q_baselines_summary.md` — metrics tables.
- `tests/test_q_labels.py`, `tests/test_q_no_leakage.py` — invariants.
