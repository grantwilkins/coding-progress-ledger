# v0 findings — coding-estimator

_Generated 2026-05-05. Recentered scientific summary, written to be the
single artifact a reader who is not in the codebase consults to learn
what v0 has and has not established. Companion to the gate report
(`reports/ESTIMATOR_GO_NO_GO.md`) and the sign-off
(`reports/sign_off_ledger_basic_v0.1.md`)._

## TL;DR

Prefix-only ledger features predict **near-future progress dynamics**
on retrospective coding-agent traces. They do **not** improve
**terminal success** prediction over elapsed time at this sample size.
The observation channel measures work-frontier dynamics before it
becomes a reliable completion estimator. That is the publishable
boundary.

`not_safe_for_control = true`. The artifact is shipped as a
**measurement result**, not as an estimator ready for any consumer.

## Primary v0 claim

> **Prefix-only ledger features predict near-future progress
> dynamics.**

Evidence: per-source LORO on `swe_agent_pilot` (20 runs, 499 labeled
checkpoints), `y_future_progress_drop_h5`:

| model           | Brier | AUROC | Δ vs G2 (Brier) |
|-----------------|------:|------:|----------------:|
| G2 time_only    | 0.142 | 0.626 |       —         |
| G4 ledger_basic | 0.039 | 0.977 |     **−0.102**  |
| G5 dynamics     | 0.078 | 0.897 |       −0.064    |
| G4 + G5         | 0.042 | 0.973 |       −0.100    |

A ten-point Brier improvement over the elapsed-time baseline at AUROC
0.977 is large for v0. G5 features alone also clear G2 by ~6 Brier
points, but they do **not** stack additively with G4 — `g4_plus_g5`
(0.042) is essentially `G4` (0.039), and on `tb_live` G5 alone is
slightly worse than G4 alone (0.124 vs 0.096). Read this as: G4
features dominate the dynamics signal; G5 carries an independent but
smaller share of the same information rather than complementary
information.

The dynamics result corroborates on `tb_live` (10 runs, 23 labeled
checkpoints, smaller absolute Briers) where G4 still beats G2 by
0.037 Brier.

`y_validation_new_work_h5` shows the same pattern at smaller absolute
Briers (very rare positives on swe_agent_pilot; richer base rate on
tb_live) — ledger features beat or match elapsed time on every cell.

Source: `reports/g5/g5_eval.md`.

## Secondary v0 claim — negative result

> **Prefix-only ledger features do not improve terminal success
> prediction over elapsed time on small retrospective data.**

Evidence: per-source LORO on `swe_agent_pilot`, `y_success_eventual`:

| model           | Brier | AUROC | Δ vs G2 (Brier) |
|-----------------|------:|------:|----------------:|
| G2 time_only    | 0.283 | 0.281 |       —         |
| G4 ledger_basic | 0.291 | 0.410 |       +0.009    |
| G5 dynamics     | 0.272 | 0.385 |       −0.010    |
| G4 + G5         | 0.292 | 0.411 |       +0.010    |

G4 is *worse* than G2 on `y_success_eventual` by 0.009. G5's
dynamics features alone do slightly better than G2 (Δ −0.010 on
`y_success_eventual`) but well within noise; combining (G4+G5) sits
back at G4 (+0.010). The strongest scientific gate (O7 in
`coding_estimator/eval/failure_modes.py`) demands a +0.02 Brier lift
of G4 over G2; none of the ledger configurations reaches that on
`y_success_eventual` on the largest retrospective source. Read this
as: even with the dynamics feature group added, prefix-only ledger
state does not unlock terminal-success prediction at this N.

`tb_live` is uninformative for this target: 12/12 successes (single-
class y).

Source: `reports/g5/g5_eval.md`, `reports/failure_modes/failure_modes.md`.

## Interpretation

The estimator's job description in the project mission is "a belief
layer over live coding-progress ledgers" that "consumes prefix-only
ledger features and outputs calibrated probabilities over successful
completion by future horizons, remaining time, and near-future
progress dynamics."

The v0 measurement says:

- The **near-future progress-dynamics** half of that mission is
  achievable today on retrospective data — by a wide margin on the
  largest source.
- The **successful-completion-by-horizon** half is dominated by a
  one-feature elapsed-time baseline at this N.

A ledger watching a coding agent measures **what the agent has
visibly done so far**. That signal is local: it predicts what the
agent will do next better than it predicts whether the run will
ultimately succeed. Terminal success is downstream and confounded by
hidden requirements (test harness specifics, failure-mode coverage,
unannotated retrospective traces). The current data plus the current
feature set are not yet enough to bridge that gap.

This is not a project failure. It is a project boundary. The
observation channel is doing what it should — it sees process
shape — and the v0 gate's strict +0.02 threshold has correctly told
us we cannot promise more.

## What the pipeline now guarantees

- **Run-disjoint splits** (LORO / LTFO / LOSO / holdout) with a
  hard-fail on row-level cross-fold leakage.
- **Forbidden-column audit** with exact + prefix + suffix matching
  (`coding_estimator/leakage/guard.py`); P1.e zero hits.
- **Run-constancy guard** for joint (feature, target) pairs in every
  G4 training fold (`coding_estimator/leakage/run_constancy.py`); P1.f
  zero hits across 128 audited cells.
- **Behavioral leakage audit (D5)** structured artifact with prefix-
  truncation invariance, label-shuffle test, run-constancy, structural
  checks (`reports/d5_audit.json`). P1.g enforces the schema; bare
  `{clean: true}` is rejected.
- **Calibrated reports** with run-disjoint k-fold isotonic
  recalibration (`coding_estimator/calibration/`); raw and post-
  isotonic ECE on every (model, source, target) cell.
- **Bootstrap CIs** are run-level (not row-level) on every Brier
  reported.
- **Model card validates against a JSON schema**
  (`schemas/model_card_schema.json`) before any bundle hits disk.
- **Versioning policy** is published (`docs/VERSIONING.md`); pre-v1
  estimators are immutable and carry `not_safe_for_control = true`.

## What the data can currently answer

- SWE-agent process-dynamics prediction (progress drop, validation
  new-work) at 20 runs / 499 labeled checkpoints.
- SWE-agent terminal-success prediction (negative result vs elapsed
  time, also at 20 runs).
- TB-12 process-dynamics prediction at 10 runs / 23 labeled
  checkpoints (corroborative; small CIs).

## What the data cannot answer

- TB success prediction (12/12 successes; single-class y).
- Combined retrospective gate (Hermes labels are upstream-unannotated;
  see `reports/HERMES_LABEL_DIAGNOSIS.md`).
- Live failure risk (zero TB live failures collected; zero
  swe_agent_live runs).
- Across-source generalization for either target at v0 N.

## Consequences for the next phase

In priority order (also reflected in `TASKS.md`):

1. **Annotate the 30 hermes_pilot_h5_v2 runs upstream.** Highest
   leverage. Unblocks P1.c (the *one* CI-exclusion gate) on ~50 runs.
   No code changes in this repo. Diagnosis in
   `reports/HERMES_LABEL_DIAGNOSIS.md`.

2. **Collect `tb_live_v2` with outcome diversity.** ≥ 30 runs with
   ≥ 10 failures, real wall-clock, same sidecar/ledger protocol, do
   not tune the agent to make tasks succeed. Without failures the TB
   cohort is uninformative for any success-prediction gate; with
   failures it can corroborate the dynamics finding on live data.

3. **Defer scheduler / online-inference / semantic-feature work.**
   Per `TASKS.md` § Workstream M, Q, R — these remain explicitly
   deferred until P passes on better data.

4. **Run the human-baseline experiment.**
   `scripts/run_human_baseline.py` is on disk; one human reads the 6
   midpoint-prefix prompts and writes their probabilities into
   `reports/human_baseline/human_predictions.csv`. The comparison
   answers whether the ledger is *readable* as a belief signal. If
   G4 matches the human on dynamics, the channel carries the signal;
   if G4 trails the human, the model is weak.

5. **Re-run the full pipeline after (1) and (2).** Every artifact
   regenerates from `scripts/`:

   ```
   scripts/run_baselines.py        # G ladder
   scripts/run_model_ladder.py     # I ladder
   scripts/run_calibration.py      # J reports
   scripts/run_g5_eval.py          # G5 dynamics evaluation
   scripts/run_tb_live_eval.py     # K1 + K3
   scripts/run_retro_to_live.py    # L3
   scripts/run_failure_modes.py    # O
   scripts/run_d5_audit.py         # D5 audit JSON
   scripts/run_go_no_go.py         # P1
   scripts/run_sign_off.py         # P2 + P3
   ```

   None takes more than ~30 seconds on the current dataset.

## What we do NOT recommend

- **Loosening the +0.02 O7 threshold** to get a pass. The strict
  threshold is informative; the failure is real signal, not noise to
  be tuned away.
- **Adding semantic / text features** before fixing data defects.
  Those would help prediction but muddy the publishable claim that
  ledger-native features carry process-dynamics signal.
- **Building any controller, scheduler, or online inference
  surface.** The repo's mission keeps these explicitly out of scope
  until P passes.

## Where to read next

- `reports/ESTIMATOR_GO_NO_GO.md` — full P1.a–h gate evidence.
- `reports/sign_off_ledger_basic_v0.1.md` — the v0 sign-off.
- `reports/g5/g5_eval.md` — per-target G2 / G4 / G5 / G4+G5
  comparison.
- `reports/d5_audit.md` and `reports/d5_audit.json` — D5 structured
  audit.
- `reports/HERMES_LABEL_DIAGNOSIS.md` — why P1.c is indeterminate
  and how to fix it.
- `reports/REVIEWER_BRIEFING.md` — fuller architectural briefing
  for an independent reviewer.
- `reports/NOT_READY_FOR_SCHEDULING.md` — prioritized BLOCKING /
  DATA / AUDIT actions for the next phase.

## Two-line summary

The v0 estimator is doing what a good measurement system should do:
revealing that current ledger features and current data support
**process-dynamics estimation** but not yet **completion-risk
estimation**. The next phase is data work, not more model polish.
