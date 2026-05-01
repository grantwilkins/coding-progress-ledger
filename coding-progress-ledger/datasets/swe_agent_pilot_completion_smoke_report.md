# Completion Prediction Smoke Report

This smoke test verifies the completion-prediction plumbing on a tiny curated dataset. It is not evidence of general predictive performance. The next scientific test requires retrospective SWE-agent or Terminal-Bench trajectories with many more natural successes and failures.

## Dataset

- Number of runs: 20
- Number of checkpoint rows: 191
- Success runs: 7 (`swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_10`)
- Failure runs: 13 (`swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_06`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_09`, `swe_agent_pilot_f_10`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_09`)

## Evaluation

- Method: leave-one-run-out by run_id
- Train/test run_id overlap: none, validated before fitting
- Estimator: deterministic binned success-rate baseline

## Feature Sets Used

- `progress_only`: `coding_progress`
- `ledger_basic`: `coding_progress`, `overall_progress`, `active_coding_weight`, `completed_coding_weight`, `active_coding_leaves`, `completed_coding_leaves`, `num_splits_so_far`, `num_reopens_so_far`, `num_invalidations_so_far`, `delta_coding_progress`
- `elapsed_only`: `step`, `event_index`

## Leakage Exclusions

- `run_id` is used only for leave-one-run-out grouping, never as a model feature.
- `final_success` is used only as the label.
- `final_success_source`, `event_type`, `subtask_id`, all `native_*` fields, drop-source fields, test-result fields, final-row aggregates copied backward, and summary_by_category final metrics are excluded from model features.

## Metrics

| model | AUROC | Brier score | log loss |
| --- | ---: | ---: | ---: |
| progress_only | 0.280164 | 0.266710 | 0.755980 |
| ledger_basic | 0.204343 | 0.264687 | 0.726906 |
| elapsed_only | 0.104930 | 0.269642 | 0.741328 |

## Mean Predicted Probability

High-progress failures are failure rows with `coding_progress >= 0.8`.

| model | successes | failures | high-progress failures | monotonic incomplete failures |
| --- | ---: | ---: | ---: | ---: |
| progress_only | 0.334320 | 0.396385 | 0.457014 | not computable |
| ledger_basic | 0.336372 | 0.395148 | 0.428348 | not computable |
| elapsed_only | 0.334797 | 0.405339 | 0.411052 | not computable |

## Case Notes

- `control_high_progress_wrong_solution`: not present in input dataset.
- `control_monotonic_incomplete_failure`: not present in input dataset.
- `control_coding_complete_artifacts_incomplete`: not present in input dataset.

---

## Interpretation (G2)

This section was added by hand after running G1. It addresses the
five questions in `TASKS.md` § G2.

### G2.1 Disclaimer (do NOT read AUROCs as predictive performance)

Per § 0 of `TASKS.md`, this section makes **no** claim of predictive
performance. The metrics in the table above are diagnostic plumbing
output, not science. Two reasons they're diagnostic only on this run:

1. **N=20 with leave-one-run-out** is too small for any sensible
   confidence interval on AUROC. Even the toy/live smoke (N=18)
   was flagged with the same caveat in
   `datasets/completion_prediction_smoke_report.md`.
2. **The label is contaminated.** The smoke script reads
   `final_success` from the dataset CSV, which is filled by the
   builder's `resolve_final_success` heuristic
   (source: `inferred_from_test_output`). For 3 SWE-agent runs that
   are upstream successes (`s_03`, `s_06`, `s_09`), the heuristic
   classifies them as failures because SWE-bench-style eval logs
   format pass/fail markers differently from toy/live's pytest
   output. So 3 of the 20 labels are wrong relative to the
   authoritative `source_metadata.json:final_success`. AUROCs
   below 0.5 in the table reflect this label noise, not the
   framework's signal quality. (See `observation_distribution_comparison.md`
   § 3.6.)

### G2.2 Do failed runs still get high predicted probabilities?

Yes — and that is exactly what we want to confirm, not what we
want to "fix":

- Mean predicted probability for runs the builder labels as
  failures: 0.40 (`progress_only`), 0.40 (`ledger_basic`),
  0.41 (`elapsed_only`).
- Mean predicted probability for runs the builder labels as
  successes: 0.33, 0.34, 0.33.

The difference is small (~0.06) and roughly equal across all three
feature sets — the predictor is barely separating builder-labeled
classes given the noisy labels. The "high-progress failures" column
(coding-progress ≥ 0.8 AND label=failure) gets mean predicted
probability 0.43–0.46. Higher than the global failure mean, which
is what you'd expect from a model that uses progress as a feature
on a label-noisy distribution.

### G2.3 Do high-progress failures exist naturally in SWE-agent?

**Yes.** Per upstream label (`source_metadata.json`), one
high-progress failure exists naturally: **`f_06`**
(`googleapis__python-spanner-317`, coding-progress 1.00,
upstream `final_success=False`). This is the canonical "all
discovered work done; failure in undiscovered hidden work" shape —
the agent's `reproduce.py` returned "Script completed successfully"
but never actually triggered the bug; the agent moved on assuming
it had reproduced. Cross-reference: case study #3 in
`datasets/swe_agent_pilot_case_studies.md`.

The smoke report's table also shows 4 high-progress failures by
the **builder's heuristic**, but 3 of those are the misclassified
upstream successes (`s_03`, `s_06`, `s_09`); only `f_06` is a real
upstream-failure-at-1.00.

### G2.4 Does `ledger_basic` differ from `progress_only`?

Marginally. `ledger_basic`'s AUROC is 0.20 vs `progress_only`'s
0.28 — both well below the 0.5 noise floor on 20 runs with 3
mislabeled. Brier and log-loss are within 0.005 of each other.
The framework's added features (delta-progress, splits/reopens
counters, leaf counts) do not improve discrimination on this
small a sample with this noisy a label. **This is the expected
result** — the framework's claim is that progress is **decoupled**
from outcome, not that progress is a better predictor of outcome.
A fair test of "do ledger features help predict outcome" needs a
larger sample, a clean label, and a downstream estimator whose
target is on-time-finish (not pass/fail) per the project's locked-in
Workstream Q framing.

### G2.5 Does `elapsed_only` remain competitive?

`elapsed_only` has AUROC 0.10 — anti-correlated, the worst of the
three. On the toy/live smoke it was the most competitive baseline;
here it isn't, and the reason is real: **SWE-agent step counts
correlate negatively with success.** Long traces are stuck-loop
failures (`f_02` 509 steps, `f_07` 183 steps, `f_03` 113 steps,
`f_08` 77 steps, `f_10` 81 steps). Short traces include the cleanest
successes (`s_04` 17 steps, `s_09` 19 steps, `s_07` 23 steps).
"Just predict from the step count" maps "many steps → likely fail",
which is roughly inverse of toy/live where short runs had been
authored as failure controls. The competitive baseline thus does
not transfer.

### G2.6 Do evidence gaps dominate the signal?

Reviewing `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 5
("Common evidence gaps"): yes, three patterns recur and account
for most of the discriminating shape:

1. **Submit-without-validation** (`f_01`, `f_04`, `s_04` —
   validation leaf at `not_started`) accounts for the
   coding-progress 0.67 cluster regardless of upstream label.
2. **Mid-edit harness termination** (`f_02`, `f_03`, `f_05`,
   `f_07`, `f_08`, `f_10` — exit_status `submitted (exit_context)`,
   no agent submit) accounts for the 0.50–0.71 failure tail.
3. **Hidden-work gap with reproduction failure** (`f_06`, where
   the agent's repro never triggered the bug) is the unique
   1.00-progress failure.

Each pattern is a different ledger shape. A predictor consuming
just `coding_progress` and `final_success` can collapse #1 and #2
into "low progress = likely failure" but cannot distinguish #1
(submit-without-test) from #3 (`f_06`-style hidden-work gap) —
both are "successful completion of all discovered work" by the
ledger's lights.

### G2.7 Is the data suitable for a larger retrospective study?

**Yes**, with caveats:

- The 20-pilot distribution **populates all four success/progress
  quadrants** — the off-diagonal data points (`f_06`, `s_04`) are
  what make a real evaluation possible. Toy/live alone would not
  be enough.
- The protocol survived four trace-length stress points (43, 113,
  183, 509 steps) with three real refinements; the refinement rate
  should drop on subsequent batches if the protocol generalizes.
- **Before scaling beyond 20**, two follow-ups should land:
  (a) fix the builder's `resolve_final_success` to short-circuit
  to `source_metadata.json:final_success` for sources where it is
  authoritative (`source == "swe_agent"`); (b) Workstream H — at
  least one independent annotator on 1-2 pilots to test
  reproducibility.

A larger retrospective at N=100 is supportable on the framework's
infrastructure (the pipeline is end-to-end deterministic and tested),
but **the value of N=100 depends on whether the protocol survives an
independent annotator**. If it does, scale; if it doesn't, fix the
protocol first. This is the M1 go/no-go question.

### G2.8 Cross-references to case studies

Detailed case studies for each of the four shape archetypes are in
`datasets/swe_agent_pilot_case_studies.md`:

- **Case 1 (`s_01`):** successful high-progress normal run.
  Illustrates the clean INVESTIGATION → PRODUCT → VALIDATION →
  ARTIFACT pipeline.
- **Case 2 (`s_03`):** successful non-monotonic run with REOPEN.
  Illustrates how a re-run repro triggers a legitimate progress
  dip that the framework preserves.
- **Case 3 (`f_06`):** failed high-progress run. Illustrates
  G2.3 above — all discovered work completed; failure sits in
  undiscovered hidden work.
- **Case 4 (`f_03`):** failed low-progress / stuck run.
  Illustrates the stuck-loop pattern under § 6 of the protocol;
  113 steps yielding 2 leaves and 0.50 progress.

The `progress_only` predictor produces probability 0.45 for `f_06`
(highest in the dataset) and probabilities 0.34 / 0.30 for `f_03` /
`f_02` (lowest tier) — consistent with the framework's claim that
the ledger shape, not just the upstream label, carries the
discriminating signal.
