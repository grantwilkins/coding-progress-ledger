# Completion Prediction Smoke Report

This smoke test verifies the completion-prediction plumbing on a tiny curated dataset. It is not evidence of general predictive performance. The next scientific test requires retrospective SWE-agent or Terminal-Bench trajectories with many more natural successes and failures.

## Dataset

- Number of runs: 20
- Number of checkpoint rows: 191
- Success runs: 10 (`swe_agent_pilot_s_01`, `swe_agent_pilot_s_02`, `swe_agent_pilot_s_03`, `swe_agent_pilot_s_04`, `swe_agent_pilot_s_05`, `swe_agent_pilot_s_06`, `swe_agent_pilot_s_07`, `swe_agent_pilot_s_08`, `swe_agent_pilot_s_09`, `swe_agent_pilot_s_10`)
- Failure runs: 10 (`swe_agent_pilot_f_01`, `swe_agent_pilot_f_02`, `swe_agent_pilot_f_03`, `swe_agent_pilot_f_04`, `swe_agent_pilot_f_05`, `swe_agent_pilot_f_06`, `swe_agent_pilot_f_07`, `swe_agent_pilot_f_08`, `swe_agent_pilot_f_09`, `swe_agent_pilot_f_10`)

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
| progress_only | 0.374226 | 0.272617 | 0.766278 |
| ledger_basic | 0.382239 | 0.278789 | 0.755488 |
| elapsed_only | 0.182803 | 0.288142 | 0.772518 |

## Mean Predicted Probability

High-progress failures are failure rows with `coding_progress >= 0.8`.

| model | successes | failures | high-progress failures | monotonic incomplete failures |
| --- | ---: | ---: | ---: | ---: |
| progress_only | 0.532926 | 0.559820 | 0.694647 | not computable |
| ledger_basic | 0.523796 | 0.572848 | 0.632753 | not computable |
| elapsed_only | 0.521591 | 0.592735 | 0.609669 | not computable |

## Case Notes

- `control_high_progress_wrong_solution`: not present in input dataset.
- `control_monotonic_incomplete_failure`: not present in input dataset.
- `control_coding_complete_artifacts_incomplete`: not present in input dataset.

---

## Interpretation (G2) — re-run after builder fix

This section re-applies after the builder's `resolve_final_success`
was fixed (commit-pending) to short-circuit to
`source_metadata.json:final_success` for sources that pin the label
authoritatively (`final_success_source == "source_label"`). Before the
fix, 3 SWE-agent upstream successes (`s_03`, `s_06`, `s_09`) were
misclassified as failures by the heuristic test_output.txt scan,
which was confused by SWE-bench eval logs that interleave "passed",
"error", and "failed" tokens. Numbers above reflect the corrected
10/10 split.

### G2.1 Disclaimer (do NOT read AUROCs as predictive performance)

Per § 0 of `TASKS.md`, this section makes **no** claim of predictive
performance. The metrics are diagnostic plumbing output, not science:

- N=20 with leave-one-run-out is too small for any sensible
  confidence interval on AUROC.
- Progress is **decoupled from outcome by design** (per
  `feedback_progress_vs_outcome_decoupling.md`); a "fair" predictor
  built from progress alone is expected to perform near chance, and
  the AUROCs in the table reflect that.

### G2.2 Do failed runs still get high predicted probabilities?

Yes — and that is the load-bearing observation:

- Mean predicted probability for failures: 0.56 / 0.57 / 0.59
  (`progress_only` / `ledger_basic` / `elapsed_only`).
- Mean predicted probability for successes: 0.53 / 0.52 / 0.52.
- **Mean for high-progress failures (≥ 0.8 coding-progress, upstream
  label = failure): 0.69 / 0.63 / 0.61** — *highest* probability of
  any subgroup. The predictor cannot distinguish `f_06`-style
  hidden-work-gap traces from successes, and it shouldn't be able
  to from progress alone — by design.

### G2.3 Do high-progress failures exist naturally in SWE-agent?

**Yes.** With the corrected label, exactly **one** trace is a
high-progress failure: `f_06` (`googleapis__python-spanner-317`),
coding-progress 1.00, upstream `final_success=False`. The agent's
`reproduce.py` returned "Script completed successfully, no errors."
but never actually triggered the bug; the agent moved on. Detail
in case study 3 (`datasets/swe_agent_pilot_case_studies.md`).

### G2.4 Does `ledger_basic` differ from `progress_only`?

Marginally, and now in the right direction. `ledger_basic` AUROC is
**0.38** vs `progress_only` 0.37 — `ledger_basic` is +0.01 better.
Brier score is 0.0061 worse (0.279 vs 0.273). Log loss is 0.011
better (0.755 vs 0.766). The added ledger features (delta-progress,
splits/reopens counters, leaf counts) move the needle a touch but
not significantly on N=20. The framework's claim is that progress
is **decoupled** from outcome, not that progress is a better
predictor of outcome. A real test of "do ledger features help"
needs Workstream Q's on-time-finish target, not pass/fail.

### G2.5 Does `elapsed_only` remain competitive?

No. `elapsed_only` AUROC is **0.18** — anti-correlated, the worst of
the three. SWE-agent step counts correlate negatively with success:
long traces are stuck-loop failures (`f_02` 509 steps, `f_07` 183,
`f_03` 113, `f_08` 77, `f_10` 81); short traces include the cleanest
successes (`s_04` 17 steps, `s_09` 19, `s_07` 23). On toy/live the
elapsed_only baseline was competitive because short runs were
authored as failure controls; that didn't transfer.

### G2.6 Do evidence gaps dominate the signal?

Reviewing `runs/swe_agent_pilot/PILOT_ANNOTATION_SUMMARY.md` § 5
("Common evidence gaps"): yes, three patterns recur and account
for the discriminating shape:

1. **Submit-without-validation** (`f_01`, `f_04`, `s_04` —
   validation leaf at `not_started`) → coding-progress 0.67 cluster
   regardless of upstream label.
2. **Mid-edit harness termination** (`f_02`, `f_03`, `f_05`,
   `f_07`, `f_08`, `f_10` — exit_status `submitted (exit_context)`,
   no agent submit) → 0.50–0.71 failure tail.
3. **Hidden-work gap with reproduction failure** (`f_06`) → unique
   1.00-progress failure.

Each pattern is a different ledger shape. A predictor consuming
just `coding_progress` and `final_success` collapses #1 and #2
into "low progress = likely failure" but cannot distinguish #1
(submit-without-test) from #3 (`f_06`-style hidden-work gap) —
both are 1.00 progress by the ledger's lights.

### G2.7 Is the data suitable for a larger retrospective study?

**Yes.** With the builder fix, the corpus now has a clean 10/10
upstream label and populates all four success/progress quadrants.
Remaining caveat: Workstream H — at least one independent
annotator on 1-2 pilots — should land before scaling to test
reproducibility.

### G2.8 Cross-references to case studies

Detailed case studies in `datasets/swe_agent_pilot_case_studies.md`:

- **Case 1 (`s_01`):** successful high-progress normal run.
- **Case 2 (`s_03`):** successful non-monotonic run with REOPEN.
  (Now correctly labeled as a SUCCESS after the builder fix.)
- **Case 3 (`f_06`):** failed high-progress run.
- **Case 4 (`f_03`):** failed low-progress / stuck run.

The `progress_only` predictor now correctly assigns `f_06` a high
probability of "success" (the label it shares-by-shape with the
nine 1.00-success runs). This is the right behavior — the predictor
sees what the ledger sees, and the ledger says "all discovered work
done." `f_06` is failure precisely because the ledger does not
know about the undiscovered hidden work.
