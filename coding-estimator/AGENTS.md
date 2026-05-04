# AGENTS.md

Guidance for any subagent or human contributor working in this repo. The
upstream `../coding-progress-ledger/AGENTS.md` rules govern ledger-side
work; everything below adds estimator-specific constraints.

## Working rules

- Every code change adds or extends a test in `tests/`.
- Hard fail over `try`/`except`. The leakage audit and schema gates must
  surface, not swallow, errors.
- Run `uv run pytest -q` after every change. Run `uv run ruff check .` too.
- Update `TASKS.md` status markers (`not started` → `in progress` →
  `done`) at the end of each task.
- Pointer to upstream rules: `../coding-progress-ledger/AGENTS.md`.

## Project rules (reproduced verbatim from TASKS.md § 0)

```text
0.1  Do not mutate any artifact under ../coding-progress-ledger/.
0.2  Do not redefine ledger semantics. Progress is what coding-progress-ledger
     says it is. We add features and labels around it, never inside it.
0.3  Do not use any ledger event with step > S when constructing features at
     checkpoint S. The replayer must be prefix-only.
0.4  Do not use any post-run artifact when constructing features at S unless
     that artifact was visible at step S in the live trace.
     Examples of forbidden post-run leakage as features: final_diff.patch,
     final test_output.txt, final eval_logs, summary_by_category.json,
     verifier_exit_code, final_success.
0.5  Do not split train/test by checkpoint; always split by run_id.
0.6  Do not train large neural models in the first ladder. Logistic
     regression and calibrated GBMs come first.
0.7  Do not output policy actions ("pause", "stop", "throttle"). The
     estimator outputs belief state only.
0.8  Do not equate progress with success. Progress is a *measurement of
     visible discovered work*, not correctness. The estimator's job is to
     learn which progress patterns *imply* completion risk.
0.9  Do not fabricate data when an upstream artifact is missing. Skip the
     run, record it in the profile report, and move on.
0.10 Hard fail over silent fallback. If a feature is undefined for a
     checkpoint (e.g. wall-clock timestamp missing on a retrospective
     row), emit `null`/`NaN` and let the leakage audit surface it.
0.11 Calibration is a first-class output. A ranked-but-uncalibrated model
     is not shippable, even if AUROC looks good.
0.12 Every model artifact must ship with a model card stating intended
     use, data sources, splits, and known failure modes. No exceptions.
```

## Conventions
- Use `coding_estimator.io.write_parquet` / `write_csv` / `write_json` for
  every artifact write. No raw `df.to_parquet`.
- Use `coding_estimator.ingest.paths` for every ledger-side path lookup.
  Do not hardcode `../coding-progress-ledger`.
- New schemas live under `schemas/`; their docs under `docs/`.
- New label columns go through `coding_estimator/labels/registry.py`.
- New feature columns go through `coding_estimator/checkpoints/features/registry.py`.

## Cross-cutting invariants (post-D plan)

These invariants apply to every workstream from D onward. Violating them
is a regression even if all per-task tests pass.

1. **Forbidden-column guard at every choke point.** The dataset-build-time
   guard is necessary but not sufficient. Every model `.fit()`, every
   `.predict()`, and every report-generator that joins frames must call
   `coding_estimator.leakage.guard.assert_no_forbidden` on its input.
   Defense in depth: a later merge must not be able to silently
   reintroduce `final_success`, `verifier_pass`, `finish_step`, or shape
   labels.

2. **Future-mutation invariance.** For any (run, t) pair, mutating events
   past `t` must not change any feature column at any checkpoint
   `t' <= t`. The prefix replay engine (D2) ships with this as a hard
   property test. Every feature group (D3a..h) inherits it.

3. **`y_submit_without_validation` is a negative control, not a headline.**
   It is run-constant and terminal. Any non-trivial AUROC at non-terminal
   `t` is a property of the data distribution, not of the estimator. It
   is reported alongside G1/G2 in the audit, never in headline tables.

4. **TB-12 measures online realism, not model performance.** TB-12 runs
   answer "can the pipeline produce honest online checkpoint features?"
   Headline modeling claims do NOT optimize TB-12-only metrics.

5. **Retrospective-leakage caveat is mechanical, not editorial.** Any
   report that consumes `swe_agent_*` or `hermes_*` data emits an
   auto-generated caveat block via the report-template helper. If a
   report is missing the caveat, that's a bug, not a stylistic choice.

6. **Absolute progress is never reported alone.** Any table that includes
   `final_progress` (or any scalar coding_progress at terminal `t`) must
   also include `final_success` and at least one dynamics column —
   `num_progress_drops_so_far`, `num_reopens_so_far`, or
   `validation_complete`. Scalar progress in isolation invites
   misinterpretation.

7. **Missingness has four explicit semantics.** The feature dataclass
   carries a `missingness_semantic` enum, not a string:
   - `not_applicable_to_source` — concept does not apply (e.g.
     `elapsed_wall_time` on `swe_agent_pilot`)
   - `applicable_absent_so_far` — applies, hasn't been observed yet at `t`
     but could become observable later in the run (e.g.
     `validation_started` at step 3 of a run that validates at step 10)
   - `applicable_never_observed_in_run` — applies, will never be observed
     in this run (e.g. `validation_failed` on a run that only ever saw
     validation passes)
   - `unknown_due_to_missing_artifact` — cannot be determined because the
     source-side artifact (e.g. `live_instrumentation.json`) is missing
     for this run

   Writing `null` vs `False` vs `0` is decided by the semantic, not by
   convenience.

8. **`reports/CHECKPOINT_CONSTRUCTION_AUDIT.md` is a hard pre-modeling
   gate.** No baseline (Workstream G) starts until this report exists,
   covers every section in D5, and reports clean on the structural,
   prefix, forbidden-column, and run-constancy audits.
