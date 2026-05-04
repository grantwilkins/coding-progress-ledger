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
