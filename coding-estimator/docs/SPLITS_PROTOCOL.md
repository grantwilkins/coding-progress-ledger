# Splits protocol

**Invariant:** no `run_id` may appear in more than one partition of any
fold. This is enforced by `splits.protocol.assert_disjoint` and tested in
`tests/test_splits_disjoint.py`.

## Schemes

| scheme | description | use when |
|---|---|---|
| `loro` | leave-one-run-out | small N; default v0 evaluator |
| `ltfo` | leave-one-task-family-out | task families known and balanced |
| `loso` | leave-one-source-out | cross-source generalization |
| `holdout` | fixed train/test by run, deterministic seed | published numbers |
| `temporal` | earliest k% of runs train; rest test | live sources only; warn on synthetic timestamps |

## Determinism

Every scheme takes a `seed` and produces a deterministic fold assignment.
`holdout` uses `numpy.random.default_rng(seed)` for shuffling.
`temporal` is deterministic given timestamps; the seed is recorded but
unused.

## Fairness rules

- Splits are **per source** for `loro`/`ltfo`; the caller filters first.
- `loso` is the only legitimate cross-source scheme.
- `temporal` warns when `timestamp_quality != "real"` for any included run
  — synthetic-timestamp temporal splits do not generalize.

## Schema

The on-disk form is described by `schemas/split_schema.json` and is
emitted via `coding_estimator.io.write_json`.
