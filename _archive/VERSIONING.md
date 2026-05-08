# Estimator versioning policy

## Format

```
<model_family>_v<major>.<minor>[_<source-slice>]
```

Where:

- `model_family` is one of `logreg`, `empirical_bin`, `constant`,
  `time_only`, `ledger_basic`. New families require a new entry in
  `schemas/model_card_schema.json`.
- `<major>.<minor>` is the version pair.
- `<source-slice>` is an optional suffix when the estimator was
  trained on a non-default source slice (e.g. `_tb_live`,
  `_retro_only`).

Examples:

- `logreg_v0.1` — v0 logistic regression, default training slice
  (all canonical sources).
- `logreg_v0.1_tb_live` — same family, trained only on `tb_live`.
- `empirical_bin_v0.1` — v0 empirical-bin estimator.

## Bumping rules

| Change                                         | Version bump |
|------------------------------------------------|--------------|
| Schema change in features (add/remove/rename)  | major        |
| Schema change in targets (add/remove/rename)   | major        |
| New canonical source enters the training slice | major        |
| Model fit change without schema change         | minor        |
| Hyperparameter retune                          | minor        |
| Training data refresh on the same schema       | minor        |
| Bugfix that doesn't change predicted values    | none*        |

*If predictions are byte-identical, no version bump is required —
note the bugfix in the changelog instead.

## Immutability

A published estimator version (anything copied into `models/<id>/`
and committed) is **immutable**. Once it ships:

- Its `model.pkl` MUST NOT be replaced.
- Its `model_card.md` and `model_card.json` MAY be replaced **only**
  to correct factual errors in the metadata (e.g. clarifying a known
  limit). Recomputed metrics that change values require a new
  `_v<major>.<minor>` directory.
- `calibration.json` is part of the model's identity. New
  calibration data → new estimator version.

This means consumers can safely cache a version string and trust
that `models/<id>/` will not silently change underneath them.

## Sign-off

A version is "signed off" iff:

1. Its model card validates against
   `schemas/model_card_schema.json`.
2. The Workstream O failure-mode tests have run and their results
   are recorded on the card.
3. The Workstream P go/no-go gate has produced a verdict and the
   verdict is recorded on the card.

The `not_safe_for_control` flag may be flipped from `true` to
`false` only after sign-off **and** with all of {O1, O5, O7, P1}
passing with margin. Flipping it requires a `<major>.<minor>` bump.

## Pre-v1

Any estimator with `<major> = 0` is pre-v1 and explicitly carries
the `not_safe_for_control = true` flag regardless of test results.
v1 is reserved for the first estimator that clears P1 with margin
on a sample size large enough for the aspirational gate
(P-future).
