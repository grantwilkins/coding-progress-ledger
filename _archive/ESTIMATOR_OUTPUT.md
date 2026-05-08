# Estimator output schema

One row per `(run_id, checkpoint_id, target_name, model_id)`. The
machine-readable form is `schemas/estimator_output_schema.json`. Output
frames are written via `coding_estimator.io.write_parquet`.

## v0 required columns

| column | type | notes |
|---|---|---|
| `run_id` | str | matches checkpoint frame |
| `source` | str | matches checkpoint frame |
| `checkpoint_id` | str | matches checkpoint frame |
| `model_id` | str | logical name (e.g. `g4_logreg_v0`) |
| `model_version` | str | semver |
| `target_name` | str | one of the v0 targets (B2) |
| `target_family` | str | success / progress_dynamics / validation / submission |
| `target_horizon` | object | `{units, value}` |
| `probability` | float | in `[0, 1]` |
| `prediction_kind` | str | `"binary"` only in v0 |
| `calibration_bucket` | int? | reliability-diagram bucket index |
| `calibration_source` | str | one of `isotonic`, `platt`, `raw`, `constant` |
| `estimator_commit_sha` | str | this repo |
| `schema_version` | str | semver |

## Reserved nullable columns

Populated only when the corresponding model un-defers:

| column | when populated |
|---|---|
| `regression_value`, `regression_units` | I3 hazard / regression models |
| `lower_ci`, `upper_ci` | CI-emitting models |
| `top_feature_attributions` | interpretability tooling |

A `null` here is **not** a leakage flag; it means the column is genuinely
not applicable.
