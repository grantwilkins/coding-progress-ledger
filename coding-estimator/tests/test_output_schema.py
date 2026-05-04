"""Estimator output schema validates and round-trips through parquet."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import pyarrow.parquet as pq
import pytest

from coding_estimator.io import write_parquet

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "estimator_output_schema.json").read_text()
)


def _row(**overrides) -> dict:
    base = {
        "run_id": "r1",
        "source": "tb_live",
        "checkpoint_id": "r1::5",
        "model_id": "g4_logreg_v0",
        "model_version": "0.1.0",
        "target_name": "y_future_progress_drop_h5",
        "target_family": "progress_dynamics",
        "target_horizon": {"units": "steps", "value": 5},
        "probability": 0.42,
        "prediction_kind": "binary",
        "calibration_bucket": 4,
        "calibration_source": "isotonic",
        "estimator_commit_sha": "deadbee",
        "schema_version": "0.1.0",
        "regression_value": None,
        "regression_units": None,
        "lower_ci": None,
        "upper_ci": None,
        "top_feature_attributions": None,
    }
    base.update(overrides)
    return base


def test_schema_valid_jsonschema() -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_basic_row_validates() -> None:
    jsonschema.validate(_row(), SCHEMA)


def test_probability_out_of_range_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row(probability=1.5), SCHEMA)


def test_bad_target_name_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row(target_name="Not_y"), SCHEMA)


def test_unknown_calibration_source_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_row(calibration_source="bayes"), SCHEMA)


def test_parquet_round_trip(tmp_path: Path) -> None:
    rows = [
        _row(run_id="r1", checkpoint_id="r1::1", probability=0.10),
        _row(run_id="r1", checkpoint_id="r1::2", probability=0.55),
        _row(run_id="r2", checkpoint_id="r2::1", probability=0.91),
    ]
    for r in rows:
        # serialize the nested dict for parquet (re-validate as dict pre-write)
        jsonschema.validate(r, SCHEMA)
        r["target_horizon"] = json.dumps(r["target_horizon"], sort_keys=True)
    df = pd.DataFrame(rows)
    path = write_parquet(df, tmp_path / "preds.parquet", sort_by=["run_id", "checkpoint_id"])
    table = pq.read_table(path)
    out = table.to_pandas()
    assert list(out["run_id"]) == ["r1", "r1", "r2"]
    # Re-validate after re-inflating target_horizon to confirm schema coverage.
    for _, row in out.iterrows():
        rec = row.to_dict()
        rec["target_horizon"] = json.loads(rec["target_horizon"])
        jsonschema.validate(rec, SCHEMA)
