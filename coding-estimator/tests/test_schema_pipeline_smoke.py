"""End-to-end schema smoke: 1-row checkpoint → label → prediction frames,
each validated against its jsonschema, with the forbidden-column guard
and split disjointness exercised on a tiny 2-run synthetic split."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd

from coding_estimator.leakage.guard import assert_no_forbidden
from coding_estimator.splits import protocol as sp

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
CKPT_SCHEMA = json.loads((SCHEMAS_DIR / "checkpoint_schema.json").read_text())
LABEL_SCHEMA = json.loads((SCHEMAS_DIR / "label_schema.json").read_text())
OUTPUT_SCHEMA = json.loads((SCHEMAS_DIR / "estimator_output_schema.json").read_text())


def test_end_to_end_schema_pipeline() -> None:
    # 1-row checkpoint frame
    ckpt = {
        "run_id": "r1",
        "source": "tb_live",
        "checkpoint_id": "r1::5",
        "checkpoint_step": 5,
        "checkpoint_event_index": 17,
        "checkpoint_wall_time": "2026-05-04T10:00:00Z",
        "checkpoint_elapsed_seconds": 90.0,
        "checkpoint_fraction_timeout": 0.05,
        "is_terminal_checkpoint": False,
        "timestamp_quality": "real",
        "ledger_path": "runs/tb_live/r1/ledger.jsonl",
        "schema_version": "0.1.0",
        "builder_commit_sha": "deadbee",
        "source_protocol_version": "v1",
        "active_leaf_count": 3,
        "coding_progress": 0.4,
    }
    jsonschema.validate(ckpt, CKPT_SCHEMA)
    assert_no_forbidden(pd.DataFrame([ckpt]))

    # 1-row label frame
    label = {
        "run_id": "r1",
        "source": "tb_live",
        "checkpoint_id": "r1::5",
        "target_name": "y_future_progress_drop_h5",
        "target_family": "progress_dynamics",
        "target_horizon": {"units": "steps", "value": 5},
        "label_value": 1,
        "is_masked": False,
        "mask_reason": None,
        "schema_version": "0.1.0",
    }
    jsonschema.validate(label, LABEL_SCHEMA)

    # 1-row prediction frame
    pred = {
        "run_id": "r1",
        "source": "tb_live",
        "checkpoint_id": "r1::5",
        "model_id": "g4_logreg_v0",
        "model_version": "0.1.0",
        "target_name": "y_future_progress_drop_h5",
        "target_family": "progress_dynamics",
        "target_horizon": {"units": "steps", "value": 5},
        "probability": 0.31,
        "prediction_kind": "binary",
        "calibration_bucket": 3,
        "calibration_source": "isotonic",
        "estimator_commit_sha": "deadbee",
        "schema_version": "0.1.0",
    }
    jsonschema.validate(pred, OUTPUT_SCHEMA)

    # 2-run synthetic split, disjointness assertion
    df = pd.DataFrame([{"run_id": "r1"}, {"run_id": "r2"}])
    sp.assert_disjoint(sp.loro(df))
