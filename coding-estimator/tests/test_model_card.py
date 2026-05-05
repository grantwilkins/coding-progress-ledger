"""
Claim:
- validate_card validates a record against `schemas/model_card_schema.json`.
  Records missing any required field, or violating the regex on
  `estimator_id`/`commit_sha`, must raise.
- estimator_id pattern is `<model_family>_v<major>.<minor>[_<source-slice>]`
  with lowercase model_family. `logreg_v0` (missing minor) is invalid.
- build_card_record assembles a record that validates and exposes
  every required field.

Plausible wrong implementations:
- Schema looser than the contract — accepts records missing
  `commit_sha` or `failure_mode_results`.
- estimator_id regex accepts `logreg_v0` (no minor) or `LogregV1`
  (capitalized).
- build_card_record forgets to populate `target_definitions` for
  every requested target.
- write_card writes the JSON without validating first (so an invalid
  card hits disk).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from coding_estimator.models.cards import (
    SCHEMA_PATH,
    build_card_record,
    load_schema,
    render_card_markdown,
    validate_card,
    write_card,
)


def _minimal_valid_record() -> dict:
    """A minimal record that should validate."""
    return {
        "estimator_id": "logreg_v0.1",
        "estimator_version": "0.1.0",
        "model_family": "logreg",
        "training_data": {
            "checkpoints_path": "datasets/checkpoints_all.parquet",
            "labels_path": "datasets/labels_all.parquet",
            "n_runs": 50,
            "n_checkpoints": 1500,
        },
        "source_versions": {"swe_agent_pilot": "0"},
        "feature_groups": ["closure", "frontier"],
        "target_definitions": {
            "y_success_eventual": {
                "family": "success",
                "horizon_units": "terminal",
                "horizon_value": None,
                "run_constant_flag": True,
            }
        },
        "split_protocol": {
            "headline_scheme": "loro",
            "diagnostic_schemes": ["loso"],
            "headline_seed": 0,
        },
        "known_limits": ["v0 small-N"],
        "not_safe_for_control": True,
        "calibration_status": {
            "y_success_eventual": {
                "brier": 0.25,
                "ece": 0.1,
                "ece_3bin": 0.05,
                "auroc": 0.6,
                "n_checkpoints": 600,
                "calibration_method": "raw",
            }
        },
        "intended_use": ["Offline eval"],
        "non_use_cases": ["Driving control"],
        "commit_sha": "abc1234",
        "failure_mode_results": {
            "O1": {
                "test_id": "O1",
                "outcome": "pass",
                "metric_name": "median_p",
                "metric_value": 0.5,
                "threshold": 0.7,
                "note": None,
            },
            "O5": {
                "test_id": "O5",
                "outcome": "indeterminate",
                "metric_name": "delta_brier",
                "metric_value": None,
                "threshold": 0.02,
                "note": "no source_task columns",
            },
            "O7": [
                {
                    "test_id": "O7",
                    "outcome": "fail",
                    "metric_name": "brier_g2_minus_brier_g4",
                    "metric_value": -0.01,
                    "threshold": 0.02,
                    "note": None,
                }
            ],
        },
    }


def test_minimal_record_validates():
    record = _minimal_valid_record()
    validate_card(record)  # must not raise


@pytest.mark.parametrize(
    "missing_field",
    [
        "estimator_id",
        "estimator_version",
        "training_data",
        "feature_groups",
        "calibration_status",
        "commit_sha",
        "failure_mode_results",
        "not_safe_for_control",
        "split_protocol",
        "known_limits",
        "intended_use",
        "non_use_cases",
    ],
)
def test_record_missing_required_field_is_rejected(missing_field):
    record = _minimal_valid_record()
    record.pop(missing_field)
    with pytest.raises(jsonschema.ValidationError):
        validate_card(record)


@pytest.mark.parametrize(
    "estimator_id,expected_valid",
    [
        ("logreg_v0.1", True),
        ("logreg_v0.1_tb_live", True),
        ("empirical_bin_v1.10", True),
        ("logreg_v0", False),  # missing minor
        ("logreg_v1", False),  # missing minor
        ("LogregV1", False),  # uppercase
        ("logreg-v0.1", False),  # dash instead of underscore
        ("v0.1", False),  # missing family
        ("logreg_V0.1", False),  # capital V
    ],
)
def test_estimator_id_pattern_matches_versioning_doc(estimator_id, expected_valid):
    record = _minimal_valid_record()
    record["estimator_id"] = estimator_id
    if expected_valid:
        validate_card(record)
    else:
        with pytest.raises(jsonschema.ValidationError):
            validate_card(record)


@pytest.mark.parametrize(
    "commit_sha,expected_valid",
    [
        ("abc1234", True),  # 7 chars
        ("a" * 40, True),
        ("0123456789abcdef", True),
        ("ABC1234", False),  # uppercase
        ("abc12", False),  # too short
        ("xyz1234", False),  # non-hex
    ],
)
def test_commit_sha_pattern(commit_sha, expected_valid):
    record = _minimal_valid_record()
    record["commit_sha"] = commit_sha
    if expected_valid:
        validate_card(record)
    else:
        with pytest.raises(jsonschema.ValidationError):
            validate_card(record)


def test_outcome_must_be_one_of_pass_fail_indeterminate():
    record = _minimal_valid_record()
    record["failure_mode_results"]["O1"]["outcome"] = "maybe"
    with pytest.raises(jsonschema.ValidationError):
        validate_card(record)


def test_calibration_method_enum_enforced():
    record = _minimal_valid_record()
    record["calibration_status"]["y_success_eventual"]["calibration_method"] = "magic"
    with pytest.raises(jsonschema.ValidationError):
        validate_card(record)


def test_write_card_validates_before_writing(tmp_path: Path):
    """An invalid record must not produce a model_card.json on disk —
    validate_card raises and the writer aborts."""
    record = _minimal_valid_record()
    record.pop("commit_sha")
    with pytest.raises(jsonschema.ValidationError):
        write_card(tmp_path, record)
    assert not (tmp_path / "model_card.json").exists()
    assert not (tmp_path / "model_card.md").exists()


def test_write_card_emits_both_json_and_md(tmp_path: Path):
    record = _minimal_valid_record()
    json_path, md_path = write_card(tmp_path, record)
    assert json_path.exists() and md_path.exists()
    # Round-trip the JSON and re-validate
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    validate_card(parsed)
    md = md_path.read_text(encoding="utf-8")
    # Critical fields must surface in the human-facing markdown
    assert record["estimator_id"] in md
    assert record["commit_sha"] in md


def test_render_card_markdown_includes_all_target_calibration_rows():
    """If a record has 3 targets in calibration_status, the rendered
    markdown MUST mention all 3 (a wrong impl might only render the
    first or last)."""
    record = _minimal_valid_record()
    for t in ("y_future_progress_drop_h5", "y_validation_new_work_h5"):
        record["target_definitions"][t] = {
            "family": "progress_dynamics",
            "horizon_units": "steps",
            "horizon_value": 5,
            "run_constant_flag": False,
        }
        record["calibration_status"][t] = {
            "brier": 0.05, "ece": 0.02, "ece_3bin": None,
            "auroc": 0.95, "n_checkpoints": 500,
            "calibration_method": "raw",
        }
    md = render_card_markdown(record)
    for t in record["calibration_status"]:
        assert f"`{t}`" in md, f"target {t} missing from rendered card"


def test_build_card_record_populates_every_requested_target():
    ck = pd.DataFrame(
        {
            "run_id": ["r0", "r1"],
            "source": ["swe_agent_pilot", "swe_agent_pilot"],
            "source_protocol_version": ["v1", "v1"],
            "checkpoint_id": ["r0_c0", "r1_c0"],
        }
    )
    lb = pd.DataFrame(columns=["run_id", "checkpoint_id", "target_name", "label_value", "is_masked"])
    targets = ["y_success_eventual", "y_future_progress_drop_h5"]
    record = build_card_record(
        estimator_id="logreg_v0.1",
        estimator_version="0.1.0",
        model_family="logreg",
        checkpoints_df=ck,
        labels_df=lb,
        feature_groups=("closure",),
        targets=targets,
        eval_cells=[],
        headline_scheme="loro",
        diagnostic_schemes=("loso",),
        headline_seed=0,
        calibration_method="raw",
        intended_use=["x"],
        non_use_cases=["y"],
        known_limits=["z"],
        not_safe_for_control=True,
        commit_sha="abc1234",
        failure_mode_results={
            "O1": _minimal_valid_record()["failure_mode_results"]["O1"],
            "O5": _minimal_valid_record()["failure_mode_results"]["O5"],
            "O7": _minimal_valid_record()["failure_mode_results"]["O7"],
        },
    )
    for t in targets:
        assert t in record["target_definitions"], f"target {t} missing"
    # Validates against schema
    validate_card(record)
