"""
Claim:
The completion-prediction smoke script uses only allowed checkpoint-level
features, evaluates by held-out run, and writes predictions plus a disclaimer
report without treating final metadata as model input.

Plausible wrong implementations:
- Split by row instead of by run, letting the same run appear in train and test.
- Include run IDs, event types, native metrics, or final-success provenance as features.
- Accept unknown labels that would make the completion target ambiguous.
- Emit a predictions table that omits the identifying columns needed for audit.
- Drop the high-progress wrong-solution control from the report notes.
"""

import csv

import pytest

from scripts import smoke_test_completion_prediction as smoke


def test_script_rejects_missing_required_columns(tmp_path):
    input_csv = tmp_path / "step.csv"
    write_input(input_csv, base_rows(), omit={"coding_progress"})

    with pytest.raises(ValueError, match="missing required columns: coding_progress"):
        smoke.run_smoke(input_csv, tmp_path / "predictions.csv", tmp_path / "report.md")


def test_script_rejects_unknown_final_success_labels(tmp_path):
    input_csv = tmp_path / "step.csv"
    rows = base_rows()
    rows[0]["final_success"] = "unknown"
    write_input(input_csv, rows)

    with pytest.raises(ValueError, match="final_success must be known true/false"):
        smoke.run_smoke(input_csv, tmp_path / "predictions.csv", tmp_path / "report.md")


def test_forbidden_features_are_not_used():
    smoke.validate_feature_sets(smoke.MODEL_FEATURES)
    used_features = {
        feature
        for features in smoke.MODEL_FEATURES.values()
        for feature in features
    }
    assert not any(smoke.is_forbidden_feature(feature) for feature in used_features)

    with pytest.raises(ValueError, match="forbidden feature used: run_id"):
        smoke.validate_feature_sets({"bad": ("run_id",)})

    with pytest.raises(ValueError, match="forbidden feature used: native_coding_progress"):
        smoke.validate_feature_sets({"bad": ("native_coding_progress",)})


def test_leave_one_run_out_split_never_places_same_run_in_train_and_test():
    run_labels = {
        "success_a": True,
        "control_high_progress_wrong_solution": False,
        "control_monotonic_incomplete_failure": False,
    }

    splits = smoke.leave_one_run_out_splits(run_labels)
    smoke.validate_leave_one_run_out_splits(splits)

    assert len(splits) == len(run_labels)
    for train_run_ids, test_run_ids in splits:
        assert train_run_ids.isdisjoint(test_run_ids)
        assert len(test_run_ids) == 1


def test_predictions_csv_is_written_with_required_columns(tmp_path):
    input_csv = tmp_path / "step.csv"
    predictions_csv = tmp_path / "predictions.csv"
    write_input(input_csv, base_rows())

    smoke.run_smoke(input_csv, predictions_csv, tmp_path / "report.md")

    with predictions_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        predictions = list(reader)

    assert reader.fieldnames == list(smoke.PREDICTION_COLUMNS)
    assert len(predictions) == len(base_rows()) * len(smoke.MODEL_FEATURES)
    assert {row["model_name"] for row in predictions} == set(smoke.MODEL_FEATURES)


def test_report_contains_disclaimer(tmp_path):
    input_csv = tmp_path / "step.csv"
    report_md = tmp_path / "report.md"
    write_input(input_csv, base_rows())

    smoke.run_smoke(input_csv, tmp_path / "predictions.csv", report_md)

    assert smoke.DISCLAIMER in report_md.read_text()


def test_high_progress_wrong_solution_control_appears_in_report_when_present(tmp_path):
    input_csv = tmp_path / "step.csv"
    report_md = tmp_path / "report.md"
    write_input(input_csv, base_rows())

    smoke.run_smoke(input_csv, tmp_path / "predictions.csv", report_md)

    report = report_md.read_text()
    assert "control_high_progress_wrong_solution" in report
    assert "final_success=false" in report


def base_rows():
    return [
        row("success_a", 0, 0, "true", 0.0, 0.0, 0.0),
        row("success_a", 1, 1, "true", 0.5, 0.4, 0.5),
        row("control_high_progress_wrong_solution", 0, 0, "false", 0.0, 0.0, 0.0),
        row("control_high_progress_wrong_solution", 1, 1, "false", 0.9, 0.9, 0.9),
        row("control_monotonic_incomplete_failure", 0, 0, "false", 0.0, 0.0, 0.0),
        row("control_monotonic_incomplete_failure", 1, 1, "false", 0.6, 0.6, 0.6),
        row("control_coding_complete_artifacts_incomplete", 0, 0, "true", 0.0, 0.0, 0.0),
        row("control_coding_complete_artifacts_incomplete", 1, 1, "true", 1.0, 0.667, 1.0),
    ]


def row(run_id, step, event_index, final_success, coding_progress, overall_progress, delta_coding_progress):
    return {
        "run_id": run_id,
        "step": str(step),
        "event_index": str(event_index),
        "coding_progress": str(coding_progress),
        "overall_progress": str(overall_progress),
        "active_coding_weight": "2.0",
        "completed_coding_weight": str(coding_progress * 2.0),
        "active_overall_weight": "3.0",
        "completed_overall_weight": str(overall_progress * 3.0),
        "active_coding_leaves": "2",
        "completed_coding_leaves": str(round(coding_progress * 2.0)),
        "active_overall_leaves": "3",
        "completed_overall_leaves": str(round(overall_progress * 3.0)),
        "num_splits_so_far": "0",
        "num_reopens_so_far": "0",
        "num_invalidations_so_far": "0",
        "delta_coding_progress": str(delta_coding_progress),
        "delta_overall_progress": str(delta_coding_progress),
        "category_resolution_mode": "native",
        "category_overrides_applied": "0",
        "final_success": final_success,
    }


def write_input(path, rows, omit=frozenset()):
    fieldnames = [field for field in ("run_id", "final_success", *smoke.ALLOWED_FEATURES) if field not in omit]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in rows:
            writer.writerow({field: value for field, value in item.items() if field in fieldnames})
