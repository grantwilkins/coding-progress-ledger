"""
Claim:
The observation audit checks a CSV artifact for internal accounting coherence,
category-resolution diagnostics, event-vs-step aggregation effects, and
success/progress diagnostic buckets without rebuilding ledgers.

Plausible wrong implementations:
- Validate progress bounds but ignore native progress columns.
- Compare deltas across steps rather than adjacent event rows.
- Miss native/resolved divergence by only checking final rows.
- Treat event-level and step-level drops as equivalent when same-step events
  cancel each other out.
- Classify high-progress failures as successes by looking only at progress.
"""

import csv
import hashlib
import json
from pathlib import Path

from scripts import audit_ledger_observation_dataset as audit
from scripts.build_ledger_observation_dataset import DATASET_FIELDS


def test_integrity_checks_catch_invalid_progress(tmp_path):
    path = write_csv(tmp_path, [
        row("bad_progress", 0, 0, coding_progress="1.2", final_success="true"),
    ])

    summary = audit.audit_dataset(path)

    assert summary["integrity"]["passed"] is False
    assert summary["integrity"]["invalid_progress"][0]["field"] == "coding_progress"
    assert any("invalid progress values" in warning for warning in summary["warnings"])


def test_delta_mismatch_detection_uses_adjacent_event_rows(tmp_path):
    path = write_csv(tmp_path, [
        row("delta", 0, 0, coding_progress="0.0", delta_coding_progress="0.0"),
        row("delta", 1, 1, coding_progress="0.5", delta_coding_progress="0.4"),
    ])

    summary = audit.audit_dataset(path)
    mismatch = summary["integrity"]["delta_mismatches"][0]

    assert mismatch["run_id"] == "delta"
    assert mismatch["field"] == "delta_coding_progress"
    assert mismatch["observed"] == 0.4
    assert mismatch["expected"] == 0.5


def test_native_resolved_mismatch_reports_run_and_max_difference(tmp_path):
    path = write_csv(tmp_path, [
        row("mismatch", 0, 0, coding_progress="0.0", native_coding_progress="0.0"),
        row("mismatch", 1, 1, coding_progress="1.0", native_coding_progress="0.25"),
    ])

    summary = audit.audit_dataset(path)

    assert summary["category_resolution"]["runs_with_native_resolved_metric_mismatch"] == ["mismatch"]
    assert summary["category_resolution"]["max_abs_native_resolved_coding_progress_diff_by_run"]["mismatch"] == 0.75
    assert any("large native/resolved divergence" in warning for warning in summary["warnings"])


def test_event_level_vs_step_level_comparison_detects_cancelled_same_step_drop(tmp_path):
    path = write_csv(tmp_path, [
        row("same_step", 0, 0, coding_progress="0.0", delta_coding_progress="0.0"),
        row("same_step", 1, 1, coding_progress="1.0", delta_coding_progress="1.0"),
        row("same_step", 2, 2, coding_progress="0.4", delta_coding_progress="-0.6", coding_drop_source="product"),
        row("same_step", 2, 3, coding_progress="1.0", delta_coding_progress="0.6"),
    ])

    summary = audit.audit_dataset(path)
    diff = summary["event_vs_step"]["runs_where_largest_coding_drop_differs"][0]

    assert diff["run_id"] == "same_step"
    assert diff["event_level_largest_coding_drop"] == 0.6
    assert diff["step_level_largest_coding_drop"] == 0.0
    assert summary["event_vs_step"]["runs_with_multiple_events_at_same_step"] == {"same_step": [2]}
    assert any("event-level and step-level largest drops differ substantially" in warning for warning in summary["warnings"])


def test_success_progress_quadrant_classification_uses_final_rows(tmp_path):
    path = write_csv(tmp_path, [
        row("success_high", 0, 0, coding_progress="0.9", final_success="true"),
        row("success_low", 0, 0, coding_progress="0.7", final_success="true"),
        row("failure_high", 0, 0, coding_progress="0.8", final_success="false"),
        row("failure_low", 0, 0, coding_progress="0.2", final_success="false"),
        row("unknown", 0, 0, coding_progress="1.0", final_success="", final_success_source="unknown"),
    ])

    quadrants = audit.audit_dataset(path)["success_progress_quadrants"]

    assert quadrants["success_high_progress"] == ["success_high"]
    assert quadrants["success_low_progress"] == ["success_low"]
    assert quadrants["failure_high_progress"] == ["failure_high"]
    assert quadrants["failure_low_progress"] == ["failure_low"]
    assert quadrants["unknown_success"] == ["unknown"]


def test_audit_does_not_mutate_input_csv_and_cli_writes_outputs(tmp_path):
    input_csv = write_csv(tmp_path, [
        row("cli", 0, 0),
        row("cli", 1, 1, coding_progress="1.0", delta_coding_progress="1.0"),
    ])
    output_md = tmp_path / "audit.md"
    output_json = tmp_path / "audit.json"
    before = sha256(input_csv)

    assert audit.main(["--input-csv", str(input_csv), "--output-md", str(output_md), "--output-json", str(output_json)]) == 0

    assert sha256(input_csv) == before
    assert "Ledger Observations v0 Audit" in output_md.read_text()
    summary = json.loads(output_json.read_text())
    assert summary["totals"]["runs"] == 1


def row(
    run_id,
    step,
    event_index,
    *,
    coding_progress="0.0",
    overall_progress=None,
    active_coding_weight="1.0",
    completed_coding_weight="0.0",
    active_overall_weight=None,
    completed_overall_weight=None,
    active_coding_leaves="1",
    completed_coding_leaves="0",
    active_overall_leaves=None,
    completed_overall_leaves=None,
    delta_coding_progress="0.0",
    delta_overall_progress=None,
    coding_drop_source="none",
    overall_drop_source="none",
    final_success="true",
    final_success_source="summary.final_success",
    native_coding_progress=None,
    native_overall_progress=None,
    native_delta_coding_progress=None,
    native_delta_overall_progress=None,
    native_coding_drop_source=None,
    native_overall_drop_source=None,
    category_resolution_mode="native",
):
    overall_progress = coding_progress if overall_progress is None else overall_progress
    active_overall_weight = active_coding_weight if active_overall_weight is None else active_overall_weight
    completed_overall_weight = completed_coding_weight if completed_overall_weight is None else completed_overall_weight
    active_overall_leaves = active_coding_leaves if active_overall_leaves is None else active_overall_leaves
    completed_overall_leaves = completed_coding_leaves if completed_overall_leaves is None else completed_overall_leaves
    delta_overall_progress = delta_coding_progress if delta_overall_progress is None else delta_overall_progress
    native_coding_progress = coding_progress if native_coding_progress is None else native_coding_progress
    native_overall_progress = overall_progress if native_overall_progress is None else native_overall_progress
    native_delta_coding_progress = delta_coding_progress if native_delta_coding_progress is None else native_delta_coding_progress
    native_delta_overall_progress = delta_overall_progress if native_delta_overall_progress is None else native_delta_overall_progress
    native_coding_drop_source = coding_drop_source if native_coding_drop_source is None else native_coding_drop_source
    native_overall_drop_source = overall_drop_source if native_overall_drop_source is None else native_overall_drop_source
    return {
        "run_id": run_id,
        "step": str(step),
        "event_index": str(event_index),
        "event_type": "init" if event_index == 0 else "update_status",
        "subtask_id": "",
        "coding_progress": coding_progress,
        "overall_progress": overall_progress,
        "active_coding_weight": active_coding_weight,
        "completed_coding_weight": completed_coding_weight,
        "active_overall_weight": active_overall_weight,
        "completed_overall_weight": completed_overall_weight,
        "active_coding_leaves": active_coding_leaves,
        "completed_coding_leaves": completed_coding_leaves,
        "active_overall_leaves": active_overall_leaves,
        "completed_overall_leaves": completed_overall_leaves,
        "num_splits_so_far": "0",
        "num_reopens_so_far": "0",
        "num_invalidations_so_far": "0",
        "delta_coding_progress": delta_coding_progress,
        "delta_overall_progress": delta_overall_progress,
        "coding_drop_source": coding_drop_source,
        "overall_drop_source": overall_drop_source,
        "final_success": final_success,
        "final_success_source": final_success_source,
        "native_coding_progress": native_coding_progress,
        "native_overall_progress": native_overall_progress,
        "native_active_coding_weight": active_coding_weight,
        "native_completed_coding_weight": completed_coding_weight,
        "native_active_overall_weight": active_overall_weight,
        "native_completed_overall_weight": completed_overall_weight,
        "native_active_coding_leaves": active_coding_leaves,
        "native_completed_coding_leaves": completed_coding_leaves,
        "native_active_overall_leaves": active_overall_leaves,
        "native_completed_overall_leaves": completed_overall_leaves,
        "native_delta_coding_progress": native_delta_coding_progress,
        "native_delta_overall_progress": native_delta_overall_progress,
        "native_coding_drop_source": native_coding_drop_source,
        "native_overall_drop_source": native_overall_drop_source,
        "category_resolution_mode": category_resolution_mode,
        "category_overrides_applied": "0",
    }


def write_csv(tmp_path, rows):
    path = tmp_path / "observations.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
