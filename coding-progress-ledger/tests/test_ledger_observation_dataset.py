"""
Claim:
The observation dataset is a deterministic event-prefix replay table. Canonical
metrics use resolved categories, native metrics preserve source-log replay, and
dataset generation never rewrites ledger.jsonl.

Plausible wrong implementations:
- Detect legacy category fields after replay, where missing categories already
  look like explicit product categories.
- Patch and reuse a native Ledger instead of replaying native and resolved event
  views separately.
- Compute deltas from non-adjacent rows or with the wrong sign.
- Infer final success from progress rather than explicit metadata.
- Rewrite ledger.jsonl while filling legacy category metadata.
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_ledger_observation_dataset as builder


def test_generated_rows_are_deterministic_and_do_not_mutate_ledger_jsonl(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "deterministic",
        [
            event(0, "init", None, {"root_task": "Deterministic"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
            event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        ],
        summary={"final_success": True, "final_success_source": "summary.final_success"},
    )
    write_run(
        runs_dir,
        "second",
        [
            event(0, "init", None, {"root_task": "Second"}),
            event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
        ],
    )
    before = {path: sha256(path) for path in runs_dir.glob("*/ledger.jsonl")}

    first = builder.build_dataset(runs_dir)
    second = builder.build_dataset(runs_dir)

    assert first.event_rows == second.event_rows
    assert first.step_rows == second.step_rows
    assert first.rows == first.event_rows
    assert {path: sha256(path) for path in runs_dir.glob("*/ledger.jsonl")} == before


def test_event_output_preserves_one_row_per_event_and_adjacent_deltas(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "reverse",
        [
            event(0, "init", None, {"root_task": "Reverse"}),
            event(1, "add_subtask", "P1", {"description": "Patch first branch", "category": "product"}),
            event(2, "update_status", "P1", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(3, "add_subtask", "P2", {"description": "Patch second branch", "category": "product"}),
        ],
    )

    result = builder.build_dataset(runs_dir)
    rows = result.event_rows

    assert len(rows) == 4
    assert [row["event_index"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["delta_coding_progress"] == 0.0
    assert rows[0]["delta_overall_progress"] == 0.0
    for before, after in zip(rows, rows[1:]):
        assert after["delta_coding_progress"] == after["coding_progress"] - before["coding_progress"]
        assert after["delta_overall_progress"] == after["overall_progress"] - before["overall_progress"]


def test_step_output_groups_by_step_and_retains_max_event_index(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "grouped",
        [
            event(0, "init", None, {"root_task": "Grouped"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
            event(1, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
            event(1, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(2, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
            event(2, "add_subtask", "A", {"description": "Export artifact bundle", "category": "artifact"}),
        ],
    )

    result = builder.build_dataset(runs_dir)
    step_rows = result.step_rows

    assert [row["step"] for row in step_rows] == [0, 1, 2]
    assert [row["event_index"] for row in step_rows] == [0, 3, 5]
    assert len(result.event_rows) == 6


def test_step_deltas_are_recomputed_from_retained_step_states(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "cancelled",
        [
            event(0, "init", None, {"root_task": "Cancelled"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
            event(1, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(2, "add_subtask", "V", {"description": "Run validation", "category": "validation"}),
            event(2, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
        ],
    )

    result = builder.build_dataset(runs_dir)
    event_rows = result.event_rows
    step_rows = result.step_rows

    assert event_rows[3]["delta_coding_progress"] < 0
    assert event_rows[4]["delta_coding_progress"] > 0
    assert step_rows[0]["delta_coding_progress"] == 0.0
    assert step_rows[1]["delta_coding_progress"] == 1.0
    assert step_rows[2]["delta_coding_progress"] == 0.0
    assert step_rows[2]["coding_drop_source"] == "none"


def test_step_drop_source_uses_material_event_drop_sources_when_step_declines(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "step_drop",
        [
            event(0, "init", None, {"root_task": "Step drop"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
            event(1, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(2, "add_subtask", "V1", {"description": "Run validation one", "category": "validation"}),
            event(2, "add_subtask", "V2", {"description": "Run validation two", "category": "validation"}),
        ],
    )

    step_rows = builder.build_dataset(runs_dir).step_rows

    assert step_rows[-1]["delta_coding_progress"] == pytest.approx(-2 / 3)
    assert step_rows[-1]["coding_drop_source"] == "validation"


def test_artifact_incomplete_control_shows_resolved_coding_overall_divergence(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "artifact_incomplete",
        [
            event(0, "init", None, {"root_task": "Artifact incomplete"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior"}),
            event(1, "add_subtask", "V", {"description": "Run validation"}),
            event(1, "add_subtask", "A", {"description": "Export artifact bundle"}),
            event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(3, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
        ],
        summary={
            "final_success": True,
            "final_success_source": "summary.final_success",
            "subtask_categories": {"P": "product", "V": "validation", "A": "artifact"},
        },
    )

    final = builder.build_dataset(runs_dir).step_rows[-1]

    assert final["coding_progress"] == 1.0
    assert final["overall_progress"] == 2 / 3
    assert final["native_coding_progress"] == 2 / 3
    assert final["native_overall_progress"] == 2 / 3
    assert final["category_resolution_mode"] == "legacy_inferred"


def test_monotonic_incomplete_control_has_no_negative_coding_deltas(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "monotonic_incomplete",
        [
            event(0, "init", None, {"root_task": "Monotonic incomplete"}),
            event(1, "add_subtask", "I", {"description": "Understand behavior", "category": "investigation"}),
            event(1, "add_subtask", "P1", {"description": "Patch simple branch", "category": "product"}),
            event(1, "add_subtask", "P2", {"description": "Patch complex branch", "category": "product"}),
            event(2, "update_status", "I", {"status": "complete", "evidence": ["task.md describes behavior"]}),
            event(3, "update_status", "P1", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        ],
        summary={"final_success": False, "final_success_source": "summary.final_success"},
    )

    rows = builder.build_dataset(runs_dir).step_rows

    assert rows[-1]["coding_progress"] == 2 / 3
    assert all(row["delta_coding_progress"] >= 0 for row in rows)


def test_high_progress_wrong_solution_keeps_success_separate_from_progress(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "wrong_solution",
        [
            event(0, "init", None, {"root_task": "Wrong solution"}),
            event(1, "add_subtask", "I", {"description": "Understand expected behavior", "category": "investigation", "weight": 1}),
            event(1, "add_subtask", "P", {"description": "Patch main behavior", "category": "product", "weight": 3}),
            event(1, "add_subtask", "V", {"description": "Run visible validation", "category": "validation", "weight": 3}),
            event(1, "add_subtask", "H", {"description": "Resolve hidden edge failure", "category": "validation", "weight": 1}),
            event(2, "update_status", "I", {"status": "complete", "evidence": ["task.md describes behavior"]}),
            event(3, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
            event(4, "update_status", "V", {"status": "complete", "evidence": ["test_output.txt shows pytest passed"]}),
        ],
        summary={"final_success": False, "final_success_source": "summary.final_success"},
    )

    final = builder.build_dataset(runs_dir).step_rows[-1]

    assert final["coding_progress"] == 7 / 8
    assert final["final_success"] == "false"
    assert final["final_success_source"] == "summary.final_success"


def test_category_modes_distinguish_native_legacy_and_mixed_from_raw_json(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "native",
        [
            event(0, "init", None, {"root_task": "Native"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        ],
    )
    write_run(
        runs_dir,
        "legacy",
        [
            event(0, "init", None, {"root_task": "Legacy"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior"}),
        ],
    )
    write_run(
        runs_dir,
        "mixed",
        [
            event(0, "init", None, {"root_task": "Mixed"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
            event(1, "add_subtask", "A", {"description": "Export artifact bundle"}),
        ],
        summary={"subtask_categories": {"A": "artifact"}},
    )

    by_run = {row["run_id"]: row for row in builder.build_dataset(runs_dir).rows if row["event_index"] == 1}

    assert by_run["native"]["category_resolution_mode"] == "native"
    assert by_run["native"]["category_overrides_applied"] == 0
    assert by_run["legacy"]["category_resolution_mode"] == "legacy_inferred"
    assert by_run["legacy"]["category_overrides_applied"] == 1
    assert by_run["mixed"]["category_resolution_mode"] == "mixed"
    assert by_run["mixed"]["category_overrides_applied"] == 1


def test_main_writes_dataset_artifacts(tmp_path):
    runs_dir = tmp_path / "runs"
    out_csv = tmp_path / "datasets" / "ledger_observations_v0.csv"
    out_event_csv = tmp_path / "datasets" / "ledger_observations_v0_event.csv"
    out_step_csv = tmp_path / "datasets" / "ledger_observations_v0_step.csv"
    out_md = tmp_path / "datasets" / "ledger_observations_v0_summary.md"
    write_run(
        runs_dir,
        "cli",
        [
            event(0, "init", None, {"root_task": "CLI"}),
            event(1, "add_subtask", "P", {"description": "Patch behavior", "category": "product"}),
        ],
    )

    assert builder.main([
        "--runs-dir", str(runs_dir),
        "--output-csv", str(out_csv),
        "--output-event-csv", str(out_event_csv),
        "--output-step-csv", str(out_step_csv),
        "--summary-md", str(out_md),
    ]) == 0

    rows = list(csv.DictReader(out_csv.open()))
    event_rows = list(csv.DictReader(out_event_csv.open()))
    step_rows = list(csv.DictReader(out_step_csv.open()))
    assert rows[0]["run_id"] == "cli"
    assert event_rows == rows
    assert len(step_rows) == 2
    assert "native_coding_progress" in rows[0]
    summary = out_md.read_text()
    assert "Total runs: 1" in summary
    assert "Event rows: 2" in summary
    assert "Step rows: 2" in summary


def test_wall_clock_columns_null_on_legacy_populated_on_timestamped(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(
        runs_dir,
        "legacy_no_ts",
        [
            event(0, "init", None, {"root_task": "Legacy"}),
            event(1, "add_subtask", "P", {"description": "Patch", "category": "product"}),
            event(2, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}),
        ],
    )
    write_run(
        runs_dir,
        "live_ts",
        [
            event(0, "init", None, {"root_task": "Live"}, timestamp="2026-05-01T00:00:00+00:00"),
            event(1, "add_subtask", "P", {"description": "Patch", "category": "product"}, timestamp="2026-05-01T00:00:30+00:00"),
            event(2, "add_subtask", "V", {"description": "Validate", "category": "validation"}, timestamp="2026-05-01T00:01:00+00:00"),
            event(3, "update_status", "P", {"status": "complete", "evidence": ["final_diff.patch shows patch"]}, timestamp="2026-05-01T00:02:00+00:00"),
        ],
    )

    result = builder.build_dataset(runs_dir)
    legacy_rows = [r for r in result.event_rows if r["run_id"] == "legacy_no_ts"]
    live_rows = [r for r in result.event_rows if r["run_id"] == "live_ts"]

    for row in legacy_rows:
        assert row["elapsed_seconds"] == ""
        assert row["seconds_since_last_event"] == ""
        assert row["seconds_since_progress_increase"] == ""
        assert row["events_per_minute"] == ""

    assert live_rows[0]["elapsed_seconds"] == 0.0
    assert live_rows[0]["seconds_since_last_event"] == 0.0
    assert live_rows[1]["elapsed_seconds"] == 30.0
    assert live_rows[1]["seconds_since_last_event"] == 30.0
    assert live_rows[3]["elapsed_seconds"] == 120.0
    # progress jumps at index 3 (P completes); seconds_since_progress_increase resets to 0
    assert live_rows[3]["seconds_since_progress_increase"] == 0.0
    # rolling rate has 4 events spanning 120s = 2 minutes -> 2 events/min
    assert live_rows[3]["events_per_minute"] == pytest.approx(2.0)

    live_step_rows = [r for r in result.step_rows if r["run_id"] == "live_ts"]
    legacy_step_rows = [r for r in result.step_rows if r["run_id"] == "legacy_no_ts"]
    for row in legacy_step_rows:
        assert row["elapsed_seconds"] == ""
        assert row["seconds_since_progress_increase"] == ""
    # step rows track step-level (not event-level) seconds_since_last_event
    assert live_step_rows[0]["seconds_since_last_event"] == 0.0
    # step 3 retains the index-3 event timestamp (00:02:00); previous retained step 2 was at 00:01:00
    assert live_step_rows[-1]["seconds_since_last_event"] == 60.0


def event(step, event_type, subtask_id, payload, reason=None, timestamp=None):
    out = {
        "step": step,
        "event_type": event_type,
        "subtask_id": subtask_id,
        "payload": payload,
        "reason": reason,
    }
    if timestamp is not None:
        out["timestamp"] = timestamp
    return out


def write_run(runs_dir, run_id, events, summary=None):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
    if summary is not None:
        (run_dir / "summary_by_category.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return run_dir


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
