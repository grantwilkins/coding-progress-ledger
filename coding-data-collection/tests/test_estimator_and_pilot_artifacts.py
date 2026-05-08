"""
Claim:
The collection repo only orchestrates estimator artifact production and
candidate selection. Estimator-facing checkpoint artifacts must carry prefix
provenance for every row, and pilot task selection must balance category
coverage before filling remaining slots by priority.

Plausible wrong implementations:
- Accept checkpoint rows that omit max_ledger_step_used or
  max_observation_step_used.
- Accept provenance that uses future observation or ledger steps.
- Compute estimator features or labels in collection instead of validating
  estimator-produced artifact files.
- Select the top-N tasks globally and accidentally drop whole task categories.
- Sort pilot candidates by operational risk in the wrong direction.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from coding_data_collection.estimator_artifacts import stage_runs_for_estimator, validate_estimator_artifacts
from coding_data_collection.task_scoring import (
    choose_pilot_tasks,
    score_candidate_row,
    write_candidate_scores,
)


def _write_estimator_surface(path: Path, checkpoints: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    checkpoints.to_parquet(path / "checkpoints.parquet", index=False)
    pd.DataFrame({"run_id": ["r1"]}).to_parquet(path / "labels.parquet", index=False)
    pd.DataFrame({"run_id": ["r1"]}).to_parquet(path / "estimator_predictions.parquet", index=False)
    (path / "checkpoint_feature_manifest.json").write_text("{}\n", encoding="utf-8")


def test_estimator_artifact_validation_requires_complete_prefix_provenance(tmp_path: Path) -> None:
    _write_estimator_surface(
        tmp_path,
        pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "checkpoint_step": 3,
                    "max_ledger_step_used": 3,
                    "max_observation_step_used": 2,
                },
                {
                    "run_id": "r1",
                    "checkpoint_step": 4,
                    "max_ledger_step_used": 4,
                    "max_observation_step_used": 4,
                },
            ]
        ),
    )

    report = validate_estimator_artifacts(tmp_path)

    assert report["passed"] is True
    assert report["prefix_provenance_complete"] is True


def test_estimator_artifact_validation_rejects_future_prefix_evidence(tmp_path: Path) -> None:
    _write_estimator_surface(
        tmp_path,
        pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "checkpoint_step": 3,
                    "max_ledger_step_used": 3,
                    "max_observation_step_used": 5,
                },
            ]
        ),
    )

    report = validate_estimator_artifacts(tmp_path)

    assert report["passed"] is False
    assert "checkpoints.parquet: max_observation_step_used exceeds checkpoint_step" in report["issues"]


def test_replace_staged_runs_removes_stale_entries(tmp_path: Path) -> None:
    estimator_root = tmp_path / "estimator"
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    old_run.mkdir()
    new_run.mkdir()

    staged_root = stage_runs_for_estimator(
        run_dirs=[old_run],
        estimator_root=estimator_root,
        source_id="terminal_bench_pilot",
    )

    assert (staged_root / "old").exists()

    stage_runs_for_estimator(
        run_dirs=[new_run],
        estimator_root=estimator_root,
        source_id="terminal_bench_pilot",
        replace=True,
    )

    assert not (staged_root / "old").exists()
    assert (staged_root / "new").exists()


def _candidate(task_id: str, category: str, priority_knobs: dict) -> dict:
    base = {
        "source": "terminal_bench_hf",
        "task_id": task_id,
        "title": task_id,
        "category": category,
        "difficulty": "medium",
        "tags": "",
        "expected_runtime_bucket": 2,
        "expected_validation_visibility": 2,
        "expected_file_edit_complexity": 2,
        "expected_environment_complexity": 2,
        "expected_failure_modes": 2,
        "oracle_test_leakage_risk": 1,
        "docker_feasibility": 5,
        "requires_internet": False,
        "large_download_or_build": False,
        "calibration_notes": "",
    }
    return score_candidate_row({**base, **priority_knobs})


def test_pilot_selection_preserves_category_coverage_before_priority_fill() -> None:
    rows = [
        _candidate("high-a", "math", {"expected_failure_modes": 5}),
        _candidate("second-a", "math", {"expected_validation_visibility": 5}),
        _candidate("only-b", "systems", {"expected_runtime_bucket": 1}),
        _candidate("only-c", "security", {"expected_runtime_bucket": 1}),
    ]

    selected = choose_pilot_tasks(rows, n=3)

    assert set(selected) == {"high-a", "only-b", "only-c"}


def test_write_candidate_scores_uses_formula_and_marks_selected_rows(tmp_path: Path) -> None:
    input_csv = tmp_path / "calibration.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "task_id",
                "title",
                "category",
                "difficulty",
                "tags",
                "expected_runtime_bucket",
                "expected_validation_visibility",
                "expected_file_edit_complexity",
                "expected_environment_complexity",
                "expected_failure_modes",
                "oracle_test_leakage_risk",
                "docker_feasibility",
                "requires_internet",
                "large_download_or_build",
                "calibration_notes",
            ],
        )
        writer.writeheader()
        writer.writerow({field: _candidate("a", "math", {}).get(field, "") for field in writer.fieldnames})
        writer.writerow(
            {
                field: _candidate("b", "systems", {"oracle_test_leakage_risk": 3}).get(field, "")
                for field in writer.fieldnames
            }
        )

    output_csv = tmp_path / "scores.csv"
    write_candidate_scores(input_csv, output_csv, selected_count=1)
    rows = list(csv.DictReader(output_csv.open(newline="", encoding="utf-8")))

    assert rows[0]["task_id"] == "a"
    assert rows[0]["selected_for_pilot"] == "True"
    assert int(rows[0]["pilot_priority"]) > int(rows[1]["pilot_priority"])
