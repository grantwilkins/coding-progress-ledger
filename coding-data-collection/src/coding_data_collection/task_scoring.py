from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskScoreInput:
    task_id: str
    expected_runtime_bucket: int
    expected_validation_visibility: int
    expected_file_edit_complexity: int
    expected_environment_complexity: int
    expected_failure_modes: int
    oracle_test_leakage_risk: int
    docker_feasibility: int
    requires_internet: bool
    large_download_or_build: bool


@dataclass(frozen=True)
class TaskScore:
    task_id: str
    trajectory_richness: int
    operational_risk: int
    pilot_priority: int


def score_task(item: TaskScoreInput) -> TaskScore:
    richness = (
        item.expected_runtime_bucket
        + item.expected_validation_visibility * 2
        + item.expected_file_edit_complexity
        + item.expected_environment_complexity
        + item.expected_failure_modes * 2
    )
    risk = item.oracle_test_leakage_risk * 3 + (5 - item.docker_feasibility)
    if item.requires_internet:
        risk += 3
    if item.large_download_or_build:
        risk += 2
    return TaskScore(
        task_id=item.task_id,
        trajectory_richness=richness,
        operational_risk=risk,
        pilot_priority=richness - risk,
    )


SCORE_INPUT_FIELDS = (
    "expected_runtime_bucket",
    "expected_validation_visibility",
    "expected_file_edit_complexity",
    "expected_environment_complexity",
    "expected_failure_modes",
    "oracle_test_leakage_risk",
    "docker_feasibility",
)

SCORE_OUTPUT_FIELDS = (
    "source",
    "task_id",
    "title",
    "category",
    "difficulty",
    "tags",
    *SCORE_INPUT_FIELDS,
    "requires_internet",
    "large_download_or_build",
    "trajectory_richness",
    "operational_risk",
    "pilot_priority",
    "selected_for_pilot",
    "selection_reason",
    "calibration_notes",
)


def score_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    item = TaskScoreInput(
        task_id=str(row["task_id"]),
        expected_runtime_bucket=int(row["expected_runtime_bucket"]),
        expected_validation_visibility=int(row["expected_validation_visibility"]),
        expected_file_edit_complexity=int(row["expected_file_edit_complexity"]),
        expected_environment_complexity=int(row["expected_environment_complexity"]),
        expected_failure_modes=int(row["expected_failure_modes"]),
        oracle_test_leakage_risk=int(row["oracle_test_leakage_risk"]),
        docker_feasibility=int(row["docker_feasibility"]),
        requires_internet=_as_bool(row["requires_internet"]),
        large_download_or_build=_as_bool(row["large_download_or_build"]),
    )
    score = score_task(item)
    return {
        **row,
        "trajectory_richness": score.trajectory_richness,
        "operational_risk": score.operational_risk,
        "pilot_priority": score.pilot_priority,
    }


def choose_pilot_tasks(rows: list[dict[str, Any]], *, n: int = 12) -> list[str]:
    scored = sorted(
        rows,
        key=lambda row: (
            -int(row["pilot_priority"]),
            int(row["operational_risk"]),
            str(row.get("category") or ""),
            str(row.get("difficulty") or ""),
            str(row["task_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for row in scored:
        category = str(row.get("category") or "uncategorized")
        if category not in seen_categories:
            selected.append(row)
            seen_categories.add(category)
        if len(selected) == n:
            return [str(row["task_id"]) for row in selected]
    selected_ids = {str(row["task_id"]) for row in selected}
    for row in scored:
        if str(row["task_id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["task_id"]))
        if len(selected) == n:
            break
    return [str(row["task_id"]) for row in selected]


def write_candidate_scores(input_csv: Path, output_csv: Path, *, selected_count: int = 12) -> Path:
    rows = [_normalize_candidate_row(row) for row in _read_csv(input_csv)]
    scored = [score_candidate_row(row) for row in rows]
    selected = set(choose_pilot_tasks(scored, n=selected_count))
    for row in scored:
        chosen = str(row["task_id"]) in selected
        row["selected_for_pilot"] = chosen
        row["selection_reason"] = (
            "selected by balanced priority ranking"
            if chosen
            else "reserve candidate"
        )
    scored.sort(
        key=lambda row: (
            not bool(row["selected_for_pilot"]),
            -int(row["pilot_priority"]),
            int(row["operational_risk"]),
            str(row.get("task_id")),
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(SCORE_OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in SCORE_OUTPUT_FIELDS}
            for row in scored
        )
    return output_csv


def _normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for field in SCORE_INPUT_FIELDS:
        out[field] = int(out[field])
    out["requires_internet"] = _as_bool(out["requires_internet"])
    out["large_download_or_build"] = _as_bool(out["large_download_or_build"])
    return out


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
