from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkTask:
    source: str
    task_id: str
    title: str
    category: str | None = None
    difficulty: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": ",".join(self.tags),
            "metadata_json": json.dumps(self.metadata, sort_keys=True),
        }


def _as_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return tuple(str(item) for item in loaded)
            except json.JSONDecodeError:
                pass
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


class TerminalBenchHFAdapter:
    source = "terminal_bench_hf"

    def inspect_row(self, row: dict[str, Any]) -> BenchmarkTask:
        task_id = str(row["task_id"])
        metadata = {
            "tar_sha256": row.get("tar_sha256"),
            "archive_bytes": row.get("archive_bytes"),
            "n_files": row.get("n_files"),
            "max_agent_timeout_sec": row.get("max_agent_timeout_sec"),
            "max_test_timeout_sec": row.get("max_test_timeout_sec"),
            "has_archive": row.get("archive") is not None,
        }
        return BenchmarkTask(
            source=self.source,
            task_id=task_id,
            title=str(row.get("base_description") or task_id)[:160],
            category=_none_or_str(row.get("category")),
            difficulty=_none_or_str(row.get("difficulty")),
            tags=_as_tags(row.get("tags")),
            metadata=metadata,
        )

    def inspect_rows(self, rows: Iterable[dict[str, Any]]) -> list[BenchmarkTask]:
        return [self.inspect_row(row) for row in rows]


class HarborTerminalBenchAdapter:
    source = "terminal_bench_harbor"
    dataset = "terminal-bench/terminal-bench-2"
    legacy_dataset_alias = "terminal-bench@2.0"

    def inspect_task_id(self, task_id: str, revision: str | None = None) -> BenchmarkTask:
        command = ["uvx", "harbor", "run", "-d", self.dataset, "-t", task_id]
        oracle_command = [*command, "-a", "oracle"]
        short_name = task_id.rsplit("/", 1)[-1]
        return BenchmarkTask(
            source=self.source,
            task_id=task_id,
            title=short_name.replace("-", " "),
            metadata={
                "dataset": self.dataset,
                "legacy_dataset_alias": self.legacy_dataset_alias,
                "revision": revision,
                "run_command": command,
                "oracle_command": oracle_command,
                "instrumentation_question": (
                    "Can Harbor expose per-step agent transcript events in the "
                    "observation_events.jsonl schema?"
                ),
            },
        )

    def inspect_task_ids(self, task_ids: Iterable[str]) -> list[BenchmarkTask]:
        return [self.inspect_task_id(task_id) for task_id in task_ids]


class SWEBenchProAdapter:
    source = "swe_bench_pro"

    def inspect_row(self, row: dict[str, Any]) -> BenchmarkTask:
        instance_id = str(row["instance_id"])
        dockerhub_tag = row.get("dockerhub_tag")
        full_image = f"jefzda/sweap-images:{dockerhub_tag}" if dockerhub_tag else None
        command_plan = {
            "repo": row.get("repo"),
            "base_commit": row.get("base_commit"),
            "dockerhub_tag": dockerhub_tag,
            "docker_image": full_image,
            "problem_statement": row.get("problem_statement"),
            "visible_test_route": row.get("selected_test_files_to_run"),
            "hidden_evaluation_route": "SWE-bench Pro evaluation repo",
            "patch_output_path": "patch_predictions.json",
            "expected_diff_format": "unified_diff",
        }
        return BenchmarkTask(
            source=self.source,
            task_id=instance_id,
            title=instance_id,
            category=_none_or_str(row.get("issue_categories")),
            difficulty=_none_or_str(row.get("issue_specificity")),
            tags=_as_tags(row.get("repo_language")),
            metadata={
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "dockerhub_tag": dockerhub_tag,
                "command_plan": command_plan,
            },
        )

    def inspect_rows(self, rows: Iterable[dict[str, Any]]) -> list[BenchmarkTask]:
        return [self.inspect_row(row) for row in rows]


def write_registry_manifest(tasks: Iterable[BenchmarkTask], path: Path) -> Path:
    rows = [task.to_manifest_row() for task in tasks]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "task_id",
                "title",
                "category",
                "difficulty",
                "tags",
                "metadata_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)
