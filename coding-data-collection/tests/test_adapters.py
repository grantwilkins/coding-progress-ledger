from __future__ import annotations

import json
from pathlib import Path

from coding_data_collection.benchmarks import (
    HarborTerminalBenchAdapter,
    SWEBenchProAdapter,
    TerminalBenchHFAdapter,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE_FIELDS = {"archive", "task_yaml", "patch", "test_patch", "fail_to_pass", "pass_to_pass"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_terminal_bench_hf_adapter_inspects_rows() -> None:
    rows = [
        {
            "task_id": f"task-{i}",
            "base_description": "Do a terminal task",
            "difficulty": "medium",
            "tags": ["software"],
            "category": "software engineering",
            "tar_sha256": "abc",
            "archive_bytes": 123,
            "n_files": 7,
        }
        for i in range(3)
    ]
    tasks = TerminalBenchHFAdapter().inspect_rows(rows)
    assert len(tasks) == 3
    assert tasks[0].source == "terminal_bench_hf"
    assert tasks[0].metadata["has_archive"] is False


def test_harbor_adapter_emits_oracle_command_plan() -> None:
    tasks = HarborTerminalBenchAdapter().inspect_task_ids(
        ["terminal-bench/fix-git", "terminal-bench/regex-log", "terminal-bench/query-optimize"]
    )
    assert len(tasks) == 3
    assert tasks[0].metadata["dataset"] == "terminal-bench/terminal-bench-2"
    assert tasks[0].metadata["legacy_dataset_alias"] == "terminal-bench@2.0"
    assert "-a" in tasks[0].metadata["oracle_command"]
    assert "instrumentation_question" in tasks[0].metadata


def test_swe_bench_pro_adapter_emits_future_command_plan() -> None:
    rows = [
        {
            "repo": "org/repo",
            "instance_id": f"inst-{i}",
            "base_commit": "a" * 40,
            "problem_statement": "Fix it",
            "repo_language": "Python",
            "issue_categories": "bug",
            "issue_specificity": "specific",
            "selected_test_files_to_run": "tests/test_x.py",
            "dockerhub_tag": f"tag-{i}",
        }
        for i in range(3)
    ]
    tasks = SWEBenchProAdapter().inspect_rows(rows)
    plan = tasks[0].metadata["command_plan"]
    assert len(tasks) == 3
    assert plan["docker_image"] == "jefzda/sweap-images:tag-0"
    assert plan["expected_diff_format"] == "unified_diff"


def test_real_redacted_terminal_bench_hf_sample_rows_are_inspectable() -> None:
    rows = _read_jsonl(ROOT / "datasets" / "source_samples" / "terminal_bench_hf_rows.jsonl")

    tasks = TerminalBenchHFAdapter().inspect_rows(rows)

    assert [task.task_id for task in tasks] == [
        "adaptive-rejection-sampler",
        "aimo-airline-departures",
        "attention-mil",
    ]
    assert all(not FORBIDDEN_SOURCE_FIELDS.intersection(row) for row in rows)
    assert tasks[0].metadata["tar_sha256"] == "67bb3bf96df062a37c8158adfb49ca8bd32e45f968d3a414b6f0b9621faf9354"


def test_real_redacted_swe_bench_pro_sample_rows_are_inspectable() -> None:
    rows = _read_jsonl(ROOT / "datasets" / "source_samples" / "swe_bench_pro_rows.jsonl")

    tasks = SWEBenchProAdapter().inspect_rows(rows)

    assert len(tasks) == 3
    assert tasks[0].task_id == "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
    assert all(not FORBIDDEN_SOURCE_FIELDS.intersection(row) for row in rows)
    assert tasks[1].metadata["command_plan"]["docker_image"].startswith(
        "jefzda/sweap-images:qutebrowser.qutebrowser-"
    )


def test_current_harbor_hub_terminal_bench_task_ids_are_inspectable() -> None:
    manifest = json.loads((ROOT / "manifests" / "harbor_terminal_bench_tasks.json").read_text(encoding="utf-8"))

    tasks = HarborTerminalBenchAdapter().inspect_task_ids(manifest["tasks"])

    assert len(tasks) == 3
    assert tasks[0].task_id == "terminal-bench/fix-git"
    assert tasks[0].metadata["oracle_command"][:4] == [
        "uvx",
        "harbor",
        "run",
        "-d",
    ]
