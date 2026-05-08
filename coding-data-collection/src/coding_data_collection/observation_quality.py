from __future__ import annotations

from collections import Counter
from statistics import median
from pathlib import Path
from typing import Any

from .observation import read_jsonl
from .validation import validate_run_dir


TERMINAL_EVENT_TYPES = {"verifier_pass", "verifier_fail", "verifier_disagreement", "expected_file_missing"}


def observation_quality_report(run_dir: Path) -> dict[str, Any]:
    transcript_path = run_dir / "transcript.jsonl"
    observation_path = run_dir / "observation_events.jsonl"
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    shell_rows = [row for row in transcript if row.get("kind") == "shell"]
    validation_issues = [
        {"artifact": issue.artifact, "message": issue.message}
        for issue in validate_run_dir(run_dir)
        if issue.artifact == "observation_events.jsonl"
    ]

    shell_count = len(shell_rows)
    stdout_present = sum(_has_capture_field(row, "stdout_snippet") or _has_capture_field(row, "obs_snippet") for row in shell_rows)
    stderr_present = sum(_has_capture_field(row, "stderr_snippet") for row in shell_rows)
    exit_code_present = sum("exit_code" in row and row.get("exit_code") is not None for row in shell_rows)
    stdout_nonempty = sum(bool(row.get("stdout_snippet") or row.get("obs_snippet")) for row in shell_rows)
    stderr_nonempty = sum(bool(row.get("stderr_snippet")) for row in shell_rows)

    terminal_events = [event for event in observations if event.get("event_type") in TERMINAL_EVENT_TYPES]
    visible_terminal_events = [
        event
        for event in terminal_events
        if event.get("payload", {}).get("visible_to_agent") is not False
    ]
    hidden_phase_visible_events = [
        event
        for event in observations
        if event.get("source_artifact") in {"verifier_output.txt", "oracle_workspace_snapshot"}
        and event.get("payload", {}).get("visible_to_agent") is not False
    ]
    missing_inputs = [
        name
        for name, path in (
            ("transcript.jsonl", transcript_path),
            ("observation_events.jsonl", observation_path),
        )
        if not path.is_file()
    ]

    return {
        "run_dir": str(run_dir),
        "missing_inputs": missing_inputs,
        "observation_schema_valid": not validation_issues,
        "observation_validation_issues": validation_issues,
        "shell_rows": shell_count,
        "shell_exit_code_coverage": _ratio(exit_code_present, shell_count),
        "shell_stdout_snippet_coverage": _ratio(stdout_present, shell_count),
        "shell_stderr_snippet_coverage": _ratio(stderr_present, shell_count),
        "shell_stdout_nonempty_fraction": _ratio(stdout_nonempty, shell_count),
        "shell_stderr_nonempty_fraction": _ratio(stderr_nonempty, shell_count),
        "event_counts": dict(sorted(Counter(event.get("event_type") for event in observations).items())),
        "observation_event_count": len(observations),
        "terminal_event_count": len(terminal_events),
        "terminal_events_visible_to_agent": len(visible_terminal_events),
        "hidden_phase_events_visible_to_agent": len(hidden_phase_visible_events),
        "passed": not missing_inputs and shell_count > 0 and not validation_issues and not visible_terminal_events and not hidden_phase_visible_events,
    }


def corpus_observation_quality_report(run_dirs: list[Path]) -> dict[str, Any]:
    reports = [observation_quality_report(run_dir) for run_dir in run_dirs]
    shell_rows = sum(report["shell_rows"] for report in reports)
    event_counts = [report["observation_event_count"] for report in reports]
    weighted_exit = sum(report["shell_exit_code_coverage"] * report["shell_rows"] for report in reports)
    weighted_stdout = sum(report["shell_stdout_snippet_coverage"] * report["shell_rows"] for report in reports)
    weighted_stderr = sum(report["shell_stderr_snippet_coverage"] * report["shell_rows"] for report in reports)
    return {
        "run_count": len(reports),
        "shell_rows": shell_rows,
        "shell_exit_code_coverage": _ratio(weighted_exit, shell_rows),
        "shell_stdout_snippet_coverage": _ratio(weighted_stdout, shell_rows),
        "shell_stderr_snippet_coverage": _ratio(weighted_stderr, shell_rows),
        "median_observation_events_per_run": median(event_counts) if event_counts else 0,
        "pilot_gates": {
            "median_observation_events_per_run_min": 10,
            "median_observation_events_per_run_passed": bool(event_counts and median(event_counts) >= 10),
        },
        "passed": all(report["passed"] for report in reports),
        "runs": reports,
    }


def _has_capture_field(row: dict[str, Any], key: str) -> bool:
    return key in row and row.get(key) is not None


def _ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator
