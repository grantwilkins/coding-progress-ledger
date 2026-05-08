from __future__ import annotations

from coding_data_collection.ledger import transcript_to_wire_events
from coding_data_collection.task_scoring import TaskScoreInput, score_task


def test_transcript_to_wire_events_uses_ledger_wire_schema() -> None:
    transcript = [
        {"step": 1, "kind": "shell", "command": "pytest -q", "ts": "2026-05-05T00:00:00Z"},
        {"step": 2, "kind": "edit_file", "path": "app.py", "summary": "edit app"},
    ]
    events = transcript_to_wire_events(transcript, run_id="r1")
    assert events[0]["schema_version"].startswith("1.")
    assert events[0]["ledger_ops"][0]["category"] == "validation"
    assert events[1]["ledger_ops"][0]["category"] == "product"


def test_task_scoring_penalizes_operational_risk() -> None:
    rich = TaskScoreInput(
        task_id="rich",
        expected_runtime_bucket=4,
        expected_validation_visibility=5,
        expected_file_edit_complexity=4,
        expected_environment_complexity=3,
        expected_failure_modes=5,
        oracle_test_leakage_risk=1,
        docker_feasibility=5,
        requires_internet=False,
        large_download_or_build=False,
    )
    risky = TaskScoreInput(
        task_id="risky",
        expected_runtime_bucket=4,
        expected_validation_visibility=5,
        expected_file_edit_complexity=4,
        expected_environment_complexity=3,
        expected_failure_modes=5,
        oracle_test_leakage_risk=5,
        docker_feasibility=1,
        requires_internet=True,
        large_download_or_build=True,
    )
    assert score_task(rich).pilot_priority > score_task(risky).pilot_priority

