from __future__ import annotations

from pathlib import Path

from coding_data_collection.artifacts import artifact_completeness_report, write_run_manifest
from coding_data_collection.protocol import (
    REQUIRED_RUN_ARTIFACTS,
    RunStatus,
    VERSIONS,
    analysis_inclusion,
    protocol_manifest_payload,
)


def test_protocol_versions_are_declared() -> None:
    payload = protocol_manifest_payload(
        coding_progress_ledger_sha="a",
        coding_data_collection_sha="c",
    )
    assert payload["run_protocol_version"] == VERSIONS.run_protocol_version
    assert payload["observation_event_schema_version"] == "0.2.0"
    assert payload["ledger_wire_schema_version"].startswith("1.")
    assert sorted(payload) == [
        "artifact_layout_version",
        "benchmark_adapter_version",
        "coding_data_collection_sha",
        "coding_progress_ledger_sha",
        "ledger_wire_schema_version",
        "observation_event_schema_version",
        "run_protocol_version",
    ]


def test_artifact_completeness_varies_by_status(tmp_path: Path) -> None:
    for name in REQUIRED_RUN_ARTIFACTS:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert artifact_completeness_report(tmp_path, RunStatus.COMPLETED_SUCCESS)["complete"]

    partial = tmp_path / "partial"
    partial.mkdir()
    for name in (
        "task_metadata.json",
        "environment_manifest.json",
        "protocol_manifest.json",
        "run_manifest.json",
        "run_notes.md",
    ):
        (partial / name).write_text("", encoding="utf-8")
    assert artifact_completeness_report(partial, RunStatus.INFRASTRUCTURE_FAILURE)["complete"]


def test_analysis_inclusion_rules() -> None:
    assert analysis_inclusion(RunStatus.COMPLETED_FAILURE)["terminal_success"] is True
    assert analysis_inclusion(RunStatus.AGENT_TIMEOUT)["terminal_success"] is False
    assert analysis_inclusion(RunStatus.AGENT_TIMEOUT)["process_dynamics"] is True


def test_write_run_manifest_records_cost_metrics(tmp_path: Path) -> None:
    write_run_manifest(
        tmp_path,
        run_id="run1",
        run_status=RunStatus.AGENT_TIMEOUT,
        final_success=None,
        termination_reason="agent_timeout",
        metrics={"tokens": 10, "tool_calls": 2, "estimated_cost_usd": 0.01},
    )
    text = (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    assert '"tokens": 10' in text
    assert '"estimated_cost_usd": 0.01' in text
    assert '"started_at":' in text
    assert '"ended_at":' in text
    assert '"wallclock_seconds": 0.0' in text
