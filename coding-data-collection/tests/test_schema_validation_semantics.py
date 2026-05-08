"""
Claim:
The collection validator enforces protocol compatibility at the run-artifact
boundary before trace collection. Completed runs require terminal artifacts,
non-terminal infrastructure states cannot enter terminal-success analysis,
post-terminal observations must stay after the transcript, and ledger wire
events must match the sidecar's semantic op contract.

Plausible wrong implementations:
- Validate JSON shape but ignore protocol/schema version drift.
- Treat a completed run as valid when verifier output or ledger artifacts are missing.
- Let infrastructure/setup failures carry terminal_success=true in run_manifest.json.
- Emit verifier observations at the agent's done step, leaking terminal facts into prefixes.
- Accept ledger ops that look well-formed but cannot replay through the sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

from coding_data_collection.artifacts import write_json, write_run_manifest
from coding_data_collection.ledger import transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events, write_jsonl
from coding_data_collection.protocol import REQUIRED_RUN_ARTIFACTS, RunStatus, VERSIONS
from coding_data_collection.validation import validate_run_dir


def _write_minimal_completed_run(run_dir: Path) -> None:
    transcript = [
        {
            "step": 1,
            "kind": "shell",
            "command": "pytest -q",
            "exit_code": 0,
            "ts": "2026-05-05T00:00:00Z",
        },
        {"step": 2, "kind": "done", "ts": "2026-05-05T00:00:01Z"},
    ]
    (run_dir / "task.md").write_text("# Task\n\nFix it.\n", encoding="utf-8")
    write_json(
        run_dir / "task_metadata.json",
        {
            "run_protocol_version": VERSIONS.run_protocol_version,
            "artifact_layout_version": VERSIONS.artifact_layout_version,
            "benchmark_adapter_version": VERSIONS.benchmark_adapter_version,
            "task_dir": "tasks/example",
            "hidden_names": ["tests"],
            "skipped_paths": ["tests/test_outputs.py"],
        },
    )
    write_json(
        run_dir / "environment_manifest.json",
        {
            "run_protocol_version": VERSIONS.run_protocol_version,
            "artifact_layout_version": VERSIONS.artifact_layout_version,
            "agent_workspace": str(run_dir / "agent_workspace"),
        },
    )
    write_json(
        run_dir / "protocol_manifest.json",
        {
            **VERSIONS.to_dict(),
            "coding_progress_ledger_sha": "ledger",
            "coding_data_collection_sha": "collection",
        },
    )
    write_jsonl(run_dir / "transcript.jsonl", transcript)
    write_jsonl(
        run_dir / "observation_events.jsonl",
        build_observation_events(transcript, run_id="run1", verifier_exit_code=0),
    )
    write_wire_events(run_dir / "events.jsonl", transcript_to_wire_events(transcript, run_id="run1"))
    (run_dir / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "progress.csv").write_text("step,score\n1,1.0\n", encoding="utf-8")
    (run_dir / "progress_by_category.csv").write_text("step,category,score\n1,product,1.0\n", encoding="utf-8")
    write_json(run_dir / "summary_by_category.json", {"final_success": True})
    (run_dir / "verifier_output.txt").write_text("pass\n", encoding="utf-8")
    (run_dir / "run_notes.md").write_text("# Notes\n", encoding="utf-8")
    write_run_manifest(
        run_dir,
        run_id="run1",
        run_status=RunStatus.COMPLETED_SUCCESS,
        final_success=True,
        termination_reason="verifier_pass",
    )


def _messages(issues: list) -> list[str]:
    return [f"{issue.artifact}: {issue.message}" for issue in issues]


def test_complete_run_validates_only_when_all_terminal_artifacts_and_versions_match(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)

    assert validate_run_dir(tmp_path) == []

    (tmp_path / "verifier_output.txt").unlink()
    messages = _messages(validate_run_dir(tmp_path))

    assert "verifier_output.txt: missing for status completed_success" in messages


def test_schema_version_drift_is_rejected_before_trace_collection(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)
    protocol = json.loads((tmp_path / "protocol_manifest.json").read_text(encoding="utf-8"))
    protocol["observation_event_schema_version"] = "0.1.0"
    (tmp_path / "protocol_manifest.json").write_text(json.dumps(protocol), encoding="utf-8")

    messages = _messages(validate_run_dir(tmp_path))

    assert any("observation_event_schema_version" in message and "'0.2.0' was expected" in message for message in messages)


def test_infrastructure_failure_cannot_be_terminal_success_analysis(tmp_path: Path) -> None:
    for name in ("task_metadata.json", "environment_manifest.json", "protocol_manifest.json", "run_notes.md"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    write_run_manifest(
        tmp_path,
        run_id="run1",
        run_status=RunStatus.INFRASTRUCTURE_FAILURE,
        final_success=None,
        termination_reason="verifier_not_run",
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["analysis_inclusion"]["terminal_success"] = True
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    messages = _messages(validate_run_dir(tmp_path))

    assert any("infrastructure_failure cannot be included" in message for message in messages)


def test_post_terminal_observation_at_done_step_is_invalid(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)
    events = json.loads((tmp_path / "observation_events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    events["step"] = 2
    write_jsonl(tmp_path / "observation_events.jsonl", [events])

    messages = _messages(validate_run_dir(tmp_path))

    assert any("post-terminal event at transcript step 2" in message for message in messages)


def test_observation_schema_requires_visibility_payload(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)
    event = json.loads((tmp_path / "observation_events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    event["payload"].pop("visible_to_agent")
    write_jsonl(tmp_path / "observation_events.jsonl", [event])

    messages = _messages(validate_run_dir(tmp_path))

    assert any("visible_to_agent' is a required property" in message for message in messages)


def test_hidden_phase_observation_cannot_be_agent_visible(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)
    event = {
        "schema_version": VERSIONS.observation_event_schema_version,
        "run_id": "run1",
        "step": 3,
        "observed_ts": "2026-05-05T00:00:02Z",
        "source_artifact": "verifier_output.txt",
        "event_type": "validation_attempt",
        "payload": {"visible_to_agent": True},
    }
    write_jsonl(tmp_path / "observation_events.jsonl", [event])

    messages = _messages(validate_run_dir(tmp_path))

    assert any("hidden phase event must not be visible_to_agent" in message for message in messages)


def test_ledger_wire_schema_rejects_non_replayable_ops(tmp_path: Path) -> None:
    _write_minimal_completed_run(tmp_path)
    write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "schema_version": VERSIONS.ledger_wire_schema_version,
                "run_id": "run1",
                "step": 1,
                "timestamp": "2026-05-05T00:00:00Z",
                "ledger_ops": [{"op": "add", "id": "s1"}],
            }
        ],
    )

    messages = _messages(validate_run_dir(tmp_path))

    assert any("description' is a required property" in message for message in messages)
    assert any("category' is a required property" in message for message in messages)


def test_required_run_artifact_list_matches_schema_validated_completed_surface() -> None:
    schema_validated = set(REQUIRED_RUN_ARTIFACTS)
    for schema_name in (
        "run_manifest.json",
        "task_metadata.json",
        "environment_manifest.json",
        "protocol_manifest.json",
        "observation_events.jsonl",
        "events.jsonl",
    ):
        assert schema_name in schema_validated
