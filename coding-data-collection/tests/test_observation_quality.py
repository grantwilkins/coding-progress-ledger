"""
Claim:
Observation quality metrics are computed at the shell-row level. Captured but
empty stdout/stderr streams count as capture coverage, missing fields do not,
and terminal verifier facts must remain non-agent-visible.

Plausible wrong implementations:
- Count non-empty snippets instead of capture-field presence, penalizing valid
  commands that produced no output.
- Average per-run percentages instead of weighting by shell rows.
- Treat terminal verifier events as safe even when visible_to_agent is true.
- Treat verifier-phase validation/error observations as safe when visible_to_agent is true.
- Ignore JSON-schema/timing validation issues when declaring quality passed.
- Pass a wrong or empty run directory because zero shell rows divide to perfect coverage.
"""

from pathlib import Path

from coding_data_collection.artifacts import write_json, write_run_manifest
from coding_data_collection.ledger import transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events, write_jsonl
from coding_data_collection.observation_quality import corpus_observation_quality_report, observation_quality_report
from coding_data_collection.protocol import RunStatus, VERSIONS


def test_shell_capture_coverage_counts_empty_captured_streams(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        transcript=[
            {"step": 1, "kind": "shell", "command": "true", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {"step": 2, "kind": "shell", "command": "pytest", "exit_code": 1, "stdout_snippet": "failed"},
            {"step": 3, "kind": "done"},
        ],
        verifier_exit_code=1,
    )

    report = observation_quality_report(tmp_path)

    assert report["shell_rows"] == 2
    assert report["shell_exit_code_coverage"] == 1.0
    assert report["shell_stdout_snippet_coverage"] == 1.0
    assert report["shell_stderr_snippet_coverage"] == 0.5
    assert report["shell_stdout_nonempty_fraction"] == 0.5
    assert report["passed"] is True


def test_corpus_coverage_is_weighted_by_shell_rows(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write_run(
        run_a,
        transcript=[
            {"step": 1, "kind": "shell", "command": "true", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {"step": 2, "kind": "shell", "command": "true", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {"step": 3, "kind": "shell", "command": "true", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {"step": 4, "kind": "done"},
        ],
        verifier_exit_code=0,
    )
    _write_run(
        run_b,
        transcript=[
            {"step": 1, "kind": "shell", "command": "true", "exit_code": 0, "stdout_snippet": ""},
            {"step": 2, "kind": "done"},
        ],
        verifier_exit_code=0,
    )

    report = corpus_observation_quality_report([run_a, run_b])

    assert report["shell_rows"] == 4
    assert report["shell_stderr_snippet_coverage"] == 0.75
    assert report["median_observation_events_per_run"] == 2.0
    assert "runs" in report


def test_terminal_visible_to_agent_event_fails_quality(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        transcript=[
            {"step": 1, "kind": "shell", "command": "pytest", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {"step": 2, "kind": "done"},
        ],
        verifier_exit_code=0,
    )
    events = [
        {
            "schema_version": VERSIONS.observation_event_schema_version,
            "run_id": "run1",
            "step": 3,
            "observed_ts": "2026-05-05T00:00:02Z",
            "source_artifact": "verifier_output.txt",
            "event_type": "verifier_pass",
            "payload": {"exit_code": 0, "visible_to_agent": True},
        }
    ]
    write_jsonl(tmp_path / "observation_events.jsonl", events)

    report = observation_quality_report(tmp_path)

    assert report["observation_schema_valid"] is False
    assert report["terminal_events_visible_to_agent"] == 1
    assert report["passed"] is False


def test_hidden_phase_visible_to_agent_event_fails_quality(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        transcript=[
            {"step": 1, "kind": "shell", "summary": "agent_phase", "command": "true", "exit_code": 0, "stdout_snippet": "", "stderr_snippet": ""},
            {
                "step": 2,
                "kind": "shell",
                "summary": "verifier_phase",
                "command": "bash /task/run-tests.sh",
                "exit_code": 1,
                "stdout_snippet": "FAILED tests/test_outputs.py::test_hidden",
                "stderr_snippet": "",
                "visible_to_agent": True,
            },
            {"step": 3, "kind": "done"},
        ],
        verifier_exit_code=1,
    )

    report = observation_quality_report(tmp_path)

    assert report["hidden_phase_events_visible_to_agent"] > 0
    assert report["passed"] is False


def test_vacuous_run_dir_does_not_pass_quality(tmp_path: Path) -> None:
    report = observation_quality_report(tmp_path)

    assert sorted(report["missing_inputs"]) == ["observation_events.jsonl", "transcript.jsonl"]
    assert report["shell_rows"] == 0
    assert report["passed"] is False


def _write_run(run_dir: Path, *, transcript: list[dict], verifier_exit_code: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
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
    write_json(run_dir / "environment_manifest.json", {"run_protocol_version": VERSIONS.run_protocol_version})
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
        build_observation_events(transcript, run_id="run1", verifier_exit_code=verifier_exit_code),
    )
    write_wire_events(run_dir / "events.jsonl", transcript_to_wire_events(transcript, run_id="run1"))
    (run_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    (run_dir / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "progress.csv").write_text("step,score\n1,1.0\n", encoding="utf-8")
    (run_dir / "progress_by_category.csv").write_text("step,category,score\n1,product,1.0\n", encoding="utf-8")
    write_json(run_dir / "summary_by_category.json", {"final_success": verifier_exit_code == 0})
    (run_dir / "verifier_output.txt").write_text("pass\n" if verifier_exit_code == 0 else "fail\n", encoding="utf-8")
    (run_dir / "run_notes.md").write_text("# Notes\n", encoding="utf-8")
    write_run_manifest(
        run_dir,
        run_id="run1",
        run_status=RunStatus.COMPLETED_SUCCESS if verifier_exit_code == 0 else RunStatus.COMPLETED_FAILURE,
        final_success=verifier_exit_code == 0,
        termination_reason="verifier_pass" if verifier_exit_code == 0 else "verifier_fail",
    )
