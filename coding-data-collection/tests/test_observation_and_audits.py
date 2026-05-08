from __future__ import annotations

from pathlib import Path

from coding_data_collection.audits import scan_agent_workspace_for_leakage
from coding_data_collection.observation import build_observation_events


def test_observation_events_put_verifier_after_transcript() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "pytest -q",
            "exit_code": 1,
            "obs_snippet": "AssertionError: nope",
        },
        {"step": 2, "ts": "2026-05-05T00:00:01Z", "kind": "done"},
    ]
    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=1)
    types = [event["event_type"] for event in events]
    verifier = [event for event in events if event["event_type"] == "verifier_fail"][0]
    assert "validation_attempt" in types
    assert "validation_fail_observed" in types
    assert "verifier_disagreement" in types
    assert verifier["step"] == 3
    assert verifier["payload"]["visible_to_agent"] is False


def test_verifier_phase_shell_observations_are_not_agent_visible() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "summary": "verifier_phase",
            "command": "pytest /task/tests",
            "exit_code": 1,
            "stdout_snippet": "FAILED tests/test_outputs.py::test_hidden",
            "stderr_snippet": "",
        },
        {"step": 2, "ts": "2026-05-05T00:00:01Z", "kind": "done"},
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=1)
    verifier_attempts = [event for event in events if event["event_type"] == "validation_attempt"]
    verifier_errors = [event for event in events if event["event_type"] == "error_observed"]

    assert verifier_attempts
    assert all(event["payload"]["visible_to_agent"] is False for event in verifier_attempts + verifier_errors)
    assert all(event["source_artifact"] == "verifier_output.txt" for event in verifier_attempts + verifier_errors)


def test_successful_shell_with_error_text_is_not_error_observed() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "summary": "verifier_phase",
            "command": "bash /task/run-tests.sh",
            "exit_code": 0,
            "stdout_snippet": "2 passed",
            "stderr_snippet": "error: Project is already initialized in `/app`",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=0)

    assert "error_observed" not in {event["event_type"] for event in events}


def test_generic_check_word_is_not_validation_attempt() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "python - <<'PY'\nprint('check available modules')\nPY",
            "exit_code": 0,
            "stdout_snippet": "check available modules",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=None)

    assert "validation_attempt" not in {event["event_type"] for event in events}


def test_cmp_and_final_check_are_validation_attempts() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "cmp -s out.txt expected.txt && echo final-check-ok",
            "exit_code": 0,
            "stdout_snippet": "final-check-ok",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=None)

    assert "validation_attempt" in {event["event_type"] for event in events}


def test_direct_smoke_success_is_validation_attempt() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "python -m py_compile app.py && ./process_data.sh",
            "exit_code": 0,
            "stdout_snippet": "Data processed successfully!",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=None)

    assert "validation_attempt" in {event["event_type"] for event in events}


def test_pip_version_failure_is_validation_failure() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "python3 -m pip --version && pip3 --version",
            "exit_code": 1,
            "stderr_snippet": "No module named pip",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=None)
    types = {event["event_type"] for event in events}

    assert "validation_attempt" in types
    assert "validation_fail_observed" in types


def test_basic_task_script_test_failure_is_validation_failure() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "shell",
            "command": "python abmil_assignment.py",
            "exit_code": 1,
            "stdout_snippet": "Running basic ABMIL test...",
            "stderr_snippet": "Traceback",
        }
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=None)
    types = {event["event_type"] for event in events}

    assert "validation_attempt" in types
    assert "validation_fail_observed" in types


def test_leakage_scanner_detects_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    assert scan_agent_workspace_for_leakage(tmp_path)["passed"]
    (tmp_path / "solution.sh").write_text("# oracle", encoding="utf-8")
    report = scan_agent_workspace_for_leakage(tmp_path)
    assert report["passed"] is False
    assert "solution.sh" in report["leakage_hits"]
