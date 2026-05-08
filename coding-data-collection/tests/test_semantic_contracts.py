"""
Claim:
The collection harness preserves the scientific boundary between visible
agent-prefix evidence and post-run verifier/oracle information. Hidden
benchmark materials must be detected if they enter the agent workspace,
terminal verifier facts must remain post-terminal, unresolved verifier state
must not become a model failure, and task scoring must prefer rich but
operationally feasible trajectories.

Plausible wrong implementations:
- Treat `tests/` as an ordinary workspace directory and miss hidden-test leakage.
- Emit verifier or expected-file observations at the done step instead of the
  post-transcript step, leaking terminal facts into preterminal checkpoints.
- Compare expected and written product paths without normalizing `./` prefixes.
- Mark a run with no verifier result as `completed_failure`, contaminating
  terminal-success labels with infrastructure failures.
- Let operational risk bonuses overwhelm trajectory-richness scoring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from coding_data_collection.audits import scan_agent_workspace_for_leakage
from coding_data_collection.ledger import transcript_to_wire_events
from coding_data_collection.observation import build_observation_events
from coding_data_collection.protocol import (
    PROCESS_DYNAMICS_REQUIRED_ARTIFACTS,
    REQUIRED_RUN_ARTIFACTS,
    RunStatus,
    required_artifacts_for_status,
)
from coding_data_collection.task_scoring import TaskScoreInput, score_task


def _event_types(events: list[dict]) -> list[str]:
    return [event["event_type"] for event in events]


def test_hidden_tests_directory_is_leakage_not_agent_workspace_content(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_outputs.py").write_text("def test_hidden(): pass\n", encoding="utf-8")
    (tmp_path / "renamed_notes.md").write_text("This contains the oracle solution.", encoding="utf-8")

    report = scan_agent_workspace_for_leakage(tmp_path)

    assert report["passed"] is False
    assert "tests" in report["leakage_hits"]
    assert "tests/test_outputs.py" in report["leakage_hits"]
    assert "renamed_notes.md" in report["leakage_hits"]


def test_prepare_run_excludes_nested_hidden_paths_and_symlinks(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    run_dir = tmp_path / "run"
    (task_dir / "src").mkdir(parents=True)
    (task_dir / "src" / "main.py").write_text("print('visible')\n", encoding="utf-8")
    (run_dir / "agent_workspace").mkdir(parents=True)
    (run_dir / "agent_workspace" / "stale_result.txt").write_text("old agent output\n", encoding="utf-8")
    (task_dir / "src" / "tests").mkdir()
    (task_dir / "src" / "tests" / "test_outputs.py").write_text("assert secret\n", encoding="utf-8")
    (task_dir / "solution_reference").mkdir()
    (task_dir / "solution_reference" / "answer.py").write_text("secret\n", encoding="utf-8")
    (task_dir / "solution.yaml").write_text("- command: echo secret\n", encoding="utf-8")
    (task_dir / "Dockerfile").write_text("terminal-bench-canary should not enter agent workspace\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text("instruction: visible task text\n", encoding="utf-8")
    (task_dir / "visible_link").symlink_to(task_dir / "src" / "main.py")
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_run.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--task-dir",
            str(task_dir),
            "--run-dir",
            str(run_dir),
            "--collection-root",
            str(Path(__file__).resolve().parents[1]),
            "--ledger-root",
            str(tmp_path),
            "--estimator-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    workspace = run_dir / "agent_workspace"
    metadata = json.loads((run_dir / "task_metadata.json").read_text(encoding="utf-8"))
    assert (workspace / "src" / "main.py").is_file()
    assert not (workspace / "stale_result.txt").exists()
    assert not (workspace / "src" / "tests").exists()
    assert not (workspace / "solution_reference").exists()
    assert not (workspace / "solution.yaml").exists()
    assert not (workspace / "Dockerfile").exists()
    assert not (workspace / "task.yaml").exists()
    assert not (workspace / "visible_link").exists()
    assert "src/tests/test_outputs.py" in metadata["skipped_paths"]
    assert "solution.yaml" in metadata["skipped_paths"]
    assert "Dockerfile" in metadata["skipped_paths"]
    assert "task.yaml" in metadata["skipped_paths"]
    assert "visible_link" in metadata["skipped_paths"]


def test_hidden_test_read_is_oracle_artifact_and_post_terminal_events_are_not_at_done_step() -> None:
    transcript = [
        {
            "step": 4,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "read_file",
            "path": "tests/test_outputs.py",
            "obs_snippet": "assert secret_case",
        },
        {
            "step": 7,
            "ts": "2026-05-05T00:00:01Z",
            "kind": "done",
        },
    ]

    events = build_observation_events(transcript, run_id="r1", verifier_exit_code=1)
    event_types = _event_types(events)
    terminal_steps = {
        event["event_type"]: event["step"]
        for event in events
        if event["event_type"] in {"verifier_fail", "verifier_disagreement"}
    }
    visibility = {
        event["event_type"]: event["payload"].get("visible_to_agent")
        for event in events
        if event["event_type"] in {"oracle_artifact_read", "agent_claims_done", "verifier_fail"}
    }

    assert "oracle_artifact_read" in event_types
    assert terminal_steps == {"verifier_fail": 8, "verifier_disagreement": 8}
    assert visibility == {
        "oracle_artifact_read": True,
        "agent_claims_done": True,
        "verifier_fail": False,
    }


def test_expected_file_missing_uses_normalized_paths_before_declaring_missing() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "write_file",
            "path": "./src/app.py",
        },
        {
            "step": 2,
            "ts": "2026-05-05T00:00:01Z",
            "kind": "done",
        },
    ]

    events = build_observation_events(
        transcript,
        run_id="r1",
        verifier_exit_code=0,
        expected_paths={"src/app.py"},
    )

    assert "expected_file_missing" not in _event_types(events)


def test_expected_file_missing_reports_only_still_missing_paths() -> None:
    transcript = [
        {
            "step": 1,
            "ts": "2026-05-05T00:00:00Z",
            "kind": "write_file",
            "path": "src/app.py",
        },
        {
            "step": 2,
            "ts": "2026-05-05T00:00:01Z",
            "kind": "done",
        },
    ]

    events = build_observation_events(
        transcript,
        run_id="r1",
        verifier_exit_code=0,
        expected_paths={"src/app.py", "src/config.py"},
    )
    missing = [event for event in events if event["event_type"] == "expected_file_missing"][0]

    assert missing["payload"]["missing_paths"] == ["src/config.py"]


def test_artifact_requirements_are_status_specific_not_all_partial() -> None:
    assert required_artifacts_for_status(RunStatus.COMPLETED_FAILURE) == REQUIRED_RUN_ARTIFACTS
    assert required_artifacts_for_status(RunStatus.AGENT_TIMEOUT) == PROCESS_DYNAMICS_REQUIRED_ARTIFACTS
    assert "verifier_output.txt" not in required_artifacts_for_status(RunStatus.AGENT_TIMEOUT)
    assert "verifier_output.txt" in required_artifacts_for_status(RunStatus.VERIFIER_CRASH)
    assert required_artifacts_for_status(RunStatus.QUARANTINED_LEAKAGE) == (
        "task_metadata.json",
        "environment_manifest.json",
        "protocol_manifest.json",
        "run_manifest.json",
        "run_notes.md",
    )


def test_prepare_run_hides_protected_and_sanitizes_visible_canary(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "maze_game.sh").write_text(
        "#!/bin/bash\n# terminal-bench-canary GUID secret\npython3 /protected/maze_server.py\n",
        encoding="utf-8",
    )
    (task_dir / "protected").mkdir()
    (task_dir / "protected" / "maze_server.py").write_text(
        "# terminal-bench-canary GUID secret\nprint('hidden')\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_run.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--task-dir",
            str(task_dir),
            "--run-dir",
            str(run_dir),
            "--collection-root",
            str(Path(__file__).resolve().parents[1]),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    workspace = run_dir / "agent_workspace"
    assert not (workspace / "protected").exists()
    assert "terminal-bench-canary" not in (workspace / "maze_game.sh").read_text(encoding="utf-8")


def test_finalize_without_verifier_result_is_infrastructure_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"step": 1, "kind": "done", "ts": "2026-05-05T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "finalize_run.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "r1",
            "--skip-sidecar",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "infrastructure_failure"
    assert manifest["analysis_inclusion"]["terminal_success"] is False


def test_task_priority_increases_with_richness_but_decreases_with_equal_risk() -> None:
    base = dict(
        expected_runtime_bucket=3,
        expected_validation_visibility=2,
        expected_file_edit_complexity=2,
        expected_environment_complexity=1,
        expected_failure_modes=2,
        oracle_test_leakage_risk=1,
        docker_feasibility=4,
        requires_internet=False,
        large_download_or_build=False,
    )
    low_richness = score_task(TaskScoreInput(task_id="low", **base))
    richer_same_risk = score_task(
        TaskScoreInput(
            task_id="rich",
            **{
                **base,
                "expected_validation_visibility": 4,
                "expected_failure_modes": 4,
            },
        )
    )
    same_richness_higher_risk = score_task(
        TaskScoreInput(
            task_id="risky",
            **{
                **base,
                "oracle_test_leakage_risk": 3,
                "requires_internet": True,
            },
        )
    )

    assert richer_same_risk.pilot_priority > low_richness.pilot_priority
    assert same_richness_higher_risk.pilot_priority < low_richness.pilot_priority


def test_task_scoring_exact_hand_checked_arithmetic() -> None:
    score = score_task(
        TaskScoreInput(
            task_id="case",
            expected_runtime_bucket=2,
            expected_validation_visibility=3,
            expected_file_edit_complexity=4,
            expected_environment_complexity=5,
            expected_failure_modes=1,
            oracle_test_leakage_risk=2,
            docker_feasibility=3,
            requires_internet=True,
            large_download_or_build=True,
        )
    )

    assert score.trajectory_richness == 2 + 2 * 3 + 4 + 5 + 2 * 1
    assert score.operational_risk == 3 * 2 + (5 - 3) + 3 + 2
    assert score.pilot_priority == score.trajectory_richness - score.operational_risk


def test_ledger_wire_events_preserve_sparse_steps_and_conservative_inferred_ops() -> None:
    transcript = [
        {"step": 2, "kind": "shell", "command": "pytest -q", "ts": "2026-05-05T00:00:00Z"},
        {"step": 9, "kind": "edit_file", "path": "src/app.py", "summary": "edit app"},
    ]

    events = transcript_to_wire_events(transcript, run_id="r1")

    assert [event["step"] for event in events] == [2, 9]
    assert events[0]["agent_step"] == transcript[0]
    assert [op["op"] for op in events[0]["ledger_ops"]] == ["add", "start"]
    assert events[0]["ledger_ops"][0]["id"] == events[0]["ledger_ops"][1]["id"]
    assert events[1]["ledger_ops"][0]["category"] == "product"
