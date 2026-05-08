"""
Claim:
- observation_events preserve transcript/verifier structure as additive,
  prefix-indexed events without redefining the ledger.
- expected-path heuristics identify likely wrong-file writes and
  missing-expected-file done claims on tb_live_v2 tasks.

Plausible wrong implementations:
- emit verifier_fail at the same step as the transcript, making
  post-verifier outcome visible to prefix checkpoints.
- miss `solution.sh` oracle reads or `done` claims.
- classify a wrong-path product write as expected.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from coding_estimator.runner.observation_events import (
    build_observation_events,
    expected_product_paths_for_task,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_expected_product_paths_for_task_reads_test_literals(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "tasks" / "tb_live_v2" / "demo" / "tests" / "test_outputs.py",
        textwrap.dedent(
            """\
            from pathlib import Path
            APP = Path.cwd() / "server.py"
            def test_answer():
                assert APP.is_file()
            """
        ),
    )
    assert expected_product_paths_for_task("demo", repo_root=repo) == {"server.py"}


def test_expected_product_paths_support_dotfiles_extensionless_and_nested_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "tasks" / "tb_live_v2" / "demo2" / "tests" / "test_outputs.py",
        textwrap.dedent(
            """\
            from pathlib import Path
            WS = Path.cwd()
            ENV = WS / ".env"
            MAKEFILE = WS / "Makefile"
            WIDGET = WS / "lib" / "widget.py"
            """
        ),
    )
    assert expected_product_paths_for_task("demo2", repo_root=repo) == {
        ".env",
        "Makefile",
        "lib/widget.py",
    }


def test_build_observation_events_emits_prefix_and_terminal_events(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "tasks" / "tb_live_v2" / "demo" / "tests" / "test_outputs.py",
        "from pathlib import Path\nAPP = Path.cwd() / 'server.py'\n",
    )
    run_dir = tmp_path / "run"
    _write(
        run_dir / "transcript.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "ts": "2026-05-05T00:00:01Z",
                        "kind": "read_file",
                        "path": "solution.sh",
                        "summary": "read solution.sh",
                    }
                ),
                json.dumps(
                    {
                        "step": 2,
                        "ts": "2026-05-05T00:00:02Z",
                        "kind": "write_file",
                        "path": "/tmp/app/wrong_name.py",
                        "summary": "write wrong file",
                    }
                ),
                json.dumps(
                    {
                        "step": 3,
                        "ts": "2026-05-05T00:00:03Z",
                        "kind": "shell",
                        "command": "pytest -q",
                        "summary": "run tests",
                        "exit_code": 1,
                        "obs_snippet": "AssertionError: server.py not created",
                    }
                ),
                json.dumps(
                    {
                        "step": 4,
                        "ts": "2026-05-05T00:00:04Z",
                        "kind": "done",
                        "summary": "implemented server",
                    }
                ),
            ]
        )
        + "\n",
    )
    _write(
        run_dir / "run_manifest.json",
        json.dumps(
            {
                "task_id": "demo",
                "final_success": False,
                "termination_reason": "verifier_fail",
                "end_time": "2026-05-05T00:00:05Z",
            }
        ),
    )
    _write(run_dir / "workspace_path.txt", str(run_dir))
    _write(run_dir / "verifier_output.txt", "AssertionError: server.py not created\n")

    events = build_observation_events(
        run_dir=run_dir,
        run_id="demo__armA__1",
        task_id="demo",
        repo_root=repo,
    )
    by_type = {}
    for event in events:
        by_type.setdefault(event["event_type"], []).append(event)

    assert by_type["solution_oracle_read"][0]["step"] == 1
    assert by_type["product_file_written"][0]["payload"]["matches_expected_path"] is False
    assert by_type["validation_attempt"][0]["step"] == 3
    assert by_type["validation_fail_observed"][0]["step"] == 3
    assert by_type["error_observed"][0]["step"] == 3
    assert by_type["agent_claims_done"][0]["step"] == 4
    assert by_type["expected_file_missing"][0]["step"] == 4
    assert by_type["verifier_fail"][0]["step"] == 5
    assert by_type["verifier_disagreement"][0]["step"] == 5


def test_relative_path_matching_rejects_nested_wrong_location(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / "tasks" / "tb_live_v2" / "demo3" / "tests" / "test_outputs.py",
        "from pathlib import Path\nAPP = Path.cwd() / 'server.py'\n",
    )
    run_dir = tmp_path / "run3"
    _write(run_dir / "workspace_path.txt", str(run_dir))
    _write(
        run_dir / "transcript.jsonl",
        json.dumps(
            {
                "step": 1,
                "ts": "2026-05-05T00:00:01Z",
                "kind": "write_file",
                "path": str(run_dir / "app" / "server.py"),
                "summary": "write nested server.py",
            }
        )
        + "\n",
    )
    _write(run_dir / "run_manifest.json", json.dumps({"task_id": "demo3", "final_success": True}))
    events = build_observation_events(
        run_dir=run_dir,
        run_id="demo3",
        task_id="demo3",
        repo_root=repo,
    )
    write_event = next(event for event in events if event["event_type"] == "product_file_written")
    assert write_event["payload"]["relative_path"] == "app/server.py"
    assert write_event["payload"]["matches_expected_path"] is False
