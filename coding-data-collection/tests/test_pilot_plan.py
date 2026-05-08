from __future__ import annotations

import csv
from pathlib import Path

from coding_data_collection.pilot_plan import PilotArm, build_pilot_plan, parse_arm


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "task_id", "selected_for_pilot", "large_download_or_build"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _task(root: Path, name: str, *, dockerfile: bool = True) -> None:
    task_dir = root / name
    task_dir.mkdir(parents=True)
    if dockerfile:
        (task_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task_dir / "run-tests.sh").write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")


def test_pilot_plan_rejects_harbor_tasks_for_strict_hf_path(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    _task(root, "local-task")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "local-task",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            },
            {
                "source": "terminal_bench_harbor",
                "task_id": "terminal-bench/remote-task",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            },
        ],
    )

    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[PilotArm("a", "true"), PilotArm("b", "false")],
        expected_tasks=2,
    )

    assert not plan["passed"]
    assert any("terminal-bench/remote-task: source terminal_bench_harbor" in item for item in plan["blockers"])
    assert any("terminal-bench/remote-task: no extracted task directory" in item for item in plan["blockers"])


def test_pilot_plan_builds_two_arm_commands_for_valid_local_tasks(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    _task(root, "task-one")
    _task(root, "task-two")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            },
            {
                "source": "terminal_bench_hf",
                "task_id": "task-two",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            },
        ],
    )

    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[PilotArm("arm-a", "echo inspect"), PilotArm("arm-b", "echo validate")],
        expected_tasks=2,
    )

    assert plan["passed"]
    assert plan["planned_run_count"] == 4
    command = plan["planned_runs"][0]["command"]
    assert "scripts/run_docker_substrate_smoke.py" in command
    assert "--agent-command" in command
    assert "echo inspect" in command
    assert "--expect-verifier-failure" in command


def test_pilot_plan_accepts_compose_style_client_dockerfile(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    task_dir = root / "task-one"
    (task_dir / "client").mkdir(parents=True)
    (task_dir / "client" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task_dir / "run-tests.sh").write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            }
        ],
    )

    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[PilotArm("arm-a", "true"), PilotArm("arm-b", "false")],
        expected_tasks=1,
    )

    assert plan["passed"]
    assert plan["planned_run_count"] == 2


def test_pilot_plan_builds_model_tool_loop_commands_for_typed_arms(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    actions = tmp_path / "actions.jsonl"
    actions.write_text('{"thought":"done","action":{"type":"done","summary":"done"}}\n', encoding="utf-8")
    _task(root, "task-one")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            }
        ],
    )

    arm = parse_arm(f"strong:type=model_tool_loop,model=scripted,scripted_actions={actions},max_steps=12")
    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[arm, PilotArm("smoke", "true")],
        expected_tasks=1,
    )

    assert plan["passed"]
    model_run = [run for run in plan["planned_runs"] if run["backend"] == "model_tool_loop"][0]
    assert model_run["eligible_for_L_gate"] is False
    assert "scripts/run_model_agent_pilot.py" in model_run["command"]
    assert "--scripted-actions" in model_run["command"]
    assert "scripted" in model_run["command"]
    assert "--max-steps" in model_run["command"]


def test_pilot_plan_marks_provider_model_tool_loop_as_gate_eligible(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    _task(root, "task-one")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            }
        ],
    )

    arm = parse_arm("strong:type=model_tool_loop,client=provider,model_provider=command,model=real,max_steps=12")
    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[arm, PilotArm("smoke", "true")],
        expected_tasks=1,
    )

    assert plan["passed"]
    model_run = [run for run in plan["planned_runs"] if run["backend"] == "model_tool_loop"][0]
    assert model_run["eligible_for_L_gate"] is True
    assert "--client" in model_run["command"]
    assert "provider" in model_run["command"]


def test_pilot_plan_blocks_missing_dockerfile_and_wrong_arm_count(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    _task(root, "task-one", dockerfile=False)
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            }
        ],
    )

    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[PilotArm("arm-a", "true")],
        expected_tasks=1,
    )

    assert not plan["passed"]
    assert "pilot requires exactly 2 arms, found 1" in plan["blockers"]
    assert any("missing Dockerfile or client/Dockerfile" in item for item in plan["blockers"])


def test_pilot_plan_supports_explicit_three_arm_plan(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    root = tmp_path / "tasks"
    _task(root, "task-one")
    _write_scores(
        scores,
        [
            {
                "source": "terminal_bench_hf",
                "task_id": "task-one",
                "selected_for_pilot": True,
                "large_download_or_build": False,
            }
        ],
    )

    plan = build_pilot_plan(
        candidate_scores_path=scores,
        task_roots=[root],
        run_root=tmp_path / "runs",
        arms=[
            PilotArm("arm-a", "true"),
            PilotArm("arm-b", "true"),
            PilotArm("arm-c", "true"),
        ],
        expected_tasks=1,
        expected_arms=3,
    )

    assert plan["passed"]
    assert plan["expected_arms"] == 3
    assert plan["planned_run_count"] == 3
