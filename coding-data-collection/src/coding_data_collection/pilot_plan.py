from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .docker_substrate import task_dockerfile_path


DEFAULT_TASK_ROOTS = (
    Path("/private/tmp/houdini_tb_d_smoke"),
    Path("/private/tmp/houdini_tb_hf"),
)


@dataclass(frozen=True)
class PilotArm:
    name: str
    agent_command: str = ""
    backend: str = "shell_command"
    client: str = "scripted"
    model_provider: str = "command"
    model: str = ""
    scripted_actions: str = ""
    model_command: str = ""
    max_steps: int = 80


def parse_arm(text: str) -> PilotArm:
    if ":type=" in text:
        name, spec = text.split(":type=", 1)
        fields = _parse_arm_fields(f"type={spec}")
        backend = fields.get("type", "").strip()
        if backend == "shell_command":
            command = fields.get("command", "").strip()
            if not name.strip() or not command:
                raise ValueError("shell_command arm requires name and command")
            return PilotArm(name=_safe_id(name.strip()), backend=backend, agent_command=command)
        if backend == "model_tool_loop":
            if not name.strip():
                raise ValueError("model_tool_loop arm requires name")
            scripted_actions = fields.get("scripted_actions", "").strip()
            return PilotArm(
                name=_safe_id(name.strip()),
                backend=backend,
                client=fields.get("client", "scripted" if scripted_actions else "provider").strip(),
                model_provider=fields.get("model_provider", "command").strip(),
                model=fields.get("model", "scripted").strip() or "scripted",
                scripted_actions=scripted_actions,
                model_command=fields.get("model_command", "").strip(),
                max_steps=int(fields.get("max_steps", "80")),
            )
        raise ValueError(f"unsupported arm backend: {backend!r}")

    name, sep, command = text.partition("=")
    if not sep or not name.strip() or not command.strip():
        raise ValueError("--arm must have the form name=agent-command")
    return PilotArm(name=_safe_id(name.strip()), backend="shell_command", agent_command=command.strip())


def build_pilot_plan(
    *,
    candidate_scores_path: Path,
    task_roots: list[Path],
    run_root: Path,
    arms: list[PilotArm],
    expected_tasks: int = 12,
    expected_arms: int = 2,
    allow_harbor: bool = False,
    network_exceptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    network_exceptions = network_exceptions or {}
    selected = _selected_rows(candidate_scores_path)
    blockers: list[str] = []
    warnings: list[str] = []

    if len(selected) != expected_tasks:
        blockers.append(f"selected task count is {len(selected)}, expected {expected_tasks}")
    if len(arms) != expected_arms:
        blockers.append(f"pilot requires exactly {expected_arms} arms, found {len(arms)}")
    if len({arm.name for arm in arms}) != len(arms):
        blockers.append("pilot arm names must be unique")
    if arms and not any(arm.backend == "model_tool_loop" for arm in arms):
        warnings.append("all configured arms are shell_command protocol-smoke arms; they are not eligible for L gate metrics")
    for arm in arms:
        if arm.backend == "model_tool_loop" and arm.client not in {"scripted", "provider"}:
            blockers.append(f"{arm.name}: model_tool_loop client must be scripted or provider")
        if arm.backend == "model_tool_loop" and arm.client == "scripted" and not arm.scripted_actions:
            blockers.append(f"{arm.name}: scripted model_tool_loop arm requires scripted_actions")

    task_entries: list[dict[str, Any]] = []
    planned_runs: list[dict[str, Any]] = []
    for row in selected:
        task_id = row["task_id"]
        source = row["source"]
        issues: list[str] = []
        task_dir = _find_task_dir(task_id, task_roots)
        if source != "terminal_bench_hf" and not allow_harbor:
            issues.append(f"{task_id}: source {source} is not allowed for strict hf_archive_custom pilot")
        if task_dir is None:
            issues.append(f"{task_id}: no extracted task directory found under task roots")
        elif task_dockerfile_path(task_dir) is None:
            issues.append(f"{task_id}: missing Dockerfile or client/Dockerfile required by run_docker_substrate_smoke.py")
        if task_dir is not None and not (task_dir / "run-tests.sh").is_file():
            issues.append(f"{task_id}: missing run-tests.sh")
        if _boolish(row.get("large_download_or_build")) and task_id not in network_exceptions:
            warnings.append(f"{task_id}: large_download_or_build=true; verifier network/cache policy should be explicit")

        task_entries.append(
            {
                "task_id": task_id,
                "source": source,
                "task_dir": str(task_dir) if task_dir is not None else None,
                "issues": issues,
            }
        )
        blockers.extend(issues)
        if issues or task_dir is None:
            continue

        for arm in arms:
            run_id = f"{_safe_id(task_id)}__{arm.name}"
            command = _planned_run_command(arm=arm, task_dir=task_dir, run_dir=run_root / run_id, run_id=run_id)
            if task_id in network_exceptions:
                command.extend(
                    [
                        "--allow-verifier-network",
                        "--network-exception-reason",
                        network_exceptions[task_id],
                    ]
                )
            planned_runs.append(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "arm": arm.name,
                    "backend": arm.backend,
                    "eligible_for_L_gate": _eligible_for_l_gate(arm),
                    "task_dir": str(task_dir),
                    "command": command,
                }
            )

    expected_runs = expected_tasks * expected_arms
    if len(planned_runs) != expected_runs:
        blockers.append(f"planned run count is {len(planned_runs)}, expected {expected_runs}")

    return {
        "candidate_scores_path": str(candidate_scores_path),
        "task_roots": [str(path) for path in task_roots],
        "run_root": str(run_root),
        "expected_tasks": expected_tasks,
        "expected_arms": expected_arms,
        "expected_runs": expected_runs,
        "selected_task_count": len(selected),
        "planned_run_count": len(planned_runs),
        "arms": [
            {
                "name": arm.name,
                "backend": arm.backend,
                "client": arm.client,
                "agent_command": arm.agent_command,
                "model_provider": arm.model_provider,
                "model": arm.model,
                "scripted_actions": arm.scripted_actions,
                "model_command": arm.model_command,
                "max_steps": arm.max_steps,
                "eligible_for_L_gate": _eligible_for_l_gate(arm),
            }
            for arm in arms
        ],
        "tasks": task_entries,
        "planned_runs": planned_runs,
        "warnings": sorted(set(warnings)),
        "blockers": sorted(set(blockers)),
        "passed": not blockers,
    }


def _planned_run_command(*, arm: PilotArm, task_dir: Path, run_dir: Path, run_id: str) -> list[str]:
    common = [
        sys.executable,
        "scripts/run_model_agent_pilot.py" if arm.backend == "model_tool_loop" else "scripts/run_docker_substrate_smoke.py",
        "--task-dir",
        str(task_dir),
        "--run-dir",
        str(run_dir),
        "--image-tag",
        f"cdc-tb-{run_id}"[:120],
    ]
    if arm.backend == "model_tool_loop":
        command = [
            *common,
            "--client",
            arm.client,
            "--model-provider",
            arm.model_provider,
            "--model-name",
            arm.model,
            "--max-steps",
            str(arm.max_steps),
            "--verifier-command",
            "bash /task/run-tests.sh",
            "--expect-verifier-failure",
        ]
        if arm.scripted_actions:
            command.extend(["--scripted-actions", arm.scripted_actions])
        if arm.model_command:
            command.extend(["--model-command", arm.model_command])
        return command
    return [
        *common,
        "--agent-command",
        arm.agent_command,
        "--verifier-command",
        "bash /task/run-tests.sh",
        "--expect-verifier-failure",
    ]


def _parse_arm_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for chunk in text.split(","):
        key, sep, value = chunk.partition("=")
        if not sep or not key.strip():
            raise ValueError("typed --arm fields must have the form key=value")
        fields[key.strip()] = value.strip()
    return fields


def _eligible_for_l_gate(arm: PilotArm) -> bool:
    return arm.backend == "model_tool_loop" and arm.client == "provider"


def execute_plan(plan: dict[str, Any]) -> int:
    if not plan.get("passed"):
        raise ValueError("cannot execute blocked pilot plan")
    for run in plan["planned_runs"]:
        proc = subprocess.run(run["command"], text=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0


def write_plan(path: Path, plan: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _selected_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if _boolish(row.get("selected_for_pilot"))]


def _find_task_dir(task_id: str, roots: list[Path]) -> Path | None:
    name = task_id.rsplit("/", 1)[-1]
    for root in roots:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_id(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
