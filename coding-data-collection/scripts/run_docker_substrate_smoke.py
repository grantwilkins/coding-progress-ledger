from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from coding_data_collection.artifacts import read_json, write_json, write_protocol_manifest, write_run_manifest
from coding_data_collection.docker_substrate import (
    DockerResourceLimits,
    agent_phase_command,
    build_docker_image,
    environment_manifest_payload,
    hydrate_workspace_from_image_app,
    oracle_phase_command,
    prepare_verifier_workspace,
    verifier_phase_command,
)
from coding_data_collection.ledger import replay_sidecar, transcript_to_wire_events, write_wire_events
from coding_data_collection.observation import build_observation_events
from coding_data_collection.observation import write_jsonl
from coding_data_collection.protocol import RunStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Docker substrate no-op smoke.")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--collection-root", default=".")
    parser.add_argument("--ledger-root", default="../coding-progress-ledger")
    parser.add_argument("--agent-command", default="true")
    parser.add_argument("--oracle-smoke", action="store_true")
    parser.add_argument("--oracle-command", default="bash /task/solution.sh")
    parser.add_argument("--verifier-command", default="bash /task/run-tests.sh")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--storage-mb", type=int, default=10240)
    parser.add_argument("--wall-clock-limit-sec", type=int, default=600)
    parser.add_argument("--allow-verifier-network", action="store_true")
    parser.add_argument("--network-exception-reason")
    parser.add_argument("--expect-verifier-failure", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "task.md").exists():
        (run_dir / "task.md").write_text(f"# Task\n\nDocker substrate smoke for {task_dir.name}.\n", encoding="utf-8")

    prepare = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_run.py")),
            "--task-dir",
            str(task_dir),
            "--run-dir",
            str(run_dir),
            "--collection-root",
            args.collection_root,
            "--ledger-root",
            args.ledger_root,
        ],
        text=True,
        capture_output=True,
    )
    if prepare.returncode != 0:
        sys.stderr.write(prepare.stderr)
        return prepare.returncode
    if args.allow_verifier_network and not args.network_exception_reason:
        sys.stderr.write("--network-exception-reason is required with --allow-verifier-network\n")
        return 2
    if args.allow_verifier_network:
        metadata = read_json(run_dir / "task_metadata.json")
        metadata.setdefault("network_exceptions", {})["verifier"] = {
            "allowed": True,
            "reason": args.network_exception_reason,
        }
        write_json(run_dir / "task_metadata.json", metadata)

    image = build_docker_image(task_dir, args.image_tag) if not args.skip_build else _inspect_image(args.image_tag)
    hydration = hydrate_workspace_from_image_app(image_tag=args.image_tag, workspace=run_dir / "agent_workspace")
    metadata = read_json(run_dir / "task_metadata.json")
    metadata["image_app_hydration"] = hydration
    write_json(run_dir / "task_metadata.json", metadata)
    shutil.copy2(run_dir / "task.md", run_dir / "agent_workspace" / "task.md")
    agent_limits = DockerResourceLimits(
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        wall_clock_limit_sec=args.wall_clock_limit_sec,
        network_enabled=False,
    )
    verifier_limits = DockerResourceLimits(
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        wall_clock_limit_sec=args.wall_clock_limit_sec,
        network_enabled=args.allow_verifier_network,
    )
    manifest = environment_manifest_payload(
        run_dir=run_dir,
        task_dir=task_dir,
        image=image,
        limits=agent_limits,
        network_exception_reason=None,
    )
    manifest.update(
        {
            "agent_network_policy": agent_limits.network_policy,
            "verifier_network_policy": verifier_limits.network_policy,
            "verifier_network_exception_reason": args.network_exception_reason,
            "oracle_smoke": args.oracle_smoke,
        }
    )
    write_json(run_dir / "environment_manifest.json", manifest)
    write_protocol_manifest(
        run_dir,
        collection_root=Path(args.collection_root),
        ledger_root=Path(args.ledger_root),
    )

    transcript: list[dict] = []
    agent_cmd = agent_phase_command(
        image_tag=args.image_tag,
        run_dir=run_dir,
        command=args.agent_command,
        limits=agent_limits,
    )
    agent_proc = _run(
        agent_cmd,
        timeout=args.wall_clock_limit_sec,
        container_name=_container_name(run_dir.name, "agent"),
    )
    transcript.append(_row(1, "agent_phase", agent_cmd, agent_proc))
    if agent_proc.returncode != 0:
        transcript.append({"step": 2, "kind": "done", "summary": "Docker substrate smoke stopped after agent failure."})
        write_jsonl(run_dir / "transcript.jsonl", transcript)
        (run_dir / "verifier_output.txt").write_text("", encoding="utf-8")
        status = RunStatus.AGENT_TIMEOUT if agent_proc.returncode == 124 else RunStatus.AGENT_CRASH
        (run_dir / "run_notes.md").write_text(
            f"# Run Notes\n\nAgent phase ended with `{status.value}` before verifier execution.\n",
            encoding="utf-8",
        )
        return _finalize_status(
            run_dir,
            args.ledger_root,
            run_status=status,
            final_success=None,
            termination_reason=status.value,
        ) or agent_proc.returncode

    verifier_workspace = run_dir / "verifier_workspace"
    try:
        prepare_verifier_workspace(run_dir / "agent_workspace", verifier_workspace)
    except ValueError as exc:
        transcript.append(
            {
                "step": 2,
                "kind": "done",
                "summary": "Agent phase complete; verifier workspace preparation follows.",
            }
        )
        transcript.append(
            {
                "step": 3,
                "kind": "shell",
                "summary": "verifier_workspace_prepare_failed",
                "exit_code": 1,
                "stderr_snippet": str(exc),
                "visible_to_agent": False,
            }
        )
        write_jsonl(run_dir / "transcript.jsonl", transcript)
        (run_dir / "verifier_output.txt").write_text(str(exc) + "\n", encoding="utf-8")
        (run_dir / "run_notes.md").write_text("# Run Notes\n\nVerifier workspace preparation failed.\n", encoding="utf-8")
        return _finalize_status(
            run_dir,
            args.ledger_root,
            run_status=RunStatus.INFRASTRUCTURE_FAILURE,
            final_success=None,
            termination_reason="verifier_workspace_prepare_failed",
        )
    transcript.append({"step": 2, "kind": "done", "summary": "Agent phase complete; hidden oracle/verifier phases follow."})
    step = 3
    if args.oracle_smoke:
        oracle_cmd = oracle_phase_command(
            image_tag=args.image_tag,
            task_dir=task_dir,
            verifier_workspace=verifier_workspace,
            command=args.oracle_command,
            limits=agent_limits,
        )
        oracle_proc = _run(
            oracle_cmd,
            timeout=args.wall_clock_limit_sec,
            container_name=_container_name(run_dir.name, "oracle"),
        )
        transcript.append(_row(step, "oracle_phase", oracle_cmd, oracle_proc))
        step += 1
        if oracle_proc.returncode != 0:
            if verifier_workspace.exists():
                shutil.rmtree(verifier_workspace)
            write_jsonl(run_dir / "transcript.jsonl", transcript)
            (run_dir / "verifier_output.txt").write_text(
                oracle_proc.stdout + oracle_proc.stderr,
                encoding="utf-8",
            )
            (run_dir / "run_notes.md").write_text(
                "# Run Notes\n\nPrivileged oracle phase failed before verifier execution.\n",
                encoding="utf-8",
            )
            finalize_code = _finalize_completed(run_dir, args.ledger_root, 1)
            return finalize_code or oracle_proc.returncode
    verifier_cmd = verifier_phase_command(
        image_tag=args.image_tag,
        task_dir=task_dir,
        verifier_workspace=verifier_workspace,
        command=args.verifier_command,
        limits=verifier_limits,
    )
    verifier_proc = _run(
        verifier_cmd,
        timeout=args.wall_clock_limit_sec,
        container_name=_container_name(run_dir.name, "verifier"),
    )
    if args.oracle_smoke and verifier_workspace.exists():
        _preserve_oracle_outputs(verifier_workspace, run_dir / "oracle_workspace_snapshot")
    if verifier_workspace.exists():
        shutil.rmtree(verifier_workspace)
    transcript.append(_row(step, "verifier_phase", verifier_cmd, verifier_proc))

    write_jsonl(run_dir / "transcript.jsonl", transcript)
    (run_dir / "verifier_output.txt").write_text(
        verifier_proc.stdout + verifier_proc.stderr,
        encoding="utf-8",
    )
    (run_dir / "run_notes.md").write_text(
        "# Run Notes\n\nDocker substrate smoke generated by scripts/run_docker_substrate_smoke.py.\n",
        encoding="utf-8",
    )

    finalize_code = _finalize_completed(run_dir, args.ledger_root, verifier_proc.returncode)
    if finalize_code != 0:
        return finalize_code
    if verifier_proc.returncode != 0 and not args.expect_verifier_failure:
        return verifier_proc.returncode
    return 0 if agent_proc.returncode == 0 else agent_proc.returncode


def _finalize_completed(run_dir: Path, ledger_root: str, verifier_exit_code: int) -> int:
    status = RunStatus.COMPLETED_SUCCESS if verifier_exit_code == 0 else RunStatus.COMPLETED_FAILURE
    return _finalize_status(
        run_dir,
        ledger_root,
        run_status=status,
        final_success=verifier_exit_code == 0,
        termination_reason="verifier_pass" if verifier_exit_code == 0 else "verifier_fail",
        verifier_exit_code=verifier_exit_code,
    )


def _finalize_status(
    run_dir: Path,
    ledger_root: str,
    *,
    run_status: RunStatus,
    final_success: bool | None,
    termination_reason: str,
    verifier_exit_code: int | None = None,
) -> int:
    transcript = [
        json.loads(line)
        for line in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    write_wire_events(
        run_dir / "events.jsonl",
        transcript_to_wire_events(
            transcript,
            run_id=run_dir.name,
            verifier_exit_code=verifier_exit_code,
        ),
    )
    write_jsonl(
        run_dir / "observation_events.jsonl",
        build_observation_events(
            transcript,
            run_id=run_dir.name,
            verifier_exit_code=verifier_exit_code,
            expected_paths=set(read_json(run_dir / "task_metadata.json").get("expected_paths", [])),
        ),
    )
    sidecar = replay_sidecar(run_dir=run_dir, ledger_root=Path(ledger_root))
    if sidecar.returncode != 0:
        sys.stderr.write(sidecar.stderr)
        write_run_manifest(
            run_dir,
            run_id=run_dir.name,
            run_status=RunStatus.INFRASTRUCTURE_FAILURE,
            final_success=None,
            termination_reason="sidecar_replay_failed",
        )
        return sidecar.returncode
    write_run_manifest(
        run_dir,
        run_id=run_dir.name,
        run_status=run_status,
        final_success=final_success,
        termination_reason=termination_reason,
        metrics={
            "agent_backend": "shell_command",
            "collection_kind": "substrate_smoke",
        },
    )
    return 0


def _run(command: list[str], *, timeout: int, container_name: str) -> subprocess.CompletedProcess[str]:
    command = _with_container_name(command, container_name)
    try:
        proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args=command, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)
        proc.kill()
        stdout, stderr = proc.communicate()
        stdout = stdout or ""
        stderr = stderr or ""
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr + f"\ncommand timed out after {timeout} seconds\n",
        )


def _row(step: int, kind: str, command: list[str], proc: subprocess.CompletedProcess[str]) -> dict:
    return {
        "step": step,
        "kind": "shell",
        "summary": kind,
        "command": " ".join(command),
        "command_argv": command,
        "exit_code": proc.returncode,
        "stdout_snippet": proc.stdout[-2000:],
        "stderr_snippet": proc.stderr[-2000:],
    }


def _with_container_name(command: list[str], container_name: str) -> list[str]:
    if command[:3] != ["docker", "run", "--rm"]:
        raise ValueError("expected docker run --rm command")
    return [*command[:3], "--name", container_name, *command[3:]]


def _container_name(run_id: str, phase: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in run_id.lower()).strip("-")
    return f"cdc-{safe}-{phase}"[:63]


def _preserve_oracle_outputs(verifier_workspace: Path, snapshot_dir: Path) -> None:
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    ignore = shutil.ignore_patterns(".venv", ".pytest_cache", "__pycache__", "*.pyc", "uv.lock")
    shutil.copytree(verifier_workspace, snapshot_dir, ignore=ignore)


def _inspect_image(image_tag: str):
    from coding_data_collection.docker_substrate import DockerImageInfo

    image_id = subprocess.check_output(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
        text=True,
    ).strip()
    return DockerImageInfo(image_tag=image_tag, image_id=image_id, image_digest=image_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
