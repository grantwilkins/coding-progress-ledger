from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from coding_data_collection.agents.base import AgentBudget
from coding_data_collection.agents.model_client import (
    ModelClientConfig,
    ProviderModelClient,
    ScriptedModelClient,
    model_client_metrics,
)
from coding_data_collection.agents.model_tool_loop import ModelToolLoopBackend
from coding_data_collection.agents.prompt_builder import SYSTEM_PROMPT, TOOL_SPECS
from coding_data_collection.agent_preflight import run_agent_readiness_preflight, write_agent_readiness_report
from coding_data_collection.artifacts import read_json, write_json, write_protocol_manifest, write_run_manifest
from coding_data_collection.docker_substrate import (
    DockerResourceLimits,
    build_docker_image,
    environment_manifest_payload,
    hydrate_workspace_from_image_app,
    prepare_verifier_workspace,
    verifier_phase_command,
)
from coding_data_collection.ledger import replay_sidecar
from coding_data_collection.protocol import RunStatus
from coding_data_collection.recording import RunRecorder
from coding_data_collection.sandbox import DockerSandboxExecutor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a host-side model-tool-loop agent against a Docker sandbox.")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--client", choices=["scripted", "provider"], default="scripted")
    parser.add_argument("--scripted-actions", type=Path)
    parser.add_argument("--model-command", help="Provider adapter command; defaults to CDC_MODEL_CLIENT_COMMAND.")
    parser.add_argument("--model-provider", default="command")
    parser.add_argument("--model-name", default="scripted")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens-out", type=int, default=2048)
    parser.add_argument("--collection-root", default=".")
    parser.add_argument("--ledger-root", default="../coding-progress-ledger")
    parser.add_argument("--estimator-root", default="../coding-estimator")
    parser.add_argument("--verifier-command", default="bash /task/run-tests.sh")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--storage-mb", type=int, default=10240)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-wall-time-s", type=int, default=1800)
    parser.add_argument("--max-tool-time-s", type=int, default=120)
    parser.add_argument(
        "--min-steps-before-done",
        type=int,
        default=15,
        help="Reject done actions before this transcript step unless the run is genuinely blocked.",
    )
    parser.add_argument(
        "--require-validation-before-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a validation-like shell attempt before accepting done unless blocked.",
    )
    parser.add_argument(
        "--allow-blocked-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow early done when transcript/summary shows a genuine blocked state.",
    )
    parser.add_argument("--allow-verifier-network", action="store_true")
    parser.add_argument("--network-exception-reason")
    parser.add_argument("--expect-verifier-failure", action="store_true")
    parser.add_argument(
        "--agent-readiness-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail before model calls if the no-network agent image lacks task-required runtimes/dependencies.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Prepare workspace/build image/run readiness checks, then stop before any model calls.",
    )
    parser.add_argument(
        "--provider-route-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed before the agent loop if a provider route cannot return valid action JSON with metadata.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-sidecar", action="store_true")
    parser.add_argument(
        "--eligible-for-L-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override L-gate eligibility for provider smoke/debug runs.",
    )
    parser.add_argument(
        "--pilot-type",
        help="Override pilot_type recorded in run manifests, e.g. openrouter_free_smoke.",
    )
    args = parser.parse_args(argv)

    task_dir = Path(args.task_dir)
    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_task_md(task_dir, run_dir)

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
            "--estimator-root",
            args.estimator_root,
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

    workspace = run_dir / "agent_workspace"
    image = build_docker_image(task_dir, args.image_tag) if not args.skip_build else _inspect_image(args.image_tag)
    hydration = hydrate_workspace_from_image_app(image_tag=args.image_tag, workspace=workspace)
    metadata = read_json(run_dir / "task_metadata.json")
    metadata["image_app_hydration"] = hydration
    write_json(run_dir / "task_metadata.json", metadata)
    shutil.copy2(run_dir / "task.md", workspace / "task.md")
    agent_limits = DockerResourceLimits(
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        wall_clock_limit_sec=args.max_wall_time_s,
        network_enabled=False,
    )
    verifier_limits = DockerResourceLimits(
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        wall_clock_limit_sec=args.max_wall_time_s,
        network_enabled=args.allow_verifier_network,
    )
    manifest = environment_manifest_payload(
        run_dir=run_dir,
        task_dir=task_dir,
        image=image,
        limits=agent_limits,
    )
    manifest.update(
        {
            "agent_backend": "model_tool_loop",
            "agent_model": args.model_name,
            "agent_controller_network_policy": "host",
            "agent_sandbox_network_policy": agent_limits.network_policy,
            "verifier_network_policy": verifier_limits.network_policy,
            "verifier_network_exception_reason": args.network_exception_reason,
        }
    )
    write_json(run_dir / "environment_manifest.json", manifest)
    write_protocol_manifest(
        run_dir,
        collection_root=Path(args.collection_root),
        ledger_root=Path(args.ledger_root),
        estimator_root=Path(args.estimator_root),
    )

    container_name = _container_name(run_id, "agent")
    start = _start_agent_container(
        image_tag=args.image_tag,
        run_dir=run_dir,
        workspace=workspace,
        limits=agent_limits,
        container_name=container_name,
    )
    if start.returncode != 0:
        sys.stderr.write(start.stderr)
        return start.returncode

    if args.agent_readiness_preflight:
        preflight = run_agent_readiness_preflight(
            image_tag=args.image_tag,
            workspace_dir=workspace,
            task_md=(run_dir / "task.md").read_text(encoding="utf-8"),
            timeout_s=min(args.max_tool_time_s, 60),
        )
        write_agent_readiness_report(preflight, run_dir / "agent_readiness_preflight.json")
        if not preflight["passed"]:
            subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)
            write_run_manifest(
                run_dir,
                run_id=run_id,
                run_status=RunStatus.ENVIRONMENT_SETUP_FAILURE,
                final_success=None,
                termination_reason="agent_readiness_preflight_failed",
                metrics={
                    "agent_backend": "model_tool_loop",
                    "pilot_type": "real_agent_pilot" if args.client == "provider" else "scripted_model_smoke",
                    "eligible_for_L_gate": False,
                    "model_provider": args.model_provider if args.client == "provider" else "scripted",
                    "model_name": args.model_name,
                    "agent_readiness_preflight_passed": False,
                    "agent_readiness_failed_checks": preflight["failed_checks"],
                    "agent_readiness_manual_review_checks": preflight["manual_review_checks"],
                },
            )
            (run_dir / "run_notes.md").write_text(
                "# Run Notes\n\n"
                "Run stopped before model calls because the no-network agent image failed readiness preflight. "
                "Bake required runtimes/dependencies into the task image or exclude/tag this task before provider collection.\n",
                encoding="utf-8",
            )
            return 2
    elif args.preflight_only:
        preflight = {
            "schema_version": "0.1.0",
            "image_tag": args.image_tag,
            "network_policy": "disabled",
            "check_count": 0,
            "passed": True,
            "failed_checks": [],
            "manual_review_checks": [],
            "results": [],
        }
        write_agent_readiness_report(preflight, run_dir / "agent_readiness_preflight.json")

    if args.preflight_only:
        subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)
        write_run_manifest(
            run_dir,
            run_id=run_id,
            run_status=RunStatus.ARTIFACT_INCOMPLETE,
            final_success=None,
            termination_reason="agent_readiness_preflight_passed",
            metrics={
                "agent_backend": "model_tool_loop",
                "pilot_type": "preflight_only",
                "eligible_for_L_gate": False,
                "model_provider": args.model_provider if args.client == "provider" else "scripted",
                "model_name": args.model_name,
                "agent_readiness_preflight_passed": True,
                "agent_readiness_failed_checks": [],
                "agent_readiness_manual_review_checks": [],
            },
        )
        (run_dir / "run_notes.md").write_text(
            "# Run Notes\n\n"
            "Run stopped before model calls after workspace preparation, image build, "
            "and agent readiness preflight passed.\n",
            encoding="utf-8",
        )
        return 0

    try:
        model_client = _build_model_client(args)
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    if args.client == "provider" and args.provider_route_preflight:
        preflight_client = _build_model_client(args)
        preflight = _run_provider_route_preflight(
            preflight_client,
            task_md=(run_dir / "task.md").read_text(encoding="utf-8"),
        )
        write_json(run_dir / "provider_route_preflight.json", preflight)
        if not preflight["passed"]:
            subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)
            write_run_manifest(
                run_dir,
                run_id=run_id,
                run_status=RunStatus.ENVIRONMENT_SETUP_FAILURE,
                final_success=None,
                termination_reason="provider_route_preflight_failed",
                metrics={
                    "agent_backend": "model_tool_loop",
                    "pilot_type": "real_agent_pilot",
                    "eligible_for_L_gate": False,
                    "model_provider": args.model_provider,
                    "model_name": args.model_name,
                    "provider_route_preflight_passed": False,
                    "provider_route_preflight_issues": preflight["issues"],
                    **_provider_metrics_for_manifest(model_client_metrics(preflight_client)),
                },
            )
            (run_dir / "run_notes.md").write_text(
                "# Run Notes\n\n"
                "Run stopped before agent-loop collection because the provider route failed preflight. "
                "Use a structured-output-capable fixed route or mark this arm L-ineligible.\n",
                encoding="utf-8",
            )
            return 2

    recorder = RunRecorder(run_dir=run_dir, run_id=run_id)
    backend = ModelToolLoopBackend(
        model_client,
        model_name=args.model_name,
        eligible_for_l_gate=args.eligible_for_L_gate,
        pilot_type=args.pilot_type,
    )
    result = None
    verifier_exit_code: int | None = None
    verifier_image_tag = args.image_tag
    commit_stderr = ""
    try:
        sandbox = DockerSandboxExecutor(container_name=container_name, workspace_dir=workspace)
        result = backend.run(
            run_id=run_id,
            task_md=(run_dir / "task.md").read_text(encoding="utf-8"),
            workspace_dir=workspace,
            sandbox=sandbox,
            budget=AgentBudget(
                max_steps=args.max_steps,
                max_wall_time_s=args.max_wall_time_s,
                max_tool_time_s=args.max_tool_time_s,
                min_steps_before_done=args.min_steps_before_done,
                require_validation_before_done=args.require_validation_before_done,
                allow_blocked_done=args.allow_blocked_done,
            ),
            recorder=recorder,
        )
    finally:
        if result is not None:
            verifier_image_tag = f"{args.image_tag}-agent-final"[:120]
            commit = _commit_agent_container(container_name=container_name, image_tag=verifier_image_tag)
            if commit.returncode != 0:
                commit_stderr = commit.stderr
                verifier_image_tag = ""
        subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)

    if result is None:
        return 1
    if not verifier_image_tag:
        (run_dir / "run_notes.md").write_text(
            "# Run Notes\n\nAgent container image commit failed before verifier execution.\n",
            encoding="utf-8",
        )
        (run_dir / "verifier_output.txt").write_text(commit_stderr, encoding="utf-8")
        write_run_manifest(
            run_dir,
            run_id=run_id,
            run_status=RunStatus.INFRASTRUCTURE_FAILURE,
            final_success=None,
            termination_reason="agent_image_commit_failed",
            metrics=_manifest_metrics(args, backend, result),
        )
        return 1
    committed_manifest = read_json(run_dir / "environment_manifest.json")
    committed_manifest["agent_committed_image_tag"] = verifier_image_tag
    committed_manifest["verifier_image_tag"] = verifier_image_tag
    committed_manifest["verifier_uses_committed_agent_image"] = True
    committed_manifest["agent_committed_image_id"] = _image_id(verifier_image_tag)
    write_json(run_dir / "environment_manifest.json", committed_manifest)
    verifier_exit_code = _run_verifier(
        args=args,
        task_dir=task_dir,
        run_dir=run_dir,
        verifier_limits=verifier_limits,
        image_tag=verifier_image_tag,
    )
    recorder.write_derived_artifacts(
        verifier_exit_code=verifier_exit_code,
        expected_paths=set(read_json(run_dir / "task_metadata.json").get("expected_paths", [])),
    )
    if not args.skip_sidecar:
        sidecar = replay_sidecar(run_dir=run_dir, ledger_root=Path(args.ledger_root))
        if sidecar.returncode != 0:
            sys.stderr.write(sidecar.stderr)
            write_run_manifest(
                run_dir,
                run_id=run_id,
                run_status=RunStatus.INFRASTRUCTURE_FAILURE,
                final_success=None,
                termination_reason="sidecar_replay_failed",
                metrics=_manifest_metrics(args, backend, result),
            )
            return sidecar.returncode

    final_success = verifier_exit_code == 0
    write_run_manifest(
        run_dir,
        run_id=run_id,
        run_status=RunStatus.COMPLETED_SUCCESS if final_success else RunStatus.COMPLETED_FAILURE,
        final_success=final_success,
        termination_reason="verifier_pass" if final_success else "verifier_fail",
        metrics=_manifest_metrics(args, backend, result),
    )
    (run_dir / "run_notes.md").write_text(
        "# Run Notes\n\nHost-side model_tool_loop backend with Docker sandbox executor.\n",
        encoding="utf-8",
    )
    if verifier_exit_code != 0 and not args.expect_verifier_failure:
        return verifier_exit_code
    return 0


def _write_task_md(task_dir: Path, run_dir: Path) -> None:
    source = task_dir / "task.md"
    target = run_dir / "task.md"
    if source.is_file():
        shutil.copy2(source, target)
    elif (task_dir / "task.yaml").is_file():
        target.write_text(_task_md_from_yaml(task_dir / "task.yaml", task_dir_name=task_dir.name), encoding="utf-8")
    elif not target.exists():
        target.write_text(f"# Task\n\nTerminal-Bench task {task_dir.name}.\n", encoding="utf-8")


def _task_md_from_yaml(path: Path, *, task_dir_name: str) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    instruction: list[str] = []
    in_instruction = False
    indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "terminal-bench-canary" in stripped.lower():
            continue
        if not in_instruction:
            if stripped.startswith("instruction: |"):
                in_instruction = True
            elif stripped.startswith("instruction: "):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                if value:
                    instruction.append(value)
                break
            continue
        if not stripped:
            instruction.append("")
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if indent is None:
            indent = current_indent
        if current_indent < indent:
            break
        instruction.append(line[indent:])

    body = "\n".join(instruction).strip()
    if not body:
        body = f"Terminal-Bench task {task_dir_name}."
    return f"# Task\n\n{body}\n"


def _start_agent_container(
    *,
    image_tag: str,
    run_dir: Path,
    workspace: Path,
    limits: DockerResourceLimits,
    container_name: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--cpus",
        str(limits.cpus),
        "--memory",
        f"{limits.memory_mb}m",
        "--network",
        "none",
        "-v",
        f"{workspace.resolve()}:/app:rw",
        "-v",
        f"{(run_dir / 'task.md').resolve()}:/task/task.md:ro",
        "-w",
        "/app",
        "--entrypoint",
        "sleep",
        image_tag,
        str(limits.wall_clock_limit_sec),
    ]
    return subprocess.run(command, text=True, capture_output=True)


def _run_verifier(
    *,
    args: argparse.Namespace,
    task_dir: Path,
    run_dir: Path,
    verifier_limits: DockerResourceLimits,
    image_tag: str,
) -> int:
    verifier_workspace = run_dir / "verifier_workspace"
    try:
        prepare_verifier_workspace(run_dir / "agent_workspace", verifier_workspace)
    except ValueError as exc:
        (run_dir / "verifier_output.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return 1
    command = verifier_phase_command(
        image_tag=image_tag,
        task_dir=task_dir,
        verifier_workspace=verifier_workspace,
        command=args.verifier_command,
        limits=verifier_limits,
    )
    proc = subprocess.run(command, text=True, capture_output=True, timeout=args.max_wall_time_s)
    if verifier_workspace.exists():
        shutil.rmtree(verifier_workspace)
    (run_dir / "verifier_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def _manifest_metrics(args: argparse.Namespace, backend: ModelToolLoopBackend, result) -> dict:
    client_metrics = model_client_metrics(backend.model_client)
    provider_metrics = _provider_metrics_for_manifest(client_metrics)
    eligible_for_l_gate = backend.eligible_for_L_gate
    if args.client == "provider":
        eligible_for_l_gate = eligible_for_l_gate and _provider_route_l_eligible(client_metrics)
    return {
        "agent_backend": backend.name,
        "pilot_type": backend.pilot_type,
        "eligible_for_L_gate": eligible_for_l_gate,
        "model_provider": args.model_provider if args.client == "provider" else "scripted",
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_model_calls": args.max_steps,
        "max_tool_calls": args.max_steps,
        "max_steps": args.max_steps,
        "min_steps_before_done": args.min_steps_before_done,
        "require_validation_before_done": args.require_validation_before_done,
        "allow_blocked_done": args.allow_blocked_done,
        "max_wall_time_s": args.max_wall_time_s,
        "max_tool_time_s": args.max_tool_time_s,
        "max_tokens_in": None,
        "max_tokens_out": args.max_tokens_out,
        "total_model_calls": client_metrics.get("total_model_calls"),
        "total_tokens_in": client_metrics.get("total_tokens_in"),
        "total_tokens_out": client_metrics.get("total_tokens_out"),
        "agent_termination_reason": result.termination_reason,
        "agent_steps_used": result.steps_used,
        "provider_route_preflight_passed": bool(args.client == "provider" and args.provider_route_preflight),
        "tokens_in": client_metrics.get("total_tokens_in", result.tokens_in),
        "tokens_out": client_metrics.get("total_tokens_out", result.tokens_out),
        "estimated_cost_usd": client_metrics.get("estimated_cost_usd", result.estimated_cost_usd),
        "total_cost_credits": client_metrics.get("total_cost_credits"),
        **provider_metrics,
    }


def _build_model_client(args: argparse.Namespace):
    if args.client == "scripted":
        if args.scripted_actions is None:
            raise ValueError("--scripted-actions is required with --client scripted")
        return ScriptedModelClient.from_jsonl(args.scripted_actions, model_name=args.model_name)
    return ProviderModelClient(
        command=args.model_command,
        config=ModelClientConfig(
            provider=args.model_provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens_out=args.max_tokens_out,
            timeout_s=args.max_tool_time_s,
        ),
    )


def _run_provider_route_preflight(model_client, *, task_md: str) -> dict:
    raw = model_client.next_action(
        system_prompt=SYSTEM_PROMPT,
        task_prompt=task_md,
        transcript_prefix=[],
        tool_specs=TOOL_SPECS,
        budget_state={"provider_route_preflight": True, "retry_attempt": 0},
    )
    metrics = model_client_metrics(model_client)
    issues: list[str] = []
    if isinstance(raw, dict) and isinstance(raw.get("provider_adapter_error"), dict):
        error = raw["provider_adapter_error"]
        issues.append(str(error.get("message") or "provider adapter error"))
        if error.get("stderr_snippet"):
            issues.append(str(error["stderr_snippet"])[-500:])
    else:
        from coding_data_collection.agents.action_schema import parse_model_action

        try:
            parse_model_action(json.loads(raw) if isinstance(raw, str) else raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"provider route did not return valid action JSON: {exc}")
    if not _provider_route_l_eligible(metrics):
        issues.append("provider route did not record L-eligible metadata with nonzero usage and no fallback")
    return {
        "passed": not issues,
        "issues": issues,
        "metrics": _provider_metrics_for_manifest(metrics),
    }


def _provider_route_l_eligible(metrics: dict) -> bool:
    return (
        bool(metrics.get("provider_calls"))
        and bool(metrics.get("resolved_models"))
        and int(metrics.get("total_tokens_in") or 0) > 0
        and int(metrics.get("total_tokens_out") or 0) > 0
        and int(metrics.get("fallback_call_count") or 0) == 0
    )


def _provider_metrics_for_manifest(metrics: dict) -> dict:
    return {
        "provider_calls": metrics.get("provider_calls"),
        "last_provider_call": metrics.get("last_provider_call"),
        "fallback_call_count": metrics.get("fallback_call_count"),
        "resolved_models": metrics.get("resolved_models"),
    }


def _container_name(run_id: str, phase: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in run_id.lower()).strip("-")
    return f"cdc-{safe}-{phase}"[:63]


def _commit_agent_container(*, container_name: str, image_tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "commit", container_name, image_tag],
        text=True,
        capture_output=True,
    )


def _image_id(image_tag: str) -> str:
    return subprocess.check_output(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
        text=True,
    ).strip()


def _inspect_image(image_tag: str):
    from coding_data_collection.docker_substrate import DockerImageInfo

    image_id = _image_id(image_tag)
    return DockerImageInfo(image_tag=image_tag, image_id=image_id, image_digest=image_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
