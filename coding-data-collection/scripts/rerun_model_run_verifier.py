from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from coding_data_collection.artifacts import read_json, write_json, write_run_manifest
from coding_data_collection.docker_substrate import (
    DockerResourceLimits,
    prepare_verifier_workspace,
    verifier_phase_command,
)
from coding_data_collection.ledger import replay_sidecar
from coding_data_collection.protocol import RunStatus
from coding_data_collection.recording import RunRecorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rerun only the verifier for an existing model-agent run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--verifier-command", default="bash /task/run-tests.sh")
    parser.add_argument("--allow-verifier-network", action="store_true")
    parser.add_argument("--network-exception-reason")
    parser.add_argument("--ledger-root", default="../coding-progress-ledger")
    parser.add_argument("--max-wall-time-s", type=int, default=1800)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    manifest = read_json(run_dir / "run_manifest.json")
    task_metadata = read_json(run_dir / "task_metadata.json")
    environment = read_json(run_dir / "environment_manifest.json")
    if not manifest:
        sys.stderr.write(f"{run_dir}: missing run_manifest.json\n")
        return 2
    if args.allow_verifier_network and not args.network_exception_reason:
        sys.stderr.write("--network-exception-reason is required with --allow-verifier-network\n")
        return 2

    task_dir = Path(task_metadata.get("task_dir", ""))
    if not task_dir.is_dir():
        sys.stderr.write(f"{run_dir}: task_dir not found: {task_dir}\n")
        return 2
    image_tag = environment.get("verifier_image_tag") or environment.get("agent_committed_image_tag") or environment.get("image_tag")
    if not image_tag:
        sys.stderr.write(f"{run_dir}: no verifier/image tag in environment_manifest.json\n")
        return 2

    limits = DockerResourceLimits(
        cpus=int(environment.get("cpu_limit") or 1),
        memory_mb=int(environment.get("memory_limit_mb") or 2048),
        storage_mb=int(environment.get("disk_limit_mb") or 10240),
        wall_clock_limit_sec=args.max_wall_time_s,
        network_enabled=args.allow_verifier_network,
    )
    verifier_workspace = run_dir / "verifier_workspace"
    try:
        prepare_verifier_workspace(run_dir / "agent_workspace", verifier_workspace)
    except ValueError as exc:
        (run_dir / "verifier_output.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return _rewrite_terminal_artifacts(args, run_dir, manifest, verifier_exit_code=1)

    command = verifier_phase_command(
        image_tag=image_tag,
        task_dir=task_dir,
        verifier_workspace=verifier_workspace,
        command=args.verifier_command,
        limits=limits,
    )
    proc = subprocess.run(command, text=True, capture_output=True, timeout=args.max_wall_time_s)
    if verifier_workspace.exists():
        shutil.rmtree(verifier_workspace)
    (run_dir / "verifier_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")

    environment["verifier_network_policy"] = limits.network_policy
    environment["verifier_network_exception_reason"] = args.network_exception_reason
    environment["verifier_rerun_from_existing_agent_artifacts"] = True
    write_json(run_dir / "environment_manifest.json", environment)
    return _rewrite_terminal_artifacts(args, run_dir, manifest, verifier_exit_code=proc.returncode)


def _rewrite_terminal_artifacts(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict,
    *,
    verifier_exit_code: int,
) -> int:
    recorder = RunRecorder(run_dir=run_dir, run_id=run_dir.name)
    expected_paths = set(read_json(run_dir / "task_metadata.json").get("expected_paths", []))
    recorder.write_derived_artifacts(verifier_exit_code=verifier_exit_code, expected_paths=expected_paths)
    sidecar = replay_sidecar(run_dir=run_dir, ledger_root=Path(args.ledger_root))
    if sidecar.returncode != 0:
        sys.stderr.write(sidecar.stderr)
        return sidecar.returncode

    metrics = dict(manifest.get("metrics") or {})
    metrics["verifier_rerun_from_existing_agent_artifacts"] = True
    metrics["verifier_network_policy"] = "enabled" if args.allow_verifier_network else "disabled"
    metrics["verifier_network_exception_reason"] = args.network_exception_reason
    final_success = verifier_exit_code == 0
    write_run_manifest(
        run_dir,
        run_id=run_dir.name,
        run_status=RunStatus.COMPLETED_SUCCESS if final_success else RunStatus.COMPLETED_FAILURE,
        final_success=final_success,
        termination_reason="verifier_pass" if final_success else "verifier_fail",
        metrics=metrics,
    )
    (run_dir / "run_notes.md").write_text(
        "# Run Notes\n\nVerifier rerun from existing provider-backed agent artifacts.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
