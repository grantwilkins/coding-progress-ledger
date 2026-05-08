from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from coding_data_collection.artifacts import read_json, write_json
from coding_data_collection.docker_substrate import DockerResourceLimits, prepare_verifier_workspace, verifier_phase_command
from coding_data_collection.verifier_determinism import determinism_report, verifier_signature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rerun a Terminal-Bench verifier and compare semantic outcomes.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--source-workspace")
    parser.add_argument("--rerun-root")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--verifier-command", default="bash /task/run-tests.sh")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--storage-mb", type=int, default=10240)
    parser.add_argument("--wall-clock-limit-sec", type=int, default=600)
    parser.add_argument("--allow-verifier-network", action="store_true")
    args = parser.parse_args(argv)

    if args.trials < 1:
        sys.stderr.write("--trials must be at least 1\n")
        return 2

    run_dir = Path(args.run_dir)
    task_dir = Path(args.task_dir)
    source_workspace = Path(args.source_workspace) if args.source_workspace else _default_source_workspace(run_dir)
    rerun_root = Path(args.rerun_root) if args.rerun_root else run_dir / "verifier_reruns"
    rerun_root.mkdir(parents=True, exist_ok=True)

    expected_exit_code = _recorded_verifier_exit_code(run_dir)
    expected_output = (run_dir / "verifier_output.txt").read_text(encoding="utf-8")
    expected = verifier_signature(stdout=expected_output, stderr="", exit_code=expected_exit_code)

    limits = DockerResourceLimits(
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        wall_clock_limit_sec=args.wall_clock_limit_sec,
        network_enabled=args.allow_verifier_network,
    )

    observed = []
    trial_records = []
    for trial in range(1, args.trials + 1):
        workspace = rerun_root / f"trial_{trial}" / "workspace"
        prepare_verifier_workspace(source_workspace, workspace)
        command = verifier_phase_command(
            image_tag=args.image_tag,
            task_dir=task_dir,
            verifier_workspace=workspace,
            command=args.verifier_command,
            limits=limits,
        )
        proc = _run(command, timeout=args.wall_clock_limit_sec, container_name=_container_name(run_dir.name, trial))
        signature = verifier_signature(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)
        observed.append(signature)
        trial_records.append(
            {
                "trial": trial,
                "command_argv": proc.args,
                "exit_code": proc.returncode,
                "signature": signature.to_dict(),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )

    report = determinism_report(expected=expected, observed=observed)
    report.update(
        {
            "run_id": read_json(run_dir / "run_manifest.json").get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "task_dir": str(task_dir),
            "source_workspace": str(source_workspace),
            "rerun_root": str(rerun_root),
            "trial_records": trial_records,
        }
    )
    write_json(run_dir / "verifier_determinism_report.json", report)
    print(json.dumps({key: report[key] for key in ["run_id", "trials", "deterministic"]}, sort_keys=True))
    return 0 if report["deterministic"] else 1


def _default_source_workspace(run_dir: Path) -> Path:
    oracle_snapshot = run_dir / "oracle_workspace_snapshot"
    if oracle_snapshot.is_dir():
        return oracle_snapshot
    return run_dir / "agent_workspace"


def _recorded_verifier_exit_code(run_dir: Path) -> int:
    transcript_path = run_dir / "transcript.jsonl"
    verifier_rows = []
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("summary") == "verifier_phase" and "exit_code" in row:
            verifier_rows.append(row)
    if not verifier_rows:
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("final_success") is True:
            return 0
        if manifest.get("final_success") is False:
            return 1
        raise ValueError(f"{transcript_path} has no verifier_phase row and run_manifest.json has no terminal result")
    return int(verifier_rows[-1]["exit_code"])


def _run(command: list[str], *, timeout: int, container_name: str) -> subprocess.CompletedProcess[str]:
    command = _with_container_name(command, container_name)
    try:
        proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args=command, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)
        proc.kill()
        stdout, stderr = proc.communicate()
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout or "",
            stderr=(stderr or "") + f"\ncommand timed out after {timeout} seconds\n",
        )
    finally:
        if container_name:
            subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True)


def _with_container_name(command: list[str], container_name: str) -> list[str]:
    if command[:3] != ["docker", "run", "--rm"]:
        raise ValueError("expected docker run --rm command")
    return [*command[:3], "--name", container_name, *command[3:]]


def _container_name(run_id: str, trial: int) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in run_id.lower()).strip("-")
    return f"cdc-{safe}-verifier-rerun-{trial}"[:63]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
