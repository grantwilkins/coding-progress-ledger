from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from coding_data_collection.artifacts import utc_now, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an explicit trace-collection plan.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append-arg", action="append", default=[])
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    runs = _plan_runs(plan)
    results: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        command = _command_for_run(run)
        command.extend(str(arg) for arg in args.append_arg)
        print(f"[{index}/{len(runs)}] {run['run_id']}", flush=True)
        proc = subprocess.run(command, text=True)
        results.append(
            {
                "run_id": run["run_id"],
                "task_dir": str(run["task_dir"]),
                "run_dir": str(run["run_dir"]),
                "returncode": proc.returncode,
            }
        )
        print(f"[{index}/{len(runs)}] returncode={proc.returncode}", flush=True)

    payload = {
        "created_at": utc_now(),
        "plan": str(args.plan),
        "run_count": len(results),
        "nonzero_count": sum(1 for row in results if row["returncode"] != 0),
        "results": results,
    }
    write_json(args.out, payload)
    return 0 if payload["nonzero_count"] == 0 else 1


def _plan_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = plan.get("defaults") if isinstance(plan.get("defaults"), dict) else {}
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("plan must contain a non-empty 'runs' list")
    out = []
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("each plan run must be an object")
        merged = {**defaults, **row}
        missing = [key for key in ("run_id", "task_dir", "run_dir", "image_tag") if not merged.get(key)]
        if missing:
            raise ValueError(f"run is missing required fields: {missing}")
        out.append(merged)
    return out


def _command_for_run(run: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_model_agent_trace.py",
        "--task-dir",
        str(run["task_dir"]),
        "--run-dir",
        str(run["run_dir"]),
        "--image-tag",
        str(run["image_tag"]),
        "--client",
        str(run.get("client", "scripted")),
        "--model-provider",
        str(run.get("model_provider", "command")),
        "--model-name",
        str(run.get("model_name", "scripted")),
        "--verifier-command",
        str(run.get("verifier_command", "bash /task/run-tests.sh")),
    ]
    optional_pairs = {
        "scripted_actions": "--scripted-actions",
        "model_command": "--model-command",
        "temperature": "--temperature",
        "max_tokens_out": "--max-tokens-out",
        "collection_root": "--collection-root",
        "ledger_root": "--ledger-root",
        "cpus": "--cpus",
        "memory_mb": "--memory-mb",
        "storage_mb": "--storage-mb",
        "max_steps": "--max-steps",
        "max_wall_time_s": "--max-wall-time-s",
        "max_tool_time_s": "--max-tool-time-s",
        "min_steps_before_done": "--min-steps-before-done",
    }
    for key, flag in optional_pairs.items():
        if key in run and run[key] is not None:
            command.extend([flag, str(run[key])])

    bool_flags = {
        "require_validation_before_done": "--require-validation-before-done",
        "allow_verifier_network": "--allow-verifier-network",
        "expect_verifier_failure": "--expect-verifier-failure",
        "agent_readiness_preflight": "--agent-readiness-preflight",
        "preflight_only": "--preflight-only",
        "provider_route_preflight": "--provider-route-preflight",
        "skip_build": "--skip-build",
        "skip_sidecar": "--skip-sidecar",
    }
    false_flags = {
        "allow_blocked_done": "--no-allow-blocked-done",
        "agent_readiness_preflight": "--no-agent-readiness-preflight",
        "provider_route_preflight": "--no-provider-route-preflight",
        "require_validation_before_done": "--no-require-validation-before-done",
    }
    for key, flag in bool_flags.items():
        if run.get(key) is True:
            command.append(flag)
    for key, flag in false_flags.items():
        if run.get(key) is False:
            command.append(flag)
    if run.get("network_exception_reason"):
        command.extend(["--network-exception-reason", str(run["network_exception_reason"])])
    return command


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
