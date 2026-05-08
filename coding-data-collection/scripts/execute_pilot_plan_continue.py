#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from coding_data_collection.artifacts import utc_now, write_json
from coding_data_collection.agents.model_client import ModelClientConfig, ProviderModelClient, model_client_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_model_agent_pilot import _provider_metrics_for_manifest, _run_provider_route_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute every run in a pilot plan and continue after failed runs.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append-arg", action="append", default=[], help="Append an argument to every planned command.")
    parser.add_argument(
        "--provider-route-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preflight each provider arm once before launching planned runs.",
    )
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("passed"):
        sys.stderr.write("refusing to execute blocked plan\n")
        return 2
    provider_preflight = _preflight_provider_arms(plan) if args.provider_route_preflight else {"passed": True, "arms": []}
    if not provider_preflight["passed"]:
        payload = {
            "created_at": utc_now(),
            "plan": str(args.plan),
            "run_count": 0,
            "nonzero_count": 0,
            "provider_route_preflight": provider_preflight,
            "results": [],
        }
        write_json(args.out, payload)
        for arm in provider_preflight["arms"]:
            if not arm["passed"]:
                sys.stderr.write(f"{arm['name']}: provider route preflight failed: {arm['issues']}\n")
        return 2

    results: list[dict[str, Any]] = []
    for index, run in enumerate(plan.get("planned_runs", []), start=1):
        command = [str(part) for part in run["command"]]
        command.extend(args.append_arg)
        print(f"[{index}/{len(plan['planned_runs'])}] {run['run_id']}", flush=True)
        proc = subprocess.run(command, text=True)
        results.append(
            {
                "run_id": run["run_id"],
                "task_id": run["task_id"],
                "arm": run["arm"],
                "backend": run["backend"],
                "eligible_for_L_gate": run["eligible_for_L_gate"],
                "returncode": proc.returncode,
            }
        )
        print(f"[{index}/{len(plan['planned_runs'])}] returncode={proc.returncode}", flush=True)

    payload = {
        "created_at": utc_now(),
        "plan": str(args.plan),
        "run_count": len(results),
        "nonzero_count": sum(1 for row in results if row["returncode"] != 0),
        "provider_route_preflight": provider_preflight,
        "results": results,
    }
    write_json(args.out, payload)
    return 0 if payload["nonzero_count"] == 0 else 1


def _preflight_provider_arms(plan: dict[str, Any]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for arm in plan.get("arms", []):
        if arm.get("backend") != "model_tool_loop" or arm.get("client") != "provider":
            continue
        client = ProviderModelClient(
            command=arm.get("model_command") or None,
            config=ModelClientConfig(
                provider=str(arm.get("model_provider") or "command"),
                model_name=str(arm.get("model") or "provider"),
                timeout_s=int(arm.get("max_tool_time_s") or 120),
            ),
        )
        report = _run_provider_route_preflight(client, task_md="Provider route preflight. Return one valid tool action.")
        arms.append(
            {
                "name": arm.get("name"),
                "model_provider": arm.get("model_provider"),
                "model": arm.get("model"),
                "model_command": arm.get("model_command"),
                "passed": report["passed"],
                "issues": report["issues"],
                "metrics": {
                    **_provider_metrics_for_manifest(model_client_metrics(client)),
                    "total_model_calls": model_client_metrics(client).get("total_model_calls"),
                    "total_tokens_in": model_client_metrics(client).get("total_tokens_in"),
                    "total_tokens_out": model_client_metrics(client).get("total_tokens_out"),
                },
            }
        )
    return {"passed": all(arm["passed"] for arm in arms), "arms": arms}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
