#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from coding_data_collection.artifacts import read_json, utc_now, write_json
from coding_data_collection.audits import corpus_hardening_report
from coding_data_collection.estimator_artifacts import build_estimator_artifacts
from coding_data_collection.pilot_gates import pilot_gate_report, write_pilot_gate_outputs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execute_pilot_plan_continue import _preflight_provider_arms


REPLACEABLE_STATUSES = {"environment_setup_failure", "setup_failure", "infrastructure_failure"}
REPLACEABLE_TERMINATIONS = {
    "agent_readiness_preflight_failed",
    "provider_route_preflight_failed",
    "agent_container_start_failed",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptively complete an OpenAI-only L1 corpus from an overcomplete plan.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted-out", type=Path, required=True)
    parser.add_argument("--rejected-out", type=Path, required=True)
    parser.add_argument("--gate-out", type=Path, required=True)
    parser.add_argument("--failure-out", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--source-id", default="terminal_bench_pilot")
    parser.add_argument("--estimator-root", type=Path, default=Path("../coding-estimator"))
    parser.add_argument("--target-eligible-runs", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=14)
    parser.add_argument("--append-arg", action="append", default=[])
    parser.add_argument("--provider-route-preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-replaceable-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, rerun existing provider/setup failures but reuse existing eligible runs.",
    )
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("passed"):
        sys.stderr.write("refusing to execute blocked plan\n")
        return 2

    provider_preflight = _preflight_provider_arms(plan) if args.provider_route_preflight else {"passed": True, "arms": []}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    if not provider_preflight["passed"]:
        rejected.extend(
            {
                "kind": "provider_arm_preflight_failed",
                "arm": arm.get("name"),
                "model": arm.get("model"),
                "issues": arm.get("issues", []),
            }
            for arm in provider_preflight["arms"]
            if not arm.get("passed")
        )
        _write_outputs(args, plan, provider_preflight, attempts, accepted, rejected, final_gate=None)
        return 2

    final_gate = None
    for run in plan.get("planned_runs", [])[: args.max_attempts]:
        command = [str(part) for part in run["command"]]
        command.extend(args.append_arg)
        run_dir = Path(run["command"][run["command"].index("--run-dir") + 1])
        existing = classify_run(run_dir) if (run_dir / "run_manifest.json").is_file() else None
        if existing and (existing["accepted"] or not args.retry_replaceable_existing):
            proc = subprocess.CompletedProcess(args=command, returncode=0)
            print(f"[adaptive] reuse {run['run_id']}", flush=True)
        else:
            print(f"[adaptive] run {run['run_id']}", flush=True)
            proc = subprocess.run(command, text=True)
        record = {
            "run_id": run["run_id"],
            "task_id": run["task_id"],
            "arm": run["arm"],
            "run_dir": str(run_dir),
            "returncode": proc.returncode,
        }
        attempts.append(record)

        decision = classify_run(run_dir)
        if decision["accepted"]:
            hardening = corpus_hardening_report([run_dir])
            if not _hard_safety_passed(hardening):
                rejected.append({**record, "kind": "hard_gate_failed", "issues": hardening})
                _write_outputs(args, plan, provider_preflight, attempts, accepted, rejected, final_gate=None)
                return 1
            accepted.append({**record, **decision})
        else:
            rejected.append({**record, **decision})

        if len(accepted) >= args.target_eligible_runs:
            final_gate = _build_and_gate(args, [Path(row["run_dir"]) for row in accepted])
            if final_gate["passed"]:
                break

    if final_gate is None and accepted:
        final_gate = _build_and_gate(args, [Path(row["run_dir"]) for row in accepted])
    _write_outputs(args, plan, provider_preflight, attempts, accepted, rejected, final_gate=final_gate)
    return 0 if final_gate and final_gate["passed"] else 1


def classify_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {"accepted": False, "kind": "missing_run_manifest", "issues": ["run_manifest.json missing"]}
    manifest = read_json(manifest_path)
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    status = str(manifest.get("run_status") or "")
    termination = str(manifest.get("termination_reason") or "")
    if bool(metrics.get("eligible_for_L_gate")):
        return {"accepted": True, "kind": "eligible_run", "run_status": status, "termination_reason": termination}
    if status in REPLACEABLE_STATUSES or termination in REPLACEABLE_TERMINATIONS:
        return {"accepted": False, "kind": "replaceable_preflight_or_setup_failure", "run_status": status, "termination_reason": termination}
    return {"accepted": False, "kind": "l_ineligible_run", "run_status": status, "termination_reason": termination}


def _hard_safety_passed(hardening: dict[str, Any]) -> bool:
    return bool(hardening.get("artifact_completeness", {}).get("passed")) and bool(
        hardening.get("redaction", {}).get("passed")
    )


def _build_and_gate(args: argparse.Namespace, run_dirs: list[Path]) -> dict[str, Any]:
    build_estimator_artifacts(
        corpus_id=args.corpus_id,
        source_id=args.source_id,
        run_dirs=run_dirs,
        estimator_root=args.estimator_root,
        artifact_dir=args.artifact_dir,
        replace_staged_runs=True,
    )
    report = pilot_gate_report(run_dirs, estimator_artifact_dir=args.artifact_dir)
    write_pilot_gate_outputs(report=report, report_path=args.gate_out, failure_path=args.failure_out)
    return report


def _write_outputs(
    args: argparse.Namespace,
    plan: dict[str, Any],
    provider_preflight: dict[str, Any],
    attempts: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    final_gate: dict[str, Any] | None,
) -> None:
    payload = {
        "created_at": utc_now(),
        "plan": str(args.plan),
        "planned_run_count": len(plan.get("planned_runs", [])),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "target_eligible_runs": args.target_eligible_runs,
        "max_attempts": args.max_attempts,
        "provider_route_preflight": provider_preflight,
        "passed": bool(final_gate and final_gate.get("passed")),
        "target_eligible_runs_met": len(accepted) >= args.target_eligible_runs,
        "final_gate_inputs": final_gate.get("gate_inputs") if final_gate else None,
        "attempts": attempts,
    }
    write_json(args.out, payload)
    write_json(args.accepted_out, {"created_at": utc_now(), "runs": accepted})
    write_json(args.rejected_out, {"created_at": utc_now(), "runs": rejected})


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
