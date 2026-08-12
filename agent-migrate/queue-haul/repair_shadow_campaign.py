"""Run the minimal RAMR shadow validation for feasibility repair."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

from repair_controller import (Assignment, Attempt, AttemptUpdate,
                               FeasibilityRepairController, ObservationBatch,
                               ProposedDiff, RepairMove, RepairRequest,
                               RepairResult, RevisedMaximum)


ROOT = Path(__file__).parent
ETA_GUARD_S = 0.13832658875000023
CONDITIONS = ("bandwidth", "replay", "both")


def _run(condition, repeat, raw):
    cut = random.Random(repeat).randint(5, 25)
    assignments = {
        "bandwidth": Assignment("kv_transfer", "t0", "p0"),
        "replay": Assignment("replay", "t1", "p1"),
    }
    attempts = tuple(Attempt(
        session, 0, assignment, "running", 280, 0, 0, 28, rate=10,
    ) for session, assignment in assignments.items())
    controller = FeasibilityRepairController(
        attempts, assignments, 20, 30, ETA_GUARD_S,
        lambda sessions: 10 * len(sessions),
    )
    affected = set(assignments) if condition == "both" else {condition}
    completed = dict.fromkeys(assignments, 0.0)
    requests = pre_cut = changes = 0
    outcome = ""
    trace = raw / f"{condition}-{repeat}.jsonl"
    with trace.open("w") as handle:
        for sample in range(1, 31):
            rates, updates = {}, []
            for session in assignments:
                rate = 10 + sample % 2 if sample <= cut or session not in affected else 1
                rates[session] = rate
                completed[session] = min(280, completed[session] + rate)
                updates.append(AttemptUpdate(
                    session, 0, "committed" if completed[session] == 280 else "running",
                    280, completed[session],
                ))
            decision = controller.observe(ObservationBatch(
                sample, sample, tuple(updates),
                (("wan", rates["bandwidth"]),),
                (("p1", rates["replay"]),),
            ))
            if isinstance(decision, RepairRequest):
                requests += 1
                pre_cut += sample <= cut
                if condition == "both":
                    decision = controller.complete_repair(RepairResult(
                        decision.request_id, decision.snapshot.budget_version,
                        (), 10, False,
                    ), sample)
                else:
                    moves = []
                    for session, attempt in controller.attempts.items():
                        if attempt.status not in {"pending", "running"}:
                            continue
                        assignment = attempt.assignment if session not in affected else \
                            Assignment(attempt.assignment.method, "spare", "spare")
                        duration = (attempt.total_work - attempt.completed_work) / 10
                        moves.append(RepairMove(
                            session, assignment, duration, attempt.total_work,
                            attempt.commit_overhead_s,
                        ))
                    decision = controller.complete_repair(RepairResult(
                        decision.request_id, decision.snapshot.budget_version,
                        tuple(moves), 20, True,
                    ), sample)
                    changes = len(decision.changes)
                    controller.acknowledge(decision.proposal_id, "shadow", sample)
            if isinstance(decision, ProposedDiff):
                outcome = "proposal"
            elif isinstance(decision, RevisedMaximum):
                outcome = "revised_maximum"
            handle.write(json.dumps({
                "sample": sample, "cut": cut, "rates": rates,
                "decision": type(decision).__name__ if decision else None,
            }) + "\n")
    row = {
        "condition": condition, "repeat": repeat, "cut_s": cut,
        "bandwidth_before_gbps": 10, "bandwidth_after_gbps": 1,
        "replay_rho_before": 0, "replay_rho_after": .8,
        "pre_cut_requests": pre_cut, "repair_requests": requests,
        "outcome": outcome, "changed_assignments": changes,
    }
    row["passed"] = requests == 1 and pre_cut == 0 and (
        outcome == "revised_maximum" and changes == 0 if condition == "both"
        else outcome == "proposal" and changes == 1
    )
    return row


def run_campaign(out: Path, raw: Path, allocation_id: str):
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    rows = [_run(condition, repeat, raw)
            for condition in CONDITIONS for repeat in range(3)]
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    report = {
        "schema": "queue-haul-repair-shadow-v1",
        "allocation_id": allocation_id,
        "gpu_inventory": os.environ.get("QH_GPU_INVENTORY", "unrecorded"),
        "eta_guard_s": ETA_GUARD_S,
        "eta_guard_source": "outputs/coding-run/profile_evaluation.csv",
        "scenarios": len(rows), "passed": all(row["passed"] for row in rows),
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("repair shadow validation failed")
    return rows


def prepare(out: Path, raw: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    sbatch = out / "run.sbatch"
    sbatch.write_text(f"""#!/bin/bash
#SBATCH --job-name=qh-repair-shadow
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=repair-shadow-%j.out
#SBATCH --error=repair-shadow-%j.err
set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
export QH_GPU_INVENTORY="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
test "$(grep -c A100-SXM4-80GB <<< "$QH_GPU_INVENTORY")" = 2
cd "{ROOT.parent}"
uv run python queue-haul/repair_shadow_campaign.py run --out "{out / 'reduced'}" --raw "{raw}"
""")
    return sbatch


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--raw", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_campaign(args.out, args.raw, os.environ.get("SLURM_JOB_ID", "local"))
    else:
        print(prepare(args.out, args.raw))


if __name__ == "__main__":
    main()
