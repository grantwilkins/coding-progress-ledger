"""Paired simulator experiment for the experimental dual-Lagrangian selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

import destination_bench as bench
from migration import ORDERED_EAGER_PARALLEL_V1
from planner import plan, source_power
from profiles import ModelProfile


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "outputs/dual-lagrangian-experiment"
SOLVERS = ("greedy", "greedy_bundle", "greedy_prefix", "greedy_coupled")


def power_limit(initial, minimum, fraction):
    if not 0 <= fraction <= 1 or minimum > initial:
        raise ValueError("invalid removable-power target")
    return initial - fraction * (initial - minimum)


def parse_solvers(value):
    solvers = tuple(value.split(","))
    if not solvers or not set(solvers) <= set(SOLVERS):
        raise ValueError("unknown experimental solver")
    return solvers


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_stack_hash(root=ROOT):
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rss_bytes(value, platform=sys.platform):
    return value if platform == "darwin" else value * 1024


def result_row(result, workload, sessions, target_fraction, scenario_id,
               source_instances=None, max_source_width=None, requested_shed_w=None,
               minimum_awake_source_power_w=None):
    migration_work = sum(
        row.used for row in result.resource_uses
        if row.name.startswith("migration:")
    )
    selected_shed = result.initial_source_power_w - result.planned_source_power_w
    achieved_shed = result.initial_source_power_w \
        - result.expected_source_power_at_deadline_w
    unmet = None if requested_shed_w is None else max(
        0, requested_shed_w - achieved_shed,
    )
    return {
        "scenario_id": scenario_id, "workload": workload,
        "seed": result.seed,
        "sessions": sessions, "target_fraction": target_fraction,
        "source_instances": source_instances, "max_source_width": max_source_width,
        "solver": result.solver, "feasible": result.feasible,
        "initial_source_power_w": result.initial_source_power_w,
        "minimum_awake_source_power_w": minimum_awake_source_power_w,
        "requested_shed_w": requested_shed_w,
        "selected_shed_w": selected_shed,
        "validated_shed_w": achieved_shed if result.feasible else 0,
        "expected_source_power_at_deadline_w":
            result.expected_source_power_at_deadline_w,
        "unmet_shed_w": unmet,
        "shortfall_w": result.power_shortfall_w,
        "migration_work_s": migration_work,
        "moves": len(result.moves),
        "replay_moves": sum(move.method == "replay" for move in result.moves),
        "kv_moves": sum(move.method == "kv_transfer" for move in result.moves),
        "solve_s": result.solve_s,
        "planner_memory_bytes": result.planner_memory_bytes,
        "makespan_s": result.predicted_migration_makespan_s,
        "packing_repairs": result.packing_repair_count,
        "service_debt_replica_s": result.service_debt_replica_s,
        "required_recovery_s": result.required_recovery_s,
        "failure": result.failure_reason,
        "binding_resources": "|".join(result.binding_resources),
        "resource_uses": json.dumps([
            {"name": row.name, "used": row.used, "capacity": row.capacity,
             "utilization": row.utilization}
            for row in result.resource_uses
        ], separators=(",", ":")),
        "execution_contract": ORDERED_EAGER_PARALLEL_V1,
        "input_provenance": "measured|fitted|assumed",
        "result_provenance": "simulated",
        "evidence_status": "sensitivity",
    }


def run_case(profile, shapes, workload, sessions, seed, target_fractions,
             solvers=SOLVERS):
    sampled = bench.sample_sessions(
        shapes, sessions, seed, bench.log_bytes_per_token(bench.WORKLOADS[workload]),
    )
    packed, replicas = bench.pack_source(sampled, profile)
    widths = {}
    for session in packed:
        widths[session.source_instance] = widths.get(session.source_instance, 0) + 1
    base = bench.scenario(profile, packed, replicas, bench.Pressure())
    profile = bench.extrapolate_replay(profile, base.sessions, base.deadline_s)
    architecture = bench.architecture(
        profile, base.sessions, replicas, bench.Pressure(),
    )
    initial = source_power(base, profile)
    minimum = source_power(base, profile, (row.session_id for row in base.sessions))
    rows = []
    for target_fraction in target_fractions:
        scenario = replace(
            base, power_limit_w=power_limit(initial, minimum, target_fraction),
        )
        scenario_id = f"{workload}:{sessions}:{seed}:{target_fraction:g}"
        for solver in solvers:
            result = plan(
                scenario, profile, {}, solver, seed=seed,
                destination=architecture,
            )
            rows.append(result_row(
                result, workload, sessions, target_fraction, scenario_id,
                len(widths), max(widths.values()), initial - scenario.power_limit_w,
                minimum,
            ))
    return rows


def summarize(rows):
    baseline = {
        row["scenario_id"]: row for row in rows
        if row["solver"] == "greedy" and row["feasible"]
    }
    summary = []
    for sessions, target, solver in sorted({
        (row["sessions"], row["target_fraction"], row["solver"])
        for row in rows
    }):
        selected = [
            row for row in rows if (row["sessions"], row["target_fraction"],
                                    row["solver"]) == (sessions, target, solver)
        ]
        feasible = [row for row in selected if row["feasible"]]
        paired = [
            row["migration_work_s"] - baseline[row["scenario_id"]]["migration_work_s"]
            for row in feasible if row["scenario_id"] in baseline
        ]
        summary.append({
            "sessions": sessions, "target_fraction": target, "solver": solver,
            "cases": len(selected), "feasible": len(feasible),
            "median_solve_s": float(np.median([row["solve_s"] for row in selected])),
            "median_feasible_migration_work_s": float(np.median([
                row["migration_work_s"] for row in feasible
            ])) if feasible else None,
            "both_greedy_feasible": len(paired),
            "median_paired_work_delta_vs_greedy_s":
                float(np.median(paired)) if paired else None,
        })
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(out=DEFAULT_OUT, counts=(80, 240), seeds=range(3),
        target_fractions=(.1, .5, .9), workload="coding", solvers=SOLVERS):
    started = perf_counter()
    profile = ModelProfile.load(bench.DEFAULT_MODEL)
    manifest = json.loads(bench.DEFAULT_MANIFEST.read_text())
    shapes = bench.trace_shapes(manifest, workload)
    counts, seeds, target_fractions = tuple(counts), tuple(seeds), tuple(target_fractions)
    rows = [
        row for sessions in counts for seed in seeds
        for row in run_case(
            profile, shapes, workload, sessions, seed, target_fractions, solvers,
        )
    ]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "results.csv", rows)
    (out / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
    (out / "run_metadata.json").write_text(json.dumps({
        "experimental": True, "core_default_changed": False,
        "execution_contract": ORDERED_EAGER_PARALLEL_V1,
        "input_provenance": "measured|fitted|assumed",
        "result_provenance": "simulated", "evidence_status": "sensitivity",
        "counts": counts, "seeds": seeds, "targets": target_fractions,
        "workload": workload, "solvers": solvers,
        "model_sha256": file_hash(bench.DEFAULT_MODEL),
        "manifest_sha256": file_hash(bench.DEFAULT_MANIFEST),
        "workload_sha256": file_hash(bench.WORKLOADS[workload]),
        "campaign_sha256": file_hash(Path(__file__)),
        "planner_code_sha256": bench.code_sha256(),
        "experiment_stack_sha256": experiment_stack_hash(),
        "campaign_wall_s": perf_counter() - started,
        "peak_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--counts", type=lambda x: tuple(map(int, x.split(","))),
                        default=(80, 240))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--targets", type=lambda x: tuple(map(float, x.split(","))),
                        default=(.1, .5, .9))
    parser.add_argument("--workload", choices=bench.CLASSES, default="coding")
    parser.add_argument("--solvers", type=parse_solvers, default=SOLVERS)
    args = parser.parse_args()
    run(args.out, args.counts, range(args.seeds), args.targets, args.workload,
        args.solvers)


if __name__ == "__main__":
    main()
