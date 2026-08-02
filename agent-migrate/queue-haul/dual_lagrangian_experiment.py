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
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

import destination_bench as bench
from migration import ORDERED_EAGER_PARALLEL_V1
from planner import plan, source_power
from pool_planner import candidate_table
from power_model import ExpectedPower
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


def chord_candidate_gains(table, power):
    if power.profile.power_scope != "gpu":
        raise ValueError("source-chord bound currently requires GPU-scoped power")
    members = {}
    for session in table.sessions:
        members.setdefault(session.source_instance, []).append(session)
    session_gain = {}
    for sessions in members.values():
        total_load = sum(power.ell[row.session_id] for row in sessions)
        full_gain = power.drain_gain(row.session_id for row in sessions)
        for row in sessions:
            session_gain[row.session_id] = (
                full_gain * power.ell[row.session_id] / total_load
                if total_load else 0
            )
    return np.asarray([
        session_gain[table.sessions[candidate.session].session_id]
        for candidate in table.candidates
    ])


def fractional_chord_work_bound(table, target, power):
    if target <= 0:
        return 0.0
    gains = chord_candidate_gains(table, power)
    scale = max(target, gains.max(initial=0), 1.0)
    matrix = vstack((
        table.incidence, table.resources,
        csr_matrix((-gains / scale).reshape(1, -1)),
    ), format="csr")
    result = linprog(
        [candidate.migration_work_s for candidate in table.candidates],
        A_ub=matrix,
        b_ub=np.concatenate((np.ones(matrix.shape[0] - 1), [-target / scale])),
        bounds=(0, None), method="highs-ds", options={"presolve": True},
    )
    if result.status == 2:
        return None
    if result.status:
        raise RuntimeError(f"HiGHS chord bound failed: {result.message}")
    return float(result.fun)


def result_row(result, workload, sessions, target_fraction, scenario_id,
               source_instances=None, max_source_width=None, requested_shed_w=None,
               minimum_awake_source_power_w=None, work_lower_bound_s=None):
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
    gap = None if not result.feasible or not work_lower_bound_s else \
        100 * (migration_work / work_lower_bound_s - 1)
    if gap is not None and gap < -1e-6:
        raise RuntimeError("feasible greedy beat its chord-LP work lower bound")
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
        "lp_chord_work_lower_bound_s": work_lower_bound_s,
        "work_gap_to_lp_percent": gap,
        "target_overshoot_percent": None if not requested_shed_w else
            100 * (achieved_shed / requested_shed_w - 1),
        "moves": len(result.moves),
        "replay_moves": sum(move.method == "replay" for move in result.moves),
        "kv_moves": sum(move.method == "kv_transfer" for move in result.moves),
        "solve_s": result.solve_s,
        "planner_memory_bytes": result.planner_memory_bytes,
        "makespan_s": result.predicted_migration_makespan_s,
        "packing_repairs": result.packing_repair_count,
        "service_debt_replica_s": result.service_debt_replica_s,
        "required_recovery_s": result.required_recovery_s,
        "admission_mode": getattr(result, "admission_mode", None),
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
        selection = replace(scenario, final_state="awake", assumed_shutdown_s=None)
        power = ExpectedPower(selection, profile)
        target = initial - scenario.power_limit_w
        bounds = {}
        for solver in solvers:
            result = plan(
                scenario, profile, {}, solver, seed=seed,
                destination=architecture,
            )
            if result.admission_mode not in bounds:
                table = candidate_table(
                    scenario, profile, architecture, result.admission_mode, power,
                )
                bounds[result.admission_mode] = fractional_chord_work_bound(
                    table, target, power,
                )
            rows.append(result_row(
                result, workload, sessions, target_fraction, scenario_id,
                len(widths), max(widths.values()), initial - scenario.power_limit_w,
                minimum, bounds[result.admission_mode],
            ))
    return rows


def plot_relaxation_gap(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted({row["target_fraction"] for row in rows})
    solvers = [solver for solver in SOLVERS if any(
        row["solver"] == solver for row in rows
    )]
    labels = {
        "greedy": "Static", "greedy_bundle": "Bundles",
        "greedy_prefix": "Lagrangian\nprefix", "greedy_coupled": "Coupled",
    }
    fig, axes = plt.subplots(2, len(targets), figsize=(3.4 * len(targets), 5.5),
                             squeeze=False)
    for column, target in enumerate(targets):
        selected = [row for row in rows if row["target_fraction"] == target]
        total = len({row["scenario_id"] for row in selected})
        counts = [sum(row["solver"] == solver and row["feasible"]
                      for row in selected) for solver in solvers]
        axis = axes[0, column]
        axis.bar(range(1, len(solvers) + 1),
                 [100 * count / total for count in counts])
        axis.bar_label(axis.containers[0],
                       [f"{count}/{total}" for count in counts])
        axis.set_ylim(0, 108)
        axis.set_title(f"{target:.0%} shed target")
        axis.set_xticks(range(1, len(solvers) + 1),
                        [labels[solver] for solver in solvers])
        axis.tick_params(axis="x", labelsize=8)

        by_scenario = {}
        for row in selected:
            by_scenario.setdefault(row["scenario_id"], {})[row["solver"]] = row
        common = [case for case in by_scenario.values() if all(
            solver in case and case[solver]["work_gap_to_lp_percent"] is not None
            for solver in solvers
        )]
        data = [[case[solver]["work_gap_to_lp_percent"] for case in common]
                for solver in solvers]
        axis = axes[1, column]
        axis.boxplot(data, tick_labels=[labels[solver] for solver in solvers],
                     showfliers=False, medianprops={"color": "black"})
        for position, values in enumerate(data, 1):
            axis.scatter(np.full(len(values), position), values, s=10, alpha=.45)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_title(f"Paired feasible n={len(common)}", fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
    axes[0, 0].set_ylabel("Plans feasible (%)")
    axes[1, 0].set_ylabel("Work above chord-LP lower bound (%)")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


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


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                for field in ("target_fraction", "work_gap_to_lp_percent"):
                    row[field] = float(row[field]) if row[field] else None
                row["feasible"] = row["feasible"] == "True"
                rows.append(row)
    return rows


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
    plot_relaxation_gap(rows, out / "work_above_chord_bound")
    (out / "run_metadata.json").write_text(json.dumps({
        "experimental": True, "core_default_changed": False,
        "gap_definition": "feasible work / fractional source-chord LP work - 1",
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
    parser.add_argument("--combine", type=Path, nargs="+")
    args = parser.parse_args()
    if args.combine:
        args.out.mkdir(parents=True, exist_ok=True)
        rows = read_rows(args.combine)
        write_csv(args.out / "results.csv", rows)
        plot_relaxation_gap(rows, args.out / "work_above_chord_bound")
        (args.out / "run_metadata.json").write_text(json.dumps({
            "experimental": True,
            "gap_definition": "feasible work / fractional source-chord LP work - 1",
            "inputs": {str(path): file_hash(path) for path in args.combine},
        }, indent=2) + "\n")
        return
    run(args.out, args.counts, range(args.seeds), args.targets, args.workload,
        args.solvers)


if __name__ == "__main__":
    main()
