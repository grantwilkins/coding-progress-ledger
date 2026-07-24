"""Sweep migration time for a fixed 100k-session solver comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planner import plan, source_power
from plot_scaling_results import COLORS, LABELS
from power_drain_experiment import DEFAULT_MODEL, build_scenario
from profiles import ModelProfile, WorkloadProfile
from simulate import predict


ROOT = Path(__file__).parent
DEFAULT_WORKLOAD = ROOT / "profiles/coding.json"
DEFAULT_OUT = ROOT / "outputs/scaling_1s_to_3600s_100k_20260721"
MIGRATION_TIMES_S = (1, 3, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600)
SOLVER_SEEDS = {"random": range(10), "greedy": (3,), "lp": (3,)}


def timed_scenario(base, migration_s: float, power_window_s: float):
    deadline = migration_s + power_window_s
    return replace(base, deadline_s=deadline, end_s=deadline)


def summarize(scenario, planned, result, migration_s: float, minimum_power_w: float,
              execute_s: float) -> dict:
    methods = [move.method for move in planned.moves]
    completed = [row.committed_s for row in result.sessions if row.committed_s is not None]
    requested = planned.initial_source_power_w - scenario.power_limit_w
    achieved = planned.initial_source_power_w - result.modeled_source_power_at_deadline_w
    nodes = {node.node_id: node for node in scenario.nodes}
    return {
        "migration_s": migration_s, "deadline_s": scenario.deadline_s,
        "sessions": len(scenario.sessions),
        "source_instances": sum(all(nodes[node].local for node in instance.gpu_nodes)
                                for instance in scenario.instances),
        "source_nodes": sum(node.local for node in scenario.nodes),
        "solver": planned.solver, "seed": planned.seed,
        "initial_source_power_w": planned.initial_source_power_w,
        "minimum_awake_source_power_w": minimum_power_w,
        "requested_source_drop_w": requested,
        "planned_source_drop_w": planned.initial_source_power_w
        - planned.planned_source_power_w,
        "modeled_source_power_at_deadline_w": result.modeled_source_power_at_deadline_w,
        "modeled_source_drop_at_deadline_w": achieved,
        "requested_power_fraction_achieved": achieved / requested,
        "plan_feasible": planned.feasible,
        "planned_moves": len(methods), "replay_moves": methods.count("replay"),
        "kv_moves": methods.count("kv_transfer"),
        "moves_completed_by_budget": sum(time <= migration_s for time in completed),
        "last_completed_migration_s": max(completed, default=0.0),
        "plan_s": planned.solve_s, "execute_s": execute_s,
    }


def bounds(rows: list[dict], value) -> tuple[list[float], ...]:
    grouped = {
        time: [value(row) for row in rows if row["migration_s"] == time]
        for time in sorted({row["migration_s"] for row in rows})
    }
    return (
        list(grouped), [float(np.mean(values)) for values in grouped.values()],
        [min(values) for values in grouped.values()],
        [max(values) for values in grouped.values()],
    )


def _line(ax, rows, value, solver, **style):
    x, mean, low, high = bounds(rows, value)
    ax.plot(x, mean, color=COLORS[solver], marker="o", **style)
    if low != high:
        ax.fill_between(x, low, high, color=COLORS[solver], alpha=.18)


def plot(rows: list[dict], output: Path, power_window_s: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    by_solver = {
        solver: [row for row in rows if row["solver"] == solver]
        for solver in SOLVER_SEEDS
    }
    values = (
        lambda row: 100 * row["planned_moves"] / row["sessions"],
        lambda row: 100 * row["modeled_source_drop_at_deadline_w"]
        / row["requested_source_drop_w"],
        lambda row: 100 * row["moves_completed_by_budget"] / row["planned_moves"]
        if row["planned_moves"] else 0,
        lambda row: row["last_completed_migration_s"],
        lambda row: row["plan_s"],
    )
    for solver, selected in by_solver.items():
        _line(axes[0, 0], selected, values[0], solver, label=LABELS[solver])
        for method, label, linestyle in (
            ("replay_moves", "Replay", "-"), ("kv_moves", "KV transfer", "--")
        ):
            _line(
                axes[0, 1], selected,
                lambda row, key=method: 100 * row[key] / row["sessions"], solver,
                label=f"{LABELS[solver]}: {label}", linestyle=linestyle,
            )
        for ax, value in zip((axes[0, 2], *axes[1]), values[1:]):
            _line(ax, selected, value, solver, label=LABELS[solver])
    axes[0, 2].axhline(100, color="black", linestyle="--", label="Target")
    times = sorted({row["migration_s"] for row in rows})
    axes[1, 1].plot(times, times, "k--", label="Time budget")
    titles = (
        "Sessions selected", "Actions chosen", "Requested power reduction achieved",
        "Last completed migration", "Planning time",
    )
    ylabels = ("Percent", "Percent of sessions", "Percent", "Seconds", "Seconds")
    for ax, title, ylabel in zip(
        (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]),
        titles, ylabels,
    ):
        ax.set(title=title, ylabel=ylabel)
    axes[1, 0].set(title="Selected moves completed before power window", ylabel="Percent")
    for ax in axes.flat:
        ax.set(xlabel="Available migration time (s)", xscale="log")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[1, 1].set_yscale("log")
    axes[1, 2].set_yscale("log")
    reference = rows[0]
    fig.suptitle(
        f"{int(reference['sessions']):,} coding sessions, 1 Gbps shared inter-site, "
        f"50% awake-state power target, {power_window_s:g} s trailing window"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(model_path: Path, workload_path: Path, sessions: int, out: Path) -> list[dict]:
    profile, workload = ModelProfile.load(model_path), WorkloadProfile.load(workload_path)
    record = min(workload.records, key=lambda row: row.context_tokens)
    workload = replace(workload, records=(record,))
    start = perf_counter()
    base, routes = build_scenario(
        workload, profile, sessions, 3, 0, profile.power_window_s,
        profile.power_window_s,
    )
    base = replace(base, sessions=tuple(
        replace(session, requests=(), expected_growth_tokens_per_s=0)
        for session in base.sessions
    ))
    initial = source_power(base, profile)
    minimum = source_power(base, profile, (session.session_id for session in base.sessions))
    base = replace(base, power_limit_w=initial - .5 * (initial - minimum))
    build_s = perf_counter() - start
    rows = []
    for migration_s in MIGRATION_TIMES_S:
        scenario = timed_scenario(base, migration_s, profile.power_window_s)
        for solver, seeds in SOLVER_SEEDS.items():
            for seed in seeds:
                planned = plan(scenario, profile, routes, solver, seed=seed)
                start = perf_counter()
                result = predict(scenario, profile, planned.moves)
                row = summarize(
                    scenario, planned, result, migration_s, minimum, perf_counter() - start,
                )
                rows.append(row)
                print(
                    f"migration_s={migration_s:g} solver={solver} seed={seed} "
                    f"moves={row['planned_moves']} plan_s={row['plan_s']:.3f}",
                    flush=True,
                )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "timing_results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_local": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "model_profile": profile.profile_id, "workload_profile": workload.profile_id,
        "sessions": sessions, "migration_times_s": MIGRATION_TIMES_S,
        "deadline_s": "migration_s + power_window_s",
        "power_window_s": profile.power_window_s, "solvers": list(SOLVER_SEEDS),
        "random_seeds": list(SOLVER_SEEDS["random"]), "scenario_seed": 3,
        "requests_enabled": False, "expected_growth_enabled": False,
        "final_state": "awake", "shared_inter_site_bandwidth_gbps": 1,
        "target_fraction_of_removable_power": .5, "build_s": build_s,
        "error_bounds": "observed minimum to maximum; line is arithmetic mean",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(rows, out / "timing_summary", profile.power_window_s)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workload-profile", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--sessions", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.model_profile, args.workload_profile, args.sessions, args.out)


if __name__ == "__main__":
    main()
