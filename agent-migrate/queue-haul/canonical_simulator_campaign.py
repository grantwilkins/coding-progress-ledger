"""Run the final paired simulator comparison and compact scale check."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planner import plan, source_power
from power_drain_experiment import (
    DEFAULT_MODEL, ExperimentRun, _summary, build_scenario,
)
from profiles import ModelProfile, WorkloadProfile
from simulate import execute, predict


ROOT = Path(__file__).parent
DEFAULT_WORKLOAD = ROOT / "profiles/coding.json"
DEFAULT_OUT = ROOT / "outputs/canonical-simulator"
POLICIES = {
    "queue_haul": "lp_work_first",
    "greedy": "greedy",
    "isolated_fastest": "isolated_fastest",
    "replay_only": "replay_only",
    "kv_only": "kv_only",
}
MIGRATION_S = 120
ROUTE_BYTES_PER_S = 1_250_000_000
TARGET_FRACTIONS = (.1, .25, .5, 1)
SCALE_TARGET_FRACTION = .1
SCALE_SESSIONS_PER_ROUTE = 10_000


def scenario(profile, workload, sessions, seed, target_fraction=.5,
             route_bytes_per_s=ROUTE_BYTES_PER_S):
    shortest = min(workload.records, key=lambda row: row.context_tokens)
    workload = replace(workload, records=(shortest,))
    base, routes = build_scenario(
        workload, profile, sessions, seed, 0, profile.power_window_s,
        profile.power_window_s, route_bytes_per_s,
    )
    base = replace(
        base, deadline_s=MIGRATION_S + profile.power_window_s,
        end_s=MIGRATION_S + profile.power_window_s,
        sessions=tuple(
            replace(row, requests=(), expected_growth_tokens_per_s=0)
            for row in base.sessions
        ),
    )
    initial = source_power(base, profile)
    minimum = source_power(
        base, profile, (session.session_id for session in base.sessions)
    )
    return replace(
        base, power_limit_w=initial - target_fraction * (initial - minimum)
    ), routes, workload


def scenario_id(scenario_) -> str:
    content = (
        scenario_.power_limit_w, scenario_.deadline_s, scenario_.end_s,
        tuple(
            (row.session_id, row.source_instance, row.context_tokens,
             row.expected_f, row.expected_g, row.log_bytes)
            for row in scenario_.sessions
        ),
    )
    return hashlib.sha256(repr(content).encode()).hexdigest()[:16]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eager(planned):
    return replace(
        planned,
        moves=tuple(replace(
            move, rate_limit_bytes_per_s=None, quiesce_s=None
        ) for move in planned.moves),
    )


def enrich(policy, target_fraction, scenario_, planned, result, execution_s,
           workload_id, profile):
    summary = _summary(ExperimentRun(
        f"{scenario_id(scenario_)}:{policy}", workload_id, scenario_,
        planned, "central", result,
    ))
    moved = {move.session_id for move in planned.moves}
    sessions = {row.session_id: row for row in scenario_.sessions}
    case = profile.case()
    completed = {
        row.session_id for row in result.sessions
        if row.committed_s is not None and row.committed_s <= scenario_.deadline_s
    }
    return {
        **summary, "policy": policy, "target_fraction": target_fraction,
        "scenario_id": scenario_id(scenario_),
        "execution_s": execution_s,
        "replay_moves": sum(move.method == "replay" for move in planned.moves),
        "kv_moves": sum(move.method == "kv_transfer" for move in planned.moves),
        "exposed_sessions": len(moved - completed),
        "landed_prefill_tokens_per_s":
            sum(sessions[name].expected_f for name in moved),
        "landed_decode_tokens_per_s":
            sum(sessions[name].expected_g for name in moved),
        "landed_prefill_replicas": sum(
            sessions[name].expected_f
            / case.prefill.rate(sessions[name].context_tokens, 1)
            for name in moved
        ),
        "landed_decode_replicas": sum(
            sessions[name].expected_g
            / case.decode.rate(sessions[name].context_tokens, 1)
            for name in moved
        ),
        "input_provenance": "measured|fitted|assumed",
        "result_provenance": "simulated",
        "evidence_status": "sensitivity",
    }


def policy_runs(profile, workload, sessions, seed, target_fractions):
    rows, details = [], {}
    for target_fraction in target_fractions:
        scenario_, routes, selected_workload = scenario(
            profile, workload, sessions, seed, target_fraction,
        )
        for policy, solver in POLICIES.items():
            planned = eager(plan(scenario_, profile, routes, solver, seed=seed))
            start = perf_counter()
            result = execute(scenario_, profile, planned.moves)
            planned = replace(
                planned,
                expected_source_power_at_deadline_w=
                    result.modeled_source_power_at_deadline_w,
                feasible=planned.planned_source_power_w <= scenario_.power_limit_w
                and result.deadline_met,
            )
            rows.append(enrich(
                policy, target_fraction, scenario_, planned, result,
                perf_counter() - start, selected_workload.profile_id, profile,
            ))
            details[target_fraction, policy] = scenario_, planned, result
    return rows, details


def pooled_scale(scenario_):
    nodes = {node.node_id: node for node in scenario_.nodes}
    sources = tuple(
        instance for instance in scenario_.instances
        if all(nodes[node].local for node in instance.gpu_nodes)
    )
    source_nodes = {
        instance.instance_id: instance.gpu_nodes[0] for instance in sources
    }

    def route(source, destination):
        return (
            f"{source_nodes[source]}-fabric", "source-dc-egress",
            "wan-source-destination", "destination-dc-ingress",
        )

    route.destinations_equivalent = True
    return scenario_, route


def scale_runs(profile, workload, counts, seed):
    rows = []
    for sessions in counts:
        scenario_, routes, _ = scenario(
            profile, workload, sessions, seed, SCALE_TARGET_FRACTION,
            ROUTE_BYTES_PER_S * sessions / SCALE_SESSIONS_PER_ROUTE,
        )
        scenario_, routes = pooled_scale(scenario_)
        planned = eager(plan(
            scenario_, profile, routes, "greedy", seed=seed
        ))
        start = perf_counter()
        result = predict(scenario_, profile, planned.moves)
        rows.append({
            "sessions": sessions, "scenario_id": scenario_id(scenario_),
            "planner": "queue_haul_greedy", "planned_moves": len(planned.moves),
            "replay_moves": sum(move.method == "replay" for move in planned.moves),
            "kv_moves": sum(move.method == "kv_transfer" for move in planned.moves),
            "requested_source_drop_w":
                planned.initial_source_power_w - scenario_.power_limit_w,
            "achieved_source_drop_w":
                planned.initial_source_power_w
                - result.modeled_source_power_at_deadline_w,
            "deadline_met": result.deadline_met,
            "last_commit_s": result.migration_makespan_s,
            "plan_s": planned.solve_s, "execution_s": perf_counter() - start,
            "topology": "pooled_destination",
            "input_provenance": "measured|fitted|assumed",
            "result_provenance": "simulated",
            "evidence_status": "sensitivity",
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_schedule(path, scenario_, planned, result):
    executions = {row.session_id: row for row in result.sessions}
    sessions = {row.session_id: row for row in scenario_.sessions}
    rows = [{
        "scenario_id": scenario_id(scenario_), "session_id": move.session_id,
        "source_instance": sessions[move.session_id].source_instance,
        "destination_instance": move.destination_instance,
        "method": move.method, "order": move.order,
        "initial_start_s": executions[move.session_id].initial_start_s,
        "initial_ready_s": executions[move.session_id].initial_ready_s,
        "commit_s": executions[move.session_id].committed_s,
    } for move in planned.moves]
    write_csv(path, rows)
    return rows


def plot(policy_rows, scale_rows, schedule_rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    maximum = max(row["requested_source_drop_w"] for row in policy_rows) / 1000
    axes[0].plot([0, maximum], [0, maximum], "k--", label="Target")
    for policy in POLICIES:
        selected = sorted(
            (row for row in policy_rows if row["policy"] == policy),
            key=lambda row: row["requested_source_drop_w"],
        )
        axes[0].plot(
            [row["requested_source_drop_w"] / 1000 for row in selected],
            [row["modeled_source_drop_at_deadline_w"] / 1000
             for row in selected],
            "o-", label=policy,
        )
    axes[0].set(
        xlabel="Requested source power (kW)",
        ylabel="Achieved source power (kW)",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        [row["sessions"] for row in scale_rows],
        [row["plan_s"] for row in scale_rows], "o-", label="Plan",
    )
    axes[1].plot(
        [row["sessions"] for row in scale_rows],
        [row["execution_s"] for row in scale_rows], "o-", label="Predict",
    )
    axes[1].set(xscale="log", yscale="log", xlabel="Sessions", ylabel="Seconds")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"canonical_simulator_summary.{suffix}", dpi=180)
    plt.close(fig)

    selected = [
        row for row in schedule_rows
        if row["initial_start_s"] is not None and row["commit_s"] is not None
    ][:20]
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, row in enumerate(selected):
        ax.barh(
            y, row["commit_s"] - row["initial_start_s"],
            left=row["initial_start_s"],
            color="#4C78A8" if row["method"] == "replay" else "#F58518",
        )
    ax.set(
        xlabel="Time (s)", ylabel="Queue-Haul order",
        yticks=range(len(selected)),
        yticklabels=[f"{row['order']}: {row['method']}" for row in selected],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"representative_schedule.{suffix}", dpi=180)
    plt.close(fig)


def run(model_path=DEFAULT_MODEL, workload_path=DEFAULT_WORKLOAD,
        out=DEFAULT_OUT, main_sessions=10_000,
        scale_sessions=(10_000, 100_000, 1_000_000), seed=3,
        target_fractions=TARGET_FRACTIONS):
    profile = ModelProfile.load(model_path)
    workload = WorkloadProfile.load(workload_path)
    out.mkdir(parents=True, exist_ok=True)
    policy_rows, details = policy_runs(
        profile, workload, main_sessions, seed, target_fractions,
    )
    scale_rows = scale_runs(profile, workload, scale_sessions, seed)
    scenario_, planned, result = details[target_fractions[0], "queue_haul"]
    schedule_rows = write_schedule(
        out / "representative_schedule.csv", scenario_, planned, result,
    )
    write_csv(out / "policy_summary.csv", policy_rows)
    write_csv(out / "scale_summary.csv", scale_rows)
    (out / "run_metadata.json").write_text(json.dumps({
        "execution_contract": "eager_one_stream_per_source",
        "main_sessions": main_sessions, "scale_sessions": scale_sessions,
        "policies": POLICIES, "scale_policy": "queue_haul_greedy",
        "scale_topology": "equivalent pooled destination",
        "scale_target_fraction": SCALE_TARGET_FRACTION,
        "scale_route_contract": "10 Gbps per 10K sessions",
        "migration_s": MIGRATION_S, "power_window_s": profile.power_window_s,
        "route_bytes_per_s": ROUTE_BYTES_PER_S,
        "target_fractions_of_removable_power": target_fractions,
        "planner_expected_growth": False,
        "sampled_requests_enabled": False,
        "destination_contract": "dedicated matched TP=1 pool; assumed sensitivity",
        "model_profile": {
            "id": profile.profile_id, "path": str(model_path),
            "sha256": file_sha256(model_path),
        },
        "workload_profile": {
            "id": workload.profile_id, "path": str(workload_path),
            "sha256": file_sha256(workload_path),
        },
        "campaign_sha256": file_sha256(Path(__file__)),
        "seed": seed,
    }, indent=2) + "\n")
    plot(policy_rows, scale_rows, schedule_rows, out)
    return policy_rows, scale_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workload-profile", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--main-sessions", type=int, default=10_000)
    parser.add_argument(
        "--scale-sessions", type=lambda value: tuple(map(int, value.split(","))),
        default=(10_000, 100_000, 1_000_000),
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--target-fractions", type=lambda value: tuple(map(float, value.split(","))),
        default=TARGET_FRACTIONS,
    )
    args = parser.parse_args()
    run(
        args.model_profile, args.workload_profile, args.out,
        args.main_sessions, args.scale_sessions, args.seed,
        args.target_fractions,
    )


if __name__ == "__main__":
    main()
