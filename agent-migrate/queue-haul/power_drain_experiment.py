"""Profile-driven power-drain planning and event simulation."""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planner import PlanResult, SOLVERS, plan
from profiles import ModelProfile, WorkloadProfile
from simulate import (ExecutionResult, ExecutionScenario, NetworkLink, PowerNode, ServingInstance,
                      SimRequest, SimSession, execute)


ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
DEFAULT_WORKLOADS = tuple(ROOT / f"profiles/{name}.json" for name in (
    "interactive_coding", "coding", "agentic_tool_loop",
))


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    workload_id: str
    scenario: ExecutionScenario
    plan: PlanResult
    case_id: str
    result: ExecutionResult


def _requests(record, end_s: float, rng) -> tuple[SimRequest, ...]:
    if record.state != "active":
        return ()
    requests, elapsed = [], 0.0
    while True:
        gap = float(rng.exponential(record.request_gap_s) + record.tool_delay_s)
        elapsed += gap
        if elapsed > end_s:
            return tuple(requests)
        requests.append(SimRequest(gap, record.prompt_tokens, record.output_tokens))


def _validate_contexts(sessions: tuple[SimSession, ...], profile: ModelProfile) -> None:
    # TODO(context): extend the measured curves before running longer workload windows.
    for case_id, case in profile.cases.items():
        for session in sessions:
            context = session.context_tokens
            try:
                case.replay.rate(context, 1)
                for request in session.requests:
                    case.prefill.rate(context, 1)
                    if request.output_tokens:
                        case.decode.rate(context + request.prompt_tokens, 1)
                    context += request.prompt_tokens + request.output_tokens
                    case.replay.rate(context, 1)
            except ValueError as exc:
                raise ValueError(
                    f"session {session.session_id} exceeds {case_id!r} measured context range: {exc}"
                ) from None


def build_scenario(workload: WorkloadProfile, profile: ModelProfile, sessions: int, seed: int,
                   power_limit_w: float, deadline_s: float, end_s: float,
                   link_bytes_per_s: float = 125_000_000.0,
                   final_state: str = "awake", controller_delay_s: float = 0.0):
    # TODO(workloads): replace the small assumed record sets with held-out traces.
    records, case = workload.sample(sessions, seed), profile.case()
    cycles = np.array([r.request_gap_s + r.tool_delay_s for r in records])
    if any(r.state == "active" and cycle <= 0 for r, cycle in zip(records, cycles)):
        raise ValueError("active sessions require a positive request cycle")
    # TODO(load-cycle): include measured service time when the workload profile records it.
    ell = np.array([
        (r.prompt_tokens / cycle) / case.F + (r.output_tokens / cycle) / case.G
        if r.state == "active" else 0.0 for r, cycle in zip(records, cycles)
    ])
    if np.any(ell > profile.max_ell):
        raise ValueError("a sampled session exceeds the calibrated load range")
    assignment, loads = np.empty(sessions, int), []
    heap = []
    for j in np.argsort(-ell, kind="stable"):
        if heap and heap[0][0] + ell[j] <= profile.max_ell:
            load, gpu = heapq.heappop(heap)
        else:
            gpu, load = len(loads), 0.0
            loads.append(0.0)
        assignment[j], loads[gpu] = gpu, load + ell[j]
        heapq.heappush(heap, (loads[gpu], gpu))
    gpu_count = len(loads)
    node_count = math.ceil(gpu_count / profile.gpus_per_node)
    nodes = tuple(
        PowerNode(f"source-node-{i}", profile.gpus_per_node, True) for i in range(node_count)
    ) + tuple(
        PowerNode(f"dest-node-{i}", profile.gpus_per_node, False) for i in range(node_count)
    )
    instances = tuple(
        ServingInstance(f"source-{i}", (f"source-node-{i // profile.gpus_per_node}",))
        for i in range(gpu_count)
    ) + tuple(
        ServingInstance(f"dest-{i}", (f"dest-node-{i // profile.gpus_per_node}",))
        for i in range(gpu_count)
    )
    rng = np.random.default_rng(seed + 1)
    request_horizon = end_s - controller_delay_s
    wake_horizon = max(0.0, deadline_s - controller_delay_s)
    # TODO(wake): fit the first-request distribution from complete session traces.
    sampled = tuple(
        SimSession(
            str(j), f"source-{assignment[j]}", r.context_tokens,
            r.prompt_tokens / cycle if r.state == "active" else 0.0,
            r.output_tokens / cycle if r.state == "active" else 0.0,
            r.log_bytes, r.log_external, _requests(r, request_horizon, rng), True,
            0.0 if r.state != "active" or wake_horizon <= r.tool_delay_s else
            1.0 if r.request_gap_s == 0 else
            1 - math.exp(-(wake_horizon - r.tool_delay_s) / r.request_gap_s),
        ) for j, (r, cycle) in enumerate(zip(records, cycles))
    )
    _validate_contexts(sampled, profile)
    # TODO(site-links): add shared site limits after they are measured.
    links = tuple(NetworkLink(f"source-node-{i}-egress", link_bytes_per_s)
                  for i in range(node_count)) + tuple(
        NetworkLink(f"dest-node-{i}-ingress", link_bytes_per_s) for i in range(node_count)
    )
    source_node = {f"source-{i}": i // profile.gpus_per_node for i in range(gpu_count)}
    dest_node = {f"dest-{i}": i // profile.gpus_per_node for i in range(gpu_count)}

    def route(source: str, destination: str) -> tuple[str, ...]:
        return (f"source-node-{source_node[source]}-egress",
                f"dest-node-{dest_node[destination]}-ingress")

    return ExecutionScenario(
        deadline_s, end_s, power_limit_w, final_state, controller_delay_s,
        nodes, instances, sampled, links,
    ), route


def run(model_path: Path = DEFAULT_MODEL, workload_paths=DEFAULT_WORKLOADS,
        sessions: int = 10_000, seed: int = 0, power_limits=(10_000.0,),
        deadlines=(120.0,), end_s: float = 180.0, solvers=SOLVERS,
        link_bytes_per_s: float = 125_000_000.0,
        final_state: str = "awake", controller_delay_s: float = 0.0) -> list[ExperimentRun]:
    profile = ModelProfile.load(model_path)
    runs = []
    for workload_path in workload_paths:
        workload = WorkloadProfile.load(workload_path)
        for power_limit in power_limits:
            for deadline in deadlines:
                scenario, routes = build_scenario(
                    workload, profile, sessions, seed, power_limit, deadline, end_s,
                    link_bytes_per_s, final_state,
                    controller_delay_s,
                )
                for solver in solvers:
                    planned = plan(scenario, profile, routes, solver, "central", seed)
                    for case_id in profile.cases:
                        result = execute(scenario, profile, planned.moves, case_id)
                        run_id = f"{workload.profile_id}:{power_limit:g}:{deadline:g}:{solver}:{case_id}:{seed}"
                        runs.append(ExperimentRun(
                            run_id, workload.profile_id, scenario, planned, case_id, result
                        ))
    return runs


def excess_energy(power, start_s: float, end_s: float, limit_w: float) -> float:
    if not power or power[0][0] > start_s or end_s < start_s:
        raise ValueError("power points must cover the violation interval")
    area, cursor, value = 0.0, start_s, power[0][1]
    for point in power:
        if point[0] <= start_s:
            value = point[1]
        elif point[0] <= end_s:
            area += (point[0] - cursor) * max(0.0, value - limit_w)
            cursor, value = point[0], point[1]
    return area + (end_s - cursor) * max(0.0, value - limit_w)


def _summary(run: ExperimentRun) -> dict:
    scenario, result = run.scenario, run.result
    completed = [row for row in result.sessions if row.committed_s is not None]
    resumed = all(row.committed_s is not None and row.committed_s <= scenario.deadline_s
                  for row in result.sessions)
    pauses = [row.committed_s - row.pause_s for row in completed if row.pause_s is not None]
    wakes = [row.wake_ready_s - row.wake_start_s for row in result.sessions
             if row.wake_ready_s is not None]
    arrivals = {
        (event.session_id, int(event.detail)): event.time_s for event in result.events
        if event.event == "request_arrival"
    }
    starts = {(row.session_id, row.request_index): row.start_s for row in result.requests}
    request_waits = [starts[key] - arrival for key, arrival in arrivals.items() if key in starts]
    requests_started = result.requests_started_by(scenario.deadline_s)
    unresumed = sum(
        max(0.0, min(scenario.end_s, row.committed_s or scenario.end_s)
            - max(scenario.deadline_s, row.pause_s or scenario.end_s))
        for row in result.sessions
    )
    return {
        "run_id": run.run_id, "model_profile": run.plan.profile_id,
        "profile_case": run.case_id, "workload_profile": run.workload_id,
        "solver": run.plan.solver, "seed": run.plan.seed,
        "sessions": len(scenario.sessions),
        "source_nodes": sum(node.local for node in scenario.nodes),
        "destination_nodes": sum(not node.local for node in scenario.nodes),
        "power_limit_w": scenario.power_limit_w, "deadline_s": scenario.deadline_s,
        "end_s": scenario.end_s, "controller_delay_s": scenario.controller_delay_s,
        "solve_s": run.plan.solve_s,
        "planned_moves": len(run.plan.moves), "plan_feasible": run.plan.feasible,
        "fractional_variables": run.plan.fractional_variables,
        "planned_source_power_w": run.plan.planned_source_power_w,
        "source_power_at_deadline_w": result.source_power_at_deadline_w,
        "power_met": result.deadline_met, "moves_committed_by_deadline": resumed,
        "requests_started_by_deadline": requests_started,
        "accepted": result.deadline_met and resumed and requests_started,
        "completed_by_end": result.completed_sessions, "makespan_s": result.makespan_s,
        "total_pause_s": sum(pauses),
        "p95_pause_s": float(np.quantile(pauses, 0.95)) if pauses else 0.0,
        "total_wake_s": sum(wakes), "unresumed_after_deadline_s": unresumed,
        "deferred_sessions": sum(row.method == "replay_on_request" for row in result.sessions),
        "request_arrivals": len(arrivals),
        "requests_completed_by_end": sum(row.end_s <= scenario.end_s for row in result.requests),
        "unfinished_requests": len(arrivals) - sum(
            row.end_s <= scenario.end_s for row in result.requests
        ),
        "total_request_wait_s": sum(request_waits),
        "p95_request_wait_s": float(np.quantile(request_waits, 0.95)) if request_waits else 0.0,
        "network_bytes": sum(row.transferred_bytes for row in result.network),
        "network_duration_s": max(
            (row.end_s or scenario.end_s for row in result.network), default=0.0
        ) - min((row.start_s for row in result.network), default=0.0),
        "excess_energy_j": excess_energy(
            result.power, scenario.deadline_s, scenario.end_s, scenario.power_limit_w
        ),
    }


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def write(runs: list[ExperimentRun], out: Path) -> None:
    if not runs:
        raise ValueError("no experiment runs")
    out.mkdir(parents=True, exist_ok=True)
    summaries = [_summary(run) for run in runs]
    events, sessions, requests, network, power, plans = [], [], [], [], [], []
    for run in runs:
        base = {
            "run_id": run.run_id, "model_profile": run.plan.profile_id,
            "workload_profile": run.workload_id, "profile_case": run.case_id,
            "solver": run.plan.solver, "seed": run.plan.seed,
            "power_limit_w": run.scenario.power_limit_w,
            "deadline_s": run.scenario.deadline_s, "end_s": run.scenario.end_s,
            "final_state": run.scenario.final_state,
            "controller_delay_s": run.scenario.controller_delay_s,
        }
        events += [{**base, "time_s": row.time_s, "event": row.event,
                    "session_id": row.session_id, "node_id": row.node_id,
                    "detail": row.detail} for row in run.result.events]
        sessions += [{**base, **row.__dict__} for row in run.result.sessions]
        requests += [{**base, **row.__dict__} for row in run.result.requests]
        network += [{**base, **row.__dict__, "path": "|".join(row.path)}
                    for row in run.result.network]
        power += [{**base, "time_s": t, "source_power_w": source,
                   "destination_power_w": destination}
                  for t, source, destination in run.result.power]
        plans += [{**base, "session_id": row.session_id,
                   "destination_instance": row.destination_instance, "method": row.method,
                   "order": row.order, "path": "|".join(row.path),
                   "external_path": "|".join(row.external_path)} for row in run.plan.moves]
    for name, rows in (("summary", summaries), ("events", events), ("sessions", sessions),
                       ("requests", requests), ("network", network), ("power", power),
                       ("plans", plans)):
        fields = tuple(rows[0]) if rows else ("run_id",)
        _write_csv(out / f"{name}.csv", rows, fields)
    _plot(runs, summaries, out)


def _plot(runs: list[ExperimentRun], summaries: list[dict], out: Path) -> None:
    central = [run for run in runs if run.case_id == "central"]
    first = central[0]
    same = [run for run in central if run.workload_id == first.workload_id
            and run.scenario.deadline_s == first.scenario.deadline_s
            and run.scenario.power_limit_w == first.scenario.power_limit_w]
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in same:
        ax.step([p[0] for p in run.result.power], [p[1] for p in run.result.power], where="post",
                label=run.plan.solver)
    ax.axhline(first.scenario.power_limit_w, color="black", linestyle="--", label="power limit")
    ax.axvline(first.scenario.deadline_s, color="black", linestyle=":", label="deadline")
    ax.set(xlabel="time (s)", ylabel="local source power (W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "power_timeline.png", dpi=160)
    plt.close(fig)

    phase = {solver: [] for solver in SOLVERS}
    for run in same:
        phase[run.plan.solver] += [row.committed_s - row.pause_s for row in run.result.sessions
                                   if row.committed_s is not None and row.pause_s is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [solver for solver, values in phase.items() if values]
    ax.boxplot([phase[label] for label in labels], tick_labels=labels, showfliers=False)
    ax.set(ylabel="session pause (s)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "session_pause.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in same:
        rows = [row for row in run.result.network if row.end_s is not None]
        methods = {row.session_id: row.method for row in run.result.sessions}
        for phase_name in sorted({row.phase for row in rows}):
            selected = [row for row in rows if row.phase == phase_name]
            labels = {methods[row.session_id] for row in selected}
            ax.scatter(
                [row.transferred_bytes / 1e6 for row in selected],
                [row.end_s - row.start_s for row in selected], s=10, alpha=0.4,
                label=f"{run.plan.solver}:{'/'.join(sorted(labels))}:{phase_name}",
            )
    ax.set(xlabel="transfer size (MB)", ylabel="network time (s)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "network_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in same:
        ax.scatter(
            [row.arrival_s for row in run.result.requests],
            [row.start_s - row.arrival_s for row in run.result.requests],
            s=10, alpha=0.5, label=run.plan.solver,
        )
    ax.axvline(first.scenario.deadline_s, color="black", linestyle=":", label="deadline")
    ax.set(xlabel="request arrival (s)", ylabel="request wait (s)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "request_wait.png", dpi=160)
    plt.close(fig)

    central_summaries = [row for row in summaries if row["profile_case"] == "central"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for solver in sorted({row["solver"] for row in central_summaries}):
        rows = [row for row in central_summaries if row["solver"] == solver]
        ax.scatter([row["planned_source_power_w"] for row in rows],
                   [row["source_power_at_deadline_w"] for row in rows], label=solver)
    values = [row[key] for row in central_summaries
              for key in ("planned_source_power_w", "source_power_at_deadline_w")]
    ax.plot([min(values), max(values)], [min(values), max(values)], "k--", label="equal")
    ax.set(xlabel="planned source power (W)", ylabel="simulated source power (W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "planned_vs_simulated_power.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = sorted({row["solver"] for row in central_summaries})
    axes[0].bar(labels, [np.mean([r["accepted"] for r in central_summaries
                                 if r["solver"] == label])
                         for label in labels])
    axes[1].bar(labels, [np.median([r["excess_energy_j"] for r in central_summaries
                                    if r["solver"] == label]) / 1000 for label in labels])
    axes[0].set(ylabel="accepted fraction")
    axes[1].set(ylabel="excess energy after deadline (kJ)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "policy_outcomes.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workload-profile", type=Path, action="append")
    parser.add_argument("--power-limit", type=float, action="append", required=True)
    parser.add_argument("--deadline", type=float, action="append", required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--solver", choices=SOLVERS, action="append")
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-bytes-per-s", type=float, default=125_000_000.0)
    parser.add_argument("--controller-delay", type=float, default=0.0)
    parser.add_argument("--final-state", choices=("awake", "sleep", "off"), default="awake")
    parser.add_argument("--out", type=Path, default=Path("queue-haul/outputs/power_drain"))
    args = parser.parse_args()
    runs = run(
        args.model_profile, args.workload_profile or DEFAULT_WORKLOADS, args.sessions, args.seed,
        args.power_limit, args.deadline, args.end, args.solver or SOLVERS,
        args.link_bytes_per_s, args.final_state, args.controller_delay,
    )
    write(runs, args.out)
    print(f"runs={len(runs)} output={args.out}")


if __name__ == "__main__":
    main()
