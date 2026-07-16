"""Profile-driven power-drain planning and event simulation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planner import InstanceCapacity, PlanResult, SOLVERS, plan
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


@dataclass(frozen=True)
class PlotRun:
    workload_id: str
    deadline_s: float
    power_limit_w: float
    solver: str
    power: tuple[tuple[float, float, float], ...]
    pauses: tuple[float, ...]
    network: tuple[tuple[float, float, str], ...]
    requests: tuple[tuple[float, float], ...]


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
    # TODO(tp-topology): construct measured multi-GPU instance and network layouts.
    if profile.tensor_parallel != 1:
        raise ValueError("scenario builder currently supports tensor parallel size 1")
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
    # TODO(prefix-sharing): count identical shared KV blocks once when traces expose them.
    resident = np.array([r.context_tokens if r.state == "active" else 0 for r in records])
    assignment, capacity = np.empty(sessions, int), InstanceCapacity(
        [], [], profile.max_ell, profile.kv_capacity_tokens
    )
    for j in np.argsort(-np.maximum(ell / profile.max_ell,
                                    resident / profile.kv_capacity_tokens), kind="stable"):
        assignment[j] = capacity.place(ell[j], int(resident[j]), grow=True)
    loads = capacity.loads
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
    request_horizon = end_s
    wake_horizon = max(0.0, deadline_s - controller_delay_s)
    # TODO(wake): fit the first-request distribution from complete session traces.
    sampled = tuple(
        SimSession(
            str(j), f"source-{assignment[j]}", r.context_tokens,
            r.prompt_tokens / cycle if r.state == "active" else 0.0,
            r.output_tokens / cycle if r.state == "active" else 0.0,
            r.log_bytes, r.log_external, _requests(r, request_horizon, rng), True,
            0.0 if r.state != "cold" or wake_horizon <= r.tool_delay_s else
            1.0 if r.request_gap_s == 0 else
            1 - math.exp(-(wake_horizon - r.tool_delay_s) / r.request_gap_s),
            state=r.state,
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


def _run_group(profile, workload, sessions, seed, power_limit, deadline, end_s, solver,
               link_bytes_per_s, final_state, controller_delay_s):
    scenario, routes = build_scenario(
        workload, profile, sessions, seed, power_limit, deadline, end_s,
        link_bytes_per_s, final_state, controller_delay_s,
    )
    planned = plan(scenario, profile, routes, solver, "central", seed)
    return tuple(
        ExperimentRun(
            f"{workload.profile_id}:{power_limit:g}:{deadline:g}:{solver}:{case_id}:{seed}",
            workload.profile_id, scenario, planned, case_id,
            execute(scenario, profile, planned.moves, case_id),
        ) for case_id in profile.cases
    )


def run(model_path: Path = DEFAULT_MODEL, workload_paths=DEFAULT_WORKLOADS,
        sessions: int = 10_000, seed: int = 0, power_limits=(10_000.0,),
        deadlines=(120.0,), end_s: float = 180.0, solvers=SOLVERS,
        link_bytes_per_s: float = 125_000_000.0,
        final_state: str = "awake", controller_delay_s: float = 0.0,
        workers: int = 1) -> Iterator[ExperimentRun]:
    if workers < 1:
        raise ValueError("workers must be positive")
    profile = ModelProfile.load(model_path)

    def tasks():
        for workload_path in workload_paths:
            workload = WorkloadProfile.load(workload_path)
            for power_limit in power_limits:
                for deadline in deadlines:
                    for solver in solvers:
                        yield (
                            profile, workload, sessions, seed, power_limit, deadline, end_s,
                            solver, link_bytes_per_s, final_state, controller_delay_s,
                        )

    if workers == 1:
        for task in tasks():
            yield from _run_group(*task)
        return
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task in tasks():
            pending.append(pool.submit(_run_group, *task))
            if len(pending) == workers:
                yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()


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
    nodes = {node.node_id: node for node in scenario.nodes}
    source_instances = {
        instance.instance_id for instance in scenario.instances
        if all(nodes[node].local for node in instance.gpu_nodes)
    }
    resident = {instance: 0 for instance in source_instances}
    for session in scenario.sessions:
        if session.state == "active":
            resident[session.source_instance] += session.context_tokens
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
    queue_waits = [row.start_s - row.arrival_s for row in result.queues
                   if row.start_s is not None]
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
        "source_instances": len(source_instances),
        "source_nodes": sum(node.local for node in scenario.nodes),
        "destination_nodes": sum(not node.local for node in scenario.nodes),
        "kv_capacity_tokens_per_instance": run.plan.kv_capacity_tokens,
        "max_source_resident_kv_tokens": max(resident.values(), default=0),
        "power_limit_w": scenario.power_limit_w, "deadline_s": scenario.deadline_s,
        "end_s": scenario.end_s, "controller_delay_s": scenario.controller_delay_s,
        "solve_s": run.plan.solve_s,
        "planned_moves": len(run.plan.moves), "plan_feasible": run.plan.feasible,
        "initial_source_power_w": run.plan.initial_source_power_w,
        "requested_source_drop_w": max(
            0.0, run.plan.initial_source_power_w - scenario.power_limit_w
        ),
        "planned_source_power_w": run.plan.planned_source_power_w,
        "expected_source_power_at_deadline_w": run.plan.expected_source_power_at_deadline_w,
        "modeled_source_power_at_deadline_w": result.modeled_source_power_at_deadline_w,
        "modeled_source_drop_at_deadline_w": max(
            0.0, run.plan.initial_source_power_w - result.modeled_source_power_at_deadline_w
        ),
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
        "destination_kv_queue_operations": len(result.queues),
        "destination_kv_queue_max_depth": max(
            (row.depth_at_arrival for row in result.queues), default=0
        ),
        "destination_kv_queue_max_bytes": max(
            (row.bytes_at_arrival for row in result.queues), default=0
        ),
        "destination_kv_queue_total_wait_s": sum(queue_waits),
        "destination_kv_queue_p95_wait_s": float(np.quantile(queue_waits, 0.95))
        if queue_waits else 0.0,
        "excess_energy_j": excess_energy(
            result.power, scenario.deadline_s, scenario.end_s, scenario.power_limit_w
        ),
    }


def _sample(rows, limit: int = 2_000):
    rows = tuple(rows)
    if len(rows) <= limit:
        return rows
    return tuple(rows[i] for i in np.linspace(0, len(rows) - 1, limit, dtype=int))


def write(runs: Iterable[ExperimentRun], out: Path) -> int:
    iterator = iter(runs)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("no experiment runs") from None
    out.mkdir(parents=True, exist_ok=True)
    names = "summary", "events", "sessions", "requests", "network", "queues", "power", "plans"
    summaries, plots, writers = [], [], {}
    with ExitStack() as stack:
        files = {name: stack.enter_context((out / f"{name}.csv").open("w", newline=""))
                 for name in names}

        def emit(name: str, rows) -> None:
            for row in rows:
                if name not in writers:
                    writers[name] = csv.DictWriter(files[name], tuple(row))
                    writers[name].writeheader()
                writers[name].writerow(row)

        for run in chain((first,), iterator):
            summary = _summary(run)
            summaries.append(summary)
            base = {
                "run_id": run.run_id, "model_profile": run.plan.profile_id,
                "workload_profile": run.workload_id, "profile_case": run.case_id,
                "solver": run.plan.solver, "seed": run.plan.seed,
                "power_limit_w": run.scenario.power_limit_w,
                "deadline_s": run.scenario.deadline_s, "end_s": run.scenario.end_s,
                "final_state": run.scenario.final_state,
                "controller_delay_s": run.scenario.controller_delay_s,
            }
            emit("summary", (summary,))
            emit("events", ({**base, **row.__dict__} for row in run.result.events))
            emit("sessions", ({**base, **row.__dict__} for row in run.result.sessions))
            emit("requests", ({**base, **row.__dict__} for row in run.result.requests))
            emit("network", ({**base, **row.__dict__, "path": "|".join(row.path)}
                             for row in run.result.network))
            emit("queues", ({**base, **row.__dict__,
                             "wait_s": None if row.start_s is None
                             else row.start_s - row.arrival_s}
                            for row in run.result.queues))
            emit("power", ({**base, "time_s": t, "modeled_source_power_w": source,
                            "modeled_destination_power_w": destination}
                           for t, source, destination in run.result.power))
            emit("plans", ({**base, "session_id": row.session_id,
                            "destination_instance": row.destination_instance,
                            "method": row.method, "order": row.order,
                            "path": "|".join(row.path),
                            "external_path": "|".join(row.external_path)}
                           for row in run.plan.moves))
            if run.case_id == "central":
                methods = {row.session_id: row.method for row in run.result.sessions}
                plots.append(PlotRun(
                    run.workload_id, run.scenario.deadline_s, run.scenario.power_limit_w,
                    run.plan.solver, _sample(run.result.power),
                    _sample(row.committed_s - row.pause_s for row in run.result.sessions
                            if row.committed_s is not None and row.pause_s is not None),
                    _sample((row.transferred_bytes / 1e6, row.end_s - row.start_s,
                             f"{methods[row.session_id]}:{row.phase}")
                            for row in run.result.network if row.end_s is not None),
                    _sample((row.arrival_s, row.start_s - row.arrival_s)
                            for row in run.result.requests),
                ))
        for name in set(names) - set(writers):
            csv.DictWriter(files[name], ("run_id",)).writeheader()
    _plot(plots, summaries, out)
    return len(summaries)


def _plot(runs: list[PlotRun], summaries: list[dict], out: Path) -> None:
    first = runs[0]
    same = [run for run in runs if run.workload_id == first.workload_id
            and run.deadline_s == first.deadline_s
            and run.power_limit_w == first.power_limit_w]
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in same:
        ax.step([p[0] for p in run.power], [p[1] for p in run.power], where="post",
                label=run.solver)
    ax.axhline(first.power_limit_w, color="black", linestyle="--", label="power limit")
    ax.axvline(first.deadline_s, color="black", linestyle=":", label="deadline")
    ax.set(xlabel="time (s)", ylabel="modeled expected source power (W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "power_timeline.png", dpi=160)
    plt.close(fig)

    phase = {solver: [] for solver in SOLVERS}
    for run in same:
        phase[run.solver] += run.pauses
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
        for phase_name in sorted({row[2] for row in run.network}):
            selected = [row for row in run.network if row[2] == phase_name]
            ax.scatter(
                [row[0] for row in selected], [row[1] for row in selected],
                s=10, alpha=0.4, label=f"{run.solver}:{phase_name}",
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
            [row[0] for row in run.requests], [row[1] for row in run.requests],
            s=10, alpha=0.5, label=run.solver,
        )
    ax.axvline(first.deadline_s, color="black", linestyle=":", label="deadline")
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
        ax.scatter([row["expected_source_power_at_deadline_w"] for row in rows],
                   [row["modeled_source_power_at_deadline_w"] for row in rows], label=solver)
    values = [row[key] for row in central_summaries
              for key in ("expected_source_power_at_deadline_w",
                          "modeled_source_power_at_deadline_w")]
    ax.plot([min(values), max(values)], [min(values), max(values)], "k--", label="equal")
    ax.set(xlabel="expected deadline-window power (W)",
           ylabel="modeled deadline-window power (W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "expected_vs_modeled_power.png", dpi=160)
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--final-state", choices=("awake", "sleep", "off"), default="awake")
    parser.add_argument("--out", type=Path, default=Path("queue-haul/outputs/power_drain"))
    args = parser.parse_args()
    runs = run(
        args.model_profile, args.workload_profile or DEFAULT_WORKLOADS, args.sessions, args.seed,
        args.power_limit, args.deadline, args.end, args.solver or SOLVERS,
        args.link_bytes_per_s, args.final_state, args.controller_delay, args.workers,
    )
    count = write(runs, args.out)
    print(f"runs={count} output={args.out}")


if __name__ == "__main__":
    main()
