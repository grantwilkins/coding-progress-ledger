"""Destination requirements without assuming a destination inventory."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from destination import DestinationType
from planner import _local_sessions
from pool_planner import _destination_duration
from power_model import ExpectedPower
from simulate import ExecutionScenario


@dataclass(frozen=True)
class RequirementAction:
    session_id: str
    source_instance: str
    method: str
    source_power_gain_w: float
    duration_s: float
    route_bytes: int
    service_work: tuple[float, float]
    kv_blocks: int
    transition_work: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class DestinationRequirement:
    target_source_power_reduction_w: float
    achieved_source_power_reduction_w: float
    selected_modeled_source_power_gain_w: float
    maximum_modeled_source_power_gain_w: float | None
    target_met: bool
    actions: tuple[RequirementAction, ...]
    destination_service_work: tuple[float, float]
    destination_transition_work: tuple[float, float]
    destination_kv_blocks: int
    destination_kv_tokens: int
    replay_migration_slot_s: float
    kv_migration_slot_s: float
    wan_bytes: int
    source_stream_occupancy_s: tuple[tuple[str, float], ...]
    minimum_source_streams_lower_bound: tuple[tuple[str, int], ...]
    source_stream_limit: int
    method_mix: tuple[tuple[str, int], ...]
    makespan_lower_bound_s: float
    migration_horizon_s: float
    route_bandwidth_bytes_per_s: float
    route_rtt_s: float
    solver_mode: str
    solver_status: str
    solver_mip_gap: float | None
    solve_s: float


def _actions(scenario: ExecutionScenario, profile, destination_type: DestinationType,
             bandwidth: float, rtt: float, horizon: float, case_id: str):
    if destination_type.compatibility.model != profile.model:
        raise ValueError("source model does not match destination type")
    if destination_type.migration is None:
        raise ValueError("destination type lacks physical migration components")
    case, power = profile.case(case_id), ExpectedPower(scenario, profile, case_id)
    actions = []
    for session in _local_sessions(scenario):
        if session.state != "active":
            continue
        tokens = math.ceil(
            session.context_tokens + session.expected_growth_tokens_per_s * horizon
        )
        service = tuple(destination_type.work(
            session.expected_f, session.expected_g, tokens, True,
        ))
        blocks = math.ceil(tokens / destination_type.kv_block_tokens)
        for method in ("replay", "kv_transfer"):
            duration = _destination_duration(
                session, method, case, ("wan",), {"wan": bandwidth}, horizon,
                destination_type.migration[method],
            ) + rtt
            if duration <= horizon:
                route_bytes = (
                    math.ceil(session.log_bytes * tokens / session.context_tokens)
                    if method == "replay" else case.kv_transfer.sealed_bytes(tokens)
                )
                transition = (
                    (max(0, duration - route_bytes / bandwidth - rtt), 0)
                    if method == "replay" else (0, 0)
                )
                actions.append(RequirementAction(
                    session.session_id, session.source_instance, method,
                    power.marginal(session.session_id), duration, route_bytes,
                    service, blocks, transition,
                ))
    return tuple(actions)


def _solve(actions: tuple[RequirementAction, ...], target: float, horizon: float,
           streams: int, bandwidth: float):
    if not actions:
        return (), 0.0
    sessions = {name: i for i, name in enumerate(dict.fromkeys(
        action.session_id for action in actions
    ))}
    sources = tuple(sorted({action.source_instance for action in actions}))
    source_rows = {
        (source, stream): len(sessions) + i * streams + stream
        for i, source in enumerate(sources) for stream in range(streams)
    }
    n = len(actions) * streams
    rows, columns, values = [], [], []
    for j, action in enumerate(actions):
        for stream in range(streams):
            column = j * streams + stream
            for row, value in (
                (sessions[action.session_id], 1),
                (source_rows[action.source_instance, stream], action.duration_s),
                (len(sessions) + len(sources) * streams, action.route_bytes),
            ):
                rows.append(row)
                columns.append(column)
                values.append(value)
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(sessions) + len(sources) * streams + 1, n),
    ).tocsr()
    upper = np.r_[
        np.ones(len(sessions)),
        np.full(len(sources) * streams, horizon),
        bandwidth * horizon,
    ]
    base = LinearConstraint(matrix, -np.inf, upper)
    bounds, integer = Bounds(0, 1), np.ones(n)
    gains = np.repeat([action.source_power_gain_w for action in actions], streams)
    work = np.repeat([action.duration_s for action in actions], streams)

    def optimize(cost, constraints):
        result = milp(
            cost, integrality=integer, bounds=bounds, constraints=constraints,
            options={"mip_rel_gap": 0},
        )
        if not result.success:
            raise RuntimeError(f"requirement MILP returned {result.message}")
        return result.x > .5

    maximum = optimize(-gains, base)
    maximum_gain = float(gains @ maximum)
    required = min(target, maximum_gain)
    gain_row = coo_matrix(gains.reshape(1, -1))
    selected = optimize(
        work,
        (base, LinearConstraint(gain_row, required - 1e-7, np.inf)),
    )
    return tuple(np.flatnonzero(selected) // streams), maximum_gain


def _greedy(actions: tuple[RequirementAction, ...], target: float, horizon: float,
            streams: int, bandwidth: float):
    bins = {
        source: [0.0] * streams
        for source in {action.source_instance for action in actions}
    }
    selected, sessions, wan, gain = [], set(), 0, 0.0

    def cost(i):
        action = actions[i]
        return action.duration_s / (streams * horizon) \
            + action.route_bytes / (bandwidth * horizon)

    def placement(i):
        action = actions[i]
        choices = [
            (used, j) for j, used in enumerate(bins[action.source_instance])
            if used + action.duration_s <= horizon + 1e-7
        ]
        return max(choices)[1] if choices else None

    small, large, promoted = [], [], set()
    by_gain = sorted(range(len(actions)), key=lambda i: (
        -actions[i].source_power_gain_w, i,
    ))
    for i, action in enumerate(actions):
        heapq.heappush(small, (
            cost(i) / max(action.source_power_gain_w, 1e-12), cost(i), i,
        ))
    cursor = 0

    def promote(remaining):
        nonlocal cursor
        while cursor < len(by_gain) \
                and actions[by_gain[cursor]].source_power_gain_w >= remaining:
            i = by_gain[cursor]
            promoted.add(i)
            heapq.heappush(large, (cost(i), i))
            cursor += 1

    def feasible(i):
        action = actions[i]
        return action.session_id not in sessions \
            and wan + action.route_bytes <= bandwidth * horizon + 1e-7 \
            and placement(i) is not None

    while gain + 1e-7 < target:
        remaining = target - gain
        promote(remaining)
        while small and (small[0][2] in promoted or not feasible(small[0][2])):
            heapq.heappop(small)
        while large and not feasible(large[0][1]):
            heapq.heappop(large)
        options = []
        if small:
            options.append((small[0][0], small[0][1], small[0][2], "small"))
        if large:
            options.append((large[0][0] / remaining, large[0][0],
                            large[0][1], "large"))
        if not options:
            break
        _, _, i, heap = min(options)
        heapq.heappop(small if heap == "small" else large)
        action, stream = actions[i], placement(i)
        bins[action.source_instance][stream] += action.duration_s
        selected.append(i)
        sessions.add(action.session_id)
        wan += action.route_bytes
        gain += action.source_power_gain_w
    _validate_greedy(actions, selected, horizon, streams, bandwidth)
    return tuple(selected)


def _validate_greedy(actions, selected, horizon, streams, bandwidth):
    bins, sessions, wan = {}, set(), 0
    for i in selected:
        action = actions[i]
        if action.session_id in sessions:
            raise RuntimeError("greedy selected two actions for one session")
        loads = bins.setdefault(action.source_instance, [0.0] * streams)
        choices = [(used, j) for j, used in enumerate(loads)
                   if used + action.duration_s <= horizon + 1e-7]
        if not choices:
            raise RuntimeError("greedy source-stream schedule exceeds deadline")
        loads[max(choices)[1]] += action.duration_s
        sessions.add(action.session_id)
        wan += action.route_bytes
    if wan > bandwidth * horizon + 1e-7:
        raise RuntimeError("greedy WAN budget exceeds deadline")


BASELINES = (
    "all_replay", "all_kv", "isolated_fastest", "network_greedy",
    "service_greedy", "power_first",
)


def _baseline(actions, target, horizon, streams, bandwidth, policy):
    if policy not in BASELINES:
        raise ValueError("unknown frontier baseline")
    eligible = list(range(len(actions)))
    if policy in {"all_replay", "all_kv"}:
        method = "replay" if policy == "all_replay" else "kv_transfer"
        eligible = [i for i in eligible if actions[i].method == method]
    if policy == "isolated_fastest":
        eligible = list({
            action.session_id: min(
                (i for i in eligible if actions[i].session_id == action.session_id),
                key=lambda i: (actions[i].duration_s, i),
            )
            for action in actions
        }.values())
    keys = {
        "network_greedy": lambda i: (
            actions[i].route_bytes / max(actions[i].source_power_gain_w, 1e-12), i,
        ),
        "service_greedy": lambda i: (
            sum(actions[i].service_work)
            / max(actions[i].source_power_gain_w, 1e-12), i,
        ),
        "power_first": lambda i: (-actions[i].source_power_gain_w, i),
    }
    key = keys.get(policy, lambda i: (
        actions[i].duration_s / max(actions[i].source_power_gain_w, 1e-12), i,
    ))
    bins = {
        source: [0.0] * streams
        for source in {action.source_instance for action in actions}
    }
    selected, sessions, wan, gain = [], set(), 0, 0.0
    for i in sorted(eligible, key=key):
        action = actions[i]
        choices = [
            (used, stream) for stream, used in enumerate(bins[action.source_instance])
            if used + action.duration_s <= horizon + 1e-7
        ]
        if action.session_id in sessions or not choices \
                or wan + action.route_bytes > bandwidth * horizon + 1e-7:
            continue
        bins[action.source_instance][max(choices)[1]] += action.duration_s
        selected.append(i)
        sessions.add(action.session_id)
        wan += action.route_bytes
        gain += action.source_power_gain_w
        if gain >= target - 1e-7:
            break
    _validate_greedy(actions, selected, horizon, streams, bandwidth)
    return tuple(selected)


def requirement_frontier(scenario: ExecutionScenario, profile,
                         destination_type: DestinationType,
                         target_source_power_reduction_w: float,
                         route_bandwidth_bytes_per_s: float, route_rtt_s: float,
                         source_streams: int, case_id: str = "central",
                         solver_mode: str = "exact"):
    """Return raw landing requirements; ``route_rtt_s`` is added once per action."""
    if target_source_power_reduction_w < 0 or route_bandwidth_bytes_per_s <= 0 \
            or route_rtt_s < 0 or source_streams < 1 \
            or solver_mode not in {"exact", "greedy", *BASELINES}:
        raise ValueError("invalid requirement-frontier input")
    start = perf_counter()
    horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    if horizon <= 0:
        raise ValueError("migration horizon must be positive")
    actions = _actions(
        scenario, profile, destination_type, route_bandwidth_bytes_per_s,
        route_rtt_s, horizon, case_id,
    )
    if solver_mode == "exact":
        selected, maximum = _solve(
            actions, target_source_power_reduction_w, horizon, source_streams,
            route_bandwidth_bytes_per_s,
        )
    elif solver_mode == "greedy":
        selected = _greedy(
            actions, target_source_power_reduction_w, horizon, source_streams,
            route_bandwidth_bytes_per_s,
        )
        maximum = None
    else:
        selected = _baseline(
            actions, target_source_power_reduction_w, horizon, source_streams,
            route_bandwidth_bytes_per_s, solver_mode,
        )
        maximum = None
    chosen = tuple(actions[i] for i in selected)
    power = ExpectedPower(scenario, profile, case_id)
    achieved = power.drain_gain(action.session_id for action in chosen)
    modeled = sum(action.source_power_gain_w for action in chosen)
    service = tuple(sum(action.service_work[i] for action in chosen) for i in range(2))
    transition = tuple(
        sum(action.transition_work[i] for action in chosen) for i in range(2)
    )
    occupancies = tuple(
        (source, sum(action.duration_s for action in chosen
                     if action.source_instance == source))
        for source in sorted({action.source_instance for action in chosen})
    )
    minimum_streams = tuple(
        (source, math.ceil(seconds / horizon - 1e-12))
        for source, seconds in occupancies
    )
    wan = sum(action.route_bytes for action in chosen)
    makespan = max(
        [0.0, wan / route_bandwidth_bytes_per_s]
        + [action.duration_s for action in chosen]
        + [seconds / source_streams for _, seconds in occupancies]
    )
    mix = tuple(
        (method, sum(action.method == method for action in chosen))
        for method in ("replay", "kv_transfer")
    )
    return DestinationRequirement(
        target_source_power_reduction_w, achieved, modeled, maximum,
        achieved + 1e-7 >= target_source_power_reduction_w, chosen, service, transition,
        sum(action.kv_blocks for action in chosen),
        sum(action.kv_blocks for action in chosen) * destination_type.kv_block_tokens,
        sum(action.duration_s for action in chosen if action.method == "replay"),
        sum(action.duration_s for action in chosen if action.method == "kv_transfer"),
        wan, occupancies, minimum_streams, source_streams, mix, makespan, horizon,
        route_bandwidth_bytes_per_s, route_rtt_s, solver_mode,
        ("optimal" if solver_mode == "exact" else
         "approximate" if solver_mode == "greedy" else "baseline")
        + ("_target_met" if achieved + 1e-7 >= target_source_power_reduction_w
           else "_best_effort"),
        0.0 if solver_mode == "exact" else None, perf_counter() - start,
    )


def sweep_frontier(scenario, profile, destination_type, targets, stream_counts,
                   route_bandwidth_bytes_per_s, route_rtt_s, case_id="central",
                   solver_mode="exact"):
    return tuple(
        requirement_frontier(
            scenario, profile, destination_type, target,
            route_bandwidth_bytes_per_s, route_rtt_s, streams, case_id, solver_mode,
        )
        for streams in stream_counts for target in targets
    )
