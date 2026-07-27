"""Destination requirements without assuming a destination inventory."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class DestinationRequirement:
    target_source_power_reduction_w: float
    achieved_source_power_reduction_w: float
    maximum_achievable_source_power_reduction_w: float
    target_met: bool
    actions: tuple[RequirementAction, ...]
    destination_service_work: tuple[float, float]
    destination_kv_blocks: int
    destination_kv_tokens: int
    replay_migration_slot_s: float
    kv_migration_slot_s: float
    wan_bytes: int
    source_stream_occupancy_s: tuple[tuple[str, float], ...]
    minimum_source_streams: tuple[tuple[str, int], ...]
    source_stream_limit: int
    method_mix: tuple[tuple[str, int], ...]
    makespan_lower_bound_s: float
    migration_horizon_s: float
    route_bandwidth_bytes_per_s: float
    route_rtt_s: float
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
                actions.append(RequirementAction(
                    session.session_id, session.source_instance, method,
                    power.marginal(session.session_id), duration, route_bytes,
                    service, blocks,
                ))
    return tuple(actions)


def _solve(actions: tuple[RequirementAction, ...], target: float, horizon: float,
           streams: int, bandwidth: float):
    if not actions:
        return (), 0.0
    sessions = {name: i for i, name in enumerate(dict.fromkeys(
        action.session_id for action in actions
    ))}
    sources = {name: i for i, name in enumerate(sorted({
        action.source_instance for action in actions
    }))}
    rows, columns, values = [], [], []
    for j, action in enumerate(actions):
        for row, value in (
            (sessions[action.session_id], 1),
            (len(sessions) + sources[action.source_instance], action.duration_s),
            (len(sessions) + len(sources), action.route_bytes),
        ):
            rows.append(row)
            columns.append(j)
            values.append(value)
    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(sessions) + len(sources) + 1, len(actions)),
    ).tocsr()
    upper = np.r_[
        np.ones(len(sessions)),
        np.full(len(sources), streams * horizon),
        bandwidth * horizon,
    ]
    base = LinearConstraint(matrix, -np.inf, upper)
    bounds, integer = Bounds(0, 1), np.ones(len(actions))
    gains = np.array([action.source_power_gain_w for action in actions])
    work = np.array([action.duration_s for action in actions])

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
    return tuple(np.flatnonzero(selected)), maximum_gain


def requirement_frontier(scenario: ExecutionScenario, profile,
                         destination_type: DestinationType,
                         target_source_power_reduction_w: float,
                         route_bandwidth_bytes_per_s: float, route_rtt_s: float,
                         source_streams: int, case_id: str = "central"):
    """Return raw landing requirements; ``route_rtt_s`` is added once per action."""
    if target_source_power_reduction_w < 0 or route_bandwidth_bytes_per_s <= 0 \
            or route_rtt_s < 0 or source_streams < 1:
        raise ValueError("invalid requirement-frontier input")
    start = perf_counter()
    horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    if horizon <= 0:
        raise ValueError("migration horizon must be positive")
    actions = _actions(
        scenario, profile, destination_type, route_bandwidth_bytes_per_s,
        route_rtt_s, horizon, case_id,
    )
    selected, maximum = _solve(
        actions, target_source_power_reduction_w, horizon, source_streams,
        route_bandwidth_bytes_per_s,
    )
    chosen = tuple(actions[i] for i in selected)
    power = ExpectedPower(scenario, profile, case_id)
    achieved = power.drain_gain(action.session_id for action in chosen)
    service = tuple(sum(action.service_work[i] for action in chosen) for i in range(2))
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
        target_source_power_reduction_w, achieved, maximum,
        achieved + 1e-7 >= target_source_power_reduction_w, chosen, service,
        sum(action.kv_blocks for action in chosen),
        sum(action.kv_blocks for action in chosen) * destination_type.kv_block_tokens,
        sum(action.duration_s for action in chosen if action.method == "replay"),
        sum(action.duration_s for action in chosen if action.method == "kv_transfer"),
        wan, occupancies, minimum_streams, source_streams, mix, makespan, horizon,
        route_bandwidth_bytes_per_s, route_rtt_s, perf_counter() - start,
    )


def sweep_frontier(scenario, profile, destination_type, targets, stream_counts,
                   route_bandwidth_bytes_per_s, route_rtt_s, case_id="central"):
    return tuple(
        requirement_frontier(
            scenario, profile, destination_type, target,
            route_bandwidth_bytes_per_s, route_rtt_s, streams, case_id,
        )
        for streams in stream_counts for target in targets
    )
