"""Whole-session planning against the measured source power curve."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
from time import perf_counter
from typing import Callable

import numpy as np

from profiles import ModelProfile, ProfileCase
from power_model import ExpectedPower
from simulate import (MOVE_METHODS_BY_STATE, ExecutionScenario, MoveMethod, PlannedMove,
                      SimSession, predict)


METHODS: tuple[MoveMethod, ...] = ("replay", "kv_transfer", "replay_on_request")
SOLVERS = ("random", "load_only", "node_aware", "node_drain")
Routes = dict[tuple[str, str], tuple[str, ...]] | Callable[[str, str], tuple[str, ...]]


class InstanceCapacity:
    def __init__(self, loads: list[float], tokens: list[int], max_load: float,
                 max_tokens: int):
        self.loads, self.tokens = loads, tokens
        self.max_load, self.max_tokens = max_load, max_tokens
        self.heap = [(self._score(i), i) for i in range(len(loads))]
        heapq.heapify(self.heap)

    def _score(self, i: int) -> float:
        return max(self.loads[i] / self.max_load, self.tokens[i] / self.max_tokens)

    def place(self, load: float, tokens: int, grow: bool = False) -> int:
        if grow:
            if self.heap:
                _, i = heapq.heappop(self.heap)
                if self.loads[i] + load > self.max_load or self.tokens[i] + tokens > self.max_tokens:
                    heapq.heappush(self.heap, (self._score(i), i))
                    i = len(self.loads)
                    self.loads.append(0.0)
                    self.tokens.append(0)
            else:
                i = 0
                self.loads.append(0.0)
                self.tokens.append(0)
            self.loads[i] += load
            self.tokens[i] += tokens
            heapq.heappush(self.heap, (self._score(i), i))
            return i
        held = []
        while self.heap:
            _, i = heapq.heappop(self.heap)
            if self.loads[i] + load <= self.max_load and self.tokens[i] + tokens <= self.max_tokens:
                break
            held.append((self._score(i), i))
        else:
            raise ValueError("no destination compute or KV capacity")
        self.loads[i] += load
        self.tokens[i] += tokens
        heapq.heappush(self.heap, (self._score(i), i))
        for item in held:
            heapq.heappush(self.heap, item)
        return i


@dataclass(frozen=True)
class PlanResult:
    solver: str
    moves: tuple[PlannedMove, ...]
    initial_source_power_w: float
    planned_source_power_w: float
    expected_source_power_at_deadline_w: float
    feasible: bool
    solve_s: float
    profile_id: str
    profile_case: str
    seed: int
    kv_capacity_tokens: int


def _ell(session: SimSession, case: ProfileCase) -> float:
    return session.expected_f / case.F + session.expected_g / case.G


def _resident_tokens(session: SimSession) -> int:
    return session.context_tokens if session.state == "active" else 0


def source_power(scenario: ExecutionScenario, profile: ModelProfile, moved=(),
                 case_id: str = "central") -> float:
    """Expected local power after committed moves and the requested final node state."""
    state, moved = ExpectedPower(scenario, profile, case_id), set(moved)
    for session in scenario.sessions:
        if session.session_id in moved:
            state.remove(session.session_id)
    return state.power(True)


def _duration(session: SimSession, method: MoveMethod, case: ProfileCase,
              path: tuple[str, ...], links: dict[str, float]) -> float:
    def link_s(size):
        return size / min(links[link] for link in path)
    def external_link_s(size):
        # TODO(external-path): replace the destination-ingress assumption with measured topology.
        return size / links[path[-1]]
    replay_s = session.context_tokens / case.replay.rate(session.context_tokens, 1)
    if method == "replay":
        transfer_s = external_link_s(session.log_bytes) if session.log_external \
            else link_s(session.log_bytes)
        return transfer_s + replay_s + case.switch_s
    if method == "kv_transfer":
        blocks = case.kv_transfer.blocks(session.context_tokens)
        return (case.kv_transfer.setup_s + link_s(blocks * case.kv_transfer.block_bytes)
                + blocks * case.kv_transfer.block_processing_s + case.kv_transfer.sync_s
                + case.switch_s)
    initial_s = 0.0 if session.log_external else link_s(session.log_bytes)
    wake_s = (external_link_s(session.log_bytes) if session.log_external else 0.0) + replay_s
    return initial_s + case.switch_s + session.wake_probability * wake_s


def _local_sessions(scenario: ExecutionScenario) -> list[SimSession]:
    nodes = {n.node_id: n for n in scenario.nodes}
    instances = {i.instance_id: i for i in scenario.instances}
    out = []
    for session in scenario.sessions:
        local = {nodes[n].local for n in instances[session.source_instance].gpu_nodes}
        if len(local) != 1:
            raise ValueError("a serving instance cannot span local and remote power scopes")
        if local == {True} and session.movable:
            out.append(session)
    return out


def _drain_groups(scenario: ExecutionScenario, sessions: list[SimSession]) -> list[list[int]]:
    instances = {i.instance_id: set(i.gpu_nodes) for i in scenario.instances}
    by_node: dict[str, set[int]] = {}
    for j, session in enumerate(sessions):
        for node in instances[session.source_instance]:
            by_node.setdefault(node, set()).add(j)
    remaining = set(range(len(sessions)))
    groups = []
    while remaining:
        group, pending = set(), [remaining.pop()]
        while pending:
            j = pending.pop()
            group.add(j)
            linked = set().union(*(by_node[node] for node in instances[sessions[j].source_instance]))
            pending.extend(linked & remaining)
            remaining -= linked
        groups.append(sorted(group))
    return groups


def _route(routes: Routes, source: str, destination: str) -> tuple[str, ...]:
    return routes(source, destination) if callable(routes) else routes[(source, destination)]


def _place(selected: list[int], sessions: list[SimSession], scenario: ExecutionScenario,
           profile: ModelProfile, routes: Routes,
           methods: list[MoveMethod], case_id: str) -> tuple[PlannedMove, ...]:
    case = profile.case(case_id)
    nodes = {n.node_id: n for n in scenario.nodes}
    destinations = [
        i for i in scenario.instances if all(not nodes[n].local for n in i.gpu_nodes)
    ]
    if not destinations or len({len(i.gpu_nodes) for i in destinations}) != 1:
        raise ValueError("destination instances must exist and use one tensor-parallel size")
    load = {i.instance_id: 0.0 for i in destinations}
    tokens = {i.instance_id: 0 for i in destinations}
    for session in scenario.sessions:
        if session.source_instance in load:
            load[session.source_instance] += _ell(session, case)
            tokens[session.source_instance] += _resident_tokens(session)
    size = len(destinations[0].gpu_nodes)
    capacity = InstanceCapacity(
        [load[i.instance_id] for i in destinations],
        [tokens[i.instance_id] for i in destinations],
        profile.max_ell * size, profile.kv_capacity_tokens,
    )
    moves = []
    for order, j in enumerate(selected):
        session, ell = sessions[j], _ell(sessions[j], case)
        destination = destinations[capacity.place(ell, _resident_tokens(session))].instance_id
        path = _route(routes, session.source_instance, destination)
        # TODO(external-path): replace the destination-ingress assumption with measured topology.
        moves.append(PlannedMove(
            session.session_id, destination, methods[j], order, path, path[-1:]
        ))
    return tuple(moves)


def plan(scenario: ExecutionScenario, profile: ModelProfile,
         paths: Routes, solver: str,
         case_id: str = "central", seed: int = 0) -> PlanResult:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    start, case = perf_counter(), profile.case(case_id)
    sessions = _local_sessions(scenario)
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    nodes = {n.node_id: n for n in scenario.nodes}
    destinations = [i for i in scenario.instances if all(not nodes[n].local for n in i.gpu_nodes)]
    if not destinations:
        raise ValueError("scenario has no destination instance")
    power_state = ExpectedPower(scenario, profile, case_id)
    initial = power_state.power(True)
    if initial <= scenario.power_limit_w:
        return PlanResult(solver, (), initial, initial, initial, True, perf_counter() - start,
                          profile.profile_id, case_id, seed, profile.kv_capacity_tokens)
    horizon = scenario.deadline_s - scenario.controller_delay_s
    valid = np.zeros((len(sessions), len(METHODS)), bool)
    costs = np.full(valid.shape, np.inf)
    for j, session in enumerate(sessions):
        # TODO(routes): measure heterogeneous destination links before optimizing over them.
        candidate_path = _route(paths, session.source_instance, destinations[0].instance_id)
        for k, method in enumerate(METHODS):
            costs[j, k] = _duration(session, method, case, candidate_path, links)
            valid[j, k] = method in MOVE_METHODS_BY_STATE[session.state] \
                and costs[j, k] <= horizon
    available = valid.any(1)
    best_method = np.argmin(np.where(valid, costs, np.inf), axis=1)
    best_cost = costs[np.arange(len(sessions)), best_method]
    groups = _drain_groups(scenario, sessions)
    base_gain = np.array([power_state.marginal(s.session_id) for s in sessions])
    gains = base_gain.copy()
    for group in groups:
        ids = [sessions[j].session_id for j in group]
        bonus = power_state.drain_gain(ids) - base_gain[group].sum()
        weight = np.array([_ell(sessions[j], case) for j in group])
        gains[group] += bonus * (weight / weight.sum() if weight.any() else 1 / len(group))
    rng = np.random.default_rng(seed)
    if solver == "random":
        order = list(rng.permutation(np.flatnonzero(available)))
        methods = [METHODS[int(rng.choice(np.flatnonzero(valid[j])))] if available[j]
                   else METHODS[0] for j in range(len(sessions))]
    elif solver == "load_only":
        order = [j for j in np.argsort(
            best_cost / np.maximum([_ell(s, case) for s in sessions], 1e-12)
        ) if available[j]]
        methods = [METHODS[k] for k in best_method]
    elif solver == "node_aware":
        order = [j for j in np.argsort(best_cost / np.maximum(gains, 1e-12)) if available[j]]
        methods = [METHODS[k] for k in best_method]
    elif solver == "node_drain":
        groups.sort(key=lambda group: best_cost[group].sum() / max(gains[group].sum(), 1e-12))
        groups = [sorted(group, key=lambda j: best_cost[j]) for group in groups]
        methods = [METHODS[k] for k in best_method]
    selected = []
    if solver == "node_drain":
        for group in groups:
            if power_state.power(True) <= scenario.power_limit_w:
                break
            for j in group:
                if valid[j, METHODS.index(methods[j])]:
                    selected.append(j)
                    power_state.remove(sessions[j].session_id)
    else:
        for j in order:
            if power_state.power(True) <= scenario.power_limit_w:
                break
            if valid[j, METHODS.index(methods[j])]:
                selected.append(j)
                power_state.remove(sessions[j].session_id)
    planned = power_state.power(True)
    moves = _place(selected, sessions, scenario, profile, paths, methods, case_id)
    expected = predict(
        replace(scenario, sessions=tuple(replace(session, requests=())
                                         for session in scenario.sessions)),
        profile, moves, case_id,
    )
    feasible = planned <= scenario.power_limit_w and expected.deadline_met and all(
        row.committed_s is not None and row.committed_s <= scenario.deadline_s
        for row in expected.sessions
    )
    return PlanResult(
        solver, moves, initial, planned, expected.modeled_source_power_at_deadline_w, feasible,
        perf_counter() - start, profile.profile_id, case_id, seed,
        profile.kv_capacity_tokens,
    )
