"""Whole-session planning against the measured source power curve."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from time import perf_counter
from typing import Callable

import cvxpy as cp
import numpy as np

from profiles import ModelProfile, ProfileCase
from power_model import ExpectedPower
from simulate import ExecutionScenario, MoveMethod, PlannedMove, SimSession


METHODS: tuple[MoveMethod, ...] = ("replay", "kv_transfer", "replay_on_request")
SOLVERS = ("random", "load_only", "node_aware", "node_drain", "rounded_lp")
Routes = dict[tuple[str, str], tuple[str, ...]] | Callable[[str, str], tuple[str, ...]]


@dataclass(frozen=True)
class PlanResult:
    solver: str
    moves: tuple[PlannedMove, ...]
    initial_source_power_w: float
    planned_source_power_w: float
    feasible: bool
    solve_s: float
    fractional_variables: int
    profile_id: str
    profile_case: str
    seed: int


def _ell(session: SimSession, case: ProfileCase) -> float:
    return session.expected_f / case.F + session.expected_g / case.G


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
    replay_s = session.context_tokens / case.replay.rate(session.context_tokens, 1)
    if method == "replay":
        return link_s(session.log_bytes) + replay_s + case.switch_s
    if method == "kv_transfer":
        blocks = case.kv_transfer.blocks(session.context_tokens)
        return (case.kv_transfer.setup_s + link_s(blocks * case.kv_transfer.block_bytes)
                + blocks * case.kv_transfer.block_processing_s + case.kv_transfer.sync_s
                + case.switch_s)
    initial_s = 0.0 if session.log_external else link_s(session.log_bytes)
    wake_s = (link_s(session.log_bytes) if session.log_external else 0.0) + replay_s
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
    remaining = set(range(len(sessions)))
    groups = []
    while remaining:
        group, nodes, changed = set(), set(), True
        group.add(remaining.pop())
        while changed:
            nodes |= set().union(*(instances[sessions[j].source_instance] for j in group))
            add = {j for j in remaining if instances[sessions[j].source_instance] & nodes}
            changed = bool(add)
            group |= add
            remaining -= add
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
    for session in scenario.sessions:
        if session.source_instance in load:
            load[session.source_instance] += _ell(session, case)
    size = len(destinations[0].gpu_nodes)
    heap = [(value / size, key) for key, value in load.items()]
    heapq.heapify(heap)
    moves = []
    for order, j in enumerate(selected):
        session, ell = sessions[j], _ell(sessions[j], case)
        current, destination = heapq.heappop(heap)
        if current + ell / size > profile.max_ell:
            raise ValueError(f"no destination capacity or path for session {session.session_id!r}")
        path = _route(routes, session.source_instance, destination)
        load[destination] += ell
        heapq.heappush(heap, (load[destination] / size, destination))
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
        return PlanResult(solver, (), initial, initial, True, perf_counter() - start, 0,
                          profile.profile_id, case_id, seed)
    horizon = scenario.deadline_s - scenario.solver_s
    valid = np.zeros((len(sessions), len(METHODS)), bool)
    costs = np.full(valid.shape, np.inf)
    for j, session in enumerate(sessions):
        # TODO(routes): measure heterogeneous destination links before optimizing over them.
        candidate_path = _route(paths, session.source_instance, destinations[0].instance_id)
        for k, method in enumerate(METHODS):
            costs[j, k] = _duration(session, method, case, candidate_path, links)
            valid[j, k] = costs[j, k] <= horizon
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
        gains[group] += bonus * weight / weight.sum()
    rng, fractional = np.random.default_rng(seed), np.zeros_like(costs)
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
        order = [j for group in groups for j in sorted(group, key=lambda j: best_cost[j])]
        methods = [METHODS[k] for k in best_method]
    else:
        x = cp.Variable(costs.shape, nonneg=True)
        objective_cost = np.where(valid, costs, 0.0)
        problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(objective_cost, x))), [
            x <= valid, cp.sum(x, axis=1) <= 1,
            gains @ cp.sum(x, axis=1) >= initial - scenario.power_limit_w,
        ])
        problem.solve(solver=cp.SCIPY)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            fractional[:, :] = valid * np.eye(len(METHODS))[best_method]
        else:
            fractional = np.asarray(x.value)
        mass = fractional.sum(1)
        order = list(np.lexsort((best_cost / np.maximum(gains, 1e-12), -mass)))
        methods = [METHODS[int(np.argmax(fractional[j]))] if mass[j] > 1e-9
                   else METHODS[best_method[j]] for j in range(len(sessions))]
    selected = []
    if solver == "node_drain":
        ordered_groups = [[j for j in order if j in group] for group in groups]
        for group in ordered_groups:
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
    return PlanResult(
        solver, moves, initial, planned, planned <= scenario.power_limit_w,
        perf_counter() - start, int(((fractional > 1e-9) & (fractional < 1 - 1e-9)).sum()),
        profile.profile_id, case_id, seed,
    )
