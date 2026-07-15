"""Whole-session planning against the measured source power curve."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np

from profiles import ModelProfile, ProfileCase
from simulate import ExecutionScenario, MoveMethod, PlannedMove, SimSession


METHODS: tuple[MoveMethod, ...] = ("replay", "kv_transfer", "replay_on_request")
SOLVERS = ("random", "load_only", "node_aware", "node_drain", "rounded_lp")


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
    case, moved = profile.case(case_id), set(moved)
    nodes = {n.node_id: n for n in scenario.nodes}
    instances = {i.instance_id: i for i in scenario.instances}
    loads = {i.instance_id: 0.0 for i in scenario.instances}
    for session in scenario.sessions:
        if session.session_id not in moved:
            loads[session.source_instance] += _ell(session, case)
    slots = {n.node_id: [0.0] * n.gpus for n in scenario.nodes}
    used = {n.node_id: 0 for n in scenario.nodes}
    for instance in scenario.instances:
        share = loads[instance.instance_id] / len(instance.gpu_nodes)
        for node_id in instance.gpu_nodes:
            if used[node_id] == len(slots[node_id]):
                raise ValueError(f"serving instances exceed GPU capacity on {node_id!r}")
            slots[node_id][used[node_id]] += share
            used[node_id] += 1
    dependents = {
        node_id: {
            s.session_id for s in scenario.sessions
            if node_id in instances[s.source_instance].gpu_nodes
        } for node_id in nodes
    }
    total = 0.0
    for node_id, node in nodes.items():
        if not node.local:
            continue
        drained = bool(dependents[node_id]) and dependents[node_id] <= moved
        if drained and scenario.final_state == "off":
            continue
        if drained and scenario.final_state == "sleep":
            total += case.sleep_power_w * (node.gpus if profile.power_scope == "gpu" else 1)
        elif profile.power_scope == "gpu":
            total += sum(case.power_curve.power(load) for load in slots[node_id])
        else:
            total += case.power_curve.power(sum(slots[node_id]))
    return total


def _duration(session: SimSession, method: MoveMethod, case: ProfileCase,
              path: tuple[str, ...], links: dict[str, float]) -> float:
    link_s = lambda size: size / min(links[link] for link in path)
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


def _place(selected: list[int], sessions: list[SimSession], scenario: ExecutionScenario,
           profile: ModelProfile, paths: dict[tuple[str, str], tuple[str, ...]],
           methods: list[MoveMethod], case_id: str) -> tuple[PlannedMove, ...]:
    case = profile.case(case_id)
    nodes = {n.node_id: n for n in scenario.nodes}
    destinations = [
        i for i in scenario.instances if all(not nodes[n].local for n in i.gpu_nodes)
    ]
    load = {i.instance_id: 0.0 for i in destinations}
    for session in scenario.sessions:
        if session.source_instance in load:
            load[session.source_instance] += _ell(session, case)
    moves = []
    for order, j in enumerate(selected):
        session, ell = sessions[j], _ell(sessions[j], case)
        choices = [
            i for i in destinations
            if (session.source_instance, i.instance_id) in paths
            and load[i.instance_id] / len(i.gpu_nodes) + ell / len(i.gpu_nodes) <= profile.max_ell
        ]
        if not choices:
            raise ValueError(f"no destination capacity or path for session {session.session_id!r}")
        dest = min(choices, key=lambda i: (load[i.instance_id] / len(i.gpu_nodes), i.instance_id))
        load[dest.instance_id] += ell
        path = paths[(session.source_instance, dest.instance_id)]
        moves.append(PlannedMove(session.session_id, dest.instance_id, methods[j], order, path))
    return tuple(moves)


def plan(scenario: ExecutionScenario, profile: ModelProfile,
         paths: dict[tuple[str, str], tuple[str, ...]], solver: str,
         case_id: str = "central", seed: int = 0) -> PlanResult:
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    start, case = perf_counter(), profile.case(case_id)
    sessions, links = _local_sessions(scenario), {l.link_id: l.bytes_per_s for l in scenario.links}
    initial = source_power(scenario, profile, case_id=case_id)
    if initial <= scenario.power_limit_w:
        return PlanResult(solver, (), initial, initial, True, perf_counter() - start, 0,
                          profile.profile_id, case_id, seed)
    horizon = scenario.deadline_s - scenario.solver_s
    valid = np.zeros((len(sessions), len(METHODS)), bool)
    costs = np.full(valid.shape, np.inf)
    for j, session in enumerate(sessions):
        candidate_paths = [path for (source, _dest), path in paths.items()
                           if source == session.source_instance]
        if not candidate_paths:
            raise ValueError(f"no destination path for source {session.source_instance!r}")
        for k, method in enumerate(METHODS):
            costs[j, k] = min(_duration(session, method, case, path, links)
                              for path in candidate_paths)
            valid[j, k] = costs[j, k] <= horizon
    available = valid.any(1)
    best_method = np.argmin(np.where(valid, costs, np.inf), axis=1)
    best_cost = costs[np.arange(len(sessions)), best_method]
    groups = _drain_groups(scenario, sessions)
    base_gain = np.array([initial - source_power(
        scenario, profile, [s.session_id], case_id
    ) for s in sessions])
    gains = base_gain.copy()
    for group in groups:
        ids = [sessions[j].session_id for j in group]
        bonus = initial - source_power(scenario, profile, ids, case_id) - base_gain[group].sum()
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
            if source_power(scenario, profile, [sessions[j].session_id for j in selected], case_id) \
                    <= scenario.power_limit_w:
                break
            selected.extend(j for j in group if valid[j, METHODS.index(methods[j])])
    else:
        for j in order:
            if source_power(scenario, profile, [sessions[i].session_id for i in selected], case_id) \
                    <= scenario.power_limit_w:
                break
            if valid[j, METHODS.index(methods[j])]:
                selected.append(j)
    moved_ids = [sessions[j].session_id for j in selected]
    planned = source_power(scenario, profile, moved_ids, case_id)
    moves = _place(selected, sessions, scenario, profile, paths, methods, case_id)
    return PlanResult(
        solver, moves, initial, planned, planned <= scenario.power_limit_w,
        perf_counter() - start, int(((fractional > 1e-9) & (fractional < 1 - 1e-9)).sum()),
        profile.profile_id, case_id, seed,
    )
