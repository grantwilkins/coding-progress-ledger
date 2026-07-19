"""Whole-session planning against the measured source power curve."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from time import perf_counter
from typing import Callable

import cvxpy as cp
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from profiles import ModelProfile, ProfileCase
from power_model import ExpectedPower
from simulate import (MOVE_METHODS_BY_STATE, ExecutionScenario, MoveMethod, PlannedMove,
                      SimSession, predict)


METHODS: tuple[MoveMethod, ...] = ("replay", "kv_transfer", "replay_on_request")
SOLVERS = ("random", "load_only", "node_aware", "node_drain")
LP_SOLVERS = ("lp", "lp_peak_first", "lp_work_first")
ALL_SOLVERS = SOLVERS + LP_SOLVERS
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
    lp_power_shortfall_w: float | None = None
    lp_peak_pressure: float | None = None


def _ell(session: SimSession, case: ProfileCase) -> float:
    return session.expected_f / case.F + session.expected_g / case.G


def _resident_tokens(session: SimSession, horizon: float = 0.0) -> int:
    return math.ceil(session.context_tokens + session.expected_growth_tokens_per_s * horizon) \
        if session.state == "active" else 0


def source_power(scenario: ExecutionScenario, profile: ModelProfile, moved=(),
                 case_id: str = "central") -> float:
    """Expected local power after committed moves and the requested final node state."""
    state, moved = ExpectedPower(scenario, profile, case_id), set(moved)
    for session in scenario.sessions:
        if session.session_id in moved:
            state.remove(session.session_id)
    return state.power(True)


def _duration(session: SimSession, method: MoveMethod, case: ProfileCase,
              path: tuple[str, ...], links: dict[str, float], horizon: float = 0.0) -> float:
    def link_s(size):
        return size / min(links[link] for link in path)
    tokens = _resident_tokens(session, horizon) or session.context_tokens
    replay_s = tokens / case.replay.rate(tokens, 1)
    if method == "replay":
        return link_s(session.log_bytes) + replay_s + case.replay_completion_s + case.switch_s
    if method == "kv_transfer":
        size = case.kv_transfer.bytes(tokens)
        return (case.kv_transfer.setup_s
                + max(link_s(size), size / case.kv_transfer.destination_bytes_per_s)
                + case.kv_transfer.initial_completion_s + case.switch_s)
    wake_s = link_s(session.log_bytes) + replay_s + case.replay_completion_s
    return case.switch_s + session.wake_probability * wake_s


def _required_kv_rate(session: SimSession, case: ProfileCase, quiesce_s: float,
                      controller_s: float, physical: float) -> float:
    growth_s = quiesce_s - controller_s
    transfer_s = growth_s - case.kv_transfer.setup_s \
        - case.kv_transfer.initial_completion_s
    if transfer_s <= 0:
        raise ValueError("KV preparation has no transfer window")
    rate = case.kv_transfer.bytes(math.ceil(
        session.context_tokens + session.expected_growth_tokens_per_s * growth_s
    )) / transfer_s
    if rate > physical:
        raise ValueError("KV preparation exceeds physical capacity")
    return rate


def _kv_schedule(scenario: ExecutionScenario, profile: ModelProfile,
                 session: SimSession, case: ProfileCase, path: tuple[str, ...],
                 links: dict[str, float]) -> tuple[float, float]:
    transition = 0.0 if scenario.final_state == "awake" else (
        case.sleep_s if scenario.final_state == "sleep"
        else scenario.assumed_shutdown_s
    )
    reserve = (
        profile.power_window_s + case.switch_s
        + case.kv_transfer.catch_up_fixed_s
        + case.kv_transfer.block_tokens / case.kv_transfer.tail_replay_tps
        + transition
    )
    quiesce = max(scenario.controller_delay_s, scenario.deadline_s - reserve)
    physical = min(
        case.kv_transfer.destination_bytes_per_s,
        *(links[link_id] for link_id in path),
    )
    return _required_kv_rate(
        session, case, quiesce, scenario.controller_delay_s, physical,
    ), quiesce


def _expected_scenario(scenario: ExecutionScenario,
                       moves: tuple[PlannedMove, ...]) -> ExecutionScenario:
    by_session = {move.session_id: move for move in moves}
    return replace(scenario, sessions=tuple(
        replace(
            session,
            context_tokens=math.ceil(
                session.context_tokens
                + session.expected_growth_tokens_per_s
                * max(
                    0.0,
                    (by_session[session.session_id].quiesce_s
                     if by_session[session.session_id].quiesce_s is not None
                     else scenario.deadline_s)
                    - scenario.controller_delay_s,
                )
            ) if session.session_id in by_session else session.context_tokens,
            requests=(), expected_growth_tokens_per_s=0,
        )
        for session in scenario.sessions
    ))


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


def _route_resources(source: str, destinations, routes: Routes, links: dict[str, float],
                     cache: dict):
    paths = tuple(dict.fromkeys(
        _route(routes, source, destination.instance_id) for destination in destinations
    ))
    if any(not path or any(link not in links for link in path) for path in paths):
        raise ValueError("LP route contains an unknown link")
    if paths in cache:
        return cache[paths]

    def summarize(options):
        common = set(options[0]).intersection(*map(set, options[1:]))
        variable = [set(path) - common for path in options]
        if any(len(path) > 1 for path in variable):
            raise ValueError("LP supports one destination-pool link per route")
        return common, set().union(*variable), min(
            min(links[link] for link in path) for path in options
        )

    cache[paths] = summarize(paths), summarize([path[-1:] for path in paths])
    return cache[paths]


def _migration_resources(scenario: ExecutionScenario, profile: ModelProfile, routes: Routes,
                         sessions: list[SimSession], destinations, case: ProfileCase,
                         horizon: float):
    n = len(sessions)
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    destination_ids = {instance.instance_id for instance in destinations}
    summaries = {}
    route_cache = {
        source: _route_resources(source, destinations, routes, links, summaries)
        for source in {session.source_instance for session in sessions}
    }
    replay_s = np.zeros(n)
    replay_valid = np.ones(n, bool)
    for j, session in enumerate(sessions):
        tokens = _resident_tokens(session, horizon)
        try:
            replay_s[j] = tokens / case.replay.rate(tokens, 1)
        except ValueError:
            replay_valid[j] = False
    kv_bytes = np.array([
        case.kv_transfer.bytes(_resident_tokens(session, horizon)) for session in sessions
    ], float)
    replay_bytes = np.array([session.log_bytes for session in sessions], float)
    durations = np.zeros((2, n))
    kv_service = np.zeros(n)
    named_links: dict[str, dict[int, float]] = {}
    flexible_links: set[str] | None = None
    flexible: dict[int, float] = {}

    for j, session in enumerate(sessions):
        internal, external = route_cache[session.source_instance]
        replay_route = internal
        durations[0, j] = replay_bytes[j] / replay_route[2] + replay_s[j] + case.switch_s
        transfer_s = max(kv_bytes[j] / internal[2],
                         kv_bytes[j] / case.kv_transfer.destination_bytes_per_s)
        kv_service[j] = transfer_s + case.kv_transfer.initial_completion_s
        durations[1, j] = case.kv_transfer.setup_s + kv_service[j] + case.switch_s
        for method, (byte_count, route) in enumerate((
            (replay_bytes[j], replay_route), (kv_bytes[j], internal)
        )):
            column = method * n + j
            for link in sorted(route[0]):
                named_links.setdefault(link, {})[column] = byte_count
            if route[1]:
                if flexible_links is None:
                    flexible_links = route[1]
                elif flexible_links != route[1]:
                    raise ValueError("planner requires one shared destination link pool")
                flexible[column] = byte_count
    if flexible_links and flexible_links & named_links.keys():
        raise ValueError("planner destination-pool links must not also be fixed route links")

    valid = (durations <= max(horizon, 0)).reshape(-1)
    valid[:n] &= replay_valid
    row, column, data, resource_count = [], [], [], 0

    def add_resource(entries, capacity):
        nonlocal resource_count
        entries = [(int(col), float(value)) for col, value in entries if value > 0]
        if not entries:
            return
        if capacity <= 0:
            valid[[col for col, _ in entries]] = False
            return
        row.extend([resource_count] * len(entries))
        column.extend(col for col, _ in entries)
        data.extend(value / capacity for _, value in entries)
        resource_count += 1

    by_source: dict[str, list[int]] = {}
    for j, session in enumerate(sessions):
        by_source.setdefault(session.source_instance, []).append(j)
    for indices in by_source.values():
        add_resource(
            ((method * n + j, durations[method, j])
             for j in indices for method in range(2)),
            horizon * profile.max_source_streams,
        )
    for link, entries in named_links.items():
        add_resource(entries.items(), links[link] * horizon)
    if flexible_links:
        add_resource(flexible.items(), sum(links[link] for link in flexible_links) * horizon)
    add_resource(((j, replay_s[j]) for j in range(n)),
                 len(destinations) * profile.max_destination_replays * horizon)
    add_resource(((n + j, kv_service[j]) for j in range(n)),
                 len(destinations) * profile.max_destination_kv_streams * horizon)

    destination_load = sum(
        _ell(session, case) for session in scenario.sessions
        if session.source_instance in destination_ids
    )
    destination_tokens = sum(
        _resident_tokens(session, horizon) for session in scenario.sessions
        if session.source_instance in destination_ids
    )
    add_resource(
        ((method * n + j, _ell(session, case))
         for j, session in enumerate(sessions) for method in range(2)),
        sum(len(instance.gpu_nodes) for instance in destinations) * profile.max_ell
        - destination_load,
    )
    add_resource(
        ((method * n + j, _resident_tokens(session, horizon))
         for j, session in enumerate(sessions) for method in range(2)),
        len(destinations) * profile.kv_capacity_tokens - destination_tokens,
    )
    return durations, valid, csr_matrix(
        (data, (row, column)), shape=(resource_count, 2 * n)
    )


def _round_lp(values: np.ndarray, valid: np.ndarray, resources: csr_matrix,
              gains: np.ndarray, work: np.ndarray, target: float):
    n = gains.size
    values = np.clip(values, 0, 1)
    usage = np.zeros(resources.shape[0])
    chosen = np.full(n, -1, int)
    matrix = csc_matrix(resources)
    gain = 0.0

    def take(column):
        nonlocal gain
        session, method = column % n, column // n
        if chosen[session] >= 0 or not valid[column]:
            return False
        start, end = matrix.indptr[column:column + 2]
        rows, added = matrix.indices[start:end], matrix.data[start:end]
        if np.any(usage[rows] + added > 1 + 1e-8):
            return False
        chosen[session] = method
        usage[rows] += added
        gain += gains[session]
        return True

    z = np.maximum(0, 1 - values[:n] - values[n:])
    preferred = [
        method * n + session
        for session in range(n)
        for method in [int(values[session + n] > values[session])]
        if values[method * n + session] + 1e-8 >= z[session]
    ]
    preferred.sort(key=lambda column: (-values[column], work[column], column))
    for column in preferred:
        if gain + 1e-8 >= target:
            break
        take(column)

    score = np.tile(gains, 2) / np.maximum(work, 1e-12)
    for column in np.lexsort((np.arange(2 * n), -score, -values)):
        if gain + 1e-8 >= target:
            break
        take(int(column))
    return chosen, usage


def _execution_feasible(scenario: ExecutionScenario, expected) -> bool:
    return expected.deadline_met and all(
        row.committed_s is not None and row.committed_s <= scenario.deadline_s
        for row in expected.sessions
    )


def _node_drain_greedy(groups, sessions, gains, durations, valid, resources, horizon,
                       power: ExpectedPower, limit: float):
    n = len(gains)
    matrix = csc_matrix(resources)
    chosen = np.full(n, -1, int)
    usage = np.zeros(resources.shape[0])

    def column(method, session):
        start, end = matrix.indptr[method * n + session:method * n + session + 2]
        return matrix.indices[start:end], matrix.data[start:end]

    pressure = np.full(n, np.inf)
    for j in range(n):
        pressure[j] = min(
            (max(column(method, j)[1], default=0.0)
             for method in range(2) if valid[method * n + j]),
            default=np.inf,
        )
    ordered_groups = []
    for group in groups:
        order = sorted(group, key=lambda j: (
            -gains[j] / max(pressure[j], 1e-12), pressure[j], j
        ))
        group_usage = np.zeros(resources.shape[0])
        peak = 0.0
        for j in order:
            options = []
            for method in range(2):
                if not valid[method * n + j]:
                    continue
                rows, added = column(method, j)
                options.append((
                    max(peak, max(group_usage[rows] + added, default=0.0)),
                    durations[method, j], method, rows, added,
                ))
            if options:
                peak, _, _, rows, added = min(options, key=lambda item: item[:3])
                group_usage[rows] += added
        score = sum(gains[j] for j in group if np.isfinite(pressure[j])) \
            / max(horizon * peak, 1e-12) if peak else 0.0
        ordered_groups.append((-score, min(group), order))

    selected = []
    for _, _, group in sorted(ordered_groups):
        if power.power(True) <= limit:
            break
        for j in group:
            options = []
            for method in range(2):
                if not valid[method * n + j]:
                    continue
                rows, added = column(method, j)
                remaining = 1 - usage[rows]
                if np.any(added > remaining + 1e-8):
                    continue
                options.append((
                    max(added / np.maximum(remaining, 1e-12), default=0.0),
                    durations[method, j], method, rows, added,
                ))
            if not options:
                continue
            _, _, method, rows, added = min(options, key=lambda item: item[:3])
            chosen[j] = method
            usage[rows] += added
            selected.append(j)
            power.remove(sessions[j].session_id)
    return selected, chosen, usage


def _solve_lp(solver: str, gains: np.ndarray, work: np.ndarray, valid: np.ndarray,
              resources: csr_matrix, target: float) -> np.ndarray:
    n, scale = gains.size, max(target, 1.0)
    x = cp.Variable(2 * n, nonneg=True)
    selected = x[:n] + x[n:]
    base = [selected <= 1, resources @ x <= 1]
    if (~valid).any():
        base.append(x[~valid] == 0)

    def solve(objective, constraints, maximize=False):
        problem = cp.Problem(
            cp.Maximize(objective) if maximize else cp.Minimize(objective), constraints
        )
        problem.solve(solver=cp.CLARABEL)
        return problem

    normalized_work = work @ x / max(work.max(), 1.0)
    if solver == "lp":
        problem = solve(
            normalized_work, base + [gains @ selected / scale >= target / scale]
        )
        if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            problem = solve(gains @ selected / scale, base, maximize=True)
    else:
        shortfall = cp.Variable(nonneg=True)
        linked = base + [gains @ selected / scale + shortfall >= target / scale]
        problem = solve(shortfall, linked)
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            linked += [shortfall <= float(shortfall.value) + 1e-8]
            phi = cp.Variable(nonneg=True)
            if solver == "lp_peak_first":
                problem = solve(phi, linked + [resources @ x <= phi, phi <= 1])
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    problem = solve(
                        normalized_work,
                        linked + [resources @ x <= phi, phi <= float(phi.value) + 1e-8],
                    )
            elif solver == "lp_work_first":
                problem = solve(normalized_work, linked)
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    problem = solve(
                        phi,
                        linked + [normalized_work <= float(problem.value) + 1e-8,
                                  resources @ x <= phi, phi <= 1],
                    )
            else:
                raise ValueError(f"unknown LP solver {solver!r}")
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP planner returned {problem.status}")
    return np.asarray(x.value)


def _plan_lp(scenario: ExecutionScenario, profile: ModelProfile, routes: Routes,
             solver: str, case_id: str, seed: int, start: float) -> PlanResult:
    if scenario.final_state != "awake":
        raise ValueError("LP supports final_state='awake'")
    sessions = _local_sessions(scenario)
    if any(session.state != "active" for session in sessions):
        raise ValueError("LP currently supports active sessions only")
    nodes = {node.node_id: node for node in scenario.nodes}
    destinations = [
        instance for instance in scenario.instances
        if all(not nodes[node].local for node in instance.gpu_nodes)
    ]
    if not destinations:
        raise ValueError("scenario has no destination instance")
    power = ExpectedPower(scenario, profile, case_id)
    initial = power.power(True)
    if initial <= scenario.power_limit_w:
        return PlanResult(
            solver, (), initial, initial, initial, True, perf_counter() - start,
            profile.profile_id, case_id, seed, profile.kv_capacity_tokens, 0.0, 0.0,
        )
    n, case = len(sessions), profile.case(case_id)
    if not n:
        return PlanResult(
            solver, (), initial, initial, initial, False, perf_counter() - start,
            profile.profile_id, case_id, seed, profile.kv_capacity_tokens,
            initial - scenario.power_limit_w, 0.0,
        )
    horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    durations, valid, resources = _migration_resources(
        scenario, profile, routes, sessions, destinations, case, horizon
    )
    gains = np.array([power.marginal(session.session_id) for session in sessions])
    target = initial - scenario.power_limit_w
    work = durations.reshape(-1)
    chosen, usage = _round_lp(
        _solve_lp(solver, gains, work, valid, resources, target),
        valid, resources, gains, work, target,
    )
    selected_indices = [j for j in range(n) if chosen[j] >= 0]
    selected_indices.sort(
        key=lambda j: durations[chosen[j], j] / max(gains[j], 1e-12)
    )
    methods = [METHODS[chosen[j]] if chosen[j] >= 0 else METHODS[0] for j in range(n)]
    moves = _place(
        selected_indices, sessions, scenario, profile, routes, methods, case_id
    )
    for j in selected_indices:
        power.remove(sessions[j].session_id)
    planned = power.power(True)
    expected = predict(
        _expected_scenario(scenario, moves),
        profile, moves, case_id,
    )
    feasible = planned <= scenario.power_limit_w and max(usage, default=0) <= 1 + 1e-8 \
        and _execution_feasible(scenario, expected)
    return PlanResult(
        solver, moves, initial, planned, expected.modeled_source_power_at_deadline_w,
        feasible, perf_counter() - start, profile.profile_id, case_id, seed,
        profile.kv_capacity_tokens, max(0.0, planned - scenario.power_limit_w),
        max(usage, default=0.0),
    )


def _place(selected: list[int], sessions: list[SimSession], scenario: ExecutionScenario,
           profile: ModelProfile, routes: Routes,
           methods: list[MoveMethod], case_id: str) -> tuple[PlannedMove, ...]:
    case = profile.case(case_id)
    link_rates = {link.link_id: link.bytes_per_s for link in scenario.links}
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
            tokens[session.source_instance] += _resident_tokens(
                session, scenario.deadline_s - scenario.controller_delay_s
            )
    size = len(destinations[0].gpu_nodes)
    capacity = InstanceCapacity(
        [load[i.instance_id] for i in destinations],
        [tokens[i.instance_id] for i in destinations],
        profile.max_ell * size, profile.kv_capacity_tokens,
    )
    moves = []
    for order, j in enumerate(selected):
        session, ell = sessions[j], _ell(sessions[j], case)
        horizon = scenario.deadline_s - scenario.controller_delay_s
        destination = destinations[capacity.place(
            ell, _resident_tokens(session, horizon)
        )].instance_id
        path = _route(routes, session.source_instance, destination)
        rate, quiesce = None, None
        if methods[j] == "kv_transfer":
            rate, quiesce = _kv_schedule(
                scenario, profile, session, case, path, link_rates,
            )
        moves.append(PlannedMove(
            session.session_id, destination, methods[j], order, path,
            rate_limit_bytes_per_s=rate, quiesce_s=quiesce,
        ))
    return tuple(moves)


def plan(scenario: ExecutionScenario, profile: ModelProfile,
         paths: Routes, solver: str,
         case_id: str = "central", seed: int = 0) -> PlanResult:
    if solver not in ALL_SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    start = perf_counter()
    if solver in LP_SOLVERS:
        return _plan_lp(scenario, profile, paths, solver, case_id, seed, start)
    case = profile.case(case_id)
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
            try:
                costs[j, k] = _duration(
                    session, method, case, candidate_path, links, horizon
                )
                if method == "kv_transfer":
                    _kv_schedule(
                        scenario, profile, session, case, candidate_path, links,
                    )
                valid[j, k] = method in MOVE_METHODS_BY_STATE[session.state] \
                    and costs[j, k] <= horizon
            except ValueError:
                pass
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
    capacity_node_drain = solver == "node_drain" and scenario.final_state == "awake" \
        and all(session.state == "active" for session in sessions)
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
    elif solver == "node_drain" and not capacity_node_drain:
        groups.sort(key=lambda group: best_cost[group].sum() / max(gains[group].sum(), 1e-12))
        groups = [sorted(group, key=lambda j: best_cost[j]) for group in groups]
        methods = [METHODS[k] for k in best_method]
    selected = []
    if capacity_node_drain:
        resource_horizon = horizon - profile.power_window_s
        durations, resource_valid, resources = _migration_resources(
            scenario, profile, paths, sessions, destinations, case, resource_horizon
        )
        selected, chosen, _ = _node_drain_greedy(
            groups, sessions, gains, durations, resource_valid, resources,
            resource_horizon, power_state, scenario.power_limit_w,
        )
        methods = [METHODS[chosen[j]] if chosen[j] >= 0 else METHODS[0]
                   for j in range(len(sessions))]
    elif solver == "node_drain":
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
        _expected_scenario(scenario, moves),
        profile, moves, case_id,
    )
    feasible = planned <= scenario.power_limit_w and _execution_feasible(scenario, expected)
    return PlanResult(
        solver, moves, initial, planned, expected.modeled_source_power_at_deadline_w, feasible,
        perf_counter() - start, profile.profile_id, case_id, seed,
        profile.kv_capacity_tokens,
    )
