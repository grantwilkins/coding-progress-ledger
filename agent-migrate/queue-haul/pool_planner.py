"""Pool-aware destination admission and deterministic replica packing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from destination import DestinationArchitecture
from planner import (_changes, _duration, _expected_scenario, _kv_catch_up_s,
                     _kv_schedule, _local_sessions, _log_bytes, _resident_tokens,
                     PlanResult, source_power)
from power_model import ExpectedPower
from simulate import ExecutionScenario, PlannedMove, SimSession, predict


@dataclass(frozen=True)
class Candidate:
    session: int
    method: str
    pool: int
    gain_w: float
    migration_work_s: float
    duration_s: float
    path: tuple[str, ...]
    route_bytes: float
    service_work: tuple[float, float]
    kv_tokens: int
    transition_work: tuple[float, float] = (0.0, 0.0)

    @property
    def replay_occupancy_s(self): return self.duration_s if self.method == "replay" else 0.0

    @property
    def kv_occupancy_s(self): return self.duration_s if self.method == "kv_transfer" else 0.0


@dataclass(frozen=True)
class CandidateTable:
    sessions: tuple[SimSession, ...]
    candidates: tuple[Candidate, ...]
    incidence: csr_matrix
    resources: csr_matrix
    resource_names: tuple[str, ...]
    migration_horizon_s: float


def _baseline(scenario: ExecutionScenario, architecture: DestinationArchitecture,
              horizon: float):
    pools = {r.replica_id: (p, architecture.type_by_id[p.type_id])
             for p in architecture.pools for r in p.replicas}
    backgrounds = [s for s in scenario.sessions if s.source_instance in pools]
    supplied = any(r.baseline_kv_tokens or any(r.baseline_work)
                   for p in architecture.pools for r in p.replicas)
    if supplied and backgrounds:
        raise ValueError("destination baselines and destination SimSession backgrounds are exclusive")
    work = {r.replica_id: np.array(r.baseline_work, float)
            for p in architecture.pools for r in p.replicas}
    kv = {
        r.replica_id: -(-r.baseline_kv_tokens // q.kv_block_tokens)
        for r, (_, q) in ((r, pools[r.replica_id])
                          for p in architecture.pools for r in p.replicas)
    }
    for session in backgrounds:
        _, q = pools[session.source_instance]
        work[session.source_instance] += q.work(
            session.expected_f, session.expected_g, session.context_tokens,
        )
        kv[session.source_instance] += -(
            -_resident_tokens(session, horizon) // q.kv_block_tokens
        )
    return work, kv


def _destination_duration(session, method, case, path, links, horizon, components):
    tokens = _resident_tokens(session, horizon) or session.context_tokens
    def route(size):
        return size / min(links[link] for link in path)
    if method == "replay":
        contexts, rates = case.replay.by_concurrency[1]
        rate = case.replay.rate(tokens, 1) if contexts[0] <= tokens <= contexts[-1] \
            else float(min(rates))
        compute = tokens / rate + case.replay_completion_s * (
            1 + _changes(session, horizon)
        )
        return route(_log_bytes(session, tokens)) \
            + components.compute_completion_factor * compute + case.switch_s
    size = case.kv_transfer.sealed_bytes(tokens)
    return max(route(size), size / case.kv_transfer.destination_bytes_per_s) \
        + components.residual_s + _kv_catch_up_s(
        session, tokens, case, horizon,
    )


def _validate_topology(scenario, architecture):
    instances = {i.instance_id for i in scenario.instances}
    links = {link.link_id for link in scenario.links}
    replicas = {r.replica_id for p in architecture.pools for r in p.replicas}
    if not replicas <= instances:
        raise ValueError("destination replica is not a ServingInstance")
    if any(not set(pool.route) <= links for pool in architecture.pools):
        raise ValueError("destination route contains an unknown link")


def _pool_rho(q, pool, work):
    total = sum((work[r.replica_id] for r in pool.replicas), start=np.zeros(2))
    return float(max(np.asarray(q.normals) @ total /
                     (len(pool.replicas) * np.asarray(q.bounds["normal"]))))


def _mode_boundary_rho(q, mode):
    return float(max(np.asarray(q.bounds[mode]) / np.asarray(q.bounds["normal"])))


def _event_bounds(q, pool, mode):
    if pool.event_flex_fraction is None:
        return np.asarray(q.bounds[mode])
    normal, stable = np.asarray(q.bounds["normal"]), np.asarray(q.bounds["stable"])
    return np.minimum(normal + pool.event_flex_fraction * stable, stable)


def service_debt(baseline, ongoing, transition, stable, horizon):
    """Return queued replica-seconds and required recovery per service row."""
    baseline, ongoing, transition, stable = map(
        lambda value: np.asarray(value, float),
        (baseline, ongoing, transition, stable),
    )
    load = baseline + ongoing
    debt = np.maximum(0, horizon * load + transition - horizon * stable)
    spare = stable - load
    recovery = np.divide(
        debt, spare, out=np.full_like(debt, np.inf), where=spare > 0,
    )
    recovery[debt == 0] = 0
    return debt, recovery


def candidate_table(scenario: ExecutionScenario, profile, architecture: DestinationArchitecture,
                    mode: str, power: ExpectedPower) -> CandidateTable:
    if mode not in {"normal", "emergency"}:
        raise ValueError("admission mode must be normal or emergency")
    _validate_topology(scenario, architecture)
    if architecture.source_compatibility.model != profile.model:
        raise ValueError("source model does not match destination architecture")
    sessions = tuple(s for s in _local_sessions(scenario) if s.state == "active")
    migration_horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    residency_horizon = architecture.residency_horizon_s
    residency_horizon = scenario.end_s - scenario.controller_delay_s \
        if residency_horizon is None else residency_horizon
    if migration_horizon <= 0:
        return CandidateTable(sessions, (), csr_matrix((len(sessions), 0)),
                              csr_matrix((0, 0)), (), migration_horizon)
    work0, kv0 = _baseline(scenario, architecture, residency_horizon)
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    candidates = []
    for j, session in enumerate(sessions):
        gain = power.marginal(session.session_id)
        for p, pool in enumerate(architecture.pools):
            q = architecture.type_by_id[pool.type_id]
            bounds = _event_bounds(q, pool, mode)
            if q.migration is None:
                profile.case("central").replay.rate(
                    _resident_tokens(session, migration_horizon), 1,
                )
            if max(np.asarray(q.normals) @ sum(
                (work0[r.replica_id] for r in pool.replicas), start=np.zeros(2)
            ) / (len(pool.replicas) * bounds)) > 1 + 1e-9:
                continue
            demand = q.work(
                session.expected_f, session.expected_g,
                _resident_tokens(session, residency_horizon),
                q.migration is not None,
            )
            baseline = sum((work0[r.replica_id] for r in pool.replicas), start=np.zeros(2))
            resident = -(-_resident_tokens(session, residency_horizon)
                         // q.kv_block_tokens)
            capacity = len(pool.replicas) * (
                q.kv_capacity_tokens // q.kv_block_tokens
            )
            if np.any(np.asarray(q.normals) @ demand >
                      len(pool.replicas) * bounds
                      - np.asarray(q.normals) @ baseline + 1e-9) \
                    or resident > capacity \
                    - sum(kv0[r.replica_id] for r in pool.replicas):
                continue
            rho, bandwidth = _pool_rho(q, pool, work0), min(links[x] for x in pool.route)
            for method in pool.methods:
                if not q.compatibility.supports(architecture.source_compatibility, method):
                    continue
                try:
                    components = None if q.migration is None else q.migration[method]
                    duration = (
                        _duration(
                            session, method, profile.case("central"), pool.route,
                            links, migration_horizon,
                        ) if components is None else
                        _destination_duration(
                            session, method, profile.case("central"), pool.route,
                            links, migration_horizon, components,
                        )
                    )
                    if method == "kv_transfer":
                        _kv_schedule(scenario, profile, session, profile.case("central"),
                                     pool.route, links)
                except ValueError:
                    continue
                if q.migration is None:
                    duration *= q.loaded[method].worst(
                        rho, _mode_boundary_rho(q, mode),
                        session.context_tokens, bandwidth,
                    )
                if duration > migration_horizon:
                    continue
                tokens = _resident_tokens(session, migration_horizon)
                route_bytes = (_log_bytes(session, tokens) if method == "replay" else
                               profile.case("central").kv_transfer.sealed_bytes(tokens))
                transition = (
                    (max(0, duration - route_bytes / bandwidth), 0)
                    if method == "replay" else (0, 0)
                )
                candidates.append(Candidate(
                    j, method, p, gain, duration, duration, pool.route, route_bytes,
                    tuple(demand), resident, transition,
                ))
    candidates = tuple(candidates)
    incidence = csr_matrix((np.ones(len(candidates)),
                            ([c.session for c in candidates], range(len(candidates)))),
                           shape=(len(sessions), len(candidates)))
    rows, capacities, names = [], [], []
    def add(values, capacity, name):
        if any(values):
            rows.append(values)
            capacities.append(capacity)
            names.append(name)
    for source in sorted({s.source_instance for s in sessions}):
        add([c.duration_s if sessions[c.session].source_instance == source else 0
             for c in candidates], migration_horizon * profile.max_source_streams,
            f"source:{source}")
    for link, rate in links.items():
        add([c.route_bytes if link in c.path else 0 for c in candidates],
            rate * migration_horizon, f"route:{link}")
    for p, pool in enumerate(architecture.pools):
        q = architecture.type_by_id[pool.type_id]
        baseline = sum((work0[r.replica_id] for r in pool.replicas), start=np.zeros(2))
        event = _event_bounds(q, pool, mode)
        for facet, (normal, bound) in enumerate(zip(q.normals, event)):
            residual = len(pool.replicas) * bound - np.asarray(normal) @ baseline
            add([np.asarray(normal) @ c.service_work if c.pool == p else 0
                 for c in candidates], residual, f"service:{pool.pool_id}:{facet}")
        if pool.event_flex_fraction is not None:
            for facet, (normal, bound) in enumerate(zip(q.normals, q.bounds["stable"])):
                capacity = migration_horizon * (
                    len(pool.replicas) * bound - np.asarray(normal) @ baseline
                    + pool.service_debt_fraction * len(pool.replicas) * bound
                )
                add([
                    migration_horizon * (np.asarray(normal) @ c.service_work)
                    + np.asarray(normal) @ c.transition_work if c.pool == p else 0
                    for c in candidates
                ], capacity, f"service-debt:{pool.pool_id}:{facet}")
        add([c.kv_tokens if c.pool == p else 0 for c in candidates],
            len(pool.replicas) * (q.kv_capacity_tokens // q.kv_block_tokens)
            - sum(kv0[r.replica_id] for r in pool.replicas), f"kv:{pool.pool_id}")
        add([c.duration_s if c.pool == p else 0 for c in candidates],
            len(pool.replicas) * migration_horizon, f"migration:{pool.pool_id}")
    data, rr, cc = [], [], []
    for i, (row, capacity) in enumerate(zip(rows, capacities)):
        for j, value in enumerate(row):
            if value:
                data.append(value / capacity)
                rr.append(i)
                cc.append(j)
    return CandidateTable(sessions, candidates, incidence,
                          csr_matrix((data, (rr, cc)), shape=(len(rows), len(candidates))),
                          tuple(names), migration_horizon)


def _greedy(table: CandidateTable, target: float):
    matrix, selected, usage = csc_matrix(table.resources), set(), np.zeros(table.resources.shape[0])
    cheapest = {}
    for i, c in enumerate(table.candidates):
        a = matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]].sum()
        if c.session not in cheapest or (a, i) < cheapest[c.session]:
            cheapest[c.session] = (a, i)
    demand = np.zeros(table.resources.shape[0])
    for _, i in cheapest.values():
        demand[matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]] += \
            matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]]
    prices, score = np.maximum(demand, 1), []
    for i, c in enumerate(table.candidates):
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, values = matrix.indices[sl], matrix.data[sl]
        score.append(c.gain_w / max(values @ prices[rows], 1e-12))
    sessions, gain = set(), 0.0
    for i in np.lexsort((np.arange(len(score)), -np.asarray(score))):
        i, c = int(i), table.candidates[int(i)]
        if gain >= target - 1e-8:
            break
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, values = matrix.indices[sl], matrix.data[sl]
        if c.session in sessions or np.any(usage[rows] + values > 1 + 1e-8):
            continue
        selected.add(i)
        sessions.add(c.session)
        usage[rows] += values
        gain += c.gain_w
    return selected


def _lp(table: CandidateTable, target: float):
    if not table.candidates:
        return set()
    n, x = len(table.candidates), cp.Variable(len(table.candidates), nonneg=True)
    gains = np.array([c.gain_w for c in table.candidates])
    work = np.array([c.migration_work_s for c in table.candidates])
    base = [table.incidence @ x <= 1, table.resources @ x <= 1, x <= 1]
    def solve(objective, constraints, maximize=False):
        p = cp.Problem(cp.Maximize(objective) if maximize else cp.Minimize(objective), constraints)
        p.solve(solver=cp.CLARABEL)
        return p
    problem = solve(work @ x, base + [gains @ x >= target])
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        problem = solve(gains @ x, base, True)
        best = float(problem.value)
        problem = solve(work @ x, base + [gains @ x >= best - 1e-7])
    values = np.asarray(x.value)
    matrix, selected, sessions = csc_matrix(table.resources), set(), set()
    usage, gain = np.zeros(table.resources.shape[0]), 0.0
    for i in np.lexsort((np.arange(n), work, -values)):
        i, c = int(i), table.candidates[int(i)]
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, added = matrix.indices[sl], matrix.data[sl]
        if c.session in sessions or np.any(usage[rows] + added > 1 + 1e-8):
            continue
        selected.add(i)
        sessions.add(c.session)
        usage[rows] += added
        gain += c.gain_w
        if gain >= target - 1e-8:
            break
    if gain < target - 1e-8:
        for i in np.lexsort((np.arange(n), work / np.maximum(gains, 1e-12))):
            i, c = int(i), table.candidates[int(i)]
            sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
            rows, added = matrix.indices[sl], matrix.data[sl]
            if c.session in sessions or np.any(usage[rows] + added > 1 + 1e-8):
                continue
            selected.add(i)
            sessions.add(c.session)
            usage[rows] += added
            gain += c.gain_w
    return selected


def _pack(table, selected, architecture, scenario, mode):
    horizon = architecture.residency_horizon_s
    horizon = scenario.end_s - scenario.controller_delay_s if horizon is None else horizon
    work, kv = _baseline(scenario, architecture, horizon)
    migration = {r.replica_id: 0.0 for p in architecture.pools for r in p.replicas}
    assignment = {}
    for p, pool in enumerate(architecture.pools):
        q = architecture.type_by_id[pool.type_id]
        normals, bounds = np.asarray(q.normals), _event_bounds(q, pool, mode)
        members = [i for i in selected if table.candidates[i].pool == p]
        members.sort(key=lambda i: (-max(
            *(normals @ table.candidates[i].service_work / bounds),
            table.candidates[i].kv_tokens
            / (q.kv_capacity_tokens // q.kv_block_tokens),
            table.candidates[i].duration_s / table.migration_horizon_s,
        ), table.sessions[table.candidates[i].session].session_id, i))
        for i in members:
            c, choices = table.candidates[i], []
            for r, replica in enumerate(pool.replicas):
                next_work, next_kv = work[replica.replica_id] + c.service_work, kv[replica.replica_id] + c.kv_tokens
                next_migration = migration[replica.replica_id] + c.duration_s
                pressure = max(*(normals @ next_work / bounds),
                               next_kv
                               / (q.kv_capacity_tokens // q.kv_block_tokens),
                               next_migration / table.migration_horizon_s)
                if pressure <= 1 + 1e-9:
                    choices.append((pressure, r, next_work, next_kv, next_migration))
            if not choices:
                return None, tuple(sorted(members))
            _, r, next_work, next_kv, next_migration = min(choices, key=lambda x: x[:2])
            work[pool.replicas[r].replica_id], kv[pool.replicas[r].replica_id] = next_work, next_kv
            migration[pool.replicas[r].replica_id] = next_migration
            assignment[i] = pool.replicas[r].replica_id
    return assignment, None


def exact_replica_assignment(table, selected, architecture, scenario, mode):
    """Small-case oracle used to validate deterministic packing."""
    selected = list(selected)
    def search(prefix, remaining):
        if not remaining:
            return prefix
        i = remaining[0]
        for replica in architecture.pools[table.candidates[i].pool].replicas:
            trial = dict(prefix)
            trial[i] = replica.replica_id
            if _assignment_valid(table, trial, architecture, scenario, mode):
                found = search(trial, remaining[1:])
                if found is not None:
                    return found
        return None
    return search({}, selected)


def _assignment_valid(table, assignment, architecture, scenario, mode):
    horizon = architecture.residency_horizon_s
    horizon = scenario.end_s - scenario.controller_delay_s if horizon is None else horizon
    work, kv = _baseline(scenario, architecture, horizon)
    migration = {r.replica_id: 0.0 for p in architecture.pools for r in p.replicas}
    pools = {r.replica_id: (architecture.type_by_id[p.type_id], p)
             for p in architecture.pools for r in p.replicas}
    for i, replica in assignment.items():
        work[replica] += table.candidates[i].service_work
        kv[replica] += table.candidates[i].kv_tokens
        migration[replica] += table.candidates[i].duration_s
    return all(np.all(np.asarray(q.normals) @ work[r] <= _event_bounds(q, pool, mode) + 1e-9)
               and kv[r] <= q.kv_capacity_tokens // q.kv_block_tokens
               and migration[r] <= table.migration_horizon_s + 1e-9
               for r, (q, pool) in pools.items())


def validate_destination_execution(scenario, architecture, moves):
    _validate_topology(scenario, architecture)
    horizon = architecture.residency_horizon_s
    horizon = scenario.end_s - scenario.controller_delay_s if horizon is None else horizon
    work, kv = _baseline(scenario, architecture, horizon)
    pools = {r.replica_id: architecture.type_by_id[p.type_id]
             for p in architecture.pools for r in p.replicas}
    sessions = {s.session_id: s for s in scenario.sessions}
    for move in moves:
        q, s = pools[move.destination_instance], sessions[move.session_id]
        work[move.destination_instance] += q.work(
            s.expected_f, s.expected_g, _resident_tokens(s, horizon),
            q.migration is not None,
        )
        kv[move.destination_instance] += -(
            -_resident_tokens(s, horizon) // q.kv_block_tokens
        )
    if any(np.any(np.asarray(q.normals) @ work[r] > np.asarray(q.bounds["stable"]) + 1e-9)
           or kv[r] > q.kv_capacity_tokens // q.kv_block_tokens
           for r, q in pools.items()):
        raise ValueError("destination replica exceeds stable envelope")


def _moves(table, selected, assignment, architecture, scenario, profile):
    links, moves = {x.link_id: x.bytes_per_s for x in scenario.links}, []
    ordered = sorted(selected, key=lambda i: (
        table.candidates[i].migration_work_s / max(table.candidates[i].gain_w, 1e-12), i,
    ))
    for order, i in enumerate(ordered):
        c, session = table.candidates[i], table.sessions[table.candidates[i].session]
        rate = quiesce = None
        if c.method == "kv_transfer" and (
            session.requests or session.expected_growth_tokens_per_s
        ):
            rate, quiesce = _kv_schedule(scenario, profile, session, profile.case("central"), c.path, links)
            rate = rate or None
        moves.append(PlannedMove(session.session_id, assignment[i], c.method, order, c.path,
                                 rate, quiesce, architecture.pools[c.pool].pool_id))
    return tuple(moves)


def _selected_service_debt(table, selected, architecture, scenario):
    horizon = table.migration_horizon_s
    residency = architecture.residency_horizon_s
    residency = scenario.end_s - scenario.controller_delay_s \
        if residency is None else residency
    work0, _ = _baseline(scenario, architecture, residency)
    debts, recoveries = [], []
    for p, pool in enumerate(architecture.pools):
        if pool.event_flex_fraction is None:
            continue
        q, normals = architecture.type_by_id[pool.type_id], np.asarray(
            architecture.type_by_id[pool.type_id].normals,
        )
        baseline = normals @ sum(
            (work0[r.replica_id] for r in pool.replicas), start=np.zeros(2),
        )
        members = [table.candidates[i] for i in selected if table.candidates[i].pool == p]
        ongoing = sum(
            (normals @ np.asarray(c.service_work) for c in members),
            start=np.zeros(len(normals)),
        )
        transition = sum(
            (normals @ np.asarray(c.transition_work) for c in members),
            start=np.zeros(len(normals)),
        )
        stable = len(pool.replicas) * np.asarray(q.bounds["stable"])
        debt, recovery = service_debt(baseline, ongoing, transition, stable, horizon)
        debts.extend(debt)
        recoveries.extend(recovery)
    return max(debts, default=0.0), max(recoveries, default=0.0)


def _mode_plan(scenario, profile, architecture, solver, mode, power, target):
    table = candidate_table(scenario, profile, architecture, mode, power)
    selected = (_lp if solver.startswith("lp") else _greedy)(table, target)
    repairs, repair_s = 0, 0.0
    while True:
        started = perf_counter()
        assignment, cut = _pack(table, selected, architecture, scenario, mode)
        if assignment is not None:
            return table, selected, assignment, repairs, repair_s
        if not cut:
            raise RuntimeError("destination packing repair did not converge")
        drop = max(cut, key=lambda i: (
            float(table.resources[:, i].sum())
            / max(table.candidates[i].gain_w, 1e-12), i,
        ))
        selected.remove(drop)
        repair_s += perf_counter() - started
        repairs += 1


def plan_destination(scenario, profile, solver, case_id, seed, architecture):
    if solver not in {"greedy", "lp", "lp_peak_first", "lp_work_first"}:
        raise ValueError("destination architecture supports pool-aware LP and greedy")
    if case_id != "central":
        raise ValueError("destination admission supports the central profile")
    selection_scenario = replace(
        scenario, final_state="awake", assumed_shutdown_s=None,
    )
    start, power = perf_counter(), ExpectedPower(selection_scenario, profile, case_id)
    initial, target = power.power(True), max(0.0, power.power(True) - scenario.power_limit_w)
    chosen = None
    for mode in ("normal", "emergency"):
        result = _mode_plan(scenario, profile, architecture, solver, mode, power, target)
        moved = [result[0].sessions[result[0].candidates[i].session].session_id for i in result[1]]
        planned = source_power(selection_scenario, profile, moved, case_id)
        chosen = mode, result, planned
        if planned <= scenario.power_limit_w + 1e-8:
            break
    mode, (table, selected, assignment, repairs, repair_s), planned = chosen
    moves = _moves(table, selected, assignment, architecture, scenario, profile)
    validate_destination_execution(scenario, architecture, moves)
    expected = predict(_expected_scenario(scenario, moves), profile, moves, case_id)
    shortfall = max(0.0, planned - scenario.power_limit_w)
    shortfall = 0.0 if shortfall <= 1e-8 else shortfall
    feasible = shortfall == 0 and expected.deadline_met
    failure = None if feasible else "target_unmet" if shortfall else "migration_deadline"
    usage = np.asarray(table.resources[:, list(selected)].sum(1)).ravel() if selected else np.zeros(len(table.resource_names))
    bottleneck = table.resource_names[int(usage.argmax())] if usage.size and usage.max() else None
    binding = tuple(
        name for name, value in zip(table.resource_names, usage)
        if value >= 1 - 1e-7
    )
    debt, recovery = _selected_service_debt(
        table, selected, architecture, scenario,
    )
    temporal = [usage[i] for i, name in enumerate(table.resource_names)
                if name.startswith(("source:", "route:", "migration:"))]
    horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    makespan = max(expected.migration_makespan_s or 0, scenario.controller_delay_s + max(
        [horizon * max(temporal, default=0)]
        + [table.candidates[i].duration_s for i in selected],
    ))
    memory = sum(a.nbytes for a in (table.resources.data, table.resources.indices,
                                   table.resources.indptr, table.incidence.data,
                                   table.incidence.indices, table.incidence.indptr))
    return PlanResult(
        solver, moves, initial, planned, expected.modeled_source_power_at_deadline_w,
        feasible, perf_counter() - start, profile.profile_id, case_id, seed,
        sum(len(p.replicas) * architecture.type_by_id[p.type_id].kv_capacity_tokens
            for p in architecture.pools), shortfall, None, mode, shortfall, failure,
        repairs, repair_s, makespan, bottleneck, memory, debt, recovery, binding,
    )
