"""Pool-aware destination admission and deterministic replica packing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter

import cvxpy as cp
import highspy
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack

from destination import DestinationArchitecture
from planner import (_changes, _duration, _expected_scenario, _kv_catch_up_s,
                     _kv_schedule, _local_sessions, _log_bytes, _resident_tokens,
                     PlanResult, ResourceUse, ServiceDebtUse, source_power)
from power_model import ExpectedPower
from simulate import (ExecutionScenario, PlannedMove, PoolServiceExecution,
                      SimSession, predict)


MAX_EXACT_COUPLED_PATTERNS = 10_000
COUPLED_PRICE_ITERATIONS = 32
DUAL_PRICE_ITERATIONS = 1
DUAL_PREFIX_BUCKETS = 8
DUAL_HIGH_TARGET_ITERATIONS = 4
DUAL_HIGH_TARGET_BUCKETS = 64
DUAL_HIGH_TARGET_FRACTION = .75
COLUMN_GROWTH_SWEEPS = 20
COLUMN_TOLERANCE = 1e-8


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
    resource_capacities: tuple[float, ...]
    resource_units: tuple[str, ...]
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


def _service_trace(initial, capacity, changes, start, end, detailed=True):
    """Fluid queue under time-varying declared demand."""
    if min(initial, capacity, start) < 0 or capacity <= 0 or end < start \
            or any(not start <= time <= end for time, _ in changes):
        raise ValueError("invalid service trace")
    demand, queue, peak, time, rows = initial, 0.0, 0.0, start, []
    grouped = {}
    for at, delta in changes:
        grouped[at] = grouped.get(at, 0.0) + delta
    for at in (*sorted(grouped), end):
        queue = max(0.0, queue + (demand - capacity) * (at - time))
        peak = max(peak, queue)
        demand += grouped.get(at, 0.0)
        if demand < -1e-9:
            raise ValueError("service demand became negative")
        demand = max(0.0, demand)
        if detailed:
            rows.append((at, demand, queue, peak))
        else:
            rows[:] = [(at, demand, queue, peak)]
        time = at
    return tuple(rows)


def destination_service_execution(
    scenario, profile, architecture, moves, result, detailed=True,
):
    """Schedule declared pool work from realized replay and commit times."""
    horizon = scenario.deadline_s - scenario.controller_delay_s - profile.power_window_s
    end = scenario.controller_delay_s + horizon
    residency = architecture.residency_horizon_s
    residency = scenario.end_s - scenario.controller_delay_s \
        if residency is None else residency
    work0, _ = _baseline(scenario, architecture, residency)
    sessions = {session.session_id: session for session in scenario.sessions}
    pools = {pool.pool_id: pool for pool in architecture.pools}
    move_pool = {move.session_id: pools[move.destination_pool] for move in moves}
    move_type = {
        session_id: architecture.type_by_id[pool.type_id]
        for session_id, pool in move_pool.items()
    }
    changes = {(pool.pool_id, facet): [] for pool in architecture.pools
               for facet in range(len(architecture.type_by_id[pool.type_id].normals))}
    for row in result.sessions:
        if row.session_id not in move_pool:
            continue
        pool, q = move_pool[row.session_id], move_type[row.session_id]
        normals = np.asarray(q.normals)
        if row.method == "replay":
            for start, finish in (
                (row.initial_replay_start_s, row.initial_ready_s),
                (row.catch_up_replay_start_s, row.catch_up_ready_s),
            ):
                if start is not None and start <= end:
                    delta = normals @ np.array((1.0, 0.0))
                    for facet, value in enumerate(delta):
                        changes[pool.pool_id, facet].append((start, float(value)))
                        if finish is not None and finish <= end:
                            changes[pool.pool_id, facet].append(
                                (finish, float(-value))
                            )
        if row.committed_s is not None and row.committed_s <= end:
            session = sessions[row.session_id]
            delta = normals @ q.work(
                session.expected_f, session.expected_g,
                _resident_tokens(session, residency), q.migration is not None,
            )
            for facet, value in enumerate(delta):
                changes[pool.pool_id, facet].append((row.committed_s, float(value)))
    rows = []
    for pool in architecture.pools:
        q, normals = architecture.type_by_id[pool.type_id], np.asarray(
            architecture.type_by_id[pool.type_id].normals,
        )
        baseline = normals @ sum(
            (work0[replica.replica_id] for replica in pool.replicas),
            start=np.zeros(2),
        )
        stable = len(pool.replicas) * np.asarray(q.bounds["stable"])
        event = stable if pool.event_flex_fraction is None else \
            len(pool.replicas) * _event_bounds(q, pool, "normal")
        budget = (
            np.full(len(normals), np.inf) if pool.event_flex_fraction is None
            else pool.service_debt_fraction * horizon * stable
        )
        for facet in range(len(normals)):
            trace = _service_trace(
                float(baseline[facet]), float(stable[facet]),
                changes[pool.pool_id, facet], scenario.controller_delay_s, end,
                detailed,
            )
            final_demand, final_queue, peak = trace[-1][1:]
            spare = stable[facet] - final_demand
            recovery = 0.0 if final_queue == 0 else (
                final_queue / spare if spare > 0 else np.inf
            )
            within = final_demand <= event[facet] + 1e-9 \
                and peak <= budget[facet] + 1e-9 and np.isfinite(recovery)
            rows.extend(PoolServiceExecution(
                pool.pool_id, facet, time, demand, float(stable[facet]), queue,
                peak_at_time, float(budget[facet]), float(recovery), bool(within),
            ) for time, demand, queue, peak_at_time in trace)
    return tuple(rows)


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
                              csr_matrix((0, 0)), (), (), (), migration_horizon)
    work0, kv0 = _baseline(scenario, architecture, residency_horizon)
    pool_work = tuple(
        sum((work0[r.replica_id] for r in pool.replicas), start=np.zeros(2))
        for pool in architecture.pools
    )
    pool_kv = tuple(
        sum(kv0[r.replica_id] for r in pool.replicas)
        for pool in architecture.pools
    )
    case, types = profile.case("central"), architecture.type_by_id
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    candidates = []
    for j, session in enumerate(sessions):
        gain = power.marginal(session.session_id)
        migration_tokens = _resident_tokens(session, migration_horizon)
        residency_tokens = _resident_tokens(session, residency_horizon)
        demand_cache, duration_cache = {}, {}
        route_bytes = {
            "replay": _log_bytes(session, migration_tokens),
            "kv_transfer": case.kv_transfer.sealed_bytes(migration_tokens),
        }
        for p, pool in enumerate(architecture.pools):
            q = types[pool.type_id]
            bounds = _event_bounds(q, pool, mode)
            if q.migration is None:
                case.replay.rate(migration_tokens, 1)
            baseline = pool_work[p]
            if max(np.asarray(q.normals) @ baseline
                   / (len(pool.replicas) * bounds)) > 1 + 1e-9:
                continue
            demand_key = (q.type_id, q.migration is not None)
            if demand_key not in demand_cache:
                demand_cache[demand_key] = q.work(
                    session.expected_f, session.expected_g,
                    residency_tokens, q.migration is not None,
                )
            demand = demand_cache[demand_key]
            resident = -(-residency_tokens // q.kv_block_tokens)
            capacity = len(pool.replicas) * (
                q.kv_capacity_tokens // q.kv_block_tokens
            )
            if np.any(np.asarray(q.normals) @ demand >
                      len(pool.replicas) * bounds
                      - np.asarray(q.normals) @ baseline + 1e-9) \
                    or resident > capacity \
                    - pool_kv[p]:
                continue
            rho = float(max(np.asarray(q.normals) @ baseline /
                            (len(pool.replicas) * np.asarray(q.bounds["normal"]))))
            bandwidth = min(links[x] for x in pool.route)
            for method in pool.methods:
                if not q.compatibility.supports(architecture.source_compatibility, method):
                    continue
                duration_key = (q.type_id, pool.route, method, rho, mode)
                if duration_key not in duration_cache:
                    try:
                        components = None if q.migration is None else q.migration[method]
                        duration = (
                            _duration(
                                session, method, case, pool.route,
                                links, migration_horizon,
                            ) if components is None else
                            _destination_duration(
                                session, method, case, pool.route,
                                links, migration_horizon, components,
                            )
                        )
                        if method == "kv_transfer":
                            _kv_schedule(scenario, profile, session, case,
                                         pool.route, links)
                        if q.migration is None:
                            duration *= q.loaded[method].worst(
                                rho, _mode_boundary_rho(q, mode),
                                session.context_tokens, bandwidth,
                            )
                        duration_cache[duration_key] = duration
                    except ValueError:
                        duration_cache[duration_key] = None
                duration = duration_cache[duration_key]
                if duration is None:
                    continue
                if duration > migration_horizon:
                    continue
                transition = (
                    (max(0, duration - route_bytes[method] / bandwidth), 0)
                    if method == "replay" else (0, 0)
                )
                candidates.append(Candidate(
                    j, method, p, gain, duration, duration, pool.route,
                    route_bytes[method],
                    tuple(demand), resident, transition,
                ))
    candidates = tuple(candidates)
    incidence = csr_matrix((np.ones(len(candidates)),
                            ([c.session for c in candidates], range(len(candidates)))),
                           shape=(len(sessions), len(candidates)))
    specs, row_for = [], {}
    def add(key, capacity, name, unit):
        row_for[key] = len(specs)
        specs.append((capacity, name, unit))
    for link, rate in links.items():
        add(("route", link), rate * migration_horizon, f"route:{link}", "bytes")
    for p, pool in enumerate(architecture.pools):
        q = types[pool.type_id]
        baseline = pool_work[p]
        event = _event_bounds(q, pool, mode)
        for facet, (normal, bound) in enumerate(zip(q.normals, event)):
            residual = len(pool.replicas) * bound - np.asarray(normal) @ baseline
            add(("service", p, facet), residual,
                f"service:{pool.pool_id}:{facet}", "replica-s/s")
        if pool.event_flex_fraction is not None:
            for facet, (normal, bound) in enumerate(zip(q.normals, q.bounds["stable"])):
                capacity = migration_horizon * (
                    len(pool.replicas) * bound - np.asarray(normal) @ baseline
                    + pool.service_debt_fraction * len(pool.replicas) * bound
                )
                add(("debt", p, facet), capacity,
                    f"service-debt:{pool.pool_id}:{facet}", "replica-s")
        add(("kv", p), len(pool.replicas)
            * (q.kv_capacity_tokens // q.kv_block_tokens) - pool_kv[p],
            f"kv:{pool.pool_id}", "blocks")
        for method in pool.methods:
            add(("migration", p, method), len(pool.replicas) * migration_horizon,
                f"migration:{pool.pool_id}:{method}", "replica-s")
    data, rr, cc = [], [], []
    def emit(key, column, value):
        if value:
            row = row_for[key]
            data.append(value / specs[row][0])
            rr.append(row)
            cc.append(column)
    for column, candidate in enumerate(candidates):
        for link in dict.fromkeys(candidate.path):
            emit(("route", link), column, candidate.route_bytes)
        p, pool = candidate.pool, architecture.pools[candidate.pool]
        q = types[pool.type_id]
        for facet, normal in enumerate(q.normals):
            emit(("service", p, facet), column,
                 np.asarray(normal) @ candidate.service_work)
            if pool.event_flex_fraction is not None:
                emit(("debt", p, facet), column,
                     migration_horizon * (np.asarray(normal) @ candidate.service_work)
                     + np.asarray(normal) @ candidate.transition_work)
        emit(("kv", p), column, candidate.kv_tokens)
        emit(("migration", p, candidate.method), column, candidate.duration_s)
    used = sorted(set(rr))
    remap = {old: new for new, old in enumerate(used)}
    rr = [remap[row] for row in rr]
    capacities = tuple(specs[row][0] for row in used)
    names = tuple(specs[row][1] for row in used)
    units = tuple(specs[row][2] for row in used)
    return CandidateTable(sessions, candidates, incidence,
                          csr_matrix((data, (rr, cc)), shape=(len(used), len(candidates))),
                          names, capacities, units,
                          migration_horizon)


def _scarcity_prices(table, matrix):
    cheapest = {}
    for i, c in enumerate(table.candidates):
        a = matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]].sum()
        if c.session not in cheapest or (a, i) < cheapest[c.session]:
            cheapest[c.session] = (a, i)
    demand = np.zeros(table.resources.shape[0])
    for _, i in cheapest.values():
        demand[matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]] += \
            matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]]
    return np.maximum(demand, 1)


def _greedy(table: CandidateTable, target: float):
    matrix, selected, usage = csc_matrix(table.resources), set(), np.zeros(table.resources.shape[0])
    prices, score = _scarcity_prices(table, matrix), []
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


def _greedy_bundle(table: CandidateTable, target: float, power: ExpectedPower):
    matrix, selected = csc_matrix(table.resources), set()
    sessions, usage, gain = set(), np.zeros(table.resources.shape[0]), 0.0
    state = ExpectedPower(power.scenario, power.profile, power.case.case_id)
    while gain < target - 1e-8:
        choices, cheapest = [], {}
        for i, c in enumerate(table.candidates):
            if c.session in sessions:
                continue
            sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
            rows, values = matrix.indices[sl], matrix.data[sl]
            if np.any(usage[rows] + values > 1 + 1e-8):
                continue
            cost = (values / (1 - usage[rows])).sum()
            choices.append((i,))
            if c.session not in cheapest or (cost, i) < cheapest[c.session]:
                cheapest[c.session] = cost, i
        groups = {}
        for cost, i in cheapest.values():
            session = table.sessions[table.candidates[i].session]
            groups.setdefault(session.source_instance, []).append(
                (power.ell[session.session_id] / cost, i)
            )
        for group in groups.values():
            ordered = [i for _, i in sorted(group, reverse=True)]
            choices.extend(tuple(ordered[:n]) for n in range(2, min(3, len(ordered)) + 1))
            if len(ordered) > 3:
                choices.append(tuple(ordered))
        best = None
        for choice in choices:
            added = np.asarray(matrix[:, choice].sum(1)).ravel()
            if np.any(usage + added > 1 + 1e-8):
                continue
            ids = [table.sessions[table.candidates[i].session].session_id for i in choice]
            cost = sum(value / (1 - usage[row])
                       for row, value in enumerate(added) if value)
            key = (state.drain_gain(ids) / cost, tuple(-i for i in choice))
            if best is None or key > best[0]:
                best = key, choice, ids, added
        if best is None:
            break
        before = state.power(True)
        for session_id in best[2]:
            state.remove(session_id)
        gain += before - state.power(True)
        selected.update(best[1])
        sessions.update(table.candidates[i].session for i in best[1])
        usage += best[3]
    return selected


def _coupled_gain(table, power, pattern, cache=None):
    sessions = frozenset(table.candidates[i].session for i in pattern)
    if cache is None or sessions not in cache:
        value = power.drain_gain([
            table.sessions[session].session_id for session in sessions
        ])
        if cache is not None:
            cache[sessions] = value
    return value if cache is None else cache[sessions]


def _source_removed_gain(power, source, removed_load):
    owned = power.instance_slots[source]
    share = removed_load / len(owned)
    if power.profile.power_scope == "gpu":
        return sum(
            power.slot_power[node][slot]
            - power.case.power_curve.power(power.slots[node][slot] - share)
            for node, slot in owned if power.nodes[node].local
        )
    slots = {}
    for node, slot in owned:
        if power.nodes[node].local:
            slots.setdefault(node, list(power.slots[node]))[slot] -= share
    return sum(
        power.node_power[node] - power._power(node, values, "awake")
        for node, values in slots.items()
    )


def _coupled_source_patterns(table, power, priced, eta, scale, cache=None):
    ordered = sorted(priced, key=lambda row: (
        row[1] / power.ell[table.sessions[table.candidates[row[0]].session].session_id],
        row[0],
    ))
    if not ordered:
        return (), ()
    source = table.sessions[table.candidates[ordered[0][0]].session].source_instance
    best, price, removed = (0.0, 0), 0.0, 0.0
    for k, (i, value) in enumerate(ordered, 1):
        session = table.sessions[table.candidates[i].session]
        if session.source_instance != source:
            raise ValueError("prefix pricing crossed a source power domain")
        price += value
        removed += power.ell[session.session_id]
        score = price - eta * _source_removed_gain(
            power, source, removed,
        ) / scale
        if score < best[0]:
            best = score, k
    order = tuple(i for i, _value in ordered)
    return order[:best[1]], order


def _coupled_source_pattern(table, power, priced, eta, scale):
    return _coupled_source_patterns(table, power, priced, eta, scale)[0]


def _retained_prefixes(pattern, ordered, buckets=None):
    if not ordered:
        return {pattern}
    buckets = DUAL_PREFIX_BUCKETS if buckets is None else buckets
    if buckets < 1:
        raise ValueError("prefix recovery needs a positive bucket count")
    n, k = len(ordered), len(pattern)
    sizes = {1, n, max(1, k - 1), k, min(n, k + 1)}
    sizes.update(
        -(-n * step // buckets) for step in range(1, buckets)
    )
    return {ordered[:size] for size in sizes}


def _dual_resource_limits(table):
    limits = np.ones(table.resources.shape[0])
    matrix = csr_matrix(table.resources)
    for row, name in enumerate(table.resource_names):
        if not name.startswith("route:"):
            continue
        columns = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        values = matrix.data[matrix.indptr[row]:matrix.indptr[row + 1]]
        tail = max((
            table.candidates[i].duration_s
            - value * table.migration_horizon_s
            for i, value in zip(columns, values)
        ), default=0.0)
        limits[row] = max(0.0, 1 - tail / table.migration_horizon_s)
    return limits


def _coupled_source_space(
    table, power, members, by_session, matrix, gain_cache,
):
    choices = [tuple(by_session.get(session, ())) for session in members
               if power.ell[table.sessions[session].session_id] > 0]
    count = 1
    for actions in choices:
        count *= len(actions) + 1
        if count > MAX_EXACT_COUPLED_PATTERNS:
            return None
    columns = {
        i: np.asarray(matrix[:, i].todense()).ravel()
        for actions in choices for i in actions
    }
    states = [((), 0.0, np.zeros(matrix.shape[0]))]
    for actions in choices:
        prior = states
        states = list(prior)
        for pattern, work, usage in prior:
            for i in actions:
                next_usage = usage + columns[i]
                if np.all(next_usage <= 1 + 1e-8):
                    states.append((
                        pattern + (i,),
                        work + table.candidates[i].migration_work_s,
                        next_usage,
                    ))
    patterns = tuple(row[0] for row in states)
    return (
        patterns,
        np.asarray([row[1] for row in states]),
        np.asarray([row[2] for row in states]),
        np.asarray([
            _coupled_gain(table, power, pattern, gain_cache)
            for pattern in patterns
        ]),
    )


def _recover_coupled(
    table, power, patterns, target, architecture, scenario, mode, gain_cache=None,
    eager_pack=False, return_assignment=False, resource_limits=None,
):
    matrix, selected, chosen = csc_matrix(table.resources), set(), {}
    limits = np.ones(matrix.shape[0]) if resource_limits is None else resource_limits
    gain, blocked, cache = 0.0, set(), {}
    usage = np.zeros(matrix.shape[0])
    sources = sorted(patterns)
    rank = {source: i for i, source in enumerate(sources)}
    pattern_ids, multipliers, multiplier = {}, {}, 1
    for source in sources:
        pattern_ids[source] = {
            pattern: i for i, pattern in enumerate(sorted(patterns[source]), 1)
        }
        multipliers[source], multiplier = multiplier, multiplier * (
            len(pattern_ids[source]) + 1
        )
    state, visited = 0, {0}
    versions = dict.fromkeys(sources, 0)
    heap, deferred, assignment = [], [], None

    def stats(pattern):
        if pattern not in cache:
            sessions = {table.candidates[i].session for i in pattern}
            sources = {
                table.sessions[session].source_instance for session in sessions
            }
            cache[pattern] = (
                _source_removed_gain(
                    power, next(iter(sources)), sum(
                        power.ell[table.sessions[session].session_id]
                        for session in sessions
                    ),
                ) if len(sources) == 1 else _coupled_gain(
                    table, power, pattern, gain_cache,
                ),
                sum(table.candidates[i].migration_work_s for i in pattern),
                set(pattern),
                np.asarray(matrix[:, list(pattern)].sum(1)).ravel()
                if pattern else np.zeros(matrix.shape[0]),
            )
        return cache[pattern]

    def entry(source, pattern):
        old = chosen.get(source, ())
        if not pattern or pattern == old or (source, old, pattern) in blocked:
            return None
        old_value, old_work, _members, _usage = stats(old)
        value, work, _members, _usage = stats(pattern)
        if value < old_value - 1e-8:
            return None
        added = value - old_value
        ratio = min(added, target - gain) / max(work - old_work, 1e-12)
        return (
            -ratio, work, tuple(pattern) + (np.inf,), rank[source], versions[source],
            source, old, pattern, added,
        )

    def add(source):
        for pattern in sorted(patterns[source]):
            item = entry(source, pattern)
            if item is not None:
                heappush(heap, item)

    for source in sources:
        add(source)
    while gain < target - 1e-8:
        best = None
        while heap:
            item = heappop(heap)
            source, old, pattern = item[5:8]
            if item[4] != versions[source] or chosen.get(source, ()) != old:
                continue
            current = entry(source, pattern)
            if current is None:
                continue
            if current[:4] != item[:4]:
                heappush(heap, current)
                continue
            old_members, old_usage = stats(old)[2:]
            members, pattern_usage = stats(pattern)[2:]
            trial_state = state + (
                pattern_ids[source][pattern] - pattern_ids[source].get(old, 0)
            ) * multipliers[source]
            if trial_state in visited:
                deferred.append(item)
                continue
            trial_usage = usage - old_usage + pattern_usage
            if np.any(trial_usage > limits + 1e-8):
                deferred.append(item)
                continue
            best = source, old, pattern, item[8], trial_usage, trial_state
            break
        if best is None:
            break
        if eager_pack:
            assignment = _pack(
                table, selected - stats(best[1])[2] | stats(best[2])[2],
                architecture, scenario, mode,
            )[0]
            if assignment is None:
                blocked.add(best[:3])
                continue
        selected.difference_update(stats(best[1])[2])
        selected.update(stats(best[2])[2])
        chosen[best[0]], state = best[2], best[5]
        visited.add(state)
        usage = best[4]
        gain += best[3]
        versions[best[0]] += 1
        if best[3] < 0:
            heap, deferred = [], []
            for source in sources:
                add(source)
            continue
        for item in deferred:
            if item[4] == versions[item[5]]:
                current = entry(item[5], item[7])
                if current is not None:
                    heappush(heap, current)
        deferred = []
        add(best[0])
    if not eager_pack and gain >= target - 1e-8:
        assignment = _pack(table, selected, architecture, scenario, mode)[0]
    if not eager_pack and (gain < target - 1e-8 or assignment is None):
        return _recover_coupled(
            table, power, patterns, target, architecture, scenario, mode,
            gain_cache, True, return_assignment, limits,
        )
    exact_usage = np.asarray(matrix[:, list(selected)].sum(1)).ravel() \
        if selected else np.zeros(matrix.shape[0])
    if np.any(exact_usage > limits + 1e-8):
        raise RuntimeError("coupled recovery exceeded an aggregate resource")
    return (selected, assignment) if return_assignment else selected


def _greedy_coupled(
    table, target, power, architecture, scenario, mode, enumerate_small=True,
    iterations=None, return_assignment=False, resource_limits=None,
    prefix_buckets=None,
):
    """Simulator-local emulation of source-local choices under shared prices."""
    iterations = COUPLED_PRICE_ITERATIONS if iterations is None else iterations
    if iterations < 1:
        raise ValueError("dual pricing needs a positive iteration budget")
    matrix = csc_matrix(table.resources)
    limits = np.ones(matrix.shape[0]) \
        if resource_limits is None else resource_limits
    by_source, by_session = {}, {}
    for j, session in enumerate(table.sessions):
        by_source.setdefault(session.source_instance, []).append(j)
    if not enumerate_small:
        for members in by_source.values():
            members.sort(key=lambda j: table.sessions[j].session_id)
    for i, candidate in enumerate(table.candidates):
        by_session.setdefault(candidate.session, []).append(i)
    prices = _scarcity_prices(table, matrix)
    eta, scale = 1.0, max(target, 1.0)
    gain_cache = {}
    retained = {source: set() for source in by_source}
    spaces = {
        source: _coupled_source_space(
            table, power, members, by_session, matrix, gain_cache,
        ) if enumerate_small else None
        for source, members in by_source.items()
    }
    for source, space in spaces.items():
        if space is not None:
            patterns, _work, _usage, gains = space
            best = gains.max()
            retained[source].update(
                pattern for pattern, gain in zip(patterns, gains)
                if gain >= best - 1e-8
            )
    for iteration in range(iterations):
        selected, chosen = set(), []
        for source_order, (source, members) in enumerate(sorted(by_source.items())):
            space = spaces[source]
            if space is not None:
                patterns, work, usage, gains = space
                score = work / table.migration_horizon_s \
                    + usage @ prices - eta * gains / scale
                pattern = patterns[int(np.argmin(score))]
            else:
                priced, alternate = [], []
                for position, session in enumerate(members):
                    if power.ell[table.sessions[session].session_id] <= 0:
                        continue
                    actions = []
                    for i in by_session.get(session, ()):
                        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
                        rows, values = matrix.indices[sl], matrix.data[sl]
                        value = table.candidates[i].migration_work_s \
                            / table.migration_horizon_s + prices[rows] @ values
                        actions.append((value, i))
                    if not actions:
                        continue
                    actions.sort()
                    low = min(value for value, _i in actions)
                    ties = sorted(
                        i for value, i in actions if abs(value - low) <= 1e-12
                    )
                    i = ties[(iteration + source_order + position) % len(ties)]
                    priced.append((i, low))
                    if not enumerate_small:
                        value, i = actions[
                            (iteration + source_order + position) % min(2, len(actions))
                        ]
                        alternate.append((i, value))
                pattern, ordered = _coupled_source_patterns(
                    table, power, priced, eta, scale, gain_cache,
                )
                retained[source].update(_retained_prefixes(
                    pattern, ordered, prefix_buckets,
                ))
                if alternate:
                    alternative, ordered = _coupled_source_patterns(
                        table, power, alternate, eta, scale, gain_cache,
                    )
                    retained[source].update(
                        _retained_prefixes(alternative, ordered, prefix_buckets)
                    )
            retained[source].add(pattern)
            selected.update(pattern)
            chosen.append(pattern)
        usage = np.asarray(table.resources[:, list(selected)].sum(1)).ravel() \
            if selected else np.zeros(table.resources.shape[0])
        shed = sum(
            _coupled_gain(table, power, pattern, gain_cache) for pattern in chosen
        )
        step = .5 / np.sqrt(iteration + 1)
        prices = np.maximum(0, prices + step * (usage - limits))
        eta = max(0, eta + step * (target - shed) / scale)
    return _recover_coupled(
        table, power, retained, target, architecture, scenario, mode, gain_cache,
        return_assignment=return_assignment,
        resource_limits=limits,
    )


def _greedy_prefix(
    table, target, power, architecture, scenario, mode, return_assignment=False,
):
    maximum = power.drain_gain(
        session.session_id for session in table.sessions
    )
    high = maximum > 0 and target >= DUAL_HIGH_TARGET_FRACTION * maximum
    return _greedy_coupled(
        table, target, power, architecture, scenario, mode, False,
        iterations=(DUAL_HIGH_TARGET_ITERATIONS if high
                    else DUAL_PRICE_ITERATIONS),
        return_assignment=return_assignment,
        resource_limits=_dual_resource_limits(table),
        prefix_buckets=(DUAL_HIGH_TARGET_BUCKETS if high
                        else DUAL_PREFIX_BUCKETS),
    )


def _round_lp(table, target, values):
    n = len(table.candidates)
    gains = np.array([c.gain_w for c in table.candidates])
    work = np.array([c.migration_work_s for c in table.candidates])
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


def _lp(table: CandidateTable, target: float, stats=None):
    if not table.candidates:
        return set()
    started, native_s, solves, iterations = perf_counter(), 0.0, 0, 0
    n, x = len(table.candidates), cp.Variable(len(table.candidates), nonneg=True)
    gains = np.array([c.gain_w for c in table.candidates])
    work = np.array([c.migration_work_s for c in table.candidates])
    base = [table.incidence @ x <= 1, table.resources @ x <= 1, x <= 1]
    def solve(objective, constraints, maximize=False):
        nonlocal native_s, solves, iterations
        p = cp.Problem(cp.Maximize(objective) if maximize else cp.Minimize(objective), constraints)
        p.solve(solver=cp.CLARABEL)
        native_s += p.solver_stats.solve_time or 0
        iterations += p.solver_stats.num_iters or 0
        solves += 1
        return p
    problem = solve(work @ x, base + [gains @ x >= target])
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        problem = solve(gains @ x, base, True)
        best = float(problem.value)
        problem = solve(work @ x, base + [gains @ x >= best - 1e-7])
    selected = _round_lp(table, target, np.asarray(x.value))
    if stats is not None:
        stats.update(wall_s=perf_counter() - started, native_s=native_s,
                     solves=solves, iterations=iterations)
    return selected


def _lp_highs(table: CandidateTable, target: float, stats=None):
    if not table.candidates:
        return set()
    started, native_s, solves, iterations = perf_counter(), 0.0, 0, 0
    gains = np.array([c.gain_w for c in table.candidates])
    work = np.array([c.migration_work_s for c in table.candidates])
    if not np.array_equal(np.asarray(table.incidence.sum(0)).ravel(),
                          np.ones(len(table.candidates))):
        raise ValueError("each LP candidate must belong to exactly one session")
    common = vstack((table.incidence, table.resources), format="csr")
    common_rhs = np.ones(common.shape[0])
    gain_scale = max(float(target), float(gains.max(initial=0)), 1.0)
    target_row = csr_matrix((-gains / gain_scale).reshape(1, -1))

    def solve(objective, minimum=None):
        nonlocal native_s, solves, iterations
        matrix, rhs = common, common_rhs
        if minimum is not None:
            matrix = vstack((common, target_row), format="csr")
            rhs = np.append(common_rhs, -minimum / gain_scale)
        solve_started = perf_counter()
        result = linprog(objective, A_ub=matrix, b_ub=rhs,
                         bounds=(0, None), method="highs-ipm",
                         options={"presolve": True})
        native_s += perf_counter() - solve_started
        solves += 1
        iterations += result.nit
        if result.status not in (0, 2):
            raise RuntimeError(f"HiGHS failed: {result.message}")
        return result

    result = solve(work / table.migration_horizon_s, target)
    if result.status == 2:
        maximum = solve(-gains / gain_scale)
        if maximum.status:
            raise RuntimeError(f"HiGHS maximum-gain LP failed: {maximum.message}")
        best = -float(maximum.fun) * gain_scale
        result = solve(work / table.migration_horizon_s, best - 1e-7)
    if result.status:
        raise RuntimeError(f"HiGHS target LP failed: {result.message}")
    selected = _round_lp(table, target, result.x)
    if stats is not None:
        stats.update(wall_s=perf_counter() - started, native_s=native_s,
                     solves=solves, iterations=iterations)
    return selected


def _column_phase(table, target, costs, active, shortfall, stats):
    sessions = np.array([c.session for c in table.candidates])
    gains = np.array([c.gain_w for c in table.candidates])
    active = set(active)
    batch = max(256, table.incidence.shape[0] // COLUMN_GROWTH_SWEEPS)
    for sweep in range(len(table.candidates) + 1):
        columns = np.array(sorted(active), int)
        represented = np.unique(sessions[columns]) if columns.size else np.array([], int)
        incidence = table.incidence[represented][:, columns]
        resources = table.resources[:, columns]
        target_row = csr_matrix((-gains[columns]).reshape(1, -1))
        matrix = vstack((incidence, resources, target_row), format="csr")
        objective = costs[columns]
        if shortfall:
            matrix = hstack((matrix, csr_matrix((
                [-1.0], ([matrix.shape[0] - 1], [0])),
                shape=(matrix.shape[0], 1),
            )), format="csr")
            objective = np.append(objective, 1)
        result = linprog(
            objective, A_ub=matrix,
            b_ub=np.concatenate((
                np.ones(len(represented) + table.resources.shape[0]), [-target],
            )), bounds=(0, None), method="highs-ds", options={"presolve": True},
        )
        if result.status:
            raise RuntimeError(f"column master failed: {result.message}")
        dual = -result.ineqlin.marginals
        alpha = np.zeros(table.incidence.shape[0])
        alpha[represented] = dual[:len(represented)]
        resource_dual = dual[len(represented):-1]
        eta = dual[-1]
        base = costs + table.resources.T @ resource_dual - eta * gains
        reduced = np.asarray(base).ravel() + alpha[sessions]
        minimum = np.full(table.incidence.shape[0], np.inf)
        np.minimum.at(minimum, sessions, reduced)
        best = np.full(table.incidence.shape[0], len(table.candidates))
        tied = reduced == minimum[sessions]
        np.minimum.at(best, sessions[tied], np.flatnonzero(tied))
        violations = best[minimum < -COLUMN_TOLERANCE]
        violations = violations[~np.isin(violations, columns, assume_unique=True)]

        correction = np.maximum(0, -alpha - minimum)
        correction[~np.isfinite(correction)] = 0
        lower = eta * target - resource_dual.sum() - (alpha + correction).sum()
        stats.update(sweeps=sweep + 1, columns=len(active), upper=float(result.fun),
                     lower=float(lower), gap=float(result.fun - lower))
        if not violations.size:
            return result, columns, active
        order = np.lexsort((violations, reduced[violations]))
        active.update(map(int, violations[order[:batch]]))
    raise RuntimeError("column generation did not converge")


def _lp_column_generation(table: CandidateTable, target: float, stats=None):
    if not table.candidates or target <= 0:
        return set()
    started, phase1, phase2 = perf_counter(), {}, {}
    zeros = np.zeros(len(table.candidates))
    first, columns, active = _column_phase(
        table, target, zeros, set(), True, phase1,
    )
    shortfall = float(first.x[-1])
    effective = max(0.0, target - shortfall)
    work = np.array([c.migration_work_s for c in table.candidates]) \
        / table.migration_horizon_s
    second, columns, active = _column_phase(
        table, max(0.0, effective - 1e-7), work, active, False, phase2,
    )
    values = np.zeros(len(table.candidates))
    values[columns] = second.x
    selected = _round_lp(table, target, values)
    if stats is not None:
        stats.update(wall_s=perf_counter() - started,
                     active_columns=len(active),
                     active_sessions=len({table.candidates[i].session for i in active}),
                     phase1_shortfall=shortfall, effective_target=effective,
                     phase1=phase1, phase2=phase2)
    return selected


def _add_priced_columns(highs, table, choices, costs, candidate_columns,
                        session_rows):
    choices = np.asarray(choices, dtype=np.int32)
    sessions = np.array([table.candidates[i].session for i in choices], np.int32)
    new_sessions = np.unique(sessions[session_rows[sessions] < 0])
    if new_sessions.size:
        first = highs.getNumRow()
        status = highs.addRows(
            len(new_sessions), np.full(len(new_sessions), -highspy.kHighsInf),
            np.ones(len(new_sessions)), 0,
            np.zeros(len(new_sessions) + 1, np.int32),
            np.array([], np.int32), np.array([], float),
        )
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS failed to add session rows")
        session_rows[new_sessions] = np.arange(first, first + len(new_sessions))
    matrix, starts, indices, values = csc_matrix(table.resources), [0], [], []
    target_row = table.resources.shape[0]
    for choice, session in zip(choices, sessions):
        sl = slice(matrix.indptr[choice], matrix.indptr[choice + 1])
        indices.extend(matrix.indices[sl])
        values.extend(matrix.data[sl])
        indices.extend((target_row, session_rows[session]))
        values.extend((table.candidates[choice].gain_w, 1.0))
        starts.append(len(indices))
    first = highs.getNumCol()
    status = highs.addCols(
        len(choices), costs[choices], np.zeros(len(choices)),
        np.full(len(choices), highspy.kHighsInf), len(indices),
        np.asarray(starts, np.int32), np.asarray(indices, np.int32),
        np.asarray(values, float),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to add priced columns")
    candidate_columns[choices] = np.arange(first, first + len(choices))


def _persistent_column_phase(highs, table, target, costs, candidate_columns,
                             session_rows, stats):
    sessions = np.array([c.session for c in table.candidates])
    gains = np.array([c.gain_w for c in table.candidates])
    batch = max(256, table.incidence.shape[0] // COLUMN_GROWTH_SWEEPS)
    pricing_s = add_s = solve_s = 0.0
    iterations = 0
    for sweep in range(len(table.candidates) + 1):
        started = perf_counter()
        highs.run()
        solve_s += perf_counter() - started
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"persistent master failed: {highs.getModelStatus()}")
        solution, info = highs.getSolution(), highs.getInfo()
        iterations += info.simplex_iteration_count
        row_dual = np.asarray(solution.row_dual)
        resource_dual = -row_dual[:table.resources.shape[0]]
        eta = row_dual[table.resources.shape[0]]
        alpha = np.zeros(table.incidence.shape[0])
        represented = np.flatnonzero(session_rows >= 0)
        alpha[represented] = -row_dual[session_rows[represented]]

        started = perf_counter()
        reduced = np.asarray(
            costs + table.resources.T @ resource_dual - eta * gains,
        ).ravel() + alpha[sessions]
        minimum = np.full(table.incidence.shape[0], np.inf)
        np.minimum.at(minimum, sessions, reduced)
        best = np.full(table.incidence.shape[0], len(table.candidates))
        tied = reduced == minimum[sessions]
        np.minimum.at(best, sessions[tied], np.flatnonzero(tied))
        violations = best[minimum < -COLUMN_TOLERANCE]
        violations = violations[candidate_columns[violations] < 0]
        correction = np.maximum(0, -alpha - minimum)
        correction[~np.isfinite(correction)] = 0
        lower = eta * target - resource_dual.sum() - (alpha + correction).sum()
        pricing_s += perf_counter() - started
        upper = float(info.objective_function_value)
        stats.update(sweeps=sweep + 1,
                     columns=int(np.count_nonzero(candidate_columns >= 0)),
                     upper=upper, lower=float(lower), gap=upper - lower,
                     pricing_s=pricing_s, add_s=add_s, solve_s=solve_s,
                     simplex_iterations=iterations)
        if not violations.size:
            return solution
        order = np.lexsort((violations, reduced[violations]))
        choices = violations[order[:batch]]
        started = perf_counter()
        _add_priced_columns(
            highs, table, choices, costs, candidate_columns, session_rows,
        )
        add_s += perf_counter() - started
        highs.setOptionValue("presolve", "off")
    raise RuntimeError("persistent column generation did not converge")


def _lp_column_generation_persistent(table: CandidateTable, target: float, stats=None):
    if not table.candidates or target <= 0:
        return set()
    started, phase1, phase2 = perf_counter(), {}, {}
    highs, resources = highspy.Highs(), table.resources.shape[0]
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    highs.addRows(
        resources + 1,
        np.concatenate((np.full(resources, -highspy.kHighsInf), [target])),
        np.concatenate((np.ones(resources), [highspy.kHighsInf])), 0,
        np.zeros(resources + 2, np.int32), np.array([], np.int32),
        np.array([], float),
    )
    highs.addCol(1, 0, highspy.kHighsInf, 1, np.array([resources], np.int32),
                 np.array([1.0]))
    candidate_columns = np.full(len(table.candidates), -1, np.int32)
    session_rows = np.full(table.incidence.shape[0], -1, np.int32)
    first = _persistent_column_phase(
        highs, table, target, np.zeros(len(table.candidates)),
        candidate_columns, session_rows, phase1,
    )
    shortfall = float(first.col_value[0])
    effective = max(0.0, target - shortfall)
    work = np.array([c.migration_work_s for c in table.candidates]) \
        / table.migration_horizon_s
    active = np.flatnonzero(candidate_columns >= 0)
    highs.changeColsCost(
        len(active), candidate_columns[active], work[active],
    )
    highs.changeColBounds(0, 0, 0)
    phase2_target = max(0.0, effective - 1e-7)
    highs.changeRowBounds(resources, phase2_target, highspy.kHighsInf)
    second = _persistent_column_phase(
        highs, table, phase2_target, work, candidate_columns, session_rows, phase2,
    )
    values = np.zeros(len(table.candidates))
    active = np.flatnonzero(candidate_columns >= 0)
    values[active] = np.asarray(second.col_value)[candidate_columns[active]]
    selected = _round_lp(table, target, values)
    if stats is not None:
        stats.update(wall_s=perf_counter() - started, active_columns=len(active),
                     active_sessions=np.count_nonzero(session_rows >= 0),
                     phase1_shortfall=shortfall, effective_target=effective,
                     phase1=phase1, phase2=phase2)
    return selected


def _pack(table, selected, architecture, scenario, mode):
    horizon = architecture.residency_horizon_s
    horizon = scenario.end_s - scenario.controller_delay_s if horizon is None else horizon
    work, kv = _baseline(scenario, architecture, horizon)
    migration = {
        r.replica_id: {"replay": 0.0, "kv_transfer": 0.0}
        for p in architecture.pools for r in p.replicas
    }
    assignment = {}
    for p, pool in enumerate(architecture.pools):
        q = architecture.type_by_id[pool.type_id]
        normals, bounds = np.asarray(q.normals), _event_bounds(q, pool, mode)
        replicas = pool.replicas
        replica_work = np.asarray([work[r.replica_id] for r in replicas])
        replica_kv = np.asarray([kv[r.replica_id] for r in replicas])
        replica_migration = np.zeros((len(replicas), 2))
        method_column = {"replay": 0, "kv_transfer": 1}
        members = [i for i in selected if table.candidates[i].pool == p]
        members.sort(key=lambda i: (-max(
            *(normals @ table.candidates[i].service_work / bounds),
            table.candidates[i].kv_tokens
            / (q.kv_capacity_tokens // q.kv_block_tokens),
            table.candidates[i].duration_s / table.migration_horizon_s,
        ), table.sessions[table.candidates[i].session].session_id, i))
        for i in members:
            c = table.candidates[i]
            next_work = replica_work + c.service_work
            pressure = np.maximum.reduce((
                np.max(next_work @ normals.T / bounds, axis=1),
                (replica_kv + c.kv_tokens)
                / (q.kv_capacity_tokens // q.kv_block_tokens),
                np.maximum(
                    np.max(replica_migration, axis=1),
                    replica_migration[:, method_column[c.method]] + c.duration_s,
                ) / table.migration_horizon_s,
            ))
            feasible = np.flatnonzero(pressure <= 1 + 1e-9)
            if not feasible.size:
                return None, tuple(sorted(members))
            r = feasible[np.argmin(pressure[feasible])]
            replica_work[r] = next_work[r]
            replica_kv[r] += c.kv_tokens
            replica_migration[r, method_column[c.method]] += c.duration_s
            replica_id = replicas[r].replica_id
            work[replica_id], kv[replica_id] = replica_work[r], replica_kv[r]
            migration[replica_id][c.method] = replica_migration[r, method_column[c.method]]
            assignment[i] = replica_id
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
    migration = {
        r.replica_id: {"replay": 0.0, "kv_transfer": 0.0}
        for p in architecture.pools for r in p.replicas
    }
    pools = {r.replica_id: (architecture.type_by_id[p.type_id], p)
             for p in architecture.pools for r in p.replicas}
    for i, replica in assignment.items():
        work[replica] += table.candidates[i].service_work
        kv[replica] += table.candidates[i].kv_tokens
        candidate = table.candidates[i]
        migration[replica][candidate.method] += candidate.duration_s
    return all(np.all(np.asarray(q.normals) @ work[r] <= _event_bounds(q, pool, mode) + 1e-9)
               and kv[r] <= q.kv_capacity_tokens // q.kv_block_tokens
               and max(migration[r].values()) <= table.migration_horizon_s + 1e-9
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
    records = []
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
        spare = stable - baseline - ongoing
        records.extend(
            ServiceDebtUse(pool.pool_id, facet, float(debt[facet]),
                           float(spare[facet]), float(recovery[facet]))
            for facet in range(len(debt))
        )
    return tuple(records)


def _mode_plan(scenario, profile, architecture, solver, mode, power, target):
    table = candidate_table(scenario, profile, architecture, mode, power)
    assignment = None
    if solver == "greedy_prefix":
        selected, assignment = _greedy_prefix(
            table, target, power, architecture, scenario, mode, True,
        )
    elif solver == "greedy_coupled":
        selected, assignment = _greedy_coupled(
            table, target, power, architecture, scenario, mode,
            return_assignment=True,
        )
    else:
        selected = (_lp_column_generation_persistent(table, target)
                    if solver == "lp_column_generation_persistent" else
                    _lp_column_generation(table, target)
                    if solver == "lp_column_generation" else
                    _lp_highs(table, target) if solver == "lp_highs" else
                    _lp(table, target) if solver.startswith("lp") else
                    _greedy_bundle(table, target, power)
                    if solver == "greedy_bundle" else _greedy(table, target))
    repairs, repair_s = 0, 0.0
    costs = np.asarray(table.resources.sum(0)).ravel()
    while True:
        started = perf_counter()
        if assignment is None:
            assignment, cut = _pack(table, selected, architecture, scenario, mode)
        else:
            cut = None
        if assignment is not None:
            return table, selected, assignment, repairs, repair_s
        if not cut:
            raise RuntimeError("destination packing repair did not converge")
        drop = max(cut, key=lambda i: (
            costs[i] / max(table.candidates[i].gain_w, 1e-12), i,
        ))
        selected.remove(drop)
        assignment = None
        repair_s += perf_counter() - started
        repairs += 1


def plan_destination(scenario, profile, solver, case_id, seed, architecture):
    if solver not in {"greedy", "greedy_bundle", "greedy_prefix", "greedy_coupled",
                      "lp", "lp_peak_first", "lp_work_first", "lp_highs",
                      "lp_column_generation", "lp_column_generation_persistent"}:
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
    expected = predict(
        _expected_scenario(scenario, moves), profile, moves, case_id, architecture,
    )
    shortfall = max(0.0, planned - scenario.power_limit_w)
    shortfall = 0.0 if shortfall <= 1e-8 else shortfall
    debt_rows = _selected_service_debt(
        table, selected, architecture, scenario,
    )
    debt = max((row.debt_replica_s for row in debt_rows), default=0.0)
    recovery = max((row.recovery_s for row in debt_rows), default=0.0)
    feasible = shortfall == 0 and expected.deadline_met and np.isfinite(recovery)
    failure = (
        None if feasible else "target_unmet" if shortfall
        else "service_debt_unrecoverable" if not np.isfinite(recovery)
        else "destination_service_queue" if any(
            not row.within_contract for row in expected.pool_service
        )
        else "migration_deadline"
    )
    usage = np.asarray(table.resources[:, list(selected)].sum(1)).ravel() if selected else np.zeros(len(table.resource_names))
    bottleneck = table.resource_names[int(usage.argmax())] if usage.size and usage.max() else None
    binding = tuple(
        name for name, value in zip(table.resource_names, usage)
        if value >= 1 - 1e-7
    )
    resources = tuple(
        ResourceUse(name, unit, float(value * capacity), capacity, float(value))
        for name, unit, capacity, value in zip(
            table.resource_names, table.resource_units,
            table.resource_capacities, usage,
        )
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
        solver=solver, moves=moves, initial_source_power_w=initial,
        planned_source_power_w=planned,
        expected_source_power_at_deadline_w=expected.modeled_source_power_at_deadline_w,
        feasible=feasible, solve_s=perf_counter() - start,
        profile_id=profile.profile_id, profile_case=case_id, seed=seed,
        kv_capacity_tokens=sum(
            len(p.replicas) * architecture.type_by_id[p.type_id].kv_capacity_tokens
            for p in architecture.pools
        ),
        lp_power_shortfall_w=shortfall, admission_mode=mode,
        power_shortfall_w=shortfall, failure_reason=failure,
        packing_repair_count=repairs, packing_repair_s=repair_s,
        predicted_migration_makespan_s=makespan, bottleneck=bottleneck,
        planner_memory_bytes=memory, service_debt_replica_s=debt,
        required_recovery_s=recovery, binding_resources=binding,
        resource_uses=resources, service_debts=debt_rows,
    )
