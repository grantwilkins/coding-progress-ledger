"""Pool-aware destination admission and deterministic replica packing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter
from types import SimpleNamespace

import cvxpy as cp
import highspy
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack

from destination import DestinationArchitecture
from planner import (_changes, _duration, _expected_scenario, _kv_catch_up_s,
                     _kv_schedule, _local_sessions, _log_bytes, _resident_tokens,
                     PlanResult, ResourceUse, ServiceDebtUse, source_power)
from power_model import ExpectedPower
from repair_controller import (Assignment, RepairMove, RepairRequest, RepairResult)
from simulate import (ExecutionScenario, PlannedMove, PoolServiceExecution,
                      SimSession, predict)


DUAL_PRICE_ITERATIONS = 1
DUAL_PREFIX_BUCKETS = 8
DUAL_HIGH_TARGET_ITERATIONS = 4
DUAL_HIGH_TARGET_BUCKETS = 64
DUAL_HIGH_TARGET_FRACTION = .75
COLUMN_GROWTH_SWEEPS = 20
COLUMN_TOLERANCE = 1e-8
COLUMN_GAP_TOLERANCE = 1e-7
NATIVE_PRICING_CHUNK = 65_536


@dataclass(frozen=True, slots=True)
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
    selection_credit: float | None = None

    @property
    def credit(self):
        return self.gain_w if self.selection_credit is None else self.selection_credit

    @property
    def objective_cost_s(self): return self.duration_s

    @property
    def replay_occupancy_s(self): return self.duration_s if self.method == "replay" else 0.0

    @property
    def kv_occupancy_s(self): return self.duration_s if self.method == "kv_transfer" else 0.0


@dataclass(frozen=True, slots=True)
class CandidateTable:
    sessions: tuple[SimSession, ...]
    candidates: tuple[Candidate, ...]
    incidence: csr_matrix
    resources: csr_matrix
    resource_names: tuple[str, ...]
    resource_capacities: tuple[float, ...]
    resource_units: tuple[str, ...]
    migration_horizon_s: float


@dataclass(frozen=True, slots=True)
class PricingSoA:
    gains: np.ndarray
    features: np.ndarray
    feasible: np.ndarray
    option_signatures: np.ndarray
    option_starts: np.ndarray
    resource_rows: np.ndarray
    resource_coefficients: np.ndarray
    session_ranks: np.ndarray


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
    bandwidth = min(links[link] for link in path)
    extrapolated = components.extrapolates(tokens, bandwidth)
    if extrapolated and not components.allow_extrapolation:
        raise ValueError(
            "migration candidate outside calibrated " + "/".join(extrapolated)
            + " range"
        )
    def route(size):
        return size / bandwidth
    if method == "replay":
        compute = tokens / case.replay.conservative_rate(tokens, 1) \
            + case.replay_completion_s * (
            1 + _changes(session, horizon)
        )
        return route(_log_bytes(session, tokens)) \
            + components.compute_completion_factor * compute + case.switch_s
    size = case.kv_transfer.sealed_bytes(tokens)
    ingest = components.kv_ingest_bytes_per_s \
        or case.kv_transfer.destination_bytes_per_s
    return max(route(size), size / ingest) \
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


def _candidate_oracle(scenario, profile, architecture, mode, power,
                      include_infeasible=False, selection_credits=None):
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
        return SimpleNamespace(
            sessions=sessions, migration_horizon_s=migration_horizon, specs=(),
            pools=architecture.pools, gains=(), marginal_gains=(), options=(), signatures=(),
            choices=lambda _: (), column=lambda _: (), feature=lambda _: (),
            templates=(),
        )
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
    marginal_gains = tuple(power.marginal(session.session_id) for session in sessions)
    gains = marginal_gains if selection_credits is None else tuple(selection_credits)
    if len(gains) != len(sessions):
        raise ValueError("selection credits must match planner sessions")
    grouped = {}
    for p, pool in enumerate(architecture.pools):
        q = types[pool.type_id]
        key = (
            q.type_id, len(pool.replicas), tuple(pool_work[p]), pool_kv[p],
            tuple(_event_bounds(q, pool, mode)), pool.route, pool.methods,
            None if pool.fluid_migration is None else (
                pool.fluid_migration.replay_speedup,
                pool.fluid_migration.kv_ingest_bytes_per_s,
                tuple(sorted(pool.fluid_migration.source_power_w.items())),
                tuple(sorted(pool.fluid_migration.destination_power_w.items())),
                pool.fluid_migration.provenance,
                pool.fluid_migration.coupling,
                pool.fluid_migration.route_overlap,
            ),
            tuple(sorted((pool.migration_headroom or {}).items())),
        )
        grouped.setdefault(key, []).append(p)
    pool_groups = tuple(map(tuple, grouped.values()))

    def records(j):
        session, values = sessions[j], []
        migration_tokens = _resident_tokens(session, migration_horizon)
        residency_tokens = _resident_tokens(session, residency_horizon)
        demand_cache, duration_cache = {}, {}
        route_bytes = {
            "replay": _log_bytes(session, migration_tokens),
            "kv_transfer": case.kv_transfer.sealed_bytes(migration_tokens),
        }
        for group in pool_groups:
            p, pool = group[0], architecture.pools[group[0]]
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
                duration_key = (
                    q.type_id, pool.route, method, rho, mode,
                    None if pool.fluid_migration is None else (
                        pool.fluid_migration.replay_speedup,
                        pool.fluid_migration.kv_ingest_bytes_per_s,
                        pool.fluid_migration.coupling,
                        pool.fluid_migration.route_overlap,
                        len(pool.replicas),
                    ),
                )
                if duration_key not in duration_cache:
                    components = None if q.migration is None else q.migration[method]
                    if components is None:
                        contexts = case.replay.by_concurrency[1][0]
                        unsupported = not contexts[0] <= migration_tokens <= contexts[-1]
                    else:
                        unsupported = bool(components.extrapolates(
                            migration_tokens, bandwidth,
                        )) and not components.allow_extrapolation
                    if unsupported:
                        duration_cache[duration_key] = None
                        continue
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
                    migration_work = duration
                    transition = None
                    if pool.fluid_migration:
                        service = pool.fluid_migration
                        replicas = len(pool.replicas)
                        if method == "replay":
                            stream_work = migration_tokens \
                                / case.replay.conservative_rate(migration_tokens, 1) \
                                / service.replay_speedup
                            tail_work = case.replay_completion_s * (
                                1 + _changes(session, migration_horizon)
                            ) / service.replay_speedup
                            transition = (stream_work + tail_work, 0)
                            try:
                                factor = q.loaded[method].worst(
                                    rho, rho, migration_tokens, bandwidth,
                                )
                            except ValueError:
                                duration_cache[duration_key] = None
                                continue
                            stream_work *= factor
                            tail_work *= factor
                            migration_work = stream_work + tail_work
                            duration = max(
                                route_bytes[method] / bandwidth,
                                stream_work / replicas,
                            ) + tail_work / replicas + case.switch_s
                        else:
                            tail_work = components.residual_s if components else \
                                case.kv_transfer.initial_completion_s
                            stream_work = route_bytes[method] \
                                / service.kv_ingest_bytes_per_s
                            try:
                                factor = q.loaded[method].worst(
                                    rho, rho, migration_tokens, bandwidth,
                                )
                            except ValueError:
                                duration_cache[duration_key] = None
                                continue
                            stream_work *= factor
                            tail_work *= factor
                            migration_work = stream_work + tail_work
                            duration = max(
                                route_bytes[method] / bandwidth,
                                stream_work / replicas,
                            ) + tail_work / replicas + case.switch_s
                        if service.coupling:
                            migration_work += case.switch_s
                            if not service.route_overlap:
                                migration_work += replicas * route_bytes[method] / bandwidth
                    if method == "kv_transfer":
                        try:
                            _kv_schedule(scenario, profile, session, case,
                                         pool.route, links)
                        except ValueError:
                            duration_cache[duration_key] = None
                            continue
                    if q.migration is None:
                        try:
                            duration *= q.loaded[method].worst(
                                rho, _mode_boundary_rho(q, mode),
                                session.context_tokens, bandwidth,
                            )
                        except ValueError:
                            duration_cache[duration_key] = None
                            continue
                    duration_cache[duration_key] = duration, migration_work, transition
                timed = duration_cache[duration_key]
                if timed is None:
                    continue
                duration, migration_work, transition = timed
                if duration > migration_horizon and not include_infeasible:
                    continue
                transition = transition or (
                    (max(0, duration - route_bytes[method] / bandwidth), 0)
                    if method == "replay" else (0, 0)
                )
                values.extend((
                    method, destination, duration, migration_work,
                    architecture.pools[destination].route, route_bytes[method],
                    tuple(demand), resident, transition,
                ) for destination in group)
        values.sort(key=lambda value: option_for[value[1], value[0]])
        return tuple(values)

    def choices(j):
        return tuple(Candidate(
            j, method, p, marginal_gains[j], migration_work, duration, route,
            route_bytes, demand, resident, transition,
            None if selection_credits is None else gains[j],
        ) for method, p, duration, migration_work, route, route_bytes, demand, resident, transition
                     in records(j))

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
            service = pool.fluid_migration
            capacity = len(pool.replicas) * migration_horizon \
                * (pool.migration_headroom or {}).get(method, 1)
            add(("migration", p, method), capacity,
                f"migration:{pool.pool_id}:{method}",
                "replica-s")

    options = sorted([
        (p, method) for p, pool in enumerate(architecture.pools)
        for method in pool.methods
        if types[pool.type_id].compatibility.supports(
            architecture.source_compatibility, method,
        )
    ], key=lambda item: (
        architecture.pools[item[0]].pool_id,
        0 if item[1] == "replay" else 1,
    ))
    signatures, signature_for = [], {}
    templates = []
    for p, method in options:
        pool, q = architecture.pools[p], types[architecture.pools[p].type_id]
        baseline = pool_work[p]
        rho = float(max(np.asarray(q.normals) @ baseline /
                        (len(pool.replicas) * np.asarray(q.bounds["normal"]))))
        signature = (q.type_id, pool.route, method, rho, mode)
        if signature not in signature_for:
            signature_for[signature] = len(signatures)
            signatures.append(signature)
        entries = []
        def emit(key, coefficients):
            row = row_for[key]
            entries.append((row, np.asarray(coefficients) / specs[row][0]))
        unit = np.eye(7)
        for link in dict.fromkeys(pool.route):
            emit(("route", link), unit[0])
        for facet, normal in enumerate(q.normals):
            ongoing = np.zeros(7)
            ongoing[1:3] = normal
            emit(("service", p, facet), ongoing)
            if pool.event_flex_fraction is not None:
                debt = np.zeros(7)
                debt[1:3], debt[5:7] = migration_horizon * np.asarray(normal), normal
                emit(("debt", p, facet), debt)
        emit(("kv", p), unit[3])
        emit(("migration", p, method), unit[4])
        service = pool.fluid_migration
        if service and service.coupling:
            other = "kv_transfer" if method == "replay" else "replay"
            if other in pool.methods:
                emit(("migration", p, other), service.coupling * unit[4])
        templates.append(tuple(entries))

    option_for = {option: i for i, option in enumerate(options)}
    option_signatures = tuple(
        signature_for[(
            types[architecture.pools[p].type_id].type_id,
            architecture.pools[p].route, method,
            float(max(np.asarray(types[architecture.pools[p].type_id].normals)
                      @ pool_work[p] / (len(architecture.pools[p].replicas)
                      * np.asarray(types[architecture.pools[p].type_id].bounds["normal"])))),
            mode,
        )] for p, method in options
    )

    def feature(candidate):
        return np.asarray((
            candidate.route_bytes, *candidate.service_work, candidate.kv_tokens,
            candidate.migration_work_s, *candidate.transition_work,
        ), float)

    def pricing(j):
        by_signature, mask = {}, 0
        for method, p, duration, migration_work, _route, route_bytes, demand, resident, transition \
                in records(j):
            option = option_for[p, method]
            signature = option_signatures[option]
            values = (route_bytes, *demand, resident, migration_work, *transition)
            if signature in by_signature and by_signature[signature] != values:
                raise ValueError("equivalent pricing signatures disagree")
            by_signature[signature] = values
            mask |= 1 << option
        return by_signature, mask

    def column(candidate):
        entries = []
        values = feature(candidate)
        for row, coefficients in templates[option_for[candidate.pool, candidate.method]]:
            value = float(coefficients @ values)
            if value:
                entries.append((row, value))
        return tuple(entries)

    return SimpleNamespace(
        sessions=sessions, migration_horizon_s=migration_horizon,
        pools=architecture.pools, gains=gains, marginal_gains=marginal_gains,
        specs=tuple(specs), choices=choices,
        column=column, feature=feature, options=tuple(options),
        pricing=pricing,
        signatures=tuple(signatures), option_signatures=option_signatures,
        templates=tuple(templates), option_for=option_for,
        pool_groups=pool_groups,
    )


def _pricing_chunk(oracle, start, stop):
    features = np.zeros((stop - start, len(oracle.signatures), 7))
    feasible = np.zeros(stop - start, np.uint16)
    for j in range(start, stop):
        local = j - start
        priced, mask = oracle.pricing(j)
        for signature, values in priced.items():
            features[local, signature] = values
        feasible[local] = mask
    return features, feasible


def _pricing_layout(oracle):
    starts, rows, coefficients = [0], [], []
    for template in oracle.templates:
        rows.extend(row for row, _ in template)
        coefficients.extend(value for _, value in template)
        starts.append(len(rows))
    return (np.asarray(oracle.option_signatures, np.uint16),
            np.asarray(starts, np.int32), np.asarray(rows, np.int32),
            np.asarray(coefficients))


def _pricing_ranks(oracle):
    order = sorted(range(len(oracle.sessions)),
                   key=lambda j: oracle.sessions[j].session_id)
    ranks = np.empty(len(order), np.uint32)
    ranks[order] = np.arange(len(order), dtype=np.uint32)
    return ranks


def _pricing_soa(oracle):
    if len(oracle.options) > 16:
        raise ValueError("native pricing supports at most 16 pool-method options")
    features, feasible = _pricing_chunk(oracle, 0, len(oracle.sessions))
    option_signatures, starts, rows, coefficients = _pricing_layout(oracle)
    return PricingSoA(
        np.asarray(oracle.gains), features, feasible,
        option_signatures, starts, rows, coefficients, _pricing_ranks(oracle),
    )


def _native_pricing_oracle(oracle):
    from _queue_haul_native import PricingOracle

    if len(oracle.options) > 16:
        raise ValueError("native pricing supports at most 16 pool-method options")
    option_signatures, starts, rows, coefficients = _pricing_layout(oracle)
    native = PricingOracle.allocate(
        len(oracle.sessions), len(oracle.signatures), len(oracle.options),
        len(oracle.specs), oracle.migration_horizon_s,
        option_signatures, starts, rows, np.ascontiguousarray(coefficients.ravel()),
    )
    ranks = _pricing_ranks(oracle)
    for start in range(0, len(oracle.sessions), NATIVE_PRICING_CHUNK):
        stop = min(start + NATIVE_PRICING_CHUNK, len(oracle.sessions))
        features, feasible = _pricing_chunk(oracle, start, stop)
        native.load(
            start, np.asarray(oracle.gains[start:stop]), features.ravel(),
            feasible, ranks[start:stop],
        )
    return native


def _materialize_candidates(oracle, candidates, prune=True):
    sessions, horizon = oracle.sessions, oracle.migration_horizon_s
    if horizon <= 0:
        return CandidateTable(sessions, (), csr_matrix((len(sessions), 0)),
                              csr_matrix((0, 0)), (), (), (), horizon)
    candidates = tuple(candidates)
    incidence = csr_matrix((np.ones(len(candidates)),
                            ([c.session for c in candidates], range(len(candidates)))),
                           shape=(len(sessions), len(candidates)))
    data, rr, cc = [], [], []
    for column, candidate in enumerate(candidates):
        for row, value in oracle.column(candidate):
            data.append(value)
            rr.append(row)
            cc.append(column)
    used = sorted(set(rr)) if prune else list(range(len(oracle.specs)))
    remap = {old: new for new, old in enumerate(used)}
    rr = [remap[row] for row in rr]
    capacities = tuple(oracle.specs[row][0] for row in used)
    names = tuple(oracle.specs[row][1] for row in used)
    units = tuple(oracle.specs[row][2] for row in used)
    return CandidateTable(sessions, candidates, incidence,
                          csr_matrix((data, (rr, cc)), shape=(len(used), len(candidates))),
                          names, capacities, units, horizon)


def candidate_table(scenario: ExecutionScenario, profile, architecture: DestinationArchitecture,
                    mode: str, power: ExpectedPower,
                    selection_credits=None) -> CandidateTable:
    oracle = _candidate_oracle(
        scenario, profile, architecture, mode, power,
        selection_credits=selection_credits,
    )
    return _materialize_candidates(oracle, (
        candidate for j in range(len(oracle.sessions))
        for candidate in oracle.choices(j)
    ))


def phase_one_capacity_duals(table: CandidateTable):
    """Maximize additive selection credit and return normalized capacity duals."""
    if not table.candidates:
        return 0.0, np.zeros(len(table.resource_names))
    gains = np.array([candidate.credit for candidate in table.candidates])
    matrix = vstack((table.incidence, table.resources), format="csr")
    result = linprog(-gains, A_ub=matrix, b_ub=np.ones(matrix.shape[0]),
                     bounds=(0, None), method="highs")
    if result.status:
        raise RuntimeError(f"HiGHS Phase-I LP failed: {result.message}")
    duals = -result.ineqlin.marginals[table.incidence.shape[0]:]
    return -float(result.fun), np.maximum(0, duals)


def _max_shed(table: CandidateTable, power: ExpectedPower):
    """Maximize exact awake-state shed for one source instance."""
    if not table.candidates:
        return set()
    session_ids = tuple(session.session_id for session in table.sessions)
    if len({power.route[session_ids[candidate.session]]
            for candidate in table.candidates}) != 1:
        raise ValueError("max_shed requires one source instance")
    loads = np.asarray([
        power.ell[session_ids[candidate.session]]
        for candidate in table.candidates
    ])
    matrix = vstack((table.incidence, table.resources), format="csr")
    base = LinearConstraint(matrix, -np.inf, 1)
    bounds, integer = Bounds(0, 1), np.ones(len(loads))

    def optimize(cost, constraints):
        result = milp(
            cost, integrality=integer, bounds=bounds, constraints=constraints,
            options={"mip_rel_gap": 0},
        )
        if not result.success:
            raise RuntimeError(f"maximum-shed MILP returned {result.message}")
        return result.x

    # Any optimizer returned by the primary MILP is an exact maximum-shed
    # reference.  A former second MILP minimized migration work while pinning
    # the floating-point optimum to within 1e-9.  That cosmetic tie-break can
    # take orders of magnitude longer to prove because HiGHS must reason about
    # an almost-exact dense equality; it does not change the reference shed.
    maximum = optimize(-loads, base)
    return set(np.flatnonzero(maximum > .5))


def _scarcity_prices(table, matrix, eligible=None):
    eligible = range(len(table.candidates)) if eligible is None else eligible
    cheapest = {}
    for i in eligible:
        c = table.candidates[i]
        a = matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]].sum()
        if c.session not in cheapest or (a, i) < cheapest[c.session]:
            cheapest[c.session] = (a, i)
    demand = np.zeros(table.resources.shape[0])
    for _, i in cheapest.values():
        demand[matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]] += \
            matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]]
    return np.maximum(demand, 1)


def _greedy(table: CandidateTable, target: float, eligible=None):
    matrix, selected, usage = csc_matrix(table.resources), set(), np.zeros(table.resources.shape[0])
    eligible = tuple(range(len(table.candidates))) if eligible is None else tuple(eligible)
    prices, score = _scarcity_prices(table, matrix, eligible), []
    for i in eligible:
        c = table.candidates[i]
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, values = matrix.indices[sl], matrix.data[sl]
        score.append((c.credit / max(values @ prices[rows], 1e-12), i))
    sessions, gain = set(), 0.0
    for _, i in sorted(score, key=lambda row: (-row[0], row[1])):
        c = table.candidates[i]
        if gain >= target - 1e-8:
            break
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, values = matrix.indices[sl], matrix.data[sl]
        if c.session in sessions or np.any(usage[rows] + values > 1 + 1e-8):
            continue
        selected.add(i)
        sessions.add(c.session)
        usage[rows] += values
        gain += c.credit
    return selected


def _baseline_policy(table: CandidateTable, target: float, policy: str, seed: int):
    if policy == "random":
        matrix, usage = csc_matrix(table.resources), np.zeros(table.resources.shape[0])
        rng, selected, gain = np.random.default_rng(seed), set(), 0.0
        choices = [[] for _ in range(table.incidence.shape[0])]
        for i, candidate in enumerate(table.candidates):
            choices[candidate.session].append(i)
        for session in rng.permutation(table.incidence.shape[0]):
            options = []
            for i in choices[int(session)]:
                column = matrix[:, i]
                if np.all(usage[column.indices] + column.data <= 1 + 1e-8):
                    options.append(i)
            if not options:
                continue
            i = options[int(rng.integers(len(options)))]
            column = matrix[:, i]
            rows, values = column.indices, column.data
            selected.add(i)
            usage[rows] += values
            gain += table.candidates[i].credit
            if gain >= target - 1e-8:
                break
        return selected
    eligible = [
        i for i, candidate in enumerate(table.candidates)
        if policy not in {"replay_only", "kv_only"}
        or candidate.method == ("replay" if policy == "replay_only" else "kv_transfer")
    ]
    if policy == "isolated_fastest":
        fastest = {}
        for i in eligible:
            candidate = table.candidates[i]
            key = (candidate.duration_s, candidate.migration_work_s, i)
            if candidate.session not in fastest or key < fastest[candidate.session][0]:
                fastest[candidate.session] = key, i
        eligible = [value[1] for value in fastest.values()]
    return _greedy(table, target, eligible)


def _lagrangian_gain(table, power, pattern, cache=None):
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
            - (power.case.phase_power.power(power.slots[node][slot] - share)
               if power.case.phase_power else
               power.case.power_curve.power(power.slots[node][slot] - share))
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


def _phase_power_target(power, sessions, target_w):
    """Convert one-source nonlinear watts into an exact additive load target."""
    if power.case.phase_power is None:
        return None, target_w, None
    session_ids = tuple(session.session_id for session in sessions)
    if not session_ids:
        return (), target_w, None
    sources = {power.route[session_id] for session_id in session_ids}
    if len(sources) != 1:
        raise ValueError("phase-aware additive planning requires one source instance")
    source = sources.pop()
    credits = tuple(power.ell[session_id] for session_id in session_ids)
    maximum_load = sum(credits)
    maximum_gain = _source_removed_gain(power, source, maximum_load)
    if target_w <= 0:
        required = 0.0
    elif target_w > maximum_gain + 1e-8:
        required = maximum_load + max(1.0, maximum_load)
    else:
        low, high = 0.0, maximum_load
        for _ in range(80):
            middle = (low + high) / 2
            if _source_removed_gain(power, source, middle) >= target_w:
                high = middle
            else:
                low = middle
        required = high
    return credits, required, source


def fractional_power_opportunity(table: CandidateTable, power: ExpectedPower):
    credits, _, source = _phase_power_target(power, table.sessions, 0)
    if credits is None:
        return phase_one_capacity_duals(table)[0]
    if source is None:
        return 0.0
    exact = replace(table, candidates=tuple(
        replace(candidate, selection_credit=credits[candidate.session])
        for candidate in table.candidates
    ))
    removed_load, _ = phase_one_capacity_duals(exact)
    return _source_removed_gain(power, source, min(removed_load, sum(credits)))


def _lagrangian_source_prefixes(table, power, priced, eta, scale):
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


def _lagrangian_source_prefix(table, power, priced, eta, scale):
    return _lagrangian_source_prefixes(table, power, priced, eta, scale)[0]


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


def _recover_lagrangian(
    table, power, patterns, target, architecture, scenario, mode, gain_cache=None,
    eager_pack=False, return_assignment=False, resource_limits=None, stats_cache=None,
):
    matrix, selected, chosen = csc_matrix(table.resources), set(), {}
    columns = tuple((matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]],
                     matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]])
                    for i in range(matrix.shape[1]))
    limits = np.ones(matrix.shape[0]) if resource_limits is None else resource_limits
    gain, blocked = 0.0, set()
    cache = {} if stats_cache is None else stats_cache
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
            pattern_usage = np.zeros(matrix.shape[0])
            for i in pattern:
                rows, values = columns[i]
                pattern_usage[rows] += values
            cache[pattern] = (
                _source_removed_gain(
                    power, next(iter(sources)), sum(
                        power.ell[table.sessions[session].session_id]
                        for session in sessions
                    ),
                ) if len(sources) == 1 else _lagrangian_gain(
                    table, power, pattern, gain_cache,
                ),
                sum(table.candidates[i].objective_cost_s for i in pattern),
                set(pattern),
                pattern_usage,
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
        return _recover_lagrangian(
            table, power, patterns, target, architecture, scenario, mode,
            gain_cache, True, return_assignment, limits, cache,
        )
    exact_usage = np.asarray(matrix[:, list(selected)].sum(1)).ravel() \
        if selected else np.zeros(matrix.shape[0])
    if np.any(exact_usage > limits + 1e-8):
        raise RuntimeError("Lagrangian recovery exceeded an aggregate resource")
    return (selected, assignment) if return_assignment else selected


def _greedy_lagrangian(
    table, target, power, architecture, scenario, mode, return_assignment=False,
):
    """Choose source-local prefixes under iterated aggregate-resource prices."""
    maximum = power.drain_gain(session.session_id for session in table.sessions)
    high = maximum > 0 and target >= DUAL_HIGH_TARGET_FRACTION * maximum
    iterations = DUAL_HIGH_TARGET_ITERATIONS if high else DUAL_PRICE_ITERATIONS
    prefix_buckets = DUAL_HIGH_TARGET_BUCKETS if high else DUAL_PREFIX_BUCKETS
    if iterations < 1:
        raise ValueError("dual pricing needs a positive iteration budget")
    matrix = csc_matrix(table.resources)
    limits = _dual_resource_limits(table)
    by_source, by_session = {}, {}
    for j, session in enumerate(table.sessions):
        by_source.setdefault(session.source_instance, []).append(j)
    for members in by_source.values():
        members.sort(key=lambda j: table.sessions[j].session_id)
    for i, candidate in enumerate(table.candidates):
        by_session.setdefault(candidate.session, []).append(i)
    prices = _scarcity_prices(table, matrix)
    eta, scale = 1.0, max(target, 1.0)
    gain_cache = {}
    retained = {source: set() for source in by_source}
    for iteration in range(iterations):
        selected, chosen = set(), []
        for source_order, (source, members) in enumerate(sorted(by_source.items())):
            priced, alternate = [], []
            for position, session in enumerate(members):
                if power.ell[table.sessions[session].session_id] <= 0:
                    continue
                actions = []
                for i in by_session.get(session, ()):
                    sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
                    rows, values = matrix.indices[sl], matrix.data[sl]
                    value = table.candidates[i].objective_cost_s \
                        / table.migration_horizon_s + prices[rows] @ values
                    actions.append((value, i))
                if not actions:
                    continue
                actions.sort()
                low = min(value for value, _i in actions)
                ties = sorted(i for value, i in actions if abs(value - low) <= 1e-12)
                i = ties[(iteration + source_order + position) % len(ties)]
                priced.append((i, low))
                value, alternate_i = actions[
                    (iteration + source_order + position) % min(2, len(actions))
                ]
                alternate.append((alternate_i, value))
            pattern, ordered = _lagrangian_source_prefixes(
                table, power, priced, eta, scale,
            )
            retained[source].update(_retained_prefixes(
                pattern, ordered, prefix_buckets,
            ))
            if alternate:
                alternative, ordered = _lagrangian_source_prefixes(
                    table, power, alternate, eta, scale,
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
            _lagrangian_gain(table, power, pattern, gain_cache) for pattern in chosen
        )
        step = .5 / np.sqrt(iteration + 1)
        prices = np.maximum(0, prices + step * (usage - limits))
        eta = max(0, eta + step * (target - shed) / scale)
    return _recover_lagrangian(
        table, power, retained, target, architecture, scenario, mode, gain_cache,
        return_assignment=return_assignment,
        resource_limits=limits,
    )


def _round_lp(table, target, values):
    n = len(table.candidates)
    gains = np.array([c.credit for c in table.candidates])
    work = np.array([c.objective_cost_s for c in table.candidates])
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
        gain += c.credit
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
            gain += c.credit
    return selected


def _lp(table: CandidateTable, target: float, stats=None):
    if not table.candidates:
        return set()
    started, native_s, solves, iterations = perf_counter(), 0.0, 0, 0
    x = cp.Variable(len(table.candidates), nonneg=True)
    gains = np.array([c.credit for c in table.candidates])
    work = np.array([c.objective_cost_s for c in table.candidates])
    base = [table.incidence @ x <= 1, table.resources @ x <= 1, x <= 1]
    def solve(objective, constraints, maximize=False):
        nonlocal native_s, solves, iterations
        p = cp.Problem(cp.Maximize(objective) if maximize else cp.Minimize(objective), constraints)
        try:
            p.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            return None
        native_s += p.solver_stats.solve_time or 0
        iterations += p.solver_stats.num_iters or 0
        solves += 1
        return p
    problem = solve(work @ x, base + [gains @ x >= target])
    if problem is None:
        return _lp_highs(table, target, stats)
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        problem = solve(gains @ x, base, True)
        if problem is None:
            return _lp_highs(table, target, stats)
        best = float(problem.value)
        problem = solve(work @ x, base + [gains @ x >= best - 1e-7])
        if problem is None:
            return _lp_highs(table, target, stats)
    values = None if x.value is None else np.asarray(x.value)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) \
            or values is None or not np.isfinite(values).all():
        return _lp_highs(table, target, stats)
    selected = _round_lp(table, target, values)
    if stats is not None:
        stats.update(wall_s=perf_counter() - started, native_s=native_s,
                     solves=solves, iterations=iterations)
    return selected


def _lp_highs(table: CandidateTable, target: float, stats=None):
    if not table.candidates:
        return set()
    started, native_s, solves, iterations = perf_counter(), 0.0, 0, 0
    gains = np.array([c.credit for c in table.candidates])
    work = np.array([c.objective_cost_s for c in table.candidates])
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
        result = solve(
            work / table.migration_horizon_s,
            max(0.0, best - 1e-7 * gain_scale),
        )
    if result.status:
        raise RuntimeError(f"HiGHS target LP failed: {result.message}")
    selected = _round_lp(table, target, result.x)
    if stats is not None:
        stats.update(wall_s=perf_counter() - started, native_s=native_s,
                     solves=solves, iterations=iterations)
    return selected


def _column_phase(table, target, costs, active, shortfall, stats):
    sessions = np.array([c.session for c in table.candidates])
    gains = np.array([c.credit for c in table.candidates])
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

        correction = np.maximum(0, -minimum)
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
    work = np.array([c.objective_cost_s for c in table.candidates]) \
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


def _add_priced_columns(highs, table, resources, choices, costs,
                        candidate_columns, session_rows):
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
    matrix, starts, indices, values = resources, [0], [], []
    target_row = table.resources.shape[0]
    for choice, session in zip(choices, sessions):
        sl = slice(matrix.indptr[choice], matrix.indptr[choice + 1])
        indices.extend(matrix.indices[sl])
        values.extend(matrix.data[sl])
        indices.extend((target_row, session_rows[session]))
        values.extend((table.candidates[choice].credit, 1.0))
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


def _persistent_column_phase(highs, table, resources, target, costs,
                             candidate_columns, session_rows, stats):
    sessions = np.array([c.session for c in table.candidates])
    gains = np.array([c.credit for c in table.candidates])
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
        correction = np.maximum(0, -minimum)
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
            highs, table, resources, choices, costs, candidate_columns,
            session_rows,
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
    status = highs.addRows(
        resources + 1,
        np.concatenate((np.full(resources, -highspy.kHighsInf), [target])),
        np.concatenate((np.ones(resources), [highspy.kHighsInf])), 0,
        np.zeros(resources + 2, np.int32), np.array([], np.int32),
        np.array([], float),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to initialize master rows")
    status = highs.addCol(
        1, 0, highspy.kHighsInf, 1, np.array([resources], np.int32),
        np.array([1.0]),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to add shortfall column")
    candidate_columns = np.full(len(table.candidates), -1, np.int32)
    session_rows = np.full(table.incidence.shape[0], -1, np.int32)
    resource_columns = csc_matrix(table.resources)
    first = _persistent_column_phase(
        highs, table, resource_columns, target, np.zeros(len(table.candidates)),
        candidate_columns, session_rows, phase1,
    )
    shortfall = float(first.col_value[0])
    effective = max(0.0, target - shortfall)
    work = np.array([c.objective_cost_s for c in table.candidates]) \
        / table.migration_horizon_s
    active = np.flatnonzero(candidate_columns >= 0)
    status = highs.changeColsCost(
        len(active), candidate_columns[active], work[active],
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to set Phase-II costs")
    if highs.changeColBounds(0, 0, 0) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to fix target shortfall")
    phase2_target = max(0.0, effective - 1e-7)
    if highs.changeRowBounds(
        resources, phase2_target, highspy.kHighsInf,
    ) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to set Phase-II target")
    second = _persistent_column_phase(
        highs, table, resource_columns, phase2_target, work, candidate_columns,
        session_rows, phase2,
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


@dataclass(slots=True)
class _PricedColumn:
    reduced: float
    order: tuple[str, str, str]
    candidate: Candidate
    entries: tuple[tuple[int, float], ...]
    cost: float

    def __lt__(self, other):
        return (self.reduced, self.order) > (other.reduced, other.order)


@dataclass(slots=True)
class _CompletionColumn:
    rank: tuple
    candidate: Candidate
    entries: tuple[tuple[int, float], ...]

    def __lt__(self, other):
        return self.rank < other.rank


def _lazy_completion(oracle, candidates, values, target):
    candidates, usage = list(candidates), np.zeros(len(oracle.specs))
    selected, sessions = set(), set()
    gain = 0.0
    def semantic(candidate):
        return (
            oracle.sessions[candidate.session].session_id,
            oracle.pools[candidate.pool].pool_id,
            "0" if candidate.method == "replay" else "1",
        )
    identities = {
        (candidate.session, candidate.pool, candidate.method): i
        for i, candidate in enumerate(candidates)
    }
    masses = {
        identity: values[i] for identity, i in identities.items()
    }
    after = {}
    for i in sorted(
        (i for i, value in enumerate(values) if value > 0),
        key=lambda i: (-values[i], candidates[i].objective_cost_s,
                       semantic(candidates[i])),
    ):
        candidate, entries = candidates[i], oracle.column(candidates[i])
        rank = (-values[i], candidate.objective_cost_s, *semantic(candidate))
        if candidate.session in sessions:
            continue
        if all(usage[row] + value <= 1 + 1e-8 for row, value in entries):
            selected.add(i)
            sessions.add(candidate.session)
            for row, value in entries:
                usage[row] += value
            gain += candidate.credit
            if gain >= target - 1e-8:
                return candidates, selected
        else:
            after[candidate.session] = rank

    def next_choice(j, after=None):
        best = None
        for candidate in oracle.choices(j):
            identity = (candidate.session, candidate.pool, candidate.method)
            rank = (
                -masses.get(identity, 0.0), candidate.objective_cost_s,
                *semantic(candidate),
            )
            if (after is None or rank > after) and (
                best is None or rank < best.rank
            ):
                best = _CompletionColumn(rank, candidate, oracle.column(candidate))
        return best

    heap = []
    for j in range(len(oracle.sessions)):
        if j not in sessions:
            item = next_choice(j, after.get(j))
            if item is not None:
                heappush(heap, item)
    while heap:
        item = heappop(heap)
        candidate = item.candidate
        if all(usage[row] + value <= 1 + 1e-8
               for row, value in item.entries):
            identity = (candidate.session, candidate.pool, candidate.method)
            i = identities.get(identity)
            if i is None:
                i = len(candidates)
                identities[identity] = i
                candidates.append(candidate)
            selected.add(i)
            sessions.add(candidate.session)
            for row, value in item.entries:
                usage[row] += value
            gain += candidate.credit
            if gain >= target - 1e-8:
                break
        else:
            item = next_choice(candidate.session, item.rank)
            if item is not None:
                heappush(heap, item)
    return candidates, selected


def _lazy_add_columns(highs, oracle, priced, session_rows, active, candidates):
    sessions = sorted({item.candidate.session for item in priced
                       if session_rows[item.candidate.session] < 0})
    if sessions:
        first = highs.getNumRow()
        status = highs.addRows(
            len(sessions), np.full(len(sessions), -highspy.kHighsInf),
            np.ones(len(sessions)), 0, np.zeros(len(sessions) + 1, np.int32),
            np.array([], np.int32), np.array([], float),
        )
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS failed to add lazy session rows")
        session_rows[sessions] = np.arange(first, first + len(sessions))
    target, starts, indices, values = len(oracle.specs), [0], [], []
    for item in priced:
        candidate = item.candidate
        indices.extend(row for row, _ in item.entries)
        values.extend(value for _, value in item.entries)
        indices.extend((target, session_rows[candidate.session]))
        values.extend((candidate.credit, 1.0))
        starts.append(len(indices))
    status = highs.addCols(
        len(priced), np.array([item.cost for item in priced]),
        np.zeros(len(priced)), np.full(len(priced), highspy.kHighsInf),
        len(indices), np.asarray(starts, np.int32), np.asarray(indices, np.int32),
        np.asarray(values, float),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to add lazy priced columns")
    for item in priced:
        candidate = item.candidate
        active.add((candidate.session, candidate.pool, candidate.method))
        candidates.append(candidate)


def _lazy_column_phase(highs, oracle, target, phase_two, session_rows,
                       active, candidates, stats):
    batch = max(256, len(oracle.sessions) // COLUMN_GROWTH_SWEEPS)
    pricing_s = add_s = solve_s = 0.0
    iterations = 0
    for sweep in range(sum(len(pool.methods) for pool in oracle.pools)
                       * len(oracle.sessions) + 1):
        started = perf_counter()
        highs.run()
        solve_s += perf_counter() - started
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"lazy master failed: {highs.getModelStatus()}")
        solution, info = highs.getSolution(), highs.getInfo()
        iterations += info.simplex_iteration_count
        dual = np.asarray(solution.row_dual)
        resources = np.maximum(0, -dual[:len(oracle.specs)])
        eta = max(0.0, dual[len(oracle.specs)])
        eta = eta if phase_two else min(1.0, eta)

        started, heap, dual_sessions = perf_counter(), [], 0.0
        for j, session in enumerate(oracle.sessions):
            row = session_rows[j]
            alpha = max(0.0, -dual[row]) if row >= 0 else 0.0
            minimum, best = np.inf, None
            for candidate in oracle.choices(j):
                entries = oracle.column(candidate)
                cost = candidate.objective_cost_s / oracle.migration_horizon_s \
                    if phase_two else 0.0
                reduced = cost + sum(
                    resources[index] * value for index, value in entries
                ) - eta * candidate.credit + alpha
                minimum = min(minimum, reduced)
                identity = (candidate.session, candidate.pool, candidate.method)
                order = (session.session_id,
                         oracle.pools[candidate.pool].pool_id,
                         "0" if candidate.method == "replay" else "1")
                if identity not in active and (
                    best is None or (reduced, order) < (best.reduced, best.order)
                ):
                    best = _PricedColumn(reduced, order, candidate, entries, cost)
            if np.isfinite(minimum):
                alpha += max(0.0, -minimum)
            dual_sessions += alpha
            if best is not None and best.reduced < 0:
                heappush(heap, best)
                if len(heap) > batch:
                    heappop(heap)
        lower = eta * target - resources.sum() - dual_sessions
        pricing_s += perf_counter() - started
        upper = float(info.objective_function_value)
        gap = upper - lower
        stats.update(
            sweeps=sweep + 1, columns=len(candidates), upper=upper,
            lower=float(lower), gap=gap, pricing_s=pricing_s,
            add_s=add_s, solve_s=solve_s, simplex_iterations=iterations,
        )
        if gap <= COLUMN_GAP_TOLERANCE:
            return solution
        if not heap:
            raise RuntimeError(f"lazy certificate gap did not close: {gap}")
        priced = sorted(heap, key=lambda item: (item.reduced, item.order))
        started = perf_counter()
        _lazy_add_columns(
            highs, oracle, priced, session_rows, active, candidates,
        )
        add_s += perf_counter() - started
        highs.setOptionValue("presolve", "off")
    raise RuntimeError("lazy column generation did not converge")


def _native_column_phase(highs, oracle, native, target, phase_two, session_rows,
                         active, candidates, stats):
    batch = max(256, len(oracle.sessions) // COLUMN_GROWTH_SWEEPS)
    pricing_s = materialize_s = add_s = solve_s = 0.0
    iterations = choice_evaluations = 0
    for sweep_index in range(len(oracle.options) * len(oracle.sessions) + 1):
        started = perf_counter()
        highs.run()
        solve_s += perf_counter() - started
        if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"native lazy master failed: {highs.getModelStatus()}")
        solution, info = highs.getSolution(), highs.getInfo()
        iterations += info.simplex_iteration_count
        dual = np.asarray(solution.row_dual)
        resources = np.maximum(0, -dual[:len(oracle.specs)])
        alpha = np.zeros(len(oracle.sessions))
        represented = np.flatnonzero(session_rows >= 0)
        alpha[represented] = np.maximum(0, -dual[session_rows[represented]])
        eta = max(0.0, dual[len(oracle.specs)])

        started = perf_counter()
        sweep = native.price(
            2 if phase_two else 1, eta, resources, alpha, batch,
            COLUMN_TOLERANCE,
        )
        choice_evaluations += int(sweep["evaluated_choices"])
        lower = (sweep["effective_eta"] * target - resources.sum()
                 - alpha.sum() - sweep["repair_sum"])
        pricing_s += perf_counter() - started
        upper = float(info.objective_function_value)
        gap = upper - lower
        stats.update(
            sweeps=sweep_index + 1, columns=len(candidates), upper=upper,
            lower=float(lower), gap=gap, pricing_s=pricing_s,
            materialize_s=materialize_s, add_s=add_s, solve_s=solve_s,
            simplex_iterations=iterations, evaluated_choices=choice_evaluations,
        )
        if gap <= COLUMN_GAP_TOLERANCE:
            if len(sweep["candidate_ids"]):
                native.discard(sweep["epoch"])
            return solution
        if not len(sweep["candidate_ids"]):
            raise RuntimeError(f"native lazy certificate gap did not close: {gap}")

        started = perf_counter()
        priced = []
        starts = sweep["resource_starts"]
        for column, (session, option, reduced, cost) in enumerate(zip(
            sweep["session_indices"], sweep["option_indices"],
            sweep["reduced_costs"], sweep["phase2_costs"],
        )):
            session, option = int(session), int(option)
            pool, method = oracle.options[option]
            feature = sweep["candidate_features"][7 * column:7 * column + 7]
            candidate = Candidate(
                session, method, pool, oracle.marginal_gains[session],
                float(feature[4]), float(feature[4]), oracle.pools[pool].route,
                float(feature[0]), (float(feature[1]), float(feature[2])),
                int(feature[3]), (float(feature[5]), float(feature[6])),
                float(sweep["gains"][column]),
            )
            start, end = starts[column:column + 2]
            entries = tuple(zip(
                map(int, sweep["resource_rows"][start:end]),
                map(float, sweep["resource_values"][start:end]),
            ))
            order = (
                oracle.sessions[session].session_id,
                oracle.pools[candidate.pool].pool_id,
                "0" if candidate.method == "replay" else "1",
            )
            priced.append(_PricedColumn(
                float(reduced), order, candidate, entries,
                float(cost) if phase_two else 0.0,
            ))
        materialize_s += perf_counter() - started
        started = perf_counter()
        _lazy_add_columns(
            highs, oracle, priced, session_rows, active, candidates,
        )
        native.commit(sweep["epoch"], sweep["candidate_ids"])
        add_s += perf_counter() - started
        highs.setOptionValue("presolve", "off")
    raise RuntimeError("native lazy column generation did not converge")


def _lp_column_generation_lazy(oracle, target, stats=None, native=False):
    if target <= 0 or oracle.migration_horizon_s <= 0:
        return _materialize_candidates(oracle, (), False), set()
    started, phase1, phase2 = perf_counter(), {}, {}
    native_oracle = _native_pricing_oracle(oracle) if native else None
    build_s = perf_counter() - started
    highs, resources = highspy.Highs(), len(oracle.specs)
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    status = highs.addRows(
        resources + 1,
        np.concatenate((np.full(resources, -highspy.kHighsInf), [target])),
        np.concatenate((np.ones(resources), [highspy.kHighsInf])), 0,
        np.zeros(resources + 2, np.int32), np.array([], np.int32),
        np.array([], float),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to initialize lazy master")
    if highs.addCol(
        1, 0, highspy.kHighsInf, 1, np.array([resources], np.int32),
        np.array([1.0]),
    ) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to add lazy shortfall column")
    session_rows = np.full(len(oracle.sessions), -1, np.int32)
    active, candidates = set(), []
    phase = _native_column_phase if native else _lazy_column_phase
    arguments = (highs, oracle) + ((native_oracle,) if native else ())
    first = phase(
        *arguments, target, False, session_rows, active, candidates, phase1,
    )
    shortfall = float(first.col_value[0])
    effective = max(0.0, target - shortfall)
    if candidates:
        columns = np.arange(1, len(candidates) + 1, dtype=np.int32)
        costs = np.array([
            candidate.objective_cost_s / oracle.migration_horizon_s
            for candidate in candidates
        ])
        if highs.changeColsCost(len(columns), columns, costs) \
                != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS failed to set lazy Phase-II costs")
    if highs.changeColBounds(0, 0, 0) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to fix lazy target shortfall")
    phase2_target = max(0.0, effective - 1e-7)
    if highs.changeRowBounds(
        resources, phase2_target, highspy.kHighsInf,
    ) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to set lazy Phase-II target")
    second = phase(
        *arguments, phase2_target, True, session_rows, active, candidates, phase2,
    )
    values = np.asarray(second.col_value)[1:len(candidates) + 1]
    master_columns = len(candidates)
    completed = perf_counter()
    candidates, selected = _lazy_completion(oracle, candidates, values, target)
    completion_s = perf_counter() - completed
    materialized = perf_counter()
    table = _materialize_candidates(oracle, candidates, False)
    table_s = perf_counter() - materialized
    if stats is not None:
        stats.update(
            wall_s=perf_counter() - started, active_columns=master_columns,
            materialized_columns=len(candidates),
            completion_columns=len(candidates) - master_columns,
            active_sessions=np.count_nonzero(session_rows >= 0),
            phase1_shortfall=shortfall, effective_target=effective,
            phase1=phase1, phase2=phase2, native_build_s=build_s if native else 0,
            completion_s=completion_s, table_s=table_s,
        )
    return table, selected


def _lp_column_generation_native(oracle, target, stats=None):
    return _lp_column_generation_lazy(oracle, target, stats, True)


def _pack(table, selected, architecture, scenario, mode, repair=False, preferred=None):
    horizon = architecture.residency_horizon_s
    horizon = scenario.end_s - scenario.controller_delay_s if horizon is None else horizon
    work, kv = _baseline(scenario, architecture, horizon)
    migration = {
        r.replica_id: {"replay": 0.0, "kv_transfer": 0.0}
        for p in architecture.pools for r in p.replicas
    }
    assignment, rejected = {}, []
    costs = np.asarray(table.resources.sum(0)).ravel() if repair else None
    for p, pool in enumerate(architecture.pools):
        q = architecture.type_by_id[pool.type_id]
        normals, bounds = np.asarray(q.normals), _event_bounds(q, pool, mode)
        replicas = pool.replicas
        replica_work = np.asarray([work[r.replica_id] for r in replicas])
        replica_kv = np.asarray([kv[r.replica_id] for r in replicas])
        replica_migration = np.zeros((len(replicas), 2))
        method_column = {"replay": 0, "kv_transfer": 1}
        members = [i for i in selected if table.candidates[i].pool == p]
        members.sort(key=(
            lambda i: (costs[i] / max(table.candidates[i].credit, 1e-12),
                       table.sessions[table.candidates[i].session].session_id, i)
        ) if repair else lambda i: (-max(
            *(normals @ table.candidates[i].service_work / bounds),
            table.candidates[i].kv_tokens
            / (q.kv_capacity_tokens // q.kv_block_tokens),
            0 if pool.fluid_migration else
            table.candidates[i].duration_s / table.migration_horizon_s,
        ), table.sessions[table.candidates[i].session].session_id, i))
        for i in members:
            c = table.candidates[i]
            next_work = replica_work + c.service_work
            pressure = np.maximum.reduce((
                np.max(next_work @ normals.T / bounds, axis=1),
                (replica_kv + c.kv_tokens)
                / (q.kv_capacity_tokens // q.kv_block_tokens),
                np.zeros(len(replicas)) if pool.fluid_migration else np.maximum(
                    np.max(replica_migration, axis=1),
                    replica_migration[:, method_column[c.method]] + c.duration_s,
                ) / table.migration_horizon_s,
            ))
            feasible = np.flatnonzero(pressure <= 1 + 1e-9)
            if not feasible.size:
                if repair:
                    rejected.append(i)
                    continue
                return None, tuple(sorted(members))
            wanted = None if preferred is None else preferred.get(
                table.sessions[c.session].session_id)
            matching = [r for r in feasible if wanted
                        and wanted.method == c.method
                        and wanted.pool == pool.pool_id
                        and replicas[r].replica_id == wanted.destination]
            r = matching[0] if matching else feasible[np.argmin(pressure[feasible])]
            replica_work[r] = next_work[r]
            replica_kv[r] += c.kv_tokens
            if not pool.fluid_migration:
                replica_migration[r, method_column[c.method]] += c.duration_s
            replica_id = replicas[r].replica_id
            work[replica_id], kv[replica_id] = replica_work[r], replica_kv[r]
            migration[replica_id][c.method] = replica_migration[r, method_column[c.method]]
            assignment[i] = replica_id
    return assignment, tuple(rejected) if repair else None


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
               and (pools[r][1].fluid_migration is not None
                    or max(migration[r].values()) <= table.migration_horizon_s + 1e-9)
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
        table.candidates[i].objective_cost_s / max(table.candidates[i].credit, 1e-12), i,
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


def _repair_selection(table, architecture, target, attempts, soft):
    if not table.candidates:
        return set(), False
    n, x = len(table.candidates), cp.Variable(len(table.candidates), nonneg=True)
    gains = np.array([c.credit for c in table.candidates])
    durations = np.array([c.duration_s for c in table.candidates])
    session_ids = [table.sessions[c.session].session_id for c in table.candidates]
    current = {a.session_id: a for a in attempts if a.status in {"pending", "running"}}
    same = np.array([
        sid in current and current[sid].assignment.method == c.method
        and current[sid].assignment.pool == architecture.pools[c.pool].pool_id
        and len(architecture.pools[c.pool].replicas) == 1
        and current[sid].assignment.destination
        == architecture.pools[c.pool].replicas[0].replica_id
        for sid, c in zip(session_ids, table.candidates)
    ])
    change = np.array([
        -1.0 if same[i] else 0.0 if session_ids[i] in current else 1.0
        for i in range(n)
    ])
    discarded = np.array([
        -current[session_ids[i]].completed_work
        / current[session_ids[i]].total_work * durations[i]
        if same[i] else 0.0 for i in range(n)
    ])
    forced = [i for i in range(n) if same[i] and soft
              and current[session_ids[i]].soft_changed]
    base = [table.incidence @ x <= 1, table.resources @ x <= 1, x <= 1]
    base += [x[i] == 1 for i in forced]

    def solve(objective, constraints, maximize=False):
        problem = cp.Problem(
            cp.Maximize(objective) if maximize else cp.Minimize(objective), constraints,
        )
        problem.solve(solver=cp.CLARABEL)
        return problem

    constraints = base + [gains @ x >= target]
    problem = solve(change @ x, constraints)
    reaches = problem.status not in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
    if reaches:
        constraints += [change @ x <= float(change @ x.value) + 1e-7]
        problem = solve(discarded @ x, constraints)
        constraints += [discarded @ x <= float(discarded @ x.value) + 1e-7]
        problem = solve(durations @ x, constraints)
    else:
        problem = solve(gains @ x, base, True)
    if x.value is None or not np.isfinite(x.value).all():
        return set(), False

    matrix, usage = csc_matrix(table.resources), np.zeros(table.resources.shape[0])
    selected, sessions = set(), set()
    for i in (*forced, *np.lexsort((durations, -np.asarray(x.value)))):
        i, candidate = int(i), table.candidates[int(i)]
        if i in selected or candidate.session in sessions:
            continue
        sl = slice(matrix.indptr[i], matrix.indptr[i + 1])
        rows, added = matrix.indices[sl], matrix.data[sl]
        if np.any(usage[rows] + added > 1 + 1e-8):
            continue
        selected.add(i)
        sessions.add(candidate.session)
        usage[rows] += added
        if reaches and sum(table.candidates[j].credit for j in selected) >= target - 1e-8:
            break
    return selected, reaches


def _repair_architecture(architecture, observations):
    """Apply independently observed prefill capacities to destination pools."""
    values = {row.pool: row for row in observations}
    if len(values) != len(observations):
        raise ValueError("duplicate repair prefill pool observation")
    unknown = set(values) - {pool.pool_id for pool in architecture.pools}
    if unknown:
        raise ValueError(f"unknown repair prefill pools: {sorted(unknown)}")
    if not values:
        return architecture
    types, pools = list(architecture.types), []
    by_type = architecture.type_by_id
    for pool in architecture.pools:
        observation = values.get(pool.pool_id)
        if observation is None:
            pools.append(pool)
            continue
        current = by_type[pool.type_id]
        scale = observation.tokens_per_s / current.prefill.at(
            observation.context_tokens)
        if scale <= 0:
            raise ValueError("repair prefill scale must be positive")
        migration = None if current.migration is None else {
            method: replace(
                component,
                compute_completion_factor=(
                    component.compute_completion_factor / scale
                    if method == "replay" else
                    component.compute_completion_factor
                ),
                provenance=(
                    f"{component.provenance}; observed {pool.pool_id} prefill "
                    f"{observation.tokens_per_s:g} tok/s at "
                    f"{observation.context_tokens:g} tokens"
                ),
            ) for method, component in current.migration.items()
        }
        repaired = replace(
            current,
            type_id=f"{current.type_id}/repair/{pool.pool_id}",
            prefill=replace(
                current.prefill,
                rates=tuple(rate * scale for rate in current.prefill.rates),
            ),
            migration=migration,
            provenance=(
                f"{current.provenance}; observed {pool.pool_id} prefill "
                f"scale={scale:g}"
            ),
        )
        replicas = tuple(replace(
            replica,
            baseline_work=(replica.baseline_work[0] / scale,
                           replica.baseline_work[1]),
        ) for replica in pool.replicas)
        types.append(repaired)
        pools.append(replace(pool, type_id=repaired.type_id, replicas=replicas))
    return replace(architecture, types=tuple(types), pools=tuple(pools))


def _land_committed_sessions(architecture, scenario, attempts, committed, horizon):
    """Charge committed repair work to its concrete destination replica."""
    if not committed:
        return architecture
    sessions = {session.session_id: session for session in scenario.sessions}
    assignments = {session_id: attempts[session_id].assignment
                   for session_id in committed if session_id in attempts}
    known = {replica.replica_id for pool in architecture.pools
             for replica in pool.replicas}
    if any(value.destination not in known for value in assignments.values()):
        raise ValueError("committed repair assignment has an unknown replica")
    pools = []
    for pool in architecture.pools:
        q = architecture.type_by_id[pool.type_id]
        replicas = []
        for replica in pool.replicas:
            work = np.asarray(replica.baseline_work, float)
            kv = replica.baseline_kv_tokens
            for session_id, assignment in assignments.items():
                if assignment.destination != replica.replica_id:
                    continue
                session = sessions[session_id]
                tokens = _resident_tokens(session, horizon)
                work += q.work(
                    session.expected_f, session.expected_g, tokens,
                    q.migration is not None,
                )
                kv += tokens
            replicas.append(replace(
                replica, baseline_work=tuple(work), baseline_kv_tokens=kv))
        pools.append(replace(pool, replicas=tuple(replicas)))
    return replace(architecture, pools=tuple(pools))


def repair_destination(scenario, profile, architecture, request: RepairRequest,
                       admission_mode="normal") -> RepairResult:
    """Solve one residual, target-restoring pool-aware repair."""
    state = request.snapshot
    if state.credit_deadline_s <= state.now_s:
        return RepairResult(request.request_id, state.budget_version, (), 0, False)
    architecture = _repair_architecture(
        architecture, state.prefill_capacities)
    attempts = {attempt.session_id: attempt for attempt in state.attempts}
    locked = {session_id: attempt for session_id, attempt in attempts.items()
              if attempt.status in {"pending", "running"}
              and not attempt.repairable}
    residency = architecture.residency_horizon_s
    residency = scenario.end_s - state.now_s if residency is None else residency
    architecture = _land_committed_sessions(
        architecture, scenario, attempts, state.committed, residency)
    sessions = tuple(replace(
        session,
        movable=(session.session_id in state.source_sessions
                 and session.session_id not in locked),
    ) for session in scenario.sessions)
    rates = dict(state.route_rates)
    links = tuple(replace(link, bytes_per_s=rates.get(link.link_id, link.bytes_per_s))
                  for link in scenario.links)
    deadline = state.credit_deadline_s + profile.power_window_s
    residual = replace(
        scenario, sessions=sessions, links=links, controller_delay_s=state.now_s,
        deadline_s=deadline, end_s=max(scenario.end_s, deadline),
        final_state="awake", assumed_shutdown_s=None,
    )
    power = ExpectedPower(residual, profile)
    oracle = _candidate_oracle(
        residual, profile, architecture, admission_mode, power, True,
    )
    active = {sid: attempt for sid, attempt in attempts.items()
              if attempt.status in {"pending", "running"}}
    selectable = {sid: attempt for sid, attempt in active.items()
                  if attempt.repairable}
    replay_rates = dict(state.replay_rates)
    full, candidates = {}, []
    case = profile.case("central")
    for j in range(len(oracle.sessions)):
        for candidate in oracle.choices(j):
            session = oracle.sessions[j]
            pool = architecture.pools[candidate.pool]
            attempt = active.get(session.session_id)
            same = attempt and attempt.assignment.method == candidate.method \
                and attempt.assignment.pool == pool.pool_id
            same = bool(same and len(pool.replicas) == 1
                        and attempt.assignment.destination
                        == pool.replicas[0].replica_id)
            full[session.session_id, candidate.method, candidate.pool] = candidate
            if same:
                fraction = (attempt.total_work - attempt.completed_work) / attempt.total_work
                duration = ((attempt.total_work - attempt.completed_work) / attempt.rate
                            + attempt.commit_overhead_s) if attempt.rate else \
                    max(0.0, attempt.planned_commit_s - state.now_s)
                candidate = replace(
                    candidate, duration_s=duration,
                    route_bytes=candidate.route_bytes * fraction,
                    migration_work_s=candidate.migration_work_s * fraction,
                    transition_work=tuple(value * fraction
                                          for value in candidate.transition_work),
                )
            elif candidate.method == "replay" and pool.pool_id in replay_rates:
                components = architecture.type_by_id[pool.type_id].migration
                factor = components["replay"].compute_completion_factor \
                    if components else 1.0
                route_s = candidate.route_bytes / min(
                    link.bytes_per_s for link in residual.links
                    if link.link_id in candidate.path)
                candidate = replace(
                    candidate,
                    duration_s=route_s
                    + session.context_tokens / replay_rates[pool.pool_id]
                    + factor * case.replay_completion_s + case.switch_s,
                )
            if candidate.duration_s + state.eta_guard_s <= oracle.migration_horizon_s:
                candidates.append(candidate)
    table = _materialize_candidates(oracle, candidates)
    original_power = ExpectedPower(replace(scenario, final_state="awake",
                                           assumed_shutdown_s=None), profile)
    fixed_moves = tuple(RepairMove(
        session_id, attempt.assignment,
        ((attempt.total_work - attempt.completed_work) / attempt.rate
         + attempt.commit_overhead_s) if attempt.rate else
        max(0.0, attempt.planned_commit_s - state.now_s),
        attempt.total_work, attempt.commit_overhead_s,
    ) for session_id, attempt in sorted(locked.items()))
    fixed_on_time = {
        move.session_id for move in fixed_moves
        if max(
            attempts[move.session_id].planned_commit_s,
            state.now_s + move.duration_s,
        ) + state.eta_guard_s <= state.credit_deadline_s
    }
    fixed_gain = original_power.drain_gain(state.committed | fixed_on_time)
    remaining_target = max(0.0, state.target_watts - fixed_gain)
    selected, lp_reaches = (set(), True) if remaining_target <= 1e-8 else \
        _repair_selection(
            table, architecture, remaining_target,
            tuple(selectable.values()), request.trigger.startswith("soft:"),
        )
    preferred = {sid: attempt.assignment for sid, attempt in selectable.items()}
    assignment, rejected = _pack(
        table, selected, architecture, residual, admission_mode, True, preferred,
    )
    selected -= set(rejected)
    moved = {table.sessions[table.candidates[i].session].session_id for i in selected}
    attainable = original_power.drain_gain(
        state.committed | fixed_on_time | moved)
    reaches = lp_reaches and not rejected and attainable >= state.target_watts - 1e-8
    moves = []
    for i in sorted(selected, key=lambda i: table.sessions[table.candidates[i].session].session_id):
        candidate = table.candidates[i]
        session = table.sessions[candidate.session]
        pool = architecture.pools[candidate.pool]
        attempt = active.get(session.session_id)
        same = attempt and attempt.assignment.method == candidate.method \
            and attempt.assignment.pool == pool.pool_id \
            and len(pool.replicas) == 1 \
            and attempt.assignment.destination == assignment[i]
        total = attempt.total_work if same else (
            session.context_tokens if candidate.method == "replay" else
            full[session.session_id, candidate.method, candidate.pool].route_bytes
        )
        components = architecture.type_by_id[pool.type_id].migration
        if candidate.method == "replay":
            factor = components["replay"].compute_completion_factor \
                if components else 1.0
            compute = session.context_tokens / (
                replay_rates[pool.pool_id]
                if pool.pool_id in replay_rates else
                case.replay.rate(session.context_tokens, 1) / factor
            )
            overhead = max(0.0, candidate.duration_s - compute)
        else:
            overhead = (components["kv_transfer"].residual_s
                        if components else case.kv_transfer.initial_completion_s)
            overhead += _kv_catch_up_s(
                session, _resident_tokens(session, oracle.migration_horizon_s),
                case, oracle.migration_horizon_s,
            )
        moves.append(RepairMove(
            session.session_id,
            Assignment(candidate.method, assignment[i], pool.pool_id),
            candidate.duration_s, total,
            attempt.commit_overhead_s if same else overhead,
        ))
    return RepairResult(
        request.request_id, state.budget_version,
        tuple(sorted((*fixed_moves, *moves), key=lambda move: move.session_id)),
        attainable, reaches,
    )


def _mode_plan(scenario, profile, architecture, solver, mode, power, target, seed=0):
    assignment = None
    streamed = solver in {"lp_column_generation_lazy", "lp_column_generation_native"}
    selection_credits = None
    if power is not None and solver not in {
            "greedy_lagrangian", "max_shed", "lp_power_blind"}:
        sessions = tuple(s for s in _local_sessions(scenario) if s.state == "active")
        selection_credits, target, _ = _phase_power_target(
            power, sessions, target,
        )
    if streamed:
        solve = (_lp_column_generation_native if solver.endswith("native")
                 else _lp_column_generation_lazy)
        oracle = _candidate_oracle(
            scenario, profile, architecture, mode, power,
        ) if selection_credits is None else _candidate_oracle(
            scenario, profile, architecture, mode, power,
            selection_credits=selection_credits,
        )
        table, selected = solve(oracle, target)
    else:
        table = candidate_table(
            scenario, profile, architecture, mode, power, selection_credits,
        )
        if solver == "lp_power_blind":
            gain = power.drain_gain(session.session_id for session in table.sessions) \
                / len(table.sessions)
            table = replace(table, candidates=tuple(
                replace(candidate, gain_w=gain) for candidate in table.candidates))
    if streamed:
        pass
    elif solver == "max_shed":
        selected = _max_shed(table, power)
    elif solver == "greedy_lagrangian":
        selected, assignment = _greedy_lagrangian(
            table, target, power, architecture, scenario, mode, True,
        )
    elif solver in {"isolated_fastest", "random", "replay_only", "kv_only"}:
        selected = _baseline_policy(table, target, solver, seed)
    else:
        selected = (_lp_column_generation_persistent(table, target)
                    if solver == "lp_column_generation_persistent" else
                    _lp_column_generation(table, target)
                    if solver == "lp_column_generation" else
                    _lp_highs(table, target)
                    if solver in {"lp_highs", "lp_power_blind"} else
                    _lp(table, target) if solver.startswith("lp") else
                    _greedy(table, target))
    repairs, repair_s = 0, 0.0
    if assignment is None:
        started = perf_counter()
        assignment, cut = _pack(
            table, selected, architecture, scenario, mode, repair=True,
        )
        if solver == "max_shed" and cut:
            raise RuntimeError("maximum-shed set is not replica-packable")
        selected.difference_update(cut)
        repairs = len(cut)
        repair_s = perf_counter() - started if cut else 0.0
    if selection_credits is not None:
        credited = sum(table.candidates[i].credit for i in selected)
        for i in sorted(selected, key=lambda i: (
                table.candidates[i].objective_cost_s, i), reverse=True):
            if credited - table.candidates[i].credit >= target - 1e-9:
                selected.remove(i)
                assignment.pop(i, None)
                credited -= table.candidates[i].credit
    return table, selected, assignment, repairs, repair_s


def plan_destination(scenario, profile, solver, case_id, seed, architecture,
                     admission_mode=None):
    if solver not in {"greedy", "greedy_lagrangian", "max_shed",
                      "isolated_fastest", "random", "replay_only", "kv_only",
                      "lp", "lp_peak_first", "lp_work_first", "lp_highs",
                      "lp_power_blind",
                      "lp_column_generation", "lp_column_generation_persistent",
                      "lp_column_generation_lazy", "lp_column_generation_native"}:
        raise ValueError("destination architecture supports pool-aware LP and greedy")
    if case_id != "central":
        raise ValueError("destination admission supports the central profile")
    if admission_mode not in {None, "normal", "emergency"}:
        raise ValueError("invalid destination admission mode")
    selection_scenario = replace(
        scenario, final_state="awake", assumed_shutdown_s=None,
    )
    start, power = perf_counter(), ExpectedPower(selection_scenario, profile, case_id)
    initial, target = power.power(True), max(0.0, power.power(True) - scenario.power_limit_w)
    chosen = None
    equal_modes = all(np.array_equal(
        _event_bounds(architecture.type_by_id[pool.type_id], pool, "normal"),
        _event_bounds(architecture.type_by_id[pool.type_id], pool, "emergency"),
    ) for pool in architecture.pools)
    for mode in ((admission_mode,) if admission_mode else ("normal", "emergency")):
        result = _mode_plan(scenario, profile, architecture, solver, mode, power, target, seed)
        moved = [result[0].sessions[result[0].candidates[i].session].session_id for i in result[1]]
        planned = source_power(selection_scenario, profile, moved, case_id)
        chosen = mode, result, planned
        if planned <= scenario.power_limit_w + 1e-8 or admission_mode:
            break
        if mode == "normal" and equal_modes:
            chosen = "emergency", result, planned
            break
    mode, (table, selected, assignment, repairs, repair_s), planned = chosen
    fluid = any(pool.fluid_migration for pool in architecture.pools)
    deadline_repairs = 0
    deadline_repair_s = 0.0
    while True:
        moves = _moves(table, selected, assignment, architecture, scenario, profile)
        validate_destination_execution(scenario, architecture, moves)
        expected = predict(
            _expected_scenario(scenario, moves), profile, moves, case_id, architecture,
        )
        deadline = scenario.controller_delay_s + table.migration_horizon_s
        if not fluid or expected.migration_makespan_s is not None \
                and expected.migration_makespan_s <= deadline + 1e-9 or not selected:
            break
        started = perf_counter()
        costs = np.asarray(table.resources.sum(0)).ravel()
        drop = max(selected, key=lambda i: (
            costs[i] / max(table.candidates[i].credit, 1e-12), i,
        ))
        selected.remove(drop)
        assignment.pop(drop)
        moved = [table.sessions[table.candidates[i].session].session_id for i in selected]
        planned = source_power(selection_scenario, profile, moved, case_id)
        deadline_repairs += 1
        deadline_repair_s += perf_counter() - started
    if power.case.phase_power is not None and solver not in {
            "greedy_lagrangian", "max_shed", "lp_power_blind"}:
        _, required_load, _ = _phase_power_target(power, table.sessions, target)
        credited_load = sum(table.candidates[i].credit for i in selected)
        if (credited_load >= required_load - 1e-9) != \
                (planned <= scenario.power_limit_w + 1e-8):
            raise RuntimeError("phase-load target disagrees with exact source power")
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
        deadline_repair_count=deadline_repairs,
        deadline_repair_s=deadline_repair_s,
        predicted_migration_makespan_s=makespan, bottleneck=bottleneck,
        planner_memory_bytes=memory, service_debt_replica_s=debt,
        required_recovery_s=recovery, binding_resources=binding,
        resource_uses=resources, service_debts=debt_rows,
    )
