"""Measured batch schedules for a pooled 2 MW installed GPT-OSS/A100 fleet."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import time
import warnings
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from itertools import product
from pathlib import Path

import highspy
import numpy as np
from scipy.sparse import csr_matrix

from pool_shed_calibration import calibration, regional_check, replay_seconds, kv_state, loaded_execution_check, resident_execution_check, service_work, source_power
from pool_shed_execution import DISPATCH_CHUNKS, destination_gpus, network_nodes
from pool_shed_planner import PLANNING_ITERATIONS, PLANNING_RESOLUTION

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/a100-pooled-agentic-2mw"
NETWORK = ROOT / "outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json"
MANIFEST = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
SCHEMA = "queue-haul-a100-pooled-service-v9"
WORKLOADS = ("measured_pack", "coding", "coding_long")
GPUS, SOURCE_LOAD = 6666, .8
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest")
ACTIONS = ("east_replay", "east_kv_transfer", "germany_replay", "germany_kv_transfer")
DEADLINES = (1, 3, 10, 30, 60, 120, 300, 600, 1800, 3600)
LOADS = (.25, .5, .75, .9, .95)
PRIMARY_TOL = 1e-9
PRIMARY_ROW_TOL = 1e-9


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt") as handle:
            handle.write(payload)
    else:
        temporary.write_text(payload)
    temporary.replace(path)


@cache
def network_samples():
    data = json.loads(NETWORK.read_text())
    routes = np.array([data["paths"][r]["simultaneous_mbps"] for r in ("east", "germany")]).T
    shared = np.array(data["aggregate_simultaneous_mbps"])
    if not np.allclose(routes.sum(1), shared) or np.any(routes <= 0):
        raise ValueError("network observations must be positive and paired")
    return np.column_stack((routes, shared)) * 125_000


def bandwidth(endpoint, nodes, wan_gbps):
    if nodes < 1 or np.any(np.asarray(endpoint) <= 0):
        raise ValueError("invalid endpoint capacity")
    return endpoint.copy() if wan_gbps == "reference" else np.minimum(endpoint * nodes, float(wan_gbps) * 1e9 / 8)


@dataclass
class Fleet:
    count: np.ndarray
    context: np.ndarray
    prompt: np.ndarray
    output: np.ndarray
    t1: np.ndarray
    kv: np.ndarray
    log: np.ndarray
    demand: np.ndarray
    templates: list
    gpus: int
    kv_capacity: float
    metadata: dict
    gpus_per_node: int = 1

    @property
    def nodes(self):
        return (self.gpus + self.gpus_per_node - 1) // self.gpus_per_node

    @property
    def gain(self):
        return self.demand / (SOURCE_LOAD * self.gpus)

    @property
    def baseline_kv(self):
        return float(self.count @ self.memory_tokens)

    @property
    def memory_tokens(self):
        peak = self.metadata.get("peak_context", self.context)
        return np.ceil(np.maximum(self.context, peak) / 16) * 16


@cache
def sample_fleet(workload, snapshot=0, gpus=GPUS, gpus_per_node=8):
    c = calibration(0)
    rng = np.random.default_rng(1001 + snapshot)
    if workload == "measured_pack":
        shapes = np.array([[context, 2048, 32] for context in (2048, 4096, 4096, 8192, 8192, 12288, 12288, 14336)], float)
        count = np.full(8, gpus)
        evidence = {"timing_scope": "measured_pack_and_background", "excluded_states": 0,
                    "turn_sequences": [[{"context": context, "prompt": prompt, "output": output, "reset": True}]
                                       for context, prompt, output in shapes],
                    "turn_offset": [0] * 8, "sequence_cycle": True,
                    "trace_end": "synthetic repeated measured request shapes, reset each turn"}
    elif workload in ("coding", "coding_long"):
        raw = json.loads(MANIFEST.read_text())
        ids = sum(raw["manifest"]["splits"]["coding"].values(), [])
        rows = [r for r in raw["traces"] if r["session_id"] in ids]
        lo, hi = min(c["replay_context_tokens"]), max(c["replay_context_tokens"])
        unsupported = {r["session_id"] for r in rows if r["input_tokens_total"] + r["output_tokens"] > hi}
        supported = [r for r in rows if lo <= r["input_tokens_total"] - r["newly_append_tokens"] <= hi
                     and r["newly_append_tokens"] + r["output_tokens"] > 0 and r["session_id"] not in unsupported]
        families = {key: [r for r in supported if r["session_id"] == key
                          and (workload != "coding_long" or r["input_tokens_total"] - r["newly_append_tokens"] >= 24576)] for key in ids}
        families = {key: value for key, value in families.items() if value}
        if not families:
            raise ValueError("no coding states in the singleton calibration support")
        picked = [families[key][rng.integers(len(families[key]))] for key in rng.choice(sorted(families), 24)]
        shapes = np.array([(r["input_tokens_total"] - r["newly_append_tokens"], r["newly_append_tokens"], r["output_tokens"]) for r in picked], float)
        count = rng.multinomial(gpus * 8, np.full(24, 1 / 24))
        evidence = {"timing_scope": "coding_background_and_subset_transfer", "sampled_states": picked,
                    "excluded_states": len(rows) - len(supported), "supported_states": len(supported),
                    "excluded_trajectories": len(unsupported),
                    "exclusion_reason": "initial context outside singleton support, no ongoing work, or future trajectory exceeds context support"}
        evidence["turn_sequences"] = [[{"context": r["input_tokens_total"] - r["newly_append_tokens"],
            "prompt": r["newly_append_tokens"], "output": r["output_tokens"], "reset": r["reset"]}
            for r in sorted(rows, key=lambda r: r["turn"])
            if r["session_id"] == state["session_id"]] for state in picked]
        evidence["turn_offset"] = [sum(r["session_id"] == state["session_id"] and r["turn"] < state["turn"] for r in rows) for state in picked]
        evidence["sequence_cycle"] = True
        evidence["trace_end"] = "replay complete recorded trajectory with context reset on wrap; explicit pacing/lifecycle assumption"
    else:
        raise ValueError(workload)
    context, prompt, output = shapes.T
    old_work = np.array([np.mean([r["prompt"] / c["F"] + r["output"] / c["G"] for r in sequence])
                         for sequence in evidence["turn_sequences"]])
    evidence["turn_work_s"] = [[float(service_work(r["context"] + r["prompt"], r["prompt"], r["output"], c)) for r in sequence]
                               for sequence in evidence["turn_sequences"]]
    evidence["turn_duration_s"] = [[v * c["resident_service"]["bound"] for v in sequence] for sequence in evidence["turn_work_s"]]
    work = np.array([np.mean(sequence) for sequence in evidence["turn_work_s"]])
    cadence = SOURCE_LOAD * gpus / (count @ work)
    evidence.update(reference_rps=float(count.sum() / (count @ work)), source_session_rps=float(cadence),
                    reference_basis="cycle-average context-dependent work / measured normal coding service bound",
                    protect_resident=False, paced_source=True,
                    source_phase_s=((np.random.default_rng(2001 + snapshot).permutation(len(count)) + .5) / (len(count) * cadence)).tolist(),
                    timing_load_factor=float((count @ old_work) / (count @ work)),
                    service_context_limit=c["resident_service"]["context_limit"],
                    service_context_extrapolated=any(r["context"] + r["prompt"] > c["resident_service"]["context_limit"]
                                                     for sequence in evidence["turn_sequences"] for r in sequence),
                    serving_work_s=work.tolist(),
                    arrivals="equal cadence, seeded stratified cohort phases, current cycle starts before time zero; trace timestamps unavailable", initial_migration_queue=0,
                    batch_context_limit=c["batch_context_limit"],
                    packing_context_tokens=c["packing_context_tokens"] if workload != "measured_pack" else None)
    evidence["kv_partial_s"] = (kv_state(context, c)[1] / c["kv_tail_replay_tps"]).tolist()
    evidence["peak_context"] = [max(r["context"] + r["prompt"] + r["output"] for r in sequence) for sequence in evidence["turn_sequences"]]
    evidence["memory_basis"] = "rounded maximum recorded cycle context reserved for resident and incoming cohorts"
    fleet = Fleet(count, context, prompt, output, replay_seconds(context, c),
                 kv_state(context, c)[0], 2 * context,
                 work * cadence, [list(range(i, i + 8)) for i in range(0, len(count), 8)],
                 gpus, gpus * c["kv_capacity_tokens"], evidence, gpus_per_node)
    evidence["source_power"] = source_power(fleet, c)
    return fleet


def batch_time(replay, t1, beta, kappa, load):
    replay, t1, kappa = np.asarray(replay), np.asarray(t1), np.asarray(kappa)
    if (np.any(replay < 0) or np.any(t1 <= 0) or np.any((kappa < 0) | (kappa > 1))
            or np.any(replay != np.floor(replay))
            or not np.isfinite(np.r_[beta, kappa.ravel(), load, t1, replay.ravel()]).all()):
        raise ValueError("invalid measured batch inputs")
    return np.exp(beta * load) * (np.sum(replay * kappa * t1, axis=1)
                                 + np.max(np.where(replay > 0, (1 - kappa) * t1, 0), axis=1))


@cache
def patterns(workload, snapshot=0, gpus=GPUS, expanded=False):
    return library(sample_fleet(workload, snapshot, gpus), expanded)


def library(fleet, expanded=False):
    n = len(fleet.count)
    found = set()
    def add(selected, replayed):
        r, k = np.zeros(n, int), np.zeros(n, int)
        np.add.at(r, replayed, 1)
        np.add.at(k, selected, 1)
        k -= r
        found.add(tuple(np.r_[r, k]))
    for i in range(n):
        add([i], [])
        add([i], [i])
    keys = [fleet.context, -fleet.context, -fleet.gain / fleet.t1, -fleet.gain / fleet.kv]
    if expanded:
        keys += [fleet.gain / fleet.t1, fleet.gain / fleet.kv, fleet.demand, -fleet.demand]
    ratio = (fleet.kv - fleet.log) / fleet.t1
    for template in fleet.templates:
        for key in keys:
            order = sorted(template, key=lambda i: (key[i], i))
            for width in range(1, len(order) + 1):
                selected = order[:width]
                ranked = sorted(selected, key=lambda i: (-ratio[i], i))
                for sequence in (ranked, ranked[::-1]) if expanded else (ranked,):
                    for cut in range(width + 1):
                        add(selected, sequence[:cut])
    values = np.array(sorted(found), dtype=float)
    return action_closure(values[:, :n], values[:, n:])


@dataclass
class Table:
    fleet: Fleet
    replay: np.ndarray
    kv: np.ndarray
    route: np.ndarray
    duration: np.ndarray
    release: np.ndarray
    kv_release: np.ndarray
    log_bytes: np.ndarray
    kv_bytes: np.ndarray
    rate: np.ndarray
    eligible: np.ndarray
    fastest: np.ndarray
    matrix: np.ndarray
    capacities: np.ndarray
    gains: np.ndarray
    deadline: float
    load: float
    endpoint: np.ndarray
    budgets: np.ndarray
    timing: dict
    debt: np.ndarray
    nominal_commit: np.ndarray
    service_time: np.ndarray
    require_recovery: bool = False


def nominal_action(fleet, counts, action, route, rate, timing, measured):
    from pool_shed_execution import _quiesce, _buffered, catchup, initial_work, source_snapshot

    if not counts.any():
        return 0., 0., 0., 0.
    origin, turns = source_snapshot(fleet, 0.)
    volume, work = initial_work(fleet, counts, action, route, origin, timing, measured)
    load_factor = np.exp(timing["beta"] * measured["forecast_load"] * fleet.metadata.get("timing_load_factor", 1.))
    pause_requested = volume / rate + work * load_factor
    pause, context, reset, _ = _quiesce(fleet, counts, pause_requested, origin_turn=turns)
    delta, tail = catchup(fleet, counts, action, route, context, reset, timing, measured, origin_context=origin)
    commit = pause + delta / rate + tail * load_factor + measured.get("switch_s", 0.)
    quiescing = fleet.metadata.get("paced_source", False)
    _, buffered = _buffered(fleet, counts, pause_requested if quiescing else pause, commit, measured, quiescing=quiescing)
    return volume + delta, (work + tail) * load_factor, commit, buffered


def isolated_methods(fleet, load, endpoint, budgets, timing):
    budgets = np.minimum(budgets, endpoint * network_nodes(fleet))
    rates = np.minimum(endpoint[:2], np.minimum(budgets[:2], budgets[2]))
    kv_rates = np.minimum(rates, timing.get("regional_kv_bytes_per_s", rates))
    replay = fleet.log / rates[:, None] + fleet.t1 * np.exp(timing["beta"] * load * fleet.metadata.get("timing_load_factor", 1.)) * np.asarray(timing.get("regional_replay_factor", [1., 1.]))[:, None]
    kv = fleet.kv / kv_rates[:, None] + (timing["kv_completion_s"] + np.asarray(fleet.metadata.get("kv_partial_s", np.zeros(len(fleet.count))))) * np.exp(timing["beta"] * load * fleet.metadata.get("timing_load_factor", 1.))
    if "turn_sequences" in fleet.metadata:
        measured = {**calibration(0), "forecast_load": load}
        shapes = np.eye(len(fleet.count))
        replay = np.array([[nominal_action(fleet, c, 0, route, rates[route], timing, measured)[2] for c in shapes] for route in (0, 1)])
        kv = np.array([[nominal_action(fleet, c, 1, route, kv_rates[route], timing, measured)[2] for c in shapes] for route in (0, 1)])
    return replay.min(0) < kv.min(0)


def action_closure(replay, kv):
    zero = np.zeros_like(replay)
    values = np.unique(np.vstack((np.c_[replay, zero], np.c_[zero, kv])), axis=0)
    return np.split(values[values.sum(1) > 0], 2, axis=1)


def include_isolated(replay, kv, fastest):
    total = replay + kv
    return action_closure(np.vstack((replay, total * fastest)), np.vstack((kv, total * ~fastest)))


def schedule_table(fleet, replay, kv, load, deadline, endpoint, budgets, timing):
    endpoint, budgets = np.asarray(endpoint), np.asarray(budgets)
    tails = np.array([timing["kv_completion_s"], timing["kv_batch_completion_s"]])
    if (not 0 <= load < 1 or deadline <= 0 or np.any(tails < 0) or np.any(endpoint <= 0) or np.any(budgets <= 0)
            or not np.isfinite(np.r_[deadline, load, tails, endpoint, budgets]).all()
            or np.any(replay < 0) or np.any(kv < 0) or np.any(kv != np.floor(kv)) or replay.shape != kv.shape):
        raise ValueError("invalid scheduling inputs")
    budgets = np.minimum(budgets, endpoint * network_nodes(fleet))
    fastest = isolated_methods(fleet, load, endpoint, budgets, timing)
    r, k = np.tile(replay, (2, 1)), np.tile(kv, (2, 1))
    route = np.repeat([0, 1], len(replay))
    long_context = np.any((r > 0) & (fleet.context > fleet.metadata.get("batch_context_limit", np.inf)), axis=1)
    knots = fleet.metadata.get("packing_context_tokens")
    kappa = np.interp(fleet.context, knots, timing["packing_kappa"]) if knots else timing["kappa"]
    duration = batch_time(r, fleet.t1, timing["beta"], np.where(long_context[:, None], 1., kappa), load * fleet.metadata.get("timing_load_factor", 1.))
    duration *= np.asarray(timing.get("regional_replay_factor", [1., 1.]))[route]
    release = deadline - duration
    kv_release = deadline - (np.interp(k.sum(1), [0, 1, 8], [0, *tails]) + k @ np.asarray(fleet.metadata.get("kv_partial_s", np.zeros(len(fleet.count))))) * np.exp(timing["beta"] * load * fleet.metadata.get("timing_load_factor", 1.))
    logs, state = r @ fleet.log, k @ fleet.kv
    rates = (logs + state) / deadline
    per_batch = np.minimum(endpoint[route], np.minimum(budgets[route], budgets[2]))
    kv_per_batch = np.minimum(per_batch, np.asarray(timing.get("regional_kv_bytes_per_s", endpoint[:2]))[route])
    r_commit, k_commit = logs / per_batch + duration, state / kv_per_batch + deadline - kv_release
    buffered = np.zeros(len(route))
    if "turn_sequences" in fleet.metadata:
        measured, cache = {**calibration(0), "forecast_load": load}, {}
        estimates = []
        for action, counts, speed in ((0, r, per_batch), (1, k, kv_per_batch)):
            rows = []
            for j, c in enumerate(counts):
                key = (action, int(route[j]), tuple(c))
                if key not in cache:
                    cache[key] = nominal_action(fleet, c, action, route[j], speed[j], timing, measured)
                rows.append(cache[key])
            estimates.append(np.array(rows).T)
        logs, duration, r_commit, r_buffer = estimates[0]
        state, k_work, k_commit, k_buffer = estimates[1]
        release, kv_release = deadline - duration, deadline - k_work
        buffered = r_buffer + k_buffer
        rates = (logs + state) / deadline
    eligible = ((duration <= deadline) & ((logs == 0) | (release > 0)) & (kv_release >= 0)
                & ((state == 0) | (kv_release > 0)) & (r_commit <= deadline + 1e-12)
                & (k_commit <= deadline + 1e-12) & (rates <= per_batch * (1 + 1e-12)))
    total = r + k
    row_masks = np.array([route == j for j in (0, 1)])
    compute = duration + deadline - kv_release
    loss = timing.get("resident_replay_loss", 0.)
    debt = load * (loss * duration + deadline - kv_release) + buffered
    service_time = ((1 - load * (1 - loss)) * duration + deadline - kv_release + buffered
                    + (r @ fleet.demand) * np.maximum(deadline - r_commit, 0)
                    + (k @ fleet.demand) * np.maximum(deadline - k_commit, 0))
    matrix = np.vstack((total.T, row_masks * compute, row_masks * (total @ fleet.demand),
                        row_masks * (total @ fleet.memory_tokens), row_masks * rates, row_masks * (state / deadline), rates))
    gpus = destination_gpus(fleet)
    capacities = np.r_[fleet.count, [gpus * deadline] * 2, [gpus * (1 - load)] * 2,
                       [(fleet.kv_capacity - fleet.baseline_kv) * gpus / fleet.gpus] * 2, budgets[:2],
                       np.asarray(timing.get("regional_kv_bytes_per_s", endpoint[:2])) * network_nodes(fleet)[:2], budgets[2]]
    if fleet.metadata.get("protect_resident"):
        service_time = compute + buffered + (r @ fleet.demand) * np.maximum(deadline - r_commit, 0) + (k @ fleet.demand) * np.maximum(deadline - k_commit, 0)
        matrix = np.vstack((matrix, row_masks * service_time))
        capacities = np.r_[capacities, [gpus * (1 - load) * deadline] * 2]
        debt = compute + buffered
    if np.any(capacities < 0):
        raise ValueError("resident state exceeds pooled KV capacity")
    return Table(fleet, r, k, route, duration, release, kv_release, logs, state, rates, eligible, fastest,
                 matrix, capacities, total @ fleet.gain, deadline, load, endpoint, budgets, timing,
                 debt, np.maximum(r_commit, k_commit), service_time)


def policy_mask(table, policy):
    mask = table.eligible.copy()
    if policy == "kv_only":
        mask &= table.replay.sum(1) == 0
    elif policy == "replay_only":
        mask &= table.kv.sum(1) == 0
    elif policy == "isolated_fastest":
        mask &= ~np.any((table.replay > 0) & ~table.fastest, axis=1)
        mask &= ~np.any((table.kv > 0) & table.fastest, axis=1)
    elif policy not in POLICIES:
        raise ValueError(policy)
    return mask


def _bounded_lp(cost, matrix, rhs, upper, certificate=None):
    matrix = csr_matrix(matrix)
    attempts = (("simplex", 0), ("ipm", 0), ("simplex", 2))
    for attempt, (algorithm, scaling) in enumerate(attempts):
        model, solver = highspy.HighsLp(), highspy.Highs()
        model.num_col_, model.num_row_ = matrix.shape[1], matrix.shape[0]
        model.col_cost_, model.col_lower_, model.col_upper_ = cost, np.zeros(len(cost)), upper
        model.row_lower_, model.row_upper_ = np.full(len(rhs), -highspy.kHighsInf), rhs
        model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
        for key, value in {"output_flag": False, "threads": 1, "solver": algorithm, "presolve": "on",
                           "simplex_scale_strategy": scaling, "small_matrix_value": 1e-12,
                           "primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-9}.items():
            if solver.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected option {key}")
        status = solver.passModel(model)
        if status == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS rejected the LP model")
        if status == highspy.HighsStatus.kWarning:
            warnings.warn("HiGHS model import warning; original constraints are checked after solving", RuntimeWarning, stacklevel=2)
        status = solver.run()
        failure = solver.modelStatusToString(solver.getModelStatus())
        if status == highspy.HighsStatus.kOk and solver.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            result = np.asarray(solver.getSolution().col_value)
            failure = certificate(result) if certificate is not None else None
            if failure is None:
                return result
        if attempt + 1 < len(attempts):
            warnings.warn(f"HiGHS {algorithm} failed: {failure}; retrying the same LP with {'IPM' if attempt == 0 else 'scaled simplex'}", RuntimeWarning, stacklevel=2)
    raise RuntimeError(failure)


def solve_lp(table, allowed, objective, primary=None):
    chosen = np.zeros(len(table.gains))
    ids = np.flatnonzero(allowed & ~np.any((table.matrix > 0) & (table.capacities[:, None] == 0), axis=0))
    if not len(ids):
        if primary is not None and primary > PRIMARY_TOL:
            raise RuntimeError("positive primary objective has no feasible variables")
        return chosen
    scale = np.where(table.capacities > 0, table.capacities, 1)
    matrix = table.matrix[:, ids] * table.fleet.gpus / scale[:, None]
    column_scale = np.maximum(matrix.max(0), 1.)
    cost = objective[ids] * table.fleet.gpus / column_scale
    constraints, limits = matrix / column_scale, (table.capacities > 0).astype(float)
    upper = np.min(np.divide(limits[:, None], constraints, out=np.full_like(constraints, np.inf), where=constraints > 0), axis=0)
    if primary is not None:
        primary_row = table.gains[ids] * table.fleet.gpus / column_scale
        primary_scale = max(abs(primary_row).max(), 1e-30)
        constraints = np.vstack((constraints, -primary_row / primary_scale))
        # Keep the secondary face away from a numerically singular boundary; bound absolute shed loss.
        limits = np.r_[limits, -max(0., primary - PRIMARY_TOL) / primary_scale + min(PRIMARY_ROW_TOL, PRIMARY_TOL / primary_scale)]
    def certificate(result):
        if not np.isfinite(result).all() or np.min(result) < -1e-9:
            return "LP returned invalid replica fractions"
        chosen[ids] = np.maximum(result, 0) * table.fleet.gpus / column_scale
        if np.max((table.matrix @ chosen - table.capacities) / np.maximum(table.capacities, 1)) > 1e-8:
            return "LP returned an infeasible resource allocation"
        if primary is not None and table.gains @ chosen < primary - PRIMARY_TOL - 1e-8:
            return "LP failed to preserve the primary objective"
    result = _bounded_lp(cost / max(abs(cost).max(), 1e-30), constraints, limits, upper, certificate)
    failure = certificate(result)
    if failure is not None:
        raise RuntimeError(failure)
    return chosen


def optimal_kv_range(table):
    chosen = select(table, "queue_haul")
    primary, kv_gain = float(table.gains @ chosen), table.kv @ table.fleet.gain
    endpoints = [solve_lp(table, policy_mask(table, "queue_haul"), sign * kv_gain, primary) for sign in (1, -1)]
    for endpoint in endpoints:
        certify(table, endpoint)
        if abs(table.gains @ endpoint - primary) > 1e-8:
            raise RuntimeError("KV diagnostic changed the primary optimum")
    return {"planned_shed_fraction": primary, "selected_kv_fraction": float(kv_gain @ chosen),
            "minimum_kv_fraction": float(kv_gain @ endpoints[0]), "maximum_kv_fraction": float(kv_gain @ endpoints[1]),
            "primary_shed_tolerance": 2 * PRIMARY_TOL,
            "scope": "primary optimum within numerical tolerance in the planning model; not an execution guarantee"}


def select(table, policy):
    allowed = policy_mask(table, policy)
    if policy != "greedy":
        chosen = solve_lp(table, allowed, -table.gains)
        return solve_lp(table, allowed, table.debt, float(table.gains @ chosen)) if table.debt.any() else chosen
    chosen = np.zeros(len(table.gains))
    if not allowed.any():
        return chosen
    scale = np.where(table.capacities > 0, table.capacities, 1)
    matrix = table.matrix * table.fleet.gpus / scale[:, None]
    limits = (table.capacities > 0).astype(float)
    gains = table.gains * table.fleet.gpus
    if policy == "greedy":
        remaining = limits.copy()
        for _ in range(len(limits) + 1):
            feasible = allowed & ~np.any((matrix > 0) & (remaining[:, None] <= 1e-10), axis=0)
            if not feasible.any():
                break
            prices = matrix / np.maximum(remaining[:, None], 1e-30)
            n = len(table.fleet.count)
            groups = [(0, n), (n, n+2), (n+2, n+4), (n+4, n+6), (n+6, n+11)]
            if table.require_recovery or table.fleet.metadata.get("protect_resident"):
                groups.append((n+11, n+13))
            cost = sum(prices[a:b].max(0) for a, b in groups)
            cost += table.debt / ((1 - table.load) * table.deadline)
            score = np.where(feasible, gains / np.maximum(cost, 1e-30), -np.inf)
            j = int(np.argmax(score))
            twin = (j + len(gains) // 2) % len(gains)
            ids = [j, twin] if feasible[twin] and np.isclose(score[j], score[twin], rtol=1e-12, atol=0) else [j]
            direction = matrix[:, ids].sum(1)
            take = np.min(remaining[direction > 0] / direction[direction > 0])
            chosen[ids] += take
            remaining = np.maximum(remaining - take * direction, 0)
        else:
            raise RuntimeError("greedy failed to exhaust a resource per iteration")
    return chosen * table.fleet.gpus


def certify(table, chosen):
    chosen = np.asarray(chosen)
    if chosen.shape != table.gains.shape or not np.isfinite(chosen).all() or np.any(chosen < -1e-10):
        raise ValueError("invalid pattern multiplicities")
    if np.any(chosen[~table.eligible] > 1e-10):
        raise ValueError("selected an ineligible batch schedule")
    used = table.matrix @ chosen
    residual = float(np.max((used - table.capacities) / np.maximum(table.capacities, 1)))
    if residual > 1e-8:
        raise RuntimeError("schedule exceeds pooled resources")
    active = chosen > 1e-10
    action_counts, action_fractions = [], []
    for route in (0, 1):
        for action in (table.replay, table.kv):
            counts = chosen[table.route == route] @ action[table.route == route]
            action_counts.append(float(counts.sum()))
            action_fractions.append(float(counts @ table.fleet.gain))
    return {"shed_fraction": float(table.gains @ chosen), "action_counts": action_counts,
            "action_fractions": action_fractions, "completed_sessions": sum(action_counts),
            "max_relative_residual": max(0., residual),
            "resource_utilization": (used / np.maximum(table.capacities, 1e-30)).tolist(),
            "binding_rows": np.flatnonzero((table.capacities > 0) & np.isclose(used, table.capacities, rtol=1e-7, atol=0)).tolist(),
            "batch_replica_seconds": [float((chosen * table.duration)[table.route == r].sum()) for r in (0, 1)],
            "forecast_induced_serving_work_s": float(table.debt @ chosen),
            "forecast_service_volume_deficit_work_s": [max(0., float((chosen * table.service_time)[table.route == r].sum())
                - destination_gpus(table.fleet) * (1 - table.load) * table.deadline) for r in (0, 1)],
            "scope": "nominal volume/work plan including catch-up and shared service occupancy; no execution certificate",
            "patterns": [{"column": int(j), "multiplicity": float(chosen[j]), "route": int(table.route[j]),
                          "replay_counts": table.replay[j].tolist(), "kv_counts": table.kv[j].tolist(),
                          "replay_release_s": float(table.release[j]), "batch_duration_s": float(table.duration[j]),
                          "kv_completion_start_s": float(table.kv_release[j]),
                          "reserved_bytes_per_s": float(table.rate[j])} for j in np.flatnonzero(active)]}


def compare(table):
    results = {policy: certify(table, select(table, policy)) for policy in POLICIES}
    for policy, result in results.items():
        result["solver_status"] = "greedy_planned" if policy == "greedy" else "optimal_nominal_plan"
    qh = results["queue_haul"]["shed_fraction"]
    if any(r["shed_fraction"] > qh + 1e-8 for r in results.values()):
        raise RuntimeError("QH LP is below a feasible baseline in the same schedule library")
    return results


def configuration(smoke=False):
    return {"schema": SCHEMA, "gpus": GPUS, "installed_gpu_w": GPUS * 300, "source_load": SOURCE_LOAD,
            "workloads": ["coding", "coding_long"],
            "solver_version": highspy.Highs().version(),
            "gpus_per_node": 8,
            "dispatch_chunks": DISPATCH_CHUNKS,
            "planning_iterations": PLANNING_ITERATIONS, "planning_resolution": PLANNING_RESOLUTION,
            "require_recovery": False,
            "resident_loads": [.25, .95] if smoke else list(LOADS),
            "deadlines": [1, 10, 60] if smoke else list(DEADLINES),
            "wan_gbps": [40] if smoke else ["reference", 40, 100, 400, 1000],
            "snapshots": 1 if smoke else 4, "draws": 1 if smoke else 8, "seed": 2001}


def cells(config):
    snapshots = [(w, i) for w in config["workloads"] for i in range(config["snapshots"] if w == "coding" else 1)]
    return list(product(snapshots, config["resident_loads"], range(config["draws"] + 1), config["wan_gbps"], config["deadlines"]))


def provenance(c):
    paths = [Path(__file__), ROOT / "pool_shed_calibration.py", ROOT / "pool_shed_execution.py", ROOT / "pool_shed_planner.py", ROOT / "plot_style.py", NETWORK, MANIFEST]
    return {**c["sources"], **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def prepare(out, smoke=False, resident_loads=None, snapshots=None, draws=None, wan_gbps=None, gpus_per_node=None, require_recovery=False):
    started = time.perf_counter()
    config = configuration(smoke)
    config["require_recovery"] = require_recovery
    if require_recovery:
        raise ValueError("the legacy static recovery constraint does not certify dynamic recovery or resident SLOs")
    for key, value in (("resident_loads", resident_loads), ("snapshots", snapshots), ("draws", draws), ("gpus_per_node", gpus_per_node)):
        if value is not None:
            config[key] = value
    if wan_gbps is not None:
        config["wan_gbps"] = ["reference", *wan_gbps]
    if (not isinstance(config["gpus_per_node"], int) or isinstance(config["gpus_per_node"], bool)
            or config["gpus_per_node"] < 1 or config["snapshots"] < 1 or config["draws"] < 0 or not config["resident_loads"]
            or len(set(config["resident_loads"])) != len(config["resident_loads"])
            or any(not 0 <= u < 1 for u in config["resident_loads"])
            or len(set(config["wan_gbps"])) != len(config["wan_gbps"])
            or any(not np.isfinite(w) or w <= 0 for w in config["wan_gbps"] if w != "reference")):
        raise ValueError("invalid campaign grid")
    c = calibration(config["draws"])
    rng = np.random.default_rng(config["seed"])
    indices = [-1, *rng.integers(len(network_samples()), size=config["draws"]).tolist()]
    sources = provenance(c)
    identity = digest({"config": config, "sources": sources})
    plan = {"identity": identity, "config": config, "sources": sources, "calibration": c,
            "network_indices": indices, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "assumptions": ["continuous pooled populations; optimum only within the common finite batch library",
                            "all methods receive the same queue/phase feedback; future rates use central calibration, not hidden execution draws",
                            "source swedencentral; equal-size eastus2 and germanywestcentral destinations",
                            "resident load is a fraction of the measured coding normal service envelope using its original contextual phase rates; no FLOP or busy-time interpretation",
                            "coding and 24K-plus coding cohorts use recorded trajectories within measured replay support; serving-context extrapolation and mixture transfers are explicit",
                            "eight resident sessions/GPU at every load; no invented initial migration backlog",
                            "paced recorded trajectories with seeded stratified phases, reset on wrap; request-duration proxies are separate from idle spacing; paused requests queue at destination after handoff",
                            "ongoing serving and KV are pooled; no discrete placement or local fragmentation model",
                            "KV ingest and GPU shutdown omitted; sealed KV, partial tails and live catch-up charged",
                            "KV response/validation tail is measured separately; batch/load overlap is a transfer assumption",
                            "receding-horizon LP uses time-indexed phase profiles and projected resident and source-buffer recovery; iterative approximation, not a global execution optimum",
                            "replay displaces resident throughput using the existing measured loss; spare compute repays debt before migrated request buffers; WAN-only transfer occupies no compute",
                            "handoff is primary, induced resident deficit and buffered work secondary; service recovery and resident latency validity are separate outputs, not handoff guarantees",
                            "replay and KV completion share pooled compute; contextual service load converts to the original offered-work reference for measured timing slowdown; neither is measured GPU utilization",
                            "common bounded-wave dispatcher maintains a bandwidth-based active window and prioritizes final deltas; each wave snapshots current completed source state at first dispatch and catches up relative to that snapshot",
                            "initial replay and changed-state catch-up submit full context at measured full-context timing, without an assumed retained-prefix hit; two bytes/token; KV uses loaded-runtime serialized geometry",
                            "WAN allocations are scenarios, not measurements of backbone capacity",
                            "measured single-A100-VM endpoints are pooled per node, shared by its GPUs and both destinations; eight GPUs/node is a transfer assumption",
                            "power uses measured context/rate-dependent phase-power calibration and bootstrap curves; partial shed allocates the modeled active-to-idle difference by released serving fraction"],
            "resource_rows": "one source-cohort row per state, then " + ", ".join(
                [f"{resource}_{route}" for resource in ("migration_replica_seconds", "serving_reference", "kv_tokens", "network", "kv_application_network")
                 for route in ("east", "germany")] + ["network_shared"]
                + ["shared_service_east", "shared_service_germany"]),
            "preparation_s": time.perf_counter() - started}
    if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text())["identity"] != identity:
        raise ValueError("existing output uses different inputs; choose a new directory")
    write_json(out / "plan.json", plan)
    return plan


def load_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if "inherited_checkpoints" in plan:
        raise ValueError("inherited checkpoint metadata must come from the pinned manifest")
    if plan["config"].get("solver_version") != highspy.Highs().version():
        raise ValueError("stale LP solver version")
    if plan["config"]["schema"] != SCHEMA or plan["sources"] != provenance(calibration(plan["config"]["draws"])):
        raise ValueError("stale schema, code, or calibration")
    if plan["identity"] != digest({"config": plan["config"], "sources": plan["sources"]}):
        raise ValueError("invalid plan identity")
    if "inherited_checkpoints_sha256" in plan:
        payload = (out / "inherited-checkpoints.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != plan["inherited_checkpoints_sha256"]:
            raise ValueError("inherited checkpoint manifest changed")
        inherited = json.loads(payload)
        if (inherited["parent_config"] != plan["config"] or inherited["parent_identity"] == plan["identity"]
                or inherited["parent_identity"] != digest({"config": inherited["parent_config"], "sources": inherited["parent_sources"]})):
            raise ValueError("invalid inherited execution provenance")
        expected = {f"{i:06d}.json.gz" for i in range(len(cells(plan["config"])))}
        if (not set(inherited["cells_sha256"]) <= expected
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                       for h in inherited["cells_sha256"].values())):
            raise ValueError("unexpected inherited checkpoint or checksum")
        plan["inherited_checkpoints"] = inherited
    return plan


def read_checkpoint(path, plan, expected):
    payload = path.read_bytes()
    value = json.loads(gzip.decompress(payload))
    inherited = plan.get("inherited_checkpoints", {})
    checksum = inherited.get("cells_sha256", {}).get(path.name)
    identity = inherited["parent_identity"] if checksum is not None else plan["identity"]
    if checksum is not None and hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("inherited checkpoint bytes changed")
    if value["identity"] != identity or value["cell"] != json.loads(json.dumps(expected)) or value["status"] != "complete":
        raise ValueError("invalid cell checkpoint")
    if set(value["results"]) != set(POLICIES):
        raise ValueError("missing policy")
    return value


@lru_cache(maxsize=64)
def forecast(workload, snapshot, gpus, gpus_per_node, load, wan, deadline, expanded=False, require_recovery=False):
    fleet = sample_fleet(workload, snapshot, gpus, gpus_per_node)
    samples = network_samples()
    endpoint = np.r_[np.median(samples[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    budgets = bandwidth(endpoint, fleet.nodes, wan)
    r, k = patterns(workload, snapshot, fleet.gpus, expanded)
    timing = calibration(0)["timing"][0]
    fastest = isolated_methods(fleet, load, endpoint, budgets, timing)
    r, k = include_isolated(r, k, fastest)
    table = schedule_table(fleet, r, k, load, deadline, endpoint, budgets, timing)
    if require_recovery:
        table = replace(table, matrix=np.vstack((table.matrix, np.array([table.route == r for r in (0, 1)]) * table.service_time)),
                        capacities=np.r_[table.capacities, [fleet.gpus * (1 - load) * deadline] * 2], require_recovery=True)
    choices = {policy: select(table, policy) for policy in POLICIES}
    certificates = {policy: certify(table, choice) for policy, choice in choices.items()}
    primary = certificates["queue_haul"]["shed_fraction"]
    if any(c["shed_fraction"] > primary + 1e-8 for c in certificates.values()):
        raise RuntimeError("LP violates baseline containment in the planning model")
    return table, choices, certificates


@lru_cache(maxsize=512)
def initial_admission(workload, snapshot, gpus, gpus_per_node, load, wan, deadline, expanded, policy, resolution, iterations):
    from pool_shed_execution import PooledExecution
    from pool_shed_planner import plan_admission

    table = forecast(workload, snapshot, gpus, gpus_per_node, load, wan, deadline, expanded)[0]
    measured = calibration(0)
    return plan_admission(PooledExecution(table, table.timing, measured), table, policy,
                          calibration=measured, resolution=resolution, iterations=iterations)


def execute_feedback(table, realized, policy, timing, measured, chunks=DISPATCH_CHUNKS,
                     resolution=PLANNING_RESOLUTION, iterations=PLANNING_ITERATIONS, initial=None):
    from pool_shed_execution import PooledExecution
    from pool_shed_planner import plan_admission

    engine = PooledExecution(realized, timing, measured, chunks)
    diagnostics, planning_s = [], 0.
    for _ in range(1024):
        if engine.now >= table.deadline:
            break
        remaining = table.fleet.count - (table.replay + table.kv).T @ engine.selected_total
        if remaining @ table.fleet.gain <= 1e-8:
            engine.advance(table.deadline)
            break
        started = time.perf_counter()
        chosen, until, audit = initial if engine.now == 0 and initial is not None else plan_admission(
            engine, table, policy, calibration=calibration(0), resolution=resolution, iterations=iterations)
        planning_s += time.perf_counter() - started
        if not engine.now < until <= table.deadline:
            raise RuntimeError("feedback planner failed to advance time")
        diagnostics.append({"time_s": engine.now, "admitted_shed_fraction": float(table.gains @ chosen), **audit})
        engine.admit(chosen)
        engine.advance(until)
    else:
        raise RuntimeError("feedback planner exceeded 1024 decisions")
    action_counts = [engine.selected_total[table.route == r] @ action[table.route == r]
                     for r in (0, 1) for action in (table.replay, table.kv)]
    return {**engine.result(),
            "admitted_shed_fraction": float(table.gains @ engine.selected_total),
            "admitted_action_counts": [float(counts.sum()) for counts in action_counts],
            "admitted_action_fractions": [float(counts @ table.fleet.gain) for counts in action_counts],
            "max_relative_residual": max((d["max_relative_residual"] for d in diagnostics), default=0.),
            "planning_steps": len(diagnostics), "planning_s": planning_s,
            "planning_diagnostics": diagnostics,
            "solver_status": "feedback_greedy" if policy == "greedy" else "receding_horizon_lp",
            "planning_scope": "common observed queue feedback; central future rates; iterative temporal approximation, no global execution optimality guarantee"}


def run_cell(plan, cell, expanded=False):

    (workload, snapshot), load, draw, wan, deadline = cell
    start = time.perf_counter()
    table, _, certificates = forecast(workload, snapshot, plan["config"]["gpus"],
        plan["config"]["gpus_per_node"], load, wan, deadline, expanded, plan["config"]["require_recovery"])
    build_s = time.perf_counter() - start
    endpoint = network_samples()[plan["network_indices"][draw]].copy() if draw else table.endpoint
    budgets = np.minimum(bandwidth(endpoint, table.fleet.nodes, wan), endpoint * network_nodes(table.fleet))
    realized = replace(table, endpoint=endpoint, budgets=budgets)
    start = time.perf_counter()
    results = {}
    for policy in POLICIES:
        initial_started = time.perf_counter()
        initial = initial_admission(workload, snapshot, plan["config"]["gpus"], plan["config"]["gpus_per_node"],
                                    load, wan, deadline, expanded, policy, plan["config"]["planning_resolution"],
                                    plan["config"]["planning_iterations"])
        initial_s = time.perf_counter() - initial_started
        executed = execute_feedback(table, realized, policy, plan["calibration"]["timing"][draw],
                                    plan["calibration"], plan["config"]["dispatch_chunks"],
                                    plan["config"]["planning_resolution"], plan["config"]["planning_iterations"], initial)
        executed["planning_s"] += initial_s
        results[policy] = {**executed, "initial_nominal_shed_fraction": certificates[policy]["shed_fraction"]}
    return {"identity": plan["identity"], "cell": list(cell), "status": "complete", "results": results,
            "columns": len(table.gains), "eligible_columns": int(table.eligible.sum()),
            "budgets_gbps": (budgets * 8e-9).tolist(), "endpoint_gbps": (endpoint * 8e-9).tolist(),
            "serving_ceiling": min(1., 2 * destination_gpus(table.fleet) * (1 - load) / (table.fleet.gpus * SOURCE_LOAD)),
            "build_s": build_s, "solve_evaluate_s": time.perf_counter() - start}


def run(out, shard=0, shards=1):
    started = time.perf_counter()
    plan = load_plan(out)
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    work = cells(plan["config"])
    variants = len(plan["config"]["wan_gbps"]) * len(plan["config"]["deadlines"])
    for index, cell in enumerate(work):
        scenario = index // ((plan["config"]["draws"] + 1) * variants) * variants + index % variants
        if scenario % shards != shard:
            continue
        path = out / "cells" / f"{index:06d}.json.gz"
        if path.exists():
            read_checkpoint(path, plan, cell)
            continue
        if path.name in plan.get("inherited_checkpoints", {}).get("cells_sha256", {}):
            raise ValueError("missing inherited checkpoint")
        write_json(path, run_cell(plan, cell))
        if index % 250 == 0:
            print(f"{index + 1}/{len(work)} cells; {time.perf_counter() - started:.1f}s", flush=True)
    write_json(out / f"runtime-{shard}.json", {"identity": plan["identity"], "shard": shard, "shards": shards,
                                              "wall_s": time.perf_counter() - started})


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def reduce(out):
    started = time.perf_counter()
    plan = load_plan(out)
    expected = cells(plan["config"])
    paths = sorted((out / "cells").glob("*.json.gz"))
    if [p.name for p in paths] != [f"{i:06d}.json.gz" for i in range(len(expected))]:
        raise ValueError("missing or unexpected cells")
    rows, groups, differences, regressions = [], {}, [], []
    maxima = {"residual": 0., "columns": 0, "initial_nominal_lp_loss": 0., "executed_lp_loss": 0.,
              "planning_steps": 0, "queue_iteration_residual": 0.}
    unsettled, decisions = 0, 0
    build_s = solve_s = 0.
    for i, path in enumerate(paths):
        value = read_checkpoint(path, plan, expected[i])
        (workload, snapshot), load, draw, wan, deadline = value["cell"]
        qh = value["results"]["queue_haul"]["shed_fraction"]
        nominal_qh = value["results"]["queue_haul"]["initial_nominal_shed_fraction"]
        maxima["columns"] = max(maxima["columns"], value["columns"])
        build_s += value["build_s"]
        solve_s += value["solve_evaluate_s"]
        for policy, result in value["results"].items():
            maxima["residual"] = max(maxima["residual"], result["max_relative_residual"])
            maxima["initial_nominal_lp_loss"] = max(maxima["initial_nominal_lp_loss"], result["initial_nominal_shed_fraction"] - nominal_qh)
            maxima["planning_steps"] = max(maxima["planning_steps"], result["planning_steps"])
            iteration_residual = max((d["fixed_point_residual"] for d in result["planning_diagnostics"]), default=0.)
            maxima["queue_iteration_residual"] = max(maxima["queue_iteration_residual"], iteration_residual)
            decisions += result["planning_steps"]
            unsettled += sum(d["fixed_point_residual"] >= 1e-3 for d in result["planning_diagnostics"])
            maxima["executed_lp_loss"] = max(maxima["executed_lp_loss"], result["shed_fraction"] - qh)
            if (not np.isfinite(result["shed_fraction"]) or result["shed_fraction"] < -1e-8
                    or result["shed_fraction"] > value["serving_ceiling"] + 1e-8
                    or result["shed_fraction"] > result["admitted_shed_fraction"] + 1e-8):
                raise ValueError("invalid shed or serving ceiling")
            if not np.allclose(np.array(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"],
                               result["pending_resident_debt_work_s"], rtol=1e-8, atol=1e-6):
                raise ValueError("resident debt conservation failed")
            if not np.allclose(np.array(result["resident_displaced_work_s"]) - result["resident_pool_compensation_work_s"],
                               result["resident_debt_generated_work_s"], rtol=1e-8, atol=1e-6):
                raise ValueError("pooled resident compensation conservation failed")
            if result["resident_latency_validated"]:
                raise ValueError("aggregate queue accounting cannot validate resident latency")
            if result["last_completion_s"] > deadline + 1e-8:
                raise ValueError("handoff after deadline")
            row = {"workload": workload, "snapshot": snapshot, "load": load, "draw": draw, "wan_gbps": wan,
                   "resident_latency_validated": False, "service_recovered_by_deadline": result["service_recovered_by_deadline"],
                   "service_recovery_scope": result["service_recovery_scope"],
                   "deadline_s": deadline, "policy": policy, "shed_fraction": result["shed_fraction"],
                   "admitted_shed_fraction": result["admitted_shed_fraction"],
                   "admitted_unfinished_fraction": result["admitted_shed_fraction"] - result["shed_fraction"],
                   "initial_nominal_shed_fraction": result["initial_nominal_shed_fraction"],
                   "last_completion_s": result["last_completion_s"], "unfinished_batch_mass": result["unfinished_batch_mass"],
                   "action_counts": result["action_counts"], "action_fractions": result["action_fractions"],
                   "admitted_action_counts": result["admitted_action_counts"], "admitted_action_fractions": result["admitted_action_fractions"],
                   "buffered_requests": result["buffered_requests"], "pending_buffered_requests": result["pending_buffered_requests"],
                   "pending_resident_debt_work_s": sum(result["pending_resident_debt_work_s"]),
                   "resident_debt_generated_work_s": sum(result["resident_debt_generated_work_s"]),
                   "resident_displaced_work_s": sum(result["resident_displaced_work_s"]),
                   "resident_pool_compensation_work_s": sum(result["resident_pool_compensation_work_s"]),
                   "peak_migration_replicas": result["peak_migration_replicas"],
                   "pending_source_buffer_work_s": result["pending_source_buffer_work_s"],
                   "pending_destination_buffer_work_s": result["pending_backlog_reference_work_s"],
                   "service_ready_s": result["service_ready_s"],
                   "planning_steps": result["planning_steps"], "planning_s": result["planning_s"],
                   "maximum_queue_iteration_residual": iteration_residual,
                   "batch_replica_seconds": result["batch_replica_seconds"],
                   "serving_ceiling": value["serving_ceiling"], "qh_minus_policy_fraction": qh - result["shed_fraction"]}
            rows.append(row)
            groups.setdefault((workload, load, wan, deadline, policy), []).append(row)
    if maxima["residual"] > 1e-8 or maxima["initial_nominal_lp_loss"] > 1e-8:
        raise RuntimeError("campaign feasibility/dominance audit failed")
    curves = {}
    for row in rows:
        if row["policy"] == "queue_haul":
            curves.setdefault((row["workload"], row["snapshot"], row["load"], row["draw"], row["wan_gbps"]), []).append(row)
    for curve in curves.values():
        curve.sort(key=lambda row: row["deadline_s"])
        if np.any(np.diff([r["shed_fraction"] for r in curve]) < -1e-8):
            regressions.append(curve[0])
    power = {(w, s): sample_fleet(w, s, plan["config"]["gpus"]).metadata["source_power"] for w, s in {tuple(cell[0]) for cell in expected}}
    power_scale = plan["config"]["gpus"] / 1e6
    summary = []
    for key, values in groups.items():
        central = [r for r in values if r["draw"] == 0]
        sampled = [r for r in values if r["draw"] > 0] or central
        fractions = np.array([r["shed_fraction"] for r in sampled])
        mw = np.array([r["shed_fraction"] * power_scale * np.array(power[r["workload"], r["snapshot"]]["delta_draws_w"]) for r in sampled])
        central_mw = [r["shed_fraction"] * power_scale * power[r["workload"], r["snapshot"]]["delta_w"] for r in central]
        summary.append({**dict(zip(("workload", "load", "wan_gbps", "deadline_s", "policy"), key)),
                        "central_shed_mw": float(np.median(central_mw)),
                        "median_shed_fraction": float(np.median(fractions)),
                        "median_nameplate_equivalent_mw": float(np.median(fractions)) * plan["config"]["installed_gpu_w"] / 1e6,
                        **dict(zip(("p05_shed_mw", "median_shed_mw", "p95_shed_mw"), map(float, np.quantile(mw, [.05, .5, .95])))),
                        "action_counts_mean": np.mean([r["action_counts"] for r in sampled], axis=0).tolist(),
                        "action_fractions_mean": np.mean([r["action_fractions"] for r in sampled], axis=0).tolist(),
                        "admitted_action_fractions_mean": np.mean([r["admitted_action_fractions"] for r in sampled], axis=0).tolist(),
                        "admitted_shed_fraction": float(np.median([r["admitted_shed_fraction"] for r in values])),
                        "median_pending_resident_debt_work_s": float(np.median([r["pending_resident_debt_work_s"] for r in sampled])),
                        "median_pending_source_buffer_work_s": float(np.median([r["pending_source_buffer_work_s"] for r in sampled])),
                        "median_pending_destination_buffer_work_s": float(np.median([r["pending_destination_buffer_work_s"] for r in sampled])),
                        "service_ready_fraction": float(np.mean([r["service_ready_s"] is not None for r in sampled])),
                        "serving_ceiling": sampled[0]["serving_ceiling"],
                        "planning_steps_mean": float(np.mean([r["planning_steps"] for r in sampled])),
                        "workload_central_range_mw": [min(central_mw), max(central_mw)],
                        "timing_network_range_fraction": [float(fractions.min()), float(fractions.max())]})
        if key[-1] != "queue_haul":
            differences.append({**dict(zip(("workload", "load", "wan_gbps", "deadline_s", "policy"), key)),
                                "minimum_qh_gap_fraction": min(r["qh_minus_policy_fraction"] for r in values),
                                "median_qh_gap_fraction": float(np.median([r["qh_minus_policy_fraction"] for r in sampled]))})
    write_csv(out / "scenarios.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", differences)
    write_json(out / "dominance-audit.json", {"identity": plan["identity"], "cells": len(paths), "maxima": maxima,
                                            "scope": "initial static LP containment; feedback outcomes have no cross-policy optimality guarantee",
                                            "planning_decisions": decisions, "unsettled_queue_iterations": unsettled,
                                            "executed_deadline_regressions": len(regressions)})
    plot_start = time.perf_counter()
    plot(summary, out)
    plot_debt(summary, out, plan["config"]["gpus"])
    metadata = {"identity": plan["identity"], "cells": len(paths), "summary": summary,
                "calibration_evidence": plan["calibration"]["evidence"],
                "workloads": {f"{w}-{s}": sample_fleet(w, s, plan["config"]["gpus"]).metadata
                              for w, s in {cell[0] for cell in expected}},
                "power_scope": "linear allocation of measured phase-power active-to-idle delta at the source request cadence; model error and workload transfer remain; no shutdown",
                "interval_scope": "identical feedback rules and central forecast calibration, paired execution draws and workload snapshots; empirical sensitivity, not coverage of unmeasured fleet transfer error",
                "timing_s": {"preparation": plan["preparation_s"], "pattern_tables": build_s, "solves_evaluation": solve_s,
                             "plotting": time.perf_counter() - plot_start, "reduction": time.perf_counter() - started}}
    if "inherited_checkpoints" in plan:
        metadata["inherited_execution"] = {"identity": plan["inherited_checkpoints"]["parent_identity"],
            "cells": len(plan["inherited_checkpoints"]["cells_sha256"]),
            "new_cells": len(paths) - len(plan["inherited_checkpoints"]["cells_sha256"]),
            "parent_sources_sha256": digest(plan["inherited_checkpoints"]["parent_sources"]),
            "manifest_sha256": plan["inherited_checkpoints_sha256"]}
    write_json(out / "summary.json", metadata)
    return metadata


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    model_label = plot_style.MODEL_NAMES["openai/gpt-oss-20b"] + " / " + plot_style.AGENTIC_HARDWARE_NAMES["a100"]
    workloads = [w for w in WORKLOADS if any(r["workload"] == w for r in summary)]
    loads = sorted({r["load"] for r in summary})
    for wan in dict.fromkeys(r["wan_gbps"] for r in summary):
        fig, axes = plt.subplots(len(workloads), len(loads), squeeze=False, figsize=(3.2 * len(loads), 3 * len(workloads)), sharex=True, sharey=True)
        for ax, (workload, load) in zip(axes.flat, product(workloads, loads)):
            for policy in POLICIES:
                series = sorted((r for r in summary if (r["workload"], r["load"], r["wan_gbps"], r["policy"]) ==
                                 (workload, load, wan, policy)), key=lambda r: r["deadline_s"])
                x = [r["deadline_s"] for r in series]
                ax.plot(x, [r["median_shed_mw"] for r in series], color=plot_style.POLICY_COLORS[policy],
                        linestyle=plot_style.POLICY_LINESTYLES[policy], label=plot_style.POLICY_NAMES[policy])
                ax.fill_between(x, [r["p05_shed_mw"] for r in series], [r["p95_shed_mw"] for r in series],
                                color=plot_style.POLICY_COLORS[policy], alpha=.12)
            ax.set(title=f"{workload.replace('_', ' ')}; load {load:g}\nServing ceiling {min(1., 2 * (1 - load) / SOURCE_LOAD):.1%}", xscale="log", xlabel="Deadline (s)")
        fig.supylabel("Shed power proxy (MW)")
        network_label = "measured endpoint reference" if wan == "reference" else f"assumed shared WAN {wan / 1000:g} Tbit/s" if wan >= 1000 else f"assumed shared WAN {wan:g} Gbit/s"
        fig.suptitle(f"{model_label}; {network_label}; feedback-policy p05–p95 sensitivity\nHandoff attainment; fleet transfer assumptions apply")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=9)
        fig.tight_layout(rect=(.02, .08, 1, .90))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"shed-{wan}.{extension}", bbox_inches="tight")
        plt.close(fig)
    numeric_wans = [r["wan_gbps"] for r in summary if r["wan_gbps"] != "reference"]
    action_wans = dict.fromkeys(([40] if 40 in numeric_wans else []) + [max(numeric_wans) if numeric_wans else "reference"])
    for wan, load in product(action_wans, loads):
        fig, axes = plt.subplots(2 * len(workloads), 5, figsize=(16, 5.5 * len(workloads)), sharey=True)
        for ax, (workload, scope, policy) in zip(axes.flat, product(workloads, ("admitted", "completed"), POLICIES)):
            series = sorted((r for r in summary if (r["workload"], r["load"], r["wan_gbps"], r["policy"]) ==
                             (workload, load, wan, policy)), key=lambda r: r["deadline_s"])
            field = "admitted_action_fractions_mean" if scope == "admitted" else "action_fractions_mean"
            ax.stackplot([r["deadline_s"] for r in series], np.array([r[field] for r in series]).T,
                         labels=[plot_style.ACTION_NAMES[a] for a in ACTIONS], colors=[plot_style.ACTION_COLORS[a] for a in ACTIONS])
            ax.set(title=f"{workload.replace('_', ' ')}\n{scope}: {plot_style.POLICY_NAMES[policy]}", xscale="log", xlabel="Deadline (s)", ylim=(0, 1))
            ax.title.set_fontsize(8)
        fig.supylabel("Source workload fraction: admitted vs completed")
        network_label = f"{wan / 1000:g} Tbit/s" if isinstance(wan, (int, float)) and wan >= 1000 else f"{wan} Gbit/s" if wan != "reference" else "measured reference"
        fig.suptitle(f"Action breakdown; resident load {load:g}; WAN budget {network_label}\nIndependent pooled execution; fleet transfer assumptions apply")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
        fig.tight_layout(rect=(.02, .07, 1, .90))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"actions-{wan}-{load:g}.{extension}", bbox_inches="tight")
        plt.close(fig)


def plot_debt(rows, out, gpus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    workloads = [w for w in WORKLOADS if any(r["workload"] == w for r in rows)]
    wans = [w for w in (40, 1000) if any(r["wan_gbps"] == w for r in rows)]
    loads = [u for u in (.5, .95) if any(r["load"] == u for r in rows)]
    if not wans or not loads:
        return
    for field, label, name in (("median_pending_resident_debt_work_s", "Resident service deficit", "resident-debt"),
                               ("median_pending_source_buffer_work_s", "Source-owned buffered work", "source-buffer")):
        fig, axes = plt.subplots(len(workloads) * len(loads), len(wans), squeeze=False,
                                 figsize=(5 * len(wans), 3 * len(workloads) * len(loads)))
        for ax, (workload, load, wan) in zip(axes.flat, product(workloads, loads, wans)):
            for policy in POLICIES:
                series = sorted((r for r in rows if (r["workload"], r["load"], r["wan_gbps"], r["policy"]) ==
                                 (workload, load, wan, policy)), key=lambda r: r["deadline_s"])
                ax.plot([r["deadline_s"] for r in series], [r[field] / (2 * gpus) for r in series], **plot_style.policy_style(policy))
            ax.set(title=f"{workload.replace('_', ' ')}; load {load:g}; {wan} Gbit/s", xscale="log", xlabel="Shed deadline (s)")
        fig.supylabel(f"{label}\n(safe-capacity seconds / destination GPU)")
        fig.suptitle(f"{label} at the deadline; shared resident service and migration")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(.03, .04, 1, .95))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)


def plot_optimal_kv(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    fig, axes = plt.subplots(len(WORKLOADS), 2, figsize=(9, 3 * len(WORKLOADS)), sharex=True, sharey=True)
    for ax, (workload, wan) in zip(axes.flat, product(WORKLOADS, (40, 1000))):
        series = sorted((r for r in rows if (r["workload"], r["wan_gbps"], r["load"]) == (workload, wan, .5)), key=lambda r: r["deadline_s"])
        x = [r["deadline_s"] for r in series]
        ax.fill_between(x, [r["minimum_kv_fraction"] for r in series], [r["maximum_kv_fraction"] for r in series],
                        color=plot_style.ACTION_COLORS["kv_transfer"], alpha=.25, label="Same maximum planned shed; secondary cost unconstrained")
        ax.plot(x, [r["selected_kv_fraction"] for r in series], **plot_style.policy_style("queue_haul"))
        ax.set(title=f"{workload.replace('_', ' ')}; {wan} Gbit/s", xscale="log", xlabel="Deadline (s)", ylim=(0, 1))
    fig.supylabel("Source workload assigned to KV")
    fig.suptitle("Action ambiguity in the initial static LP; resident load 50%\nRanges do not certify equivalent executed completion")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.tight_layout(rect=(0, .07, 1, .91))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"optimal-kv-ranges.{extension}", bbox_inches="tight")
    plt.close(fig)


def plot_scale(scales, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, scope, title in zip(axes, ("fixed_total_wan", "fixed_wan_per_node"),
                               ("40 Gbit/s total WAN budget", "40 Gbit/s per node WAN budget")):
        rows = sorted((r for r in scales if r["scope"] == scope), key=lambda r: r["source_gpus"])
        for policy, key in (("queue_haul", "qh_shed_fraction"), ("replay_only", "replay_shed_fraction")):
            ax.plot([r["source_gpus"] for r in rows], [100 * r[key] for r in rows], marker="o", **plot_style.policy_style(policy))
        ax.set(title=title, xlabel="Source GPU count (8 GPUs/node)", xscale="log", ylim=(0, 105))
        ax.text(.04, .08, f"Largest fleet: QH migrates {rows[-1]['qh_kv_sessions']:,.0f} sessions via KV", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel("Removed source workload (%)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Scaling diagnostic: measured pack, 30-second deadline, 50% resident load")
    fig.text(.5, .02, "Proportional WAN is a scaling diagnostic, not a region-pair capacity measurement.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"scale-comparison.{extension}", bbox_inches="tight")
    plt.close(fig)


def resolution_check(c):
    rows, differences = [], []
    cases = (("measured_pack", .5, 1000, 30), ("coding", .5, 1000, 60),
             ("coding", .95, 40, 3600), ("coding", .5, 1000, 3600),
             ("coding_long", .5, 1000, 30), ("coding_long", .95, 1000, 300))
    settings = {"default": (DISPATCH_CHUNKS, PLANNING_RESOLUTION, PLANNING_ITERATIONS),
                "dispatch": (2 * DISPATCH_CHUNKS, PLANNING_RESOLUTION, PLANNING_ITERATIONS),
                "feedback": (DISPATCH_CHUNKS, PLANNING_RESOLUTION / 2, PLANNING_ITERATIONS),
                "iterations": (DISPATCH_CHUNKS, PLANNING_RESOLUTION, 2 * PLANNING_ITERATIONS),
                "combined": (2 * DISPATCH_CHUNKS, PLANNING_RESOLUTION / 2, 2 * PLANNING_ITERATIONS)}
    for workload, load, wan, deadline in cases:
        table, _, _ = forecast(workload, 0, GPUS, 8, load, wan, deadline)
        results = {}
        for name, (chunks, resolution, iterations) in settings.items():
            started = time.perf_counter()
            evaluated = {p: execute_feedback(table, table, p, c["timing"][0], c, chunks, resolution, iterations) for p in POLICIES}
            results[name] = {p: r["shed_fraction"] for p, r in evaluated.items()}
            rows.append({"workload": workload, "load": load, "wan_gbps": wan, "deadline_s": deadline,
                         "setting": name, "chunks": chunks, "planning_resolution": resolution, "iterations": iterations,
                         "seconds": time.perf_counter() - started, "shed_fraction": results[name],
                         "resident_debt_work_s_per_gpu": {p: sum(r["pending_resident_debt_work_s"]) / (2 * GPUS) for p, r in evaluated.items()},
                         "maximum_iteration_residual": max((d["fixed_point_residual"] for r in evaluated.values() for d in r["planning_diagnostics"]), default=0.)})
        for setting in settings.keys() - {"default"}:
            for policy in POLICIES:
                gap = lambda name: results[name]["queue_haul"] - results[name][policy]
                differences.append({"workload": workload, "load": load, "wan_gbps": wan, "deadline_s": deadline,
                                    "policy": policy, "setting": setting,
                                    "absolute_shed_difference": abs(results["default"][policy] - results[setting][policy]),
                                    "absolute_qh_gap_difference": abs(gap("default") - gap(setting))})
    maximum = max(max(r["absolute_shed_difference"], r["absolute_qh_gap_difference"]) for r in differences if r["setting"] == "dispatch")
    return {"rows": rows, "differences": differences, "default_chunks": DISPATCH_CHUNKS,
            "maximum_dispatch_shed_or_gap_difference": maximum, "gate_pass": maximum <= .02,
            "scope": "Six central replay/buffer/WAN cases including long contexts, all five policies. Dispatch refinement tests numerical sensitivity. The feedback setting jointly changes geometric decision anchors, reservation bins, and candidate starts; it is planning-grid sensitivity, not isolated feedback cadence. All policies share the decision rule, including observed recovery events, with state-dependent timestamps. Iteration changes test policy sensitivity. None bounds global optimality or hardware-transfer error."}


def validate(out):
    from pool_shed_execution import regional_execution_check
    from loaded_service_model import historical_execution_check

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    c = calibration(0)
    config = configuration(True)
    config.update(draws=0, resident_loads=[.25, .75, .95], deadlines=[1, 3, 10, 60], wan_gbps=[10, 40, 400])
    plan = {"identity": "validation", "config": config, "calibration": c, "network_indices": [-1]}
    errors = []
    for cell in cells(config):
        (workload, snapshot), load, _, wan, deadline = cell
        a, b = [forecast(workload, snapshot, config["gpus"], config["gpus_per_node"], load, wan, deadline, expanded)[2]
                for expanded in (False, True)]
        errors.append(abs(a["queue_haul"]["shed_fraction"] - b["queue_haul"]["shed_fraction"]))
    scales = []
    for scope, gpus in product(("fixed_total_wan", "fixed_wan_per_node"), (8, 64, 512, 4096, 66664)):
        plan["config"] = {**config, "gpus": gpus}
        wan = 40 if scope == "fixed_total_wan" else 40 * (gpus // config["gpus_per_node"])
        result = run_cell(plan, (("measured_pack", 0), .5, 0, wan, 30))["results"]
        qh = result["queue_haul"]
        scales.append({"scope": scope, "source_gpus": gpus, "gpus_per_node": config["gpus_per_node"],
                       "wan_gbps": wan, "qh_shed_fraction": qh["shed_fraction"],
                       "replay_shed_fraction": result["replay_only"]["shed_fraction"],
                       "qh_kv_sessions": sum(qh["action_counts"][1::2])})
    faces = []
    for workload, load, wan, deadline in product(WORKLOADS, (.5, .95), (40, 1000), (30, 60, 300, 3600)):
        table, _, _ = forecast(workload, 0, GPUS, 8, load, wan, deadline)
        faces.append({"workload": workload, "load": load, "wan_gbps": wan, "deadline_s": deadline, **optimal_kv_range(table)})
    write_csv(out / "optimal-kv-ranges.csv", faces)
    runtime_scaling = []
    for gpus in (6400, 64000, 640000):
        forecast.cache_clear()
        plan["config"] = {**config, "gpus": gpus}
        before = time.perf_counter()
        result = run_cell(plan, (("measured_pack", 0), .5, 0, 1000 * gpus / GPUS, 60))
        runtime_scaling.append({"gpus": gpus, "seconds": time.perf_counter() - before, "columns": result["columns"],
            "completion_events": sum(len(r["completion_events"]) for r in result["results"].values()),
            "shed": {p: r["shed_fraction"] for p, r in result["results"].items()}})
    if any(abs(row["shed"][p] - runtime_scaling[0]["shed"][p]) > 1e-8 for row in runtime_scaling for p in POLICIES):
        raise RuntimeError("proportional pooled scaling changed executed outcomes")
    report = {"sources": provenance(c), "solver_version": highspy.Highs().version(), "calibration": c["evidence"], "regional_fidelity": regional_check(c),
              "regional_execution": regional_execution_check(c), "loaded_execution": loaded_execution_check(c),
              "regional_components": c["regional_components"]["validation"],
              "library_audit_cells": len(errors), "optimal_kv_ranges": faces, "runtime_scaling": runtime_scaling,
              "runtime_scaling_scope": "Proportional GPU/WAN scaling above the per-batch endpoint bottleneck boundary; 100-fold fleet range with unchanged nominal action costs",
              "scale_comparison": scales,
              "scale_scope": "Measured-pack workload, 30s deadline, .5 load; fixed WAN vs constant network/compute ratio. Scaled budgets are diagnostics, not inferred WAN allocations.",
              "library_p95_difference_fraction": float(np.quantile(errors, .95)),
              "library_max_difference_fraction": max(errors), "seconds": time.perf_counter() - started,
              "scope": "library sensitivity, not a bound on global scheduling optimality"}
    report["historical_execution"] = historical_execution_check(c)
    report["resident_service"] = c["resident_service"]
    report["resident_interference"] = c["resident_interference"]
    report["resident_execution"] = resident_execution_check(c, report["regional_execution"])
    report["execution_resolution"] = resolution_check(c)
    report["seconds"] = time.perf_counter() - started
    write_json(out / "validation.json", report)
    plot_scale(scales, out)
    plot_optimal_kv(faces, out)
    if np.quantile(errors, .95) > .01 or max(errors) > .02:
        raise RuntimeError("batch-library sensitivity exceeds the promotion gate")
    if not report["regional_execution"]["gate_pass"]:
        raise RuntimeError("pool timing fails the regional hardware holdout; see validation.json")
    if not report["historical_execution"]["gate_pass"]:
        raise RuntimeError("historical engine reproduction failed; see validation.json")
    if not report["loaded_execution"]["gate_pass"]:
        raise RuntimeError("pool timing fails the loaded replay holdout; see validation.json")
    if not report["execution_resolution"]["gate_pass"]:
        raise RuntimeError("dispatch-resolution sensitivity exceeds two percentage points; see validation.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "reduce", "validate"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resident-loads", type=float, nargs="+")
    parser.add_argument("--wan-gbps", type=float, nargs="+")
    parser.add_argument("--snapshots", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--require-recovery", action="store_true", help="legacy static option; dynamic recovery is modeled and reported separately")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command != "prepare" and (args.require_recovery or any(v is not None for v in (args.resident_loads, args.wan_gbps, args.snapshots, args.draws, args.gpus_per_node))):
        parser.error("grid overrides apply only to prepare")
    if args.command == "prepare":
        plan = prepare(args.out, args.smoke, args.resident_loads, args.snapshots, args.draws, args.wan_gbps, args.gpus_per_node, args.require_recovery)
        print(f"Prepared {len(cells(plan['config']))} cells")
    elif args.command == "run":
        run(args.out, args.shard, args.shards)
    elif args.command == "reduce":
        reduce(args.out)
    else:
        print(json.dumps(validate(args.out), indent=2))


if __name__ == "__main__":
    main()
