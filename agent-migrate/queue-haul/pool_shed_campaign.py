"""Measured batch schedules for a pooled 20 MW installed GPT-OSS/A100 fleet."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from pool_shed_calibration import calibration, replay_seconds

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/a100-batch-shed"
NETWORK = ROOT / "outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json"
MANIFEST = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
SCHEMA = "queue-haul-a100-batch-shed-v1"
GPUS, SOURCE_LOAD = 66666, .8
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest")
ACTIONS = ("east_replay", "east_kv_transfer", "germany_replay", "germany_kv_transfer")
DEADLINES = (1, 3, 10, 30, 60, 120, 300, 600, 1800, 3600)
LOADS = (.25, .5, .75, .9, .95)


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


def bandwidth(endpoint, gpus, wan_gbps):
    if gpus < 1 or np.any(np.asarray(endpoint) <= 0):
        raise ValueError("invalid endpoint capacity")
    return endpoint.copy() if wan_gbps == "reference" else np.minimum(endpoint * gpus, float(wan_gbps) * 1e9 / 8)


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

    @property
    def gain(self):
        return self.demand / (SOURCE_LOAD * self.gpus)

    @property
    def baseline_kv(self):
        return float(self.count @ self.memory_tokens)

    @property
    def memory_tokens(self):
        return np.ceil(self.context / 16) * 16


@cache
def sample_fleet(workload, snapshot=0, gpus=GPUS):
    c = calibration(0)
    rng = np.random.default_rng(1001 + snapshot)
    if workload == "measured_pack":
        shapes = np.array([[context, 2048, 32] for context in (2048, 4096, 4096, 8192, 8192, 12288, 12288, 14336)], float)
        count = np.full(8, gpus)
        evidence = {"timing_scope": "measured_pack_and_background", "excluded_states": 0}
    elif workload == "coding":
        raw = json.loads(MANIFEST.read_text())
        ids = sum(raw["manifest"]["splits"]["coding"].values(), [])
        rows = [r for r in raw["traces"] if r["session_id"] in ids]
        lo, hi = min(c["replay_context_tokens"]), max(c["replay_context_tokens"])
        supported = [r for r in rows if lo <= r["input_tokens_total"] - r["newly_append_tokens"] <= hi
                     and r["newly_append_tokens"] + r["output_tokens"] > 0]
        families = {key: [r for r in supported if r["session_id"] == key] for key in ids}
        families = {key: value for key, value in families.items() if value}
        if not families:
            raise ValueError("no coding states in the singleton calibration support")
        picked = [families[key][rng.integers(len(families[key]))] for key in rng.choice(sorted(families), 24)]
        shapes = np.array([(r["input_tokens_total"] - r["newly_append_tokens"], r["newly_append_tokens"], r["output_tokens"]) for r in picked], float)
        count = rng.multinomial(gpus * 8, np.full(24, 1 / 24))
        evidence = {"timing_scope": "coding_background_and_subset_transfer", "sampled_states": picked,
                    "excluded_states": len(rows) - len(supported), "supported_states": len(supported),
                    "exclusion_reason": "context outside singleton support or no ongoing work"}
    else:
        raise ValueError(workload)
    context, prompt, output = shapes.T
    work = prompt / c["F"] + output / c["G"]
    cadence = SOURCE_LOAD * gpus / (count @ work)
    evidence.update(reference_rps=float(count.sum() / (count @ work)), source_session_rps=float(cadence),
                    reference_basis="derived_phase_normalized_reference_not_measured_saturation",
                    arrivals="equal paced session cadence; trace timestamps unavailable", initial_migration_queue=0,
                    batch_context_limit=c["batch_context_limit"],
                    packing_context_tokens=c["packing_context_tokens"] if workload == "coding" else None)
    return Fleet(count, context, prompt, output, replay_seconds(context, c),
                 np.ceil(context / c["kv_block_tokens"]) * c["kv_block_bytes"], 2 * context,
                 work * cadence, [list(range(i, i + 8)) for i in range(0, len(count), 8)],
                 gpus, gpus * c["kv_capacity_tokens"], evidence)


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
    return values[:, :n], values[:, n:]


@dataclass
class Table:
    fleet: Fleet
    replay: np.ndarray
    kv: np.ndarray
    route: np.ndarray
    duration: np.ndarray
    release: np.ndarray
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


def schedule_table(fleet, replay, kv, load, deadline, endpoint, budgets, timing):
    endpoint, budgets = np.asarray(endpoint), np.asarray(budgets)
    if (not 0 <= load < 1 or deadline <= 0 or np.any(endpoint <= 0) or np.any(budgets <= 0)
            or not np.isfinite(np.r_[deadline, load, endpoint, budgets]).all()
            or np.any(replay < 0) or np.any(kv < 0) or np.any(kv != np.floor(kv)) or replay.shape != kv.shape):
        raise ValueError("invalid scheduling inputs")
    r, k = np.tile(replay, (2, 1)), np.tile(kv, (2, 1))
    route = np.repeat([0, 1], len(replay))
    long_context = np.any((r > 0) & (fleet.context > fleet.metadata.get("batch_context_limit", np.inf)), axis=1)
    knots = fleet.metadata.get("packing_context_tokens")
    kappa = np.interp(fleet.context, knots, timing["packing_kappa"]) if knots else timing["kappa"]
    duration = batch_time(r, fleet.t1, timing["beta"], np.where(long_context[:, None], 1., kappa), load)
    release = deadline - duration
    logs, state = r @ fleet.log, k @ fleet.kv
    rates = np.divide(logs, release, out=np.zeros_like(logs), where=release > 0) + state / deadline
    eligible = (duration <= deadline) & ((logs == 0) | (release > 0)) & (rates <= endpoint[route] * (1 + 1e-12))
    total = r + k
    row_masks = np.array([route == j for j in (0, 1)])
    matrix = np.vstack((total.T, row_masks, row_masks * (total @ fleet.demand),
                        row_masks * (total @ fleet.memory_tokens), row_masks * rates, rates))
    capacities = np.r_[fleet.count, [fleet.gpus] * 2, [fleet.gpus * (1 - load)] * 2,
                       [fleet.kv_capacity - fleet.baseline_kv] * 2, budgets]
    if np.any(capacities < 0):
        raise ValueError("resident state exceeds pooled KV capacity")
    isolated_bw = max(min(endpoint[j], budgets[j], budgets[2]) for j in (0, 1))
    fastest = fleet.log / isolated_bw + fleet.t1 * np.exp(timing["beta"] * load) < fleet.kv / isolated_bw
    return Table(fleet, r, k, route, duration, release, logs, state, rates, eligible, fastest,
                 matrix, capacities, total @ fleet.gain, deadline, load, endpoint, budgets)


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


def select(table, policy):
    allowed = policy_mask(table, policy)
    chosen = np.zeros(len(table.gains))
    if not allowed.any():
        return chosen
    scale = np.where(table.capacities > 0, table.capacities, 1)
    matrix = table.matrix * table.fleet.gpus / scale[:, None]
    limits = (table.capacities > 0).astype(float)
    gains = table.gains * table.fleet.gpus
    if policy != "greedy":
        ids = np.flatnonzero(allowed)
        result = linprog(-gains[ids] / gains[ids].max(), A_ub=csr_matrix(matrix[:, ids]), b_ub=limits,
                         bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            raise RuntimeError(result.message)
        if np.min(result.x) < -1e-9:
            raise RuntimeError("LP returned negative replica fractions")
        chosen[ids] = np.maximum(result.x, 0)
    else:
        remaining = limits.copy()
        for _ in range(len(limits) + 1):
            feasible = allowed & ~np.any((matrix > 0) & (remaining[:, None] <= 1e-10), axis=0)
            if not feasible.any():
                break
            cost = np.sum(matrix / np.maximum(remaining[:, None], 1e-30), axis=0)
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


def execute(table, chosen):
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
    if (np.any(table.log_bytes[active] > np.maximum(table.release[active], 0) *
               (table.rate[active] - table.kv_bytes[active] / table.deadline) * (1 + 1e-8) + 1e-5)
            or np.any(table.release[active] < -1e-10)):
        raise RuntimeError("replay begins before its log transfer can finish")
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
            "last_completion_s": table.deadline if active.any() else 0.,
            "patterns": [{"column": int(j), "multiplicity": float(chosen[j]), "route": int(table.route[j]),
                          "replay_counts": table.replay[j].tolist(), "kv_counts": table.kv[j].tolist(),
                          "replay_release_s": float(table.release[j]), "batch_duration_s": float(table.duration[j]),
                          "reserved_bytes_per_s": float(table.rate[j])} for j in np.flatnonzero(active)]}


def compare(table):
    results = {policy: execute(table, select(table, policy)) for policy in POLICIES}
    for policy, result in results.items():
        result["solver_status"] = "greedy_feasible" if policy == "greedy" else "optimal_within_library"
    qh = results["queue_haul"]["shed_fraction"]
    if any(r["shed_fraction"] > qh + 1e-8 for r in results.values()):
        raise RuntimeError("QH LP is below a feasible baseline in the same schedule library")
    return results


def configuration(smoke=False):
    return {"schema": SCHEMA, "gpus": GPUS, "installed_gpu_w": GPUS * 300, "source_load": SOURCE_LOAD,
            "resident_loads": [.25, .95] if smoke else list(LOADS),
            "deadlines": [1, 10, 60] if smoke else list(DEADLINES),
            "wan_gbps": [40] if smoke else ["reference", 10, 40, 100, 400],
            "snapshots": 1 if smoke else 4, "draws": 1 if smoke else 8, "seed": 2001}


def cells(config):
    snapshots = [("measured_pack", 0)] + [("coding", i) for i in range(config["snapshots"])]
    return list(product(snapshots, config["resident_loads"], range(config["draws"] + 1), config["wan_gbps"], config["deadlines"]))


def provenance(c):
    paths = [Path(__file__), ROOT / "pool_shed_calibration.py", ROOT / "plot_style.py", NETWORK, MANIFEST]
    return {**c["sources"], **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def prepare(out, smoke=False, resident_loads=None, snapshots=None, draws=None, wan_gbps=None):
    started = time.perf_counter()
    config = configuration(smoke)
    for key, value in (("resident_loads", resident_loads), ("snapshots", snapshots), ("draws", draws)):
        if value is not None:
            config[key] = value
    if wan_gbps is not None:
        config["wan_gbps"] = ["reference", *wan_gbps]
    if (config["snapshots"] < 1 or config["draws"] < 0 or not config["resident_loads"]
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
                            "scenario-reoptimized sensitivity, not fixed-plan robustness or transfer-error confidence bounds",
                            "source swedencentral; equal-size eastus2 and germanywestcentral destinations",
                            "resident load uses derived phase-reference work, not FLOPs, busy time, or validated SLO capacity",
                            "coding uses measured singleton support and transferred batch/background response",
                            "eight resident sessions/GPU at every load; no invented initial migration backlog",
                            "paced arrivals; fixed context snapshots, resident weights; handoff at deadline",
                            "ongoing serving and KV are pooled; no discrete placement or local fragmentation model",
                            "KV ingest, catch-up, context growth, and shutdown are omitted",
                            "replay logs assume two bytes/token; KV uses loaded-runtime serialized geometry",
                            "WAN allocations are scenarios, not measurements of backbone capacity",
                            "power is linear workload-share allocation of direct coding active-to-awake-idle anchors"],
            "resource_rows": "one source-cohort row per state, then " + ", ".join(
                [f"{resource}_{route}" for resource in ("migration_replicas", "serving_reference", "kv_tokens", "network")
                 for route in ("east", "germany")] + ["network_shared"]),
            "preparation_s": time.perf_counter() - started}
    if (out / "plan.json").exists() and json.loads((out / "plan.json").read_text())["identity"] != identity:
        raise ValueError("existing output uses different inputs; choose a new directory")
    write_json(out / "plan.json", plan)
    return plan


def load_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if plan["config"]["schema"] != SCHEMA or plan["sources"] != provenance(calibration(plan["config"]["draws"])):
        raise ValueError("stale schema, code, or calibration")
    if plan["identity"] != digest({"config": plan["config"], "sources": plan["sources"]}):
        raise ValueError("invalid plan identity")
    return plan


def run_cell(plan, cell, expanded=False):
    (workload, snapshot), load, draw, wan, deadline = cell
    fleet = sample_fleet(workload, snapshot, plan["config"]["gpus"])
    samples = network_samples()
    endpoint = samples[plan["network_indices"][draw]].copy() if draw else np.r_[np.median(samples[:, :2], axis=0), 0.]
    if not draw:
        endpoint[2] = endpoint[:2].sum()
    budgets = bandwidth(endpoint, fleet.gpus, wan)
    r, k = patterns(workload, snapshot, fleet.gpus, expanded)
    start = time.perf_counter()
    table = schedule_table(fleet, r, k, load, deadline, endpoint, budgets, plan["calibration"]["timing"][draw])
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    results = compare(table)
    return {"identity": plan["identity"], "cell": list(cell), "status": "complete", "results": results,
            "columns": len(table.gains), "eligible_columns": int(table.eligible.sum()),
            "budgets_gbps": (budgets * 8e-9).tolist(), "endpoint_gbps": (endpoint * 8e-9).tolist(),
            "serving_ceiling": min(1., 2 * (1 - load) / SOURCE_LOAD),
            "build_s": build_s, "solve_evaluate_s": time.perf_counter() - start}


def run(out, shard=0, shards=1):
    started = time.perf_counter()
    plan = load_plan(out)
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    work = cells(plan["config"])
    for index, cell in enumerate(work):
        if index % shards != shard:
            continue
        path = out / "cells" / f"{index:06d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as handle:
                previous = json.load(handle)
            if previous["identity"] != plan["identity"] or previous["cell"] != json.loads(json.dumps(cell)):
                raise ValueError("stale cell checkpoint")
            continue
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
    maxima = {"residual": 0., "columns": 0, "lp_loss": 0.}
    build_s = solve_s = 0.
    for i, path in enumerate(paths):
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
        if value["identity"] != plan["identity"] or value["cell"] != json.loads(json.dumps(expected[i])) or value["status"] != "complete":
            raise ValueError("invalid cell checkpoint")
        (workload, snapshot), load, draw, wan, deadline = value["cell"]
        if set(value["results"]) != set(POLICIES):
            raise ValueError("missing policy")
        qh = value["results"]["queue_haul"]["shed_fraction"]
        maxima["columns"] = max(maxima["columns"], value["columns"])
        build_s += value["build_s"]
        solve_s += value["solve_evaluate_s"]
        for policy, result in value["results"].items():
            maxima["residual"] = max(maxima["residual"], result["max_relative_residual"])
            maxima["lp_loss"] = max(maxima["lp_loss"], result["shed_fraction"] - qh)
            if (not np.isfinite(result["shed_fraction"]) or result["shed_fraction"] < -1e-8
                    or result["shed_fraction"] > value["serving_ceiling"] + 1e-8):
                raise ValueError("invalid shed or serving ceiling")
            row = {"workload": workload, "snapshot": snapshot, "load": load, "draw": draw, "wan_gbps": wan,
                   "deadline_s": deadline, "policy": policy, "shed_fraction": result["shed_fraction"],
                   "action_counts": result["action_counts"], "action_fractions": result["action_fractions"],
                   "resource_utilization": result["resource_utilization"], "batch_replica_seconds": result["batch_replica_seconds"],
                   "serving_ceiling": value["serving_ceiling"], "qh_minus_policy_fraction": qh - result["shed_fraction"]}
            rows.append(row)
            groups.setdefault((workload, load, wan, deadline, policy), []).append(row)
    if maxima["residual"] > 1e-8 or maxima["lp_loss"] > 1e-8:
        raise RuntimeError("campaign feasibility/dominance audit failed")
    curves = {}
    for row in rows:
        if row["policy"] == "queue_haul":
            curves.setdefault((row["workload"], row["snapshot"], row["load"], row["draw"], row["wan_gbps"]), []).append(row)
    for curve in curves.values():
        curve.sort(key=lambda row: row["deadline_s"])
        if np.any(np.diff([r["shed_fraction"] for r in curve]) < -1e-8):
            regressions.append(curve[0])
    if regressions:
        raise RuntimeError("LP shed declined with deadline")
    power = np.array(plan["calibration"]["power_draws_w"]) * plan["config"]["gpus"] / 1e6
    central_power = plan["config"]["gpus"] * (plan["calibration"]["active_w"] - plan["calibration"]["idle_w"]) / 1e6
    summary = []
    for key, values in groups.items():
        central = [r for r in values if r["draw"] == 0]
        sampled = [r for r in values if r["draw"] > 0] or central
        fractions = np.array([r["shed_fraction"] for r in sampled])
        mw = fractions[:, None] * power
        summary.append({**dict(zip(("workload", "load", "wan_gbps", "deadline_s", "policy"), key)),
                        "central_shed_mw": float(np.median([r["shed_fraction"] for r in central]) * central_power),
                        "median_shed_fraction": float(np.median(fractions)),
                        **dict(zip(("p05_shed_mw", "median_shed_mw", "p95_shed_mw"), map(float, np.quantile(mw, [.05, .5, .95])))),
                        "action_counts_mean": np.mean([r["action_counts"] for r in sampled], axis=0).tolist(),
                        "action_fractions_mean": np.mean([r["action_fractions"] for r in sampled], axis=0).tolist(),
                        "serving_ceiling": sampled[0]["serving_ceiling"],
                        "resource_utilization_mean": np.mean([r["resource_utilization"] for r in sampled], axis=0).tolist(),
                        "workload_central_range_mw": [min(r["shed_fraction"] for r in central) * central_power,
                                                     max(r["shed_fraction"] for r in central) * central_power],
                        "timing_network_range_fraction": [float(fractions.min()), float(fractions.max())]})
        if key[-1] != "queue_haul":
            differences.append({**dict(zip(("workload", "load", "wan_gbps", "deadline_s", "policy"), key)),
                                "minimum_qh_gap_fraction": min(r["qh_minus_policy_fraction"] for r in values),
                                "median_qh_gap_fraction": float(np.median([r["qh_minus_policy_fraction"] for r in sampled]))})
    write_csv(out / "scenarios.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", differences)
    write_json(out / "dominance-audit.json", {"identity": plan["identity"], "cells": len(paths), "maxima": maxima,
                                            "deadline_regressions": 0})
    plot_start = time.perf_counter()
    plot(summary, out)
    metadata = {"identity": plan["identity"], "cells": len(paths), "summary": summary,
                "calibration_evidence": plan["calibration"]["evidence"],
                "workloads": {f"{w}-{s}": sample_fleet(w, s, plan["config"]["gpus"]).metadata
                              for w, s in {cell[0] for cell in expected}},
                "power_scope": "linear measured-anchor proxy; workload transfer; no shutdown",
                "interval_scope": "scenario-reoptimized workload/calibration sensitivity; not transfer-error or fixed-plan coverage",
                "timing_s": {"preparation": plan["preparation_s"], "pattern_tables": build_s, "solves_evaluation": solve_s,
                             "plotting": time.perf_counter() - plot_start, "reduction": time.perf_counter() - started}}
    write_json(out / "summary.json", metadata)
    return metadata


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    model_label = plot_style.MODEL_NAMES["openai/gpt-oss-20b"] + " / " + plot_style.AGENTIC_HARDWARE_NAMES["a100"]
    workloads = ("measured_pack", "coding")
    loads = sorted({r["load"] for r in summary})
    for wan in dict.fromkeys(r["wan_gbps"] for r in summary):
        fig, axes = plt.subplots(2, len(loads), squeeze=False, figsize=(3.2 * len(loads), 6), sharex=True, sharey=True)
        for ax, (workload, load) in zip(axes.flat, product(workloads, loads)):
            for policy in POLICIES:
                series = sorted((r for r in summary if (r["workload"], r["load"], r["wan_gbps"], r["policy"]) ==
                                 (workload, load, wan, policy)), key=lambda r: r["deadline_s"])
                x = [r["deadline_s"] for r in series]
                ax.plot(x, [r["median_shed_mw"] for r in series], color=plot_style.POLICY_COLORS[policy],
                        linestyle=plot_style.POLICY_LINESTYLES[policy], label=plot_style.POLICY_NAMES[policy])
                ax.fill_between(x, [r["p05_shed_mw"] for r in series], [r["p95_shed_mw"] for r in series],
                                color=plot_style.POLICY_COLORS[policy], alpha=.12)
            ax.set(title=f"{workload.replace('_', ' ')}; load {load:g}", xscale="log", xlabel="Deadline (s)")
        fig.supylabel("Shed power proxy (MW)")
        network_label = "measured endpoint reference" if wan == "reference" else f"assumed shared WAN {wan} Gbit/s"
        fig.suptitle(f"{model_label}; {network_label}; scenario-reoptimized p05–p95")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=9)
        fig.tight_layout(rect=(.02, .08, 1, .94))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"shed-{wan}.{extension}", bbox_inches="tight")
        plt.close(fig)
    wan = 40 if any(r["wan_gbps"] == 40 for r in summary) else summary[0]["wan_gbps"]
    for load in loads:
        fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=True)
        for ax, (workload, policy) in zip(axes.flat, product(workloads, POLICIES)):
            series = sorted((r for r in summary if (r["workload"], r["load"], r["wan_gbps"], r["policy"]) ==
                             (workload, load, wan, policy)), key=lambda r: r["deadline_s"])
            ax.stackplot([r["deadline_s"] for r in series], np.array([r["action_fractions_mean"] for r in series]).T,
                         labels=[plot_style.ACTION_NAMES[a] for a in ACTIONS], colors=[plot_style.ACTION_COLORS[a] for a in ACTIONS])
            ax.set(title=f"{workload.replace('_', ' ')}\n{plot_style.POLICY_NAMES[policy]}", xscale="log", xlabel="Deadline (s)", ylim=(0, 1))
        fig.supylabel("Removed source workload fraction")
        fig.suptitle(f"Action breakdown; resident load {load:g}; shared WAN {wan} Gbit/s")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9)
        fig.tight_layout(rect=(.02, .07, 1, .92))
        for extension in ("png", "pdf"):
            fig.savefig(out / f"actions-{load:g}.{extension}", bbox_inches="tight")
        plt.close(fig)


def validate(out):
    started = time.perf_counter()
    c = calibration(0)
    config = configuration(True)
    config.update(draws=0, resident_loads=[.25, .75, .95], deadlines=[1, 3, 10, 60], wan_gbps=[10, 40, 400])
    plan = {"identity": "validation", "config": config, "calibration": c, "network_indices": [-1]}
    errors = []
    for cell in cells(config):
        a, b = run_cell(plan, cell), run_cell(plan, cell, expanded=True)
        errors.append(abs(a["results"]["queue_haul"]["shed_fraction"] - b["results"]["queue_haul"]["shed_fraction"]))
    report = {"calibration": c["evidence"], "library_audit_cells": len(errors),
              "library_p95_difference_fraction": float(np.quantile(errors, .95)),
              "library_max_difference_fraction": max(errors), "seconds": time.perf_counter() - started,
              "scope": "library sensitivity, not a bound on global scheduling optimality"}
    write_json(out / "validation.json", report)
    if np.quantile(errors, .95) > .01 or max(errors) > .02:
        raise RuntimeError("batch-library sensitivity exceeds the promotion gate")
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
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command != "prepare" and any(v is not None for v in (args.resident_loads, args.wan_gbps, args.snapshots, args.draws)):
        parser.error("grid overrides apply only to prepare")
    if args.command == "prepare":
        plan = prepare(args.out, args.smoke, args.resident_loads, args.snapshots, args.draws, args.wan_gbps)
        print(f"Prepared {len(cells(plan['config']))} cells")
    elif args.command == "run":
        run(args.out, args.shard, args.shards)
    elif args.command == "reduce":
        reduce(args.out)
    else:
        print(json.dumps(validate(args.out), indent=2))


if __name__ == "__main__":
    main()
