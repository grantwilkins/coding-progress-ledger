"""GPT-OSS/A100 whole-session shed with pooled compute and ideal flow timing."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/a100-pool-shed"
MODELS = ("gpt-oss-20b",)
GPUS = int(20e6 / 300)
CALIBRATION = ROOT / "outputs/a100-parity-20260905/power"
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest")
ACTIONS = ("east_replay", "east_kv_transfer", "germany_replay", "germany_kv_transfer")
PREFILL_METRICS = ("prefill_contention_session_s", "prefill_peak_ready_sessions", "prefill_ready_at_deadline",
                   "prefill_network_blocked_at_deadline", "prefill_remaining_gpu_s")
DEADLINES = (1, 3, 10, 30, 60, 120, 300, 600, 1800, 3600)
NETWORK = ROOT / "outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json"
MANIFEST = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
SCHEMA = "queue-haul-a100-pool-shed-v1"
SOURCE_LOAD, BASELINE_LOAD = .8, .5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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
def calibrations(model):
    metadata = json.loads((CALIBRATION / "metadata.json").read_text())
    log = (CALIBRATION / "server.log").read_text()
    rows = [json.loads(line) for line in (CALIBRATION / "cells.jsonl").read_text().splitlines()]
    if (model != MODELS[0] or metadata["model"] != "openai/gpt-oss-20b"
            or metadata["gpu"]["name"] != "NVIDIA A100 80GB PCIe" or metadata["gpu"]["power_limit_w"] != 300
            or not metadata["optimized_runtime"] or "enforce_eager=False" not in log
            or [r["sequence"] for r in rows] != list(range(111)) or any(r["cached_prompt_tokens"] for r in rows)):
        raise ValueError("invalid optimized GPT-OSS/A100 raw calibration")
    keys = {(r["family"], r["prompt_tokens"], r["output_tokens"], r["concurrency"]) for r in rows}
    groups = {key: [r for r in rows if (r["family"], r["prompt_tokens"], r["output_tokens"], r["concurrency"]) == key] for key in sorted(keys)}
    F, G = [max(np.median([r[field] for r in group]) for key, group in groups.items()
                if key[0] == family and len(group) > 1)
            for family, field in (("prefill", "realized_prefill_tps"), ("decode", "realized_decode_tps"))]
    prefill = {"model": metadata["model"], "kv_capacity_tokens": int(re.search(r"GPU KV cache size: ([\d,]+) tokens", log)[1].replace(",", "")),
               "curve": [{"context_tokens": key[1], "prefill_tps_median": float(np.median([r["realized_prefill_tps"] for r in group]))}
                         for key, group in groups.items() if key[0] == "prefill" and key[3] == 1]}
    idle = [r["power_mean_w"] for r in rows if r["family"] == "idle" and r["sequence"] > 0]
    active = [group for key, group in groups.items() if key[0] == "campaign"]
    active.sort(key=lambda group: np.median([r["realized_prefill_tps"] / F + r["realized_decode_tps"] / G for r in group]))
    xs = [0., *[float(np.median([r["realized_prefill_tps"] / F + r["realized_decode_tps"] / G for r in group])) for group in active]]
    watts = [idle, *[[r["power_mean_w"] for r in group] for group in active]]
    if np.any(np.diff(xs) <= 0) or len(idle) != 2:
        raise ValueError("invalid empirical power support or warm-idle anchors")
    rng = np.random.default_rng(1)
    draws = Counter(tuple(float(rng.choice(values)) for values in watts) for _ in range(200))
    power = {"model": metadata["model"], "hardware": "A100", "F_prefill_tps": float(F), "G_decode_tps": float(G),
             "bootstrap_curve_counts": list(draws.values()),
             "phase_power": {"measured_power_curve": list(zip(xs, map(float, map(np.median, watts)))),
                             "measured_power_bootstrap": [list(zip(xs, values)) for values in draws]},
             "evidence": {"status": "empirical_anchors_only; pooled workload extrapolation provisional",
                          "runtime": "vLLM 0.22.0, optimized TP1, MXFP4 weights, BF16 KV, chunked prefill 8192",
                          "power_shape": [604, 64], "cold_start_idle_excluded": True,
                          "normalization": "maximum repeated-cell median achieved prefill/decode throughput",
                          "prior_rational_fit_status": json.loads((CALIBRATION / "fit.json").read_text())["status"]}}
    return prefill, power


def network_samples():
    data = json.loads(NETWORK.read_text())
    routes = np.array([data["paths"][r]["simultaneous_mbps"] for r in ("east", "germany")]).T
    shared = np.array(data["aggregate_simultaneous_mbps"])
    if not np.allclose(routes.sum(1), shared) or np.any(routes <= 0):
        raise ValueError("network repetitions must be positive and paired")
    return np.column_stack((routes, shared)) * 125_000


def bandwidth(endpoint, gpus, wan_gbps):
    """WAN allocations are independent of GPU count and measured TCP asymmetry."""
    budgets = endpoint.copy() if wan_gbps == "reference" else np.minimum(
        np.full(3, float(wan_gbps) * 1e9 / 8), endpoint * gpus)
    if not np.all(np.isfinite(budgets)) or np.any(budgets <= 0):
        raise ValueError("WAN budgets must be finite and positive")
    return budgets


def kv_bytes(model, context):
    """Same analytical BF16 state geometry as matched_action_campaign's source."""
    if model == "gpt-oss-20b":
        return 4 * 8 * 64 * (12 * context + 12 * np.minimum(context, 128))
    raise ValueError(model)


@dataclass
class Fleet:
    count: np.ndarray
    context: np.ndarray
    demand: np.ndarray
    replay: np.ndarray
    kv: np.ndarray
    log: np.ndarray
    gpus: int
    kv_capacity: float
    power_w: float
    idle_w: float
    metadata: dict

    @property
    def gain(self):
        return self.demand / SOURCE_LOAD * (self.power_w - self.idle_w)

    @property
    def baseline_kv(self):
        return float(self.count @ self.context) * BASELINE_LOAD / SOURCE_LOAD


def sample_fleet(model, workload, density, snapshot, gpus=GPUS):
    prefill, power = calibrations(model)
    curve = np.array([[r["context_tokens"], r["prefill_tps_median"]] for r in prefill["curve"]])
    rng = np.random.default_rng(1001 + snapshot)
    if workload == "coding":
        raw = json.loads(MANIFEST.read_text())
        ids = sum(raw["manifest"]["splits"]["coding"].values(), [])
        traces = [r for r in raw["traces"] if r["session_id"] in ids]
        supported = [r for r in traces if curve[0, 0] <= r["input_tokens_total"] - r["newly_append_tokens"] <= curve[-1, 0]
                     and r["newly_append_tokens"] + r["output_tokens"] > 0]
        families = {key: [r for r in supported if r["session_id"] == key] for key in ids}
        families = {key: value for key, value in families.items() if value}
        if not families:
            raise ValueError("no coding states within measured context support")
        picked = [families[key][rng.integers(len(families[key]))]
                  for key in rng.choice(sorted(families), len(ids))]
        shapes = np.array([(r["input_tokens_total"] - r["newly_append_tokens"],
                            r["newly_append_tokens"], r["output_tokens"]) for r in picked], float)
        logs, cadence = 2 * shapes[:, 0], np.ones(len(shapes))
        evidence = {"supported_states": len(supported), "excluded_states": len(traces) - len(supported),
                    "sampled_trajectories": len(ids), "log_density_assumed_bytes_per_token": 2}
    else:
        raw = json.loads((ROOT / f"profiles/{workload}.json").read_text())
        records = raw["records"]
        shapes = np.array([(r["context_tokens"], r["prompt_tokens"], r["output_tokens"]) for r in records], float)
        logs = np.array([r["log_bytes"] for r in records], float)
        cadence = 1 / np.array([r["request_gap_s"] + r["tool_delay_s"]
                                + r["prompt_tokens"] / power["F_prefill_tps"]
                                + r["output_tokens"] / power["G_decode_tps"] for r in records])
        evidence = {"workload_source": raw["source"], "cadence": "profile gaps, then common load normalization"}
    records, frequencies = np.unique(np.column_stack((shapes, logs, cadence)), axis=0, return_counts=True)
    counts = rng.multinomial(gpus * density, frequencies / frequencies.sum())
    records, counts = records[counts > 0], counts[counts > 0]
    context, prompt, output, logs, cadence = records.T
    if np.any((context < curve[0, 0]) | (context > curve[-1, 0])):
        raise ValueError("workload contexts outside measured A100 prefill support")
    work = cadence * (prompt / power["F_prefill_tps"] + output / power["G_decode_tps"])
    demand = work * (gpus * SOURCE_LOAD / (counts @ work))
    watts = np.array(power["phase_power"]["measured_power_curve"])
    if SOURCE_LOAD > watts[-1, 0]:
        raise ValueError("source load outside measured power support")
    return Fleet(counts, context, demand, context / np.interp(context, *curve.T),
                 kv_bytes(model, context), logs, gpus, gpus * prefill["kv_capacity_tokens"],
                 float(np.interp(SOURCE_LOAD, *watts.T)), float(watts[0, 1]), evidence)


def resources(fleet, deadline, budgets, endpoint):
    """Columns are cohort-major: four actions per cohort; rows use physical units."""
    n = len(fleet.count)
    route = np.tile((0, 0, 1, 1), n)
    replay = np.tile((True, False, True, False), n)
    volume = np.where(replay, np.repeat(fleet.log, 4), np.repeat(fleet.kv, 4))
    work = np.repeat(fleet.replay, 4) * replay
    demand, context = np.repeat(fleet.demand, 4), np.repeat(fleet.context, 4)
    matrix = np.array([volume * (route == r) for r in (0, 1)] + [volume]
                      + [(work + deadline * demand) * (route == r) for r in (0, 1)]
                      + [context * (route == r) for r in (0, 1)])
    capacities = np.r_[budgets * deadline, np.repeat(fleet.gpus * (1 - BASELINE_LOAD) * deadline, 2),
                       np.repeat(fleet.kv_capacity - fleet.baseline_kv, 2)]
    isolated = volume / np.minimum(endpoint[route], np.minimum(budgets[route], budgets[2])) + work
    return matrix, capacities, isolated


def greedy_fill(count, gains, matrix, capacities, eligible, chosen=None):
    """Count-weighted version of the existing fixed scarcity-price greedy."""
    chosen = np.zeros(len(gains), dtype=np.int64) if chosen is None else chosen.copy()
    left = count - chosen.reshape(-1, 4).sum(1)
    normalized = matrix / capacities[:, None]
    costs = np.where(eligible, normalized.sum(0), np.inf).reshape(-1, 4)
    cheapest = np.argmin(costs, axis=1) + 4 * np.arange(len(count))
    valid = np.isfinite(costs.min(1))
    prices = np.maximum(normalized[:, cheapest[valid]] @ count[valid], 1)
    score = np.where(eligible, gains / np.maximum(prices @ normalized, 1e-30), -np.inf)
    usage = matrix @ chosen
    for j in np.argsort(-score, kind="stable"):
        i = j // 4
        if not eligible[j] or not left[i]:
            continue
        positive = matrix[:, j] > 0
        take = min(left[i], max(0, int(np.floor(np.min((capacities - usage)[positive] / matrix[positive, j]) + 1e-9))))
        chosen[j] += take
        left[i] -= take
        usage += take * matrix[:, j]
    return chosen


def select(fleet, deadline, budgets, endpoint, policy):
    matrix, capacities, isolated = resources(fleet, deadline, budgets, endpoint)
    gains = np.repeat(fleet.gain, 4)
    eligible = isolated <= deadline
    if np.any(capacities <= 0):
        raise ValueError("source/destination baseline leaves no migration resources")
    if policy in ("kv_only", "replay_only"):
        eligible &= np.arange(len(gains)) % 2 == (policy == "kv_only")
    elif policy == "isolated_fastest":
        # Prefer less replay work, then action ID, when isolated times tie.
        fastest = np.array([min(range(4), key=lambda a: (isolated[4*i+a],
                            fleet.replay[i] if a % 2 == 0 else 0, a)) for i in range(len(fleet.count))])
        eligible &= np.tile(np.arange(4) % 2, len(fleet.count)) == np.repeat(fastest % 2, 4)
    chosen, bound = None, None
    if policy == "queue_haul":
        population = np.repeat(fleet.count, 4)
        incidence = csr_matrix((np.ones(len(gains)), (np.repeat(np.arange(len(fleet.count)), 4),
                                                    np.arange(len(gains)))), shape=(len(fleet.count), len(gains)))
        # Solve cohort fractions so tiny per-session byte coefficients survive HiGHS scaling.
        constraints = vstack((incidence, csr_matrix(matrix * population / capacities[:, None])), format="csr")
        objective = gains * population
        result = linprog(-objective / objective.max(), A_ub=constraints,
                         b_ub=np.ones(len(fleet.count) + len(capacities)),
                         bounds=np.column_stack((np.zeros(len(gains)), eligible)),
                         method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                                  "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            raise RuntimeError(result.message)
        bound = float(-result.fun * objective.max())
        # Preserve maximum shed, then minimize peak resource pressure (including log transport).
        pressure = matrix * population / capacities[:, None]
        balanced = linprog(np.r_[np.zeros(len(gains)), 1.],
                          A_ub=vstack((hstack((constraints, csr_matrix((constraints.shape[0], 1)))),
                                       csr_matrix(np.column_stack((pressure, -np.ones(len(capacities))))))),
                          b_ub=np.r_[np.ones(constraints.shape[0]), np.zeros(len(capacities))],
                          A_eq=csr_matrix(np.r_[objective / objective.max(), 0.][None, :]),
                          b_eq=[-result.fun], bounds=[*zip(np.zeros(len(gains)), eligible), (0, 1)],
                          method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                                   "dual_feasibility_tolerance": 1e-9})
        if not balanced.success:
            raise RuntimeError(balanced.message)
        counts = balanced.x[:-1] * population
        chosen = np.floor(np.maximum(counts, 0)).astype(np.int64)
    elif policy not in POLICIES:
        raise ValueError(policy)
    chosen = greedy_fill(fleet.count, gains, matrix, capacities, eligible, chosen)
    residual = float(np.max((matrix @ chosen - capacities) / capacities))
    if residual > 1e-8 or np.any(chosen.reshape(-1, 4).sum(1) > fleet.count):
        raise RuntimeError("selection violates pooled capacity or session conservation")
    return chosen, {"planned_shed_w": float(gains @ chosen), "lp_bound_w": bound,
                    "solver_status": "optimal" if policy == "queue_haul" else "heuristic",
                    "rounding_gap_w": None if bound is None else max(0., bound - gains @ chosen),
                    "max_relative_capacity_residual": residual}


def fair_rates(count, caps, capacity):
    """Max-min per-session rates, compressed over identical session cohorts."""
    if not len(count) or capacity <= 0:
        return np.zeros(len(count))
    order = np.argsort(caps)
    c, weights = caps[order], count[order]
    spent = np.r_[0., np.cumsum(c * weights)[:-1]]
    remaining = np.cumsum(weights[::-1])[::-1]
    levels = (capacity - spent) / remaining
    index = np.flatnonzero(levels <= c)
    level = levels[index[0]] if len(index) else c[-1]
    return np.minimum(caps, max(0., level))


def execute(fleet, chosen, deadline, budgets, endpoint, service=1.):
    """Reserve serving, then run exact fluid stage events; commit integer cohorts."""
    chosen = np.asarray(chosen)
    if (chosen.shape != (4 * len(fleet.count),) or np.any(chosen < 0)
            or np.any(chosen != np.floor(chosen)) or np.any(chosen.reshape(-1, 4).sum(1) > fleet.count)
            or not np.all(np.isfinite(np.r_[deadline, service, budgets, endpoint]))
            or deadline <= 0 or service <= BASELINE_LOAD or np.any(budgets <= 0) or np.any(endpoint <= 0)):
        raise ValueError("invalid whole-session execution inputs")
    chosen = chosen.astype(np.int64)
    admitted = chosen.copy()
    demand = np.repeat(fleet.demand, 4) / service
    context = np.repeat(fleet.context, 4)
    route = np.tile((0, 0, 1, 1), len(fleet.count))
    compute = np.full(2, fleet.gpus * (1 - BASELINE_LOAD / service))
    memory = np.full(2, fleet.kv_capacity - fleet.baseline_kv)
    if np.any(compute <= 0) or np.any(memory <= 0):
        raise ValueError("sampled destination baseline itself is infeasible")
    for j in np.flatnonzero(chosen):
        r = route[j]
        admitted[j] = max(0, min(chosen[j], int(np.floor(min(compute[r] / demand[j], memory[r] / context[j]) + 1e-9))))
        compute[r] -= admitted[j] * demand[j]
        memory[r] -= admitted[j] * context[j]
    ids = np.flatnonzero(admitted)
    count, routes = admitted[ids], route[ids]
    replay = ids % 2 == 0
    net = np.where(replay, np.repeat(fleet.log, 4)[ids], np.repeat(fleet.kv, 4)[ids]).copy()
    work = (np.repeat(fleet.replay, 4)[ids] / service) * replay
    committed = (net <= 1e-6) & (work <= 1e-12)
    transferred, computed, contention = np.zeros(3), np.zeros(2), np.zeros(2)
    peak_ready = np.zeros(2, dtype=np.int64)
    now = 0.
    while now < deadline and np.any(~committed):
        sending = net > 1e-6
        ready = ~sending & (work > 1e-12)
        net_rate, compute_rate = np.zeros(len(ids)), np.zeros(len(ids))
        active = np.array([count[sending & (routes == r)].sum() for r in (0, 1)])
        caps = np.minimum(endpoint[:2], np.divide(budgets[:2], active, out=np.zeros(2), where=active > 0))
        net_rate[sending] = fair_rates(count[sending], caps[routes[sending]], budgets[2])
        for r in (0, 1):
            mask = ready & (routes == r)
            compute_rate[mask] = min(1., max(0., compute[r]) / max(1, count[mask].sum()))
            peak_ready[r] = max(peak_ready[r], count[mask].sum())
        times = np.r_[np.divide(net, net_rate, out=np.full(len(ids), np.inf), where=net_rate > 0),
                      np.divide(work, compute_rate, out=np.full(len(ids), np.inf), where=compute_rate > 0)]
        step = min(deadline - now, times.min(initial=np.inf))
        if step <= 0:
            raise RuntimeError("fluid executor made no progress")
        network_work, gpu_work = np.minimum(net, net_rate * step), np.minimum(work, compute_rate * step)
        for r in (0, 1):
            transferred[r] += count[routes == r] @ network_work[routes == r]
            computed[r] += count[routes == r] @ gpu_work[routes == r]
            mask = ready & (routes == r)
            contention[r] += count[mask] @ (1 - compute_rate[mask]) * step
        transferred[2] = transferred[:2].sum()
        net -= network_work
        work -= gpu_work
        now += step
        committed |= (net <= 1e-6) & (work <= 1e-12)
    completed = np.zeros(len(chosen), dtype=np.int64)
    completed[ids[committed]] = count[committed]
    ready_at_deadline = [int(count[replay & (net <= 1e-6) & ~committed & (routes == r)].sum()) for r in (0, 1)]
    peak_ready = np.maximum(peak_ready, ready_at_deadline)
    if np.any(transferred > budgets * deadline * (1 + 1e-8)) or np.any(computed > np.maximum(compute, 0) * deadline * (1 + 1e-8)):
        raise RuntimeError("execution exceeded flow-volume capacity")
    return completed, {"selected_sessions": int(chosen.sum()), "completed_sessions": int(completed.sum()),
                       "rejected_sessions": int((chosen - admitted).sum()),
                       "incomplete_sessions": int((admitted - completed).sum()),
                       "unselected_sessions": int(fleet.count.sum() - chosen.sum()),
                       "full_plan_success": bool(np.array_equal(completed, chosen)),
                       "network_utilization": (transferred / (budgets * deadline)).tolist(),
                       "compute_utilization": (computed / (fleet.gpus * deadline)).tolist(),
                       "prefill_contention_session_s": contention.tolist(),
                       "prefill_peak_ready_sessions": peak_ready.tolist(),
                       "prefill_ready_at_deadline": ready_at_deadline,
                       "prefill_network_blocked_at_deadline": [int(count[replay & (net > 1e-6) & (routes == r)].sum()) for r in (0, 1)],
                       "prefill_remaining_gpu_s": [float(count[routes == r] @ work[routes == r]) for r in (0, 1)],
                       "reserved_serving_utilization": (1 - compute / fleet.gpus).tolist(),
                       "kv_utilization": (1 - memory / fleet.kv_capacity).tolist(),
                       "last_event_s": now}


def configuration(smoke=False):
    return {"schema": SCHEMA, "models": list(MODELS), "gpus": GPUS,
            "hardware": "NVIDIA A100 80GB PCIe", "gpu_power_limit_w": 300,
            "installed_gpu_w": GPUS * 300, "deadlines": list(DEADLINES),
            "wan_gbps": [40, 400] if smoke else ["reference", 10, 40, 100, 400],
            "snapshots": 1 if smoke else 20, "draws": 20 if smoke else 200,
            "service_factors": [.8, 1., 1.2],
            "cases": [["coding", 8]] if smoke else [["coding", d] for d in (8, 4, 16, 32)]}


def cells(config):
    return list(product(config["models"], config["cases"], range(config["snapshots"]),
                        config["wan_gbps"], config["deadlines"]))


def provenance():
    paths = [Path(__file__), NETWORK, MANIFEST, ROOT / "plot_style.py"]
    paths += [CALIBRATION / name for name in ("cells.jsonl", "metadata.json", "server.log", "fit.json")]
    paths += [ROOT / f"profiles/{w}.json" for w in ("interactive_coding", "agentic_tool_loop", "agentic_rps_shape")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare(out, smoke=False, wan_gbps=None):
    config = configuration(smoke)
    if wan_gbps is not None:
        if len(set(wan_gbps)) != len(wan_gbps) or not np.all(np.isfinite(wan_gbps)) or min(wan_gbps) <= 0:
            raise ValueError("WAN sweep requires distinct positive finite Gbit/s budgets")
        config["wan_gbps"] = ["reference", *sorted(wan_gbps)]
    metadata = {"config": config, "sources": provenance(), "calibration": calibrations(MODELS[0])[1]["evidence"],
                "source_region": "swedencentral", "destination_regions": ["eastus2", "germanywestcentral"],
                "seeds": {"workload": "1001 + snapshot", "calibration": "2001 + snapshot"},
                "network_references": [
                    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-tcpip-performance-tuning",
                    "https://learn.microsoft.com/en-us/azure/virtual-network/virtual-network-peering-overview"],
                "wan_literature": [
                    {"paper": "SWAN, SIGCOMM 2013, section 6.1", "scope": "production inter-DC capacities: tens of Gbit/s to Tbit/s",
                     "url": "https://www.microsoft.com/en-us/research/wp-content/uploads/2013/08/Achieving-High-Utilization-with-Software-Driven-WAN.pdf"},
                    {"paper": "B4, SIGCOMM 2013", "scope": "shared WAN links, application-priority allocation; not GPU-count capacity",
                     "url": "https://conferences.sigcomm.org/sigcomm/2013/papers/sigcomm/p3.pdf"},
                    {"paper": "B4 and After, SIGCOMM 2018, section 3", "scope": "up to 6.4 Tbit/s Saturn WAN/site; 81.92 Tbit/s Stargate includes cluster and sidelinks",
                     "url": "https://cs538.github.io/readings/hong18.pdf"},
                    {"paper": "RADWAN, SIGCOMM 2018", "scope": "100 Gbit/s IP links and 100-200 Gbit/s rate adaptation; global sums are not route budgets",
                     "url": "https://www.microsoft.com/en-us/research/uploads/prod/2018/03/Rate_Adaptive_WAN.pdf"},
                    {"paper": "OneWAN, NSDI 2023", "scope": "regional aggregation and backbone links shared by traffic classes",
                     "url": "https://www.usenix.org/system/files/nsdi23-krishnaswamy.pdf"},
                    {"paper": "TEAL, SIGCOMM 2023, section 5.1", "scope": "uses measured SWAN demand; assigns some missing topology capacities for evaluation",
                     "url": "https://minlanyu.seas.harvard.edu/writeup/sigcomm23-teal.pdf"},
                    {"paper": "HEDGE, NSDI 2026", "scope": "600 Gbit/s hardware LAG; stochastic production link capacity; 3/5 Tbit/s targets are modeled",
                     "url": "https://www.usenix.org/system/files/nsdi26-devraj.pdf"}],
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "network_observations_bytes_per_s": network_samples().tolist(),
                "assumptions": ["20 MW target installed; 66666 300W A100 GPUs = 19.9998 MW",
                    "80% source and 50% destination normalized F/G demand are assumed; no calibrated utilization or latency SLO claim",
                    "empirical 604/64 active power anchors are provisional for sampled serving mixes; failed rational fit unused",
                    "bootstrap only within warm idle and same-shape/concurrency groups; no workload-transfer confidence claim",
                    "busy equivalent capacity drains to idle; no discrete GPU packing or shutdown",
                    "frozen state; weights resident; no ingest/setup/catch-up cost",
                    "configured Gbit/s shared migration allocations are assumed, not Azure route measurements",
                    "each route capped at the shared WAN budget; aggregate caps do not inherit TCP throughput asymmetry",
                    "GPU count scales endpoint ceilings only, never the available WAN allocation",
                    "network repeats vary endpoint limits only; WAN allocation held fixed, no invented backbone error distribution",
                    "per-session network ceiling assumes one measured eight-stream endpoint bundle",
                    "replay per-session compute ceiling is one GPU; capacity uses contextual prefill times",
                    "all compute after serving reservation is available to replay with ideal processor sharing; no measured loaded-queue claim",
                    "LP tie-break minimizes peak normalized resource use while preserving maximum shed",
                    "stable session IDs are cohort-major then action-major within selected counts",
                    "greedy uses the native scarcity-price primary pass; no target-recovery scans",
                    "KV geometry follows matched_action analytical BF16 formulas, not measured wire bytes",
                    "pooled KV tokens use measured startup capacity; no placement/fragmentation model",
                    "coding logs assume 2 bytes/token; equal cadence normalized to source load",
                    "coding snapshots resample 24 trajectories and one supported joint state each",
                    "destination-only service sensitivity is assumed +/-20%; source demand/power fixed",
                    "network bootstrap has only three paired repetitions; no regional capacity confidence claim",
                    "power draw is fleet-wide, weighted by compressed bootstrap multiplicities",
                    "LP is a volume relaxation bound; executor reports completed whole sessions"]}
    metadata["identity"] = digest({"config": config, "sources": metadata["sources"]})
    path = out / "plan.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != metadata["identity"]:
            raise ValueError("existing plan differs; use a new output directory")
    else:
        write_json(path, metadata)
    return metadata


def run_cell(config, cell):
    model, (workload, density), snapshot, wan_gbps, deadline = cell
    fleet = sample_fleet(model, workload, density, snapshot, config["gpus"])
    source_memory = float(fleet.count @ fleet.context)
    base = {"model": model, "workload": workload, "density": density, "snapshot": snapshot,
            "wan_gbps": wan_gbps, "deadline_s": deadline,
            "population_sessions": int(fleet.count.sum()), "source_kv_utilization": source_memory / fleet.kv_capacity}
    if source_memory > fleet.kv_capacity or fleet.baseline_kv >= fleet.kv_capacity:
        return {"case": base, "status": "memory_infeasible", "metadata": fleet.metadata}
    samples = network_samples()
    central = np.median(samples, axis=0)
    central[2] = central[:2].sum()
    budgets = bandwidth(central, fleet.gpus, wan_gbps)
    _, power = calibrations(model)
    rng = np.random.default_rng(2001 + snapshot)
    network_draws = rng.integers(len(samples), size=config["draws"])
    power_draws = rng.choice(len(power["bootstrap_curve_counts"]), size=config["draws"],
                            p=np.array(power["bootstrap_curve_counts"]) / 200)
    executions, plans = [], {}
    for policy in POLICIES:
        chosen, planning = select(fleet, deadline, budgets, central, policy)
        plans[policy] = {**planning, "counts": chosen.reshape(-1, 4).tolist()}
        central_done, central_result = execute(fleet, chosen, deadline, budgets, central)
        plans[policy]["central_execution"] = {**central_result, "shed_w": float(np.repeat(fleet.gain, 4) @ central_done)}
        for service in config["service_factors"]:
            for k in np.unique(network_draws):
                actual_budgets = bandwidth(samples[k], fleet.gpus, wan_gbps)
                completed, result = execute(fleet, chosen, deadline, actual_budgets, samples[k], service)
                action_gpu = (completed.reshape(-1, 4) * fleet.demand[:, None]).sum(0) / SOURCE_LOAD
                executions.append({**planning, **result, "policy": policy, "service_factor": service,
                                   "network_draw": int(k), "action_gpu": action_gpu.tolist(),
                                   "action_counts": completed.reshape(-1, 4).sum(0).tolist(),
                                   "network_budget_gbps": (actual_budgets * 8e-9).tolist()})
    curves = [np.array(power["phase_power"]["measured_power_bootstrap"][p]) for p in power_draws]
    return {"case": base, "status": "complete", "executions": executions, "plans": plans, "metadata": fleet.metadata,
            "gpus": fleet.gpus, "draws": {"network": network_draws.tolist(), "power": power_draws.tolist(),
                "active_w": [float(np.interp(SOURCE_LOAD, *curve.T)) for curve in curves],
                "idle_w": [float(curve[0, 1]) for curve in curves]},
            "cohorts": {"counts": fleet.count.tolist(), "context": fleet.context.tolist(),
                        "demand": fleet.demand.tolist(), "replay_gpu_s": fleet.replay.tolist()}}


def draw_rows(result):
    """Expand compact, paired execution/power draws only when needed."""
    for execution in result["executions"]:
        for draw, k in enumerate(result["draws"]["network"]):
            if execution["network_draw"] != k:
                continue
            active, idle = result["draws"]["active_w"][draw], result["draws"]["idle_w"][draw]
            action_w = np.array(execution["action_gpu"]) * (active - idle)
            yield {**result["case"], **execution, "draw": draw, "power_draw": result["draws"]["power"][draw],
                   "initial_source_w": result["gpus"] * active, "idle_source_w": result["gpus"] * idle,
                   "shed_w": float(action_w.sum()), "shed_fraction": sum(execution["action_gpu"]) / result["gpus"],
                   "action_shed_w": action_w.tolist()}


def run(out, shard=0, shards=1):
    plan = json.loads((out / "plan.json").read_text())
    if plan["sources"] != provenance() or not 0 <= shard < shards:
        raise ValueError("input/code changed or invalid shard")
    for index, cell in enumerate(cells(plan["config"])):
        if index % shards != shard:
            continue
        path = out / "cells" / f"{index:06d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as handle:
                previous = json.load(handle)
            if previous["identity"] != plan["identity"] or previous["cell"] != list(cell):
                raise ValueError(f"stale cell: {path}")
            continue
        result = run_cell(plan["config"], cell)
        write_json(path, {"identity": plan["identity"], "cell": cell, **result})
        print(f"{index + 1}/{len(cells(plan['config']))} {model_label(cell)} {result['status']}", flush=True)


def model_label(cell):
    return f"{cell[0]} {cell[1]} snapshot={cell[2]} network={cell[3]} deadline={cell[4]}"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def reduce(out):
    plan = json.loads((out / "plan.json").read_text())
    if plan["sources"] != provenance():
        raise ValueError("input/code changed since preparation")
    expected = cells(plan["config"])
    paths = sorted((out / "cells").glob("*.json.gz"))
    if {p.name for p in paths} != {f"{i:06d}.json.gz" for i in range(len(expected))}:
        raise ValueError("missing or unexpected campaign cells")
    groups, pairs, invalid = {}, {}, []
    grouping = ("model", "workload", "density", "wan_gbps", "deadline_s", "policy", "service_factor")
    for i, path in enumerate(paths):
        with gzip.open(path, "rt") as handle:
            result = json.load(handle)
        if result["identity"] != plan["identity"] or result["cell"] != list(expected[i]):
            raise ValueError(f"stale or duplicate cell: {path}")
        if result["status"] == "memory_infeasible":
            invalid.append(result["case"])
            continue
        if result["status"] != "complete" or any(len(v) != plan["config"]["draws"] for v in result["draws"].values()):
            raise ValueError(f"invalid status or draw lengths: {path}")
        rows = list(draw_rows(result))
        keys = [(r["policy"], r["service_factor"], r["draw"]) for r in rows]
        required = set(product(POLICIES, plan["config"]["service_factors"], range(plan["config"]["draws"])))
        if len(keys) != len(required) or set(keys) != required:
            raise ValueError(f"missing or duplicate policy/draw rows: {path}")
        local = {}
        for row in rows:
            local.setdefault(tuple(row[k] for k in grouping), []).append(row)
        for key, group in local.items():
            group.sort(key=lambda r: r["draw"])
            groups.setdefault(key, []).append({
                "watts": np.array([r["shed_w"] for r in group]),
                "fractions": np.array([r["shed_fraction"] for r in group]),
                "completed": np.array([r["completed_sessions"] for r in group]),
                "success": np.mean([r["full_plan_success"] for r in group]),
                "actions": np.mean([r["action_counts"] for r in group], axis=0),
                "action_watts": np.mean([r["action_shed_w"] for r in group], axis=0),
                "network_usage": np.mean([r["network_utilization"] for r in group], axis=0),
                "compute_usage": np.mean([r["compute_utilization"] for r in group], axis=0),
                "serving_usage": np.mean([r["reserved_serving_utilization"] for r in group], axis=0),
                **{metric: np.mean([r[metric] for r in group], axis=0) for metric in PREFILL_METRICS},
                "planned_shed_w": np.mean([r["planned_shed_w"] for r in group]),
                "memory_usage": np.mean([r["kv_utilization"] for r in group], axis=0),
                "not_completed": np.mean([r["population_sessions"] - r["completed_sessions"] for r in group]),
                "budget": np.median([r["network_budget_gbps"] for r in group], axis=0)})
            if key[-2] != "queue_haul":
                qh_key = (*key[:-2], "queue_haul", key[-1])
                qh = sorted(local[qh_key], key=lambda r: r["draw"])
                pairs.setdefault(key, []).append(np.array([a["shed_w"] - b["shed_w"] for a, b in zip(qh, group)]))
    if not groups:
        raise ValueError("no memory-feasible campaign cells")
    summary = []
    for key, group in groups.items():
        if len(group) != plan["config"]["snapshots"]:
            # Memory feasibility can depend on the sampled context distribution.
            status = "conditional_on_memory_feasible_snapshots"
        else:
            status = "complete"
        watts = np.concatenate([r["watts"] for r in group])
        snapshot_medians = [np.median(r["watts"]) for r in group]
        calibration_width = [np.diff(np.quantile(r["watts"], [.05, .95]))[0] for r in group]
        completed = np.concatenate([r["completed"] for r in group])
        summary.append({**dict(zip(grouping, key)), "draws": len(watts), "snapshots": len(group), "status": status,
                        "p05_shed_mw": float(np.quantile(watts, .05) / 1e6),
                        "median_shed_mw": float(np.median(watts) / 1e6), "p95_shed_mw": float(np.quantile(watts, .95) / 1e6),
                        "median_shed_fraction": float(np.median(np.concatenate([r["fractions"] for r in group]))),
                        "workload_p05_mw": float(np.quantile(snapshot_medians, .05) / 1e6),
                        "workload_p95_mw": float(np.quantile(snapshot_medians, .95) / 1e6),
                        "median_calibration_width_mw": float(np.median(calibration_width) / 1e6),
                        "full_plan_success": float(np.mean([r["success"] for r in group])),
                        "p05_completed_sessions": float(np.quantile(completed, .05)),
                        "median_completed_sessions": float(np.median(completed)),
                        "p95_completed_sessions": float(np.quantile(completed, .95)),
                        "mean_network_utilization": np.mean([r["network_usage"] for r in group], axis=0).tolist(),
                        "mean_migration_compute_utilization": np.mean([r["compute_usage"] for r in group], axis=0).tolist(),
                        "mean_serving_utilization": np.mean([r["serving_usage"] for r in group], axis=0).tolist(),
                        **{f"mean_{metric}": np.mean([r[metric] for r in group], axis=0).tolist() for metric in PREFILL_METRICS},
                        "mean_planned_shed_mw": float(np.mean([r["planned_shed_w"] for r in group]) / 1e6),
                        "mean_kv_utilization": np.mean([r["memory_usage"] for r in group], axis=0).tolist(),
                        "action_counts_mean": np.mean([r["actions"] for r in group], axis=0).tolist(),
                        "action_shed_mw_mean": (np.mean([r["action_watts"] for r in group], axis=0) / 1e6).tolist(),
                        "mean_not_completed": float(np.mean([r["not_completed"] for r in group])),
                        "network_budget_gbps_median": np.median([r["budget"] for r in group], axis=0).tolist()})
    paired = []
    for key, arrays in pairs.items():
        values = np.concatenate(arrays)
        paired.append({**dict(zip(grouping, key)), "draws": len(values),
                       **dict(zip(("p05_qh_minus_baseline_w", "median_qh_minus_baseline_w", "p95_qh_minus_baseline_w"),
                                  np.quantile(values, [.05, .5, .95]))), "qh_win_fraction": float(np.mean(values > 0))})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_differences.csv", paired)
    curves = {}
    for row in summary:
        key = tuple(row[k] for k in grouping if k != "deadline_s")
        curves.setdefault(key, []).append(row)
    regressions = []
    for curve in curves.values():
        curve.sort(key=lambda r: r["deadline_s"])
        for before, after in zip(curve, curve[1:]):
            if after["median_shed_mw"] + 1e-9 < before["median_shed_mw"]:
                regressions.append({**{k: after[k] for k in grouping}, "previous_deadline_s": before["deadline_s"],
                                    "median_shed_drop_mw": before["median_shed_mw"] - after["median_shed_mw"]})
    write_json(out / "summary.json", {"identity": plan["identity"], "complete_cells": len(paths),
                                      "memory_infeasible": invalid, "deadline_regressions": regressions, "summary": summary})
    plot(summary, out)
    return summary


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    for workload, density in sorted({(r["workload"], r["density"]) for r in summary}):
        selected = [r for r in summary if (r["workload"], r["density"], r["service_factor"]) == (workload, density, 1.)]
        models = [m for m in MODELS if any(r["model"] == m for r in selected)]
        wan_points = list(dict.fromkeys(r["wan_gbps"] for r in selected))
        fig, axes = plt.subplots(len(models), len(wan_points), squeeze=False,
                                 figsize=(3.5 * len(wan_points), max(4.2, 2.8 * len(models))), sharex=True, sharey="row")
        for ax, (model, wan_gbps) in zip(axes.flat, product(models, wan_points)):
            for policy in POLICIES:
                series = sorted((r for r in selected if (r["model"], r["wan_gbps"], r["policy"]) == (model, wan_gbps, policy)), key=lambda r: r["deadline_s"])
                if not series:
                    continue
                x = [r["deadline_s"] for r in series]
                ax.plot(x, [r["median_shed_mw"] for r in series], color=plot_style.POLICY_COLORS[policy],
                        linestyle=plot_style.POLICY_LINESTYLES[policy], label=plot_style.POLICY_NAMES[policy])
                ax.fill_between(x, [r["p05_shed_mw"] for r in series], [r["p95_shed_mw"] for r in series], color=plot_style.POLICY_COLORS[policy], alpha=.1)
            budget = next(r["network_budget_gbps_median"][2] for r in selected if r["model"] == model and r["wan_gbps"] == wan_gbps)
            ax.set(title=f"{model}\nshared {budget:.3g} Gbit/s", xscale="log", xlabel="Deadline (s)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
        fig.suptitle(f"GPT-OSS/A100; provisional power and pooled compute\n{workload}, {density} sessions/GPU; bands: paired draw p05–p95", fontsize=11)
        fig.supylabel("Attained shed (MW)")
        fig.tight_layout(rect=(.03, .1, 1, .94))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"frontier-{workload}-{density}.{suffix}", bbox_inches="tight")
        plt.close(fig)
        for wan_gbps, metric in product(wan_points, ("sessions", "watts")):
            fig, axes = plt.subplots(len(models), len(POLICIES), squeeze=False, figsize=(17.5, max(4.2, 2.8 * len(models))), sharey="row")
            actions = (*ACTIONS, "not_moved") if metric == "sessions" else ACTIONS
            for ax, (model, policy) in zip(axes.flat, product(models, POLICIES)):
                series = sorted((r for r in selected if (r["model"], r["wan_gbps"], r["policy"]) == (model, wan_gbps, policy)), key=lambda r: r["deadline_s"])
                if metric == "sessions":
                    values = np.array([r["action_counts_mean"] + [r["mean_not_completed"]] for r in series]).T
                    values /= values.sum(0)
                    ax.set_ylim(0, 1)
                else:
                    values = np.array([r["action_shed_mw_mean"] for r in series]).T
                ax.stackplot([r["deadline_s"] for r in series], values, labels=[plot_style.ACTION_NAMES[a] for a in actions],
                             colors=[plot_style.ACTION_COLORS[a] for a in actions])
                ax.set(title=f"{model}\n{plot_style.POLICY_NAMES[policy]}", xscale="log", xlabel="Deadline (s)",
                       ylabel="Session share" if metric == "sessions" else "Mean shed (MW)")
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=9)
            caption = "measured endpoint reference" if wan_gbps == "reference" else f"assumed shared WAN {wan_gbps:g} Gbit/s"
            fig.suptitle(f"A100; provisional power/compute; {workload}, {density} sessions/GPU; {caption}", fontsize=11)
            fig.tight_layout(rect=(0, .1, 1, .96))
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"actions-{workload}-{density}-{wan_gbps}-{metric}.{suffix}", bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "reduce"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wan-gbps", type=float, nargs="+", help="prepare: shared migration budgets; e.g. 10 40 100 400 1000")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command != "prepare" and (args.smoke or args.wan_gbps is not None):
        parser.error("--smoke and --wan-gbps apply only to prepare")
    if args.command == "prepare":
        plan = prepare(args.out, args.smoke, args.wan_gbps)
        print(f"Prepared {len(cells(plan['config']))} cells: {args.out}")
    elif args.command == "run":
        run(args.out, args.shard, args.shards)
    else:
        reduce(args.out)


if __name__ == "__main__":
    main()
