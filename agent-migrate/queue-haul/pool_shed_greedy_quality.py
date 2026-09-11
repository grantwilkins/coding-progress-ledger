"""Bounded native packing comparisons against the frozen bandwidth study."""

import argparse
import gzip
import hashlib
import inspect
import json
import platform
import time
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
from _queue_haul_native import _queue_haul_native as native

import pool_shed_bandwidth_sweep as sweep
import pool_shed_campaign as q
import pool_shed_planner as planner
import pool_shed_priced_greedy as pricing
from pool_shed_execution import PooledExecution
from pool_shed_network import transport_workers

REFERENCE = q.ROOT / "outputs/a100-bandwidth-ttft-20260911"
OUT = q.ROOT / "outputs/a100-native-greedy-20260911"
WORKLOADS = ("coding", "coding_long")
FEEDBACK_CASES = ((30, 10.), (120, 1.), (120, 100.))
MATRIX_CASES = tuple(product((30, 120), (1., 10., 100.)))
CHANGED = {"pool_shed_planner.py", "pool_shed_campaign.py", "pool_shed_bandwidth_sweep.py", "plot_style.py",
           "pool_shed_execution.py", "pool_shed_resident_queue.py"}
SCOPE = "Same frozen fleet, measured timing, candidate library, network limits and feedback clocks. " \
        "Rust heap seeding, a greedy covering-dual bound, sparse coordinate refinement with diagonal row penalties and symmetric sweeps, and feasible support compression replace multiplicative-weight iterations; exact CPU memoization reuses unchanged simulation queries. " \
        "Pricing certificates bound the primary handoff objective of individual admission matrices within 0.001 of total source work (0.1 percentage point), not the receding-horizon execution optimum. " \
        "Debt breaks heap ties; the LP's secondary minimization is not reproduced. " \
        "No request-timing audit is rerun; handoff and local fluid recovery do not certify resident SLOs."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
        return json.load(handle)


def prepare(out):
    archived = read(REFERENCE / "config.json")
    if q.digest({k: v for k, v in archived.items() if k != "identity"}) != archived["identity"]:
        raise ValueError("archived configuration identity changed")
    frozen = {**archived["measurement_sha256"], **archived["source_sha256"]}
    for name, expected in frozen.items():
        if name not in CHANGED and sha(q.ROOT / name) != expected:
            raise ValueError(f"frozen physical input changed: {name}")
    measured = q.calibration(0)
    fleets = {w: sweep.fleet_for(w, measured) for w in WORKLOADS}
    for name, fleet in fleets.items():
        digest = q.digest(dict(metadata=fleet.metadata, count=fleet.count.tolist(), context=fleet.context.tolist()))
        if digest != archived["fleet_contract_sha256"][name]:
            raise ValueError(f"frozen fleet contract changed: {name}")
    config = dict(scope=SCOPE, archive_identity=archived["identity"], archive_config_sha256=sha(REFERENCE / "config.json"),
        frozen_input_sha256=frozen, source_sha256={str(p.relative_to(q.ROOT)): sha(p) for p in [
            *q.ROOT.glob("pool_shed*.py"), q.ROOT / "plot_style.py", *q.ROOT.glob("native/src/*.rs"),
            q.ROOT / "native/Cargo.toml", q.ROOT / "native/Cargo.lock", q.ROOT / "native/rust-toolchain.toml"]},
        feedback_cases=FEEDBACK_CASES, matrix_cases=MATRIX_CASES, workloads=WORKLOADS,
        planning=dict(chunks=q.DISPATCH_CHUNKS, resolution=q.PLANNING_RESOLUTION, iterations=q.PLANNING_ITERATIONS),
        environment=dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                         numpy=np.__version__, highs=q.highspy.Highs().version(),
                         native_binary_sha256=sha(Path(native.__file__))),
        matrix_timing="LP/native each have three serial warm-process repeats including normalization, native seed/fill, support compression and certification; old greedy has one observed sample. LP primary and secondary stages are timed separately; imports and matrix construction excluded.",
        feedback_timing="LP/native are each rerun serially once with identical current execution code; both timings exclude common table construction and result validation. Old greedy timing is archived and includes table construction and validation.")
    config["identity"] = q.digest(config)
    path = out / "config.json"
    if path.exists() and read(path)["identity"] != config["identity"]:
        raise ValueError("comparison inputs changed; choose a fresh output directory")
    q.write_json(path, config)
    return config, measured, fleets


def paired_inputs(workload, fleet, replay, kv, endpoint, deadline, bandwidth, measured, config):
    budgets = np.full(3, bandwidth * 1e12 / 8)
    table = q.schedule_table(fleet, replay, kv, sweep.LOAD, deadline, endpoint, budgets, measured["timing"][0])
    candidates = q.digest(dict(replay=replay.tolist(), kv=kv.tolist()))
    references, hashes = {}, {}
    for policy in ("greedy", "queue_haul"):
        path = REFERENCE / f"{workload}-d{deadline}-b{bandwidth:g}-{policy}.json.gz"
        record = read(path)
        if (record["identity"] != config["archive_identity"] or record["candidate_sha256"] != candidates
                or q.digest(record["fleet_metadata"]) != q.digest(fleet.metadata)
                or not np.array_equal(record["requested_budgets_tbps"], budgets * 8e-12)
                or not np.array_equal(record["effective_budgets_tbps"], table.budgets * 8e-12)
                or not np.array_equal(record["transport_worker_equivalents"], transport_workers(fleet))):
            raise ValueError(f"paired controls differ: {path.name}")
        references[policy], hashes[policy] = record, sha(path)
    return table, references, dict(identity=config["identity"], workload=workload, deadline_s=deadline,
        bandwidth_tbps=bandwidth, candidate_sha256=candidates, reference_sha256=hashes,
        requested_budgets_tbps=(budgets * 8e-12).tolist(), effective_budgets_tbps=(table.budgets * 8e-12).tolist())


def metrics(result, wall_s, wall_scope):
    actions = np.asarray(result["action_counts"])
    widths, replicas = [], []
    for route in (0, 1):
        waves = [w for w in result["wave_schedules"] if w["route"] == route]
        mass = sum(w["mass"] for w in waves)
        replicas.append(mass)
        widths.append(sum(w["mass"] * sum(w["counts"]) for w in waves) / mass if mass else None)
    return dict(handoff_fraction=result["shed_fraction"], recovered_handoff_fraction=result["recovered_handoff_fraction"],
        action_order=q.ACTIONS, completed_action_counts=actions.tolist(), completed_action_work=result["action_fractions"],
        kv_share_of_completed_sessions=float(actions[1::2].sum() / actions.sum()) if actions.sum() else None,
        destination_admitted_mean_width=widths, reserved_destination_replicas=replicas,
        dispatch_waves=len(result["wave_schedules"]), wall_s=wall_s, wall_scope=wall_scope,
        planning_s=result["planning_s"], planning_steps=result["planning_steps"])


def feedback(table, measured, references):
    archived = {policy: metrics(record["result"], record["simulate_s"], "Archived table construction, feedback and validation")
                for policy, record in references.items()}
    paired = dict(archived)
    for policy in ("queue_haul", "greedy_priced"):
        started = time.perf_counter()
        result = q.execute_feedback(table, table, policy, measured["timing"][0], measured)
        wall_s = time.perf_counter() - started
        sweep.validate_result(table.fleet, result, table.deadline, table.budgets, policy)
        paired[policy] = metrics(result, wall_s, "Feedback execution including planning; current shared simulation code")
        if policy == "queue_haul" and not np.isclose(result["shed_fraction"], archived[policy]["handoff_fraction"], atol=1e-8, rtol=1e-8):
            raise RuntimeError("LP feedback objective changed from the frozen reference")
    certificates = [dict(time_s=d["time_s"], iteration=i, **certificate)
                    for d in result["planning_diagnostics"] for i, certificate in enumerate(d.get("pricing_certificates", []))]
    if not certificates or any(not c["converged"] for c in certificates):
        raise RuntimeError("priced execution lacks converged admission certificates")
    return dict(result=result, comparisons=paired, archived_comparisons=archived, pricing_certificates=certificates,
        signed_handoff_difference_pp={policy: 100 * (result["shed_fraction"] - values["handoff_fraction"])
                                      for policy, values in paired.items() if policy != "greedy_priced"},
        resident_latency_validated=False)


def initial_matrix(table, measured, references, path):
    captured, old_choose = {}, planner._choose

    def observe(matrix, capacity, gains, debt, fleet, greedy):
        if captured:
            raise RuntimeError("expected one initial admission matrix")
        caller = inspect.currentframe().f_back.f_locals
        started = time.perf_counter()
        old = old_choose(matrix, capacity, gains, debt, fleet, greedy)
        old_s = time.perf_counter() - started
        captured.update(matrix=matrix.copy(), capacity=capacity.copy(), gains=gains.copy(), debt=debt.copy(),
            gpus=fleet.gpus, old=old.copy(), old_s=old_s,
            original=caller["original"].copy(), starts=caller["starts"].copy(), edges=caller["edges"].copy(),
            static_rows=len(caller["capacity"]))
        return old

    with patch.object(planner, "_choose", observe):
        planner.plan_admission(PooledExecution(table, table.timing, measured), table, "greedy", iterations=1)
    matrix, capacity, gains, debt = [captured[k] for k in ("matrix", "capacity", "gains", "debt")]
    original_lp, original_priced = q.solve_lp, pricing.priced_greedy
    stages = {"primary": [], "secondary": []}

    def timed_lp(table, allowed, objective, primary=None):
        started = time.perf_counter()
        result = original_lp(table, allowed, objective, primary)
        stages["primary" if primary is None else "secondary"].append(time.perf_counter() - started)
        return result

    def with_prices(*args, **kwargs):
        result, info = original_priced(*args, **kwargs, return_prices=True)
        captured["priced_dual"] = info.pop("dual_prices")
        return result, info

    for method in ("lp", "priced"):
        times = []
        with patch.object(q, "solve_lp", timed_lp), patch.object(pricing, "priced_greedy", with_prices):
            for _ in range(3):
                started = time.perf_counter()
                if method == "lp":
                    captured[method] = old_choose(matrix, capacity, gains, debt, table.fleet, False)
                else:
                    captured[method], certificate = planner._choose_priced(matrix, capacity, gains, debt, table.fleet)
                times.append(time.perf_counter() - started)
        captured[method + "_samples_s"], captured[method + "_s"] = times, float(np.median(times))
    for stage, times in stages.items():
        captured["lp_" + stage + "_samples_s"] = times
    compared = {}
    for method in ("old", "lp", "priced"):
        chosen = captured[method]
        residual = float(np.max((matrix @ chosen - capacity) / np.maximum(capacity, 1.), initial=0.))
        if np.any(chosen < 0) or not np.isfinite(chosen).all() or residual > 1e-8:
            raise RuntimeError(f"infeasible {method} matrix allocation")
        immediate = captured["starts"] == 0
        compared[method] = dict(gain=float(gains @ chosen), immediate_gain=float(gains[immediate] @ chosen[immediate]),
            secondary_work=float(debt @ chosen), positive_columns=int(np.count_nonzero(chosen > 0)),
            wall_s=captured[method + "_s"], max_relative_residual=residual,
            samples_s=captured.get(method + "_samples_s", [captured[method + "_s"]]))
    compared["lp"].update({stage + "_wall_s": float(np.median(times)) for stage, times in stages.items()})
    prices = captured["priced_dual"]
    if (np.any(matrix.T @ prices < gains * (1 - 1e-10))
            or not np.isclose(capacity @ prices, certificate["upper_bound"], rtol=1e-12, atol=1e-12)):
        raise RuntimeError("saved native dual does not certify the original matrix")
    for method, policy in (("old", "greedy"), ("lp", "queue_haul")):
        expected = references[policy]["result"]["planning_diagnostics"][0]["predicted_shed_fraction"]
        if not np.isclose(compared[method]["gain"], expected, rtol=1e-8, atol=1e-8):
            raise ValueError(f"initial {policy} objective differs from the frozen comparison")
    if compared["priced"]["gain"] > compared["lp"]["gain"] + 1e-8:
        raise RuntimeError("priced solution exceeds the same-matrix LP reference")
    if compared["lp"]["gain"] > certificate["upper_bound"] + 1e-8:
        raise RuntimeError("priced upper bound is below the same-matrix LP reference")
    captured.update(replay=table.replay, kv=table.kv, route=table.route)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **captured)
    return dict(comparisons=compared, certificate=certificate, matrix_sha256=sha(path),
        signed_objective_difference_pp={method: 100 * (compared["priced"]["gain"] - compared[method]["gain"])
                                        for method in ("old", "lp")}, resident_latency_validated=False)


def run(out, mode="feedback", workloads=WORKLOADS):
    config, measured, fleets = prepare(out)
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    modes = ("feedback", "matrix") if mode == "both" else (mode,)
    for workload in workloads:
        fleet = fleets[workload]
        replay, kv = q.include_isolated(*q.library(fleet), q.isolated_methods(
            fleet, sweep.LOAD, endpoint, np.full(3, 1e12 / 8), measured["timing"][0]))
        for selected_mode in modes:
            for deadline, bandwidth in FEEDBACK_CASES if selected_mode == "feedback" else MATRIX_CASES:
                name = f"{workload}-d{deadline}-b{bandwidth:g}"
                path = out / selected_mode / f"{name}.json.gz"
                table, references, common = paired_inputs(workload, fleet, replay, kv, endpoint, deadline, bandwidth, measured, config)
                if path.exists():
                    record = read(path)
                    if any(record[k] != value for k, value in common.items()):
                        raise ValueError(f"comparison checkpoint changed: {path}")
                    if selected_mode == "matrix" and sha(path.with_suffix("").with_suffix(".npz")) != record["matrix_sha256"]:
                        raise ValueError("saved matrix changed")
                else:
                    result = feedback(table, measured, references) if selected_mode == "feedback" else initial_matrix(
                        table, measured, references, path.with_suffix("").with_suffix(".npz"))
                    record = {**common, **result, "mode": selected_mode}
                    q.write_json(path, record)
                print(json.dumps(dict(case=name, mode=selected_mode, comparisons=record["comparisons"]), sort_keys=True), flush=True)
    for name, expected in {**{k: v for k, v in config["frozen_input_sha256"].items() if k not in CHANGED},
                           **config["source_sha256"]}.items():
        if sha(q.ROOT / name) != expected:
            raise ValueError(f"source changed during comparison: {name}")
    if sha(Path(native.__file__)) != config["environment"]["native_binary_sha256"]:
        raise ValueError("native binary changed during comparison")
    paths = sorted(out.glob("*/*.json.gz"))
    records = [read(p) for p in paths]
    if any(r["identity"] != config["identity"] for r in records):
        raise ValueError("mixed comparison identities")
    for path, record in zip(paths, records):
        if record["mode"] == "matrix" and sha(path.with_suffix("").with_suffix(".npz")) != record["matrix_sha256"]:
            raise ValueError("saved matrix changed")
    q.write_json(out / "summary.json", dict(identity=config["identity"], scope=SCOPE, resident_latency_validated=False,
        rows=[{k: v for k, v in r.items() if k not in ("result", "pricing_certificates")} for r in records]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--mode", choices=("feedback", "matrix", "both"), default="feedback")
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    args = parser.parse_args()
    run(args.out, args.mode, args.workloads)
