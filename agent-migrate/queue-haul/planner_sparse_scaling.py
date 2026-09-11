"""Bounded first-admission planning timings with one source constraint per history."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from _queue_haul_native import _queue_haul_native as native
from scipy.sparse import csc_matrix, issparse

import pool_shed_campaign as q
import pool_shed_planner as planner
import pool_shed_priced_greedy as pricing
from pool_shed_execution import PooledExecution, kv_transfer_bytes

OUT = q.ROOT / "outputs/a100-sparse-planning-20260911"
SESSIONS, POLICIES = (64, 128, 256, 512, 1024), ("queue_haul", "greedy_priced")
LOAD, DEADLINE, REFERENCE_GPUS, REFERENCE_WAN_TBPS = .5, 30., 6666, 10.
SCOPE = [
    "One source constraint per individual history, count=1. A fixed 64-history block expands the measured coding trajectory sample at eight source GPUs; larger cases repeat that block with distinct history ownership and unchanged declared phases. Repeated trajectories and phases are not new measurements.",
    "Source and each destination have sessions/8 GPUs, eight GPUs per host. Source cadence and offered-load normalization, resident load, per-host reservations and per-GPU WAN allocation remain fixed. Allocations retain the existing fractional planning relaxation; this is not integer placement.",
    "Preparation includes individual-fleet construction, sparse candidate library, isolated comparisons and nominal table construction. First admission includes the execution object, temporal profiles, sparse resource assembly, actual production selection and post-selection checks. Calibration/data loading and Python imports are recorded separately or excluded.",
    "The total-time panel includes production LP primary plus secondary minimization. The primary-selection panel compares LP primary selection alone with the complete native selection helper. Native certifies primary loss at most .001 of total source work; it does not reproduce the LP secondary optimum.",
    "Serial paired repeats use the same freshly prepared table, alternate policy order and pin numerical threads to one. Repeat count is recorded in the configuration. Per-stage timing observers retain witnesses without substituting decisions. No DES progression, later feedback, request queue audit or resident SLO claim is included.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.dtype.str, value.shape)).encode() + value.tobytes()).hexdigest()


def sparse_arrays(matrix):
    matrix = csc_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return dict(data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=np.asarray(matrix.shape))


def inputs():
    measured = q.calibration(0)
    q.network_samples()
    base = q.replica_fleet(q.sample_fleet("coding", gpus=8, gpus_per_node=8))
    base = replace(base, metadata={**base.metadata, "planning_reference_s": 4.,
        "require_local_recovery": True, "fixed_host_shares": True, "host_migration_gbps": 80.,
        "record_wave_schedules": False, "replay_cached_tokens": [0.] * len(base.count),
        "kv_shared_tokens": [0.] * len(base.count), "kv_wire_scale": 800_000_000 / (32768 * 49152)})
    return measured, replace(base, kv=kv_transfer_bytes(base, base.context, measured))


def individual_fleet(base, sessions):
    if isinstance(sessions, bool) or not isinstance(sessions, (int, np.integer)) or sessions < 64 or sessions % 64:
        raise ValueError("sessions must be a positive multiple of the fixed 64-history block")
    if base.gpus != 8 or base.count.sum() != 64 or np.any(base.count != np.floor(base.count)):
        raise ValueError("individual scaling requires the frozen eight-GPU source block")
    ids = np.tile(np.repeat(np.arange(len(base.count)), base.count.astype(int)), sessions // 64)
    fleet = q.compact_fleet(base, ids)
    metadata = {**fleet.metadata, "destination_gpus": sessions // 8,
                "individual_history_ids": list(range(sessions)), "history_block_indices": ids.tolist()}
    metadata.pop("source_power", None)
    return replace(fleet, count=np.ones(sessions), gpus=sessions // 8,
                   kv_capacity=base.kv_capacity * sessions / 64,
                   templates=[list(range(i, i + 8)) for i in range(0, sessions, 8)], metadata=metadata)


def prepare(base, sessions, measured):
    started = time.perf_counter()
    fleet = individual_fleet(base, sessions)
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    budgets = np.full(3, REFERENCE_WAN_TBPS * 1e12 / 8 * fleet.gpus / REFERENCE_GPUS)
    sampled_s = time.perf_counter() - started
    started = time.perf_counter()
    replay, kv = q.include_isolated(*q.library(fleet, sparse=True),
        q.isolated_methods(fleet, LOAD, endpoint, budgets, measured["timing"][0], compact=True))
    library_s = time.perf_counter() - started
    started = time.perf_counter()
    table = q.schedule_table(fleet, replay, kv, LOAD, DEADLINE, endpoint, budgets, measured["timing"][0])
    table_s = time.perf_counter() - started
    if not all(issparse(value) for value in (replay, kv, table.matrix)):
        raise RuntimeError("session benchmark requires the production sparse preparation path")
    return table, dict(fleet_s=sampled_s, library_s=library_s, nominal_table_s=table_s,
                       preparation_s=sampled_s + library_s + table_s,
                       requested_wan_tbps=float(budgets[2] * 8e-12))


def admission(table, policy, measured):
    stages, captured = {"primary": [], "secondary": [], "selection": []}, {}
    old_lp, old_native = q.solve_lp, pricing.priced_greedy
    old_choose = planner._choose_priced if policy == "greedy_priced" else planner._choose

    def timed_lp(table_, allowed, objective, primary=None):
        started = time.perf_counter()
        chosen = old_lp(table_, allowed, objective, primary)
        stages["primary" if primary is None else "secondary"].append(time.perf_counter() - started)
        if primary is None:
            captured["lp_primary"] = chosen
        return chosen

    def with_prices(*args, **kwargs):
        chosen, certificate = old_native(*args, **kwargs, return_prices=True)
        captured["native_dual"] = certificate.pop("dual_prices")
        return chosen, certificate

    def timed_choose(matrix, capacity, gains, debt, fleet, *args):
        if "matrix" in captured:
            raise RuntimeError("fixed-affinity initial admission unexpectedly needed another load iteration")
        started = time.perf_counter()
        result = old_choose(matrix, capacity, gains, debt, fleet, *args)
        stages["selection"].append(time.perf_counter() - started)
        captured.update(matrix=matrix, capacity=capacity, gains=gains, debt=debt,
                        chosen=result[0] if policy == "greedy_priced" else result)
        return result

    started = time.perf_counter()
    with patch.object(q, "solve_lp", timed_lp), patch.object(pricing, "priced_greedy", with_prices), \
         patch.object(planner, "_choose_priced" if policy == "greedy_priced" else "_choose", timed_choose):
        admitted, next_decision, audit = planner.plan_admission(
            PooledExecution(table, table.timing, measured), table, policy, calibration=measured)
    elapsed = time.perf_counter() - started
    if "matrix" not in captured or not issparse(captured["matrix"]):
        raise RuntimeError("expected a nonempty sparse first-admission problem")
    matrix, capacity, gains, chosen = (captured[k] for k in ("matrix", "capacity", "gains", "chosen"))
    usage = np.asarray(matrix @ chosen).ravel()
    positive = capacity > 0
    residual = float(np.max(usage[positive] / capacity[positive] - 1, initial=0.))
    if (not np.isfinite(chosen).all() or not np.isfinite(usage).all() or np.any(chosen < 0) or np.any(usage[~positive] > 0)
            or residual > 1e-8):
        raise RuntimeError("first-admission allocation violates original constraints")
    objective = float(gains @ chosen)
    if policy == "greedy_priced":
        dual = captured["native_dual"]
        if not np.isfinite(dual).all() or np.any(dual < 0) or np.any(matrix.T @ dual < gains * (1 - 1e-10)):
            raise RuntimeError("native first-admission dual does not cover original columns")
        bound = float(capacity @ dual)
        if bound < objective * (1 - 1e-10) or bound - objective > .001:
            raise RuntimeError("native first-admission certificate misses its primary target")
    arrays = {**{f"matrix_{k}": v for k, v in sparse_arrays(matrix).items()},
              **{k: np.asarray(captured[k]) for k in ("capacity", "gains", "debt")}}
    chosen_hash = array_sha(chosen)
    stats = dict(policy=policy, first_admission_s=elapsed, selection_s=sum(stages["selection"]),
        lp_primary_s=sum(stages["primary"]), lp_secondary_s=sum(stages["secondary"]),
        temporal_and_validation_s=elapsed - sum(stages["selection"]),
        primary_selection_s=sum(stages["selection"] if policy == "greedy_priced" else stages["primary"]),
        full_horizon_objective=objective, admitted_fraction=float(table.gains @ admitted),
        max_relative_constraint_residual=max(0., residual), positive_columns=int(np.count_nonzero(chosen)),
        next_decision_s=next_decision, audit=audit, chosen_sha256=chosen_hash,
        matrix_shape=list(matrix.shape), matrix_nnz=matrix.nnz,
        input_array_sha256={k: array_sha(v) for k, v in arrays.items()})
    witnesses = {"chosen": chosen, "admitted": admitted}
    if policy == "greedy_priced":
        stats.update(upper_bound=bound, absolute_gap=max(0., bound - objective),
                     dual_sha256=array_sha(captured["native_dual"]))
        witnesses["dual"] = captured["native_dual"]
    else:
        stats["primary_objective"] = float(gains @ captured["lp_primary"])
        witnesses["primary"] = captured["lp_primary"]
    return stats, arrays, witnesses


def cell(sessions, repeat, out):
    started = time.perf_counter()
    measured, base = inputs()
    loading_s = time.perf_counter() - started
    table, preparation = prepare(base, sessions, measured)
    rows, witnesses, shared = [], {}, None
    for policy in POLICIES if repeat % 2 else POLICIES[::-1]:
        stats, arrays, chosen = admission(table, policy, measured)
        if shared is not None and rows[0]["input_array_sha256"] != stats["input_array_sha256"]:
            raise RuntimeError("LP and native first-admission inputs differ")
        shared = arrays
        witnesses.update({f"{policy}_{key}": value for key, value in chosen.items()})
        rows.append({**stats, **preparation, "total_planning_s": preparation["preparation_s"] + stats["first_admission_s"]})
    by_policy = {r["policy"]: r for r in rows}
    lp, native = by_policy["queue_haul"], by_policy["greedy_priced"]
    if lp["primary_objective"] > native["upper_bound"] + 1e-8:
        raise RuntimeError("native dual falls below the LP primary objective")
    witness = out / "cells" / f"{sessions}-{repeat}.npz"
    witness.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(witness, **shared, **witnesses)
    return dict(sessions=sessions, repeat=repeat, input_loading_s=loading_s,
        source_gpus=table.fleet.gpus, destination_gpus_each=table.fleet.gpus,
        candidate_rows=table.replay.shape[0], candidate_nnz=table.replay.nnz + table.kv.nnz,
        fleet_sha256=q.digest(dict(metadata=table.fleet.metadata, count=table.fleet.count.tolist(), context=table.fleet.context.tolist())),
        primary_shortfall_pp=100 * (lp["primary_objective"] - native["full_horizon_objective"]),
        witness_file=str(witness.relative_to(q.ROOT)), witness_sha256=sha(witness), policies=rows)


def identity(args, measured, base):
    paths = [Path(__file__), *q.ROOT.glob("pool_shed*.py"), q.ROOT / "plot_style.py",
             *q.ROOT.glob("native/src/*.rs"), *(q.ROOT / f"native/{name}" for name in
                ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml"))]
    source_inputs = base.metadata.get("source_timing_inputs", {})
    if any(sha(q.ROOT / name) != digest for name, digest in source_inputs.items()):
        raise ValueError("source timing evidence changed")
    return dict(workload="coding", sessions=args.sessions, repeats=args.repeats, deadline_s=DEADLINE, resident_load=LOAD,
        reference_gpus=REFERENCE_GPUS, reference_wan_tbps=REFERENCE_WAN_TBPS, scope=SCOPE,
        source_sha256={str(p.resolve().relative_to(q.ROOT)): sha(p) for p in paths},
        measurement_sha256=q.provenance(measured), source_timing_input_sha256=source_inputs,
        base_fleet_sha256=q.digest(dict(metadata=base.metadata, count=base.count.tolist(), context=base.context.tolist())),
        native_binary_sha256=sha(native.__file__),
        environment=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                         highs=q.highspy.Highs().version(), numerical_threads=1),
        cell_timeout_s=args.cell_timeout)


def plot(records, out):
    import matplotlib
    import plot_style

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("planning plot requires completed paired records")
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for axis, metric, title in zip(axes, ("total_planning_s", "primary_selection_s"),
            ("Preparation + first admission", "Primary selection")):
        for policy in POLICIES:
            grouped = {n: [p[metric] for r in records if r["sessions"] == n
                          for p in r["policies"] if p["policy"] == policy]
                       for n in sorted({r["sessions"] for r in records})}
            x = list(grouped)
            axis.plot(x, [np.median(v) for v in grouped.values()], marker=plot_style.POLICY_MARKERS[policy],
                      **plot_style.policy_style(policy))
            axis.fill_between(x, [min(v) for v in grouped.values()], [max(v) for v in grouped.values()],
                              color=plot_style.POLICY_COLORS[policy], alpha=.14)
        axis.set(xscale="log", yscale="log", xlabel="Individual source histories", ylabel="Time (s)", title=title)
        axis.set_xticks(x, [f"{n:,}" for n in x])
        axis.tick_params(labelsize=10)
        axis.grid(True, which="both", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Sparse planning · one history per source constraint", fontsize=14)
    fig.text(.5, .015, "Repeated measured trajectories; fractional allocations. Total LP time includes secondary minimization.\n"
             "Medians and observed ranges; calibration/imports and DES excluded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"planner_scaling.{suffix}", bbox_inches="tight")
    plt.close(fig)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    measured, base = inputs()
    config = identity(args, measured, base)
    config["identity"] = q.digest(config)
    manifest = args.out / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("planning benchmark configuration changed; use a fresh output directory")
    q.write_json(manifest, config)
    records = []
    for n in args.sessions:
        for repeat in range(1, args.repeats + 1):
            path = args.out / "cells" / f"{n}-{repeat}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.with_suffix(".log").open("w") as stream:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "cell", "--sessions", str(n),
                        "--repeat", str(repeat), "--out", str(args.out)], check=True, timeout=args.cell_timeout,
                        stdout=stream, stderr=subprocess.STDOUT, cwd=q.ROOT)
            record = json.loads(path.read_text())
            if record["identity"] != config["identity"] or sha(q.ROOT / record["witness_file"]) != record["witness_sha256"]:
                raise ValueError("cell identity or saved witness changed")
            records.append(record)
            print(json.dumps(dict(sessions=n, repeat=repeat, policies=[
                {key: p[key] for key in ("policy", "total_planning_s", "selection_s")} for p in record["policies"]])), flush=True)
    if identity(args, measured, base) != {k: v for k, v in config.items() if k != "identity"}:
        raise RuntimeError("benchmark inputs changed during execution")
    q.write_json(args.out / "summary.json", dict(identity=config["identity"], records=records, scope=SCOPE))
    rows = [dict(sessions=r["sessions"], repeat=r["repeat"], **{k: v for k, v in p.items()
            if not isinstance(v, (dict, list))}) for r in records for p in r["policies"]]
    with (args.out / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, sorted(set().union(*(r.keys() for r in rows))), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(records, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "cell", "plot"))
    parser.add_argument("--sessions", type=int, nargs="+", default=list(SESSIONS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cell-timeout", type=float, default=60.)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.repeats < 1 or not np.isfinite(args.cell_timeout) or args.cell_timeout <= 0 or any(n < 64 or n % 64 for n in args.sessions):
        parser.error("positive repeats/time limit and session counts divisible by 64 required")
    if args.command == "run":
        run(args)
    elif args.command == "cell":
        if len(args.sessions) != 1:
            parser.error("cell requires one session count")
        config = json.loads((args.out / "config.json").read_text())
        record = cell(args.sessions[0], args.repeat, args.out)
        q.write_json(args.out / "cells" / f"{args.sessions[0]}-{args.repeat}.json", {**record, "identity": config["identity"]})
    else:
        plot(json.loads((args.out / "summary.json").read_text())["records"], args.out)
