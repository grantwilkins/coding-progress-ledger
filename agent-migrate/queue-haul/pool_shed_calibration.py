"""Frozen A100 endpoint timing and direct coding-power anchors for pooled shed."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from functools import cache
from pathlib import Path

import numpy as np

import loaded_service_model as loaded
from profiles import KVTransfer, ModelProfile

ROOT = Path(__file__).resolve().parent
POWER_POINTS = ROOT / "outputs/azure-compact-calibration-20260813/pack-power-gated-points.csv"
POWER_PROFILE = ROOT / "outputs/azure-compact-calibration-20260813/gpt_oss_20b_a100_tp1_azure_300w_pack_power_gated.json"
TRANSITIONS = ROOT / "outputs/service-admission-transition-a100-20260816/summary.json"
CROSSOVER = ROOT / "outputs/policy-hardware-crossover-20260730"
LONG_PACKS = ROOT / "outputs/policy-hardware-width8-frontier-20260730"
PACKING = ROOT / "outputs/policy-hardware-width8-packing-20260730"
REGIONAL = ROOT / "outputs/a100-parity-20260907/timing"
SERVICE_REFERENCE = ROOT / "outputs/destination-v7-20260722/baseline-profile.json"
PATHS = (Path(__file__), ROOT / "loaded_service_model.py", ROOT / "profiles.py",
         loaded.PROFILE, loaded.OUT, POWER_POINTS, POWER_PROFILE, TRANSITIONS, SERVICE_REFERENCE,
         loaded.TRAINING, *loaded.BLOCKS, loaded.STANDALONE,
         CROSSOVER / "migrations.csv", CROSSOVER / "run_metadata.json",
         LONG_PACKS / "migrations.csv", LONG_PACKS / "scenarios.csv", LONG_PACKS / "policy_episodes.csv", LONG_PACKS / "run_metadata.json",
         PACKING / "policy_episodes.csv", PACKING / "policy_migrations.csv", PACKING / "plan.json", PACKING / "run_metadata.json",
         *(p.parent / "live_plan.json" for p in (loaded.TRAINING, *loaded.BLOCKS, loaded.STANDALONE)))


def replay_seconds(contexts, calibration, cached_tokens=0):
    """Match the historical profile's conservative singleton rate, including its tail."""
    context = np.asarray(contexts, dtype=float)
    if not np.all(np.isfinite(context)) or np.any(context <= 0):
        raise ValueError("replay contexts must be finite and positive")
    cached = np.asarray(cached_tokens, dtype=float)
    if not np.isfinite(cached).all() or np.any(cached < 0):
        raise ValueError("cached tokens must be finite and nonnegative")
    x, y = np.asarray(calibration["replay_context_tokens"]), np.asarray(calibration["replay_tps"])
    rate = np.interp(context, x, y, left=y.min(), right=y.min())
    return (context - np.minimum(cached, context)) / rate + calibration["replay_completion_s"]


def kv_state(contexts, calibration):
    """Native sealed payload and residual tokens; the latter still require replay."""
    context = np.asarray(contexts, dtype=float)
    if not np.isfinite(context).all() or np.any(context < 0) or np.any(context != np.floor(context)):
        raise ValueError("KV contexts must be finite nonnegative token counts")
    transfer = KVTransfer.parse(calibration["kv_transfer"])
    return tuple(np.fromiter((method(int(n)) for n in context.flat), dtype=np.int64).reshape(context.shape)
                 for method in (transfer.sealed_bytes, transfer.tail_tokens))


def service_work(context, prompt, output, calibration):
    """Request demand in units of the measured normal serving envelope."""
    contract = calibration["resident_service"]
    context, prompt, output = np.broadcast_arrays(context, prompt, output)
    if not np.isfinite([context, prompt, output]).all() or np.any(context <= 0) or np.any(prompt < 0) or np.any(output < 0):
        raise ValueError("invalid resident request shape")
    rates = [np.interp(context, *np.asarray(contract[phase]).T,
                       left=min(row[1] for row in contract[phase]), right=min(row[1] for row in contract[phase]))
             for phase in ("prefill", "decode")]
    return (prompt / rates[0] + output / rates[1]) / contract["bound"]


def source_power(fleet, calibration):
    """Existing phase-power curve at the source's cycle-average offered token rates."""
    if hashlib.sha256(POWER_PROFILE.read_bytes()).hexdigest() != calibration["sources"][str(POWER_PROFILE.relative_to(ROOT))]:
        raise ValueError("source power calibration changed")
    power = ModelProfile.load(POWER_PROFILE).case().phase_power
    means = np.array([[np.mean([r[key] for r in sequence]) for key in ("prompt", "output")]
                      for sequence in fleet.metadata["turn_sequences"]])
    rates = fleet.count @ means * fleet.metadata["source_session_rps"] / fleet.gpus
    if not np.isfinite(rates).all() or not power.contains(*rates):
        raise ValueError("source token rates lie outside the measured power hull")
    load = power.load(*rates)
    if not power.measured_power_bootstrap or any(load > curve[-1][0] for curve in power.measured_power_bootstrap):
        raise ValueError("source power lacks supported measured bootstrap curves")
    active, idle = power.power(load), power.power(0.)
    draws = [float(np.interp(load, *np.asarray(curve).T) - curve[0][1]) for curve in power.measured_power_bootstrap]
    return {"active_w": active, "idle_w": idle, "delta_w": active - idle, "delta_draws_w": draws,
            "prefill_tokens_per_s_per_gpu": float(rates[0]), "decode_tokens_per_s_per_gpu": float(rates[1]),
            "power_load": load, "grouped_cv_rmse_w": power.grouped_cv_rmse_w,
            "within_5w_fraction": power.within_5w_fraction,
            "scope": "Whole-GPU cycle-average active power at the declared source rates minus awake idle; within the measured phase-rate hull. Bootstrap samples curve uncertainty, not the reported grouped-CV model error or trajectory transfer. Multiplying this full delta by shed fraction is a separate linear allocation proxy, not evaluation of the nonlinear remaining-workload power."}


def _resident_service():
    root = ROOT / "outputs/destination-v7-20260722"
    paths = [root / name for name in ("plan.json", "baseline-profile.json", "anchor-gate.json", "service/frontier.json", "acceptance.json")]
    plan, profile, anchors, frontier, acceptance = [json.loads(path.read_text()) for path in paths]
    if hashlib.sha256(paths[1].read_bytes()).hexdigest() != plan["baseline_profile"]["sha256"] or not anchors["within_limit"]:
        raise ValueError("resident service normalization or anchor gate changed")
    curves = {phase: dict(profile["cases"]["central"][f"{phase}_tps"]["1"]) for phase in ("prefill", "decode")}
    for row in anchors["anchors"]:
        curves[row["metric"]][row["context_tokens"]] = row["observed_tokens_per_s"]
    cases = []
    for split in ("tune", "validation"):
        for path in sorted((root / "service" / split / "coding").glob("normal-*/result.json")):
            row = json.loads(path.read_text())
            if row["status"] != "complete" or not row["classification"]["normal"] or not row["drained"]:
                raise ValueError("recorded coding resident service check failed")
            cases.append({"path": str(path.relative_to(ROOT)), **row})
            paths.append(path)
    bound = frontier["nested_bounds"]["normal"]
    if len(cases) != 4 or not min(row["radius"] for row in cases) <= bound <= max(row["radius"] for row in cases):
        raise ValueError("resident service requires four bracketing coding checks")
    return {"bound": bound, **{phase: sorted(values.items()) for phase, values in curves.items()},
            "context_limit": min(max(values) for values in curves.values()),
            "targets": plan["service"]["slos"]["normal"], "evidence": cases,
            "full_profile_accepted": acceptance["accepted"],
            "scope": "Retrospective coding serving envelope with its original context-dependent baseline curves and live anchor replacements; four recorded tuning/validation checks pass. Full destination profile was not accepted. Other request mixtures and shared migration occupancy are declared transfers, not a generic TTFT/TPOT guarantee.",
            "extrapolation": "Outside measured context curves hold the slowest measured phase rate; explicit sensitivity, not a validated bound.",
            "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    errors = abs(predicted - observed) / observed
    return {"episodes": len(observed), "median_relative_error": float(np.median(errors)),
            "p90_relative_error": float(np.quantile(errors, .9)),
            "false_feasible_25s": int(np.sum((predicted <= 25) & (observed > 25)))}


def _regional_components(value):
    """Fit primitive regional rates on the existing pure-action training split."""
    paths = [REGIONAL / name for name in ("plan.json", "scale-protocol.json", "scale-fit.json", "results.csv")]
    plan, protocol, frozen = [json.loads(path.read_text()) for path in paths[:3]]
    if (hashlib.sha256(paths[0].read_bytes()).hexdigest() != protocol["plan_sha256"]
            or hashlib.sha256(paths[1].read_bytes()).hexdigest() != frozen["protocol_sha256"]
            or set(protocol["training_ids"]) & set(protocol["holdout_ids"])):
        raise ValueError("regional component split provenance changed")
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]}
    rows, timing = [], value["timing"][0]
    for row in _read(paths[3]):
        training = row["scenario_id"] in protocol["training_ids"]
        if not training and row["scenario_id"] not in protocol["holdout_ids"] or training and row["policy"] == "fixed_mixed":
            continue
        candidates = sorted((REGIONAL / "scenarios" / row["scenario_id"]).glob("attempt-*/result.json"))
        path = candidates[-1]
        actual = json.loads(path.read_text())
        if (row["status"] != "complete" or actual["status"] != "complete"
                or abs(actual["migration_s"] - float(row["migration_s"])) > 1e-8):
            raise ValueError("regional component observations must match completed results")
        paths.append(path)
        scenario, features, observed, descriptors = scenarios[row["scenario_id"]], [], [], []
        contexts = {s["session_id"]: s["initial_tokens"] for s in scenario["sessions"]}
        for destination in ("east", "germany"):
            moves = [m for m in scenario["moves"] if m["destination_instance"] == destination]
            replay, kv = [np.array([contexts[m["session_id"]] for m in moves if m["method"] == method])
                          for method in ("replay", "kv_transfer")]
            work = replay_seconds(replay, value)
            kappa = np.ones_like(work) if np.any(replay > value["batch_context_limit"]) else np.interp(
                replay, value["packing_context_tokens"], timing["packing_kappa"])
            duration = float(np.exp(timing["beta"] * scenario["background"][destination][0])
                             * (np.sum(kappa * work) + max((1 - kappa) * work, default=0)))
            state, residual = kv_state(kv, value)
            tail = np.interp(len(kv), [0, 1, 8], [0, timing["kv_completion_s"], timing["kv_batch_completion_s"]])
            features.append((duration, state.sum(), tail + residual.sum() / value["kv_tail_replay_tps"]))
            descriptors.append((replay, scenario["background"][destination][0], len(kv), residual.sum()))
            requests = [m["request"] for m in actual["requests"] if m["destination_instance"] == destination]
            observed.append((max(r["end_ns"] for r in requests) - actual["started_ns"]) / 1e9)
        rows.append((training, row["policy"], features, observed, float(row["migration_s"]), descriptors, row["scenario_id"]))
    if sum(r[0] for r in rows) != 53 or sum(not r[0] for r in rows) != 24:
        raise ValueError("regional components require 53 pure-action training and 24 validation episodes")
    def fit(training, timing_draw):
        replay_factors, rates = [], []
        for destination in range(2):
            for policy, column, output in (("fixed_replay", 0, replay_factors), ("fixed_kv_transfer", 1, rates)):
                selected = [r for r in training if r[1] == policy]
                features = []
                for row in selected:
                    context, load, count, residual = row[5][destination]
                    if column:
                        features.append((row[2][destination][1], np.interp(count, [0, 1, 8],
                            [0, timing_draw["kv_completion_s"], timing_draw["kv_batch_completion_s"]]) + residual / value["kv_tail_replay_tps"]))
                    else:
                        work = replay_seconds(context, value)
                        kappa = np.ones_like(work) if np.any(context > value["batch_context_limit"]) else np.interp(
                            context, value["packing_context_tokens"], timing_draw["packing_kappa"])
                        features.append((np.exp(timing_draw["beta"] * load) * (np.sum(kappa * work)
                                         + max((1 - kappa) * work, default=0)), 0.))
                x, tail = np.asarray(features, dtype=float).T
                y = np.array([r[3][destination] for r in selected]) - tail
                coefficient = float(x @ y / (x @ x))
                if not np.isfinite(coefficient) or coefficient <= 0:
                    raise ValueError("regional primitive calibration must be finite and positive")
                output.append(1 / coefficient if column else coefficient)
        return replay_factors, rates

    training = [r for r in rows if r[0]]
    rng = np.random.default_rng(5)
    bootstrap_ids = []
    for draw, timing_draw in enumerate(value["timing"]):
        selected = training if not draw else [training[i] for policy in ("fixed_replay", "fixed_kv_transfer")
            for i in rng.choice([i for i, r in enumerate(training) if r[1] == policy], sum(r[1] == policy for r in training))]
        timing_draw["regional_replay_factor"], timing_draw["regional_kv_bytes_per_s"] = fit(selected, timing_draw)
        bootstrap_ids.append([r[6] for r in selected])
    replay_factors, rates = fit(training, value["timing"][0])
    reports = {}
    for policy in ("fixed_replay", "fixed_kv_transfer", "fixed_mixed", "aggregate"):
        selected = [r for r in rows if not r[0] and (policy == "aggregate" or r[1] == policy)]
        observed = np.array([r[4] for r in selected])
        predicted = np.array([max(max(f[0] * replay_factors[d], f[1] / rates[d] + f[2])
                                  for d, f in enumerate(r[2])) for r in selected])
        reports[policy] = {**_metrics(observed, predicted), "mae_s": float(np.mean(abs(predicted - observed))),
            "r2": float(1 - np.sum((predicted - observed) ** 2) / np.sum((observed - observed.mean()) ** 2)),
            "residual_s": (observed - predicted).tolist()}
    reports["gate_pass"] = reports["aggregate"]["mae_s"] <= protocol["gates"]["mae_s"] and reports["aggregate"]["r2"] >= protocol["gates"]["r2"]
    return {"replay_factor": replay_factors, "endpoint_bytes_per_s": rates, "validation": reports,
            "gates": protocol["gates"], "training_episodes": 53, "heldout_episodes": 24,
            "bootstrap_episode_ids": bootstrap_ids,
            "bootstrap_seed": 5, "bootstrap_scope": "Resample complete training episodes within each pure action; routes remain paired; refit features using each draw's batch/completion parameters; no heldout residuals enter the fits",
            "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "scope": "Retrospective fixed-split component calibration: 26 pure replay and 27 pure KV episodes fit regional replay work scales and application transfer rates after measured completion cost; all 27 mixed training episodes unused; 24 pre-existing holdouts check max-overlapped route completion. Effective node rates include application transport overhead, not backbone capacity or a KV ingest constraint. This check does not validate fleet scheduling or changing-load execution."}


def _resident_interference(value):
    """Resident completion deficit during replay, normalized by same-episode service."""
    plan, protocol = [json.loads((REGIONAL / name).read_text()) for name in ("plan.json", "scale-protocol.json")]
    profile_path = ROOT / "profiles" / Path(plan["model_profile"]["path"]).name
    if hashlib.sha256(profile_path.read_bytes()).hexdigest() != plan["model_profile"]["sha256"]:
        raise ValueError("resident interference service reference changed")
    profile = ModelProfile.load(profile_path).case()
    work = 604 / profile.prefill.rate(604, 1) + 64 / profile.decode.rate(604, 1)
    scenarios = {r["scenario_id"]: r for r in plan["scenarios"]}
    rows, excluded, sources = [], [], {str(profile_path.relative_to(ROOT)): hashlib.sha256(profile_path.read_bytes()).hexdigest()}
    for name, checksum in value["regional_components"]["sources"].items():
        if not name.endswith("/result.json"):
            continue
        path = ROOT / name
        actual = json.loads(path.read_text())
        scenario = scenarios[actual["scenario_id"]]
        if scenario["policy"] == "fixed_mixed":
            continue
        sources[name] = checksum
        for route in ("east", "germany"):
            load = scenario["background"][route][0]
            if not load:
                continue
            trace = path.parent / f"sink_load_{route}.jsonl"
            sources[str(trace.relative_to(ROOT))] = hashlib.sha256(trace.read_bytes()).hexdigest()
            requests = [json.loads(line) for line in trace.read_text().splitlines()]
            if any(r["status_code"] != 200 or (r["prompt_tokens"], r["output_tokens"]) != (604, 64) for r in requests):
                raise ValueError("resident interference requests violate the fixed 604/64 contract")
            ends = (np.array([r["end_ns"] for r in requests], dtype=np.int64) - actual["started_ns"]) / 1e9
            baseline = ends[(ends >= -20) & (ends < 0)]
            rate = (len(baseline) - 1) / np.ptp(baseline) if len(baseline) >= 3 and np.ptp(baseline) > 0 else 0.
            if not rate or abs(rate / (load / work) - 1) > .1:
                excluded.append({"scenario_id": actual["scenario_id"], "route": route, "load": load,
                                 "observed_rps": float(rate), "intended_rps": load / work})
                continue
            duration = (max(m["request"]["end_ns"] for m in actual["requests"] if m["destination_instance"] == route)
                        - actual["started_ns"]) / 1e9
            expected, completed = rate * duration, int(np.sum((ends >= 0) & (ends < duration)))
            rows.append({"scenario_id": actual["scenario_id"], "route": route, "policy": scenario["policy"],
                "training": actual["scenario_id"] in protocol["training_ids"], "load": load,
                "baseline_rps": float(rate), "duration_s": float(duration), "expected_completions": float(expected),
                "observed_completions": completed, "deficit_requests": float(expected - completed)})

    def fit(selected):
        x = np.array([r["expected_completions"] for r in selected])
        y = np.array([r["deficit_requests"] for r in selected])
        loss = float(x @ y / (x @ x))
        if not np.isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError("resident replay loss must be a measured fraction between zero and one")
        return loss

    training = [r for r in rows if r["training"] and r["policy"] == "fixed_replay"]
    if len(training) != 31 or len({r["scenario_id"] for r in training}) != 23:
        raise ValueError("resident replay calibration requires 31 routes across 23 training episodes")
    for timing, ids in zip(value["timing"], value["regional_components"]["bootstrap_episode_ids"], strict=True):
        timing["resident_replay_loss"] = fit([r for scenario_id in ids for r in training if r["scenario_id"] == scenario_id])
    validation = {}
    for policy in ("fixed_replay", "fixed_kv_transfer"):
        selected = [r for r in rows if not r["training"] and r["policy"] == policy]
        expected = np.array([r["expected_completions"] for r in selected])
        actual = np.array([r["deficit_requests"] for r in selected])
        prediction = fit(training) * expected if policy == "fixed_replay" else np.zeros(len(selected))
        validation[policy] = {"routes": len(selected), "episodes": len({r["scenario_id"] for r in selected}),
            "mae_requests": float(np.mean(abs(prediction - actual))),
            "normalized_mae": float(np.mean(abs(prediction - actual) / expected)),
            "median_observed_loss": float(np.median(actual / expected)), "residual_requests": (actual - prediction).tolist()}
    return {"replay_loss": fit(training), "training_routes": 31, "training_episodes": 23,
            "validation": validation, "measurements": rows, "excluded_baselines": excluded, "sources": sources,
            "reference_request_tokens": [604, 64], "reference_work_s": work,
            "selection": "At least three baseline completions in the preceding 20 seconds; observed baseline rate within 10% of intended rate excludes client-backpressured cases; accepted loads are 0.25 and 0.5",
            "bootstrap_scope": "Same complete training-episode resamples as regional timing; both routes stay paired; no heldout observations fit the coefficient",
            "scope": "Retrospective resident completion deficit relative to same-episode baseline throughput; not GPU occupancy. KV is a zero-loss negative-control diagnostic, not a fitted GPU cost throughout network transfer. Shape/batch/load transfer is assumed. Post-replay recovery uses the declared reference service capacity; continued-arrival recovery is not measured because episode shutdown stops load generation."}


def regional_check(value):
    from scipy.optimize import brentq
    from pool_shed_campaign import batch_time

    root = ROOT / "outputs/a100-parity-20260907/timing"
    paths = [root / name for name in ("plan.json", "results.csv", "scale-protocol.json", "scale-fit.json")]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    plan, protocol, frozen = [json.loads(path.read_text()) for path in (paths[0], paths[2], paths[3])]
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]}
    oracle = {row["scenario_id"]: row["predicted_s"] for row in frozen["predictions"]}
    if (protocol["plan_sha256"] != hashes[str(paths[0].relative_to(ROOT))]
            or frozen["protocol_sha256"] != hashes[str(paths[2].relative_to(ROOT))]
            or set(oracle) != set(protocol["holdout_ids"]) or len(oracle) != 24):
        raise ValueError("regional prospective holdout provenance changed")
    rows, timing = [], value["timing"][0]
    for row in _read(paths[1]):
        if row["scenario_id"] not in oracle:
            continue
        if row["status"] != "complete" or int(row["started_wall_ns"]) <= frozen["frozen_wall_ns"]:
            raise ValueError("regional holdout was incomplete or observed before freezing predictions")
        scenario, routes = scenarios[row["scenario_id"]], []
        contexts = {s["session_id"]: s["initial_tokens"] for s in scenario["sessions"]}
        for destination in ("east", "germany"):
            moves = [m for m in scenario["moves"] if m["destination_instance"] == destination]
            replay, kv = [np.array([contexts[m["session_id"]] for m in moves if m["method"] == method])
                          for method in ("replay", "kv_transfer")]
            work = replay_seconds(replay, value)
            kappa = np.ones_like(work) if np.any(replay > value["batch_context_limit"]) else np.interp(
                replay, value["packing_context_tokens"], timing["packing_kappa"])
            duration = float(batch_time(np.ones((1, len(work))), work, timing["beta"], kappa,
                                       scenario["background"][destination][0])[0]) if len(work) else 0.
            tail = np.interp(len(kv), [0, 1, 8], [0, timing["kv_completion_s"], timing["kv_batch_completion_s"]])
            state, residual = kv_state(kv, value)
            tail += residual.sum() / value["kv_tail_replay_tps"]
            state = state.sum()
            routes.append((duration, tail, 2 * replay.sum(), state, scenario["bandwidth_mbps"][destination] * 125_000))
        shared = plan["network_contract"]["aggregate"]["natural_mbps"] * 125_000

        def excess(deadline):
            rates = np.array([logs / (deadline - duration) + state / (deadline - tail)
                              for duration, tail, logs, state, _ in routes])
            return max(np.max(rates / np.array([r[4] for r in routes])), rates.sum() / shared) - 1

        lower = max(max(r[:2]) for r in routes) + 1e-8
        upper = lower + sum(r[2] + r[3] for r in routes) / min(shared, *(r[4] for r in routes)) + 1
        rows.append((row["policy"], float(row["migration_s"]), brentq(excess, lower, upper), oracle[row["scenario_id"]]))
    policies = sorted({r[0] for r in rows})
    if len(rows) != 24 or len(policies) != 3 or any(sum(r[0] == p for r in rows) != 8 for p in policies):
        raise ValueError("regional holdout requires eight episodes per action family")
    reports = {}
    for name, column in (("current_pool", 2), ("frozen_oracle", 3)):
        reports[name] = {}
        for policy in [*policies, "aggregate"]:
            selected = [r for r in rows if policy == "aggregate" or r[0] == policy]
            observed, predicted = np.array([r[1] for r in selected]), np.array([r[column] for r in selected])
            reports[name][policy] = {**_metrics(observed, predicted), "mae_s": float(np.mean(abs(predicted - observed))),
                "r2": float(1 - np.sum((predicted - observed) ** 2) / np.sum((observed - observed.mean()) ** 2))}
        aggregate = reports[name]["aggregate"]
        reports[name]["gate_pass"] = aggregate["mae_s"] <= protocol["gates"]["mae_s"] and aggregate["r2"] >= protocol["gates"]["r2"]
    return {**reports, "sources": hashes, "gates": protocol["gates"],
            "scope": "Fixed recorded actions/routes/loads; one GPU per destination; final migration completion; 24 prospective episodes; timing transfer diagnostic, not optimizer or fleet-admission validation; frozen oracle is a queue-family fit"}


def loaded_execution_check(value):
    """Check base loaded-batch progress through the independent execution engine."""
    from types import SimpleNamespace
    from pool_shed_execution import execute_pooled

    _, validation, _ = loaded.load_evidence()
    context = np.array(loaded.CONTEXTS, dtype=float)
    fleet = SimpleNamespace(count=np.ones(8), context=context, demand=np.zeros(8), gain=np.ones(8) / 8,
        memory_tokens=context, baseline_kv=0., kv_capacity=1e12, gpus=1, nodes=1,
        t1=replay_seconds(context, value), log=2 * context, kv=kv_state(context, value)[0],
        metadata={"turn_sequences": [[] for _ in context], "source_session_rps": 0.,
                  "batch_context_limit": max(context)})
    timing = {**value["timing"][0], "regional_replay_factor": [1., 1.]}
    base = {**value, "regional_components": {"replay_factor": [1., 1.]}}
    observed, predicted = [], []
    for row in validation:
        if row["method"] != "replay":
            continue
        endpoint = np.full(2, row["bandwidth_mbps"] * 125_000)
        table = SimpleNamespace(fleet=fleet, replay=np.ones((1, 8)), kv=np.zeros((1, 8)), route=np.array([0]),
            deadline=300., endpoint=endpoint, budgets=np.r_[endpoint, endpoint.sum()], load=row["rho"])
        result = execute_pooled(table, np.ones(1), timing, base, chunks=1)
        if result["completed_sessions"] != 8:
            raise RuntimeError("loaded execution check did not finish all recorded sessions")
        observed.append(row["commit_s"])
        predicted.append(result["last_completion_s"])
    report = _metrics(observed, predicted)
    return {**report, "gate_pass": report["episodes"] == 220 and report["p90_relative_error"] <= .05
            and report["false_feasible_25s"] == 0,
            "scope": "220 pre-existing replay holdouts; original fixed eight-context batch, resident offered-load reference, frozen source. Regional compute factors disabled to reproduce the original runtime. This validates base loaded progress, not its composition with regional scaling or changing-load admission."}


def resident_execution_check(value, executed=None):
    """Compare engine debt at its handoff with recorded resident completion deficits."""
    from pool_shed_execution import regional_execution_check

    episodes = {r["scenario_id"]: r for r in (executed or regional_execution_check(value))["predictions"]}
    rows = []
    for observed in value["resident_interference"]["measurements"]:
        if observed["training"] or observed["policy"] != "fixed_replay":
            continue
        route = int(observed["route"] == "germany")
        events = [e for e in episodes[observed["scenario_id"]]["completion_events"] if e["route"] == route]
        event = max(events, key=lambda e: e["completion_s"])
        predicted = event["resident_debt_work_s"] * observed["baseline_rps"] / observed["load"]
        rows.append({"scenario_id": observed["scenario_id"], "route": observed["route"],
            "observed_deficit_requests": observed["deficit_requests"], "predicted_deficit_requests": predicted,
            "observed_migration_s": observed["duration_s"], "predicted_migration_s": event["completion_s"],
            "resident_debt_work_s": event["resident_debt_work_s"],
            "baseline_rps": observed["baseline_rps"], "offered_load": observed["load"],
            "normalization_completions": observed["expected_completions"]})
    if (len(rows) != 6 or not np.isfinite([[r[k] for k in ("predicted_deficit_requests", "observed_deficit_requests",
            "normalization_completions", "resident_debt_work_s")] for r in rows]).all()
            or any(not 0 <= r["resident_debt_work_s"] <= r["offered_load"] * r["predicted_migration_s"] + 1e-8 for r in rows)):
        raise ValueError("resident execution check requires six finite heldout route observations")
    errors = np.array([r["predicted_deficit_requests"] - r["observed_deficit_requests"] for r in rows])
    return {"routes": len(rows), "episodes": len({r["scenario_id"] for r in rows}),
            "mae_requests": float(np.mean(abs(errors))),
            "normalized_mae": float(np.mean(abs(errors) / [r["normalization_completions"] for r in rows])),
            "predictions": rows,
            "scope": "Independent engine debt at each predicted regional replay handoff versus six pre-existing route deficits at observed handoff. Reference work converts to requests using each route's baseline completion rate / offered load. Timing-window mismatch is retained. This checks replay-created debt, not post-migration recovery under continuing arrivals."}


def _kv_completion(value, draws):
    singleton = [[float(r["initial_response_s"]) + float(r["initial_validation_s"])
                  for r in _read(CROSSOVER / "migrations.csv")
                  if (r["method"], r["activity"], int(r["concurrency"]), int(r["repeat"]))
                  == ("kv_transfer", "none", 1, repeat)] for repeat in range(3)]
    migrations, groups = {}, {}
    scenarios = {r["scenario_id"]: r for r in _read(LONG_PACKS / "scenarios.csv")}
    for row in _read(LONG_PACKS / "migrations.csv"):
        migrations.setdefault(row["scenario_id"], []).append(row)
    for row in _read(LONG_PACKS / "policy_episodes.csv"):
        if row["policy"] != "kv_only":
            continue
        moves, scenario = migrations[row["scenario_id"]], scenarios[row["scenario_id"]]
        if (row["status"] != "complete" or len(moves) != 8 or int(row["completed_migrations"]) != 8
                or (scenario["activity"], int(scenario["concurrency"])) != ("none", 8)
                or any(m["method"] != "kv_transfer" for m in moves)):
            raise ValueError("KV completion requires complete zero-background eight-session KV batches")
        tails = np.array([float(m["initial_response_s"]) + float(m["initial_validation_s"]) for m in moves])
        if not np.isfinite(tails).all() or np.any(tails <= 0):
            raise ValueError("KV batch completion observations must be finite and positive")
        groups.setdefault(row["condition"], []).append((int(row["episode"]), float(tails.max())))
    if any(len(rows) != 24 for rows in singleton) or len(groups) != 24 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("KV completion requires three repeats of 24 singleton and batch conditions")
    batches = np.array([[tail for _, tail in sorted(rows)] for rows in groups.values()]).T
    singleton = np.asarray(singleton)
    if not np.isfinite(np.r_[singleton.ravel(), batches.ravel()]).all() or min(singleton.min(), batches.min()) <= 0:
        raise ValueError("KV completion observations must be finite and positive")
    rng = np.random.default_rng(4)
    for i, timing in enumerate(value["timing"]):
        for name, samples in (("kv_completion_s", singleton), ("kv_batch_completion_s", batches)):
            timing[name] = float(np.median(samples[rng.choice(2, 2) if i else np.arange(2)]))
    if len(value["timing"]) != draws + 1:
        raise ValueError("KV completion timing draw count changed")
    return {"training_episodes_per_width": 48, "heldout_episodes_per_width": 24, "bootstrap_seed": 4,
            "heldout": {name: {"median_absolute_error_s": float(np.median(abs(samples[2] - np.median(samples[:2])))),
                "p90_relative_error": float(np.quantile(abs(np.median(samples[:2]) / samples[2] - 1), .9))}
                for name, samples in (("singleton", singleton), ("width8", batches))},
            "scope": "Response generation plus validation after first response; excludes ingestion. First two repeats per condition fit, third checks; bootstrap complete repeat-position groups independently for widths 1 and 8. Batch tail is maximum per-request tail for synchronized readiness; intermediate widths, mixed replay/KV, resident load and overlap without KV compute accounting remain transfers, not GPU scheduling guarantees."}


def _policy_checks(value):
    scenarios = {r["scenario_id"]: r for r in json.loads((PACKING / "plan.json").read_text())["scenarios"]}
    migrations, groups = {}, {}
    for row in _read(PACKING / "policy_migrations.csv"):
        migrations.setdefault(row["scenario_id"], []).append(row)
    for row in _read(PACKING / "policy_episodes.csv"):
        scenario, moves = scenarios[row["scenario_id"]], migrations[row["scenario_id"]]
        if (len(moves) != 8 or row["status"] != "complete" or scenario["activity"] != "none"
                or any(m["method"] not in ("replay", "kv_transfer") for m in moves)):
            raise ValueError("policy timing requires eight complete zero-resident-load recorded moves")
        actual = float(row["commit_100_s"])
        if abs(actual - max(float(m["reaction_commit_s"]) for m in moves)) > 1e-8:
            raise ValueError("policy timing commit milestones differ")
        context = np.array([int(m["context_tokens"]) for m in moves])
        replay = np.array([m["method"] == "replay" for m in moves])
        work, timing = replay_seconds(context[replay], value), value["timing"][0]
        kappa = np.interp(context[replay], value["packing_context_tokens"], timing["packing_kappa"])
        if np.any(context[replay] > value["batch_context_limit"]):
            kappa = np.ones_like(work)
        duration = float((kappa * work).sum() + max((1 - kappa) * work, default=0))
        logs = 2 * context[replay].sum()
        state, residual = kv_state(context[~replay], value)
        state = state.sum()
        tail = (np.interp((~replay).sum(), [1, 8], [timing["kv_completion_s"], timing["kv_batch_completion_s"]])
                + residual.sum() / value["kv_tail_replay_tps"]) if (~replay).any() else 0
        bandwidth = scenario["bandwidth_mbps"] * 125_000
        a = duration + tail + (logs + state) / bandwidth
        discriminant = (duration - tail + (logs - state) / bandwidth) ** 2 + 4 * logs * state / bandwidth ** 2
        predicted = float((a + np.sqrt(discriminant)) / 2)
        groups.setdefault((row["policy"], row["condition"]), []).append(
            (int(row["episode"]), actual, predicted, int(replay.sum())))
    if len(groups) != 160 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("policy timing requires four policies and forty three-repeat conditions")
    policies = sorted({policy for policy, _ in groups})
    heldout = {policy: [max(rows) for (p, _), rows in groups.items() if p == policy] for policy in policies}
    return {"heldout_third_repeat": {p: _metrics([r[1] for r in rows], [r[2] for r in rows]) for p, rows in heldout.items()},
            "selected_conditions": {f"{p}/{condition}": {"episodes": len(rows),
                "observed_median_s": float(np.median([r[1] for r in rows])),
                "predicted_median_s": float(np.median([r[2] for r in rows])),
                "recorded_replay_counts": sorted({r[3] for r in rows})}
                for (p, condition), rows in groups.items()
                if condition.startswith(("mixed-fixed-10000", "large-fixed-5000", "large-fixed-10000"))},
            "scope": "Recorded actions, one destination, zero resident load, configured WAN; final route commit; third within-condition repeat held out; timing diagnostic, not planner optimality or mixed-action accuracy certification"}


def _packing(value, draws):
    scenarios = {r["scenario_id"]: r for r in json.loads((PACKING / "plan.json").read_text())["scenarios"]}
    groups = {}
    for row in _read(PACKING / "policy_episodes.csv"):
        if row["policy"] == "replay_only":
            groups.setdefault(row["condition"], []).append(row)
    if len(groups) != 40 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("packing requires forty matched three-repeat conditions")
    training, heldout = [], []
    for rows in groups.values():
        for repeat, row in enumerate(sorted(rows, key=lambda r: int(r["episode"]))):
            scenario = scenarios[row["scenario_id"]]
            context = np.array([s["initial_tokens"] for s in scenario["sessions"]])
            if row["status"] != "complete" or int(row["completed_migrations"]) != 8 or len(context) != 8:
                raise ValueError("packing calibration requires complete width-eight episodes")
            work = replay_seconds(context, value)
            route = 2 * context.sum() / (scenario["bandwidth_mbps"] * 125_000) + 8 * value["switch_s"]
            item = (repeat, context, work, route, float(row["commit_100_s"]))
            (training if repeat < 2 and len(set(context)) == 1 else heldout).append(item)
    value["packing_context_tokens"] = sorted({int(row[1][0]) for row in training})
    value["batch_context_limit"] = max(value["packing_context_tokens"])
    if len(training) != 64 or len(heldout) != 56 or value["packing_context_tokens"] != [2048, 4096, 8192, 16384]:
        raise ValueError("packing fit/holdout split changed")

    def fit(rows):
        knots = [float(np.median([(observed - route - work.max()) / (work.sum() - work.max())
                                  for _, context, work, route, observed in rows if context[0] == anchor]))
                 for anchor in value["packing_context_tokens"]]
        if not np.all(np.isfinite(knots)) or not np.all((np.asarray(knots) >= 0) & (np.asarray(knots) <= 1)):
            raise ValueError("packing coefficients must be finite and between zero and one")
        return knots

    rng = np.random.default_rng(3)
    fits = [fit(training)] + [fit([row for repeat in rng.choice(2, 2) for row in training if row[0] == repeat])
                             for _ in range(draws)]
    for timing, knots in zip(value["timing"], fits, strict=True):
        timing["packing_kappa"] = knots
    observed, predicted, previous = [], [], []
    for _, context, work, route, actual in heldout:
        kappa = np.interp(context, value["packing_context_tokens"], fits[0])
        observed.append(actual)
        predicted.append(route + (kappa * work).sum() + ((1 - kappa) * work).max())
        previous.append(route + work.max() + value["timing"][0]["kappa"] * (work.sum() - work.max()))
    check = _metrics(observed, predicted)
    if check["p90_relative_error"] > .20 or check["false_feasible_25s"]:
        raise ValueError("coding packing model failed its independent replay holdout")
    return {"training_uniform_episodes": 64, "heldout": check, "old_loaded_formula": _metrics(observed, previous),
            "split": "Within each condition, first two uniform episodes fit; third uniform and all mixed episodes validate",
            "bootstrap_seed": 3, "bootstrap_groups": 2}


def _transfer_checks(value):
    singleton = [r for r in _read(CROSSOVER / "migrations.csv") if r["method"] == "replay" and int(r["repeat"]) == 2]
    checks = {"singleton_repeat2": _metrics([float(r["initial_request_s"]) for r in singleton],
              replay_seconds([int(r["measured_prompt_tokens"]) for r in singleton], value))}
    migrations = {}
    for row in _read(LONG_PACKS / "migrations.csv"):
        migrations.setdefault(row["scenario_id"], []).append(row)
    observed, predicted, previous, endpoint_errors = [], [], [], []
    for row in _read(LONG_PACKS / "scenarios.csv"):
        if (row["kind"], row["method"], row["activity"], int(row["concurrency"])) != ("migration", "replay", "none", 8):
            continue
        context = np.array([int(r["measured_prompt_tokens"]) for r in migrations[row["scenario_id"]]])
        work = replay_seconds(context, value)
        if context.max() <= value["batch_context_limit"]:
            raise ValueError("long-pack serial calibration support changed")
        observed.append(float(row["migration_s"]))
        route = 2 * context.sum() / (float(row["bandwidth_mbps"]) * 125_000) + len(work) * value["switch_s"]
        predicted.append(route + work.sum())
        previous.append(route + work.max() + value["timing"][0]["kappa"] * (work.sum() - work.max()))
        endpoint = (max(int(r["switch_end_ns"]) for r in migrations[row["scenario_id"]])
                    - min(int(r["initial_start_ns"]) for r in migrations[row["scenario_id"]])) / 1e9
        endpoint_errors.append(abs(endpoint - observed[-1]))
    checks["long_context_width8"] = _metrics(observed, predicted)
    checks["long_context_old_loaded_formula"] = _metrics(observed, previous)
    checks["last_commit_endpoint_max_difference_s"] = max(endpoint_errors)
    check = checks["long_context_width8"]
    if check["p90_relative_error"] > .20 or check["false_feasible_25s"] or max(endpoint_errors) > 1e-8:
        raise ValueError("serial long-pack model failed its external calibration check")
    checks["scope"] = "Long-pack serial family chosen after inspecting these 72 episodes: external calibration check, not untouched holdout; same recorded vLLM0.22/LMCache0.5.1 image path, not a verified identical image hash; older vLLM0.10 c2/c4 runs excluded"
    identities = [json.loads((path / "run_metadata.json").read_text())["config"]["sandbox"] for path in (CROSSOVER, LONG_PACKS, PACKING)]
    if len(set(identities)) != 1 or len(singleton) != 24 or len(observed) != 72:
        raise ValueError("historical timing transfer evidence changed")
    return checks


@cache
def calibration(draws=8):
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 0:
        raise ValueError("timing draw count must be a nonnegative integer")
    model = loaded.validate_model(json.loads(loaded.OUT.read_text()))
    training, validation, provenance = loaded.load_evidence()
    profile = ModelProfile.load(loaded.PROFILE)
    case, physics = profile.case(), loaded._physics(profile)
    reference = json.loads((loaded.TRAINING.parent / "live_plan.json").read_text())["calibration"]["service_calibration"]
    if hashlib.sha256(SERVICE_REFERENCE.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("loaded serving reference profile changed")
    x, y = case.replay.by_concurrency[1]
    result = {"schema": "queue-haul-pool-calibration-v1", "model": profile.model,
              "replay_context_tokens": x.tolist(), "replay_tps": y.tolist(),
              "replay_completion_s": case.replay_completion_s, "switch_s": case.switch_s,
              "kv_block_tokens": case.kv_transfer.block_tokens, "kv_block_bytes": case.kv_transfer.block_bytes,
              "kv_transfer": asdict(case.kv_transfer), "kv_tail_replay_tps": case.kv_transfer.tail_replay_tps,
              "kv_capacity_tokens": profile.kv_capacity_tokens,
              "F": reference["prefill_tokens_per_s"], "G": reference["decode_tokens_per_s"],
              "reference_request_tokens": [2048, 32], "reference_rps": 1 / reference["total_s"],
              "service_reference": {"profile_sha256": reference["sha256"], "request_tokens": [2048, 32],
                  "scope": "Measured loaded-replay offered-rate reference; phase costs normalize request demand, not context-dependent saturation or GPU utilization. Long-context resident mixtures are a declared transfer. Historical decode peaks change concurrency and include >300W samples; they are not a compatible replacement."},
              "calibration_contexts": list(loaded.CONTEXTS), "load_range": [0., .975]}
    singleton = replay_seconds(loaded.CONTEXTS, result)
    if not np.isclose(singleton.sum(), physics["endpoint_work_s"]["replay"], rtol=0, atol=1e-12):
        raise ValueError("singleton proxies changed the fitted width-eight work")

    def fit(rows):
        intercept, beta = loaded._fits(rows, physics, "commit_s")["replay"]
        kappa = (intercept * singleton.sum() - singleton.max()) / (singleton.sum() - singleton.max())
        if not np.all(np.isfinite([beta, kappa])) or beta < 0 or not 0 <= kappa <= 1:
            raise ValueError("joint timing fit violates positive, sublinear batch service")
        return {"beta": float(beta), "kappa": float(kappa)}

    rng = np.random.default_rng(1)
    repeats = sorted({row["repeat"] for row in training})
    if repeats != list(range(10)):
        raise ValueError("loaded timing requires the ten frozen repeat groups")
    result["timing"] = [fit(training)] + [fit([row for repeat in rng.choice(repeats, len(repeats))
                                              for row in training if row["repeat"] == repeat]) for _ in range(draws)]
    packing = _packing(result, draws)
    kv_completion = _kv_completion(result, draws)
    result["regional_components"] = _regional_components(result)
    result["resident_interference"] = _resident_interference(result)
    timing = result["timing"][0]
    batch = singleton.max() + timing["kappa"] * (singleton.sum() - singleton.max())
    observed, predicted = [], []
    for row in validation:
        if row["method"] == "replay":
            observed.append(row["commit_s"])
            predicted.append(physics["route_bytes"]["replay"] / (row["bandwidth_mbps"] * 125_000)
                             + batch * np.exp(timing["beta"] * row["rho"]) + physics["switch_s"])
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    check = {"episodes": len(observed), "p90_relative_error": float(np.quantile(abs(predicted - observed) / observed, .9)),
             "false_feasible_25s": int(np.sum((predicted <= 25) & (observed > 25)))}
    if check["episodes"] != 220 or check["p90_relative_error"] > .05 or check["false_feasible_25s"]:
        raise ValueError("calibrated batch formula failed the replay holdout")
    with POWER_POINTS.open(newline="") as handle:
        points = [row for row in csv.DictReader(handle) if row["window"] == "pre"]
    groups = {repeat: [float(row["power_w"]) for row in points if int(row["repeat"]) == repeat]
              for repeat in range(3)}
    if len(points) != 42 or any(len(values) != 14 for values in groups.values()):
        raise ValueError("coding power requires the frozen three groups of fourteen pre windows")
    result["active_w"] = float(np.median([float(row["power_w"]) for row in points]))
    result["idle_w"] = json.loads(POWER_PROFILE.read_text())["cases"]["central"]["phase_power"]["p0_w"]
    if not 0 < result["idle_w"] < result["active_w"] <= 300:
        raise ValueError("invalid measured 300 W A100 source power anchors")
    rng = np.random.default_rng(2)
    result["power_draws_w"] = [float(np.median([w for repeat in rng.choice(3, 3) for w in groups[repeat]])
                                     - result["idle_w"]) for _ in range(200)]
    transitions = json.loads(TRANSITIONS.read_text())
    if (not transitions["campaign_pass"] or len(transitions["decisions"]) != 9
            or any(not row["pass"] or not all(row["checks"].values()) for row in transitions["decisions"])):
        raise ValueError("the recorded nine safe serving transitions did not pass")
    result["evidence"] = {
        "timing_fit_episodes": provenance["training_episodes"], "replay_holdout": check,
        "batch_equation": "exp(beta*rho) * (max(singleton_s) + kappa*(sum(singleton_s)-max(singleton_s))); empty=0",
        "coding_batch_equation": "exp(beta*rho)*(sum(kappa_i*t_i)+max((1-kappa_i)*t_i)); interpolate four packing knots; ANY context>16384 serializes entire batch; empty=0",
        "coding_packing": packing,
        "kv_completion": kv_completion, "recorded_policy_checks": _policy_checks(result),
        "idle_width8_endpoint_s": float(batch), "timing_seed": 1, "power_seed": 2,
        "power_windows": 42, "power_repeat_groups": 3, "power_bootstrap_draws": 200,
        "power_scope": "Direct coding pre-window median and separate awake-idle anchor; partial/fleet/workload transfer assumed; fitted partial-power curve unused; idle held fixed",
        "timing_scope": "Fixed width-eight 2048-14336-token pack; 2048/32 prefill-heavy resident requests; offered-rate normalization, not GPU utilization; wall endpoint time, not intrinsic GPU work",
        "timing_transfer": "Coding uses independently calibrated width-eight packing; intermediate widths, context interpolation, the 16384-token switching boundary and loaded coding resident mixtures remain transfers; beta comes from the fixed loaded pack; do not also divide service by 1-rho",
        "transition_passes": 9, "transition_scope": transitions["claim_scope"],
        "transition_recipes": transitions["recipes"], "transition_targets": transitions["targets"],
        "transition_limit": "Three discrete eager-A100/4K recipes at combined W=.50; no generic SLO cap or transfer to new mixtures",
        "loaded_validation": model["width8_relative_factor_validation"]["replay"],
        "historical_transfer_checks": _transfer_checks(result)}
    result["resident_service"] = _resident_service()
    result["sources"] = {**{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS},
                         **result["resident_service"]["sources"],
                         **result["regional_components"]["sources"], **result["resident_interference"]["sources"]}
    return result
