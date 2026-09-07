"""Frozen A100 endpoint timing and direct coding-power anchors for pooled shed."""

from __future__ import annotations

import csv
import hashlib
import json
from functools import cache
from pathlib import Path

import numpy as np

import loaded_service_model as loaded
from profiles import ModelProfile

ROOT = Path(__file__).resolve().parent
POWER_POINTS = ROOT / "outputs/azure-compact-calibration-20260813/pack-power-gated-points.csv"
POWER_PROFILE = ROOT / "outputs/azure-compact-calibration-20260813/gpt_oss_20b_a100_tp1_azure_300w_pack_power_gated.json"
TRANSITIONS = ROOT / "outputs/service-admission-transition-a100-20260816/summary.json"
CROSSOVER = ROOT / "outputs/policy-hardware-crossover-20260730"
LONG_PACKS = ROOT / "outputs/policy-hardware-width8-frontier-20260730"
PACKING = ROOT / "outputs/policy-hardware-width8-packing-20260730"
PATHS = (Path(__file__), ROOT / "loaded_service_model.py", ROOT / "profiles.py",
         loaded.PROFILE, loaded.OUT, POWER_POINTS, POWER_PROFILE, TRANSITIONS,
         loaded.TRAINING, *loaded.BLOCKS, loaded.STANDALONE,
         CROSSOVER / "migrations.csv", CROSSOVER / "run_metadata.json",
         LONG_PACKS / "migrations.csv", LONG_PACKS / "scenarios.csv", LONG_PACKS / "run_metadata.json",
         PACKING / "policy_episodes.csv", PACKING / "plan.json", PACKING / "run_metadata.json",
         *(p.parent / "live_plan.json" for p in (loaded.TRAINING, *loaded.BLOCKS, loaded.STANDALONE)))


def replay_seconds(contexts, calibration):
    """Match the historical profile's conservative singleton rate, including its tail."""
    context = np.asarray(contexts, dtype=float)
    if not np.all(np.isfinite(context)) or np.any(context <= 0):
        raise ValueError("replay contexts must be finite and positive")
    x, y = np.asarray(calibration["replay_context_tokens"]), np.asarray(calibration["replay_tps"])
    rate = np.interp(context, x, y, left=y.min(), right=y.min())
    return context / rate + calibration["replay_completion_s"]


def _read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    errors = abs(predicted - observed) / observed
    return {"episodes": len(observed), "median_relative_error": float(np.median(errors)),
            "p90_relative_error": float(np.quantile(errors, .9)),
            "false_feasible_25s": int(np.sum((predicted <= 25) & (observed > 25)))}


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
    x, y = case.replay.by_concurrency[1]
    result = {"schema": "queue-haul-pool-calibration-v1", "model": profile.model,
              "replay_context_tokens": x.tolist(), "replay_tps": y.tolist(),
              "replay_completion_s": case.replay_completion_s, "switch_s": case.switch_s,
              "kv_block_tokens": case.kv_transfer.block_tokens, "kv_block_bytes": case.kv_transfer.block_bytes,
              "kv_capacity_tokens": profile.kv_capacity_tokens,
              "F": reference["prefill_tokens_per_s"], "G": reference["decode_tokens_per_s"],
              "reference_request_tokens": [2048, 32], "reference_rps": 1 / reference["total_s"],
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
    result["sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS}
    return result
