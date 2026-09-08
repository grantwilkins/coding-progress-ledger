"""Fit destination-load migration factors from paired fixed-width A100 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from profiles import ModelProfile


ROOT = Path(__file__).parent
PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
TRAINING = ROOT / "outputs/capacity-load-publication-20260807/live_capacity.csv"
BLOCKS = tuple(ROOT / f"outputs/capacity-full-drain-block{i}-20260807/full_drain_capacity.csv"
               for i in range(2))
STANDALONE = ROOT / "outputs/capacity-full-drain-10000-20260807/full_drain_capacity.csv"
OUT = ROOT / "outputs/loaded-service-model-20260815/model.json"
CONTEXTS = (2048, 4096, 4096, 8192, 8192, 12288, 12288, 14336)
PROFILE_SHA256 = "99ec3d2e3099f01103e7a0609a9d99beabfe7ec86eeb372d65f5829090658ab8"
METHODS = ("replay", "kv_transfer")
RHO_GRID = (0, .25, .5, .65, .75, .8, .85, .875, .9, .925, .95, .975)
EQUATION = "endpoint_work(rho)=endpoint_work(0)*exp(beta_method*rho)"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) \
            and all(_close(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) \
            and all(_close(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12,
        )
    return actual == expected


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _row(row: dict, source: Path) -> dict:
    method = {"replay_only": "replay", "kv_only": "kv_transfer"}[row["policy"]]
    return {
        "method": method, "rho": float(row["offered_rho"]),
        "load_fraction": float(row["load_fraction"]),
        "commit_s": float(row["last_route_commit_s"]),
        "resume_s": float(row["last_first_token_s"]),
        "bandwidth_mbps": float(row["configured_goodput_mbps"]),
        "repeat": int(row["repeat"]), "scenario_id": row["scenario_id"],
        "trace_id": row["trace_id"], "source": str(source.relative_to(ROOT)),
        "planned_sessions": int(row["planned_sessions"]),
        "credited_sessions": int(row["credited_sessions"]),
    }


def _initial_queue(row: dict) -> float:
    if row["queue_at_start"]:
        return float(row["queue_at_start"])
    if float(row["load_fraction"]) != 0:
        raise ValueError("missing initial queue outside the zero-load cell")
    return 0.0


def load_evidence() -> tuple[list[dict], list[dict], dict]:
    plan_paths = tuple(path.parent / "live_plan.json"
                       for path in (TRAINING, *BLOCKS, STANDALONE))
    plans = [json.loads(path.read_text()) for path in plan_paths]
    if _hash(PROFILE) != PROFILE_SHA256 or any(
            plan["profile"]["sha256"] != PROFILE_SHA256 for plan in plans):
        raise ValueError("loaded-service evidence profile changed")
    raw_training = _read(TRAINING)
    raw_validation = [(row, path) for path in BLOCKS for row in _read(path)]
    raw_validation += [(row, STANDALONE) for row in _read(STANDALONE)
                       if int(row["repeat"]) >= 4]
    if any(row["right_censored"] != "False" or _initial_queue(row)
           for row in raw_training) or any(
               row["right_censored"] != "False" or _initial_queue(row)
               for row, _ in raw_validation):
        raise ValueError("loaded-service evidence is censored or initially queued")
    training = [_row(row, TRAINING) for row in raw_training
                if row["policy"] in ("replay_only", "kv_only")
                and int(row["planned_sessions"]) == 8
                and float(row["load_fraction"]) <= .875]
    validation = [_row(row, path) for row, path in raw_validation]
    if len(training) != 160 or len(validation) != 440 \
            or len({row["scenario_id"] for row in validation}) != 440 \
            or {row["scenario_id"] for row in training} \
            & {row["scenario_id"] for row in validation} \
            or {row["trace_id"] for row in training} \
            & {row["trace_id"] for row in validation} \
            or any(row["planned_sessions"] != 8 for row in training + validation) \
            or any(row["credited_sessions"] != 8 for row in training):
        raise ValueError("loaded-service evidence split changed")
    hashes = {str(path.relative_to(ROOT)): _hash(path)
              for path in (*plan_paths, TRAINING, *BLOCKS, STANDALONE)}
    return training, validation, {
        "training_episodes": len(training),
        "validation_episodes": len(validation),
        "profile": str(PROFILE.relative_to(ROOT)),
        "profile_sha256": PROFILE_SHA256, "input_sha256": hashes,
        "validation_split": "full-drain blocks 0-1 plus standalone 10G repeats 4-9",
    }


def fit_log_cells(rows: list[dict], metric: str, route_s: float,
                  endpoint_work_s: float, switch_s: float) -> tuple[float, float]:
    cells = sorted({row.get("load_fraction", row["rho"]) for row in rows})
    points = []
    for cell in cells:
        selected = [row for row in rows
                    if row.get("load_fraction", row["rho"]) == cell]
        rho = float(np.median([row["rho"] for row in selected]))
        residual = float(np.median([row[metric] for row in selected])) \
            - route_s - switch_s
        if residual <= 0 or endpoint_work_s <= 0:
            raise ValueError("load fit requires positive endpoint residual work")
        points.append((rho, math.log(residual / endpoint_work_s)))
    matrix = np.column_stack((np.ones(len(points)), [row[0] for row in points]))
    intercept, slope = np.linalg.lstsq(
        matrix, [row[1] for row in points], rcond=None,
    )[0]
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("loaded-service fit is nonfinite")
    return math.exp(float(intercept)), float(slope)


def _physics(profile: ModelProfile) -> dict:
    case = profile.case()
    route_bytes = {
        "replay": sum(2 * context for context in CONTEXTS),
        "kv_transfer": sum(case.kv_transfer.sealed_bytes(context)
                           for context in CONTEXTS),
    }
    endpoint = {
        "replay": sum(context / case.replay.conservative_rate(context, 1)
                      + case.replay_completion_s for context in CONTEXTS),
        "kv_transfer": sum(
            case.kv_transfer.sealed_bytes(context)
            / case.kv_transfer.destination_bytes_per_s
            + case.kv_transfer.initial_completion_s for context in CONTEXTS
        ),
    }
    return {"route_bytes": route_bytes, "endpoint_work_s": endpoint,
            "switch_s": len(CONTEXTS) * case.switch_s}


def _fits(rows: list[dict], physics: dict, metric: str) -> dict:
    output = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        bandwidths = {row["bandwidth_mbps"] for row in selected}
        if len(bandwidths) != 1:
            raise ValueError("training fit requires one fixed bandwidth")
        route = physics["route_bytes"][method] / (bandwidths.pop() * 125_000)
        output[method] = fit_log_cells(
            selected, metric, route, physics["endpoint_work_s"][method],
            physics["switch_s"],
        )
    return output


def _bootstrap(rows: list[dict], physics: dict, samples: int, seed: int) -> dict:
    repeats = sorted({row["repeat"] for row in rows})
    rng, values = np.random.default_rng(seed), {method: [] for method in METHODS}
    for _ in range(samples):
        draw = rng.choice(repeats, len(repeats), replace=True)
        sampled = [row for repeat in draw for row in rows if row["repeat"] == repeat]
        fitted = _fits(sampled, physics, "commit_s")
        for method in METHODS:
            values[method].append(fitted[method][1])
    return {method: {
        "p05": float(np.quantile(value, .05)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, .95)),
        "positive_fraction": float(np.mean(np.asarray(value) > 0)),
    } for method, value in values.items()}


def _validate(rows: list[dict], physics: dict, fits: dict, metric: str) -> dict:
    output = {}
    for method in METHODS:
        errors, absolute, false_feasible, false_infeasible = [], [], 0, 0
        kappa, beta = fits[method]
        for row in (item for item in rows if item["method"] == method):
            route = physics["route_bytes"][method] / (
                row["bandwidth_mbps"] * 125_000)
            predicted = route + kappa * math.exp(beta * row["rho"]) \
                * physics["endpoint_work_s"][method] + physics["switch_s"]
            actual = row[metric]
            errors.append(abs(predicted - actual) / actual)
            absolute.append(abs(predicted - actual))
            false_feasible += predicted <= 25 < actual
            false_infeasible += actual <= 25 < predicted
        output[method] = {
            "episodes": len(errors),
            "median_absolute_percentage_error": float(np.median(errors)),
            "p90_absolute_percentage_error": float(np.quantile(errors, .9)),
            "p90_absolute_error_s": float(np.quantile(absolute, .9)),
            "false_feasible_25s": false_feasible,
            "false_infeasible_25s": false_infeasible,
        }
    return output


def fit_evidence(samples: int = 1000, seed: int = 1) -> tuple[dict, dict]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    training, validation, provenance = load_evidence()
    physics = _physics(ModelProfile.load(PROFILE))
    commit, resume = (_fits(training, physics, metric)
                      for metric in ("commit_s", "resume_s"))
    selected = {"replay": commit["replay"][1], "kv_transfer": 0.0}
    selected_fit = {method: (commit[method][0], selected[method])
                    for method in METHODS}
    model = {
        "schema": "queue-haul-loaded-service-v1",
        "equation": EQUATION,
        "selected_commit_log_slope_per_rho": selected,
        "fitted_commit_log_slope_per_rho": {
            method: commit[method][1] for method in METHODS},
        "width8_intercept_diagnostic": {
            method: commit[method][0] for method in METHODS},
        "resume_log_slope_per_rho_diagnostic": {
            method: resume[method][1] for method in METHODS},
        "slowdown_at_rho_0": {method: 1.0 for method in METHODS},
        "slowdown_at_rho_0_95": {
            method: math.exp(selected[method] * .95) for method in METHODS},
        "rho_grid": list(RHO_GRID),
        "slowdown": {method: [math.exp(selected[method] * rho)
                               for rho in RHO_GRID] for method in METHODS},
        "fit_context_tokens": [min(CONTEXTS), max(CONTEXTS)],
        "training_bandwidth_mbps": [10000, 10000],
        "validation_bandwidth_mbps": [1000, 10000],
        "bootstrap": _bootstrap(training, physics, samples, seed),
        "bootstrap_samples": samples, "bootstrap_seed": seed,
        "width8_relative_factor_validation": _validate(
            validation, physics, selected_fit, "commit_s"),
        "resume_validation_diagnostic": _validate(
            validation, physics, resume, "resume_s"),
        "physics": physics,
        "limitations": [
            "the width-eight intercept is diagnostic and is not transported to regional concurrency-one timing",
            "the load shape is prefill-heavy; no separate decode-load coefficient is identified",
            "KV load slope is zero centrally because its paired bootstrap spans zero",
            "the fitted KV intercept uses a serial-tail diagnostic equation and is not deployed",
            "loaded context transport outside the fixed 2,048-14,336-token pack is a sensitivity",
            "resume TTFT is diagnostic and is not charged as migration resource work",
        ],
    }
    validate_model({**model, "provenance": provenance})
    return model, provenance


def validate_model(value: dict) -> dict:
    betas = value.get("selected_commit_log_slope_per_rho", {})
    grid, slowdown = value.get("rho_grid"), value.get("slowdown", {})
    bootstrap = value.get("bootstrap", {})
    if value.get("schema") != "queue-haul-loaded-service-v1" \
            or value.get("equation") != EQUATION or grid != list(RHO_GRID) \
            or set(betas) != set(METHODS) or set(slowdown) != set(METHODS) \
            or value.get("fit_context_tokens") != [min(CONTEXTS), max(CONTEXTS)] \
            or value.get("training_bandwidth_mbps") != [10000, 10000] \
            or value.get("validation_bandwidth_mbps") != [1000, 10000] \
            or any(not math.isfinite(betas[method]) or betas[method] < 0
                   for method in METHODS) \
            or betas["kv_transfer"] != 0 \
            or any(len(slowdown[method]) != len(grid) or not np.allclose(
                slowdown[method], np.exp(betas[method] * np.asarray(grid)),
                rtol=0, atol=1e-12,
            ) for method in METHODS) \
            or bootstrap.get("replay", {}).get("p05", 0) <= 0 \
            or not bootstrap.get("kv_transfer", {}).get("p05", 1) <= 0 \
            <= bootstrap.get("kv_transfer", {}).get("p95", -1):
        raise ValueError("invalid loaded-service model")
    validation = value.get("width8_relative_factor_validation", {})
    if set(validation) != set(METHODS) or any(
            validation[method]["false_feasible_25s"]
            or validation[method]["p90_absolute_percentage_error"]
            > ({"replay": .1, "kv_transfer": .15}[method])
            for method in METHODS):
        raise ValueError("loaded-service validation gate failed")
    training, retained, expected = load_evidence()
    physics = _physics(ModelProfile.load(PROFILE))
    commit = _fits(training, physics, "commit_s")
    selected = {"replay": commit["replay"][1], "kv_transfer": 0.0}
    samples, seed = value.get("bootstrap_samples"), value.get("bootstrap_seed")
    selected_fit = {method: (commit[method][0], selected[method])
                    for method in METHODS}
    if not isinstance(samples, int) or samples < 1 or not isinstance(seed, int) \
            or not _close(betas, selected) \
            or not _close(value.get("physics"), physics) \
            or not _close(bootstrap, _bootstrap(training, physics, samples, seed)) \
            or not _close(validation, _validate(
                retained, physics, selected_fit, "commit_s")) \
            or not _close(value.get("slowdown_at_rho_0"), {
                method: 1.0 for method in METHODS} \
            ) or not _close(value.get("slowdown_at_rho_0_95"), {
                method: math.exp(selected[method] * .95) for method in METHODS}) \
            or value.get("provenance") != expected:
        raise ValueError("loaded-service provenance changed")
    return value


def historical_execution_check(value):
    """Execute recorded actions; local training adapters do not alter fleet calibration."""
    from types import SimpleNamespace
    from pool_shed_calibration import PACKING, LONG_PACKS, kv_state, replay_seconds, _metrics
    from pool_shed_execution import execute_pooled

    def execute(context, replay, bandwidth, load, tail=None, packing=True):
        context, replay = np.asarray(context, float), np.asarray(replay, float)[None, :]
        endpoint = np.full(2, bandwidth * 125_000)
        timing = {**value["timing"][0], "regional_replay_factor": [1., 1.],
                  "regional_kv_bytes_per_s": endpoint.tolist()}
        if tail is not None:
            timing["kv_batch_completion_s"] = tail
            if not replay.any():
                timing["beta"] = 0.
        metadata = {"turn_sequences": [[] for _ in context], "source_session_rps": 0.,
                    "batch_context_limit": value["batch_context_limit"], "protect_resident": False}
        if packing:
            metadata["packing_context_tokens"] = value["packing_context_tokens"]
        fleet = SimpleNamespace(count=np.ones(len(context)), context=context, demand=np.zeros(len(context)),
            gain=np.ones(len(context)) / len(context), memory_tokens=context, baseline_kv=0., kv_capacity=1e12,
            gpus=1, nodes=1, t1=replay_seconds(context, value), log=2 * context,
            kv=kv_state(context, value)[0], metadata=metadata)
        table = SimpleNamespace(fleet=fleet, replay=replay, kv=1 - replay, route=np.array([0]),
            deadline=300., endpoint=endpoint, budgets=np.r_[endpoint, endpoint.sum()], load=load)
        result = execute_pooled(table, np.ones(1), timing, value, chunks=1)
        if result["completed_sessions"] != len(context):
            raise RuntimeError("historical execution failed to complete recorded actions")
        return result["last_completion_s"]

    def summarize(rows, key, limits):
        metrics = {group: _metrics([r["observed_s"] for r in rows if r[key] == group],
                                  [r["predicted_s"] for r in rows if r[key] == group]) for group in limits}
        return {"metrics": metrics, "predictions": rows, "p90_relative_error_limits": limits,
                "gate_pass": all(not m["false_feasible_25s"] and m["p90_relative_error"] <= limits[g]
                                 for g, m in metrics.items())}

    training, heldout, provenance = load_evidence()
    physics = _physics(ModelProfile.load(PROFILE))
    loaded_tail = physics["endpoint_work_s"]["kv_transfer"] * _fits(training, physics, "commit_s")["kv_transfer"][0]
    report = {}
    for mode, tail in (("loaded_transferred_tail", None), ("loaded_matched_runtime", loaded_tail)):
        rows = [{"scenario_id": r["scenario_id"], "method": r["method"], "observed_s": r["commit_s"],
                 "predicted_s": execute(CONTEXTS, [r["method"] == "replay"] * 8, r["bandwidth_mbps"],
                                        r["rho"], tail, packing=False)} for r in heldout]
        report[mode] = summarize(rows, "method", {"replay": .05, "kv_transfer": .15})
    report["loaded_local_tail_s"] = loaded_tail
    scenarios = {r["scenario_id"]: r for r in json.loads((PACKING / "plan.json").read_text())["scenarios"]}
    moves, groups = {}, {}
    for r in _read(PACKING / "policy_migrations.csv"):
        moves.setdefault(r["scenario_id"], []).append(r)
    for r in _read(PACKING / "policy_episodes.csv"):
        groups.setdefault((r["policy"], r["condition"]), []).append(r)
    if len(groups) != 160 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("historical policy execution requires 160 three-repeat conditions")
    training = [r for rows in groups.values() for r in sorted(rows, key=lambda r: int(r["episode"]))[:2]]
    heldout = [max(rows, key=lambda r: int(r["episode"])) for rows in groups.values()]
    residuals = []
    for r in training:
        if r["policy"] == "kv_only":
            context = [int(m["context_tokens"]) for m in moves[r["scenario_id"]]]
            state, partial = kv_state(context, value)
            residuals.append(float(r["commit_100_s"]) - state.sum() / (scenarios[r["scenario_id"]]["bandwidth_mbps"] * 125_000)
                             - partial.sum() / value["kv_tail_replay_tps"] - value["switch_s"])
    policy_tail = float(np.median(residuals))
    if len(residuals) != 80 or not np.isfinite(policy_tail) or policy_tail <= 0:
        raise ValueError("historical policy tail requires 80 valid pure-KV training episodes")
    limits = {p: m["p90_relative_error"] for p, m in value["evidence"]["recorded_policy_checks"]["heldout_third_repeat"].items()}
    for mode, tail in (("policy_transferred_tail", None), ("policy_matched_runtime", policy_tail)):
        rows = []
        for r in heldout:
            selected = moves[r["scenario_id"]]
            rows.append({"scenario_id": r["scenario_id"], "policy": r["policy"], "observed_s": float(r["commit_100_s"]),
                "predicted_s": execute([int(m["context_tokens"]) for m in selected], [m["method"] == "replay" for m in selected],
                    scenarios[r["scenario_id"]]["bandwidth_mbps"], 0., tail)})
        report[mode] = summarize(rows, "policy", limits)
    report["policy_local_tail_s"] = policy_tail
    moves, rows = {}, []
    for r in _read(LONG_PACKS / "migrations.csv"):
        moves.setdefault(r["scenario_id"], []).append(r)
    for r in _read(LONG_PACKS / "scenarios.csv"):
        if (r["kind"], r["method"], r["activity"], int(r["concurrency"])) == ("migration", "replay", "none", 8):
            rows.append({"scenario_id": r["scenario_id"], "method": "replay", "observed_s": float(r["migration_s"]),
                "predicted_s": execute([int(m["measured_prompt_tokens"]) for m in moves[r["scenario_id"]]],
                                       [True] * 8, float(r["bandwidth_mbps"]), 0.)})
    if len(rows) != 72:
        raise ValueError("historical long-context execution requires 72 recorded batches")
    report["long_context"] = summarize(rows, "method", {"replay": .20})
    report["gate_pass"] = all(report[k]["gate_pass"] for k in ("loaded_matched_runtime", "policy_matched_runtime", "long_context"))
    return {**report, "loaded_provenance": provenance,
        "scope": "Recorded one-GPU actions, original unprotected resident offered load, frozen source; local endpoint and training-only tail adapters. Loaded/long gates are existing gates; policy limits preserve analytic-reference error. Transferred-tail failures remain visible. This validates engine progress, not protected serving, moving trajectories, fleet SLOs or new policy decisions."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    model, provenance = fit_evidence(args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    value = validate_model({**model, "provenance": provenance})
    args.out.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
