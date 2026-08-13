"""Fit and freeze the Azure phase-aware source-power model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

import power_rate_sweep
from profiles import ModelProfile


MIXTURES = ("prefill", "prefill75", "mixed", "decode75", "decode")
LOADS = (.1, .25, .45, .65, .8, 1.0)


def campaign_plan(repeats: int = 3, seed: int = 1) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    fractions = dict(zip(MIXTURES, (1, .75, .5, .25, 0)))
    cells = [{"mixture": mixture, "prefill_fraction": fractions[mixture],
              "target_service_load": load, "repeat": repeat,
              "warmup_s": 10, "measurement_s": 30, "power_interval_s": .1}
             for repeat in range(repeats) for mixture in MIXTURES for load in LOADS]
    random.Random(seed).shuffle(cells)
    return {"schema": "queue-haul-phase-power-plan-v1", "cells": cells,
            "repeats": repeats, "adaptive_repeats": 2, "seed": seed,
            "gates": {"grouped_cv_rmse_w": 5, "within_5w_fraction": .8}}


def _metrics(host: str, port: int) -> tuple[float, float]:
    text = urlopen(f"http://{host}:{port}/metrics", timeout=10).read().decode()
    def total(name):
        values = [float(match.group(1)) for match in re.finditer(
            rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", text, re.MULTILINE)]
        if not values:
            raise RuntimeError(f"missing {name} counter")
        return sum(values)
    return total("vllm:prompt_tokens_total"), total("vllm:generation_tokens_total")


def _shape(fraction: float, load: float, F: float, G: float) -> tuple[str, int, float, int]:
    f, g = fraction * load * F, (1 - fraction) * load * G
    if fraction == 0:
        prompt_tokens, output_tokens = 1, 4096
        return "x", output_tokens, g / output_tokens, max(1, math.ceil(g * 40 / output_tokens))
    prompt_tokens = 4096 if fraction == 1 else 2048
    output_tokens = 1 if fraction == 1 else max(1, round(prompt_tokens * g / f))
    return "x " * prompt_tokens, output_tokens, f / prompt_tokens, 0


def run_cell(host: str, port: int, root: Path, cell: dict,
             F: float, G: float, workers: int = 512) -> dict:
    fraction, load = float(cell["prefill_fraction"]), float(cell["target_service_load"])
    prompt, output_tokens, rate, batch = _shape(fraction, load, F, G)
    warmup, window = float(cell["warmup_s"]), float(cell["measurement_s"])
    label = f"{cell['mixture']}-l{load:g}-r{cell['repeat']}".replace(".", "p")
    stop, path = threading.Event(), root / f"power-{label}.csv"
    sampler = threading.Thread(target=power_rate_sweep.power,
                               args=(path, stop, float(cell["power_interval_s"])))
    sampler.start(); started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            count = batch or math.ceil(rate * (warmup + window))
            def scheduled_request(request_id):
                if not batch:
                    delay = started + request_id / rate - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                return power_rate_sweep.request(
                    f"http://{host}:{port}/v1/completions", prompt,
                    output_tokens, f"{label}:{request_id}")
            futures = [pool.submit(scheduled_request, request_id)
                       for request_id in range(count)]
            remaining = started + warmup - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            metric_start, start_ns = _metrics(host, port), time.monotonic_ns()
            time.sleep(window)
            metric_end, end_ns = _metrics(host, port), time.monotonic_ns()
            for future in futures:
                future.result()
    finally:
        stop.set(); sampler.join()
    with path.open(newline="") as handle:
        watts = [float(row["power_w"]) for row in csv.DictReader(handle)
                 if start_ns <= int(row["monotonic_ns"]) < end_ns]
    if len(watts) < window / float(cell["power_interval_s"]) * .8:
        raise RuntimeError("insufficient power samples")
    return {"mixture": cell["mixture"], "repeat": cell["repeat"],
            "target_service_load": load, "f_tps": (metric_end[0] - metric_start[0]) / window,
            "g_tps": (metric_end[1] - metric_start[1]) / window,
            "power_mean_w": float(np.mean(watts)), "power_samples": len(watts),
            "start_ns": start_ns, "end_ns": end_ns, "power_path": str(path)}


def run_plan(plan_path: Path, profile_path: Path, out: Path,
             host: str = "127.0.0.1", port: int = 8100,
             resume: bool = False) -> list[dict]:
    plan, profile = json.loads(plan_path.read_text()), ModelProfile.load(profile_path)
    if plan.get("schema") not in {"queue-haul-phase-power-plan-v1",
                                  "queue-haul-phase-power-adaptive-plan-v1"}:
        raise ValueError("invalid phase power plan")
    power_rate_sweep.validate_gpu("NVIDIA A100 80GB PCIe", 300)
    out.mkdir(parents=True, exist_ok=resume)
    measurement_path = out / "measurements.csv"
    rows = []
    if resume and measurement_path.exists():
        with measurement_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {(row["mixture"], float(row["target_service_load"]), int(row["repeat"]))
                 for row in rows}
    case = profile.case()
    for cell in plan["cells"]:
        key = cell["mixture"], float(cell["target_service_load"]), int(cell["repeat"])
        if key in completed:
            continue
        rows.append(run_cell(host, port, out, cell, case.F, case.G))
        with measurement_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(rows[-1]), flush=True)
    return rows


def _predict(parameters, f, g, p0):
    delta, a, b = parameters
    z = a * f + b * g
    return p0 + delta * z / (1 + z)


def _fit(rows: list[dict], p0: float):
    f = np.asarray([float(row["f_tps"]) for row in rows])
    g = np.asarray([float(row["g_tps"]) for row in rows])
    watts = np.asarray([float(row["power_mean_w"]) for row in rows])
    if len(rows) < 8 or np.linalg.matrix_rank(np.column_stack((f, g))) != 2:
        raise ValueError("phase power fit needs identifiable prefill/decode coverage")
    scale_f, scale_g = max(f.max(), 1), max(g.max(), 1)
    result = least_squares(
        lambda x: _predict(x, f, g, p0) - watts,
        (max(watts.max() - p0, 1), 1 / scale_f, 1 / scale_g),
        bounds=((1e-9, 1e-12, 1e-12), (np.inf, np.inf, np.inf)),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def _hull(rows: list[dict]) -> list[list[float]]:
    points = np.unique(np.asarray([[0, 0]] + [
        [float(row["f_tps"]), float(row["g_tps"])] for row in rows
    ]), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) != 2:
        raise ValueError("valid hull needs non-collinear phase points")
    return points[ConvexHull(points).vertices].tolist()


def fit(rows: list[dict], idle_power_w: float, bootstrap_samples: int = 200,
        seed: int = 1) -> dict:
    required = {"mixture", "repeat", "f_tps", "g_tps", "power_mean_w"}
    if idle_power_w <= 0 or bootstrap_samples < 1 or not rows \
            or any(not required <= row.keys() for row in rows):
        raise ValueError("invalid phase power measurements")
    groups = sorted({str(row["mixture"]) for row in rows})
    if len(groups) < 3 or any(sum(str(row["mixture"]) == group for row in rows) < 2
                              for group in groups):
        raise ValueError("grouped validation needs three mixtures with repeats")
    parameters = _fit(rows, idle_power_w)
    errors, heldout = [], []
    for group in groups:
        train = [row for row in rows if str(row["mixture"]) != group]
        test = [row for row in rows if str(row["mixture"]) == group]
        fitted = _fit(train, idle_power_w)
        for row in test:
            error = float(_predict(fitted, float(row["f_tps"]), float(row["g_tps"]),
                                   idle_power_w) - float(row["power_mean_w"]))
            errors.append(error)
            heldout.append({"mixture": group,
                            "target_service_load": row.get("target_service_load"),
                            "error_w": error})
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    within = float(np.mean(np.abs(errors) <= 5))
    group_bias = {group: float(np.mean([row["error_w"] for row in heldout
                                        if row["mixture"] == group]))
                  for group in groups}
    rng, bootstrap = np.random.default_rng(seed), []
    grouped = {group: [row for row in rows if str(row["mixture"]) == group]
               for group in groups}
    for _ in range(bootstrap_samples):
        sample = [items[i] for items in grouped.values()
                  for i in rng.integers(0, len(items), len(items))]
        delta, a, b = _fit(sample, idle_power_w)
        bootstrap.append([idle_power_w, float(delta), float(a), float(b)])
    delta, a, b = map(float, parameters)
    return {
        "schema": "queue-haul-phase-power-fit-v1",
        "p0_w": idle_power_w, "delta_w": delta,
        "a_s_per_prefill_token": a, "b_s_per_decode_token": b,
        "valid_hull": _hull(rows), "grouped_cv_rmse_w": rmse,
        "within_5w_fraction": within, "bootstrap": bootstrap,
        "group_bias_w": group_bias,
        "gate_passed": rmse <= 5 and within >= .8
        and max(map(abs, group_bias.values())) <= 5,
        "groups": groups, "measurements": len(rows), "heldout": heldout,
    }


def adaptive_plan(plan: dict, summary: dict) -> dict:
    if summary.get("gate_passed") is not False:
        raise ValueError("adaptive repeats require a failed fit gate")
    ranked = sorted(summary["heldout"], key=lambda row: abs(float(row["error_w"])),
                    reverse=True)
    keys = []
    for row in ranked:
        key = row["mixture"], row.get("target_service_load")
        if key not in keys:
            keys.append(key)
        if len(keys) == 6:
            break
    templates = {(row["mixture"], row["target_service_load"]): row
                 for row in plan["cells"]}
    cells = [{**templates[key], "repeat": repeat}
             for key in keys for repeat in range(plan["repeats"], plan["repeats"] + 2)]
    random.Random(plan["seed"] + 1).shuffle(cells)
    return {**plan, "schema": "queue-haul-phase-power-adaptive-plan-v1",
            "parent_schema": plan["schema"], "cells": cells, "repeats": 2}


def freeze_profile(base_path: Path, measurements_path: Path, out: Path,
                   idle_power_w: float, bootstrap_samples: int = 200) -> dict:
    with measurements_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = fit(rows, idle_power_w, bootstrap_samples)
    digest = hashlib.sha256(measurements_path.read_bytes()).hexdigest()
    phase = {key: result[key] for key in (
        "p0_w", "delta_w", "a_s_per_prefill_token",
        "b_s_per_decode_token", "valid_hull", "grouped_cv_rmse_w",
        "within_5w_fraction", "bootstrap")}
    phase["provenance_sha256"] = digest
    profile = json.loads(base_path.read_text())
    central = profile["cases"]["central"]
    max_load = max(phase["a_s_per_prefill_token"] * float(row["f_tps"])
                   + phase["b_s_per_decode_token"] * float(row["g_tps"])
                   for row in rows)
    central["phase_power"] = phase
    central["power_curve"] = [[
        max_load * i / 256,
        phase["p0_w"] + phase["delta_w"] * (max_load * i / 256)
        / (1 + max_load * i / 256),
    ] for i in range(257)]
    profile.update(schema="queue-haul-model-profile-v5", status="fitted",
                   profile_id=profile["profile_id"] + "-phase-power-v1",
                   max_power_load=max_load, cases={"central": central})
    profile["sources"]["power"] = {
        "kind": "measured", "reference": str(measurements_path),
        "valid_range": [0, max_load],
        "relative_error": result["grouped_cv_rmse_w"] / phase["delta_w"],
    }
    out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--repeats", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=1)
    run_command = sub.add_parser("run")
    run_command.add_argument("--plan", type=Path, required=True)
    run_command.add_argument("--profile", type=Path, required=True)
    run_command.add_argument("--out", type=Path, required=True)
    run_command.add_argument("--host", default="127.0.0.1")
    run_command.add_argument("--port", type=int, default=8100)
    run_command.add_argument("--resume", action="store_true")
    augment = sub.add_parser("augment")
    augment.add_argument("--plan", type=Path, required=True)
    augment.add_argument("--summary", type=Path, required=True)
    augment.add_argument("--out", type=Path, required=True)
    fit_command = sub.add_parser("fit")
    fit_command.add_argument("--base-profile", type=Path, required=True)
    fit_command.add_argument("--measurements", type=Path, required=True)
    fit_command.add_argument("--out-profile", type=Path, required=True)
    fit_command.add_argument("--summary", type=Path, required=True)
    fit_command.add_argument("--idle-power-w", type=float, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "plan.json").write_text(
            json.dumps(campaign_plan(args.repeats, args.seed), indent=2, sort_keys=True) + "\n")
    elif args.command == "run":
        run_plan(args.plan, args.profile, args.out, args.host, args.port, args.resume)
    elif args.command == "augment":
        args.out.write_text(json.dumps(adaptive_plan(
            json.loads(args.plan.read_text()), json.loads(args.summary.read_text())),
            indent=2, sort_keys=True) + "\n")
    else:
        result = freeze_profile(args.base_profile, args.measurements,
                                args.out_profile, args.idle_power_w)
        args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
