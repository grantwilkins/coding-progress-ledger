"""Fit and freeze the Azure phase-aware source-power model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

import power_rate_sweep
from profiles import ModelProfile


MIXTURES = ("prefill", "prefill75", "mixed", "decode75", "decode")
LOADS = (.1, .25, .45, .65, .8, 1.0)
MIN_POWER_SAMPLE_HZ = 7


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
            "idle_measurement_s": 30,
            "gates": {"grouped_cv_rmse_w": 5, "within_5w_fraction": .8}}


def _metrics(host: str, port: int) -> tuple[float, float, float]:
    text = urlopen(f"http://{host}:{port}/metrics", timeout=10).read().decode()
    def total(name):
        values = [float(match.group(1)) for match in re.finditer(
            rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", text, re.MULTILINE)]
        if not values:
            raise RuntimeError(f"missing {name} counter")
        return sum(values)
    return (total("vllm:prompt_tokens_total"), total("vllm:generation_tokens_total"),
            total("vllm:prompt_tokens_cached_total"))


def _shape(fraction: float, load: float, F: float, G: float) -> tuple[str, int, float, int]:
    f, g = fraction * load * F, (1 - fraction) * load * G
    if fraction == 0:
        prompt_tokens, output_tokens = 1, 4096
        return "x", output_tokens, g / output_tokens, max(1, math.ceil(g * 40 / output_tokens))
    prompt_tokens = 4096 if fraction == 1 else 2048
    output_tokens = 1 if fraction == 1 else max(1, round(prompt_tokens * g / f))
    return "x " * prompt_tokens, output_tokens, f / prompt_tokens, 0


def run_cell(host: str, port: int, root: Path, cell: dict,
             F: float, G: float, model: str = "openai/gpt-oss-20b",
             workers: int = 512) -> dict:
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
                    output_tokens, f"{label}:{request_id}", model)
            futures = [pool.submit(scheduled_request, request_id)
                       for request_id in range(count)]
            remaining = started + warmup - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            metric_start, start_ns = _metrics(host, port), time.monotonic_ns()
            time.sleep(window)
            metric_end, end_ns = _metrics(host, port), time.monotonic_ns()
            results = [future.result() for future in futures]
    finally:
        stop.set(); sampler.join()
    with path.open(newline="") as handle:
        watts = [float(row["power_w"]) for row in csv.DictReader(handle)
                 if start_ns <= int(row["monotonic_ns"]) < end_ns]
    if len(watts) < window * MIN_POWER_SAMPLE_HZ:
        raise RuntimeError("insufficient power samples")
    cached = metric_end[2] - metric_start[2]
    if cached or any((row["usage"].get("prompt_tokens_details") or {})
                     .get("cached_tokens", 0) for row in results):
        raise RuntimeError("cached prompt tokens observed")
    f, g = (metric_end[index] - metric_start[index] for index in (0, 1))
    if min(f, g) < 0 or f + g <= 0:
        raise RuntimeError("invalid realized token counters")
    return {"mixture": cell["mixture"], "repeat": cell["repeat"],
            "target_service_load": load, "f_tps": f / window,
            "g_tps": g / window, "cached_prompt_tokens": 0,
            "power_mean_w": float(np.mean(watts)), "power_samples": len(watts),
            "start_ns": start_ns, "end_ns": end_ns, "power_path": str(path)}


def measure_idle(host: str, port: int, root: Path, sequence: int,
                 seconds: float) -> dict:
    path = root / f"power-idle-{sequence:03d}.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=power_rate_sweep.power,
                               args=(path, stop, .1))
    sampler.start()
    try:
        metric_start, start_ns = _metrics(host, port), time.monotonic_ns()
        time.sleep(seconds)
        metric_end, end_ns = _metrics(host, port), time.monotonic_ns()
    finally:
        stop.set(); sampler.join()
    with path.open(newline="") as handle:
        watts = [float(row["power_w"]) for row in csv.DictReader(handle)
                 if start_ns <= int(row["monotonic_ns"]) < end_ns]
    if any(end != start for start, end in zip(metric_start, metric_end)):
        raise RuntimeError("idle anchor processed inference work")
    if len(watts) < seconds * MIN_POWER_SAMPLE_HZ:
        raise RuntimeError("insufficient idle power samples")
    row = {"sequence": sequence, "window_s": (end_ns - start_ns) / 1e9,
           "power_mean_w": float(np.mean(watts)), "power_samples": len(watts),
           "start_ns": start_ns, "end_ns": end_ns, "power_path": str(path)}
    with (root / "idle.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n"); handle.flush(); os.fsync(handle.fileno())
    return row


def calibration_target(profile_path: Path | None, model: str | None,
                       hardware: str | None, F: float | None,
                       G: float | None) -> tuple[str, str, float, float]:
    if profile_path is not None:
        if any(value is not None for value in (model, hardware, F, G)):
            raise ValueError("profile and explicit calibration target are mutually exclusive")
        profile = ModelProfile.load(profile_path)
        return profile.model, profile.hardware, profile.case().F, profile.case().G
    if not model or hardware not in {"a100", "h100"} \
            or F is None or G is None or min(F, G) <= 0:
        raise ValueError("explicit target requires model, hardware, F, and G")
    return model, hardware, F, G


def run_plan(plan_path: Path, profile_path: Path | None, out: Path,
             host: str = "127.0.0.1", port: int = 8100,
             resume: bool = False, *, model: str | None = None,
             hardware: str | None = None, F: float | None = None,
             G: float | None = None, provenance: dict | None = None) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") not in {"queue-haul-phase-power-plan-v1",
                                  "queue-haul-phase-power-adaptive-plan-v1"}:
        raise ValueError("invalid phase power plan")
    model, hardware, F, G = calibration_target(profile_path, model, hardware, F, G)
    expected_gpu = {"a100": ("NVIDIA A100 80GB PCIe", 300),
                    "h100": ("NVIDIA H100 NVL", 400)}[hardware]
    power_rate_sweep.validate_gpu(*expected_gpu)
    out.mkdir(parents=True, exist_ok=resume)
    metadata = {"schema": "queue-haul-phase-power-run-v1", "model": model,
                "hardware": hardware, "F_prefill_tps": F, "G_decode_tps": G,
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                **(provenance or {})}
    metadata_path = out / "metadata.json"
    if resume:
        if json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError("phase power resume metadata changed")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    measurement_path = out / "measurements.csv"
    rows = []
    if resume and measurement_path.exists():
        with measurement_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {(row["mixture"], float(row["target_service_load"]), int(row["repeat"]))
                 for row in rows}
    idle_s = plan.get("idle_measurement_s")
    if idle_s:
        idle_path = out / "idle.jsonl"
        sequence = len(idle_path.read_text().splitlines()) if idle_path.exists() else 0
        measure_idle(host, port, out, sequence, float(idle_s))
    for cell in plan["cells"]:
        key = cell["mixture"], float(cell["target_service_load"]), int(cell["repeat"])
        if key in completed:
            continue
        rows.append(run_cell(host, port, out, cell, F, G, model))
        with measurement_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(rows[-1]), flush=True)
    if idle_s:
        measure_idle(host, port, out, sequence + 1, float(idle_s))
    return rows


def run_with_server(plan_path: Path, out: Path, model: str, hardware: str,
                    F: float, G: float, vllm: str, host: str, port: int,
                    resume: bool) -> list[dict]:
    if hardware != "h100":
        raise ValueError("managed phase-power server launch currently requires H100")
    import power_model_campaign as power
    gpu = power.validate_gpu()
    args = SimpleNamespace(model=model, vllm=vllm, host=host, port=port)
    command = power.server_command(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = len(list(out.parent.glob(f"{out.name}-server*.log")))
    suffix = "" if not prior else f"-resume-{prior:03d}"
    log_path = out.parent / f"{out.name}-server{suffix}.log"
    log = log_path.open("x")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        power.wait_ready(f"http://{host}:{port}", server)
        rows = run_plan(
            plan_path, None, out, host, port, resume, model=model,
            hardware=hardware, F=F, G=G, provenance={
                "gpu": gpu, "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True).strip(),
                "server_command": command,
            })
        log.flush()
        power.testbed.validate_h100_optimized_runtime(
            " ".join(command), log_path.read_text(errors="replace"))
        return rows
    finally:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.close()


def _predict(parameters, f, g, p0):
    delta, a, b = parameters
    z = a * f + b * g
    return p0 + delta * z / (1 + z)


def _fit(rows: list[dict], p0: float, minimum: int = 3):
    f = np.asarray([float(row["f_tps"]) for row in rows])
    g = np.asarray([float(row["g_tps"]) for row in rows])
    watts = np.asarray([float(row["power_mean_w"]) for row in rows])
    if len(rows) < minimum or np.linalg.matrix_rank(np.column_stack((f, g))) != 2:
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
    if idle_power_w <= 0 or bootstrap_samples < 1 or len(rows) < 8 \
            or any(not required <= row.keys() for row in rows):
        raise ValueError("invalid phase power measurements")
    group = lambda row: str(row.get("validation_group", row["mixture"]))
    groups = sorted({group(row) for row in rows})
    if len(groups) < 3 or any(sum(group(row) == name for row in rows) < 2
                              for name in groups):
        raise ValueError("grouped validation needs three mixtures with repeats")
    parameters = _fit(rows, idle_power_w)
    errors, heldout = [], []
    for name in groups:
        train = [row for row in rows if group(row) != name]
        test = [row for row in rows if group(row) == name]
        fitted = _fit(train, idle_power_w)
        for row in test:
            error = float(_predict(fitted, float(row["f_tps"]), float(row["g_tps"]),
                                   idle_power_w) - float(row["power_mean_w"]))
            errors.append(error)
            heldout.append({"mixture": row["mixture"], "validation_group": name,
                            "target_service_load": row.get("target_service_load"),
                            "error_w": error})
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    within = float(np.mean(np.abs(errors) <= 5))
    group_bias = {name: float(np.mean([row["error_w"] for row in heldout
                                      if row["validation_group"] == name]))
                  for name in groups}
    rng, bootstrap = np.random.default_rng(seed), []
    grouped = {name: [row for row in rows if group(row) == name] for name in groups}
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
    run_command.add_argument("--profile", type=Path)
    run_command.add_argument("--model")
    run_command.add_argument("--hardware", choices=("a100", "h100"))
    run_command.add_argument("--prefill-tps", type=float)
    run_command.add_argument("--decode-tps", type=float)
    run_command.add_argument("--vllm")
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
    idle = fit_command.add_mutually_exclusive_group(required=True)
    idle.add_argument("--idle-power-w", type=float)
    idle.add_argument("--idle-measurements", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "plan.json").write_text(
            json.dumps(campaign_plan(args.repeats, args.seed), indent=2, sort_keys=True) + "\n")
    elif args.command == "run":
        if args.vllm:
            model, hardware, F, G = calibration_target(
                args.profile, args.model, args.hardware,
                args.prefill_tps, args.decode_tps)
            run_with_server(args.plan, args.out, model, hardware, F, G,
                            args.vllm, args.host, args.port, args.resume)
        else:
            run_plan(args.plan, args.profile, args.out, args.host, args.port, args.resume,
                     model=args.model, hardware=args.hardware,
                     F=args.prefill_tps, G=args.decode_tps)
    elif args.command == "augment":
        args.out.write_text(json.dumps(adaptive_plan(
            json.loads(args.plan.read_text()), json.loads(args.summary.read_text())),
            indent=2, sort_keys=True) + "\n")
    else:
        idle_power = args.idle_power_w
        if args.idle_measurements:
            idle_power = statistics.median(json.loads(line)["power_mean_w"]
                                           for line in args.idle_measurements.read_text().splitlines())
        result = freeze_profile(args.base_profile, args.measurements,
                                args.out_profile, idle_power)
        args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
