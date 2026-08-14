"""Measure and fit H100 power from synchronized realized-token windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

PROMPT_COUNTER = "vllm:prompt_tokens_total"
DECODE_COUNTER = "vllm:generation_tokens_total"
CACHED_COUNTER = "vllm:prompt_tokens_cached_total"
COUNTER_TOLERANCE_FRACTION = .001
COUNTER_TOLERANCE_TOKENS = 1


@dataclass(frozen=True)
class Cell:
    stage: str
    family: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    replicate: int


def cells(seed: int = 1) -> list[Cell]:
    discovery = (
        ((2048, 1), (8192, 1), (28672, 1), (256, 512), (8192, 512),
         (28672, 512), (604, 64), (8192, 64), (16384, 64)),
        (1, 2, 4, 8, 16), 2,
    )
    confirmation = (
        ((4096, 1), (16384, 1), (4096, 512), (16384, 512),
         (604, 64), (12288, 64)),
        (3, 6, 12), 1,
    )
    rng = random.Random(seed)
    stages = []
    for stage, (work, concurrencies, reps) in (
            ("discovery", discovery), ("confirmation", confirmation)):
        stage_cells = [Cell(stage, family(p, g), p, g, c, r)
                       for p, g in work for c in concurrencies for r in range(reps)]
        rng.shuffle(stage_cells)
        stages.append(stage_cells)
    idle = [Cell("idle", "idle", 0, 0, 0, r) for r in range(3)]
    return [idle[0], *stages[0], idle[1], *stages[1], idle[2]]


def family(prompt: int, output: int) -> str:
    if output == 1:
        return "prefill"
    if output == 512:
        return "decode"
    if prompt == 604:
        return "campaign"
    return "agentic"


def parse_metrics(text: str) -> dict[str, float]:
    wanted = {PROMPT_COUNTER, DECODE_COUNTER, CACHED_COUNTER}
    totals = {name: 0.0 for name in wanted}
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in wanted:
            totals[name] += float(line.rsplit(None, 1)[1])
            seen.add(name)
    missing = wanted - seen
    if missing:
        raise RuntimeError(f"missing vLLM counters: {sorted(missing)}")
    return totals


def metrics(url: str) -> tuple[dict[str, float], int]:
    before = time.monotonic_ns()
    text = urlopen(url, timeout=10).read().decode()
    after = time.monotonic_ns()
    return parse_metrics(text), (before + after) // 2


def gpu_sample() -> tuple:
    before = time.monotonic_ns()
    row = subprocess.check_output([
        "nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    after = time.monotonic_ns()
    return (before + after) // 2, time.time_ns(), *map(str.strip, row)


def sample_power(path: Path, stop: threading.Event, errors: list[BaseException]) -> None:
    try:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "wall_ns", "gpu_timestamp", "power_w",
                             "utilization_pct", "memory_mib"))
            while not stop.is_set():
                writer.writerow(gpu_sample())
                handle.flush()
                stop.wait(.1)
    except BaseException as exc:
        errors.append(exc)
        stop.set()


def completion(url: str, cell: Cell, request_id: int) -> dict:
    prompt = [17] * cell.prompt_tokens
    if prompt:
        prompt[0] = 100 + request_id % 1000
    body = json.dumps({"model": "openai/gpt-oss-20b", "prompt": prompt,
                       "max_tokens": cell.output_tokens, "ignore_eos": True,
                       "temperature": 0}).encode()
    start = time.monotonic_ns()
    with urlopen(Request(url, body, {"Content-Type": "application/json"}),
                 timeout=600) as response:
        result = json.load(response)
    return {"start_ns": start, "end_ns": time.monotonic_ns(),
            "usage": result["usage"]}


def batch(url: str, cell: Cell, first_id: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=cell.concurrency) as pool:
        rows = list(pool.map(lambda i: completion(url, cell, first_id + i),
                             range(cell.concurrency)))
    if len(rows) != cell.concurrency:
        raise RuntimeError("incomplete synchronous batch")
    return rows


def accounting(cell: Cell, requests: list[dict], batches: int,
               deltas: dict[str, float]) -> dict:
    expected = batches * cell.concurrency
    if len(requests) != expected or (cell.concurrency and len(requests) % cell.concurrency):
        raise RuntimeError(f"completed {len(requests)} of {expected} measured requests")
    prompt = sum(int(row["usage"]["prompt_tokens"]) for row in requests)
    decode = sum(int(row["usage"]["completion_tokens"]) for row in requests)
    cached = sum(int((row["usage"].get("prompt_tokens_details") or {})
                     .get("cached_tokens", 0)) for row in requests)
    expected_prompt = expected * cell.prompt_tokens
    expected_decode = expected * cell.output_tokens
    if prompt != expected_prompt or decode != expected_decode:
        raise RuntimeError(f"API usage {(prompt, decode)} != scheduled {(expected_prompt, expected_decode)}")
    if cached or deltas[CACHED_COUNTER]:
        raise RuntimeError(f"cached prompt tokens observed: API={cached}, counter={deltas[CACHED_COUNTER]}")
    if cell.concurrency and prompt + decode == 0:
        raise RuntimeError("non-idle cell completed zero work")
    if not cell.concurrency and (requests or sum(deltas.values())):
        raise RuntimeError("idle cell reported inference work")
    for name, observed, exact in ((PROMPT_COUNTER, deltas[PROMPT_COUNTER], prompt),
                                  (DECODE_COUNTER, deltas[DECODE_COUNTER], decode)):
        tolerance = max(COUNTER_TOLERANCE_TOKENS,
                        COUNTER_TOLERANCE_FRACTION * exact)
        if abs(observed - exact) > tolerance:
            raise RuntimeError(f"{name} counter/API disagreement: {observed} != {exact} ± {tolerance}")
    return {"reported_prompt_tokens": deltas[PROMPT_COUNTER],
            "realized_prefill_tokens": prompt, "realized_decode_tokens": decode,
            "cached_prompt_tokens": cached, "counter_tolerance_fraction":
            COUNTER_TOLERANCE_FRACTION, "counter_tolerance_tokens": COUNTER_TOLERANCE_TOKENS}


def run_cell(cell: Cell, base_url: str, out: Path, window_s: float,
             cooldown_s: float, sequence: int) -> dict:
    label = f"{sequence:03d}-{cell.stage}-{cell.family}-p{cell.prompt_tokens}-g{cell.output_tokens}-c{cell.concurrency}-r{cell.replicate}"
    power_path = out / "power" / f"{label}.csv"
    stop = threading.Event()
    errors: list[BaseException] = []
    sampler = threading.Thread(target=sample_power, args=(power_path, stop, errors))
    sampler.start()
    try:
        url = f"{base_url}/v1/completions"
        first_id = sequence * 1_000_000
        if cell.concurrency:
            batch(url, cell, first_id)
        start_metrics, _ = metrics(f"{base_url}/metrics")
        start_ns = time.monotonic_ns()
        requests: list[dict] = []
        batches = 0
        if cell.concurrency:
            while (time.monotonic_ns() - start_ns) / 1e9 < window_s:
                requests.extend(batch(url, cell, first_id + (batches + 1) * cell.concurrency))
                batches += 1
        else:
            time.sleep(window_s)
        end_ns = time.monotonic_ns()
        end_metrics, _ = metrics(f"{base_url}/metrics")
    finally:
        stop.set(); sampler.join()
    if errors:
        raise errors[0]
    duration = (end_ns - start_ns) / 1e9
    deltas = {name: end_metrics[name] - start_metrics[name] for name in start_metrics}
    if min(deltas.values()) < 0:
        raise RuntimeError("vLLM counter reset during cell")
    work = accounting(cell, requests, batches, deltas)
    if any(row["start_ns"] < start_ns or row["end_ns"] > end_ns for row in requests):
        raise RuntimeError("measured request crossed a power boundary")
    with power_path.open() as handle:
        power = [float(row["power_w"]) for row in csv.DictReader(handle)
                 if start_ns <= int(row["monotonic_ns"]) < end_ns]
    if len(power) < 5 * duration:
        raise RuntimeError(f"only {len(power)} synchronized power samples in {duration:.1f}s")
    row = {**asdict(cell), "sequence": sequence, "start_ns": start_ns,
           "end_ns": end_ns, "window_s": duration, "batches": batches, **work,
           "realized_prefill_tps": work["realized_prefill_tokens"] / duration,
           "realized_decode_tps": work["realized_decode_tokens"] / duration,
           "power_mean_w": statistics.fmean(power),
           "power_p50_w": statistics.median(power), "power_samples": len(power),
           "completed_requests": len(requests),
           "request_count": len(requests), "power_path": str(power_path)}
    with (out / "requests.jsonl").open("a") as handle:
        for request_row in requests:
            handle.write(json.dumps({"cell": label, **request_row}) + "\n")
    time.sleep(cooldown_s)
    return row


def saturating_fit(rows: list[dict]) -> dict:
    train = [r for r in rows if r["stage"] == "discovery"]
    idle = [r["power_mean_w"] for r in rows if r["stage"] == "idle"]
    if len(train) != 90 or len(idle) != 3:
        raise ValueError("fit requires the complete discovery grid and three idle anchors")
    p0 = statistics.median(idle)
    fcap = max(r["realized_prefill_tps"] for r in train if r["family"] == "prefill")
    gcap = max(r["realized_decode_tps"] for r in train if r["family"] == "decode")
    alpha = 1 / fcap
    f = np.array([r["realized_prefill_tps"] for r in train])
    g = np.array([r["realized_decode_tps"] for r in train])
    power = np.array([r["power_mean_w"] for r in train])
    best = None
    for beta in np.geomspace(.2 / gcap, 5 / gcap, 161):
        ell = alpha * f + beta * g
        for knee in np.geomspace(.02, 5, 161):
            x = 1 - np.exp(-ell / knee)
            amplitude = max(0.0, float(np.dot(x, power - p0) / np.dot(x, x)))
            predicted = p0 + amplitude * x
            mse = float(np.mean((power - predicted) ** 2))
            candidate = (mse, beta, knee, amplitude)
            best = candidate if best is None or candidate < best else best
    mse, beta, knee, amplitude = best
    return {"power_idle_w": p0, "power_max_w": p0 + amplitude,
            "alpha_s_per_prefill_token": alpha,
            "beta_s_per_decode_token": beta, "F_prefill_tps": fcap,
            "G_decode_tps": gcap, "ell_knee": knee,
            "discovery_rmse_w": math.sqrt(mse)}


def predict(row: dict, fit: dict) -> float:
    ell = (fit["alpha_s_per_prefill_token"] * row["realized_prefill_tps"]
           + fit["beta_s_per_decode_token"] * row["realized_decode_tps"])
    return fit["power_idle_w"] + (fit["power_max_w"] - fit["power_idle_w"]) * (
        1 - math.exp(-ell / fit["ell_knee"]))


def validate(rows: list[dict], fit: dict) -> dict:
    held = [r for r in rows if r["stage"] == "confirmation"]
    errors = [abs(r["power_mean_w"] - predict(r, fit)) for r in held]
    family_mae = {name: statistics.fmean(
        abs(r["power_mean_w"] - predict(r, fit)) for r in held if r["family"] == name)
        for name in sorted({r["family"] for r in held})}
    halves = [saturating_fit(
        [r for r in rows if r["stage"] == "discovery" and r["replicate"] == replicate] * 2
        + [r for r in rows if r["stage"] == "idle"])
              for replicate in (0, 1)]
    stability = abs(halves[0]["beta_s_per_decode_token"]
                    - halves[1]["beta_s_per_decode_token"]) / fit["beta_s_per_decode_token"]
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        if row["stage"] == "discovery":
            key = (row["family"], row["prompt_tokens"], row["output_tokens"], row["concurrency"])
            grouped.setdefault(key, []).append(row["power_mean_w"])
    replicate_p90 = float(np.quantile([abs(x[0] - x[1]) for x in grouped.values()], .9))
    mean = statistics.fmean(r["power_mean_w"] for r in held)
    ss_res = sum((r["power_mean_w"] - predict(r, fit)) ** 2 for r in held)
    ss_tot = sum((r["power_mean_w"] - mean) ** 2 for r in held)
    report = {"holdout_cells": len(held), "holdout_mae_w": statistics.fmean(errors),
              "holdout_p90_abs_error_w": float(np.quantile(errors, .9)),
              "holdout_r2": 1 - ss_res / ss_tot, "family_mae_w": family_mae,
              "beta_split_relative_difference": stability,
              "replicate_p90_difference_w": replicate_p90,
              "cached_prompt_tokens": sum(r["cached_prompt_tokens"] for r in rows)}
    report["gates"] = {
        "holdout_mae_le_5w": report["holdout_mae_w"] <= 5,
        "holdout_p90_le_10w": report["holdout_p90_abs_error_w"] <= 10,
        "holdout_r2_ge_0p95": report["holdout_r2"] >= .95,
        "each_family_mae_le_8w": max(family_mae.values()) <= 8,
        "beta_split_difference_le_20pct": stability <= .2,
        "replicate_p90_difference_le_8w": replicate_p90 <= 8,
        "zero_cached_prompt_tokens": report["cached_prompt_tokens"] == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report


def validate_gpu() -> dict:
    fields = "name,uuid,power.limit"
    lines = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}",
                                     "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(lines) != 1 or not lines[0].startswith("NVIDIA H100 NVL,") \
            or not lines[0].endswith(", 400.00"):
        raise RuntimeError(f"expected one NVIDIA H100 NVL at 400 W, got {lines}")
    name, uuid, limit = map(str.strip, lines[0].split(","))
    return {"name": name, "uuid": uuid, "power_limit_w": float(limit)}


def wait_ready(base_url: str, server: subprocess.Popen, timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            urlopen(f"{base_url}/health", timeout=2).read()
            metrics(f"{base_url}/metrics")
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy")


def run(args) -> None:
    gpu = validate_gpu()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "power").mkdir()
    metadata = {"started_wall_ns": time.time_ns(), "gpu": gpu,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "minimum_window_s": args.window_s, "warmup": "one complete batch",
                "cooldown_s": args.cooldown_s, "seed": args.seed}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    server_log = (args.out / "server.log").open("w")
    server_cmd = [args.vllm, "serve", str(args.model), "--host", args.host,
                  "--port", str(args.port), "--served-model-name", "openai/gpt-oss-20b",
                  "--tensor-parallel-size", "1", "--max-model-len", "32768",
                  "--max-num-seqs", "256", "--max-num-batched-tokens", "8192",
                  "--kv-cache-dtype", "auto", "--block-size", "16",
                  "--enable-chunked-prefill", "--enforce-eager",
                  "--gpu-memory-utilization", ".75", "--disable-hybrid-kv-cache-manager",
                  "--no-enable-prefix-caching"]
    server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        base_url = f"http://{args.host}:{args.port}"
        wait_ready(base_url, server)
        rows = []
        for sequence, cell in enumerate(cells(args.seed)):
            row = run_cell(cell, base_url, args.out, args.window_s,
                           args.cooldown_s, sequence)
            rows.append(row)
            with (args.out / "cells.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        fit = saturating_fit(rows)
        report = validate(rows, fit)
        result = {"status": "calibrated" if report["passed"] else "holdout_failed",
                  "fit": fit, "validation": report}
        (args.out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("power calibration holdout gates failed")
    finally:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--window-s", type=float, default=12)
    parser.add_argument("--cooldown-s", type=float, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)
    if min(args.window_s, args.cooldown_s) <= 0:
        raise ValueError("cell durations must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
