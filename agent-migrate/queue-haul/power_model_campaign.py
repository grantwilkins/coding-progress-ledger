"""Measure and fit H100 power from synchronized realized-token windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


def cell_label(sequence: int, cell: Cell) -> str:
    return (f"{sequence:03d}-{cell.stage}-{cell.family}-p{cell.prompt_tokens}"
            f"-g{cell.output_tokens}-c{cell.concurrency}-r{cell.replicate}")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_row_numbers(row: dict, watts: list[float]) -> None:
    duration = (row["end_ns"] - row["start_ns"]) / 1e9
    if not math.isclose(row["window_s"], duration, abs_tol=1e-9) \
            or not math.isclose(row["realized_prefill_tps"],
                                row["realized_prefill_tokens"] / duration, rel_tol=1e-12) \
            or not math.isclose(row["realized_decode_tps"],
                                row["realized_decode_tokens"] / duration, rel_tol=1e-12):
        raise RuntimeError(f"row {row['sequence']} has inconsistent window/rates")
    tolerance = max(row["counter_tolerance_tokens"],
                    row["counter_tolerance_fraction"] * row["realized_prefill_tokens"])
    if abs(row["reported_prompt_tokens"] - row["realized_prefill_tokens"]) > tolerance:
        raise RuntimeError(f"row {row['sequence']} has counter/API disagreement")
    if not math.isclose(row["power_mean_w"], statistics.fmean(watts), rel_tol=1e-12) \
            or not math.isclose(row["power_p50_w"], statistics.median(watts), rel_tol=1e-12):
        raise RuntimeError(f"row {row['sequence']} has inconsistent power reduction")


def validate_request_evidence(label: str, row: dict, evidence: list[dict]) -> None:
    if len(evidence) != row["request_count"]:
        raise RuntimeError(f"{label} request evidence count mismatch")
    if any(item["start_ns"] < row["start_ns"] or item["end_ns"] > row["end_ns"]
           for item in evidence):
        raise RuntimeError(f"{label} request crossed a committed boundary")
    cached = sum(int((item["usage"].get("prompt_tokens_details") or {})
                     .get("cached_tokens", 0)) for item in evidence)
    if cached or sum(item["usage"]["prompt_tokens"] for item in evidence) \
            != row["realized_prefill_tokens"] \
            or sum(item["usage"]["completion_tokens"] for item in evidence) \
            != row["realized_decode_tokens"]:
        raise RuntimeError(f"{label} request token/cache evidence mismatch")


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
    label = cell_label(sequence, cell)
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
    append_jsonl(out / "requests.jsonl",
                 [{"cell": label, **request_row} for request_row in requests])
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
    base_alpha = 1 / fcap
    f = np.array([r["realized_prefill_tps"] for r in train])
    g = np.array([r["realized_decode_tps"] for r in train])
    power = np.array([r["power_mean_w"] for r in train])
    best = None
    for base_beta in np.geomspace(.2 / gcap, 5 / gcap, 161):
        ell = base_alpha * f + base_beta * g
        for scale in np.geomspace(.2, 50, 161):
            z = scale * ell
            x = z / (1 + z)
            amplitude = max(0.0, float(np.dot(x, power - p0) / np.dot(x, x)))
            predicted = p0 + amplitude * x
            mse = float(np.mean((power - predicted) ** 2))
            candidate = (mse, base_beta, scale, amplitude)
            best = candidate if best is None or candidate < best else best
    mse, base_beta, scale, amplitude = best
    return {"schema": "queue-haul-rational-power-v1",
            "link": "P=P0+A*z/(1+z); z=alpha*f+beta*g",
            "power_idle_w": p0, "power_amplitude_w": amplitude,
            "power_max_w": p0 + amplitude,
            "alpha_s_per_prefill_token": scale * base_alpha,
            "beta_s_per_decode_token": scale * base_beta,
            "F_prefill_tps": fcap, "G_decode_tps": gcap,
            "discovery_rmse_w": math.sqrt(mse)}


def predict(row: dict, fit: dict) -> float:
    z = (fit["alpha_s_per_prefill_token"] * row["realized_prefill_tps"]
         + fit["beta_s_per_decode_token"] * row["realized_decode_tps"])
    return fit["power_idle_w"] + fit["power_amplitude_w"] * z / (1 + z)


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


def fit_result(rows: list[dict]) -> dict:
    fit = saturating_fit(rows)
    report = validate(rows, fit)
    return {"schema": "queue-haul-rational-power-fit-v1",
            "status": "calibrated" if report["passed"] else "holdout_failed",
            "fit": fit, "interaction_diagnostic": interaction_diagnostic(rows, fit),
            "validation": report}


def interaction_diagnostic(rows: list[dict], fit: dict) -> dict:
    result = {}
    for stage in ("discovery", "confirmation"):
        selected = [row for row in rows if row["stage"] == stage]
        interaction = np.array([(row["realized_prefill_tps"] / fit["F_prefill_tps"])
                                * (row["realized_decode_tps"] / fit["G_decode_tps"])
                                for row in selected])
        residual = np.array([row["power_mean_w"] - predict(row, fit) for row in selected])
        slope = float(np.dot(interaction, residual) / np.dot(interaction, interaction))
        result[stage] = {"residual_w_per_normalized_f_g": slope,
                         "base_rmse_w": float(np.sqrt(np.mean(residual ** 2))),
                         "adjusted_rmse_w": float(np.sqrt(np.mean((residual - slope * interaction) ** 2))),
                         "residual_interaction_correlation": float(np.corrcoef(
                             interaction, residual)[0, 1])}
    return result


def complete_rows(out: Path, seed: int) -> list[dict]:
    plan = cells(seed)
    lines = (out / "cells.jsonl").read_text().splitlines()
    if len(lines) != len(plan):
        raise RuntimeError(f"offline refit requires all {len(plan)} cells, found {len(lines)}")
    rows = [json.loads(line) for line in lines]
    for sequence, (row, cell) in enumerate(zip(rows, plan, strict=True)):
        if any(row.get(key) != value for key, value in
               {**asdict(cell), "sequence": sequence}.items()):
            raise RuntimeError(f"row {sequence} does not match deterministic grid")
    return rows


def refit(args) -> None:
    rows = complete_rows(args.out, args.seed)
    fit_path = args.out / "fit.json"
    if not fit_path.is_file():
        raise RuntimeError("offline refit requires the provisional in-run fit.json")
    provisional = args.out / "fit-exponential-provisional.json"
    if provisional.exists():
        raise RuntimeError("provisional exponential fit is already archived")
    with provisional.open("xb") as handle:
        handle.write(fit_path.read_bytes()); handle.flush(); os.fsync(handle.fileno())
    result = fit_result(rows)
    temporary = fit_path.with_suffix(".rational.tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, fit_path)
    if not result["validation"]["passed"]:
        raise RuntimeError("rational power calibration holdout gates failed")


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


def validate_resume(args, gpu: dict, plan: list[Cell]) -> list[dict]:
    if not args.expected_sha or len(args.expected_sha) != 40:
        raise ValueError("resume requires the full original --expected-sha")
    metadata = json.loads((args.out / "metadata.json").read_text())
    expected = {"gpu": gpu, "git_sha": args.expected_sha,
                "minimum_window_s": args.window_s,
                "warmup": "one complete batch", "cooldown_s": args.cooldown_s,
                "seed": args.seed}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"resume metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    if str(args.model) not in (args.out / "server.log").read_text(errors="replace"):
        raise RuntimeError("resume model does not match original server log")
    raw = (args.out / "cells.jsonl").read_text().splitlines()
    if not raw or any(not line.strip() for line in raw):
        raise RuntimeError("resume requires a nonempty unambiguous cells prefix")
    rows = [json.loads(line) for line in raw]
    if len(rows) >= len(plan) or (args.out / "fit.json").exists():
        raise RuntimeError("campaign is not an incomplete resumable prefix")
    valid_power = set()
    valid_labels = {}
    for sequence, (row, cell) in enumerate(zip(rows, plan, strict=False)):
        identity = {**asdict(cell), "sequence": sequence}
        if any(row.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"row {sequence} does not match deterministic prefix")
        expected_requests = row["batches"] * cell.concurrency
        if row["request_count"] != expected_requests \
                or row["completed_requests"] != expected_requests:
            raise RuntimeError(f"row {sequence} has incomplete request accounting")
        if row["cached_prompt_tokens"] or (cell.concurrency and
                                            not row["realized_prefill_tokens"] + row["realized_decode_tokens"]):
            raise RuntimeError(f"row {sequence} has invalid realized work")
        path = args.out / "power" / f"{cell_label(sequence, cell)}.csv"
        if row["power_path"] != str(path) or not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"row {sequence} has invalid power evidence")
        with path.open() as handle:
            watts = [float(sample["power_w"]) for sample in csv.DictReader(handle)
                     if row["start_ns"] <= int(sample["monotonic_ns"]) < row["end_ns"]]
        if len(watts) != row["power_samples"] or len(watts) < 5 * row["window_s"]:
            raise RuntimeError(f"row {sequence} has incomplete power samples")
        validate_row_numbers(row, watts)
        valid_power.add(path)
        valid_labels[cell_label(sequence, cell)] = row
    requests = {label: [] for label in valid_labels}
    for line in (args.out / "requests.jsonl").read_text().splitlines():
        request_row = json.loads(line)
        if request_row.get("cell") not in requests:
            raise RuntimeError("request evidence falls outside committed prefix")
        requests[request_row["cell"]].append(request_row)
    for label, row in valid_labels.items():
        validate_request_evidence(label, row, requests[label])
    discard = set(args.discard_orphan_sequences)
    if any(sequence < len(rows) or sequence >= len(plan) for sequence in discard):
        raise RuntimeError("orphan discard sequence is not in the uncommitted suffix")
    discard_paths = {args.out / "power" / f"{cell_label(sequence, plan[sequence])}.csv"
                     for sequence in discard}
    artifacts = set((args.out / "power").glob("*.csv"))
    if artifacts != valid_power | discard_paths or not all(path.exists() for path in discard_paths):
        raise RuntimeError("power artifacts do not equal committed prefix plus explicit orphans")
    for path in discard_paths:
        path.unlink()
    current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    append_jsonl(args.out / "resumes.jsonl", [{"wall_ns": time.time_ns(),
                                                "original_sha": args.expected_sha,
                                                "resume_sha": current_sha,
                                                "next_sequence": len(rows),
                                                "discarded_orphans": sorted(discard)}])
    return rows


def run(args) -> None:
    gpu = validate_gpu()
    plan = cells(args.seed)
    if args.resume:
        rows = validate_resume(args, gpu, plan)
        log_path = args.out / f"server-resume-{len(list(args.out.glob('server-resume-*.log'))) + 1:03d}.log"
    else:
        if args.expected_sha or args.discard_orphan_sequences:
            raise ValueError("orphan/SHA options require --resume")
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "power").mkdir()
        metadata = {"started_wall_ns": time.time_ns(), "gpu": gpu,
                    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "minimum_window_s": args.window_s, "warmup": "one complete batch",
                    "cooldown_s": args.cooldown_s, "seed": args.seed}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows, log_path = [], args.out / "server.log"
    server_log = log_path.open("x")
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
        for sequence in range(len(rows), len(plan)):
            cell = plan[sequence]
            row = run_cell(cell, base_url, args.out, args.window_s,
                           args.cooldown_s, sequence)
            rows.append(row)
            append_jsonl(args.out / "cells.jsonl", [row])
            print(json.dumps(row), flush=True)
        result = fit_result(rows)
        (args.out / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["validation"]["passed"]:
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-sha")
    parser.add_argument("--discard-orphan-sequences", nargs="*", type=int, default=[])
    parser.add_argument("--refit-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.window_s, args.cooldown_s) <= 0:
        raise ValueError("cell durations must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    refit(parsed) if parsed.refit_only else run(parsed)
