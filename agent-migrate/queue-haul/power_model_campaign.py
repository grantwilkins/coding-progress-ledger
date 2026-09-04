"""Measure and fit GPU power from synchronized realized-token windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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

import migration_testbed as testbed

PROMPT_COUNTER = "vllm:prompt_tokens_total"
DECODE_COUNTER = "vllm:generation_tokens_total"
CACHED_COUNTER = "vllm:prompt_tokens_cached_total"
COUNTER_TOLERANCE_FRACTION = .001
COUNTER_TOLERANCE_TOKENS = 1
MAX_ELL = 16.0
GPU_CONTRACTS = {
    "a100": ("NVIDIA A100 80GB PCIe", 300.0),
    "h100": ("NVIDIA H100 NVL", 400.0),
}
REPLICATION_SCHEMA = "queue-haul-rational-power-replication-v1"


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


def followup_cells(seed: int = 1) -> list[Cell]:
    work = [Cell(stage, "decode", 4096, 512, concurrency, replicate)
            for stage in ("targeted_calibration", "targeted_validation")
            for concurrency in (3, 6, 12) for replicate in range(3)]
    random.Random(seed).shuffle(work)
    plan = [Cell("idle", "idle", 0, 0, 0, 0)]
    for replicate, cell in enumerate(work, 1):
        plan += [cell, Cell("idle", "idle", 0, 0, 0, replicate)]
    return plan


def replication_cells(seed: int = 1) -> list[Cell]:
    """Acquire a prospectively split extension of the original power grid."""
    discovery_work = (
        (2048, 1), (8192, 1), (28672, 1), (256, 512), (8192, 512),
        (28672, 512), (604, 64), (8192, 64), (16384, 64),
    )
    confirmation_work = (
        (4096, 1), (16384, 1), (4096, 512), (16384, 512),
        (604, 64), (12288, 64),
    )
    rng = random.Random(seed)
    discovery = [
        Cell("replication_discovery", family(prompt, output), prompt,
             output, concurrency, replicate)
        for prompt, output in discovery_work
        for concurrency in (1, 2, 4, 8, 16)
        for replicate in (2, 3)
    ]
    confirmation = [
        Cell("replication_confirmation", family(prompt, output), prompt,
             output, concurrency, 0)
        for prompt, output in confirmation_work
        for concurrency in (3, 6, 12)
    ]
    rng.shuffle(discovery)
    rng.shuffle(confirmation)

    # Finish all training acquisition before exposing the prospective holdout.
    # The original three idle anchors remain training data; these three fresh
    # anchors make startup transients unable to determine P0 by themselves.
    plan = [Cell("replication_training_idle", "idle", 0, 0, 0, 3)]
    plan += discovery[:45]
    plan += [Cell("replication_training_idle", "idle", 0, 0, 0, 4)]
    plan += discovery[45:]
    plan += [Cell("replication_training_idle", "idle", 0, 0, 0, 5)]

    # Six independent idle anchors broaden the held-out operating envelope and
    # are interspersed with the 18 unseen active cells. They are never used to
    # fit P0 or any other parameter.
    for replicate in range(6):
        plan += confirmation[replicate * 3:(replicate + 1) * 3]
        plan += [Cell("replication_holdout_idle", "idle", 0, 0, 0,
                      replicate)]
    return plan


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
        "nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,"
        "clocks.sm,clocks.mem,temperature.gpu,pstate,clocks_event_reasons.active",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    after = time.monotonic_ns()
    return (before + after) // 2, time.time_ns(), *map(str.strip, row)


def sample_power(path: Path, stop: threading.Event, errors: list[BaseException]) -> None:
    try:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "wall_ns", "gpu_timestamp", "power_w",
                             "utilization_pct", "memory_mib", "sm_clock_mhz",
                             "memory_clock_mhz", "temperature_c", "pstate",
                             "active_clock_event_reasons"))
            while not stop.is_set():
                writer.writerow(gpu_sample())
                handle.flush()
                stop.wait(.1)
    except BaseException as exc:
        errors.append(exc)
        stop.set()


def completion(url: str, cell: Cell, request_id: int, model: str) -> dict:
    prompt = [17] * cell.prompt_tokens
    if prompt:
        prompt[0] = 100 + request_id % 1000
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": cell.output_tokens, "ignore_eos": True,
                       "temperature": 0}).encode()
    start = time.monotonic_ns()
    with urlopen(Request(url, body, {"Content-Type": "application/json"}),
                 timeout=600) as response:
        result = json.load(response)
    return {"start_ns": start, "end_ns": time.monotonic_ns(),
            "usage": result["usage"]}


def batch(url: str, cell: Cell, first_id: int, model: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=cell.concurrency) as pool:
        rows = list(pool.map(lambda i: completion(url, cell, first_id + i, model),
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
             cooldown_s: float, sequence: int, model: str) -> dict:
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
            batch(url, cell, first_id, model)
        start_metrics, _ = metrics(f"{base_url}/metrics")
        start_ns = time.monotonic_ns()
        requests: list[dict] = []
        batches = 0
        if cell.concurrency:
            while (time.monotonic_ns() - start_ns) / 1e9 < window_s:
                requests.extend(batch(url, cell, first_id + (batches + 1) * cell.concurrency,
                                      model))
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
        samples = [row for row in csv.DictReader(handle)
                   if start_ns <= int(row["monotonic_ns"]) < end_ns]
    telemetry = ("sm_clock_mhz", "memory_clock_mhz", "temperature_c", "pstate",
                 "active_clock_event_reasons")
    if any(not sample.get(field) for sample in samples for field in telemetry):
        raise RuntimeError("missing synchronized GPU telemetry")
    power = [float(row["power_w"]) for row in samples]
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


def rational_fit(train: list[dict], idle: list[dict]) -> dict:
    idle_power = [r["power_mean_w"] for r in idle]
    p0 = statistics.median(idle_power)
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
            "power_idle_w": float(p0), "power_amplitude_w": float(amplitude),
            "power_max_w": float(p0 + amplitude),
            "alpha_s_per_prefill_token": float(scale * base_alpha),
            "beta_s_per_decode_token": float(scale * base_beta),
            "F_prefill_tps": float(fcap), "G_decode_tps": float(gcap),
            "discovery_rmse_w": math.sqrt(mse)}


def saturating_fit(rows: list[dict]) -> dict:
    train = [r for r in rows if r["stage"] == "discovery"]
    idle = [r for r in rows if r["stage"] == "idle"]
    if len(train) != 90 or len(idle) != 3:
        raise ValueError("fit requires the complete discovery grid and three idle anchors")
    return rational_fit(train, idle)


def exponential_fit(rows: list[dict]) -> dict:
    train = [r for r in rows if r["stage"] == "discovery"]
    idle = [r["power_mean_w"] for r in rows if r["stage"] == "idle"]
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
            mse = float(np.mean((power - p0 - amplitude * x) ** 2))
            candidate = (mse, beta, knee, amplitude)
            best = candidate if best is None or candidate < best else best
    mse, beta, knee, amplitude = best
    return {"schema": "queue-haul-exponential-power-provisional-v1",
            "status": "provisional_reconstructed_after_serialization_failure",
            "power_idle_w": float(p0), "power_max_w": float(p0 + amplitude),
            "alpha_s_per_prefill_token": float(alpha),
            "beta_s_per_decode_token": float(beta),
            "F_prefill_tps": float(fcap), "G_decode_tps": float(gcap),
            "ell_knee": float(knee), "discovery_rmse_w": math.sqrt(mse)}


def predict(row: dict, fit: dict) -> float:
    z = (fit["alpha_s_per_prefill_token"] * row["realized_prefill_tps"]
         + fit["beta_s_per_decode_token"] * row["realized_decode_tps"])
    return fit["power_idle_w"] + fit["power_amplitude_w"] * z / (1 + z)


def power_curve(fit: dict) -> list[dict]:
    return [{"ell": ell, "power_w": fit["power_idle_w"]
             + fit["power_amplitude_w"] * ell / (1 + ell)}
            for ell in (0, .25, .5, 1, 2, 4, 8, MAX_ELL)]


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
    report = {"holdout_cells": int(len(held)), "holdout_mae_w": float(statistics.fmean(errors)),
              "holdout_p90_abs_error_w": float(np.quantile(errors, .9)),
              "holdout_r2": float(1 - ss_res / ss_tot),
              "family_mae_w": {name: float(value) for name, value in family_mae.items()},
              "beta_split_relative_difference": float(stability),
              "replicate_p90_difference_w": replicate_p90,
              "cached_prompt_tokens": int(sum(r["cached_prompt_tokens"] for r in rows))}
    report["gates"] = {
        "holdout_mae_le_5w": bool(report["holdout_mae_w"] <= 5),
        "holdout_p90_le_10w": bool(report["holdout_p90_abs_error_w"] <= 10),
        "holdout_r2_ge_0p95": bool(report["holdout_r2"] >= .95),
        "each_family_mae_le_8w": bool(max(family_mae.values()) <= 8),
        "beta_split_difference_le_20pct": bool(stability <= .2),
        "replicate_p90_difference_le_8w": bool(replicate_p90 <= 8),
        "zero_cached_prompt_tokens": bool(report["cached_prompt_tokens"] == 0),
    }
    report["passed"] = all(report["gates"].values())
    return report


def fit_result(rows: list[dict]) -> dict:
    fit = saturating_fit(rows)
    report = validate(rows, fit)
    return {"schema": "queue-haul-rational-power-fit-v1",
            "status": "calibrated" if report["passed"] else "holdout_failed",
            "fit": fit, "max_ell": MAX_ELL, "power_curve": power_curve(fit),
            "interaction_diagnostic": interaction_diagnostic(rows, fit),
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


def prediction_report(rows: list[dict], fit: dict) -> dict:
    errors = [r["power_mean_w"] - predict(r, fit) for r in rows]
    mean = statistics.fmean(r["power_mean_w"] for r in rows)
    ss_res = sum(error ** 2 for error in errors)
    ss_tot = sum((r["power_mean_w"] - mean) ** 2 for r in rows)
    by_concurrency = {str(concurrency): statistics.fmean(
        abs(r["power_mean_w"] - predict(r, fit)) for r in rows
        if r["concurrency"] == concurrency)
        for concurrency in sorted({r["concurrency"] for r in rows})}
    return {"cells": len(rows), "mae_w": float(statistics.fmean(map(abs, errors))),
            "p90_abs_error_w": float(np.quantile(np.abs(errors), .9)),
            "rmse_w": float(math.sqrt(ss_res / len(rows))),
            "r2": float(1 - ss_res / ss_tot),
            "concurrency_mae_w": by_concurrency,
            "max_abs_error_w": float(max(map(abs, errors)))}


def _replication_grid(rows: list[dict]) -> tuple[list[dict], list[dict],
                                                  list[dict], list[dict]]:
    discovery = [row for row in rows
                 if row["stage"] == "replication_discovery"]
    training_idle = [row for row in rows
                     if row["stage"] == "replication_training_idle"]
    confirmation = [row for row in rows
                    if row["stage"] == "replication_confirmation"]
    holdout_idle = [row for row in rows
                    if row["stage"] == "replication_holdout_idle"]
    expected_discovery = {
        (family(prompt, output), prompt, output, concurrency, replicate)
        for prompt, output in (
            (2048, 1), (8192, 1), (28672, 1), (256, 512),
            (8192, 512), (28672, 512), (604, 64), (8192, 64),
            (16384, 64),
        )
        for concurrency in (1, 2, 4, 8, 16)
        for replicate in (2, 3)
    }
    expected_confirmation = {
        (family(prompt, output), prompt, output, concurrency, 0)
        for prompt, output in (
            (4096, 1), (16384, 1), (4096, 512), (16384, 512),
            (604, 64), (12288, 64),
        )
        for concurrency in (3, 6, 12)
    }
    identity = lambda row: (
        row["family"], row["prompt_tokens"], row["output_tokens"],
        row["concurrency"], row["replicate"])
    if len(discovery) != 90 \
            or {identity(row) for row in discovery} != expected_discovery \
            or len(training_idle) != 3 \
            or {row["replicate"] for row in training_idle} != {3, 4, 5} \
            or len(confirmation) != 18 \
            or {identity(row) for row in confirmation} != expected_confirmation \
            or len(holdout_idle) != 6 \
            or {row["replicate"] for row in holdout_idle} != set(range(6)):
        raise RuntimeError("replication extension grid is incomplete or changed")
    return discovery, training_idle, confirmation, holdout_idle


def _replicate_p90(rows: list[dict]) -> float:
    grouped: dict[tuple, dict[int, float]] = {}
    for row in rows:
        key = (row["family"], row["prompt_tokens"], row["output_tokens"],
               row["concurrency"])
        values = grouped.setdefault(key, {})
        replicate = int(row["replicate"])
        if replicate in values:
            raise RuntimeError("duplicate discovery replicate")
        values[replicate] = float(row["power_mean_w"])
    if len(grouped) != 45 or any(set(values) != set(range(4))
                                for values in grouped.values()):
        raise RuntimeError("replicate diagnostic requires four complete grids")
    differences = [
        abs(left - right)
        for values in grouped.values()
        for left, right in itertools.combinations(values.values(), 2)
    ]
    return float(np.quantile(differences, .9))


def replication_result(base: list[dict], rows: list[dict]) -> dict:
    base_discovery = [row for row in base if row["stage"] == "discovery"]
    base_idle = [row for row in base if row["stage"] == "idle"]
    if len(base) != 111 or len(base_discovery) != 90 or len(base_idle) != 3:
        raise RuntimeError("replication requires the complete original grid")
    discovery, training_idle, active, held_idle = _replication_grid(rows)
    training = base_discovery + discovery
    idle = base_idle + training_idle
    fit = rational_fit(training, idle)

    halves = [rational_fit(
        [row for row in training if int(row["replicate"]) % 2 == parity],
        idle,
    ) for parity in (0, 1)]
    stability = abs(halves[0]["beta_s_per_decode_token"]
                    - halves[1]["beta_s_per_decode_token"]) \
        / fit["beta_s_per_decode_token"]
    replicate_p90 = _replicate_p90(training)

    held = active + held_idle
    errors = [abs(row["power_mean_w"] - predict(row, fit)) for row in held]
    family_mae = {
        name: statistics.fmean(
            abs(row["power_mean_w"] - predict(row, fit))
            for row in held if row["family"] == name)
        for name in sorted({row["family"] for row in held})
    }
    mean = statistics.fmean(row["power_mean_w"] for row in held)
    ss_res = sum((row["power_mean_w"] - predict(row, fit)) ** 2
                 for row in held)
    ss_tot = sum((row["power_mean_w"] - mean) ** 2 for row in held)
    if ss_tot <= 0:
        raise RuntimeError("replication holdout has no power range")
    cached = int(sum(row["cached_prompt_tokens"] for row in base + rows))
    validation = {
        "design": "prospective_active_plus_idle_envelope",
        "training_cells": len(training),
        "training_idle_cells": len(idle),
        "holdout_cells": len(held),
        "holdout_active_cells": len(active),
        "holdout_idle_cells": len(held_idle),
        "holdout_power_range_w": [
            float(min(row["power_mean_w"] for row in held)),
            float(max(row["power_mean_w"] for row in held)),
        ],
        "holdout_mae_w": float(statistics.fmean(errors)),
        "holdout_p90_abs_error_w": float(np.quantile(errors, .9)),
        "holdout_r2": float(1 - ss_res / ss_tot),
        "family_mae_w": {name: float(value)
                         for name, value in family_mae.items()},
        "active_only_diagnostic": prediction_report(active, fit),
        "beta_split_relative_difference": float(stability),
        "beta_split_values": {
            "even_replicates_0_2": halves[0]["beta_s_per_decode_token"],
            "odd_replicates_1_3": halves[1]["beta_s_per_decode_token"],
        },
        "replicate_p90_difference_w": replicate_p90,
        "cached_prompt_tokens": cached,
        "excluded_prior_confirmation_cells": sum(
            row["stage"] == "confirmation" for row in base),
    }
    validation["gates"] = {
        "holdout_mae_le_5w": bool(validation["holdout_mae_w"] <= 5),
        "holdout_p90_le_10w": bool(
            validation["holdout_p90_abs_error_w"] <= 10),
        "holdout_r2_ge_0p95": bool(validation["holdout_r2"] >= .95),
        "each_family_mae_le_8w": bool(max(family_mae.values()) <= 8),
        "beta_split_difference_le_20pct": bool(stability <= .2),
        "replicate_p90_difference_le_8w": bool(replicate_p90 <= 8),
        "zero_cached_prompt_tokens": bool(cached == 0),
    }
    validation["passed"] = all(validation["gates"].values())
    return {
        "schema": REPLICATION_SCHEMA,
        "status": "calibrated" if validation["passed"] else "holdout_failed",
        "fit": fit,
        "max_ell": MAX_ELL,
        "power_curve": power_curve(fit),
        "validation": validation,
    }


def followup_result(base: list[dict], rows: list[dict]) -> dict:
    calibration = [row for row in rows if row["stage"] == "targeted_calibration"]
    held = [row for row in rows if row["stage"] == "targeted_validation"]
    expected = {3: 3, 6: 3, 12: 3}
    for selected in (calibration, held):
        if {c: sum(row["concurrency"] == c for row in selected) for c in expected} != expected:
            raise RuntimeError("targeted follow-up requires three independent reps per concurrency")
    fit = rational_fit([row for row in base if row["stage"] == "discovery"] + calibration,
                       [row for row in base + rows if row["stage"] == "idle"])
    original = validate(base, fit)
    targeted = prediction_report(held, fit)
    gates = {"original_holdout_passed": bool(original["passed"]),
             "targeted_mae_le_5w": bool(targeted["mae_w"] <= 5),
             "targeted_p90_le_10w": bool(targeted["p90_abs_error_w"] <= 10),
             "targeted_r2_ge_0p95": bool(targeted["r2"] >= .95),
             "each_targeted_concurrency_mae_le_8w": bool(
                 max(targeted["concurrency_mae_w"].values()) <= 8),
             "zero_cached_prompt_tokens": bool(sum(
                 row["cached_prompt_tokens"] for row in rows) == 0)}
    passed = all(gates.values())
    return {"schema": "queue-haul-rational-power-followup-v1",
            "status": "calibrated" if passed else "holdout_failed", "fit": fit,
            "max_ell": MAX_ELL, "power_curve": power_curve(fit),
            "validation": {"original_v5": original, "targeted": targeted,
                           "gates": gates, "passed": passed}}


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replication_base(path: Path, model: str,
                     hardware: str) -> tuple[list[dict], dict, dict]:
    path = path.resolve()
    metadata_path = path / "metadata.json"
    cells_path = path / "cells.jsonl"
    fit_path = path / "fit.json"
    metadata = json.loads(metadata_path.read_text())
    prior = json.loads(fit_path.read_text())
    expected_name, expected_limit = GPU_CONTRACTS[hardware]
    base_gpu = metadata.get("gpu")
    try:
        limit_matches = math.isclose(
            float(base_gpu.get("power_limit_w", math.nan)),
            expected_limit, abs_tol=.01)
    except (AttributeError, TypeError, ValueError):
        limit_matches = False
    gpu_matches = isinstance(base_gpu, dict) \
        and base_gpu.get("name") == expected_name \
        and isinstance(base_gpu.get("uuid"), str) \
        and bool(base_gpu["uuid"]) \
        and limit_matches
    if metadata.get("model") != model \
            or str(metadata.get("hardware", "")).lower() != hardware \
            or metadata.get("revision") != testbed.model_spec(model).revision \
            or not gpu_matches \
            or not isinstance(metadata.get("seed"), int) \
            or not isinstance(metadata.get("git_sha"), str) \
            or len(metadata["git_sha"]) != 40 \
            or prior.get("schema") != "queue-haul-rational-power-fit-v1" \
            or prior.get("status") != "holdout_failed" \
            or prior.get("validation", {}).get("passed") is not False:
        raise RuntimeError(
            "replication base must be the matching complete failed holdout")
    rows = complete_rows(path, int(metadata["seed"]))
    if any(row["cached_prompt_tokens"] for row in rows):
        raise RuntimeError("replication base contains cached prompt tokens")
    evidence = {
        "path": str(path),
        "metadata_sha256": _sha256(metadata_path),
        "cells_sha256": _sha256(cells_path),
        "fit_sha256": _sha256(fit_path),
        "model_revision": metadata["revision"],
        "gpu": base_gpu,
        "base_git_sha": metadata["git_sha"],
        "prior_schema": prior["schema"],
        "prior_status": prior["status"],
    }
    return rows, metadata, evidence


def refit(args) -> None:
    rows = complete_rows(args.out, args.seed)
    fit_path = args.out / "fit.json"
    provisional = args.out / "fit-exponential-provisional.json"
    if provisional.exists():
        raise RuntimeError("provisional exponential fit is already archived")
    with provisional.open("xb") as handle:
        payload = fit_path.read_bytes() if fit_path.is_file() else (
            json.dumps(exponential_fit(rows), indent=2) + "\n").encode()
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    result = fit_result(rows)
    temporary = fit_path.with_suffix(".rational.tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, fit_path)
    if not result["validation"]["passed"]:
        raise RuntimeError("rational power calibration holdout gates failed")


def validate_gpu(hardware: str = "h100") -> dict:
    if hardware not in GPU_CONTRACTS:
        raise ValueError(f"unsupported power hardware: {hardware}")
    expected_name, expected_limit = GPU_CONTRACTS[hardware]
    fields = "name,uuid,power.limit"
    lines = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}",
                                     "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one {expected_name} at {expected_limit:g} W, got {lines}")
    name, uuid, limit = map(str.strip, lines[0].split(","))
    if name != expected_name or not math.isclose(
            float(limit), expected_limit, abs_tol=.01):
        raise RuntimeError(
            f"expected one {expected_name} at {expected_limit:g} W, got {lines}")
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


def server_command(args) -> list[str]:
    spec = testbed.model_spec(args.model)
    model_args = testbed.model_vllm_args(testbed.Config(model=args.model))
    return [args.vllm, "serve", args.model, "--revision", spec.revision,
            "--host", args.host, "--port", str(args.port),
            "--served-model-name", args.model, "--tensor-parallel-size", "1",
            "--max-model-len", "32768", "--max-num-seqs", "256",
            "--max-num-batched-tokens", str(spec.batched_tokens),
            "--dtype", "bfloat16", "--kv-cache-dtype", "auto",
            "--block-size", "16", "--enable-chunked-prefill",
            "--gpu-memory-utilization", ".9", "--no-enable-prefix-caching",
            *model_args]


def validate_resume(args, gpu: dict, plan: list[Cell]) -> list[dict]:
    if not args.expected_sha or len(args.expected_sha) != 40:
        raise ValueError("resume requires the full original --expected-sha")
    metadata = json.loads((args.out / "metadata.json").read_text())
    if str(metadata.get("hardware", "h100")).lower() != args.hardware:
        raise RuntimeError("resume metadata mismatch for hardware")
    expected = {"gpu": gpu, "git_sha": args.expected_sha,
                "minimum_window_s": args.window_s,
                "warmup": "one complete batch", "cooldown_s": args.cooldown_s,
                "seed": args.seed}
    if getattr(args, "replication_base", None):
        _rows, _metadata, evidence = replication_base(
            args.replication_base, args.model, args.hardware)
        expected["replication_base"] = evidence
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
    discard = set(args.discard_orphan_sequences)
    if any(sequence < len(rows) or sequence >= len(plan) for sequence in discard):
        raise RuntimeError("orphan discard sequence is not in the uncommitted suffix")
    discard_labels = {cell_label(sequence, plan[sequence]) for sequence in discard}
    request_path = args.out / "requests.jsonl"
    raw_requests = request_path.read_text().splitlines()
    kept_requests = []
    requests = {label: [] for label in valid_labels}
    for line in raw_requests:
        request_row = json.loads(line)
        if request_row.get("cell") in discard_labels:
            continue
        if request_row.get("cell") not in requests:
            raise RuntimeError("request evidence falls outside committed prefix")
        kept_requests.append(line)
        requests[request_row["cell"]].append(request_row)
    for label, row in valid_labels.items():
        validate_request_evidence(label, row, requests[label])
    discard_paths = {args.out / "power" / f"{cell_label(sequence, plan[sequence])}.csv"
                     for sequence in discard}
    artifacts = set((args.out / "power").glob("*.csv"))
    if artifacts != valid_power | discard_paths or not all(path.exists() for path in discard_paths):
        raise RuntimeError("power artifacts do not equal committed prefix plus explicit orphans")
    if len(kept_requests) != len(raw_requests):
        request_path.write_text("".join(f"{line}\n" for line in kept_requests))
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
    gpu = validate_gpu(args.hardware)
    extending = bool(getattr(args, "replication_base", None))
    plan = (replication_cells(args.seed) if extending else
            followup_cells(args.seed) if args.followup_base else
            cells(args.seed))
    server_cmd = server_command(args)
    base_rows = base_metadata = base_evidence = None
    if extending:
        base_rows, base_metadata, base_evidence = replication_base(
            args.replication_base, args.model, args.hardware)
        if gpu != base_metadata["gpu"]:
            raise RuntimeError(
                "replication extension requires the same physical GPU")
    if args.followup_base:
        base_metadata = json.loads((args.followup_base / "metadata.json").read_text())
        if gpu != base_metadata["gpu"]:
            raise RuntimeError("targeted follow-up requires the same v5 GPU")
        if args.resume:
            raise ValueError("targeted follow-up resume is not implemented")
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
                    "model": args.model,
                    "revision": testbed.model_spec(args.model).revision,
                    "hardware": args.hardware, "optimized_runtime": True,
                    **({"optimized_h100": True} if args.hardware == "h100" else {}),
                    "server_command": server_cmd,
                    "minimum_window_s": args.window_s, "warmup": "one complete batch",
                    "cooldown_s": args.cooldown_s, "seed": args.seed}
        if args.followup_base:
            metadata["followup_base"] = str(args.followup_base)
        if extending:
            metadata["replication_base"] = base_evidence
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows, log_path = [], args.out / "server.log"
    server_log = log_path.open("x")
    server = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        base_url = f"http://{args.host}:{args.port}"
        wait_ready(base_url, server)
        for sequence in range(len(rows), len(plan)):
            cell = plan[sequence]
            row = run_cell(cell, base_url, args.out, args.window_s,
                           args.cooldown_s, sequence, args.model)
            rows.append(row)
            append_jsonl(args.out / "cells.jsonl", [row])
            print(json.dumps(row), flush=True)
        testbed.validate_optimized_runtime(
            " ".join(map(str, server_cmd)), log_path.read_text(errors="replace"))
        result = (replication_result(base_rows, rows) if extending else
                  followup_result(complete_rows(args.followup_base, args.seed), rows)
                  if args.followup_base else fit_result(rows))
        if extending:
            result["base_evidence"] = base_evidence
        with (args.out / "fit.json").open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
            handle.flush(); os.fsync(handle.fileno())
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
    parser.add_argument("--model", choices=tuple(testbed.MODEL_SPECS), required=True)
    parser.add_argument("--hardware", choices=tuple(GPU_CONTRACTS), default="h100")
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
    parser.add_argument("--followup-base", type=Path)
    parser.add_argument("--replication-base", type=Path)
    args = parser.parse_args(argv)
    if min(args.window_s, args.cooldown_s) <= 0:
        raise ValueError("cell durations must be positive")
    if args.followup_base and args.replication_base:
        raise ValueError(
            "--followup-base and --replication-base are mutually exclusive")
    if args.refit_only and (args.followup_base or args.replication_base):
        raise ValueError("follow-up/replication refit-only is not supported")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    refit(parsed) if parsed.refit_only else run(parsed)
