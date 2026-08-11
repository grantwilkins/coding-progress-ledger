"""Measure steady inference load and GPU power across fixed request rates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def validate_gpu(name: str, power_limit_w: float) -> None:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if rows != [f"{name}, {power_limit_w:.2f}"]:
        raise RuntimeError(f"expected one {name} at {power_limit_w:.2f} W, got {rows}")


def calibration_prompt(words: int) -> str:
    return "power calibration " + "x " * words


def power(path: Path, stop: threading.Event, interval: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("monotonic_ns", "wall_ns", "power_w", "utilization_pct", "memory_mib"))
        while not stop.is_set():
            row = subprocess.check_output([
                "nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ], text=True).strip().split(",")
            writer.writerow((time.monotonic_ns(), time.time_ns(), *(float(x) for x in row)))
            handle.flush()
            stop.wait(interval)


def request(url: str, prompt: str, output_tokens: int, request_id: str) -> dict:
    nonce = hashlib.sha256(request_id.encode()).hexdigest()
    body = json.dumps({"model": "openai/gpt-oss-20b", "prompt": f"{nonce} {prompt}",
                       "max_tokens": output_tokens, "ignore_eos": True,
                       "temperature": 0}).encode()
    started = time.monotonic_ns()
    response = urlopen(Request(url, body, {"Content-Type": "application/json"}), timeout=600)
    result = json.load(response)
    result["start_ns"], result["end_ns"] = started, time.monotonic_ns()
    return result


def reduce(out: Path, prefill_tps: float, decode_tps: float,
           idle_power_w: float, curve_max_rate: float) -> None:
    rows = json.loads((out / "levels.json").read_text())
    for row in rows:
        label = f"{row['rate_rps']:g}".replace(".", "p")
        with (out / f"power-r{label}.csv").open() as handle:
            watts = sorted(float(sample["power_w"]) for sample in csv.DictReader(handle)
                           if row["start_ns"] <= int(sample["monotonic_ns"]) < row["end_ns"])
        row["ell"] = row["prompt_tokens"] / row["window_s"] / prefill_tps \
            + row["output_tokens"] / row["window_s"] / decode_tps
        row["power_p50_w"] = statistics.median(watts)
        row["power_p95_w"] = watts[math.ceil(.95 * len(watts)) - 1]
    (out / "reduced.json").write_text(json.dumps(rows, indent=2) + "\n")
    points = [[0.0, idle_power_w]]
    for row in rows:
        if row["rate_rps"] > curve_max_rate:
            continue
        points.append([row["ell"], max(row["power_mean_w"], points[-1][1])])
        while len(points) > 2:
            left = (points[-2][1] - points[-3][1]) / (points[-2][0] - points[-3][0])
            right = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])
            if left >= right:
                break
            points.pop(-2)
    with (out / "power_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("ell", "power_w")); writer.writerows(points)


def level(host: str, port: int, rate: float, window: float, warmup: float,
          prompt: str, output_tokens: int, workers: int, out: Path) -> dict:
    url = f"http://{host}:{port}/v1/completions"
    stop = threading.Event()
    label = f"{rate:g}".replace(".", "p")
    power_path = out / f"power-r{label}.csv"
    sampler = threading.Thread(target=power, args=(power_path, stop, .1))
    sampler.start()
    started = time.monotonic()
    measured_start = int((started + warmup) * 1e9)
    measured_end = measured_start + int(window * 1e9)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for request_id in range(math.ceil(rate * (warmup + window))):
            delay = started + request_id / rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            identity = f"{out}:{rate}:{request_id}"
            futures.append(pool.submit(request, url, prompt, output_tokens, identity))
        completed = [future.result() for future in futures]
    stop.set(); sampler.join()
    rows = [row for row in completed if measured_start <= row["start_ns"] < measured_end]
    usage = [row["usage"] for row in rows]
    seconds = (measured_end - measured_start) / 1e9
    with power_path.open() as handle:
        watts = [float(row["power_w"]) for row in csv.DictReader(handle)
                 if measured_start <= int(row["monotonic_ns"]) < measured_end]
    return {"rate_rps": rate, "window_s": seconds, "requests": len(rows),
            "completed_in_window": sum(row["end_ns"] < measured_end for row in rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in usage),
            "output_tokens": sum(int(row.get("completion_tokens", 0)) for row in usage),
            "cached_prompt_tokens": sum(int((row.get("prompt_tokens_details") or {})
                                            .get("cached_tokens", 0)) for row in usage),
            "power_mean_w": statistics.fmean(watts), "power_samples": len(watts),
            "start_ns": measured_start, "end_ns": measured_end}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=15)
    parser.add_argument("--warmup-s", type=float, default=5)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--prompt-words", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=512)
    parser.add_argument("--expected-gpu", default="NVIDIA H100 NVL")
    parser.add_argument("--expected-power-limit-w", type=float, default=400)
    parser.add_argument("--reduce-only", action="store_true")
    parser.add_argument("--prefill-capacity-tps", type=float)
    parser.add_argument("--decode-capacity-tps", type=float)
    parser.add_argument("--idle-power-w", type=float)
    parser.add_argument("--curve-max-rate", type=float, default=12)
    parser.add_argument("--rates", nargs="+", type=float,
                        default=(.25, .5, .75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 12, 14, 16, 20))
    args = parser.parse_args()
    if args.reduce_only:
        if not all(value is not None and value > 0 for value in
                   (args.prefill_capacity_tps, args.decode_capacity_tps,
                    args.idle_power_w, args.curve_max_rate)):
            raise ValueError("reduction constants must be positive")
        reduce(args.out, args.prefill_capacity_tps, args.decode_capacity_tps,
               args.idle_power_w, args.curve_max_rate)
        return
    if args.window_s <= 0 or args.warmup_s < 0 or args.output_tokens < 1 \
            or args.prompt_words < 1 or args.workers < 1 or args.expected_power_limit_w <= 0 \
            or not args.rates or min(args.rates) <= 0:
        raise ValueError("invalid sweep settings")
    validate_gpu(args.expected_gpu, args.expected_power_limit_w)
    args.out.mkdir(parents=True, exist_ok=False)
    prompt = calibration_prompt(args.prompt_words)
    rows = []
    for rate in args.rates:
        rows.append(level(args.host, args.port, rate, args.window_s,
                          args.warmup_s, prompt, args.output_tokens,
                          args.workers, args.out))
        (args.out / "levels.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
