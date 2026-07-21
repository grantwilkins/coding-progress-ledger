"""Self-contained live destination load measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REQUIRED_METRICS = {
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc", "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
}
MODES = ("normal", "emergency", "stable")


def deterministic_tokens(label: str, count: int, vocabulary: int, seed: int) -> list[int]:
    if count < 0 or vocabulary <= 32:
        raise ValueError("invalid deterministic token request")
    state, span = int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little"), vocabulary - 16
    out = []
    for _ in range(count):
        state = (6364136223846793005 * state + 1442695040888963407) % 2**64
        out.append(16 + state % span)
    return out


def poisson_schedule(rate: float, count: int, seed: int) -> tuple[float, ...]:
    if rate <= 0 or count < 1:
        raise ValueError("arrival rate and count must be positive")
    rng, elapsed, out = random.Random(seed), 0.0, []
    for _ in range(count):
        elapsed += rng.expovariate(rate)
        out.append(elapsed)
    return tuple(out)


@dataclass
class Session:
    session_id: str
    prefix_tokens: int
    append_tokens: int
    output_tokens: int
    vocabulary: int
    seed: int
    history: list[int] = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        if min(self.prefix_tokens, self.append_tokens, self.output_tokens) < 1:
            raise ValueError("session token counts must be positive")
        self.history = deterministic_tokens(f"{self.session_id}:prefix", self.prefix_tokens,
                                            self.vocabulary, self.seed)

    def prompt(self, index: int) -> tuple[list[int], int]:
        added = deterministic_tokens(f"{self.session_id}:{index}:input", self.append_tokens,
                                     self.vocabulary, self.seed)
        forced = deterministic_tokens(f"{self.session_id}:{index}:output", 1,
                                      self.vocabulary, self.seed)[0]
        return self.history + added, forced

    def commit(self, prompt: list[int], forced: int) -> None:
        self.history = prompt + [forced] * self.output_tokens


def parse_metrics(text: str) -> dict[str, float]:
    grouped = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.split(" ", 1)
            grouped.setdefault(name.split("{", 1)[0], []).append(float(value))
        except ValueError:
            continue
    return {key: sum(values) / len(values) if key == "vllm:gpu_cache_usage_perc"
            else sum(values) for key, values in grouped.items()}


class MetricsSampler:
    def __init__(self, host: str, port: int, path: Path, period_s: float = .25):
        self.url, self.path, self.period_s = f"http://{host}:{port}/metrics", path, period_s
        self.stop, self.rows, self.error = threading.Event(), [], None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                with urllib.request.urlopen(self.url, timeout=5) as response:
                    metrics = parse_metrics(response.read().decode())
                missing = REQUIRED_METRICS - metrics.keys()
                if missing:
                    raise RuntimeError(f"missing destination metrics: {sorted(missing)}")
                self.rows.append({"monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), **metrics})
                self.stop.wait(self.period_s)
        except Exception as exc:
            self.error = exc

    def close(self):
        self.stop.set(); self.thread.join(10)
        if self.thread.is_alive() or self.error or not self.rows:
            raise RuntimeError("destination metrics sampler failed") from self.error
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)


def _completion(host: str, port: int, model: str, prompt: list[int], output_tokens: int,
                forced: int, timeout_s: float) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": output_tokens,
                       "ignore_eos": True, "temperature": 0, "allowed_token_ids": [forced],
                       "stream": True, "stream_options": {"include_usage": True}})
    start, first, usage, chunks = time.monotonic_ns(), None, {}, []
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    connection.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode(errors="ignore"); connection.close()
        return {"status": response.status, "error": error, "start_ns": start,
                "end_ns": time.monotonic_ns()}
    while line := response.readline():
        if not line.strip().startswith(b"data:"):
            continue
        now, data = time.monotonic_ns(), line.strip()[5:].strip()
        if data == b"[DONE]":
            break
        item = json.loads(data); usage = item.get("usage") or usage
        if item.get("choices"):
            first = first or now; chunks.append(now)
    connection.close(); end = time.monotonic_ns()
    prompt_tokens, completion_tokens = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return {"status": response.status, "error": "", "start_ns": start, "first_ns": first or end,
            "end_ns": end, "prompt_tokens": prompt_tokens, "output_tokens": completion_tokens,
            "cached_tokens": cached, "ttft_s": ((first or end) - start) / 1e9,
            "mean_tpot_s": (end - (first or end)) / 1e9 / max(1, completion_tokens - 1)}


def drive(host: str, port: int, model: str, sessions: list[Session], rate: float,
          count: int, seed: int, timeout_s: float = 720) -> list[dict]:
    schedule, epoch = poisson_schedule(rate, count, seed), time.monotonic()
    def one(index):
        scheduled = epoch + schedule[index]; time.sleep(max(0, scheduled - time.monotonic()))
        session = sessions[index % len(sessions)]
        with session.lock:
            prompt, forced = session.prompt(index)
            row = _completion(host, port, model, prompt, session.output_tokens, forced, timeout_s)
            row.update({"request_index": index, "session_id": session.session_id,
                        "scheduled_ns": int(scheduled * 1e9), "input_tokens": session.append_tokens,
                        "planned_output_tokens": session.output_tokens,
                        "prompt_sha256": hashlib.sha256(bytes(np.asarray(prompt, dtype=np.uint32))).hexdigest()})
            if row["status"] == 200 and row["output_tokens"] == session.output_tokens:
                session.commit(prompt, forced)
            return row
    with ThreadPoolExecutor(max_workers=min(256, count)) as pool:
        return list(pool.map(one, range(count)))


def offered_work(rows: list[dict], duration_s: float) -> tuple[float, float]:
    if duration_s <= 0:
        raise ValueError("measurement duration must be positive")
    return (sum(int(r["input_tokens"]) for r in rows) / duration_s,
            sum(int(r["planned_output_tokens"]) for r in rows) / duration_s)


def anchor_gate(rows: list[dict], expected: dict[tuple[str, int], float], limit: float = .15) -> None:
    observed = {(r["metric"], int(r["context_tokens"])): float(r["tokens_per_s"]) for r in rows}
    if observed.keys() != expected.keys() or any(abs(observed[key] / value - 1) > limit + 1e-12
                                                  for key, value in expected.items()):
        raise ValueError("service anchor drift exceeds the frozen limit")


def manifest_sessions(bundle: dict, job_class: str, split: str, vocabulary: int,
                      seed: int) -> list[Session]:
    ids = set(bundle["manifest"]["splits"][job_class][split])
    grouped = {}
    for row in bundle["traces"]:
        if row["session_id"] in ids and not row.get("reset") and int(row["input_tokens_total"]) <= 24576:
            grouped.setdefault(row["session_id"], []).append(row)
    if grouped.keys() != ids:
        raise ValueError(f"missing usable {job_class} {split} session shapes")
    sessions = []
    for session_id, rows in sorted(grouped.items()):
        row = max(rows, key=lambda r: (int(r["input_tokens_total"]), int(r["turn"])))
        total = int(row["input_tokens_total"])
        appended = min(int(row["newly_append_tokens"]), max(1, total // 4))
        sessions.append(Session(session_id, max(1, total - appended), appended,
                                int(row["output_tokens"]), vocabulary, seed))
    return sessions


def find_boundaries(probe, resolution: float = .05) -> dict[str, tuple[float, float]]:
    cache = {}
    def at(radius):
        cache.setdefault(radius, probe(radius))
        return cache[radius]
    high = .5
    while at(high)["stable"] and high < 4:
        high *= 2
    if at(high)["stable"]:
        raise ValueError("stable boundary is unbracketed")
    out = {}
    for mode in MODES:
        lo, hi = 0.0, high
        while hi - lo > resolution * max(hi, 1):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if at(mid)[mode] else (lo, mid)
        out[mode] = (lo, hi)
    return out


def queue_drift_upper(rows: list[dict], block_s: float = 30) -> float:
    if len(rows) < 2:
        raise ValueError("queue drift needs sampled metrics")
    t0 = int(rows[0]["monotonic_ns"]); points = [((int(r["monotonic_ns"]) - t0) / 1e9,
                                                   float(r["vllm:num_requests_waiting"])) for r in rows]
    points = [p for p in points if p[0] >= points[-1][0] / 3]
    slopes = []
    for block in range(max(1, math.ceil((points[-1][0] - points[0][0]) / block_s))):
        selected = [p for p in points if block * block_s <= p[0] - points[0][0] < (block + 1) * block_s]
        if len(selected) >= 2 and np.ptp([p[0] for p in selected]) > 0:
            slopes.append(float(np.polyfit(*zip(*selected), 1)[0]))
    if not slopes:
        raise ValueError("queue drift lacks complete blocks")
    return float(np.mean(slopes) + (1.645 * np.std(slopes, ddof=1) / math.sqrt(len(slopes)) if len(slopes) > 1 else 0))


def classify(requests: list[dict], metrics: list[dict], drained: bool,
             slos: dict) -> dict[str, bool]:
    complete = bool(requests) and all(not r.get("error") and r.get("status") == 200 and
                                      r.get("output_tokens") == r.get("planned_output_tokens") for r in requests)
    ttft = np.quantile([r["ttft_s"] for r in requests], .9) if complete else math.inf
    tpot = np.quantile([r["mean_tpot_s"] for r in requests], .9) if complete else math.inf
    result = {mode: bool(complete and ttft <= policy["p90_ttft_s"] and
                         tpot <= policy["p90_mean_tpot_s"]) for mode, policy in slos.items()}
    result["stable"] = bool(complete and drained and queue_drift_upper(metrics) <= 1e-12)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from destination_campaign import validate_plan
    validate_plan(json.loads(args.plan.read_text()))
    raise RuntimeError("live campaign orchestration is completed in the migration integration stage")


if __name__ == "__main__":
    main()
