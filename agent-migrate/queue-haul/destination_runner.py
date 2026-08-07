"""Self-contained live destination load measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import migration_profiler as profiler
import migration_testbed as testbed
from destination_evaluation import service_cache_state


REQUIRED_METRICS = {
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc", "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
}
MODES = ("normal", "emergency", "stable")
FORCED_VOCABULARY = 200000


def retry_call(action, path: Path, attempts: int, delay_s: float):
    attempts, delay_s = max(1, attempts), max(0, delay_s)
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps({"attempt": attempt + 1, "error": str(exc),
                                         "type": type(exc).__name__}) + "\n")
            if attempt + 1 == attempts:
                raise
            time.sleep(delay_s)


def archive_checkpoint(path: Path) -> None:
    path.replace(path.with_name(f"{path.stem}.invalid-{time.time_ns()}{path.suffix}"))


def read_checkpoint(path: Path, required: tuple[str, ...] = (), valid=None) -> dict | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        result = None
    if (result and result.get("status") == "complete" and
            all(key in result for key in required) and (valid is None or valid(result))):
        return result
    archive_checkpoint(path)
    return None


def read_anchor_checkpoint(path: Path, expected: dict, repeats: int) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
        keys = {(row["metric"], int(row["context_tokens"])) for row in rows}
        complete = keys == set(expected) and all(
            sum((row["metric"], int(row["context_tokens"])) == key for row in rows) >= repeats
            for key in keys)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        complete = False
    if complete:
        return rows
    archive_checkpoint(path)
    return None


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


def poisson_window(rate: float, seconds: float, seed: int) -> tuple[float, ...]:
    if min(rate, seconds) <= 0:
        raise ValueError("arrival rate and window must be positive")
    rng, elapsed, out = random.Random(seed), 0.0, []
    while (elapsed := elapsed + rng.expovariate(rate)) <= seconds:
        out.append(elapsed)
    return tuple(out)


def uniform_schedule(rate: float, count: int, _seed: int) -> tuple[float, ...]:
    if rate <= 0 or count < 1:
        raise ValueError("arrival rate and count must be positive")
    return tuple(index / rate for index in range(count))


def anchor_rate(expected: float, tokens: int) -> float:
    return expected / tokens


@dataclass
class Session:
    session_id: str
    prefix_tokens: int
    append_tokens: int
    output_tokens: int
    vocabulary: int
    seed: int
    history: list[int] = field(init=False)

    def __post_init__(self):
        if min(self.prefix_tokens, self.append_tokens, self.output_tokens) < 1:
            raise ValueError("session token counts must be positive")
        self.history = deterministic_tokens(f"{self.session_id}:prefix", self.prefix_tokens,
                                            self.vocabulary, self.seed)

    def prompt(self, index: int) -> tuple[list[int], int]:
        added = deterministic_tokens(f"{self.session_id}:{index}:input", self.append_tokens,
                                     self.vocabulary, self.seed)
        forced = deterministic_tokens(f"{self.session_id}:{index}:output", 1,
                                      min(self.vocabulary, FORCED_VOCABULARY), self.seed)[0]
        return self.history[:self.prefix_tokens] + added, forced

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
    metrics = {key: sum(values) / len(values) if key in
               ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
               else sum(values) for key, values in grouped.items()}
    if "vllm:gpu_cache_usage_perc" not in metrics and "vllm:kv_cache_usage_perc" in metrics:
        metrics["vllm:gpu_cache_usage_perc"] = metrics["vllm:kv_cache_usage_perc"]
    return metrics


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


def completion_payload(model: str, prompt: list[int], output_tokens: int,
                       forced: int, bypass_lmcache: bool = False) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
               "ignore_eos": True, "temperature": 0, "allowed_token_ids": [forced],
               "stream": True, "stream_options": {"include_usage": True}}
    if bypass_lmcache:
        payload["kv_transfer_params"] = {
            "qh_bypass_lmcache": True, "lmcache.skip_save": True,
        }
    return payload


def _completion(host: str, port: int, model: str, prompt: list[int], output_tokens: int,
                forced: int, timeout_s: float, bypass_lmcache: bool = False) -> dict:
    body = json.dumps(completion_payload(
        model, prompt, output_tokens, forced, bypass_lmcache,
    ))
    start, first, usage, chunks, done = time.monotonic_ns(), None, {}, [], False
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
            done = True
            break
        item = json.loads(data); usage = item.get("usage") or usage
        if item.get("choices"):
            first = first or now; chunks.append(now)
    connection.close(); end = time.monotonic_ns()
    prompt_tokens, completion_tokens = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return {"status": response.status, "error": "", "start_ns": start, "first_ns": first or end,
            "end_ns": end, "prompt_tokens": prompt_tokens, "output_tokens": completion_tokens,
            "cached_tokens": cached, "done": done,
            "ttft_s": ((first or end) - start) / 1e9,
            "mean_tpot_s": (end - (first or end)) / 1e9 / max(1, completion_tokens - 1)}


def agentic_messages(session: Session, index: int) -> list[dict]:
    return [
        {"role": "system", "content": "You are a tool-using coding agent."},
        {"role": "user", "content":
         f"Session {session.session_id} turn {index}; analyze tool results."
         + " x" * session.append_tokens},
    ]


def issue_chat(host: str, port: int, model: str, session: Session, index: int,
               scheduled_ns: int, timeout_s: float,
               bypass_lmcache: bool = False) -> dict:
    messages = agentic_messages(session, index)
    result, error = profiler.stream_chat(
        SimpleNamespace(host=host, model=model), port, messages,
        session.output_tokens, profiler.messages_hash(messages), timeout_s,
        bypass_lmcache,
    )
    return {
        "status": result.status_code,
        "error": "" if result.status_code == 200 else error,
        "start_ns": result.start_ns, "first_ns": result.first_byte_ns,
        "end_ns": result.end_ns, "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens, "cached_tokens": result.cached_tokens,
        "done": result.status_code == 200, "request_index": index,
        "session_id": session.session_id, "scheduled_ns": scheduled_ns,
        "input_tokens": session.append_tokens,
        "planned_prompt_tokens": result.prompt_tokens,
        "planned_output_tokens": session.output_tokens,
        "ttft_s": (result.first_byte_ns - result.start_ns) * 1e-9,
        "mean_tpot_s": (result.end_ns - result.first_byte_ns) * 1e-9
            / max(1, result.output_tokens - 1),
    }


def issue(host: str, port: int, model: str, session: Session, index: int,
          scheduled_ns: int, timeout_s: float,
          bypass_lmcache: bool = False) -> dict:
    prompt, forced = session.prompt(index)
    row = _completion(host, port, model, prompt, session.output_tokens, forced, timeout_s,
                      bypass_lmcache)
    row.update({"request_index": index, "session_id": session.session_id,
                "scheduled_ns": scheduled_ns, "input_tokens": session.append_tokens,
                "planned_prompt_tokens": len(prompt),
                "planned_output_tokens": session.output_tokens,
                "prompt_sha256": hashlib.sha256(bytes(np.asarray(prompt, dtype=np.uint32))).hexdigest()})
    return row


def drive(host: str, port: int, model: str, sessions: list[Session], rate: float,
          count: int, seed: int, timeout_s: float = 720,
          scheduler=poisson_schedule, window_s: float | None = None,
          stop: threading.Event | None = None,
          bypass_lmcache: bool = False, request=issue) -> list[dict]:
    schedule = poisson_window(rate, window_s, seed) if window_s else scheduler(rate, count, seed)
    count, epoch = len(schedule), time.monotonic()
    def one(index):
        scheduled = epoch + schedule[index]
        delay = max(0, scheduled - time.monotonic())
        if stop and stop.wait(delay):
            return None
        if not stop:
            time.sleep(delay)
        return request(host, port, model, sessions[index % len(sessions)], index,
                     int(scheduled * 1e9), timeout_s, bypass_lmcache)
    if not count:
        return []
    with ThreadPoolExecutor(max_workers=min(256, count)) as pool:
        return [row for row in pool.map(one, range(count)) if row is not None]


def prewarm(host: str, port: int, model: str, sessions: list[Session], timeout_s=720,
            bypass_lmcache: bool = False) -> list[dict]:
    rows = []
    for session in sessions:
        prompt, forced = session.prompt(-1)
        row = _completion(host, port, model, prompt[:session.prefix_tokens], 1, forced,
                          timeout_s, bypass_lmcache)
        if row["status"] != 200 or row["error"] or not row.get("done") \
                or row.get("prompt_tokens") != session.prefix_tokens \
                or row.get("output_tokens") != 1:
            raise RuntimeError(f"failed to prewarm {session.session_id}")
        rows.append(row)
    return rows


class DestinationLoad:
    def __init__(self, host: str, port: int, model: str, sessions: list[Session],
                 target_rho: float, prefill_rate: float, decode_rate: float,
                 root: Path, seed: int, chunk_s: float = 15, normal_bound: float = 1,
                 timeout_s: float = 720, rps: float | None = None,
                 max_inflight: int = 0, prewarm_timeout_s: float = 300,
                 bypass_lmcache: bool = False, chat: bool = False,
                 arrival_schedule: tuple[float, ...] | None = None,
                 warmup_s: float = 0, measurement_s: float = 0):
        work = np.mean([s.append_tokens / prefill_rate + s.output_tokens / decode_rate
                        for s in sessions])
        if min(target_rho, work, normal_bound) <= 0:
            raise ValueError("destination load needs positive target and work")
        if rps is not None and rps <= 0:
            raise ValueError("open-loop arrival rate must be positive")
        if (rps is None) != (max_inflight <= 0):
            raise ValueError("open-loop mode needs both rps and max_inflight")
        if arrival_schedule is not None and (
                not rps or warmup_s <= 0 or measurement_s <= 0
                or tuple(sorted(arrival_schedule)) != arrival_schedule
                or any(value < 0 or value >= warmup_s + measurement_s
                       for value in arrival_schedule)):
            raise ValueError("invalid deterministic arrival trace")
        self.host, self.port, self.model, self.sessions = host, port, model, sessions
        self.target, self.prefill_rate, self.decode_rate = target_rho, prefill_rate, decode_rate
        self.normal_bound = normal_bound
        self.root, self.seed, self.chunk_s, self.timeout_s = root, seed, chunk_s, timeout_s
        self.prewarm_timeout_s = prewarm_timeout_s
        self.bypass_lmcache = bypass_lmcache
        self.request = issue_chat if chat else issue
        self.rate = rps if rps else target_rho * normal_bound / work
        self.work, self.max_inflight = float(work), max_inflight
        self.arrival_schedule = arrival_schedule
        self.warmup_s, self.measurement_s = warmup_s, measurement_s
        self.epoch, self.queue_at_start = None, None
        self.stop, self.rows = threading.Event(), []
        self.admit, self.blocked_arrivals = threading.Event(), 0
        self.admit.set()
        self.sampler = MetricsSampler(host, port, root / "engine.csv")
        self.thread, self.failure = threading.Thread(target=self._run, daemon=True), None
        self.achieved = None

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.request is issue_chat:
            row = self.request(self.host, self.port, self.model, self.sessions[0],
                               -1, time.time_ns(), self.prewarm_timeout_s,
                               self.bypass_lmcache)
            if row["status"] != 200 or row["error"]:
                raise RuntimeError("failed to prewarm agentic chat load")
        elif self.bypass_lmcache:
            prewarm(self.host, self.port, self.model, self.sessions,
                    self.prewarm_timeout_s, True)
        else:
            prewarm(self.host, self.port, self.model, self.sessions,
                    self.prewarm_timeout_s)
        self.sampler.start(); self.thread.start()

    def pause(self):
        """Stop new arrivals; in-flight requests keep draining."""
        self.admit.clear()

    def resume(self):
        self.admit.set()

    def _run(self):
        try:
            self._run_open() if self.max_inflight else self._run_chunked()
        except Exception as exc:
            self.failure = exc

    def _run_chunked(self):
        index = 0
        while not self.stop.is_set():
            count = max(1, math.ceil(self.rate * self.chunk_s))
            self.rows += drive(self.host, self.port, self.model, self.sessions,
                               self.rate, count, self.seed + index, self.timeout_s,
                               stop=self.stop, bypass_lmcache=getattr(self, "bypass_lmcache", False),
                               request=getattr(self, "request", issue))
            index += 1

    def _run_open(self):
        """Open-loop arrivals never wait on completions."""
        rng = random.Random(self.seed)
        gate = threading.Semaphore(self.max_inflight)
        lock = threading.Lock()

        def one(index, scheduled_ns):
            try:
                row = getattr(self, "request", issue)(
                    self.host, self.port, self.model,
                    self.sessions[index % len(self.sessions)], index,
                    scheduled_ns, self.timeout_s,
                    getattr(self, "bypass_lmcache", False),
                )
                with lock:
                    self.rows.append(row)
            finally:
                gate.release()

        with ThreadPoolExecutor(max_workers=self.max_inflight) as pool:
            self.epoch = time.monotonic()
            schedule = getattr(self, "arrival_schedule", None)
            if schedule is not None:
                for index, offset in enumerate(schedule):
                    scheduled = self.epoch + offset
                    if self.stop.wait(max(0, scheduled - time.monotonic())):
                        return
                    if time.monotonic() - scheduled > .25 or not gate.acquire(blocking=False):
                        raise RuntimeError("deterministic arrival trace slipped")
                    pool.submit(one, index, time.time_ns())
                return
            index = 0
            while not self.stop.wait(rng.expovariate(self.rate)):
                while not self.admit.is_set():
                    if self.stop.wait(.1):
                        return
                scheduled_ns = time.time_ns()
                if not gate.acquire(blocking=False):
                    self.blocked_arrivals += 1
                    while not gate.acquire(blocking=False):
                        if self.stop.wait(.05):
                            return
                pool.submit(one, index, scheduled_ns)
                index += 1

    def wait_ready(self):
        if getattr(self, "arrival_schedule", None) is not None:
            while self.epoch is None and not self.failure:
                time.sleep(.01)
            deadline = self.epoch + self.warmup_s
            while time.monotonic() < deadline and not self.failure:
                time.sleep(min(.1, deadline - time.monotonic()))
            if self.failure:
                raise RuntimeError("deterministic destination load failed") from self.failure
            self.queue_at_start = self.sampler.rows[-1].get(
                "vllm:num_requests_waiting") if self.sampler.rows else None
            return
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not self.failure:
            if len(self.sampler.rows) > 1 and (self.sampler.rows[-1]["monotonic_ns"] -
                                               self.sampler.rows[0]["monotonic_ns"]) >= 30e9:
                self.achieved = measured_rho(self.sampler.rows, self.prefill_rate,
                                             self.decode_rate, self.normal_bound)
                if load_within_target(self.achieved, self.target):
                    return
            time.sleep(.25)
        raise RuntimeError(
            f"destination rho {self.achieved} misses target {self.target}"
        ) from self.failure

    def wait_deadline(self):
        if getattr(self, "arrival_schedule", None) is None:
            return
        deadline = self.epoch + self.warmup_s + self.measurement_s
        while time.monotonic() < deadline and not self.failure:
            time.sleep(min(.1, deadline - time.monotonic()))
        if self.failure:
            raise RuntimeError("deterministic destination load failed") from self.failure

    def close(self):
        self.stop.set()
        if getattr(self.thread, "ident", 1) is None:
            return
        self.thread.join(self.chunk_s + self.timeout_s + 10); self.sampler.close()
        if self.thread.is_alive() or self.failure:
            raise RuntimeError("destination foreground failed") from self.failure
        (self.root / "requests.json").write_text(json.dumps(self.rows, indent=2) + "\n")

    def summary(self):
        schedule = getattr(self, "arrival_schedule", None)
        warmup = getattr(self, "warmup_s", 0)
        measurement = getattr(self, "measurement_s", 0)
        selected = [self.sessions[index % len(self.sessions)]
                    for index, offset in enumerate(schedule or ())
                    if warmup <= offset < warmup + measurement]
        prefill = sum(row.append_tokens / self.prefill_rate for row in selected) / measurement if selected else 0
        decode = sum(row.output_tokens / self.decode_rate for row in selected) / measurement if selected else 0
        return {
            "target_rho": self.target, "achieved_rho": self.achieved,
            "offered_rho_prefill": prefill, "offered_rho_decode": decode,
            "offered_rho": prefill + decode, "offered_rps": self.rate,
            "max_inflight": self.max_inflight,
            "blocked_arrivals": self.blocked_arrivals,
            "work_per_request_s": self.work, "request_count": len(self.rows),
            "queue_at_start": getattr(self, "queue_at_start", None) if schedule is not None
            else self.sampler.rows[0].get("vllm:num_requests_waiting") if self.sampler.rows else None,
        }

def offered_work(rows: list[dict], duration_s: float) -> tuple[float, float]:
    if duration_s <= 0:
        raise ValueError("measurement duration must be positive")
    return (sum(int(r["input_tokens"]) for r in rows) / duration_s,
            sum(int(r["planned_output_tokens"]) for r in rows) / duration_s)


def measured_rho(rows: list[dict], prefill_rate: float, decode_rate: float,
                 normal_bound: float = 1) -> float:
    if len(rows) < 2 or min(prefill_rate, decode_rate, normal_bound) <= 0:
        raise ValueError("rho needs valid sampled rates and capacity")
    first, last = rows[0], rows[-1]
    seconds = (int(last["monotonic_ns"]) - int(first["monotonic_ns"])) / 1e9
    if seconds < 30:
        raise ValueError("rho gate needs thirty seconds of destination state")
    prompt = float(last["vllm:prompt_tokens_total"] - first["vllm:prompt_tokens_total"]) / seconds
    decode = float(last["vllm:generation_tokens_total"] - first["vllm:generation_tokens_total"]) / seconds
    return (prompt / prefill_rate + decode / decode_rate) / normal_bound

def require_rho(rows: list[dict], target: float, prefill_rate: float,
                decode_rate: float, normal_bound: float = 1,
                tolerance: float = .05) -> float:
    achieved = measured_rho(rows, prefill_rate, decode_rate, normal_bound)
    if not load_within_target(achieved, target, tolerance):
        raise RuntimeError(f"destination rho {achieved:.3f} misses target {target:.3f}")
    return achieved


def load_within_target(achieved: float | None, target: float,
                       tolerance: float = .05) -> bool:
    return achieved is not None and abs(achieved - target) <= tolerance


def anchor_gate(rows: list[dict], expected: dict[tuple[str, int], float], limit: float = .15) -> dict:
    observed = {key: statistics.median(float(r["tokens_per_s"]) for r in rows
                                       if (r["metric"], int(r["context_tokens"])) == key)
                for key in {(r["metric"], int(r["context_tokens"])) for r in rows}}
    if observed.keys() != expected.keys():
        raise ValueError("service anchor checkpoint is incomplete")
    anchors = [{"metric": key[0], "context_tokens": key[1],
                "expected_tokens_per_s": expected[key], "observed_tokens_per_s": observed[key],
                "ratio": observed[key] / expected[key]} for key in sorted(expected)]
    return {"within_limit": all(r["ratio"] >= 1 - limit - 1e-12 for r in anchors),
            "underdelivery_limit": limit, "anchors": anchors}


def apply_anchor_rates(profile: dict, report: dict) -> None:
    for row in report["anchors"]:
        points = profile["cases"]["central"][f'{row["metric"]}_tps']["1"]
        points[:] = sorted([p for p in points if int(p[0]) != row["context_tokens"]]
                           + [[row["context_tokens"], row["observed_tokens_per_s"]]])


def manifest_sessions(bundle: dict, job_class: str, split: str, vocabulary: int,
                      seed: int) -> list[Session]:
    ids = set(bundle["manifest"]["splits"][job_class][split])
    grouped = {}
    for row in bundle["traces"]:
        if row["session_id"] in ids and not row.get("reset") \
                and 256 <= int(row["input_tokens_total"]) <= 24576:
            grouped.setdefault(row["session_id"], []).append(row)
    if set(grouped) != ids:
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
    if resolution <= 0:
        raise ValueError("boundary resolution must be positive")
    cache = {}
    def at(radius):
        cache.setdefault(radius, probe(radius))
        return cache[radius]
    high = .5
    while at(high)["stable"] and high < 4:
        high *= 2
    low = resolution / 2
    low_value, out = at(low), {}
    for mode in MODES:
        if not low_value[mode]:
            out[mode] = (low, low)
            continue
        if at(high)[mode]:
            out[mode] = (high, high)
            continue
        lo, hi = low, high
        while hi - lo > resolution * max(hi, 1):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if at(mid)[mode] else (lo, mid)
        out[mode] = (lo, hi)
    return out


def nest_bounds(bounds: dict[str, float]) -> dict[str, float]:
    bounds = dict(bounds)
    bounds["emergency"] = min(bounds["emergency"], bounds["stable"])
    bounds["normal"] = min(bounds["normal"], bounds["emergency"])
    return bounds


def queue_drift_upper(rows: list[dict], requests=(), block_s: float = 30,
                      samples: int = 2000) -> float:
    if len(rows) < 2:
        raise ValueError("queue drift needs sampled metrics")
    t0 = int(rows[0]["monotonic_ns"]); points = [
        ((int(r["monotonic_ns"]) - t0) / 1e9,
         float(r["vllm:num_requests_waiting"]) + sum(
             int(q.get("scheduled_ns", 0)) <= int(r["monotonic_ns"]) < int(q.get("start_ns", 0))
             for q in requests)) for r in rows]
    points = [p for p in points if p[0] >= points[-1][0] / 3]
    slopes = []
    for block in range(max(1, math.ceil((points[-1][0] - points[0][0]) / block_s))):
        selected = [p for p in points if block * block_s <= p[0] - points[0][0] < (block + 1) * block_s]
        if len(selected) >= 2 and np.ptp([p[0] for p in selected]) > 0:
            slopes.append(float(np.polyfit(*zip(*selected), 1)[0]))
    if not slopes or samples < 1:
        raise ValueError("queue drift lacks complete blocks")
    rng = random.Random(0)
    return float(np.quantile([statistics.mean(rng.choices(slopes, k=len(slopes)))
                              for _ in range(samples)], .95))


def classify(requests: list[dict], metrics: list[dict], drained: bool,
             slos: dict, block_s: float = 30, samples: int = 2000) -> dict[str, bool]:
    complete = bool(requests) and all(not r.get("error") and r.get("status") == 200 and
                                      r.get("output_tokens") == r.get("planned_output_tokens") for r in requests)
    ttft = np.quantile([r["ttft_s"] for r in requests], .9) if complete else math.inf
    tpot = np.quantile([r["mean_tpot_s"] for r in requests], .9) if complete else math.inf
    result = {mode: bool(complete and ttft <= policy["p90_ttft_s"] and
                         tpot <= policy["p90_mean_tpot_s"]) for mode, policy in slos.items()}
    seconds = (int(metrics[-1]["monotonic_ns"]) - int(metrics[0]["monotonic_ns"])) / 1e9
    result["stable"] = bool(complete and drained and
                            queue_drift_upper(metrics, requests, block_s, samples) <= 1 / seconds)
    return result


def profile_rate(profile: dict, metric: str, context: int) -> float:
    points = profile["cases"]["central"][f"{metric}_tps"]["1"]
    if not points[0][0] <= context <= points[-1][0]:
        raise ValueError(f"{metric} context is outside the reused profile")
    return float(np.interp(context, *zip(*points)))


def integrity_preflight(cfg: testbed.Config, plan: dict, smoke: dict,
                        vocabulary: int = 201088) -> dict:
    a = deterministic_tokens("preflight-a", 4096, vocabulary, 0)
    b = deterministic_tokens("preflight-b", 4096, vocabulary, 0)
    forced = deterministic_tokens("preflight-output", 1, vocabulary, 0)[0]
    first = _completion(cfg.host, cfg.sink_port, cfg.model, a, 1, forced, 720)
    same = _completion(cfg.host, cfg.sink_port, cfg.model, a, 1, forced, 720)
    cross = _completion(cfg.host, cfg.sink_port, cfg.model, b, 1, forced, 720)
    tokenized = testbed.http_json(cfg.host, cfg.sink_port, "POST", "/tokenize",
                                 {"model": cfg.model, "prompt": "queue haul"})
    report = {"image_sha256": plan["image_sha256"], "gpu_count": 2,
              "same_session_cache_hit": same["cached_tokens"] > first["cached_tokens"],
              "cross_session_cache_hits": cross["cached_tokens"],
              "tokenizer_ok": int(tokenized.get("count", 0)) > 0,
              "migration_smoke": smoke}
    from destination_campaign import check_gate
    check_gate(report, "preflight")
    return report


def load_inputs(plan_path: Path) -> tuple[dict, dict, dict]:
    from destination_campaign import validate_plan
    plan = json.loads(plan_path.read_text()); validate_plan(plan)
    def load(record):
        path = plan_path.parent / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"campaign input changed: {path.name}")
        return json.loads(path.read_text())
    manifest_path = plan_path.parent / plan["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != plan["manifest"]["sha256"]:
        raise RuntimeError("campaign manifest changed")
    return plan, manifest, load(plan["baseline_profile"])


def measure_anchors(host: str, port: int, model: str, contexts: list[int],
                    vocabulary: int, expected: dict[tuple[str, int], float],
                    root: Path, stack: testbed.Stack, cfg: testbed.Config,
                    repeats: int = 3, hold_s: float = 60) -> list[dict]:
    path = root / "anchors.json"
    checkpoint = read_anchor_checkpoint(path, expected, repeats)
    if checkpoint:
        return checkpoint
    rows = []
    for context in contexts:
        for metric in ("prefill", "decode"):
            tokens, counter = ((context - 1, "vllm:prompt_tokens_total") if metric == "prefill"
                               else (256, "vllm:generation_tokens_total"))
            for repeat in range(repeats):
                testbed.flush_lmcache(stack, cfg)
                sessions = [Session(f"anchor:{metric}:{context}:{repeat}:{i}",
                                    1 if metric == "prefill" else context - 1, 1 if metric == "decode" else context - 1,
                                    1 if metric == "prefill" else 256, vocabulary, repeat)
                            for i in range(32)]
                if metric == "decode":
                    prewarm(host, port, model, sessions)
                cell = root / f"{metric}-{context}-{repeat}"; cell.mkdir(parents=True, exist_ok=True)
                sampler = MetricsSampler(host, port, cell / "engine.csv"); sampler.start()
                try:
                    started = time.monotonic()
                    rate = anchor_rate(expected[metric, context], tokens)
                    requests = drive(host, port, model, sessions, rate,
                                     math.ceil(rate * hold_s), repeat, scheduler=uniform_schedule)
                    time.sleep(max(0, hold_s - (time.monotonic() - started)))
                finally:
                    sampler.close()
                samples = [r for r in sampler.rows
                           if r["monotonic_ns"] <= sampler.rows[0]["monotonic_ns"] + hold_s * 1e9]
                seconds = (samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"]) / 1e9
                if seconds < hold_s * .9 or any(r["status"] != 200 for r in requests):
                    raise RuntimeError("service anchor request failed")
                rows.append({"metric": metric, "context_tokens": context, "run_id": repeat,
                             "tokens_per_s": (samples[-1][counter] - samples[0][counter]) / seconds})
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def service_probe(host: str, port: int, model: str, sessions: list[Session],
                  radius: float, prefill_rate: float, decode_rate: float,
                  hold_s: float, slos: dict, root: Path, seed: int,
                  cache_block_tokens: int, block_s: float = 30,
                  samples: int = 2000) -> dict:
    result_path = root / "result.json"
    checkpoint = read_checkpoint(
        result_path, ("classification", "cache"),
        lambda row: set(MODES) <= set(row["classification"])
        and row["cache"].get("state") == "private_prefix",
    )
    if checkpoint:
        return checkpoint
    work = statistics.mean(s.append_tokens / prefill_rate + s.output_tokens / decode_rate
                           for s in sessions)
    rate, count = radius / work, max(1, math.ceil(radius / work * hold_s))
    root.mkdir(parents=True, exist_ok=True)
    warm = prewarm(host, port, model, sessions)
    sampler = MetricsSampler(host, port, root / "engine.csv"); sampler.start()
    try:
        started = time.monotonic()
        requests = drive(host, port, model, sessions, rate, count, seed, window_s=hold_s)
        time.sleep(max(0, hold_s - (time.monotonic() - started)))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and sampler.rows and any(
                sampler.rows[-1][key] for key in
                ("vllm:num_requests_running", "vllm:num_requests_waiting")):
            time.sleep(.1)
    finally:
        sampler.close()
    metrics = [r for r in sampler.rows
               if r["monotonic_ns"] <= sampler.rows[0]["monotonic_ns"] + hold_s * 1e9]
    drained = not sampler.rows[-1]["vllm:num_requests_running"] \
        and not sampler.rows[-1]["vllm:num_requests_waiting"]
    cache = service_cache_state(requests, cache_block_tokens)
    (root / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    if cache["state"] != "private_prefix":
        result_path.write_text(json.dumps(
            {"status": "invalid", "cache": cache}, indent=2, sort_keys=True
        ) + "\n")
        raise RuntimeError(f"service probe cache state is {cache['state']}")
    result = {"status": "complete", "radius": radius,
              "cache": cache, "prewarm": warm,
              "classification": classify(requests, metrics, drained, slos, block_s, samples),
              "queue_drift_upper": queue_drift_upper(metrics, requests, block_s, samples),
              "offered_prefill_tps": rate * statistics.mean(s.append_tokens for s in sessions),
              "offered_decode_tps": rate * statistics.mean(s.output_tokens for s in sessions),
              "request_count": len(requests), "drained": drained}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def reset_service_cache(stack: testbed.Stack, cfg: testbed.Config) -> None:
    testbed.flush_lmcache(stack, cfg)
    testbed.reset_vllm_caches(
        cfg, (stack.run_root / "source.log", stack.run_root / "sink.log")
    )


def measure_frontier(plan: dict, bundle: dict, profile: dict, cfg: testbed.Config,
                     stack: testbed.Stack, root: Path,
                     vocabulary: int = 201088) -> tuple[list[dict], dict]:
    rows = []
    for direction in plan["service"]["directions"]:
        sessions = manifest_sessions(bundle, direction, "fit", vocabulary, 0)
        context = round(statistics.mean(s.prefix_tokens for s in sessions))
        prefill, decode = profile_rate(profile, "prefill", context), profile_rate(profile, "decode", context)
        def pilot(radius):
            reset_service_cache(stack, cfg)
            cell = root / "fit" / direction / f"pilot-{radius:.6f}"
            return service_probe(cfg.host, cfg.sink_port, cfg.model, sessions, radius,
                                 prefill, decode, plan["service"]["hold_min_s"],
                                 plan["service"]["slos"], cell, 0,
                                 plan["service"]["cache_block_tokens"],
                                 plan["service"]["block_bootstrap_s"],
                                 plan["service"]["bootstrap_samples"])["classification"]
        boundaries = find_boundaries(pilot, plan["service"]["radial_resolution"])
        from destination_campaign import boundary_decision
        for mode, (inside, outside) in boundaries.items():
            labels = {inside: [], outside: []}
            for repeat in range(plan["service"]["disagreement_repeats"]):
                pending = labels if repeat < plan["service"]["initial_repeats"] else {
                    radius: values for radius, values in labels.items() if len(set(values)) > 1}
                for radius in pending:
                    reset_service_cache(stack, cfg)
                    cell = root / "fit" / direction / f"{mode}-r{repeat}-{radius:.6f}"
                    labels[radius].append(service_probe(
                        cfg.host, cfg.sink_port, cfg.model, sessions, radius, prefill, decode,
                        plan["service"]["hold_min_s"], plan["service"]["slos"], cell,
                        repeat + 1, plan["service"]["cache_block_tokens"],
                        plan["service"]["block_bootstrap_s"],
                        plan["service"]["bootstrap_samples"])["classification"][mode])
                if (repeat + 1 >= plan["service"]["initial_repeats"] and
                        all(len(set(values)) == 1 for values in labels.values())):
                    break
            decisions = {radius: boundary_decision([
                "feasible" if value else "infeasible" for value in values])
                for radius, values in labels.items()}
            rows += [{"split": "fit", "direction": direction, "mode": mode,
                      "facet": 0, "run_id": f"{direction}-{repeat}", "bound": inside,
                      "outside": outside, "inside_decision": decisions[inside],
                      "outside_decision": decisions[outside],
                      "inside_feasible_votes": sum(labels[inside]),
                      "inside_repeats": len(labels[inside]),
                      "outside_feasible_votes": sum(labels[outside]),
                      "outside_repeats": len(labels[outside]),
                      "cache_state": "private_prefix"}
                     for repeat in range(len(labels[inside]))]
    measured = {mode: statistics.median(float(r["bound"]) for r in rows
                                         if r["split"] == "fit" and r["mode"] == mode)
                for mode in MODES}
    fit = nest_bounds(measured)
    for row in rows:
        row["measured_bound"] = row["bound"]
        row["bound"] *= fit[row["mode"]] / measured[row["mode"]]
    root.mkdir(parents=True, exist_ok=True)
    (root / "frontier.json").write_text(json.dumps(
        {"measured_bounds": measured, "nested_bounds": fit}, indent=2) + "\n")
    validation = []
    delta = .15 + 1e-9
    for split in ("tune", "validation"):
        for direction in plan["service"]["directions"]:
            sessions = manifest_sessions(bundle, direction, split, vocabulary, 0)
            context = round(statistics.mean(s.prefix_tokens for s in sessions))
            prefill, decode = profile_rate(profile, "prefill", context), profile_rate(profile, "decode", context)
            for mode, bound in fit.items():
                actuals = []
                for expected_feasible, radius in ((True, bound * (1 - delta)),
                                                  (False, bound * (1 + delta))):
                    reset_service_cache(stack, cfg)
                    cell = root / split / direction / f"{mode}-{radius:.6f}"
                    actual = service_probe(cfg.host, cfg.sink_port, cfg.model, sessions,
                                           radius, prefill, decode,
                                           plan["service"]["hold_min_s"],
                                           plan["service"]["slos"], cell, 0,
                                           plan["service"]["cache_block_tokens"],
                                           plan["service"]["block_bootstrap_s"],
                                           plan["service"]["bootstrap_samples"])[
                                               "classification"
                                           ][mode]
                    actuals.append(actual)
                validation.append({"cell": f"{split}/{direction}/{mode}",
                                   "actual_bound": bound if actuals == [True, False]
                                   else bound * (1 - delta if not actuals[0] else 1 + delta),
                                   "predicted_bound": bound, "actual_feasible": actuals[0],
                                   "predicted_feasible": True})
    root.mkdir(parents=True, exist_ok=True)
    (root / "service.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in validation))
    return rows, fit


def migration_manifest(bundle: dict) -> dict:
    sessions = []
    for rank, job_class in enumerate(bundle["manifest"]["splits"]):
        session_id = bundle["manifest"]["splits"][job_class]["validation"][0]
        rows = [r for r in bundle["traces"] if r["session_id"] == session_id and not r.get("reset")]
        row = max(rows, key=lambda r: int(r["input_tokens_total"]))
        sessions.append({"id": session_id, "job_class": job_class, "rank": rank,
                         "state_code": hashlib.sha256(session_id.encode()).hexdigest()[:12].upper(),
                         "turns": [{"time_s": 0, "input_tokens": int(row["input_tokens_total"]),
                                    "append_tokens": int(row["newly_append_tokens"]),
                                    "output_tokens": int(row["output_tokens"]), "reset": False}]})
    return {"schema": profiler.MANIFEST_SCHEMA, "workload": "mixed", "sessions": sessions}


def migration_scenario(session: dict, method: str, context: int, bandwidth: float,
                       repeat: int) -> dict:
    item = {"session_id": session["id"], "job_class": session["job_class"],
            "turn_index": 0, "initial_tokens": context, "order": 0}
    key = hashlib.sha256(f"{method}:{context}:{bandwidth}:{repeat}".encode()).hexdigest()[:16]
    return {"scenario_id": f"loaded-{key}", "kind": "migration", "method": method,
            "activity": "none", "request_schedule": [], "repeat": repeat,
            "deadline_s": 720,
            "sessions": [item], "moves": [{**item, "method": method}],
            "serving_concurrency": 1, "concurrency": 1, "move_concurrency": 1,
            "copy_policy": "initial_final", "final_state": "awake",
            "bandwidth_mbps": bandwidth}


def validate_loaded_scenario(manifest: dict, scenario: dict) -> None:
    required = {"scenario_id", "kind", "method", "activity", "repeat",
                "deadline_s", "sessions", "moves", "concurrency",
                "bandwidth_mbps"}
    if required - scenario.keys() \
            or scenario.get("kind") != "migration" \
            or scenario.get("method") not in profiler.METHODS \
            or scenario.get("activity") != "none" \
            or min(float(scenario.get(key, 0)) for key in (
                "deadline_s", "concurrency", "move_concurrency",
                "serving_concurrency", "bandwidth_mbps")) <= 0 \
            or {(row.get("session_id"), row.get("order"))
                for row in scenario.get("sessions", [])} != {
                    (row.get("session_id"), row.get("order"))
                    for row in scenario.get("moves", [])} \
            or {row.get("method") for row in scenario.get("moves", [])} \
            != {scenario.get("method")}:
        raise ValueError("invalid loaded migration scenario")
    profiler.validate_plan(
        {"schema": profiler.PLAN_SCHEMA, "scenarios": [scenario]}, manifest,
    )


@contextmanager
def loaded_stack(cfg: testbed.Config, root: Path, bandwidth: float, extra: list[str]):
    stack = testbed.start_stack(cfg, root, bandwidth, extra)
    try:
        testbed.start_sink(stack, cfg, extra)
        yield stack
    finally:
        testbed.stop_stack(stack)


def measure_loaded(plan: dict, bundle: dict, profile: dict, cfg: testbed.Config,
                   bounds: dict, root: Path, extra: list[str],
                   vocabulary: int = 201088, rehearsal: bool = False) -> list[dict]:
    manifest, normal = migration_manifest(bundle), bounds["normal"]
    background = sum((manifest_sessions(bundle, job, "validation", vocabulary, 7)
                      for job in plan["service"]["directions"]), [])
    high = plan["migration"]["emergency_inside_fraction"] * bounds["emergency"] / normal
    rhos = [high if rho == "emergency_inside" else float(rho)
            for rho in plan["migration"]["rho"]]
    cells = [(rho, plan["migration"]["context_tokens"],
              plan["migration"]["bandwidth_gbps"] * 1000) for rho in rhos]
    cells += [(high, plan["migration"]["heldout_context_tokens"],
               plan["migration"]["heldout_bandwidth_gbps"] * 1000)]
    rows = []
    for rho, context, bandwidth in cells:
        prefill, decode = profile_rate(profile, "prefill", context), profile_rate(profile, "decode", context)
        for repeat in range(plan["migration"]["repeats"]):
            control_root = root / f"rho{rho:.6f}-t{context}-b{bandwidth:g}-r{repeat}" / "control"
            if rho and not read_checkpoint(
                control_root / "result.json", ("destination_load",),
                lambda row: load_within_target(
                    row["destination_load"].get("achieved_rho"), rho
                ),
            ):
                with loaded_stack(cfg, control_root / "testbed", bandwidth, extra):
                    load = DestinationLoad(cfg.host, cfg.sink_port, cfg.model, background, rho,
                                           prefill, decode, control_root / "foreground", repeat,
                                           normal_bound=normal)
                    load.start()
                    try:
                        load.wait_ready(); time.sleep(30)
                    finally:
                        load.close()
                (control_root / "result.json").write_text(json.dumps(
                    {"status": "complete", "destination_load": load.summary()}, indent=2) + "\n")
            for method in plan["migration"]["methods"]:
                scenario = migration_scenario(manifest["sessions"][repeat % len(manifest["sessions"])],
                                              method, context, bandwidth, repeat)
                validate_loaded_scenario(manifest, scenario)
                cell_root = control_root.parent / method
                if not read_checkpoint(cell_root / "result.json", ("migrations",),
                                       lambda row: len(row["migrations"]) == 1
                                       and (row.get("destination_load") is None if not rho
                                       else load_within_target(
                                           (row.get("destination_load") or {}).get("achieved_rho"),
                                           rho,
                                       ))):
                    with loaded_stack(cfg, cell_root, bandwidth, extra) as stack:
                        load = None if not rho else DestinationLoad(
                            cfg.host, cfg.sink_port, cfg.model, background, rho,
                            prefill, decode, cell_root / "foreground", repeat,
                            normal_bound=normal,
                        )
                        profiler.run_scenario(stack, cfg, manifest, scenario, cell_root,
                                              scenario["scenario_id"], load,
                                              configure_proxy=False)
                rows.append({"rho": rho, "context_tokens": context,
                             "bandwidth_mbps": bandwidth, "repeat": repeat,
                             "method": method, "path": str(cell_root / "result.json")})
            if rehearsal:
                root.mkdir(parents=True, exist_ok=True)
                (root / "loaded.json").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows))
                return rows
    root.mkdir(parents=True, exist_ok=True)
    (root / "loaded.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def unloaded_duration(profile: dict, method: str, context: int, bytes_per_s: float) -> float:
    case = profile["cases"]["central"]
    if method == "replay":
        return context / float(np.interp(context, *zip(*case["replay_tps"]["1"]))) \
            + case["replay_completion_s"] + case["switch_s"]
    kv = case["kv_transfer"]; blocks, tail = divmod(context, kv["block_tokens"])
    size = blocks * kv["block_bytes"]
    return kv["setup_s"] + max(size / bytes_per_s, size / kv["destination_bytes_per_s"]) \
        + kv["initial_completion_s"] + case["switch_s"] \
        + (kv["catch_up_fixed_s"] + tail / kv["tail_replay_tps"] if tail else 0)


def reduce_loaded_results(profile: dict, index: list[dict], root: Path) -> tuple[list[dict], list[dict]]:
    rows, validation = [], []
    for item in index:
        result = json.loads(Path(item["path"]).read_text())
        moves = result.get("migrations", [])
        if result.get("status") != "complete" or len(moves) != 1:
            raise RuntimeError(f"invalid loaded migration: {item['path']}")
        target = float(item["rho"])
        achieved = (result.get("destination_load") or {}).get("achieved_rho")
        if target == 0 and result.get("destination_load") is None:
            achieved = 0
        elif not load_within_target(achieved, target):
            raise RuntimeError(
                f"loaded migration rho {achieved} misses target {item['rho']}"
            )
        move = moves[0]; observed = (move["switch_end_ns"] - move["initial_start_ns"]) / 1e9
        bandwidth = float(item["bandwidth_mbps"]) * 125000
        base = unloaded_duration(profile, item["method"], int(item["context_tokens"]), bandwidth)
        correct = all(r.get("status_code") == 200 and
                      r.get("context_hash") == r.get("committed_context_hash")
                      for r in result.get("continuations", []))
        row = {"method": item["method"], "rho": target,
               "run_id": item["repeat"], "duration_factor": observed / base,
               "context_tokens": item["context_tokens"],
               "bandwidth_bytes_per_s": bandwidth, "achieved_rho": achieved,
               "correct": correct,
               "_observed_s": observed, "_base_s": base}
        rows.append(row)
    baseline = {method: statistics.median(
        r["duration_factor"] for r in rows
        if r["method"] == method and r["rho"] == 0
    ) for method in ("replay", "kv_transfer")}
    factors = {method: max(
        1, *(r["duration_factor"] / baseline[method] for r in rows
             if r["method"] == method and r["rho"] > 0
             and int(r["context_tokens"]) == 16384)
    )
               for method in ("replay", "kv_transfer")}
    for row in rows:
        if int(row["context_tokens"]) == 24576:
            validation.append({"cell": f"{row['method']}-heldout-r{row['run_id']}",
                               "observed_s": row["_observed_s"],
                               "predicted_s": row["_base_s"] * baseline[row["method"]]
                               * factors[row["method"]],
                               "correct": row["correct"]})
        row.pop("_observed_s"); row.pop("_base_s")
    (root / "reduction.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in validation))
    return rows, validation


def runtime_identity(cfg: testbed.Config, plan: dict, bundle: dict,
                     profile: dict, provenance: str) -> dict:
    reference = cfg.hf_home / "hub" / f"models--{cfg.model.replace('/', '--')}" / "refs" / "main"
    if not reference.is_file():
        raise RuntimeError("pinned model revision is unavailable")
    revision = reference.read_text().strip()
    fractions = []
    for job in plan["service"]["directions"]:
        for session in manifest_sessions(bundle, job, "fit", 201088, 0):
            context = session.prefix_tokens
            f = session.append_tokens / profile_rate(profile, "prefill", context)
            g = session.output_tokens / profile_rate(profile, "decode", context)
            fractions.append(f / (f + g))
    fingerprint = hashlib.sha256(json.dumps(
        [plan["image_sha256"], cfg.model, revision, cfg.max_model_len,
         cfg.max_num_seqs, cfg.max_num_batched_tokens,
         plan["service"]["cache_block_tokens"]], separators=(",", ":")).encode()).hexdigest()
    return {"compatibility": {"model": f"{cfg.model}@{revision}",
                              "tokenizer": f"{cfg.model}@{revision}",
                              "durable_log": f"lmcache-{testbed.lmcache_mode()}-{fingerprint}",
                              "kv_abi": fingerprint},
            "kv_capacity_tokens": profile["kv_capacity_tokens"],
            "workload_prefill_fraction_range": [min(fractions), max(fractions)],
            "provenance": provenance}


def write_run_metadata(path: Path, metadata: dict, resume_from_git_sha: str | None = None) -> None:
    metadata = dict(metadata)
    if path.exists():
        previous = json.loads(path.read_text())
        if all(previous.get(key) == value for key, value in metadata.items()):
            return
        immutable = set(metadata) - {"git_sha", "dirty"}
        if (resume_from_git_sha != previous.get("git_sha") or
                any(previous.get(key) != metadata[key] for key in immutable)):
            raise RuntimeError("run root belongs to a different campaign or commit")
        metadata["git_history"] = previous.get("git_history", [previous["git_sha"]]) + [metadata["git_sha"]]
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def finalize(plan: dict, bundle: dict, profile: dict, cfg: testbed.Config,
             run_root: Path, service: list[dict], loaded: list[dict],
             loaded_validation: list[dict]) -> dict:
    from destination_campaign import acceptance_report, reduce_profile
    anchors = json.loads((run_root / "anchors" / "anchors.json").read_text())
    identity = runtime_identity(cfg, plan, bundle, profile,
                                hashlib.sha256((run_root / "run.json").read_bytes()).hexdigest())
    reduced = reduce_profile(anchors, service, loaded, identity, [[1, 1]])
    service_validation = [json.loads(line) for line in
                          (run_root / "service" / "validation.jsonl").read_text().splitlines()]
    report = acceptance_report(service_validation, loaded_validation)
    (run_root / "profile-fragments.json").write_text(json.dumps(reduced, indent=2) + "\n")
    (run_root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_campaign(plan_path: Path, run_root: Path, cfg: testbed.Config,
                 extra: list[str], resume_from_git_sha: str | None = None) -> None:
    from destination_campaign import IMAGE_SHA256, write_checksums
    plan, bundle, profile = load_inputs(plan_path)
    git_sha, dirty = profiler.git_state(False)
    metadata = {"schema": plan["schema"],
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "image_sha256": IMAGE_SHA256, "git_sha": git_sha, "dirty": dirty}
    run_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_root / "run.json", metadata, resume_from_git_sha)
    rehearsal_value = os.environ.get("QH_LOADED_REHEARSAL", "0")
    if rehearsal_value not in {"0", "1"}:
        raise ValueError("QH_LOADED_REHEARSAL must be 0 or 1")
    rehearsal = rehearsal_value == "1"
    def attempt():
        stack = testbed.start_stack(cfg, run_root / "testbed", 10000, extra)
        try:
            testbed.start_sink(stack, cfg, extra)
            smoke = testbed.run_smoke2_probe(cfg, stack.run_root, 10000)
            preflight = integrity_preflight(cfg, plan, smoke)
            (run_root / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
            expected = {(metric, context): profile_rate(profile, metric, context)
                        for metric in ("prefill", "decode") for context in plan["service"]["anchors"]}
            anchors = measure_anchors(cfg.host, cfg.sink_port, cfg.model,
                                      plan["service"]["anchors"], 201088, expected,
                                      run_root / "anchors", stack, cfg)
            report = anchor_gate(anchors, expected, plan["anchor_drift_limit"])
            (run_root / "anchor-gate.json").write_text(json.dumps(report, indent=2) + "\n")
            apply_anchor_rates(profile, report)
            testbed.flush_lmcache(stack, cfg)
            service, bounds = measure_frontier(plan, bundle, profile, cfg, stack, run_root / "service")
            shared, stack = stack, None
            testbed.stop_stack(shared)
            loaded_index = measure_loaded(plan, bundle, profile, cfg, bounds,
                                          run_root / "loaded", extra,
                                          rehearsal=rehearsal)
            if rehearsal:
                return
            loaded, validation = reduce_loaded_results(profile, loaded_index, run_root / "loaded")
            finalize(plan, bundle, profile, cfg, run_root, service, loaded, validation)
        finally:
            if stack:
                testbed.stop_stack(stack)
    retry_call(attempt, run_root / "retries.jsonl",
               int(os.environ.get("QH_CAMPAIGN_ATTEMPTS", "4")),
               float(os.environ.get("QH_RETRY_DELAY_S", "30")))
    if not rehearsal:
        write_checksums(run_root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from-git-sha", default=os.environ.get("QH_RESUME_FROM_GIT_SHA"))
    testbed.add_common(parser)
    parser.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
    run_campaign(args.plan, args.run_root, testbed.config_from_args(args), extra,
                 args.resume_from_git_sha)


if __name__ == "__main__":
    main()
