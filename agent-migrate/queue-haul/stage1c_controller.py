from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import json
import os
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import stage1b_drain_sink as b
from migration import MigrationController, Move, RequestResult, SessionState, StreamChunk


MANIFEST_SCHEMA = "queue-haul-migration-manifest-v2"
PLAN_SCHEMA = "queue-haul-migration-plan-v2"
RESULT_SCHEMA = "queue-haul-migration-result-v2"
RUN_SCHEMA = "queue-haul-migration-run-v2"
METHODS = ("replay", "kv_transfer")
ACTIVITIES = ("none", "one_turn")
JOB_CLASSES = ("interactive_coding", "coding", "agentic_tool_loop")
RESET_SUCCESS = "Successfully reset prefix cache"
PROBE_MAX_TOKENS = 512
MAX_MODEL_TOKENS = 32768


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_number
                yield row


def field(row: dict, *names, default=None):
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def trace_time(row: dict) -> float:
    value = field(row, "timestamp", "ts", "created_at", "started_at", "start_time")
    if value is None:
        value = next((event.get("timestamp") for event in row.get("timing_events", []) if event.get("timestamp") is not None), None)
    if value is None:
        raise ValueError(f"trace row {row['_line']} has no timestamp")
    if isinstance(value, (int, float)):
        return float(value)
    return __import__("datetime").datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def trace_tokens(row: dict) -> tuple[int, int, int]:
    total = int(field(row, "input_tokens_total", "input_tokens", "prompt_tokens", default=0))
    prefix = int(field(row, "prefix_tokens", "cached_input_tokens", "cache_read_input_tokens", default=0))
    append = int(field(row, "newly_append_tokens", "append_tokens", default=max(1, total - prefix)))
    output = int(field(row, "output_tokens", "completion_tokens", "generated_tokens", default=32))
    if total < 1:
        raise ValueError(f"trace row {row['_line']} has no input token count")
    return total, max(1, append), max(1, output)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile needs data")
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def make_manifest(input_path: Path, workload: str, sessions: int, seed: int) -> dict:
    if workload not in JOB_CLASSES:
        raise ValueError(f"workload must be one of {', '.join(JOB_CLASSES)}")
    groups: dict[str, list[dict]] = {}
    for row in read_rows(input_path):
        session_id = str(field(row, "session_id", "session", "conversation_id", "trace_key", default=""))
        if not session_id:
            raise ValueError(f"trace row {row['_line']} has no session id")
        groups.setdefault(session_id, []).append(row)
    candidates = []
    for session_id, rows in sorted(groups.items()):
        parsed = []
        for row in rows:
            try:
                parsed.append((trace_time(row), *trace_tokens(row), row))
            except ValueError:
                continue
        parsed.sort(key=lambda item: item[0])
        if not parsed:
            continue
        span = parsed[-1][0] - parsed[0][0]
        rate = (len(parsed) - 1) / span if span > 0 else 0.0
        candidates.append({
            "id": session_id,
            "turn_rate_hz": rate,
            "human_fraction": sum(bool(item[4].get("current_user_message_count")) for item in parsed) / len(parsed),
            "tool_fraction": sum(bool(item[4].get("tools")) for item in parsed) / len(parsed),
            "turns": [{"time_s": item[0], "input_tokens": item[1], "append_tokens": item[2], "output_tokens": item[3], "reset": index > 0 and item[1] < parsed[index - 1][1]} for index, item in enumerate(parsed)],
        })
    if not candidates:
        raise ValueError("trace has no usable sessions")
    q25, q75 = quantile([row["turn_rate_hz"] for row in candidates], .25), quantile([row["turn_rate_hz"] for row in candidates], .75)
    for row in candidates:
        row["job_class"] = (
            "interactive_coding" if row["turn_rate_hz"] <= q25 and row["human_fraction"] >= .25
            else "agentic_tool_loop" if row["turn_rate_hz"] >= q75 and row["tool_fraction"] >= .95
            else "coding"
        )
        row["state_code"] = hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()[:12].upper()
    eligible = [row for row in candidates if row["job_class"] == workload]
    if len(eligible) < sessions:
        raise ValueError(f"need {sessions} {workload} sessions, found {len(eligible)}")
    selected = random.Random(seed).sample(sorted(eligible, key=lambda row: row["id"]), sessions)
    for rank, row in enumerate(selected):
        row["rank"] = rank
    return {
        "schema": MANIFEST_SCHEMA,
        "source": {"path": str(input_path), "sha256": file_hash(input_path)},
        "seed": seed,
        "workload": workload,
        "classification": {"turn_rate_q25_hz": q25, "turn_rate_q75_hz": q75},
        "message_generator": "deterministic_trace_tokens_v2",
        "sessions": selected,
    }


def token_text(label: str, count: int) -> str:
    return f"{label} " + "x " * max(1, count)


def session_messages(session: dict, turn_index: int) -> list[dict]:
    turns = session["turns"]
    if turn_index < 0 or turn_index >= len(turns):
        raise ValueError(f"invalid turn index for {session['id']}: {turn_index}")
    code = session["state_code"]
    messages = [{"role": "system", "content": f"Session state code {code}. Include {code} in every reply."}]
    start = max((index for index in range(turn_index + 1) if turns[index].get("reset")), default=0)
    for index in range(start, turn_index + 1):
        turn = turns[index]
        user_tokens = turn["input_tokens"] if index == start else turn["append_tokens"]
        messages.append({"role": "user", "content": token_text(f"{session['id']} user turn {index}", user_tokens)})
        if index < turn_index:
            messages.append({"role": "assistant", "content": token_text(f"{code} assistant turn {index}", turn["output_tokens"])})
    return messages


def estimated_prompt_tokens(session: dict, turn_index: int) -> int:
    messages = session_messages(session, turn_index)
    return sum(len(row["content"].split()) for row in messages) + 64 * len(messages) + PROBE_MAX_TOKENS


def nearest_turn(session: dict, context_size: int) -> int:
    return min(range(len(session["turns"])), key=lambda index: (abs(session["turns"][index]["input_tokens"] - context_size), index))


def stable_seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported manifest schema")
    if not manifest.get("sessions") or len({row["id"] for row in manifest["sessions"]}) != len(manifest["sessions"]):
        raise ValueError("manifest sessions must be nonempty and unique")
    classes = {row["job_class"] for row in manifest["sessions"]}
    if classes != {manifest.get("workload")} or not classes <= set(JOB_CLASSES):
        raise ValueError("a manifest must contain one standard job class")


def make_plan(manifest_path: Path, context_sizes: list[int], concurrency: list[int],
              bandwidth_mbps: list[float], methods: list[str], activity: list[str],
              repeats: int, seed: int, deadline_s: float = 300.0,
              session_ids: tuple[str, ...] | list[str] = ()) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if not all(value > 0 for value in [*context_sizes, *concurrency, *bandwidth_mbps, repeats, deadline_s]):
        raise ValueError("plan values must be positive")
    if not methods or not activity or not set(methods) <= set(METHODS) \
            or not set(activity) <= set(ACTIVITIES):
        raise ValueError("unknown method or activity")
    count = max(concurrency)
    if len(manifest["sessions"]) < count:
        raise ValueError(f"manifest needs at least {count} sessions")
    sessions = {row["id"]: row for row in manifest["sessions"]}
    if session_ids and (len(session_ids) != count or len(set(session_ids)) != count
                        or not set(session_ids) <= sessions.keys()):
        raise ValueError(f"session ids must name exactly {count} manifest sessions")
    scenarios = []
    for size in context_sizes:
        for active in activity:
            for repeat in range(repeats):
                if session_ids:
                    chosen = [sessions[session_id] for session_id in session_ids]
                else:
                    chosen = sorted(manifest["sessions"], key=lambda row: row["id"])
                    random.Random(stable_seed(seed, size, active, repeat)).shuffle(chosen)
                    chosen = chosen[:count]
                session_rows = [{"session_id": row["id"], "job_class": row["job_class"],
                                 "turn_index": nearest_turn(row, size), "order": order}
                                for order, row in enumerate(chosen)]
                for width in concurrency:
                    match_id = object_hash([size, active, repeat, width, session_rows])[:16]
                    base = {"match_id": match_id, "context_size": size,
                            "concurrency": width, "activity": active, "repeat": repeat,
                            "deadline_s": deadline_s, "sessions": session_rows}
                    control = {**base, "method": methods[0],
                               "bandwidth_mbps": bandwidth_mbps[0], "kind": "control",
                               "scenario_id": f"c-{match_id}", "moves": []}
                    scenarios.append(control)
                    for method in methods:
                        for link in bandwidth_mbps:
                            scenario_id = object_hash([match_id, method, link])[:16]
                            scenario = {**base, "method": method, "bandwidth_mbps": link,
                                        "kind": "migration", "scenario_id": f"m-{scenario_id}"}
                            scenario["moves"] = [{**row, "method": method}
                                                 for row in session_rows]
                            scenarios.append(scenario)
    random.Random(seed).shuffle(scenarios)
    plan = {"schema": PLAN_SCHEMA, "manifest": {"path": str(manifest_path), "sha256": file_hash(manifest_path)}, "seed": seed, "scenarios": scenarios}
    validate_plan(plan, manifest)
    return plan


def validate_plan(plan: dict, manifest: dict) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported plan schema")
    validate_manifest(manifest)
    sessions = {row["id"]: row for row in manifest["sessions"]}
    ids = set(sessions)
    scenario_ids = [row["scenario_id"] for row in plan.get("scenarios", [])]
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("plan scenarios must be nonempty and unique")
    for scenario in plan["scenarios"]:
        rows = scenario["sessions"]
        if not {row["session_id"] for row in rows} <= ids or len(rows) < scenario["concurrency"]:
            raise ValueError(f"invalid sessions in {scenario['scenario_id']}")
        if scenario["kind"] == "migration" and len(scenario["moves"]) != len(rows):
            raise ValueError(f"migration {scenario['scenario_id']} does not move every selected session")
        if len({row.get("method") for row in scenario["moves"]}) > 1:
            pass  # Hand-authored mixed plans are valid; generated profiles use one method.
        for row in rows:
            tokens = estimated_prompt_tokens(sessions[row["session_id"]], row["turn_index"])
            if tokens > MAX_MODEL_TOKENS:
                raise ValueError(f"scenario {scenario['scenario_id']} prompt estimate {tokens} exceeds {MAX_MODEL_TOKENS}")


class EventLog:
    def __init__(self, path: Path, run_id: str, scenario_id: str):
        self.handle = path.open("w", buffering=1)
        self.lock = threading.Lock()
        self.fixed = {"run_id": run_id, "scenario_id": scenario_id}

    def write(self, event: str, **fields) -> None:
        row = {**self.fixed, "event": event, "monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns(), **fields}
        with self.lock:
            self.handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.handle.close()


def messages_hash(messages: list[dict] | tuple[dict, ...]) -> str:
    return object_hash(list(messages))


def chat_payload(cfg: b.Config, messages: list[dict], max_tokens: int, bypass_lmcache: bool = False) -> dict:
    payload = {"model": cfg.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if bypass_lmcache:
        payload["kv_transfer_params"] = {"qh_bypass_lmcache": True}
    return payload


def stream_chat(cfg: b.Config, port: int, messages: list[dict], max_tokens: int, context_hash: str, timeout_s: float, bypass_lmcache: bool = False) -> tuple[RequestResult, str]:
    body = json.dumps(chat_payload(cfg, messages, max_tokens, bypass_lmcache))
    start = time.monotonic_ns()
    conn = http.client.HTTPConnection(cfg.host, port, timeout=timeout_s)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    chunks, text, request_id, first, prompt_tokens, output_tokens = [], [], "", None, 0, 0
    if response.status != 200:
        error = response.read().decode(errors="ignore")
        conn.close()
        end = time.monotonic_ns()
        return RequestResult("", response.status, context_hash, start, end), error
    while line := response.readline():
        now = time.monotonic_ns()
        if not line.strip().startswith(b"data:"):
            continue
        chunks.append(StreamChunk(now, len(line)))
        data = line.strip()[5:].strip()
        if data == b"[DONE]":
            break
        item = json.loads(data)
        request_id = request_id or item.get("id", "")
        usage = item.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
        output_tokens = int(usage.get("completion_tokens", output_tokens))
        content = (item.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        if content and first is None:
            first = now
        text.append(content)
    conn.close()
    end = time.monotonic_ns()
    return RequestResult(request_id, response.status, context_hash, start, end, first or end, prompt_tokens, output_tokens, stream_chunks=tuple(chunks)), "".join(text)


def cache_operations(path: Path, start_ns: int = 0, end_ns: int = 2**63 - 1) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if start_ns <= int(row.get("monotonic_ns", 0)) <= end_ns and row.get("operation"):
            rows.append(row)
    return rows


def kv_layout(path: Path, end_ns: int) -> dict:
    rows = [row for row in cache_operations(path, end_ns=end_ns) if row["operation"] == "source_write" and int(row["bytes"]) > 0]
    if not rows:
        raise RuntimeError("no source KV layout was logged")
    chunk_bytes = max(int(row["bytes"]) for row in rows)
    full = [row for row in rows if int(row["bytes"]) == chunk_bytes]
    layouts = {(row.get("dtype"), tuple(row.get("shape", []))) for row in full}
    if len(layouts) != 1:
        raise RuntimeError(f"inconsistent full KV chunk layouts: {layouts}")
    dtype, shape = layouts.pop()
    return {"chunk_tokens": 256, "chunk_bytes": chunk_bytes, "bytes_per_token": chunk_bytes / 256, "dtype": dtype, "shape": list(shape)}


def kv_metrics(hit: int, layout: dict) -> tuple[int, int]:
    tokens = layout["chunk_tokens"]
    return (hit + tokens - 1) // tokens, hit * layout["chunk_bytes"] // tokens


def expected_hits(method: str, phase: str, total: int, source_prompt_tokens: int | None = None) -> int:
    if method != "kv_transfer":
        return 0
    if phase == "initial":
        return total
    if source_prompt_tokens is None:
        raise ValueError("catch-up requires measured source prompt tokens")
    return source_prompt_tokens // 256 * 256


def lookup_tokens(path: Path, request_id: str) -> tuple[int, int]:
    import re
    pattern = re.compile(r"Reqid:\s*([^,]+),\s*Total tokens\s*(\d+),\s*LMCache hit tokens:\s*(\d+)")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        matches = [(int(match.group(2)), int(match.group(3))) for match in pattern.finditer(path.read_text(errors="ignore")) if match.group(1).strip() == request_id]
        if matches:
            return matches[-1]
        time.sleep(.05)
    raise RuntimeError(f"LMCache did not report request {request_id}")


class LiveSession:
    def __init__(self, cfg: b.Config, session: dict, turn_index: int, event_log: EventLog, source_log: Path, cache_log: Path, timeout_s: float):
        self.cfg, self.row, self.event_log = cfg, session, event_log
        self.source_log, self.cache_log, self.timeout_s = source_log, cache_log, timeout_s
        self.session_id, self.state_code = session["id"], session["state_code"]
        self.messages = session_messages(session, turn_index)
        self.generation, self.route, self.paused = 0, cfg.src_port, False
        self.lock = threading.Lock()
        self.activity_thread: threading.Thread | None = None
        self.activity_error: Exception | None = None
        self.activity_times: tuple[int, int] | None = None
        self.cache_keys: set[str] = set()
        self.activity_prompt_tokens: int | None = None

    def probe(self, messages: list[dict], prompt: str | None = None) -> list[dict]:
        return messages + [{"role": "user", "content": prompt or f"Reply with session state code {self.state_code}."}]

    def request(self, port: int, messages: list[dict], label: str, prompt: str | None = None, bypass_lmcache: bool = False) -> tuple[RequestResult, str]:
        context_hash = messages_hash(messages)
        self.event_log.write("request_start", session_id=self.session_id, request_id=label, route_port=port, context_hash=context_hash)
        result, text = stream_chat(self.cfg, port, self.probe(messages, prompt), PROBE_MAX_TOKENS, context_hash, self.timeout_s, bypass_lmcache)
        self.event_log.write("request_end", session_id=self.session_id, request_id=result.request_id, route_port=port, status_code=result.status_code, context_hash=context_hash, first_byte_ns=result.first_byte_ns, chunks=[asdict(chunk) for chunk in result.stream_chunks])
        if result.status_code != 200 or self.state_code not in text:
            raise RuntimeError(f"{label} failed state check for {self.session_id}: HTTP {result.status_code}")
        return result, text

    def warm(self) -> None:
        before = time.monotonic_ns()
        self.request(self.cfg.src_port, list(self.messages), "source_warm")
        after = time.monotonic_ns()
        self.cache_keys |= {row["key_hash"] for row in cache_operations(self.cache_log, before, after) if row["operation"] == "source_write"}

    def snapshot(self) -> SessionState:
        with self.lock:
            return SessionState(self.session_id, self.generation, tuple(self.messages), messages_hash(self.messages))

    def start_activity(self) -> None:
        if self.activity_thread:
            return
        self.activity_thread = threading.Thread(target=self._activity, daemon=True)
        self.activity_thread.start()

    def _activity(self) -> None:
        start = time.monotonic_ns()
        try:
            with self.lock:
                base = list(self.messages)
            user = {"role": "user", "content": f"Controlled turn: reply only with session state code {self.state_code}."}
            result, text = self.request(self.cfg.src_port, base, "controlled_turn", user["content"])
            with self.lock:
                self.messages = base + [user, {"role": "assistant", "content": text}]
                self.activity_prompt_tokens = result.prompt_tokens
                self.generation += 1
            self.cache_keys |= {row["key_hash"] for row in cache_operations(self.cache_log, start, result.end_ns) if row["operation"] == "source_write"}
        except Exception as exc:
            self.activity_error = exc
        self.activity_times = start, time.monotonic_ns()

    def wait_activity(self) -> None:
        if self.activity_thread:
            self.activity_thread.join(self.timeout_s)
            if self.activity_thread.is_alive():
                raise TimeoutError(f"activity timed out for {self.session_id}")
        if self.activity_error:
            raise self.activity_error

    def continuation(self) -> RequestResult:
        with self.lock:
            if self.paused:
                raise RuntimeError(f"session {self.session_id} is paused")
            messages, port, generation = list(self.messages), self.route, self.generation
        user = {"role": "user", "content": f"Continuation: reply only with session state code {self.state_code}."}
        result, text = self.request(port, messages, "continuation", user["content"])
        with self.lock:
            if generation != self.generation:
                raise RuntimeError(f"session {self.session_id} changed during continuation")
            self.messages = messages + [user, {"role": "assistant", "content": text}]
            self.generation += 1
        return result


class LiveRuntime:
    def __init__(self, sessions: dict[str, LiveSession], cfg: b.Config, activity: str, sink_log: Path, cache_log: Path, event_log: EventLog, request_log: Path):
        self.sessions, self.cfg, self.activity = sessions, cfg, activity
        self.sink_log, self.cache_log, self.event_log = sink_log, cache_log, event_log
        self.requests = request_log.open("w", buffering=1)
        self.lock = threading.Lock()

    def snapshot(self, move: Move) -> SessionState:
        state = self.sessions[move.session_id].snapshot()
        self.event_log.write("snapshot", move_id=move.order, session_id=move.session_id, generation=state.generation, context_hash=state.context_hash)
        return state

    def prepare(self, move: Move, state: SessionState, phase: str) -> RequestResult:
        session = self.sessions[move.session_id]
        if phase == "initial" and self.activity == "one_turn":
            session.start_activity()
        self.event_log.write("copy_start", move_id=move.order, session_id=move.session_id, method=move.method, phase=phase)
        result, _text = session.request(self.cfg.api_proxy_port, list(state.messages), f"{move.method}_{phase}", bypass_lmcache=move.method == "replay")
        total, hit = lookup_tokens(self.sink_log, result.request_id)
        expected = expected_hits(move.method, phase, total, session.activity_prompt_tokens)
        if hit != expected:
            raise RuntimeError(f"{move.method} request {result.request_id} hit {hit} tokens, expected {expected}")
        layout = kv_layout(self.cache_log, result.end_ns)
        logical_chunks, logical_bytes = kv_metrics(hit, layout)
        result = replace(result, processed_tokens=total - hit, logical_kv_chunks=logical_chunks, logical_kv_bytes=logical_bytes)
        with self.lock:
            self.requests.write(json.dumps({"move_id": move.order, "session_id": move.session_id, "method": move.method, "phase": phase, "kv_layout": layout, **asdict(result)}, separators=(",", ":")) + "\n")
        self.event_log.write("copy_end", move_id=move.order, session_id=move.session_id, method=move.method, phase=phase, processed_tokens=result.processed_tokens, logical_kv_bytes=logical_bytes, logical_kv_chunks=result.logical_kv_chunks, kv_layout=layout)
        return result

    def pause(self, session_id: str) -> None:
        session = self.sessions[session_id]
        with session.lock:
            session.paused = True
        self.event_log.write("pause", session_id=session_id)

    def wait_idle(self, session_id: str) -> SessionState:
        session = self.sessions[session_id]
        session.wait_activity()
        state = session.snapshot()
        self.event_log.write("idle", session_id=session_id, generation=state.generation, context_hash=state.context_hash)
        return state

    def commit(self, move: Move, state: SessionState) -> None:
        session = self.sessions[move.session_id]
        with session.lock:
            if session.generation != state.generation or messages_hash(session.messages) != state.context_hash:
                raise RuntimeError(f"session {move.session_id} changed before route switch")
            session.messages = list(state.messages)
            session.route = self.cfg.api_proxy_port
            session.paused = False
        self.event_log.write("route_switch", move_id=move.order, session_id=move.session_id, route_port=self.cfg.api_proxy_port, context_hash=state.context_hash)

    def resume_source(self, session_id: str) -> None:
        session = self.sessions[session_id]
        with session.lock:
            session.route, session.paused = self.cfg.src_port, False
        self.event_log.write("resume_source", session_id=session_id)

    def close(self) -> None:
        self.requests.close()


class PowerSampler:
    def __init__(self, path: Path):
        self.path, self.stop = path, threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            with self.path.open("w", newline="", buffering=1) as handle:
                writer = csv.writer(handle)
                writer.writerow(["monotonic_ns", "wall_ns", "gpu", "power_w", "utilization_pct", "memory_mib", "valid"])
                while not self.stop.is_set():
                    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
                    mono, wall = time.monotonic_ns(), time.time_ns()
                    for line in output.splitlines():
                        values = [value.strip() for value in line.split(",")]
                        valid = all(value not in {"N/A", "[N/A]"} for value in values)
                        writer.writerow([mono, wall, values[0], *values[1:], int(valid)])
                    self.stop.wait(.25)
        except Exception as exc:
            self.error = exc

    def close(self) -> None:
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("power sampler did not stop")
        if self.error:
            raise self.error


def parse_node_power(text: str) -> float:
    fields = dict(item.split("=", 1) for item in text.split() if "=" in item)
    try:
        watts = float(fields["CurrentWatts"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Slurm node power is unavailable: {text.strip()}") from exc
    if watts <= 0:
        raise RuntimeError(f"Slurm node power is disabled: {text.strip()}")
    return watts


def node_power_reading() -> tuple[str, float]:
    node = os.environ.get("SLURMD_NODENAME")
    if not node:
        raise RuntimeError("SLURMD_NODENAME is required for node power")
    output = subprocess.check_output(
        ["scontrol", "show", "node", node, "--oneliner"], text=True,
    )
    return node, parse_node_power(output)


class NodePowerSampler:
    def __init__(self, path: Path):
        self.path, self.stop = path, threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            with self.path.open("w", newline="", buffering=1) as handle:
                writer = csv.writer(handle)
                writer.writerow(["monotonic_ns", "wall_ns", "node", "current_watts"])
                while not self.stop.is_set():
                    node, watts = node_power_reading()
                    writer.writerow([time.monotonic_ns(), time.time_ns(), node, watts])
                    self.stop.wait(1)
        except Exception as exc:
            self.error = exc

    def close(self) -> None:
        self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("node power sampler did not stop")
        if self.error:
            raise self.error


def time_weighted_mean(rows: list[dict], start_ns: int, end_ns: int,
                       key: str) -> float:
    points = sorted(rows, key=lambda row: row["monotonic_ns"])
    if end_ns <= start_ns or not any(row["monotonic_ns"] <= start_ns for row in points) \
            or not any(row["monotonic_ns"] >= end_ns for row in points):
        raise RuntimeError("power samples do not cover state window")
    value = next(row[key] for row in reversed(points)
                 if row["monotonic_ns"] <= start_ns)
    area, cursor = 0.0, start_ns
    for row in points:
        sample = row["monotonic_ns"]
        if sample <= start_ns:
            continue
        stop = min(sample, end_ns)
        area += (stop - cursor) * value
        if sample >= end_ns:
            return area / (end_ns - start_ns)
        cursor, value = sample, row[key]
    raise RuntimeError("power samples do not cover state window")


def power_state_summary(gpu_path: Path, node_path: Path | None,
                        cycles: list[dict]) -> list[dict]:
    gpu_rows = power_rows(gpu_path)
    gpus = sorted({row["gpu"] for row in gpu_rows})
    if len(gpus) != 2:
        raise RuntimeError(f"expected two measured GPUs in {gpu_path}, found {gpus}")
    node_rows = []
    if node_path:
        with node_path.open() as handle:
            node_rows = [
                {**row, "monotonic_ns": int(row["monotonic_ns"]),
                 "current_watts": float(row["current_watts"])}
                for row in csv.DictReader(handle)
            ]
    summary = []
    for cycle, windows in enumerate(cycles):
        for state in ("awake", "sleep"):
            start, end = windows[f"{state}_ns"]
            for role, gpu in zip(("source", "destination"), gpus):
                rows = [row for row in gpu_rows if row["gpu"] == gpu]
                summary.append({
                    "cycle": cycle, "state": state, "scope": "gpu", "device": role,
                    "duration_s": duration(start, end),
                    "mean_power_w": time_weighted_mean(rows, start, end, "power_w"),
                    "mean_memory_mib": time_weighted_mean(
                        rows, start, end, "memory_mib"
                    ),
                })
            if node_rows:
                summary.append({
                    "cycle": cycle, "state": state, "scope": "node",
                    "device": node_rows[0]["node"],
                    "duration_s": duration(start, end),
                    "mean_power_w": time_weighted_mean(
                        node_rows, start, end, "current_watts"
                    ),
                })
    source = [row for row in summary
              if row["scope"] == "gpu" and row["device"] == "source"]
    awake = statistics.median(
        row["mean_memory_mib"] for row in source if row["state"] == "awake"
    )
    sleeping = statistics.median(
        row["mean_memory_mib"] for row in source if row["state"] == "sleep"
    )
    if sleeping >= awake:
        raise RuntimeError("source GPU memory did not fall during sleep")
    return summary


def profile_power_states(stack: b.Stack, cfg: b.Config, root: Path, cycles: int,
                         window_s: float, node_power: bool) -> None:
    if cycles < 1 or window_s <= 0:
        raise ValueError("power-state cycles and window must be positive")
    root.mkdir(parents=True, exist_ok=True)
    b.set_source_sleep(cfg, False)
    gpu_sampler = PowerSampler(root / "gpu_power.csv")
    node_sampler = NodePowerSampler(root / "node_power.csv") if node_power else None
    gpu_sampler.start()
    if node_sampler:
        node_sampler.start()
    rows, sleeping = [], False
    try:
        for cycle in range(cycles):
            b.reset_vllm_caches(
                cfg, (stack.run_root / "source.log", stack.run_root / "sink.log")
            )
            time.sleep(10)
            awake_start = time.monotonic_ns()
            time.sleep(window_s)
            awake_end = time.monotonic_ns()
            sleep_start = time.monotonic_ns()
            b.set_source_sleep(cfg, True)
            sleeping = True
            sleep_end = time.monotonic_ns()
            time.sleep(10)
            steady_sleep_start = time.monotonic_ns()
            time.sleep(window_s)
            steady_sleep_end = time.monotonic_ns()
            wake_start = time.monotonic_ns()
            b.set_source_sleep(cfg, False)
            sleeping = False
            wake_end = time.monotonic_ns()
            messages = [{"role": "user", "content": "Reply with exactly OK."}]
            probe, text = stream_chat(
                cfg, cfg.src_port, messages, 4, object_hash(messages), 600
            )
            if probe.status_code != 200 or probe.first_byte_ns is None or "OK" not in text:
                raise RuntimeError("source did not serve a verified wake probe")
            rows.append({
                "cycle": cycle, "awake_ns": [awake_start, awake_end],
                "sleep_transition_ns": [sleep_start, sleep_end],
                "sleep_ns": [steady_sleep_start, steady_sleep_end],
                "wake_transition_ns": [wake_start, wake_end],
                "wake_probe": asdict(probe),
            })
    finally:
        if sleeping:
            b.set_source_sleep(cfg, False)
        try:
            gpu_sampler.close()
        finally:
            if node_sampler:
                node_sampler.close()
    summary = power_state_summary(
        root / "gpu_power.csv", root / "node_power.csv" if node_power else None,
        rows,
    )
    write_csv(root / "summary.csv", summary)
    write_json(root / "result.json", {
        "cycles": rows, "window_s": window_s, "node_power": node_power,
    })


def write_cache_slice(source: Path, destination: Path, start_ns: int, end_ns: int) -> None:
    with destination.open("w") as handle:
        for row in cache_operations(source, start_ns, end_ns):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def restart_proxy(stack: b.Stack, cfg: b.Config, scenario_root: Path, mbps: float) -> None:
    b.stop_proc(stack.proxy)
    stack.proxy = b.start_logged(b.proxy_cmd(cfg, mbps, scenario_root / "proxy_bytes.csv"), scenario_root / "proxy.log")
    b.wait_tcp_process(cfg.host, cfg.kv_proxy_port, 30, stack.proxy, scenario_root / "proxy.log")
    b.wait_tcp(cfg.host, cfg.api_proxy_port, 30)


def run_scenario(stack: b.Stack, cfg: b.Config, manifest: dict, scenario: dict, root: Path, run_id: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "scenario.json", scenario)
    restart_proxy(stack, cfg, root, scenario["bandwidth_mbps"])
    event_log = EventLog(root / "events.jsonl", run_id, scenario["scenario_id"])
    sampler = PowerSampler(root / "power.csv")
    sampler.start()
    start_ns = time.monotonic_ns()
    cache_log, source_log, sink_log = stack.run_root / "lmcache.log", stack.run_root / "source.log", stack.run_root / "sink.log"
    runtime = None
    sleeping = False
    try:
        try:
            b.flush_lmcache(stack)
            b.reset_vllm_caches(cfg, (source_log, sink_log))
        except Exception as exc:
            raise ScenarioResetError(str(exc)) from exc
        rows = {row["id"]: row for row in manifest["sessions"]}
        sessions = {item["session_id"]: LiveSession(cfg, rows[item["session_id"]], item["turn_index"], event_log, source_log, cache_log, scenario["deadline_s"]) for item in scenario["sessions"]}
        method_by_session = {row["session_id"]: row["method"] for row in scenario["moves"]} or {session_id: scenario["method"] for session_id in sessions}
        replay = [session for session_id, session in sessions.items() if method_by_session[session_id] == "replay"]
        kv = [session for session in sessions.values() if session not in replay]
        for session in replay:
            session.warm()
        if replay:
            b.flush_lmcache(stack)
        for session in kv:
            session.warm()
        moves = [Move(row["session_id"], row["method"], row["order"]) for row in scenario["moves"]]
        runtime = LiveRuntime(sessions, cfg, scenario["activity"], sink_log, cache_log, event_log, root / "requests.jsonl")
        if scenario["kind"] == "migration":
            migration_results = MigrationController(runtime, scenario["concurrency"]).run(moves)
            if any(not row.succeeded for row in migration_results):
                raise RuntimeError("; ".join(row.error for row in migration_results if row.error))
        else:
            migration_results = []
            if scenario["activity"] == "one_turn":
                with ThreadPoolExecutor(max_workers=scenario["concurrency"]) as pool:
                    for session in sessions.values():
                        session.start_activity()
                    list(pool.map(lambda session: session.wait_activity(), sessions.values()))
        full_drain = scenario["kind"] == "migration" and set(sessions) == {row["id"] for row in manifest["sessions"]} and all(row.succeeded for row in migration_results)
        sleep_times = None
        if full_drain:
            sleep_start = time.monotonic_ns()
            b.set_source_sleep(cfg, True)
            sleeping = True
            sleep_end = time.monotonic_ns()
            sleep_times = [sleep_start, sleep_end]
            event_log.write("source_sleep", start_ns=sleep_start, end_ns=sleep_end)
        continuations = []
        for item in sorted(scenario["sessions"], key=lambda row: row["order"]):
            session = sessions[item["session_id"]]
            expected_port = cfg.api_proxy_port if scenario["kind"] == "migration" else cfg.src_port
            if session.route != expected_port:
                raise RuntimeError(f"wrong continuation route for {session.session_id}")
            committed_hash = messages_hash(session.messages)
            request = session.continuation()
            continuations.append({"session_id": session.session_id, "route_port": session.route, "committed_context_hash": committed_hash, **asdict(request)})
        activities = []
        for session in sessions.values():
            if not session.activity_times:
                continue
            activity_start, activity_end = session.activity_times
            move_result = next((row for row in migration_results if row.move.session_id == session.session_id), None)
            activities.append({
                "session_id": session.session_id, "start_ns": activity_start, "end_ns": activity_end,
                "overlapped_initial_copy": bool(move_result and activity_start < move_result.initial_end_ns and activity_end > move_result.initial_start_ns),
                "overlapped_request_wait": bool(move_result and activity_start < move_result.idle_ns and activity_end > move_result.pause_start_ns),
            })
        elapsed_s = (time.monotonic_ns() - start_ns) / 1e9
        result = {
            "schema": RESULT_SCHEMA, "scenario_id": scenario["scenario_id"], "status": "complete",
            "started_ns": start_ns, "ended_ns": time.monotonic_ns(), "elapsed_s": elapsed_s,
            "deadline_s": scenario["deadline_s"], "deadline_met": elapsed_s <= scenario["deadline_s"],
            "full_drain": full_drain, "source_sleep_ns": sleep_times,
            "migrations": [asdict(row) for row in migration_results], "activities": activities, "continuations": continuations,
        }
    finally:
        if sleeping:
            b.set_source_sleep(cfg, False)
        if runtime:
            runtime.close()
        sampler.close()
        time.sleep(.3)
        event_log.close()
        write_cache_slice(cache_log, root / "cache_operations.jsonl", start_ns, time.monotonic_ns())
    result["wire_bytes"] = b.proxy_counts(root / "proxy_bytes.csv")
    write_json(root / "result.json", result)
    return result


def git_state(allow_dirty: bool) -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    if dirty and not allow_dirty:
        raise RuntimeError("formal runs require a clean worktree; pass --allow-dirty for development")
    return sha, dirty


class ScenarioResetError(RuntimeError):
    pass


def config_record(cfg: b.Config) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}


def merge_run_metadata(current: dict, previous: dict | None, resume_from: str | None) -> dict:
    if previous is None:
        return current
    provenance = {"git_sha", "git_shas"}
    core = lambda row: {key: value for key, value in row.items() if key not in provenance}
    if core(previous) != core(current):
        raise RuntimeError("run metadata changed; resume requires the same plan, manifest, and settings")
    prior = previous["git_sha"]
    if prior != current["git_sha"] and resume_from != prior:
        raise RuntimeError(f"run code changed; resume requires --resume-from-git-sha {prior}")
    current["git_shas"] = list(dict.fromkeys(previous.get("git_shas", [prior]) + current["git_shas"]))
    return current


def run_plan(plan_path: Path, run_root: Path, cfg: b.Config, allow_dirty: bool,
             extra: list[str], resume_from: str | None = None,
             power_state_cycles: int = 0, power_state_window_s: float = 60,
             node_power: bool = False) -> None:
    if power_state_cycles < 0 or power_state_window_s <= 0:
        raise ValueError("invalid power-state settings")
    if node_power:
        node_power_reading()
    plan = json.loads(plan_path.read_text())
    manifest_path = Path(plan["manifest"]["path"])
    if file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("manifest hash changed after planning")
    manifest = json.loads(manifest_path.read_text())
    validate_plan(plan, manifest)
    sha, dirty = git_state(allow_dirty)
    metadata = {"schema": RUN_SCHEMA, "plan_sha256": file_hash(plan_path), "plan_object_sha256": object_hash(plan), "manifest_sha256": file_hash(manifest_path), "git_sha": sha, "git_shas": [sha], "dirty": dirty, "config": config_record(cfg), "extra_vllm_args": extra, "power_state_cycles": power_state_cycles, "power_state_window_s": power_state_window_s, "node_power": node_power}
    metadata_path = run_root / "run_metadata.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    metadata = merge_run_metadata(metadata, previous, resume_from)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    write_json(run_root / "plan.json", plan)
    failures, stack, attempt = [], None, 0

    def start_stack(mbps: float):
        nonlocal attempt
        attempt += 1
        debug = run_root / "debug" / f"testbed_{attempt}"
        new_stack = b.start_stack(cfg, debug, mbps, extra)
        b.start_sink(new_stack, cfg, extra)
        return new_stack

    try:
        for scenario in plan["scenarios"]:
            root = run_root / "scenarios" / scenario["scenario_id"]
            if (root / "result.json").exists() and json.loads((root / "result.json").read_text()).get("status") == "complete":
                continue
            if stack is None:
                stack = start_stack(scenario["bandwidth_mbps"])
                power_result = run_root / "power_states" / "result.json"
                if power_state_cycles and not power_result.exists():
                    profile_power_states(
                        stack, cfg, power_result.parent, power_state_cycles,
                        power_state_window_s, node_power,
                    )
            for reset_attempt in range(2):
                try:
                    run_scenario(stack, cfg, manifest, scenario, root, metadata["plan_sha256"][:16])
                    break
                except ScenarioResetError:
                    b.stop_stack(stack)
                    stack = None
                    if reset_attempt:
                        raise
                    stack = start_stack(scenario["bandwidth_mbps"])
                except Exception as exc:
                    failures.append(scenario["scenario_id"])
                    write_json(root / "result.json", {"schema": RESULT_SCHEMA, "scenario_id": scenario["scenario_id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    b.stop_stack(stack)
                    stack = None
                    break
    finally:
        if stack:
            b.stop_stack(stack)
    if failures:
        raise RuntimeError(f"failed scenarios: {', '.join(failures)}")


def csv_value(value):
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: csv_value(value) for key, value in row.items()} for row in rows])


def duration(start, end) -> float:
    return (end - start) / 1e9 if start is not None and end is not None else 0.0


def first_stream_ns(request: dict) -> int:
    chunks = request.get("stream_chunks") or []
    return int(chunks[0]["monotonic_ns"] if chunks else request.get("first_byte_ns", request["end_ns"]))


def request_measurements(prefix: str, request: dict | None, controller_end_ns: int | None) -> dict:
    if not request:
        return {
            f"{prefix}_prompt_tokens": 0, f"{prefix}_processed_tokens": 0,
            f"{prefix}_kv_chunks": 0, f"{prefix}_kv_bytes": 0,
            f"{prefix}_request_s": 0.0, f"{prefix}_time_to_first_response_s": 0.0,
            f"{prefix}_response_s": 0.0, f"{prefix}_validation_s": 0.0,
        }
    first = first_stream_ns(request)
    return {
        f"{prefix}_prompt_tokens": int(request.get("prompt_tokens", 0)),
        f"{prefix}_processed_tokens": int(request.get("processed_tokens", 0)),
        f"{prefix}_kv_chunks": int(request.get("logical_kv_chunks", 0)),
        f"{prefix}_kv_bytes": int(request.get("logical_kv_bytes", 0)),
        f"{prefix}_request_s": duration(request["start_ns"], request["end_ns"]),
        f"{prefix}_time_to_first_response_s": duration(request["start_ns"], first),
        f"{prefix}_response_s": duration(first, request["end_ns"]),
        f"{prefix}_validation_s": duration(request["end_ns"], controller_end_ns),
    }


def flatten_migration(scenario: dict, result: dict, row: dict) -> dict:
    initial, catch_up = row.get("initial") or {}, row.get("catch_up") or {}
    session = next(item for item in scenario["sessions"] if item["session_id"] == row["move"]["session_id"])
    initial_metrics = request_measurements("initial", initial, row["initial_end_ns"])
    catch_up_metrics = request_measurements("catch_up", catch_up, row.get("catch_up_end_ns"))
    catch_up_metrics.pop("catch_up_processed_tokens")
    return {
        "scenario_id": scenario["scenario_id"], "match_id": scenario["match_id"], "job_class": session.get("job_class", ""),
        "session_id": row["move"]["session_id"], "method": row["move"]["method"], "order": row["move"]["order"],
        "concurrency": scenario["concurrency"], "bandwidth_mbps": scenario["bandwidth_mbps"], "activity": scenario["activity"], "repeat": scenario["repeat"],
        "success": not row.get("error"),
        "request_wait_s": duration(row["pause_start_ns"], row["idle_ns"]), "catch_up_s": duration(row.get("catch_up_start_ns"), row.get("catch_up_end_ns")),
        "service_pause_s": duration(row["pause_start_ns"], row["switch_end_ns"]), "route_switch_s": duration(row["switch_start_ns"], row["switch_end_ns"]),
        "measured_prompt_tokens": initial_metrics.pop("initial_prompt_tokens"),
        "measured_processed_tokens": initial_metrics.pop("initial_processed_tokens"),
        "measured_kv_chunks": initial_metrics.pop("initial_kv_chunks"),
        "measured_kv_bytes": initial_metrics.pop("initial_kv_bytes"),
        "catch_up_prompt_tokens": catch_up_metrics.pop("catch_up_prompt_tokens"),
        "catch_up_new_tokens": max(0, catch_up.get("prompt_tokens", 0) - initial.get("prompt_tokens", 0)),
        "catch_up_cache_hit_chunks": catch_up_metrics.pop("catch_up_kv_chunks"),
        "catch_up_cache_hit_bytes": catch_up_metrics.pop("catch_up_kv_bytes"),
        **initial_metrics, **catch_up_metrics,
        "initial_start_ns": row["initial_start_ns"], "initial_end_ns": row["initial_end_ns"], "pause_start_ns": row["pause_start_ns"], "idle_ns": row["idle_ns"], "catch_up_start_ns": row.get("catch_up_start_ns"), "catch_up_end_ns": row.get("catch_up_end_ns"), "switch_start_ns": row["switch_start_ns"], "switch_end_ns": row["switch_end_ns"],
    }


def network_measurements(path: Path) -> dict:
    rows = [row for row in b.proxy_rows(path) if row.get("billed") == "1" and row["route"] == "kv"
            and row["direction"] == "target_to_client" and int(row["bytes"]) > 0]
    if not rows:
        return {"measured_kv_wire_bytes": 0, "kv_network_window_s": 0.0,
                "measured_kv_throughput_mbps": 0.0}
    start = min(int(row["monotonic_ns"]) for row in rows)
    end = max(int(row["monotonic_ns"]) + int(row["interval_ns"]) for row in rows)
    nbytes, window = sum(int(row["bytes"]) for row in rows), duration(start, end)
    return {"measured_kv_wire_bytes": nbytes, "kv_network_window_s": window,
            "measured_kv_throughput_mbps": nbytes * 8 / window / 1e6}


def power_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [{**row, "monotonic_ns": int(row["monotonic_ns"]), "gpu": int(row["gpu"]),
                 "power_w": float(row["power_w"]), "utilization_pct": float(row["utilization_pct"]),
                 "memory_mib": float(row["memory_mib"])}
                for row in csv.DictReader(handle) if row["valid"] == "1"]


def power_measurements(path: Path, start_ns: int, end_ns: int,
                       baselines: tuple[float, float] | None = None) -> dict:
    rows, values = power_rows(path), {}
    gpus = sorted({row["gpu"] for row in rows})
    if len(gpus) != 2:
        raise RuntimeError(f"expected two measured GPUs in {path}, found {gpus}")
    for role, gpu, baseline in zip(("source", "destination"), gpus, baselines or (None, None)):
        before = [row["power_w"] for row in rows if row["gpu"] == gpu and row["monotonic_ns"] < start_ns]
        active = [row["power_w"] for row in rows if row["gpu"] == gpu and start_ns <= row["monotonic_ns"] <= end_ns]
        if not before or not active:
            raise RuntimeError(f"power samples do not cover migration window in {path}")
        baseline = statistics.median(before) if baseline is None else baseline
        mean = statistics.fmean(active)
        values.update({f"{role}_baseline_power_w": baseline, f"{role}_mean_power_w": mean,
                       f"{role}_added_power_w": mean - baseline,
                       f"{role}_added_energy_j": (mean - baseline) * duration(start_ns, end_ns)})
    values["total_added_energy_j"] = values["source_added_energy_j"] + values["destination_added_energy_j"]
    return values


def summary(values: list[float], seed: int = 0) -> dict:
    row = {"n": len(values), "median": statistics.median(values), "q25": quantile(values, .25), "q75": quantile(values, .75)}
    if len(values) >= 20:
        row["p95"] = quantile(values, .95)
    if len(values) >= 10:
        rng = random.Random(seed)
        medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(1000)]
        row.update({"median_ci_low": quantile(medians, .025), "median_ci_high": quantile(medians, .975)})
    return row


def plot_timeline(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    rows = result["migrations"]
    fig, ax = plt.subplots(figsize=(10, max(2.5, .55 * len(rows))))
    if rows:
        base = min(row["queued_ns"] for row in rows)
        colors = {"initial": "#4C78A8", "wait": "#F2CF5B", "catch": "#72B7B2", "switch": "#54A24B"}
        for y, row in enumerate(rows):
            spans = [(row["initial_start_ns"], row["initial_end_ns"], "initial"), (row["pause_start_ns"], row["idle_ns"], "wait"), (row.get("catch_up_start_ns"), row.get("catch_up_end_ns"), "catch"), (row["switch_start_ns"], row["switch_end_ns"], "switch")]
            for start, end, label in spans:
                if start is not None and end is not None:
                    ax.barh(y, (end - start) / 1e9, left=(start - base) / 1e9, color=colors[label], label=label if y == 0 else None)
            ax.barh(y, (row["switch_end_ns"] - row["pause_start_ns"]) / 1e9, left=(row["pause_start_ns"] - base) / 1e9, fill=False, edgecolor="red", linewidth=1.5)
        ax.set_yticks(range(len(rows)), [row["move"]["session_id"] for row in rows])
        for continuation in result.get("continuations", []):
            ax.plot((continuation["start_ns"] - base) / 1e9, -.6, marker="|", color="black")
    ax.set(xlabel="Seconds", ylabel="Session", title="Migration timeline")
    if rows:
        ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_resource(root: Path, result: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    proxy = b.proxy_rows(root / "proxy_bytes.csv")
    if proxy:
        base = min(int(row["monotonic_ns"]) for row in proxy)
        for route in sorted({row["route"] for row in proxy}):
            rows = [row for row in proxy if row["route"] == route and row["billed"] == "1"]
            axes[0].step([(int(row["monotonic_ns"]) - base) / 1e9 for row in rows], [int(row["bytes"]) * 8 / int(row["interval_ns"]) * 1e3 for row in rows], where="post", label=route)
        axes[0].legend(); axes[0].set_ylabel("Network (Mb/s)")
    for method, color in (("replay", "#4C78A8"), ("kv_transfer", "#F58518")):
        intervals = [(row["initial_start_ns"], row["switch_end_ns"]) for row in result["migrations"] if row["move"]["method"] == method]
        if intervals:
            points = sorted([(time_ns, delta) for start, end in intervals for time_ns, delta in ((start, 1), (end, -1))])
            active, xs, ys = 0, [], []
            base = min(start for start, _ in intervals)
            for time_ns, delta in points:
                active += delta; xs.append((time_ns - base) / 1e9); ys.append(active)
            axes[1].step(xs, ys, where="post", label=method, color=color)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    axes[1].set_ylabel("Active moves")
    power_path = root / "power.csv"
    if power_path.exists():
        power = list(csv.DictReader(power_path.open()))
        if power:
            base = min(int(row["monotonic_ns"]) for row in power)
            for gpu in sorted({row["gpu"] for row in power}):
                rows = [row for row in power if row["gpu"] == gpu and row["valid"] == "1"]
                values = [float(row["power_w"]) for row in rows]
                rolling = [statistics.median(values[max(0, index - 3):index + 1]) for index in range(len(values))]
                axes[2].plot([(int(row["monotonic_ns"]) - base) / 1e9 for row in rows], values, alpha=.3)
                axes[2].plot([(int(row["monotonic_ns"]) - base) / 1e9 for row in rows], rolling, label=f"GPU {gpu}")
            axes[2].legend()
    axes[2].set(xlabel="Seconds", ylabel="Power (W)")
    fig.tight_layout(); fig.savefig(root / "resource_trace.png", dpi=180); plt.close(fig)


def save_both(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))


def style_maps(rows: list[dict]) -> tuple[dict, dict]:
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    markers = ("o", "s", "^", "D", "P")
    links, widths = sorted({row["bandwidth_mbps"] for row in rows}), sorted({row["concurrency"] for row in rows})
    return ({value: colors[i % len(colors)] for i, value in enumerate(links)},
            {value: markers[i % len(markers)] for i, value in enumerate(widths)})


def add_legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def cross_plots(root: Path, migrations: list[dict], scenarios: list[dict]) -> None:
    import matplotlib.pyplot as plt
    if not migrations:
        return
    links, widths = style_maps(migrations)
    methods, activities = ("kv_transfer", "replay"), sorted({row["activity"] for row in migrations})

    fig, axes = plt.subplots(len(activities), 2, figsize=(12, 4.5 * len(activities)), squeeze=False)
    for i, activity in enumerate(activities):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            rows = [row for row in migrations if row["activity"] == activity and row["method"] == method]
            xkey, scale, xlabel = (("measured_kv_bytes", 1e9, "Measured initial KV bytes (GB)") if method == "kv_transfer"
                                  else ("measured_processed_tokens", 1e3, "Measured initial processed tokens (thousands)"))
            for link in sorted(links):
                for width in sorted(widths):
                    data = [row for row in rows if row["bandwidth_mbps"] == link and row["concurrency"] == width]
                    if data:
                        ax.scatter([row[xkey] / scale for row in data], [row["initial_time_to_first_response_s"] for row in data],
                                   color=links[link], marker=widths[width], alpha=.55, label=f"{link:g} Mb/s, {width} concurrent")
            ax.set(title=f"{method.replace('_', ' ')}, {activity.replace('_', ' ')}", xlabel=xlabel,
                   ylabel="Request to first streamed response (s)")
            add_legend(ax)
    fig.tight_layout(); save_both(fig, root / "initial_time"); plt.close(fig)

    migration = [row for row in scenarios if row["kind"] == "migration"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    kv = [row for row in migration if row["method"] == "kv_transfer" and row["measured_kv_throughput_mbps"] > 0]
    for width in sorted(widths):
        for activity, marker in zip(activities, ("o", "x")):
            data = [row for row in kv if row["concurrency"] == width and row["activity"] == activity]
            if data:
                axes[0].scatter([row["bandwidth_mbps"] for row in data], [row["measured_kv_throughput_mbps"] for row in data],
                                marker=marker, alpha=.55, label=f"{width} concurrent, {activity.replace('_', ' ')}")
    if kv:
        bounds = [min(row["bandwidth_mbps"] for row in kv), max(row["bandwidth_mbps"] for row in kv)]
        axes[0].plot(bounds, bounds, "k--", label="configured rate")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set(xlabel="Configured bandwidth (Mb/s)", ylabel="Measured KV throughput (Mb/s)", title="KV network rate")
    add_legend(axes[0])
    replay = [row for row in migration if row["method"] == "replay" and row["measured_replay_throughput_tokens_s"] > 0]
    offsets = {link: offset for link, offset in zip(sorted(links), (-.08, 0, .08, .16, -.16))}
    for link in sorted(links):
        for activity, marker in zip(activities, ("o", "x")):
            data = [row for row in replay if row["bandwidth_mbps"] == link and row["activity"] == activity]
            if data:
                axes[1].scatter([row["concurrency"] + offsets[link] for row in data],
                                [row["measured_replay_throughput_tokens_s"] for row in data],
                                color=links[link], marker=marker, alpha=.55, label=f"{link:g} Mb/s, {activity.replace('_', ' ')}")
    axes[1].set(xlabel="Concurrent requests", ylabel="Measured processed tokens/s", title="Replay processing rate")
    add_legend(axes[1]); fig.tight_layout(); save_both(fig, root / "throughput"); plt.close(fig)

    paired: dict[tuple, dict[int, list[float]]] = {}
    for row in migration:
        key = (row["method"], row["bandwidth_mbps"], row["activity"], row["repeat"],
               row["session_set"], row["measured_prompt_tokens"])
        paired.setdefault(key, {}).setdefault(row["concurrency"], []).append(row["migration_s"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, method in zip(axes, methods):
        for key, by_width in paired.items():
            if key[0] != method or 1 not in by_width:
                continue
            base = statistics.median(by_width[1])
            for width, values in by_width.items():
                if width > 1:
                    ax.scatter(width, base / statistics.median(values), color=links[key[1]], alpha=.5)
        limit = max(widths); ax.plot([1, limit], [1, limit], "k--", label="ideal")
        ax.axhline(1, color="grey", linewidth=1); ax.set(title=method.replace("_", " "), xlabel="Concurrency",
                                                          ylabel="Speedup from concurrency 1")
    axes[1].legend(handles=[plt.Line2D([], [], color=color, marker="o", linestyle="", label=f"{link:g} Mb/s")
                            for link, color in links.items()] + [plt.Line2D([], [], color="black", linestyle="--", label="ideal")], fontsize=8)
    fig.tight_layout(); save_both(fig, root / "concurrency_scaling"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for j, method in enumerate(methods):
        active = [row for row in migrations if row["method"] == method and row["activity"] == "one_turn"]
        for width in sorted(widths):
            data = [row for row in active if row["concurrency"] == width]
            axes[0][j].scatter([row["catch_up_new_tokens"] for row in data],
                               [row["catch_up_time_to_first_response_s"] for row in data], marker=widths[width], alpha=.5,
                               label=f"{width} concurrent")
            axes[1][j].scatter([row["catch_up_new_tokens"] for row in data], [row["service_pause_s"] for row in data],
                               marker=widths[width], alpha=.5, label=f"{width} concurrent")
        matched = [row for row in migration if row["method"] == method and row.get("continuation_difference_s") is not None]
        for activity in activities:
            data = [row for row in matched if row["activity"] == activity]
            axes[2][j].scatter([row["measured_prompt_tokens"] for row in data], [row["continuation_difference_s"] for row in data],
                               alpha=.5, label=activity.replace("_", " "))
        axes[0][j].set(title=method.replace("_", " "), xlabel="Measured new prompt tokens",
                       ylabel="Catch-up to first response (s)")
        axes[1][j].set(xlabel="Measured new prompt tokens", ylabel="Service pause (s)")
        axes[2][j].axhline(0, color="grey", linewidth=1)
        axes[2][j].set(xlabel="Measured initial prompt tokens", ylabel="Continuation time difference (s)")
        for ax in axes[:, j]:
            add_legend(ax)
    fig.tight_layout(); save_both(fig, root / "service_effects"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    method_colors = {"kv_transfer": "#F58518", "replay": "#4C78A8"}
    for method in methods:
        for width in sorted(widths):
            data = [row for row in migration if row["method"] == method and row["concurrency"] == width]
            axes[0].scatter([row["migration_s"] for row in data], [row["total_added_energy_j"] / 1000 for row in data],
                            color=method_colors[method], marker=widths[width], alpha=.5,
                            label=f"{method.replace('_', ' ')}, {width} concurrent")
    axes[0].set(xlabel="Migration time (s)", ylabel="Baseline-adjusted GPU energy (kJ)", title="Time and measured GPU energy")
    add_legend(axes[0])
    for ax, method, xkey, xlabel in ((axes[1], "kv_transfer", "measured_kv_throughput_mbps", "Measured KV throughput (Mb/s)"),
                                     (axes[2], "replay", "measured_replay_throughput_tokens_s", "Measured processed tokens/s")):
        for width in sorted(widths):
            data = [row for row in migration if row["method"] == method and row["concurrency"] == width and row[xkey] > 0]
            ax.scatter([row[xkey] for row in data], [row["destination_added_power_w"] for row in data],
                       marker=widths[width], alpha=.5, label=f"{width} concurrent")
        ax.set(xlabel=xlabel, ylabel="Baseline-adjusted destination power (W)", title=method.replace("_", " ")); add_legend(ax)
    fig.tight_layout(); save_both(fig, root / "power_energy"); plt.close(fig)

    checked = [row for row in migrations if row.get("current_model_time_s") is not None]
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in methods:
        data = [row for row in checked if row["method"] == method]
        ax.scatter([row["current_model_time_s"] for row in data], [row["initial_time_to_first_response_s"] for row in data],
                   color=method_colors[method], alpha=.6, label=method.replace("_", " "))
    if checked:
        limit = max(max(row["current_model_time_s"], row["initial_time_to_first_response_s"]) for row in checked)
        ax.plot([0, limit], [0, limit], "k--", label="equal")
    ax.set(xlabel="Current timing equation (s)", ylabel="Measured request to first response (s)",
           title="Concurrency 1, no activity, recorded valid range")
    add_legend(ax); fig.tight_layout(); save_both(fig, root / "model_check"); plt.close(fig)


def pooled_power_baselines(run_root: Path, plan: dict, results: dict[str, dict]) -> tuple[float, float]:
    pooled = ([], [])
    for scenario in plan["scenarios"]:
        moves = results[scenario["scenario_id"]].get("migrations", [])
        if not moves:
            continue
        start = min(row["initial_start_ns"] for row in moves)
        rows = power_rows(run_root / "scenarios" / scenario["scenario_id"] / "power.csv")
        gpus = sorted({row["gpu"] for row in rows})
        if len(gpus) != 2:
            raise RuntimeError(f"expected two measured GPUs for {scenario['scenario_id']}")
        for values, gpu in zip(pooled, gpus):
            values.extend(row["power_w"] for row in rows if row["gpu"] == gpu
                          and row["monotonic_ns"] < start and row["utilization_pct"] == 0)
    if not all(pooled):
        raise RuntimeError("no idle power samples before migrations")
    return tuple(statistics.median(values) for values in pooled)


def current_model_time(row: dict, profile) -> float | None:
    if row["concurrency"] != 1 or row["activity"] != "none":
        return None
    source = profile.sources[row["method"]]
    work = row["measured_kv_bytes"] if row["method"] == "kv_transfer" \
        else row["measured_prompt_tokens"]
    if not source.valid_range[0] <= work <= source.valid_range[1]:
        return None
    case = profile.case()
    if row["method"] == "kv_transfer":
        transfer = row["measured_kv_bytes"] / (row["bandwidth_mbps"] * 1e6 / 8)
        ingestion = row["measured_kv_bytes"] / case.kv_transfer.destination_bytes_per_s
        return case.kv_transfer.setup_s + max(transfer, ingestion) + case.kv_transfer.sync_s
    return row["measured_processed_tokens"] / case.replay.rate(row["measured_prompt_tokens"], 1)


def reduce_run(run_root: Path) -> None:
    metadata = json.loads((run_root / "run_metadata.json").read_text())
    if metadata.get("schema") != RUN_SCHEMA:
        raise ValueError("unsupported run schema")
    plan = json.loads((run_root / "plan.json").read_text())
    if object_hash(plan) != metadata.get("plan_object_sha256", object_hash(plan)):
        raise RuntimeError("plan changed while reducing")
    results = {}
    for scenario in plan["scenarios"]:
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing result for {scenario['scenario_id']}")
        result = json.loads(path.read_text())
        if result.get("schema") != RESULT_SCHEMA or result.get("scenario_id") != scenario["scenario_id"]:
            raise ValueError(f"invalid result for {scenario['scenario_id']}")
        results[scenario["scenario_id"]] = result
    baselines = pooled_power_baselines(run_root, plan, results)
    migrations, scenarios = [], []
    for scenario in plan["scenarios"]:
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        result = results[scenario["scenario_id"]]
        migration_rows = [flatten_migration(scenario, result, row) for row in result.get("migrations", [])]
        migrations.extend(migration_rows)
        continuation = [duration(row["start_ns"], row["first_byte_ns"]) for row in result.get("continuations", [])]
        row = {**{key: scenario[key] for key in ("scenario_id", "match_id", "kind", "method", "concurrency", "bandwidth_mbps", "activity", "repeat")},
               "session_set": ",".join(item["session_id"] for item in scenario["sessions"]),
               "status": result["status"], "elapsed_s": result.get("elapsed_s"),
               "deadline_s": result.get("deadline_s"), "deadline_met": result.get("deadline_met"),
               "continuation_ttft_s": statistics.median(continuation) if continuation else None,
               "measured_prompt_tokens": sum(item["measured_prompt_tokens"] for item in migration_rows),
               "measured_processed_tokens": sum(item["measured_processed_tokens"] for item in migration_rows),
               "measured_kv_payload_bytes": sum(item["measured_kv_bytes"] for item in migration_rows),
               "catch_up_new_tokens": sum(item["catch_up_new_tokens"] for item in migration_rows),
               "median_service_pause_s": statistics.median([item["service_pause_s"] for item in migration_rows]) if migration_rows else 0.0}
        if migration_rows:
            start = min(item["initial_start_ns"] for item in migration_rows)
            initial_end = max(item["initial_end_ns"] for item in migration_rows)
            first_end = max(first_stream_ns(item["initial"]) for item in result["migrations"])
            end = max(item["switch_end_ns"] for item in migration_rows)
            row.update({"migration_s": duration(start, end), "initial_requests_s": duration(start, initial_end),
                        "initial_time_to_first_responses_s": duration(start, first_end),
                        **network_measurements(path.parent / "proxy_bytes.csv"),
                        **power_measurements(path.parent / "power.csv", start, end, baselines)})
            row["measured_replay_throughput_tokens_s"] = (
                row["measured_processed_tokens"] / row["initial_time_to_first_responses_s"]
                if row["method"] == "replay" else 0.0)
        else:
            row.update({"migration_s": 0.0, "initial_requests_s": 0.0,
                        "initial_time_to_first_responses_s": 0.0, "measured_kv_wire_bytes": 0,
                        "kv_network_window_s": 0.0, "measured_kv_throughput_mbps": 0.0,
                        "measured_replay_throughput_tokens_s": 0.0})
        scenarios.append(row)
        if result["status"] == "complete":
            if not (path.parent / "migration_timeline.png").exists():
                plot_timeline(result, path.parent / "migration_timeline.png")
            if not (path.parent / "resource_trace.png").exists():
                plot_resource(path.parent, result)
    controls = {row["match_id"]: row for row in scenarios if row["kind"] == "control" and row["continuation_ttft_s"] is not None}
    for row in scenarios:
        control = controls.get(row["match_id"])
        if row["kind"] == "migration" and control and row["continuation_ttft_s"] is not None:
            row["continuation_difference_s"] = row["continuation_ttft_s"] - control["continuation_ttft_s"]
    from profiles import ModelProfile
    profile = ModelProfile.load(Path(__file__).with_name("profiles") / "gpt_oss_20b_a100_tp1.json")
    for row in migrations:
        row["current_model_time_s"] = current_model_time(row, profile)
    write_csv(run_root / "migrations.csv", migrations)
    write_csv(run_root / "scenarios.csv", scenarios)
    groups: dict[tuple, list[dict]] = {}
    for row in scenarios:
        if row["kind"] == "migration":
            key = tuple(row[name] for name in ("method", "concurrency", "bandwidth_mbps", "activity"))
            groups.setdefault(key, []).append(row)
    metrics = ("migration_s", "initial_time_to_first_responses_s", "measured_kv_throughput_mbps",
               "measured_replay_throughput_tokens_s", "median_service_pause_s", "total_added_energy_j")
    benchmark = []
    for key, rows in sorted(groups.items()):
        for metric in metrics:
            if metric == "measured_kv_throughput_mbps" and key[0] != "kv_transfer" \
                    or metric == "measured_replay_throughput_tokens_s" and key[0] != "replay":
                continue
            values = [row[metric] for row in rows if row.get(metric) is not None and
                      (row[metric] > 0 or metric in {"median_service_pause_s", "total_added_energy_j"})]
            if values:
                benchmark.append({"method": key[0], "concurrency": key[1], "bandwidth_mbps": key[2],
                                  "activity": key[3], "measurement": metric, **summary(values, stable_seed(*key, metric))})
    write_csv(run_root / "benchmark_summary.csv", benchmark)
    cross_plots(run_root, migrations, scenarios)


def csv_list(value: str, cast=str):
    return [cast(item) for item in value.split(",") if item]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Profile live session migration")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("make-manifest")
    command.add_argument("--input", type=Path, required=True); command.add_argument("--out", type=Path, required=True)
    command.add_argument("--workload", required=True); command.add_argument("--sessions", type=int, required=True); command.add_argument("--seed", type=int, required=True)
    command = sub.add_parser("make-plan")
    command.add_argument("--manifest", type=Path, required=True); command.add_argument("--out", type=Path, required=True)
    command.add_argument("--context-sizes", type=lambda value: csv_list(value, int), required=True)
    command.add_argument("--concurrency", type=lambda value: csv_list(value, int), required=True)
    command.add_argument("--bandwidth-mbps", type=lambda value: csv_list(value, float), required=True)
    command.add_argument("--methods", type=lambda value: csv_list(value), default=list(METHODS))
    command.add_argument("--activity", type=lambda value: csv_list(value), default=list(ACTIVITIES))
    command.add_argument("--session-ids", type=lambda value: csv_list(value), default=[])
    command.add_argument("--repeats", type=int, required=True); command.add_argument("--seed", type=int, required=True); command.add_argument("--deadline-s", type=float, default=300)
    command = sub.add_parser("run")
    command.add_argument("--plan", type=Path, required=True); command.add_argument("--run-root", type=Path, required=True); command.add_argument("--allow-dirty", action="store_true"); command.add_argument("--resume-from-git-sha")
    command.add_argument("--power-state-cycles", type=int, default=0)
    command.add_argument("--power-state-window-s", type=float, default=60)
    command.add_argument("--node-power", action="store_true")
    b.add_common(command); command.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    command = sub.add_parser("reduce"); command.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "make-manifest":
        write_json(args.out, make_manifest(args.input, args.workload, args.sessions, args.seed))
    elif args.command == "make-plan":
        write_json(args.out, make_plan(
            args.manifest, args.context_sizes, args.concurrency, args.bandwidth_mbps,
            args.methods, args.activity, args.repeats, args.seed, args.deadline_s,
            args.session_ids,
        ))
    elif args.command == "run":
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        run_plan(
            args.plan, args.run_root, b.config_from_args(args), args.allow_dirty,
            extra, args.resume_from_git_sha, args.power_state_cycles,
            args.power_state_window_s, args.node_power,
        )
    else:
        reduce_run(args.run_root)


if __name__ == "__main__":
    main()
