from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import migration_testbed as testbed
from migration_profiler import JOB_CLASSES, object_hash, write_json


SCHEMA = "queue-haul-destination-campaign-v1"
TRACE_SCHEMA = "queue-haul-trace-v1"
EVIDENCE = {
    "service_envelopes": "measure",
    "loaded_migration_slowdown": "measure",
    "foreground_impact": "measure",
    "kv_correctness": "measure",
    "continuation": "measure",
    "wan_capacity": "public_constant",
    "kv_bytes": "derive",
    "trace_growth": "derive",
    "residency_horizon": "derive",
    "headroom_grid": "derive",
    "old_service_results": "prior_only",
    "old_migration_results": "prior_only",
}


def audit_evidence(inventory: dict = EVIDENCE) -> dict:
    allowed = {"measure", "derive", "public_constant", "prior_only"}
    bad = sorted(set(inventory.values()) - allowed)
    if bad:
        raise ValueError(f"unclassified evidence: {bad}")
    return {"schema": SCHEMA, "inventory": inventory,
            "gpu_measurements": sorted(k for k, v in inventory.items() if v == "measure")}


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _messages(row: dict) -> list[dict]:
    value = next((row[k] for k in ("messages", "trajectory", "events") if row.get(k)), None)
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list):
        raise ValueError("trace row has no message list")
    return [{"role": str(m.get("role", m.get("type", ""))).lower(),
             "content": _text(m.get("content", m.get("message", m.get("text", "")))),
             **({"timestamp": m[k]} if (k := next((x for x in ("timestamp", "created_at", "time") if m.get(x) is not None), None)) else {})}
            for m in value]


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return __import__("datetime").datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def normalize_traces(rows: list[dict], source: str, revision: str, count_tokens) -> list[dict]:
    out = []
    for index, row in enumerate(rows):
        messages = _messages(row)
        session_id = str(next((row[k] for k in ("session_id", "id", "instance_id", "trace_id") if row.get(k) is not None), index))
        previous = 0
        for turn, stop in enumerate(i for i, m in enumerate(messages) if m["role"] == "assistant"):
            prefix, answer = messages[:stop], messages[stop]
            total = count_tokens(prefix)
            if total < 1 or total > 32768:
                continue
            timestamp = next((m.get("timestamp") for m in reversed(prefix) if m.get("timestamp") is not None), None)
            out.append({
                "schema": TRACE_SCHEMA, "source": source, "revision": revision,
                "license": "CC-BY-4.0", "session_id": f"{source}:{session_id}",
                "turn": turn, "time_s": _seconds(timestamp) if timestamp is not None else None,
                "input_tokens_total": total, "newly_append_tokens": max(1, total - previous),
                "output_tokens": max(1, count_tokens(answer["content"])),
                "current_user_message_count": sum(m["role"] == "user" for m in prefix),
                "tool_message_count": sum(m["role"] == "tool" for m in prefix),
                "reset": total < previous,
                "content_sha256": hashlib.sha256(json.dumps(messages[:stop + 1], sort_keys=True).encode()).hexdigest(),
            })
            previous = total
    return out


def _sessions(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    return grouped


def classify(rows: list[dict]) -> list[dict]:
    grouped = _sessions(rows)
    nvidia = {k for k in grouped if k.startswith("nvidia/")}
    timed = []
    for session_id, turns in grouped.items():
        times = sorted(r["time_s"] for r in turns if r["time_s"] is not None)
        if not session_id.startswith("nvidia/") and len(times) > 1:
            timed.append((session_id, (times[-1] - times[0]) / (len(times) - 1)))
    if len(timed) < 48 or len(nvidia) < 24:
        raise ValueError("need 48 timestamped Trace Commons and 24 NVIDIA sessions")
    interactive = {k for k, _ in sorted(timed, key=lambda x: (-x[1], x[0]))[:len(timed) // 2]}
    return [dict(row, job_class="agentic_tool_loop" if row["session_id"] in nvidia
                 else "interactive_coding" if row["session_id"] in interactive else "coding") for row in rows]


def build_manifests(rows: list[dict], seed: int = 0) -> dict:
    rows = classify(rows)
    grouped = _sessions(rows)
    manifests = {}
    for job_class in JOB_CLASSES:
        sessions = [(sid, sorted(turns, key=lambda r: r["turn"])) for sid, turns in grouped.items()
                    if turns[0]["job_class"] == job_class]
        sessions.sort(key=lambda item: (max(r["input_tokens_total"] for r in item[1]),
                                        object_hash([seed, item[0]])))
        if len(sessions) < 24:
            raise ValueError(f"need 24 {job_class} sessions, found {len(sessions)}")
        chosen = [sessions[round(i * (len(sessions) - 1) / 23)] for i in range(24)]
        manifests[job_class] = {
            split: [sid for i, (sid, _) in enumerate(chosen) if ("fit", "fit", "tune", "validation")[i % 4] == split]
            for split in ("fit", "tune", "validation")
        }
    return {"schema": SCHEMA, "trace_schema": TRACE_SCHEMA, "seed": seed,
            "rows_sha256": object_hash(sorted(rows, key=lambda r: (r["session_id"], r["turn"]))), "splits": manifests}


def token_counter(host: str, port: int, model: str):
    def count(value) -> int:
        payload = {"model": model, "add_generation_prompt": isinstance(value, list)}
        payload["messages" if isinstance(value, list) else "prompt"] = value
        result = testbed.http_json(host, port, "POST", "/tokenize", payload)
        if result.get("count") is None:
            raise RuntimeError("vLLM tokenizer returned no count")
        return int(result["count"])
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Destination measurement campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-evidence"); audit.add_argument("--out", type=Path, required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--trace-commons", type=Path, required=True); build.add_argument("--nvidia", type=Path, required=True)
    build.add_argument("--trace-revision", required=True); build.add_argument("--nvidia-revision", required=True)
    build.add_argument("--host", default="127.0.0.1"); build.add_argument("--port", type=int, default=8000)
    build.add_argument("--model", default=testbed.MODEL); build.add_argument("--seed", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-evidence":
        write_json(args.out, audit_evidence())
        return
    counter = token_counter(args.host, args.port, args.model)
    load = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = normalize_traces(load(args.trace_commons), "trace-commons/agent-traces", args.trace_revision, counter)
    rows += normalize_traces(load(args.nvidia), "nvidia/SWE-Hero-openhands-trajectories", args.nvidia_revision, counter)
    write_json(args.out, {"manifest": build_manifests(rows, args.seed), "traces": rows})


if __name__ == "__main__":
    main()
