from __future__ import annotations

import argparse
import csv
import http.client
import json
import re
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import stage1b_drain_sink as b

BLOCK_TOKENS = 256
PREFETCH = re.compile(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\).*external_request_id=([^,\)]+)")
RETRIEVED = re.compile(r"Retrieved (\d+) tokens")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def max_distinct_overlap(xs: list[dict], owners: dict[str, str]) -> int:
    events = []
    for row in xs:
        if int(row["payload_bytes"]) > 0:
            owner = owners[row["key_hashes"]]
            events += [(int(row["start_ns"]), 1, owner), (int(row["end_ns"]), -1, owner)]
    counts: dict[str, int] = {}
    peak = 0
    for _, delta, owner in sorted(events, key=lambda item: (item[0], -item[1])):
        counts[owner] = counts.get(owner, 0) + delta
        if not counts[owner]:
            del counts[owner]
        peak = max(peak, len(counts))
    return peak


def reduce_transfers(xs: list[dict], owners: dict[str, str]) -> dict:
    gets = [row for row in xs if row["command"] == "GET" and int(row["payload_bytes"]) > 0]
    if any(row["key_hashes"] not in owners for row in gets):
        raise RuntimeError("a remotely loaded block has no unique source-session owner")
    sizes = {int(row["payload_bytes"]) for row in gets}
    if len(sizes) > 1:
        raise RuntimeError(f"remote KV blocks have inconsistent payload sizes: {sizes}")
    for row in gets:
        payload = int(row["payload_bytes"])
        if int(row["response_wire_bytes"]) != payload + len(str(payload)) + 5:
            raise RuntimeError("RESP wire bytes do not equal the remotely loaded body plus framing")
    return {
        "remote_blocks": len(gets),
        "remote_payload_bytes": sum(int(row["payload_bytes"]) for row in gets),
        "remote_wire_bytes": sum(int(row["response_wire_bytes"]) for row in gets),
        "block_payload_bytes": next(iter(sizes), 0),
        "distinct_session_overlap": max_distinct_overlap(gets, owners),
    }


def request_json(cfg: b.Config, port: int, path: str, payload: dict) -> dict:
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(cfg.host, port, timeout=600)
    try:
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        text = response.read().decode()
    finally:
        conn.close()
    if response.status != 200:
        raise RuntimeError(f"POST {path} failed {response.status}: {text[:500]}")
    return json.loads(text)


def token_count(cfg: b.Config, prompt: str) -> int:
    return request_json(cfg, cfg.src_port, "/tokenize", {"model": cfg.model, "prompt": prompt})["count"]


def prompt_at(cfg: b.Config, session: str, code: str, target: int) -> tuple[str, int]:
    prefix = f"Session {session}. The state code is {code}. Always end every answer with {code}. Body: "
    words = [f"{code}_{session}_{i % 97}" for i in range(target)]
    lo, hi = 0, target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_count(cfg, prefix + " ".join(words[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    prompt = prefix + " ".join(words[:lo])
    count = token_count(cfg, prompt)
    if target - count >= BLOCK_TOKENS:
        raise RuntimeError(f"could not construct a {target}-token session: {count}")
    return prompt, count


def post(cfg: b.Config, prompt: str, code: str, port: int | None = None) -> dict:
    start = time.monotonic_ns()
    result = b.post_chat(cfg, port or cfg.api_proxy_port, prompt, 1024)
    result["start_ns"], result["end_ns"], result["code_ok"] = start, time.monotonic_ns(), code in result["content"]
    b.check_chat(result, code)
    return result


def parallel_posts(cfg: b.Config, items: list[tuple[str, str]], concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(post, cfg, prompt, code) for prompt, code in items]
        return [future.result() for future in futures]


def add_source_keys(stack: b.Stack, cfg: b.Config, prompt: str, session: str,
                    owners: dict[str, str]) -> dict:
    path = stack.run_root / "resp_transfers.csv"
    offset = len(rows(path))
    result, _ = b.warm_source(cfg, stack.run_root, prompt, f"warm {session}")
    new = rows(path)[offset:]
    keys = {row["key_hashes"] for row in new if row["command"] == "SET"}
    if not keys:
        raise RuntimeError(f"{session} stored no remote blocks")
    for key in keys:
        if key in owners and owners[key] != session:
            raise RuntimeError(f"KV key shared by distinct session bodies: {key}")
        owners[key] = session
    return {"session": session, "stored_blocks": len(keys), "source": result}


def cache_report(log: Path, offset: int, request_ids: set[str]) -> dict:
    text = log.read_text(errors="ignore")[offset:]
    found = {}
    for match in PREFETCH.finditer(text):
        retained, queried, l1, l2, external_id = match.groups()
        matches = [request_id for request_id in request_ids
                   if external_id == request_id or external_id.startswith(request_id + "-")]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous LMCache request id: {external_id}")
        if matches:
            found[matches[0]] = tuple(map(int, (retained, queried, l1, l2)))
    missing = request_ids - found.keys()
    if missing:
        raise RuntimeError(f"LMCache did not report requests: {sorted(missing)}")
    retrieved = sum(map(int, RETRIEVED.findall(text)))
    total_blocks = sum(value[0] for value in found.values())
    l2_blocks = sum(value[3] for value in found.values())
    return {"requests": found, "prefetched_tokens": total_blocks * BLOCK_TOKENS,
            "loaded_tokens": retrieved, "l2_blocks": l2_blocks, "retrieved_tokens": retrieved}


def usage_tokens(result: dict) -> int:
    value = result["usage"].get("prompt_tokens")
    if value is None:
        raise RuntimeError("vLLM omitted prompt_tokens usage")
    return int(value)


def cached_tokens(result: dict) -> int:
    details = result["usage"].get("prompt_tokens_details") or {}
    if details.get("cached_tokens") is None:
        raise RuntimeError("vLLM omitted reported cached_tokens")
    return int(details["cached_tokens"])


def measure(cfg: b.Config, stack: b.Stack, items: list[tuple[str, str]], owners: dict[str, str],
            concurrency: int, continuation: bool) -> dict:
    path = stack.run_root / "resp_transfers.csv"
    log = stack.run_root / "lmcache-sink.log"
    transfer0, log0 = len(rows(path)), log.stat().st_size
    initial = parallel_posts(cfg, items, concurrency)
    initial_end = len(rows(path))
    follow = parallel_posts(cfg, [(prompt + f"\nContinue and end with {code}.", code) for prompt, code in items], concurrency) if continuation else []
    time.sleep(1)
    transfer_rows = rows(path)
    initial_rows = transfer_rows[transfer0:initial_end]
    measured_rows = transfer_rows[transfer0:]
    requests = initial + follow
    ids = {result["id"] for result in requests}
    if "" in ids:
        raise RuntimeError("vLLM omitted a request id")
    cache = cache_report(log, log0, ids)
    transfer = reduce_transfers(measured_rows, owners)
    initial_transfer = reduce_transfers(initial_rows, owners)
    reported_cached = sum(cached_tokens(result) for result in requests)
    local = reported_cached - cache["loaded_tokens"]
    cacheable = sum(usage_tokens(result) // BLOCK_TOKENS * BLOCK_TOKENS for result in requests)
    explained = local + cache["loaded_tokens"]
    transfer_window = ([int(row["start_ns"]) for row in initial_rows if row["command"] == "GET" and int(row["payload_bytes"]) > 0],
                       [int(row["end_ns"]) for row in initial_rows if row["command"] == "GET" and int(row["payload_bytes"]) > 0])
    duration = (max(transfer_window[1]) - min(transfer_window[0])) / 1e9 if transfer_window[0] else 0
    accounting_ok = (local >= 0 and explained == reported_cached and
                     explained >= cacheable - BLOCK_TOKENS * len(requests) and
                     explained <= cacheable)
    wire_ok = (transfer["remote_blocks"] == cache["l2_blocks"] and
               transfer["remote_payload_bytes"] == transfer["remote_blocks"] * transfer["block_payload_bytes"])
    return {
        "requests": requests,
        "vllm_cached_tokens": reported_cached,
        "vllm_local_tokens": local,
        "lmcache_loaded_tokens": cache["loaded_tokens"],
        "lmcache_prefetched_tokens": cache["prefetched_tokens"],
        "lmcache_retrieved_tokens": cache["retrieved_tokens"],
        "target_cacheable_tokens": cacheable,
        "accounting_ok": accounting_ok,
        "wire_ok": wire_ok,
        "continuations_ok": all(result["code_ok"] for result in requests),
        "throughput_bps": initial_transfer["remote_payload_bytes"] / duration if duration else 0,
        **transfer,
        "distinct_session_overlap": initial_transfer["distinct_session_overlap"],
    }


def reset(stack: b.Stack, cfg: b.Config) -> None:
    b.flush_lmcache(stack, cfg)
    b.reset_vllm_caches(cfg, (stack.run_root / "source.log", stack.run_root / "sink.log"))


def acceptance(parallel: list[dict], append: list[dict]) -> dict:
    serial = max(row["throughput_bps"] for row in parallel if row["concurrency"] == 1)
    medians = {c: statistics.median(row["throughput_bps"] for row in parallel if row["concurrency"] == c) for c in (2, 4)}
    best = max(medians.values())
    gates = {
        "three_repeats_each": len(parallel) == 9 and len(append) == 3,
        "distinct_session_overlap": max(row["distinct_session_overlap"] for row in parallel) >= 2,
        "aggregate_throughput_gain": best > serial,
        "token_accounting": all(row["accounting_ok"] for row in parallel + append),
        "remote_wire_accounting": all(row["wire_ok"] for row in parallel + append),
        "continuations_correct": all(row["continuations_ok"] for row in parallel + append),
    }
    return {"ok": all(gates.values()), "gates": gates, "serialized_ceiling_bps": serial,
            "parallel_median_bps": medians, "best_parallel_median_bps": best}


def write_report(path: Path, parallel: list[dict], append: list[dict], final: dict | None = None) -> None:
    path.write_text(json.dumps({"schema": "queue-haul-mp-campaign-v1", "bandwidth_mbps": 10000,
                                "parallel": parallel, "append": append, "acceptance": final}, indent=2, sort_keys=True))


def run(root: Path) -> dict:
    cfg = b.Config()
    stack = b.start_stack(cfg, root, 10000)
    parallel, append = [], []
    report = root / "report.json"
    try:
        b.start_sink(stack, cfg)
        sessions = [(f"mp16k-{i}", f"QHMP{i}C0DE") for i in range(4)]
        prompts = {session: prompt_at(cfg, session, code, 16384)[0] for session, code in sessions}
        for concurrency in (1, 2, 4):
            for repeat in range(3):
                reset(stack, cfg)
                owners, warms = {}, []
                for session, _ in sessions:
                    warms.append(add_source_keys(stack, cfg, prompts[session], session, owners))
                result = measure(cfg, stack, [(prompts[session], code) for session, code in sessions], owners, concurrency, False)
                result.update({"repeat": repeat, "concurrency": concurrency, "sessions": warms})
                parallel.append(result)
                write_report(report, parallel, append)
        for repeat in range(3):
            reset(stack, cfg)
            session, code, owners, stages = f"append-{repeat}", f"QHAP{repeat}C0DE", {}, []
            for target in (12288, 13653, 15018, 16384):
                prompt, tokens = prompt_at(cfg, session, code, target)
                warm = add_source_keys(stack, cfg, prompt, session, owners)
                stage = measure(cfg, stack, [(prompt, code)], owners, 1, False)
                stage.update({"target_tokens": target, "prompt_tokens": tokens, "source": warm})
                stages.append(stage)
            result = {"repeat": repeat, "stages": stages,
                      "accounting_ok": all(stage["accounting_ok"] for stage in stages),
                      "wire_ok": all(stage["wire_ok"] for stage in stages),
                      "continuations_ok": all(stage["continuations_ok"] for stage in stages),
                      "throughput_bps": statistics.mean(stage["throughput_bps"] for stage in stages),
                      "distinct_session_overlap": 1}
            append.append(result)
            write_report(report, parallel, append)
        final = acceptance(parallel, append)
        write_report(report, parallel, append, final)
        if not final["ok"]:
            raise RuntimeError(f"MP campaign acceptance failed: {final}")
        return final
    finally:
        b.stop_stack(stack)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.run_root), indent=2, sort_keys=True))
    except BaseException:
        args.run_root.mkdir(parents=True, exist_ok=True)
        (args.run_root / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
