from __future__ import annotations

import argparse, dataclasses, hashlib, http.client, json, re, subprocess, time, traceback
from pathlib import Path
import stage1b_drain_sink as b
import stage1g_mp_campaign as g

TARGETS = (12288, 13653, 15018, 16384)
EXPECTED_WAN_BLOCKS = (48, 5, 5, 6)
MODEL_RE = re.compile(r"Registered non-GPU context.*model=([^,]+), world_size=(\d+)")


def acceptance(stages: list[dict]) -> dict:
    prior_source, prior_wan = set(), set()
    gates = dict.fromkeys(("incremental_wire_transfer", "no_duplicate_prefix_traffic",
                           "exact_target_accounting", "exact_wire_accounting", "complete_state"), True)
    gates["real_continuation"] = bool(stages and stages[-1]["continuation_ok"])
    for stage in stages:
        source, new = set(stage["source_block_keys"]), set(stage["new_source_block_keys"])
        wan, before, after = stage["wan_get_block_keys"], set(stage["target_before_keys"]), set(stage["target_after_keys"])
        wan_set = set(wan)
        gates["incremental_wire_transfer"] &= wan_set == new == source - before and len(wan) == len(wan_set)
        gates["no_duplicate_prefix_traffic"] &= not wan_set & prior_wan
        gates["exact_target_accounting"] &= stage["vllm_local_cached_tokens"] + stage["lmcache_retrieved_tokens"] == stage["reported_cached_tokens"]
        gates["exact_wire_accounting"] &= stage["wan_payload_bytes"] == len(wan) * stage["measured_block_bytes"]
        gates["complete_state"] &= after == before | wan_set == source and stage["reported_cached_tokens"] == stage["complete_cacheable_source_prefix"]
        prior_source, prior_wan = source, prior_wan | wan_set
    observed = [len(stage["wan_get_block_keys"]) for stage in stages]
    gates["expected_block_counts"] = observed == list(EXPECTED_WAN_BLOCKS)
    return {"ok": all(gates.values()), "gates": gates, "observed_wan_blocks": observed}


def http_json(host, port, method, path, payload=None, statuses=(200,)):
    conn = http.client.HTTPConnection(host, port, timeout=600)
    try:
        conn.request(method, path, json.dumps(payload) if payload is not None else None,
                     {"Content-Type": "application/json"})
        response, text = conn.getresponse(), None
        text = response.read().decode()
    finally:
        conn.close()
    if response.status not in statuses:
        raise RuntimeError(f"{method} {path} failed {response.status}: {text[:500]}")
    return json.loads(text)


def chat_tokens(cfg, prompt):
    result = http_json(cfg.host, cfg.src_port, "POST", "/tokenize",
                       {"model": cfg.model, "messages": [{"role": "user", "content": prompt}],
                        "add_generation_prompt": True})
    tokens = result.get("tokens")
    if not tokens or len(tokens) != result.get("count"):
        raise RuntimeError(f"vLLM did not return exact chat token IDs: {result.keys()}")
    return tokens


def warm_prefetch(cfg, tokens, model, world_size):
    result = http_json(cfg.host, cfg.sink_lmc_http_port, "POST", "/cache/prefetches",
                       {"model_name": model, "world_size": world_size, "token_ids": tokens}, (202,))
    request_id = result.get("request_id")
    if not request_id:
        raise RuntimeError(f"warm prefetch was not submitted: {result}")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = http_json(cfg.host, cfg.sink_lmc_http_port, "GET", f"/cache/prefetches/{request_id}")
        if status["status"] == "completed":
            return status
        time.sleep(.05)
    raise TimeoutError(f"warm prefetch {request_id} did not complete")


def post_messages(cfg, messages):
    started = time.time()
    parsed = http_json(cfg.host, cfg.api_proxy_port, "POST", "/v1/chat/completions",
                       {"model": cfg.model, "messages": messages, "max_tokens": 64,
                        "temperature": 0, "reasoning_effort": "low"})
    return {"status": 200, "id": parsed["id"], "usage": parsed["usage"],
            "content": parsed["choices"][0]["message"].get("content") or "",
            "elapsed_s": time.time() - started}


def gets(rows):
    return [row for row in rows if row["command"] == "GET" and int(row["payload_bytes"]) > 0]


def provenance(cfg):
    digest = hashlib.sha256()
    with cfg.sandbox.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    vllm, lmcache = b.runtime_versions(cfg)
    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "image_path": str(cfg.sandbox), "image_sha256": digest.hexdigest(),
            "vllm_version": vllm, "lmcache_version": lmcache,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()},
            "bandwidth_mbps": 10000, "lmcache_connector": "LMCacheMPConnector",
            "lmcache_transfer_mode": "engine_driven", "block_tokens": g.BLOCK_TOKENS,
            "targets": list(TARGETS)}


def incremental(cfg, stack):
    g.reset(stack, cfg)
    transfers, log = stack.run_root / "resp_transfers.csv", stack.run_root / "lmcache-sink.log"
    match = MODEL_RE.search(log.read_text(errors="ignore"))
    if not match:
        raise RuntimeError("LMCache sink did not report its model layout")
    model, world_size = match.group(1), int(match.group(2))
    source_keys, target_keys, gpu_keys, stages = set(), set(), set(), []
    session, code, final_prompt, final_response = "mp-incremental", "QHMPSTAGEC0DE", "", None
    for index, target in enumerate(TARGETS):
        prompt, prompt_tokens = g.prompt_at(cfg, session, code, target)
        final_prompt = prompt
        offset = len(g.rows(transfers))
        source_result, _ = b.warm_source(cfg, stack.run_root, prompt, f"source stage {target}")
        rows = g.rows(transfers)
        stored = {row["key_hashes"] for row in rows[offset:] if row["command"] == "SET"}
        new_source = stored - source_keys
        source_keys |= stored
        if len(new_source) != EXPECTED_WAN_BLOCKS[index]:
            raise RuntimeError(f"stage {target} created {len(new_source)} blocks, expected {EXPECTED_WAN_BLOCKS[index]}")
        offset = len(rows)
        warm = warm_prefetch(cfg, chat_tokens(cfg, prompt), model, world_size)
        rows = g.rows(transfers)
        warm_gets = gets(rows[offset:])
        log_offset, offset = log.stat().st_size, len(rows)
        result = g.post(cfg, prompt, code)
        final_response = result
        time.sleep(.5)
        rows = g.rows(transfers)
        request_gets = gets(rows[offset:])
        cache = g.cache_report(log, log_offset, {result["id"]})
        wan_rows, missing_gpu = warm_gets + request_gets, source_keys - gpu_keys
        wan_keys = [row["key_hashes"] for row in wan_rows]
        sizes = {int(row["payload_bytes"]) for row in wan_rows}
        l1 = sum(value[2] for value in cache["requests"].values())
        if len(sizes) != 1 or l1 != len(missing_gpu) or cache["l2_blocks"] or request_gets:
            raise RuntimeError(f"stage {target} not exact L1-only: sizes={sizes}, L1={l1}, expected={len(missing_gpu)}, L2={cache['l2_blocks']}, request_WAN={len(request_gets)}")
        if warm["found_keys"] != warm["total_keys"] or warm["total_keys"] != len(source_keys):
            raise RuntimeError(f"stage {target} warm prefetch incomplete: {warm}")
        reported, retrieved = g.cached_tokens(result), cache["retrieved_tokens"]
        stages.append({"target_tokens": target, "prompt_tokens": prompt_tokens,
            "source_block_keys": sorted(source_keys), "newly_stored_source_block_keys": sorted(new_source),
            "new_source_block_keys": sorted(new_source), "destination_l1_hit_keys": sorted(missing_gpu),
            "destination_l1_hit_blocks": l1, "destination_l2_prefetched_block_keys": wan_keys,
            "warm_prefetch_status": warm, "wan_get_block_keys": wan_keys,
            "wan_payload_bytes": sum(int(row["payload_bytes"]) for row in wan_rows),
            "wan_response_wire_bytes": sum(int(row["response_wire_bytes"]) for row in wan_rows),
            "measured_block_bytes": next(iter(sizes)), "target_before_keys": sorted(target_keys),
            "target_after_keys": sorted(target_keys | set(wan_keys)),
            "vllm_local_cached_tokens": reported - retrieved, "lmcache_retrieved_tokens": retrieved,
            "reported_cached_tokens": reported, "complete_cacheable_source_prefix": len(source_keys) * g.BLOCK_TOKENS,
            "source_result": source_result, "destination_result": result,
            "continuation_ok": index < len(TARGETS) - 1})
        target_keys |= set(wan_keys)
        gpu_keys = set(source_keys)
    offset = len(g.rows(transfers))
    continuation = post_messages(cfg, [{"role": "user", "content": final_prompt},
        {"role": "assistant", "content": final_response["content"]},
        {"role": "user", "content": "What is the session state code? End with that code."}])
    continuation["code_ok"] = code in continuation["content"]
    time.sleep(.5)
    stages[-1].update(continuation_ok=continuation["code_ok"], continuation=continuation,
                      continuation_wan_get_block_keys=[row["key_hashes"] for row in gets(g.rows(transfers)[offset:])])
    return stages, acceptance(stages)


def concurrency_four(cfg, stack):
    g.reset(stack, cfg)
    sessions = [(f"mp-c4-{i}", f"QHC4{i}C0DE") for i in range(4)]
    prompts = {session: g.prompt_at(cfg, session, code, 16384)[0] for session, code in sessions}
    owners = {}
    warms = [g.add_source_keys(stack, cfg, prompts[session], session, owners) for session, _ in sessions]
    result = g.measure(cfg, stack, [(prompts[s], c) for s, c in sessions], owners, 4, False)
    result.update(sessions=warms)
    result["ok"] = result["distinct_session_overlap"] >= 2 and result["throughput_bps"] >= 1e9 and result["accounting_ok"] and result["wire_ok"] and result["continuations_ok"]
    return result


def write_report(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def run(root):
    cfg, report = b.Config(), root / "report.json"
    stack = b.start_stack(cfg, root, 10000)
    data = {"schema": "queue-haul-mp-incremental-v1", "provenance": provenance(cfg)}
    try:
        b.start_sink(stack, cfg)
        stages, staged = incremental(cfg, stack)
        data.update(stages=stages, incremental_acceptance=staged)
        write_report(report, data)
        if not staged["ok"]:
            raise RuntimeError(f"incremental staging rejected: {staged}")
        data["concurrency_four"] = concurrency_four(cfg, stack)
        data["ok"] = staged["ok"] and data["concurrency_four"]["ok"]
        write_report(report, data)
        if not data["ok"]:
            raise RuntimeError("concurrency-four confirmation failed")
        return data
    finally:
        b.stop_stack(stack)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({"ok": run(args.run_root)["ok"]}, indent=2))
    except BaseException:
        args.run_root.mkdir(parents=True, exist_ok=True)
        (args.run_root / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
