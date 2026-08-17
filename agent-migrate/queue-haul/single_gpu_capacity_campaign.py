"""Non-gating single-A100 model/context capacity discovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import service_headroom_campaign as headroom


SCHEMA = "queue-haul-single-gpu-capacity-v1"
MODELS = tuple(testbed.MODEL_SPECS)
CONTEXTS = {
    "openai/gpt-oss-20b": (4096, 8192, 16384, 24576, 32256),
    "Qwen/Qwen3.8-27B": (3920, 7840, 15680, 24304, 32144),
    "google/gemma-4-26B-A4B-it": (4096, 8192, 16384, 24576, 32256),
}
WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OUTPUT_TOKENS = 32
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 256
REQUEST_TIMEOUT_S = 1800.0


class RuntimeContractError(RuntimeError):
    """A formal runtime proof failed; never reinterpret this as capacity."""


def digest(value) -> str:
    return profiler.object_hash(value)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_plan(seed: int = 1) -> dict:
    cells = [{
        "cell_id": f"a100-{slug(model)}-ctx{context}",
        "hardware": "a100", "model": model,
        "revision": testbed.model_spec(model).revision,
        "context_tokens": context,
    } for model in MODELS for context in CONTEXTS[model]]
    order = [row["cell_id"] for row in cells]
    random.Random(seed).shuffle(order)
    plan = {
        "schema": SCHEMA,
        "campaign": "single_gpu_capacity_discovery",
        "hardware": "a100",
        "models": list(MODELS),
        "model_revisions": {model: testbed.model_spec(model).revision
                            for model in MODELS},
        "contexts": {model: list(CONTEXTS[model]) for model in MODELS},
        "widths": list(WIDTHS),
        "output_tokens": OUTPUT_TOKENS,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "runtime": {
            "gpu_count": 1, "tensor_parallel_size": 1,
            "dtype": "bfloat16", "kv_cache_dtype": "auto",
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "gpu_memory_utilization": .9,
            "chunked_prefill": True, "prefix_caching": True,
            "enforce_eager": True, "lmcache_mode": "mp",
        },
        "semantics": {
            "fresh_engine_per_model_context": True,
            "unique_private_token_ids": True,
            "synchronized_bursts": True,
            "adaptive_stop": "two consecutive saturated bursts with no peak-running growth",
            "launch_oom_is_outcome": True,
            "campaign_gate": False,
        },
        "cells": cells, "run_order": order, "seed": seed,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    expected = {
        (model, testbed.model_spec(model).revision, context)
        for model in MODELS for context in CONTEXTS[model]
    }
    cells = plan.get("cells", [])
    actual = {(row.get("model"), row.get("revision"),
               row.get("context_tokens")) for row in cells}
    ids = [row.get("cell_id") for row in cells]
    qwen_contexts = plan.get("contexts", {}).get("Qwen/Qwen3.8-27B", [])
    if plan.get("schema") != SCHEMA \
            or plan.get("campaign") != "single_gpu_capacity_discovery" \
            or plan.get("hardware") != "a100" or actual != expected \
            or len(cells) != len(expected) or len(ids) != len(set(ids)) \
            or set(plan.get("run_order", [])) != set(ids) \
            or len(plan.get("run_order", [])) != len(ids) \
            or tuple(plan.get("widths", ())) != WIDTHS \
            or plan.get("output_tokens") != OUTPUT_TOKENS \
            or plan.get("runtime") != make_runtime_contract() \
            or any(context % 784 for context in qwen_contexts) \
            or plan.get("semantics", {}).get("campaign_gate") is not False:
        raise ValueError("invalid single-GPU capacity plan")


def make_runtime_contract() -> dict:
    return {
        "gpu_count": 1, "tensor_parallel_size": 1,
        "dtype": "bfloat16", "kv_cache_dtype": "auto",
        "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": .9, "chunked_prefill": True,
        "prefix_caching": True, "enforce_eager": True,
        "lmcache_mode": "mp",
    }


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def model_config(model: str) -> testbed.Config:
    spec = testbed.model_spec(model)
    base = testbed.Config(model=model)
    return replace(
        base, max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=spec.batched_tokens,
        capacity_discovery=True,
    )


def gpu_snapshot(expected: str = "A100") -> dict:
    devices = testbed.allocated_gpu_ids()
    if len(devices) != 1:
        raise RuntimeError("capacity discovery requires exactly one visible GPU")
    fields = ("name", "uuid", "driver_version", "memory.total", "power.limit",
              "clocks.applications.graphics", "clocks.applications.memory")
    text = subprocess.check_output([
        "nvidia-smi", "-i", devices[0],
        f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
    ], text=True).strip()
    values = [value.strip() for value in text.split(",")]
    if len(values) != len(fields) or expected not in values[0]:
        raise RuntimeError(f"expected one {expected}, saw {text}")
    return {"device": devices[0], **dict(zip(fields, values))}


def stack_commands(cfg: testbed.Config) -> dict[str, list[str]]:
    return {
        "redis": list(map(str, testbed.redis_cmd(cfg))),
        "cache": list(map(str, testbed.mp_server_cmd(
            cfg, "sink", l2_port=cfg.lmc_port,
        ))),
        "vllm": list(map(str, testbed.vllm_cmd(
            cfg, "sink", [], gpu_index=0,
        ))),
    }


def runtime_identity(plan: dict, cfg: testbed.Config, commands: dict) -> dict:
    sha, dirty = profiler.git_state(False)
    if dirty:
        raise RuntimeError("capacity discovery requires a clean launch commit")
    identity = {
        "plan_sha256": digest(plan), "git_sha": sha,
        "model": cfg.model, "revision": testbed.model_spec(cfg.model).revision,
        "hardware": gpu_snapshot(), "runtime_mode": testbed.runtime_mode(),
        "runtime_versions": testbed.runtime_versions(cfg),
        "scheduler": {
            "tensor_parallel_size": 1, "max_model_len": cfg.max_model_len,
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "dtype": "bfloat16", "kv_cache_dtype": "auto",
            "gpu_memory_utilization": .9, "block_size": 16,
            "chunked_prefill": True, "prefix_caching": True,
            "enforce_eager": True,
        },
        "commands": commands,
    }
    identity["sha256"] = digest(identity)
    return identity


def http_json(host: str, port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=30) as response:
        return json.loads(response.read().decode())


def available_kv_gib(log: Path) -> float | None:
    match = re.search(r"Available KV cache memory:\s+([\d.]+) GiB",
                      testbed.read_text(log))
    return float(match.group(1)) if match else None


def failure_kind(text: str) -> str:
    lower = text.lower()
    if any(marker in lower for marker in (
            "expected a (num_blocks, 2, block_size",
            "kv cache group", "did not prove", "runtime geometry changed",
            "vllm serve: error: argument")):
        return "runtime_contract"
    if any(marker in lower for marker in ("nvrm: xid", "gpu has fallen off",
                                          "preempted", "telemetry sampler",
                                          "metrics sampler", "power sampler",
                                          "can't start new thread",
                                          "cannot schedule new futures")):
        return "infrastructure"
    if re.search(r"out of memory|\boom\b|not enough memory|free memory on device",
                 text, re.IGNORECASE):
        return "oom"
    if "maximum model length" in lower:
        return "context_rejected"
    return "service_error"


def recordable_outcome(exc: Exception, engine_ready: bool, kind: str) -> bool:
    if isinstance(exc, RuntimeContractError) \
            or kind in {"infrastructure", "runtime_contract"}:
        return False
    return engine_ready or kind in {"oom", "context_rejected"}


@contextmanager
def engine_stack(cfg: testbed.Config, root: Path, identity: dict,
                 commands: dict) -> Iterator[SimpleNamespace]:
    if testbed.lmcache_mode() != "mp" or not testbed.prefix_caching():
        raise RuntimeError("capacity discovery requires the pinned MP/APC stack")
    testbed.preflight(cfg, required_gpus=1)
    for port in (cfg.sink_port, cfg.lmc_port, cfg.sink_lmc_port,
                 cfg.sink_lmc_http_port):
        if not testbed.port_free(cfg.host, port):
            raise RuntimeError(f"capacity-discovery port is busy: {port}")
    root.mkdir(parents=True, exist_ok=True)
    redis = testbed.start_logged(commands["redis"], root / "redis.log")
    cache = engine = None
    try:
        testbed.wait_tcp_process(cfg.host, cfg.lmc_port, 60, redis,
                                 root / "redis.log")
        cache = testbed.start_logged(commands["cache"], root / "lmcache-sink.log")
        testbed.wait_tcp_process(cfg.host, cfg.sink_lmc_port, 300, cache,
                                 root / "lmcache-sink.log")
        engine = testbed.start_logged(commands["vllm"], root / "sink.log")
        testbed.wait_health_process(cfg.host, cfg.sink_port,
                                    testbed.health_timeout(), engine,
                                    root / "sink.log")
        server_info = http_json(cfg.host, cfg.sink_port,
                                "/server_info?config_format=json")
        write_json(root / "server-info.json", server_info)
        log_text = testbed.read_text(root / "sink.log")
        try:
            testbed.validate_model_runtime_log(cfg, log_text, server_info)
        except RuntimeError as exc:
            raise RuntimeContractError(str(exc)) from exc
        yield SimpleNamespace(
            port=cfg.sink_port, engine=engine, log=root / "sink.log",
            kv_capacity_tokens=headroom.vllm_kv_capacity(root / "sink.log"),
            available_kv_gib=available_kv_gib(root / "sink.log"),
            server_info=server_info, identity=identity,
        )
    finally:
        for process in (engine, cache, redis):
            if process:
                testbed.stop_proc(process)


def request_body(model: str, context: int, width: int, index: int,
                 seed: int) -> tuple[str, int]:
    prompt = serving.deterministic_tokens(
        f"capacity:{model}:{context}:{width}:{index}", context, 1024, seed,
    )
    forced = 16 + (seed + width + index) % 1000
    return json.dumps(serving.completion_payload(
        model, prompt, OUTPUT_TOKENS, forced, bypass_lmcache=True,
    )), forced


def wait_drain(sampler: serving.MetricsSampler, engine, timeout_s: float = 60) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if engine.poll() is not None or sampler.error:
            return False
        if sampler.rows and not any(sampler.rows[-1].get(key, 0) for key in
                                    ("vllm:num_requests_running",
                                     "vllm:num_requests_waiting")):
            return True
        time.sleep(.05)
    return False


def run_burst(plan: dict, cell: dict, cfg: testbed.Config, stack,
              sampler: serving.MetricsSampler, width: int, root: Path) -> dict:
    prepared = [request_body(cfg.model, cell["context_tokens"], width, index,
                             plan["seed"]) for index in range(width)]
    start = time.monotonic_ns()

    def issue(item, epoch):
        index, (body, _forced) = item
        row = serving._completion(
            cfg.host, stack.port, cfg.model, [], OUTPUT_TOKENS, 0,
            plan["request_timeout_s"], prepared_body=body,
        )
        return {**row, "request_index": index, "burst_width": width,
                "planned_prompt_tokens": cell["context_tokens"],
                "planned_output_tokens": OUTPUT_TOKENS,
                "scheduled_ns": epoch,
                "send_lateness_s": (row["start_ns"] - epoch) / 1e9}

    items = list(enumerate(prepared))
    with ThreadPoolExecutor(max_workers=width) as executor:
        futures, epoch = headroom.submit_synchronized(executor, items, issue,
                                                       lead_s=.1)
        rows, error = headroom.settle_futures(futures)
    drained = wait_drain(sampler, stack.engine)
    if sampler.error:
        raise RuntimeError("engine telemetry sampler failed") from sampler.error
    end = time.monotonic_ns()
    metrics = [row for row in sampler.rows if start <= row["monotonic_ns"] <= end]
    complete = [row for row in rows if serving.service_completion(row)]
    exact = [row for row in complete if serving.exact_token_timing(row)]
    peak_running = int(max((row["vllm:num_requests_running"]
                            for row in metrics), default=0))
    peak_waiting = int(max((row["vllm:num_requests_waiting"]
                            for row in metrics), default=0))
    result = {
        "width": width, "offered": width, "completed": len(complete),
        "exact_timing": len(exact), "all_completed": len(complete) == width,
        "all_exact": len(exact) == width, "drained": drained,
        "engine_exited": stack.engine.poll() is not None,
        "peak_running_requests": peak_running,
        "peak_waiting_requests": peak_waiting,
        "peak_in_system_requests": peak_running + peak_waiting,
        "peak_kv_usage": max((row["vllm:gpu_cache_usage_perc"]
                              for row in metrics), default=None),
        "p90_ttft_s": float(np.quantile(
            [row["ttft_s"] for row in complete], .9)) if complete else None,
        "p90_mean_tpot_s": float(np.quantile(
            [row["mean_tpot_s"] for row in exact], .9)) if exact else None,
        "max_send_lateness_s": max((row["send_lateness_s"] for row in rows),
                                   default=None),
        "saturated": bool(peak_waiting > 0 and peak_running < width),
        "request_error": f"{type(error).__name__}: {error}" if error else None,
        "duration_s": (end - start) / 1e9,
    }
    write_json(root / f"width-{width:03d}-requests.json", rows)
    write_json(root / f"width-{width:03d}.json", result)
    return result


def runtime_geometry(cfg: testbed.Config, stack, root: Path) -> dict:
    log = testbed.read_text(stack.log)
    cache_log = testbed.read_text(root / "stack" / "lmcache-sink.log")
    cache_config = stack.server_info.get("vllm_config", {}).get(
        "cache_config", {})
    model_config = stack.server_info.get("vllm_config", {}).get(
        "model_config", {})
    requested_dtype = cache_config.get("cache_dtype")
    model_dtype = model_config.get("dtype")
    effective_dtype = testbed.effective_kv_cache_dtype(stack.server_info)
    return {
        "kv_capacity_tokens": stack.kv_capacity_tokens,
        "available_kv_cache_gib": stack.available_kv_gib,
        "requested_kv_cache_dtype": requested_dtype,
        "model_dtype": model_dtype,
        "effective_kv_cache_dtype": effective_dtype,
        "kv_cache_dtype_proof": effective_dtype.lower() in {
            "bfloat16", "torch.bfloat16"},
        "qwen_unified_block_tokens": (
            sorted(set(map(int, re.findall(
                r"attention block size to (\d+) tokens", log, re.IGNORECASE))))
            if cfg.model == "Qwen/Qwen3.8-27B" else []),
        "lmcache_chunk_tokens": testbed.model_chunk_tokens(cfg),
        "separate_object_groups": testbed.model_spec(
            cfg.model).separate_object_groups,
        "raw_kv_group_log_lines": [line for line in
                                   f"{log}\n{cache_log}".splitlines()
                                   if any(marker in line for marker in
                                          ("KV layer groups", "KernelGroupInfo",
                                           "ObjectGroupInfo", "Engine KV Format"))],
        "server_cache_config": stack.server_info.get(
            "vllm_config", {}).get("cache_config"),
    }


def run_cell(plan: dict, cell: dict, root: Path) -> dict:
    cfg = model_config(cell["model"])
    if not testbed.model_path(cfg).is_dir():
        raise RuntimeError(f"pinned model snapshot is missing: {cell['model']}")
    commands = stack_commands(cfg)
    identity = runtime_identity(plan, cfg, commands)
    started = time.time_ns()
    probes, geometry, outcome_error = [], None, None
    launchable, engine_ready = False, False
    sampler = power = None
    try:
        with engine_stack(cfg, root / "stack", identity, commands) as stack:
            launchable = engine_ready = True
            geometry = runtime_geometry(cfg, stack, root)
            sampler = serving.MetricsSampler(cfg.host, stack.port,
                                             root / "engine.csv", period_s=.05)
            power = profiler.PowerSampler(root / "power.csv")
            sampler.start()
            power.start()
            try:
                headroom.wait_sampler(sampler)
                prior_saturated_peak = None
                for width in plan["widths"]:
                    testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
                    probe = run_burst(plan, cell, cfg, stack, sampler, width, root)
                    probes.append(probe)
                    if probe["engine_exited"] or not probe["all_completed"] \
                            or probe["request_error"]:
                        detail = "\n".join(filter(None, (
                            probe["request_error"], testbed.read_text(stack.log))))
                        outcome_error = {
                            "phase": "service", "kind": failure_kind(detail),
                            "type": "probe_failure", "message":
                            probe["request_error"] or "engine exited or burst incomplete",
                        }
                        break
                    if probe["saturated"]:
                        if prior_saturated_peak is not None \
                                and probe["peak_running_requests"] \
                                <= prior_saturated_peak + 1:
                            break
                        prior_saturated_peak = probe["peak_running_requests"]
            finally:
                closing_sampler, sampler = sampler, None
                try:
                    closing_sampler.close()
                except Exception as exc:
                    raise RuntimeError("engine telemetry sampler failed") from exc
                closing_power, power = power, None
                try:
                    closing_power.close()
                except Exception as exc:
                    raise RuntimeError("power sampler failed") from exc
    except Exception as exc:
        text = testbed.read_text(root / "stack" / "sink.log")
        kind = failure_kind(f"{type(exc).__name__}: {exc}\n{text}")
        # Launch OOM/context rejection and an in-service failure are capacity
        # outcomes.  A semantic/runtime mismatch before readiness is not.
        if not recordable_outcome(exc, engine_ready, kind):
            raise
        outcome_error = {
            "phase": "service" if engine_ready else "launch",
            "kind": kind,
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if sampler:
            sampler.close()
        if power:
            power.close()
    attempted = {row["width"] for row in probes}
    result = {
        "schema": SCHEMA, "plan_sha256": digest(plan), **cell,
        "status": "complete", "discovery_only": True,
        "started_wall_ns": started, "runtime_identity": identity,
        "runtime_identity_sha256": identity["sha256"],
        "launchable": launchable, "outcome_error": outcome_error,
        "runtime_geometry": geometry, "probes": probes,
        "not_run_widths": [width for width in plan["widths"]
                           if width not in attempted],
    }
    write_json(root / "result.json", result)
    return result


def validate_result(plan: dict, cell: dict, result: dict) -> None:
    widths = [row.get("width") for row in result.get("probes", [])]
    identity = result.get("runtime_identity", {})
    if result.get("schema") != SCHEMA or result.get("status") != "complete" \
            or result.get("plan_sha256") != digest(plan) \
            or any(result.get(key) != value for key, value in cell.items()) \
            or widths != list(plan["widths"][:len(widths)]) \
            or set(widths) | set(result.get("not_run_widths", [])) \
            != set(plan["widths"]) \
            or identity.get("sha256") != digest({
                key: value for key, value in identity.items() if key != "sha256"
            }) or result.get("runtime_identity_sha256") != identity.get("sha256"):
        raise RuntimeError(f"invalid capacity result: {cell['cell_id']}")


def quarantine(path: Path) -> None:
    if path.exists():
        path.replace(path.with_name(f"{path.name}.invalid-{time.time_ns()}"))


def run_campaign(plan: dict, root: Path, max_attempts: int = 3) -> None:
    cells = {row["cell_id"]: row for row in plan["cells"]}
    for index, cell_id in enumerate(plan["run_order"], 1):
        cell, final = cells[cell_id], root / "cells" / cell_id
        result_path = final / "result.json"
        if result_path.exists():
            validate_result(plan, cell, json.loads(result_path.read_text()))
            continue
        if final.exists():
            quarantine(final)
        for attempt in range(1, max_attempts + 1):
            write_json(root / "status.json", {
                "state": "running", "cell_id": cell_id, "cell_index": index,
                "cell_count": len(plan["run_order"]), "attempt": attempt,
                "plan_sha256": digest(plan),
            })
            temporary = final.with_name(f".{cell_id}.attempt-{attempt}-{time.time_ns()}")
            try:
                result = run_cell(plan, cell, temporary)
                validate_result(plan, cell, result)
                temporary.replace(final)
                break
            except Exception:
                quarantine(temporary)
                if attempt == max_attempts:
                    raise
                time.sleep(5)
    summary = reduce(plan, root)
    write_json(root / "summary.json", summary)
    write_json(root / "status.json", {
        "state": "complete", "completed_cells": len(plan["run_order"]),
        "plan_sha256": digest(plan), "discovery_only": True,
    })


def cell_summary(result: dict) -> dict:
    probes = result["probes"]
    completed = [row for row in probes if row["all_completed"]]
    saturated = [row for row in probes if row["saturated"]]
    first_saturated = min(saturated, key=lambda row: row["width"],
                          default=None)
    failed = [row for row in probes if not row["all_completed"]
              or row["engine_exited"] or row["request_error"]]
    kv_usage = [row["peak_kv_usage"] for row in probes
                if row.get("peak_kv_usage") is not None]
    geometry = result.get("runtime_geometry") or {}
    return {
        "model": result["model"], "revision": result["revision"],
        "context_tokens": result["context_tokens"],
        "launchable": result["launchable"],
        "outcome_error_phase": (result.get("outcome_error") or {}).get("phase"),
        "outcome_error_kind": (result.get("outcome_error") or {}).get("kind"),
        "kv_capacity_tokens": geometry.get("kv_capacity_tokens"),
        "available_kv_cache_gib": geometry.get("available_kv_cache_gib"),
        "kv_cache_dtype_proof": geometry.get("kv_cache_dtype_proof", False),
        "qwen_unified_block_tokens": geometry.get("qwen_unified_block_tokens", []),
        "max_completed_burst_width": max((row["width"] for row in completed),
                                         default=0),
        "max_peak_running_requests": max((row["peak_running_requests"]
                                          for row in probes), default=0),
        "max_peak_waiting_requests": max((row["peak_waiting_requests"]
                                           for row in probes), default=0),
        "max_peak_kv_usage": max(kv_usage, default=None),
        "first_saturated_width": (first_saturated["width"]
                                  if first_saturated else None),
        "first_saturated_peak_kv_usage": (
            first_saturated.get("peak_kv_usage") if first_saturated else None),
        "first_saturated_p90_ttft_s": (
            first_saturated.get("p90_ttft_s") if first_saturated else None),
        "first_saturated_p90_mean_tpot_s": (
            first_saturated.get("p90_mean_tpot_s")
            if first_saturated else None),
        "first_saturated_exact_timing": (
            first_saturated.get("exact_timing") if first_saturated else None),
        "first_saturated_completed": (
            first_saturated.get("completed") if first_saturated else None),
        "first_failed_width": min((row["width"] for row in failed), default=None),
        "largest_width_attempted": max((row["width"] for row in probes), default=0),
        # A successful last probe is a lower bound, whether the geometric
        # sweep ended at 256 or stopped after repeating a running-capacity
        # plateau.  No larger completion boundary was observed.
        "right_censored": bool(probes and probes[-1]["all_completed"]),
        "probes": len(probes),
    }


def reduce(plan: dict, root: Path) -> dict:
    rows = []
    for cell in plan["cells"]:
        path = root / "cells" / cell["cell_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing capacity result: {cell['cell_id']}")
        result = json.loads(path.read_text())
        validate_result(plan, cell, result)
        rows.append(cell_summary(result))
    return {
        "schema": SCHEMA, "stage": "reduced", "plan_sha256": digest(plan),
        "hardware": "a100", "discovery_only": True, "campaign_gate": False,
        "rows": rows,
        "model_summary": {model: {
            "launchable_contexts": sum(row["launchable"] for row in rows
                                       if row["model"] == model),
            "maximum_context_with_width1_completion": max((
                row["context_tokens"] for row in rows if row["model"] == model
                and row["max_completed_burst_width"] >= 1), default=None),
        } for model in MODELS},
    }


def write_summary_csv(summary: dict, path: Path) -> None:
    rows = [{key: (json.dumps(value, separators=(",", ":"))
                   if isinstance(value, (list, dict)) else value)
             for key, value in row.items()} for row in summary["rows"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=1)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--max-attempts", type=int, default=3)
    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("--plan", type=Path, required=True)
    reduce_parser.add_argument("--run-root", type=Path, required=True)
    reduce_parser.add_argument("--out", type=Path, required=True)
    reduce_parser.add_argument("--csv", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        write_json(args.out, make_plan(args.seed))
    elif args.command == "run":
        run_campaign(read_plan(args.plan), args.run_root, args.max_attempts)
    else:
        summary = reduce(read_plan(args.plan), args.run_root)
        write_json(args.out, summary)
        write_summary_csv(summary, args.csv or args.out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
