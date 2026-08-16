"""Measure incumbent serving health as Queue-Haul consumes service headroom."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
from destination_campaign import IMAGE_SHA256


SCHEMA = "queue-haul-service-headroom-v1"
CONFIRM_SCHEMA = "queue-haul-service-headroom-confirmation-v1"
HARDWARE = {"a100": "NVIDIA A100", "h100": "NVIDIA H100"}
LOADS = (.25, .50, .70, .85, .95, 1.10)
PREFILL_CONCURRENCY = (1, 4, 16)
DECODE_CONCURRENCY = (16, 64, 128)
BLOCKS = 3
BASE_RHO = .25
CONTEXT = 4096
SESSIONS_PER_SHAPE = 8
BALANCED_OUTPUT_TOKENS = 128
BALANCED_SHARE_RANGE = (.4, .6)


@dataclass(frozen=True)
class Shape:
    prefix_tokens: int
    append_tokens: int
    output_tokens: int


SHAPES = {
    "incumbent": Shape(3840, 256, 128),
    "prefill_heavy": Shape(2048, 2048, 32),
    "decode_heavy": Shape(4032, 64, 512),
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def make_plan(model: str = testbed.MODEL) -> dict:
    spec = testbed.model_spec(model)
    specialized = model != testbed.MODEL
    calibration = [
        {"cell_id": f"{hardware}-cal-{phase}-n{concurrency}-b{block}",
         "hardware": hardware, "kind": "calibration", "phase": phase,
         "concurrency": concurrency, "block": block}
        for hardware in HARDWARE for block in range(BLOCKS)
        for phase, levels in (("prefill", PREFILL_CONCURRENCY),
                              ("decode", DECODE_CONCURRENCY))
        for concurrency in levels
    ]
    headroom = [
        {"cell_id": f"{hardware}-baseline-rho{BASE_RHO:.2f}-b{block}",
         "hardware": hardware, "kind": "headroom", "direction": "baseline",
         "target_rho": BASE_RHO, "block": block}
        for hardware in HARDWARE for block in range(BLOCKS)
    ] + [
        {"cell_id": f"{hardware}-{direction}-rho{rho:.2f}-b{block}",
         "hardware": hardware, "kind": "headroom", "direction": direction,
         "target_rho": rho, "block": block}
        for hardware in HARDWARE for block in range(BLOCKS)
        for direction in ("prefill_heavy", "decode_heavy") for rho in LOADS[1:]
    ]
    controls = [
        {"cell_id": f"{hardware}-no-resident-control-b{block}",
         "hardware": hardware, "kind": "residency_control",
         "direction": "prefill_heavy", "target_rho": BASE_RHO,
         "block": block, "resident_state": False}
        for hardware in HARDWARE for block in range(BLOCKS)
    ]
    for cell in headroom:
        cell["resident_state"] = True
    cells = calibration + headroom + controls
    return {
        "schema": SCHEMA, "image_sha256": IMAGE_SHA256,
        "model": model, "context_tokens": CONTEXT,
        "base_rho": BASE_RHO, "loads": list(LOADS), "blocks": BLOCKS,
        "warmup_s": 60, "measurement_s": 240, "drain_s": 180,
        "request_timeout_s": 180, "max_send_lateness_s": .05,
        "max_metric_gap_s": 1,
        "kv_match_tolerance": .02,
        "residency_control_max_relative_degradation": .15,
        "p99_min_incumbent_requests": 1000,
        "stack": {
            "tensor_parallel_size": 1, "max_model_len": 32768,
            "max_num_seqs": 256, "max_num_batched_tokens": spec.batched_tokens,
            "kv_cache_dtype": "auto", "block_size": 16,
            "gpu_memory_utilization": .9 if specialized else .75,
            "chunked_prefill": True,
            "prefix_caching": True, "enforce_eager": True,
            "disable_hybrid_kv_cache_manager": not specialized,
            "async_scheduling": False, "stream_interval": 1,
            "runtime_versions": {
                "apptainer": list(testbed.MP_RUNTIME_VERSIONS),
                "native": list(testbed.NATIVE_RUNTIME_VERSIONS),
            },
        },
        "shapes": {**{name: asdict(shape) for name, shape in SHAPES.items()},
                   "balanced": {"context_tokens": CONTEXT,
                                "output_tokens": BALANCED_OUTPUT_TOKENS,
                                "prefill_share_range": list(BALANCED_SHARE_RANGE)}},
        "cells": cells,
        "run_order": {hardware: {
            kind: sorted((cell["cell_id"] for cell in cells
                          if cell["hardware"] == hardware and cell["kind"] in kinds),
                         key=lambda value: digest(["run-order", value]))
            for kind, kinds in (("calibration", {"calibration"}),
                                ("measurement", {"headroom", "residency_control"}))}
            for hardware in HARDWARE},
        "confirmation": {
            "heldout": "repeat one shared baseline and each direction's last pass and first fail in three new restart blocks",
            "balanced": "three restart blocks at the selected boundary",
            "shape_checks": ["matched-rho long/few versus short/many decode",
                             "matched-rho smooth versus microburst prefill"],
        },
    }


def validate_plan(plan: dict) -> None:
    if plan.get("schema") == CONFIRM_SCHEMA:
        validate_confirmation_plan(plan)
    elif plan != make_plan(plan.get("model", testbed.MODEL)):
        raise ValueError("service-headroom plan changed from the frozen design")


def confirmation_cells(hardware: str, selection: dict) -> list[dict]:
    cells = []
    for block in range(BLOCKS, 2 * BLOCKS):
        cells.append({
            "cell_id": f"{hardware}-confirm-baseline-b{block}",
            "hardware": hardware, "kind": "confirmation",
            "direction": "baseline", "role": "baseline",
            "target_rho": BASE_RHO, "block": block, "resident_state": True,
        })
        for direction in ("prefill_heavy", "decode_heavy"):
            for role, rho in (("last_pass", selection[direction]["last_pass"]),
                              ("first_fail", selection[direction]["first_fail"])):
                cells.append({
                    "cell_id": f"{hardware}-confirm-{direction}-{role}-b{block}",
                    "hardware": hardware, "kind": "confirmation",
                    "direction": direction, "role": role, "target_rho": rho,
                    "block": block, "resident_state": True,
                })
        cells.append({
            "cell_id": f"{hardware}-confirm-balanced-b{block}",
            "hardware": hardware, "kind": "confirmation",
            "direction": "balanced", "role": "balanced",
            "target_rho": selection["supported_candidate"], "block": block,
            "resident_state": True,
        })
    return cells


def make_confirmation_plan(core: dict, scout: dict, hardware: str) -> dict:
    validate_plan(core)
    try:
        validate_scout_evidence(core, scout, hardware)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError("scout evidence is not eligible for confirmation") from exc
    if not scout["selection_ready"] or not scout["residency_control"]["pass"]:
        raise RuntimeError("scout is not eligible for confirmation")
    selection = {direction: {
        "last_pass": scout["direction_results"][direction]["slo_last_pass"],
        "first_fail": scout["direction_results"][direction]["slo_first_fail"],
    } for direction in ("prefill_heavy", "decode_heavy")}
    selection["supported_candidate"] = min(
        row["last_pass"] for row in selection.values())
    common = {key: core[key] for key in (
        "image_sha256", "model", "context_tokens", "base_rho", "loads", "blocks",
        "warmup_s", "measurement_s", "drain_s", "request_timeout_s",
        "max_send_lateness_s", "max_metric_gap_s", "kv_match_tolerance",
        "residency_control_max_relative_degradation",
        "p99_min_incumbent_requests", "stack", "shapes",
    )}
    plan = {"schema": CONFIRM_SCHEMA, **common, "hardware": hardware,
            "source_plan_sha256": digest(core),
            "source_scout_sha256": digest(scout), "targets": scout["targets"],
            "runtime_identity": scout["runtime_identity"],
            "runtime_identity_sha256": scout["runtime_identity_sha256"],
            "normalization_sha256": scout["normalization_sha256"],
            "source_residency_control_pass": scout["residency_control"]["pass"],
            "expected_resident_preloaded_kv_usage":
            scout["residency_control"]["resident_preloaded_kv_median"],
            "kv_capacity_tokens": scout["residency_control"]["kv_capacity_tokens"],
            "planned_parked_prefix_tokens":
            scout["residency_control"]["planned_parked_prefix_tokens"],
            "selection": selection,
            "cells": confirmation_cells(hardware, selection)}
    plan["run_order"] = sorted((cell["cell_id"] for cell in plan["cells"]),
                               key=lambda value: digest(["confirm-order", value]))
    validate_confirmation_plan(plan)
    return plan


def validate_confirmation_plan(plan: dict) -> None:
    core = make_plan(plan.get("model", testbed.MODEL))
    common = ("image_sha256", "model", "context_tokens", "base_rho", "loads",
              "blocks", "warmup_s", "measurement_s", "drain_s",
              "request_timeout_s", "max_send_lateness_s", "kv_match_tolerance",
              "max_metric_gap_s",
              "residency_control_max_relative_degradation",
              "p99_min_incumbent_requests", "stack", "shapes")
    selection = plan.get("selection", {})
    directions = [selection.get(name, {}) for name in
                  ("prefill_heavy", "decode_heavy")]
    valid_brackets = all(row.get("last_pass") in LOADS
                         and row.get("first_fail") in LOADS
                         and LOADS.index(row["first_fail"])
                         == LOADS.index(row["last_pass"]) + 1 for row in directions)
    candidate = min((row.get("last_pass") for row in directions), default=None)
    if plan.get("schema") != CONFIRM_SCHEMA or plan.get("hardware") not in HARDWARE \
            or any(plan.get(key) != core[key] for key in common) \
            or not valid_brackets or selection.get("supported_candidate") != candidate \
            or plan.get("cells") != confirmation_cells(plan["hardware"], selection) \
            or plan.get("run_order") != sorted(
                (cell["cell_id"] for cell in plan.get("cells", [])),
                key=lambda value: digest(["confirm-order", value])) \
            or plan.get("runtime_identity_sha256") \
            != identity_sha(plan.get("runtime_identity", {})) \
            or not 0 <= plan.get("expected_resident_preloaded_kv_usage", -1) <= 1 \
            or min(plan.get("kv_capacity_tokens", 0),
                   plan.get("planned_parked_prefix_tokens", 0)) <= 0:
        raise ValueError("service-headroom confirmation plan is invalid")


def service_work(shape: Shape, rates: dict) -> tuple[float, float]:
    return shape.append_tokens / rates["prefill_tps"], \
        shape.output_tokens / rates["decode_tps"]


def phase_share(shape: Shape, rates: dict) -> float:
    prefill, decode = service_work(shape, rates)
    return prefill / (prefill + decode)


def balanced_shape(rates: dict) -> Shape:
    append = 16 * round(BALANCED_OUTPUT_TOKENS * rates["prefill_tps"]
                        / rates["decode_tps"] / 16)
    if not 16 <= append <= CONTEXT - 16:
        raise ValueError("normalization cannot form a balanced 4K shape")
    return Shape(CONTEXT - append, append, BALANCED_OUTPUT_TOKENS)


def shape_for(name: str, rates: dict) -> Shape:
    return balanced_shape(rates) if name == "balanced" else SHAPES[name]


def parked_prefix_tokens(rates: dict) -> int:
    chunk = rates.get("cache_chunk_tokens", 16)
    return SESSIONS_PER_SHAPE * sum(
        (shape_for(name, rates).prefix_tokens - 1) // chunk * chunk
        for name in ("prefill_heavy", "decode_heavy", "balanced"))


def validate_rates(rates: dict) -> None:
    try:
        balanced = phase_share(balanced_shape(rates), rates)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("normalization does not separate the phase directions") from exc
    if min(rates["prefill_tps"], rates["decode_tps"]) <= 0 \
            or phase_share(SHAPES["prefill_heavy"], rates) < .7 \
            or phase_share(SHAPES["decode_heavy"], rates) > .2 \
            or not BALANCED_SHARE_RANGE[0] <= balanced <= BALANCED_SHARE_RANGE[1]:
        raise ValueError("normalization does not separate the phase directions")
    shape = asdict(balanced_shape(rates))
    if rates.get("balanced_shape", shape) != shape \
            or rates.get("planned_parked_prefix_tokens",
                         parked_prefix_tokens(rates)) != parked_prefix_tokens(rates):
        raise ValueError("normalization has a stale balanced shape or parked stock")


def runtime_contract(plan: dict, cfg: testbed.Config, extra: list[str], gpu: dict,
                     versions: tuple[str, str], runtime: dict,
                     git_sha: str, commands: dict[str, list[str]]) -> dict:
    if extra:
        raise RuntimeError("formal cells forbid extra vLLM arguments")
    commands = semantic_commands(commands)
    stack = plan["stack"]
    specialized = cfg.model != testbed.MODEL
    actual = {"tensor_parallel_size": 1,
              "max_model_len": cfg.max_model_len,
              "max_num_seqs": cfg.max_num_seqs,
              "max_num_batched_tokens": cfg.max_num_batched_tokens,
              "kv_cache_dtype": "auto", "block_size": 16,
              "gpu_memory_utilization": .9 if specialized else .75,
              "chunked_prefill": True,
              "prefix_caching": True, "enforce_eager": True,
              "disable_hybrid_kv_cache_manager": not specialized,
              "async_scheduling": False, "stream_interval": 1}
    expected = {key: value for key, value in stack.items()
                if key != "runtime_versions"}
    mode = runtime.get("mode")
    environment = runtime.get("environment")
    valid_runtime = mode == "apptainer" \
        and runtime.get("image_sha256") == plan["image_sha256"] \
        or mode == "native" and environment is not None \
        and runtime.get("environment_sha256") == digest(environment)
    if cfg.model != plan["model"] \
            or set(commands) != {"vllm", "cache", "redis"} \
            or actual != expected \
            or list(versions) != stack["runtime_versions"].get(mode) \
            or not valid_runtime:
        raise RuntimeError("serving stack differs from the frozen plan")
    identity = {"model": cfg.model,
                "model_revision": testbed.model_spec(cfg.model).revision,
                "runtime": runtime, "runtime_versions": list(versions),
                "git_sha": git_sha,
                "gpu": gpu, "scheduler": actual, "commands": commands}
    return {**identity, "sha256": digest(identity)}


def uniform_offsets(rate: float, horizon_s: float) -> tuple[float, ...]:
    if rate < 0 or horizon_s <= 0:
        raise ValueError("invalid offered trace")
    return tuple((index + .5) / rate for index in range(math.floor(rate * horizon_s))) \
        if rate else ()


def offered_trace(plan: dict, rates: dict, direction: str,
                  target_rho: float, block: int) -> list[dict]:
    if direction not in ("baseline", "prefill_heavy", "decode_heavy", "balanced") \
            or target_rho not in LOADS or block < 0 \
            or direction == "baseline" and target_rho != BASE_RHO:
        raise ValueError("unsupported headroom cell")
    horizon = plan["warmup_s"] + plan["measurement_s"]
    streams = [("incumbent", BASE_RHO)]
    if target_rho > BASE_RHO:
        streams.append((direction, target_rho - BASE_RHO))
    rows = []
    for population, rho in streams:
        shape = shape_for(population, rates)
        work = sum(service_work(shape, rates))
        for index, offset in enumerate(uniform_offsets(rho / work, horizon)):
            rows.append({"offset_s": offset, "population": population,
                         "session_id": f"{population}-{index % SESSIONS_PER_SHAPE}",
                         "request_index": index, "prefix_tokens": shape.prefix_tokens,
                         "append_tokens": shape.append_tokens,
                         "output_tokens": shape.output_tokens,
                         "prefill_work_s": shape.append_tokens / rates["prefill_tps"],
                         "decode_work_s": shape.output_tokens / rates["decode_tps"]})
    return sorted(rows, key=lambda row: (row["offset_s"], row["population"]))


def measurement_rows(plan: dict, rows: list[dict]) -> list[dict]:
    lo, hi = plan["warmup_s"], plan["warmup_s"] + plan["measurement_s"]
    return [row for row in rows if lo <= row["offset_s"] < hi]


def offered_phase_rho(plan: dict, rows: list[dict]) -> tuple[float, float]:
    measured = measurement_rows(plan, rows)
    return tuple(sum(row[f"{phase}_work_s"] for row in measured)
                 / plan["measurement_s"] for phase in ("prefill", "decode"))


def offered_rho(plan: dict, rows: list[dict]) -> float:
    return sum(offered_phase_rho(plan, rows))


def quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def cache_mismatches(rows: list[dict]) -> list[dict]:
    return [row for row in rows if serving.service_completion(row)
            and row.get("cached_tokens") != row.get("prefix_tokens", 0) // 16 * 16]


def validate_prewarm(rows: list[dict], sessions_: list[serving.Session]) -> None:
    if len(rows) != len(sessions_) or any(
            not serving.service_completion(row)
            or row.get("prompt_tokens") != session.prefix_tokens
            or row.get("cached_tokens") != 0
            for row, session in zip(rows, sessions_)):
        raise RuntimeError("private-prefix prewarm contract failed")


def resident_tokens(rows: list[dict], sessions_: list[serving.Session]) -> int:
    if len(rows) != len(sessions_) or any(
            not serving.service_completion(row)
            or row.get("prompt_tokens") != session.prefix_tokens
            or row.get("cached_tokens") != (session.prefix_tokens - 1) // 16 * 16
            for row, session in zip(rows, sessions_)):
        raise RuntimeError("private-prefix residency contract failed")
    return sum(row["cached_tokens"] for row in rows)


def engine_failure_kind(log: Path, exited: bool) -> str | None:
    if not exited:
        return None
    text = log.read_text(errors="ignore").lower() if log.exists() else ""
    return "infrastructure" if any(marker in text for marker in
                                   ("nvrm: xid", "gpu has fallen off", "preempted")) \
        else "service"


def vllm_kv_capacity(path: Path) -> int:
    match = re.search(r"GPU KV cache size:\s+([\d,]+) tokens",
                      path.read_text(errors="ignore"))
    if not match:
        raise RuntimeError("vLLM did not report its GPU KV capacity")
    return int(match.group(1).replace(",", ""))


def invalid_reason(plan: dict, trace: list[dict], requests: list[dict],
                   summary: dict, engine_failure: str | None) -> str | None:
    if engine_failure == "infrastructure":
        return "infrastructure engine failure"
    if len(requests) != len(trace) or not requests:
        return "incomplete offered trace"
    if max(row["send_lateness_s"] for row in requests) \
            > plan["max_send_lateness_s"]:
        return "offered trace schedule slip"
    if summary["cache_mismatch_count"]:
        return "cache contract mismatch"
    if summary.get("incumbent_exact", 1) and not summary["tpot_reportable"]:
        return "token timing is not observable"
    if not summary["telemetry_window_complete"] and engine_failure != "service":
        return "measurement telemetry is incomplete"
    return None


def in_system(row: dict, requests: list[dict]) -> float:
    now = int(row["monotonic_ns"])
    pending = sum(int(request.get("scheduled_ns", 0)) <= now
                  < int(request.get("start_ns", 0)) for request in requests)
    return float(row["vllm:num_requests_running"]
                 + row["vllm:num_requests_waiting"] + pending)


def summarize(plan: dict, offered: list[dict], requests: list[dict],
              metrics: list[dict], drained: bool) -> dict:
    planned = measurement_rows(plan, offered)
    if not requests:
        raise ValueError("headroom cell has no request records")
    epochs = [row["scheduled_ns"] - int(row["offset_s"] * 1e9) for row in requests]
    if max(epochs) - min(epochs) > 1:
        raise ValueError("request clocks do not share one offered-trace epoch")
    lo = round(statistics.median(epochs)) + int(plan["warmup_s"] * 1e9)
    hi = lo + int(plan["measurement_s"] * 1e9)
    observed = [row for row in requests if lo <= row["scheduled_ns"] < hi]
    window_metrics = [row for row in metrics if lo <= row["monotonic_ns"] < hi]
    incumbent = [row for row in observed if row["population"] == "incumbent"]
    if not incumbent:
        raise ValueError("headroom cell has no incumbent requests")
    good = [row for row in incumbent if serving.service_completion(row)]
    timed = [row for row in good if serving.exact_token_timing(row)]
    completed = [row for row in observed if serving.service_completion(row)]
    all_good = len(observed) == len(planned) and all(
        serving.service_completion(row) for row in observed)
    ttft = [row["ttft_s"] for row in good]
    tpot = [row["mean_tpot_s"] for row in timed]
    gaps = [gap for row in timed for gap in row["token_itls_s"]]
    try:
        drift = serving.queue_drift_upper(window_metrics, observed,
                                          include_running=True)
    except ValueError:
        drift = math.inf
    telemetry_complete = False
    if len(window_metrics) >= 2:
        gaps_ns = [right["monotonic_ns"] - left["monotonic_ns"]
                   for left, right in zip(window_metrics, window_metrics[1:])]
        tolerance = plan["max_metric_gap_s"] * 1e9
        telemetry_complete = window_metrics[0]["monotonic_ns"] - lo <= tolerance \
            and hi - window_metrics[-1]["monotonic_ns"] <= tolerance \
            and max(gaps_ns) <= tolerance
    loads = [in_system(row, observed) for row in window_metrics]
    p99 = len(good) >= plan["p99_min_incumbent_requests"]
    prefill_rho, decode_rho = offered_phase_rho(plan, offered)
    return {
        "offered_prefill_rho": prefill_rho,
        "offered_decode_rho": decode_rho,
        "offered_rho": prefill_rho + decode_rho,
        "offered_requests": len(observed),
        "incumbent_offered": len(incumbent),
        "incumbent_exact": len(good),
        "incumbent_exact_completion_rate": len(good) / len(incumbent),
        "incumbent_service_failure_rate": 1 - len(good) / len(incumbent),
        "all_offered_exact_completion_rate": len(completed) / len(observed),
        "all_offered_service_failure_rate": 1 - len(completed) / len(observed),
        "p50_ttft_s": quantile(ttft, .5), "p90_ttft_s": quantile(ttft, .9),
        "p95_ttft_s": quantile(ttft, .95),
        "p99_ttft_s": quantile(ttft, .99) if p99 else None,
        "p50_mean_tpot_s": quantile(tpot, .5),
        "p90_mean_tpot_s": quantile(tpot, .9),
        "p95_mean_tpot_s": quantile(tpot, .95),
        "p99_mean_tpot_s": quantile(tpot, .99) if p99 else None,
        "p90_token_itl_s": quantile(gaps, .9),
        "p95_token_itl_s": quantile(gaps, .95),
        "p99_token_itl_s": quantile(gaps, .99),
        "p99_reportable": p99,
        "tpot_reportable": bool(good) and len(timed) == len(good),
        "cache_mismatch_count": len(cache_mismatches(observed)),
        "queue_drift_upper_requests_per_s": drift,
        "initial_in_system_requests": loads[0] if loads else None,
        "late_window_p90_in_system_requests": quantile(loads[len(loads) * 2 // 3:], .9),
        "initial_kv_usage": window_metrics[0]["vllm:gpu_cache_usage_perc"]
        if window_metrics else None,
        "maximum_kv_usage": max((row["vllm:gpu_cache_usage_perc"]
                                  for row in window_metrics), default=None),
        "telemetry_window_complete": telemetry_complete,
        "drained": drained,
        "stable": bool(all_good and drained and telemetry_complete
                       and drift <= 1 / plan["measurement_s"]),
        "max_send_lateness_s": max(row["send_lateness_s"] for row in observed),
    }


def hardware_snapshot(hardware: str) -> dict:
    devices = testbed.allocated_gpu_ids()
    if len(devices) != 1:
        raise RuntimeError("formal service cells require exactly one allocated GPU")
    fields = ("name", "uuid", "driver_version", "memory.total", "power.limit",
              "clocks.applications.graphics", "clocks.applications.memory")
    text = subprocess.check_output([
        "nvidia-smi", "-i", devices[0],
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    values = [value.strip() for value in text.split(",")]
    if len(values) != len(fields) or HARDWARE[hardware] not in values[0]:
        raise RuntimeError(f"expected one {HARDWARE[hardware]}, saw {text}")
    return {"device": devices[0], **dict(zip(fields, values))}


def semantic_commands(commands: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: [re.sub(r"/tmp/qh-([A-Za-z0-9_-]+)-\d+",
                          r"/tmp/qh-\1-{pid}", token) for token in command]
            for name, command in commands.items()}


def stack_commands(cfg: testbed.Config, extra: list[str]) -> dict[str, list[str]]:
    return {"redis": list(map(str, testbed.redis_cmd(cfg))),
            "cache": list(map(str, testbed.mp_server_cmd(
                cfg, "sink", l2_port=cfg.lmc_port,
            ))),
            "vllm": list(map(str, testbed.vllm_cmd(
                cfg, "sink", ["--no-async-scheduling", "--stream-interval", "1",
                              *extra], gpu_index=0,
            )))}


def cached_image_hash(path: Path, cache: Path) -> str:
    stat = path.stat()
    signature = {"path": str(path.resolve()), "inode": stat.st_ino,
                 "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "ctime_ns": stat.st_ctime_ns}
    if cache.exists():
        record = json.loads(cache.read_text())
        if record.get("file") == signature and len(record.get("sha256", "")) == 64:
            return record["sha256"]
    sha256 = profiler.file_hash(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"file": signature, "sha256": sha256},
                                indent=2) + "\n")
    return sha256


def runtime_provenance(cfg: testbed.Config, cache: Path) -> dict:
    mode = testbed.runtime_mode()
    if mode == "apptainer":
        return {"mode": mode,
                "image_sha256": cached_image_hash(cfg.sandbox, cache)}
    packages = sorted(
        (distribution.metadata["Name"].lower(), distribution.version,
         hashlib.sha256((distribution.read_text("RECORD")
                         or distribution.read_text("METADATA") or "").encode()).hexdigest())
        for distribution in importlib.metadata.distributions()
    )
    environment = {"python": sys.version, "packages": packages}
    return {"mode": mode, "environment": environment,
            "environment_sha256": digest(environment)}


def collect_runtime_identity(plan: dict, cfg: testbed.Config, hardware: str,
                             extra: list[str], runtime: dict) -> dict:
    if testbed.lmcache_mode() != "mp" or not testbed.prefix_caching():
        raise RuntimeError("service-headroom requires the pinned MP stack")
    git_sha, _dirty = profiler.git_state(False)
    return runtime_contract(plan, cfg, extra, hardware_snapshot(hardware),
                            testbed.runtime_versions(cfg),
                            runtime, git_sha,
                            stack_commands(cfg, extra))


def identity_sha(identity: dict) -> str:
    checksum = identity.get("sha256")
    body = {key: value for key, value in identity.items() if key != "sha256"}
    if checksum is not None and checksum != digest(body):
        raise RuntimeError("runtime identity checksum changed")
    return checksum or digest(identity)


def validate_resume(result: dict, plan: dict, cell: dict, identity: dict,
                    normalization_sha256: str | None) -> None:
    if result.get("schema") != plan["schema"] \
            or result.get("plan_sha256") != digest(plan) \
            or any(result.get(key) != value for key, value in cell.items()) \
            or result.get("runtime_identity_sha256") != identity_sha(identity) \
            or result.get("normalization_sha256") != normalization_sha256:
        raise RuntimeError("resume evidence does not match this plan and runtime")


def validate_result_rows(plan: dict, hardware: str, kind: str,
                         rows: list[dict]) -> list[dict]:
    expected = [cell for cell in plan["cells"]
                if cell["hardware"] == hardware and cell["kind"] == kind]
    indexed = {row.get("cell_id"): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {
            cell["cell_id"] for cell in expected}:
        raise RuntimeError(f"{kind} result set does not match the plan")
    ordered = []
    for cell in expected:
        row = indexed[cell["cell_id"]]
        if row.get("schema") != plan["schema"] \
                or row.get("plan_sha256") != digest(plan) \
                or any(row.get(key) != value for key, value in cell.items()) \
                or row.get("runtime_identity_sha256") \
                != identity_sha(row.get("runtime_identity", {})):
            raise RuntimeError(f"result identity mismatch: {cell['cell_id']}")
        ordered.append(row)
    if len({row["runtime_identity_sha256"] for row in ordered}) != 1:
        raise RuntimeError("results mix runtime identities")
    return ordered


def load_results(plan: dict, hardware: str, root: Path, kind: str) -> list[dict]:
    expected = [cell for cell in plan["cells"]
                if cell["hardware"] == hardware and cell["kind"] == kind]
    rows = []
    for cell in expected:
        path = root / cell["cell_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing result: {cell['cell_id']}")
        rows.append(json.loads(path.read_text()))
    return validate_result_rows(plan, hardware, kind, rows)


def validate_run_order(plan: dict, hardware: str, stage: str,
                       rows: list[dict]) -> None:
    expected = plan["run_order"] if plan["schema"] == CONFIRM_SCHEMA \
        else plan["run_order"][hardware][stage]
    starts = [row.get("started_wall_ns", 0) for row in rows]
    observed = [row["cell_id"] for row in sorted(
        rows, key=lambda row: row.get("started_wall_ns", 0))]
    if len(set(starts)) != len(starts) or min(starts, default=0) <= 0 \
            or observed != expected:
        raise RuntimeError(f"{stage} run order differs from the frozen plan")


@contextmanager
def destination_stack(cfg: testbed.Config, root: Path, hardware: str,
                      extra: list[str], identity: dict):
    if testbed.lmcache_mode() != "mp":
        raise RuntimeError("service-headroom cells require QH_LMCACHE_MODE=mp")
    if not testbed.port_free(cfg.host, cfg.sink_port):
        raise RuntimeError(f"port busy: {cfg.host}:{cfg.sink_port}")
    preflight = testbed.preflight(cfg, required_gpus=1)
    commands = stack_commands(cfg, extra)
    if semantic_commands(commands) != identity["commands"]:
        raise RuntimeError("launch commands differ from runtime identity")
    root.mkdir(parents=True, exist_ok=True)
    redis = testbed.start_logged(commands["redis"], root / "redis.log")
    cache = engine = None
    try:
        testbed.wait_tcp_process(cfg.host, cfg.lmc_port, 60, redis, root / "redis.log")
        cache = testbed.start_logged(
            commands["cache"],
            root / "lmcache-sink.log",
        )
        testbed.wait_tcp_process(cfg.host, cfg.sink_lmc_port, 300, cache,
                                 root / "lmcache-sink.log")
        engine = testbed.start_logged(
            commands["vllm"], root / "sink.log",
        )
        testbed.wait_health_process(cfg.host, cfg.sink_port, testbed.health_timeout(),
                                    engine, root / "sink.log")
        testbed.validate_model_runtime_log(cfg, testbed.read_text(root / "sink.log"))
        capacity = vllm_kv_capacity(root / "sink.log")
        metadata = {"hardware": identity["gpu"], "preflight": preflight,
                    "runtime_identity": identity,
                    "runtime_versions": testbed.runtime_versions(cfg),
                    "commands": commands, "kv_capacity_tokens": capacity}
        (root / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
        yield SimpleNamespace(port=cfg.sink_port, log=root / "sink.log", engine=engine,
                              identity=identity, kv_capacity_tokens=capacity)
    finally:
        for process in (engine, cache, redis):
            if process:
                testbed.stop_proc(process)


def sessions(block: int, rates: dict) -> dict[str, serving.Session]:
    return {
        f"{name}-{index}": serving.Session(
            f"{name}-{index}", shape.prefix_tokens, shape.append_tokens,
            shape.output_tokens, 201088, block,
        )
        for name in (*SHAPES, "balanced") for shape in (shape_for(name, rates),)
        for index in range(SESSIONS_PER_SHAPE)
    }


def drain(sampler: serving.MetricsSampler, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if sampler.error:
            return False
        if sampler.rows and not any(sampler.rows[-1][key] for key in
                                    ("vllm:num_requests_running",
                                     "vllm:num_requests_waiting")):
            return True
        time.sleep(.1)
    return False


def wait_sampler(sampler: serving.MetricsSampler, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not sampler.rows and not sampler.error:
        time.sleep(.05)
    if not sampler.rows or sampler.error:
        raise RuntimeError("engine telemetry did not start") from sampler.error


def close_samplers(sampler: serving.MetricsSampler,
                   power: profiler.PowerSampler, engine) -> bool:
    try:
        try:
            sampler.close()
        except RuntimeError:
            if engine.poll() is None:
                raise
            return True
        return engine.poll() is not None
    finally:
        power.close()


def settle_futures(futures) -> tuple[list[dict], Exception | None]:
    rows, error = [], None
    for future in futures:
        try:
            rows.append(future.result())
        except Exception as exc:
            error = error or exc
    return rows, error


def synchronized_submit(executor, items, fn):
    items = list(items)
    ready, gate = threading.Barrier(len(items) + 1), threading.Event()
    def run(item):
        ready.wait()
        gate.wait()
        return fn(item)
    futures = [executor.submit(run, item) for item in items]
    ready.wait()
    return futures, gate


def validate_stage_inputs(plan: dict, rates: dict, identity: dict) -> None:
    if rates["runtime_identity_sha256"] != identity_sha(identity) \
            or rates.get("kv_capacity_tokens", 0) <= 0 \
            or rates.get("balanced_shape") != asdict(balanced_shape(rates)) \
            or rates.get("planned_parked_prefix_tokens") \
            != parked_prefix_tokens(rates):
        raise RuntimeError("normalization and headroom runtime identities differ")
    if plan["schema"] == CONFIRM_SCHEMA and (
            plan["runtime_identity_sha256"] != identity_sha(identity)
            or plan["normalization_sha256"] != rates["sha256"]):
        raise RuntimeError("confirmation differs from discovery runtime or normalization")


def run_headroom(plan: dict, cell: dict, rates: dict, cfg: testbed.Config,
                 root: Path, extra: list[str], identity: dict) -> dict:
    validate_rates(rates)
    validate_stage_inputs(plan, rates, identity)
    trace = offered_trace(plan, rates, cell["direction"], cell["target_rho"],
                          cell["block"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "offered.json").write_text(json.dumps(trace, indent=2) + "\n")
    pool = sessions(cell["block"], rates)
    requests, futures, error = [], [], None
    started_wall_ns = time.time_ns()
    epoch = wall = 0
    preloaded_kv_usage = None
    kv_capacity_tokens = None
    planned_parked_tokens = rates["planned_parked_prefix_tokens"]
    drained, engine_exited, failure_kind = False, False, None
    try:
        with destination_stack(cfg, root / "stack", cell["hardware"], extra,
                               identity) as stack:
            kv_capacity_tokens = stack.kv_capacity_tokens
            if kv_capacity_tokens != rates["kv_capacity_tokens"]:
                raise RuntimeError("live KV capacity differs from normalization")
            testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
            warm_sessions = list(pool.values()) if cell["resident_state"] else [
                session for name, session in pool.items() if name.startswith("incumbent-")]
            warm = serving.prewarm(cfg.host, stack.port, cfg.model, warm_sessions,
                                   plan["request_timeout_s"], True)
            (root / "prewarm.json").write_text(json.dumps(warm, indent=2) + "\n")
            validate_prewarm(warm, warm_sessions)
            resident = serving.prewarm(cfg.host, stack.port, cfg.model, warm_sessions,
                                       plan["request_timeout_s"], True)
            (root / "residency.json").write_text(
                json.dumps(resident, indent=2) + "\n")
            preloaded_kv_usage = resident_tokens(resident, warm_sessions) \
                / kv_capacity_tokens
            sampler = serving.MetricsSampler(cfg.host, stack.port, root / "engine.csv")
            power = profiler.PowerSampler(root / "power.csv")
            sampler.start()
            power.start()
            try:
                wait_sampler(sampler)
                epoch, wall = time.monotonic_ns(), time.time_ns()
                def issue(row):
                    result = serving.issue(
                        cfg.host, stack.port, cfg.model, pool[row["session_id"]],
                        row["request_index"], epoch + int(row["offset_s"] * 1e9),
                        plan["request_timeout_s"], True,
                    )
                    return {**result, "population": row["population"],
                            "offset_s": row["offset_s"]}
                with ThreadPoolExecutor(max_workers=512) as executor:
                    for row in trace:
                        scheduled = epoch / 1e9 + row["offset_s"]
                        time.sleep(max(0, scheduled - time.monotonic()))
                        futures.append(executor.submit(issue, row))
                requests, error = settle_futures(futures)
                if error is None:
                    drained = drain(sampler, plan["drain_s"])
            except Exception as exc:
                error = error or exc
            finally:
                if futures and not requests:
                    requests, future_error = settle_futures(futures)
                    error = error or future_error
                try:
                    engine_exited = close_samplers(sampler, power, stack.engine)
                except Exception as exc:
                    error = error or exc
                failure_kind = engine_failure_kind(stack.log, engine_exited)
    except Exception as exc:
        error = error or exc
    (root / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    base = {"schema": plan["schema"], **cell, "plan_sha256": digest(plan),
            "runtime_identity": identity,
            "runtime_identity_sha256": identity_sha(identity),
            "normalization_sha256": rates["sha256"],
            "started_wall_ns": started_wall_ns,
            "epoch_monotonic_ns": epoch, "epoch_wall_ns": wall,
            "preloaded_kv_usage": preloaded_kv_usage,
            "kv_capacity_tokens": kv_capacity_tokens,
            "planned_parked_prefix_tokens": planned_parked_tokens,
            "engine_exited": engine_exited, "engine_failure_kind": failure_kind,
            "added_prefill_share": (None if cell["direction"] == "baseline" else
                                     phase_share(shape_for(cell["direction"], rates),
                                                 rates))}
    if error:
        result = {**base, "status": "invalid",
                  "measurement_error": f"{type(error).__name__}: {error}"}
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("service-headroom measurement is invalid") from error
    summary = summarize(plan, trace, requests, sampler.rows, drained)
    reason = invalid_reason(plan, trace, requests, summary, failure_kind)
    summary.update({**base, "status": "invalid" if reason else "complete",
                    "measurement_error": reason})
    (root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    if reason:
        raise RuntimeError(f"service-headroom measurement is invalid: {reason}")
    return summary


def calibration_sessions(phase: str, concurrency: int, block: int) -> list[serving.Session]:
    shape = Shape(1, CONTEXT - 1, 1) if phase == "prefill" \
        else Shape(CONTEXT - 1, 1, 256)
    return [serving.Session(f"cal-{phase}-{index}", shape.prefix_tokens,
                            shape.append_tokens, shape.output_tokens, 201088, block)
            for index in range(concurrency)]


def run_calibration(plan: dict, cell: dict, cfg: testbed.Config,
                    root: Path, extra: list[str], identity: dict) -> dict:
    group = calibration_sessions(cell["phase"], cell["concurrency"], cell["block"])
    root.mkdir(parents=True, exist_ok=True)
    requests, futures, error = [], [], None
    started_wall_ns = time.time_ns()
    epoch, drained, engine_exited, failure_kind = 0, False, False, None
    kv_capacity_tokens = None
    try:
        with destination_stack(cfg, root / "stack", cell["hardware"], extra,
                               identity) as stack:
            kv_capacity_tokens = stack.kv_capacity_tokens
            warm = serving.prewarm(cfg.host, stack.port, cfg.model, group,
                                   plan["request_timeout_s"], True)
            (root / "prewarm.json").write_text(json.dumps(warm, indent=2) + "\n")
            validate_prewarm(warm, group)
            sampler = serving.MetricsSampler(cfg.host, stack.port, root / "engine.csv")
            power = profiler.PowerSampler(root / "power.csv")
            sampler.start()
            power.start()
            try:
                wait_sampler(sampler)
                prepared = [session.prompt(index)
                            for index, session in enumerate(group)]
                bodies = [serving.completion_body(
                    cfg.model, prompt, session.output_tokens, forced, True,
                ) for session, (prompt, forced) in zip(group, prepared)]
                def issue(item):
                    time.sleep(max(0, epoch / 1e9 - time.monotonic()))
                    return serving.issue(
                        cfg.host, stack.port, cfg.model, item[1], item[0], epoch,
                        plan["request_timeout_s"], True, prepared[item[0]],
                        bodies[item[0]],
                    )
                with ThreadPoolExecutor(max_workers=cell["concurrency"]) as executor:
                    futures, gate = synchronized_submit(
                        executor, enumerate(group), issue,
                    )
                    epoch = time.monotonic_ns() + 100_000_000
                    gate.set()
                requests, error = settle_futures(futures)
                if error is None:
                    drained = drain(sampler, plan["drain_s"])
            except Exception as exc:
                error = error or exc
            finally:
                if futures and not requests:
                    requests, future_error = settle_futures(futures)
                    error = error or future_error
                try:
                    engine_exited = close_samplers(sampler, power, stack.engine)
                except Exception as exc:
                    error = error or exc
                failure_kind = engine_failure_kind(stack.log, engine_exited)
    except Exception as exc:
        error = error or exc
    (root / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    complete = len(requests) == len(group) and all(
        serving.service_completion(row) for row in requests)
    schedule_valid = bool(requests) and max(row["send_lateness_s"] for row in requests) \
        <= plan["max_send_lateness_s"]
    cache_valid = not cache_mismatches(requests)
    duration = (max(row["last_token_ns"] for row in requests)
                - min(row["start_ns"] for row in requests)) / 1e9 if complete else None
    tokens = sum(row["planned_prompt_tokens"] - row["cached_tokens"]
                 if cell["phase"] == "prefill"
                 else row["planned_output_tokens"] for row in requests)
    invalid = error or failure_kind == "infrastructure" or not schedule_valid \
        or not cache_valid
    result = {"schema": plan["schema"], "plan_sha256": digest(plan),
              "runtime_identity": identity,
              "runtime_identity_sha256": identity_sha(identity),
              "normalization_sha256": None,
              "started_wall_ns": started_wall_ns,
              "kv_capacity_tokens": kv_capacity_tokens,
              "status": "invalid" if invalid else "complete", **cell,
              "service_completion": complete, "drained": drained,
              "engine_exited": engine_exited, "engine_failure_kind": failure_kind,
              "cache_mismatch_count": len(cache_mismatches(requests)),
              "measurement_error": f"{type(error).__name__}: {error}" if error else None,
              "max_send_lateness_s": max((row["send_lateness_s"]
                                           for row in requests), default=None),
              "tokens_per_s": tokens / duration
              if complete and drained and schedule_valid and cache_valid else None}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if invalid:
        raise RuntimeError("service calibration measurement is invalid") from error
    return result


def reduce_calibration(plan: dict, hardware: str, root: Path) -> dict:
    rows = load_results(plan, hardware, root, "calibration")
    validate_run_order(plan, hardware, "calibration", rows)
    if any(row["status"] != "complete" for row in rows):
        raise RuntimeError("calibration contains invalid measurements")
    capacities = {row["kv_capacity_tokens"] for row in rows}
    if len(capacities) != 1 or next(iter(capacities)) <= 0:
        raise RuntimeError("calibration mixes live KV capacities")
    rates = {"schema": SCHEMA, "hardware": hardware,
             "context_tokens": plan["context_tokens"],
             "plan_sha256": digest(plan),
             "runtime_identity": rows[0]["runtime_identity"],
             "runtime_identity_sha256": rows[0]["runtime_identity_sha256"],
             "normalizer_kind": "synchronized_burst_throughput",
             "kv_capacity_tokens": next(iter(capacities))}
    if plan["model"] != testbed.MODEL:
        rates["cache_chunk_tokens"] = testbed.model_spec(plan["model"]).chunk_tokens
    edge_censored = False
    for phase in ("prefill", "decode"):
        peaks, peak_concurrency = [], []
        for block in range(plan["blocks"]):
            candidates = [row for row in rows if row["phase"] == phase
                          and row["block"] == block and row["tokens_per_s"] is not None]
            if not candidates:
                raise RuntimeError(f"{phase} calibration block {block} has no stable level")
            selected = max(candidates, key=lambda row: row["tokens_per_s"])
            peaks.append(selected["tokens_per_s"])
            peak_concurrency.append(selected["concurrency"])
            edge_censored |= selected["concurrency"] == max(
                row["concurrency"] for row in rows if row["phase"] == phase)
        rates[f"{phase}_tps"] = statistics.median(peaks)
        rates[f"{phase}_block_peaks_tps"] = peaks
        rates[f"{phase}_block_peak_concurrency"] = peak_concurrency
    rates["edge_censored"] = edge_censored
    rates["balanced_shape"] = asdict(balanced_shape(rates))
    rates["planned_parked_prefix_tokens"] = parked_prefix_tokens(rates)
    rates["sha256"] = digest({key: value for key, value in rates.items()
                              if key != "sha256"})
    validate_rates(rates)
    return rates


def joint_attainment(plan: dict, requests: list[dict], ttft_target_s: float,
                     tpot_target_s: float) -> float:
    eligible = [row for row in measurement_rows(plan, requests)
                if row["population"] == "incumbent"]
    if not eligible:
        raise RuntimeError("joint attainment has no offered incumbents")
    if any(serving.service_completion(row) and not serving.exact_token_timing(row)
           for row in eligible):
        raise RuntimeError("joint attainment needs exact token timing")
    good = sum(serving.service_completion(row)
               and row["ttft_s"] <= ttft_target_s
               and row["mean_tpot_s"] <= tpot_target_s for row in eligible)
    return good / len(eligible)


def bracket(labels: list[bool], name: str) -> dict:
    first_fail = next((index for index, value in enumerate(labels) if not value),
                      len(labels))
    if any(labels[first_fail:]):
        raise RuntimeError(f"nonmonotone {name} boundary")
    return {"last_pass": LOADS[first_fail - 1] if first_fail else None,
            "first_fail": LOADS[first_fail] if first_fail < len(LOADS) else None,
            "left_censored": first_fail == 0,
            "right_censored": first_fail == len(LOADS)}


def build_scout(plan: dict, hardware: str, rows: list[dict], controls: list[dict],
                targets: dict) -> dict:
    if set(targets) != {"p90_ttft_s", "p90_mean_tpot_s"} \
            or min(targets.values()) <= 0:
        raise ValueError("latency targets must be positive")
    rows = validate_result_rows(plan, hardware, "headroom", rows)
    controls = validate_result_rows(plan, hardware, "residency_control", controls)
    validate_run_order(plan, hardware, "measurement", rows + controls)
    if len({row["runtime_identity_sha256"] for row in rows + controls}) != 1 \
            or len({row["normalization_sha256"] for row in rows + controls}) != 1:
        raise RuntimeError("headroom results mix runtime or normalization identities")
    if any(row["status"] != "complete" for row in rows):
        raise RuntimeError("headroom reduction contains invalid measurements")
    if len(controls) != plan["blocks"] \
            or any(row["status"] != "complete" for row in controls):
        raise RuntimeError("residency controls are incomplete")
    if any(row["incumbent_exact"] and not row["tpot_reportable"]
           for row in rows + controls):
        raise RuntimeError("headroom reduction lacks exact TPOT measurements")
    resident_kv = [row["preloaded_kv_usage"] for row in rows]
    control_kv = [row["preloaded_kv_usage"] for row in controls]
    measurement_kv = [row["initial_kv_usage"] for row in rows]
    if any(value is None for value in resident_kv + control_kv):
        raise RuntimeError("headroom reduction lacks preloaded KV telemetry")
    if max(resident_kv) - min(resident_kv) > plan["kv_match_tolerance"]:
        raise RuntimeError("resident KV stock changed across the headroom curve")
    capacities = {row["kv_capacity_tokens"] for row in rows + controls}
    parked = {row["planned_parked_prefix_tokens"] for row in rows + controls}
    if len(capacities) != 1 or len(parked) != 1:
        raise RuntimeError("headroom results mix KV capacity or parked stock")
    capacity, planned_parked = next(iter(capacities)), next(iter(parked))
    direction_results = {}
    for direction in ("prefill_heavy", "decode_heavy"):
        slo_labels = [all(
            row["stable"] and row["p90_ttft_s"] <= targets["p90_ttft_s"]
            and row["p90_mean_tpot_s"] <= targets["p90_mean_tpot_s"]
            for row in rows if row["target_rho"] == rho
            and (rho == BASE_RHO or row["direction"] == direction)) for rho in LOADS]
        physical_labels = [all(row["stable"] for row in rows
                               if row["target_rho"] == rho
                               and (rho == BASE_RHO or row["direction"] == direction))
                           for rho in LOADS]
        slo, physical = bracket(slo_labels, direction), bracket(
            physical_labels, f"{direction} physical",
        )
        direction_results[direction] = {
            **{f"slo_{key}": value for key, value in slo.items()},
            **{f"physical_{key}": value for key, value in physical.items()},
        }
    resident_base = [row for row in rows if row["target_rho"] == BASE_RHO]
    healthy_controls = all(
        row["stable"] and row["incumbent_exact_completion_rate"] == 1
        and row["all_offered_exact_completion_rate"] == 1 for row in controls)
    healthy_resident_base = all(
        row["stable"] and row["incumbent_exact_completion_rate"] == 1
        and row["all_offered_exact_completion_rate"] == 1 for row in resident_base)
    ratios = {
        metric: statistics.median(row[metric] for row in resident_base)
        / statistics.median(row[metric] for row in controls) - 1
        for metric in ("p90_ttft_s", "p90_mean_tpot_s")
    } if healthy_controls and healthy_resident_base else {
        "p90_ttft_s": None, "p90_mean_tpot_s": None,
    }
    stock_delta = statistics.median(resident_kv) - statistics.median(control_kv)
    expected_stock_delta = planned_parked / capacity
    stock_match = abs(stock_delta - expected_stock_delta) \
        <= plan["kv_match_tolerance"]
    residency_pass = healthy_controls and healthy_resident_base \
        and stock_match \
        and max(ratios.values()) \
        <= plan["residency_control_max_relative_degradation"]
    bounds = [row["slo_last_pass"] for row in direction_results.values()]
    conservative = min(bounds) if all(value is not None for value in bounds) else None
    selection_ready = residency_pass and conservative is not None and all(
        row["slo_first_fail"] is not None for row in direction_results.values())
    return {"schema": SCHEMA, "stage": "scout", "hardware": hardware,
            "plan_sha256": digest(plan), "planner_usable": False,
            "targets": targets,
            "runtime_identity": rows[0]["runtime_identity"],
            "runtime_identity_sha256": rows[0]["runtime_identity_sha256"],
            "normalization_sha256": rows[0]["normalization_sha256"],
            "direction_results": direction_results,
            "scout_conservative_bound": conservative,
            "selection_ready": selection_ready,
            "residency_control": {
                "pass": residency_pass,
                "healthy_controls": healthy_controls,
                "healthy_resident_baseline": healthy_resident_base,
                "stock_match": stock_match,
                "preloaded_kv_usage_delta": stock_delta,
                "expected_preloaded_kv_usage_delta": expected_stock_delta,
                "kv_capacity_tokens": capacity,
                "planned_parked_prefix_tokens": planned_parked,
                "relative_degradation": ratios,
                "resident_preloaded_kv_median": statistics.median(resident_kv),
                "control_preloaded_kv_median": statistics.median(control_kv),
                "measurement_start_kv_range": ([min(observed_kv), max(observed_kv)]
                                               if (observed_kv := [value for value
                                                   in measurement_kv if value is not None])
                                               else [None, None]),
                "resident_baseline_p90_ttft_median_s": quantile([
                    row["p90_ttft_s"] for row in rows
                    if row["target_rho"] == BASE_RHO and row["p90_ttft_s"] is not None
                ], .5),
                "control_p90_ttft_median_s": quantile([
                    row["p90_ttft_s"] for row in controls
                    if row["p90_ttft_s"] is not None
                ], .5),
            },
            "controls": controls,
            "rows": rows}


def validate_scout_evidence(plan: dict, scout: dict, hardware: str) -> None:
    try:
        expected = build_scout(plan, hardware, scout["rows"], scout["controls"],
                               scout["targets"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("scout evidence is invalid") from exc
    if scout != expected:
        raise RuntimeError("scout evidence summary does not match its rows")


def reduce_headroom(plan: dict, hardware: str, root: Path,
                    ttft_target_s: float, tpot_target_s: float) -> dict:
    rows = load_results(plan, hardware, root, "headroom")
    controls = load_results(plan, hardware, root, "residency_control")
    for row in rows:
        requests = json.loads((root / row["cell_id"] / "requests.json").read_text())
        row["joint_slo_attainment"] = joint_attainment(
            plan, requests, ttft_target_s, tpot_target_s,
        )
    return build_scout(plan, hardware, rows, controls, {
        "p90_ttft_s": ttft_target_s, "p90_mean_tpot_s": tpot_target_s,
    })


def row_feasible(row: dict, targets: dict) -> bool:
    return bool(row["stable"] and row["p90_ttft_s"] <= targets["p90_ttft_s"]
                and row["p90_mean_tpot_s"] <= targets["p90_mean_tpot_s"])


def validate_confirmation_source(plan: dict, core: dict, scout: dict) -> None:
    try:
        expected = make_confirmation_plan(core, scout, plan["hardware"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError("confirmation source scout is invalid") from exc
    if plan != expected:
        raise RuntimeError("confirmation plan does not match its source scout")


def reduce_confirmation(plan: dict, root: Path, core: dict, scout: dict) -> dict:
    validate_confirmation_source(plan, core, scout)
    rows = load_results(plan, plan["hardware"], root, "confirmation")
    if any(row["status"] != "complete" or row["incumbent_exact"]
           and not row["tpot_reportable"] for row in rows):
        raise RuntimeError("confirmation contains invalid or incomplete measurements")
    if {row["runtime_identity_sha256"] for row in rows} \
            != {plan["runtime_identity_sha256"]} \
            or {row["normalization_sha256"] for row in rows} \
            != {plan["normalization_sha256"]}:
        raise RuntimeError("confirmation runtime differs from discovery")
    for row in rows:
        requests = json.loads((root / row["cell_id"] / "requests.json").read_text())
        row["joint_slo_attainment"] = joint_attainment(
            plan, requests, plan["targets"]["p90_ttft_s"],
            plan["targets"]["p90_mean_tpot_s"],
        )
    return build_confirmation(plan, rows)


def build_confirmation(plan: dict, rows: list[dict]) -> dict:
    rows = validate_result_rows(plan, plan["hardware"], "confirmation", rows)
    validate_run_order(plan, plan["hardware"], "confirmation", rows)
    if any(row["status"] != "complete" or row["incumbent_exact"]
           and not row["tpot_reportable"] for row in rows):
        raise RuntimeError("confirmation contains invalid or incomplete measurements")
    if {row["runtime_identity_sha256"] for row in rows} \
            != {plan["runtime_identity_sha256"]} \
            or {row["normalization_sha256"] for row in rows} \
            != {plan["normalization_sha256"]}:
        raise RuntimeError("confirmation runtime differs from discovery")
    if any(row["kv_capacity_tokens"] != plan["kv_capacity_tokens"]
           or row["planned_parked_prefix_tokens"]
           != plan["planned_parked_prefix_tokens"]
           or abs(row["preloaded_kv_usage"]
                  - plan["expected_resident_preloaded_kv_usage"])
           > plan["kv_match_tolerance"] for row in rows):
        raise RuntimeError("confirmation parked KV stock differs from discovery")
    checks = {}
    baseline = all(row_feasible(row, plan["targets"]) for row in rows
                   if row["role"] == "baseline")
    for direction in ("prefill_heavy", "decode_heavy"):
        checks[f"{direction}:baseline"] = baseline
        for role in ("last_pass", "first_fail"):
            labels = [row_feasible(row, plan["targets"]) for row in rows
                      if row["direction"] == direction and row["role"] == role]
            checks[f"{direction}:{role}"] = all(labels) if role != "first_fail" \
                else not any(labels)
    balanced = [row_feasible(row, plan["targets"]) for row in rows
                if row["role"] == "balanced"]
    checks["balanced"] = all(balanced)
    usable = plan["source_residency_control_pass"] and all(checks.values())
    return {"schema": CONFIRM_SCHEMA, "stage": "confirmation",
            "hardware": plan["hardware"], "plan_sha256": digest(plan),
            "source_plan_sha256": plan["source_plan_sha256"],
            "source_scout_sha256": plan["source_scout_sha256"],
            "targets": plan["targets"], "checks": checks,
            "runtime_identity": plan["runtime_identity"],
            "runtime_identity_sha256": plan["runtime_identity_sha256"],
            "normalization_sha256": plan["normalization_sha256"],
            "planner_usable": usable,
            "supported_bound": plan["selection"]["supported_candidate"] if usable else None,
            "rows": rows}


def supported_bound(result: dict, plan: dict, core: dict, scout: dict) -> float:
    validate_confirmation_source(plan, core, scout)
    try:
        expected = build_confirmation(plan, result["rows"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("confirmation result is not eligible for planner use") from exc
    if result != expected or not expected["planner_usable"]:
        raise RuntimeError("confirmation result is not eligible for planner use")
    return float(result["supported_bound"])


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def read_rates(path: Path, hardware: str, plan_sha256: str) -> dict:
    rates = json.loads(path.read_text())
    checksum = rates.get("sha256")
    body = {key: value for key, value in rates.items() if key != "sha256"}
    if rates.get("schema") != SCHEMA or rates.get("hardware") != hardware \
            or rates.get("context_tokens") != CONTEXT or checksum != digest(body) \
            or rates.get("kv_capacity_tokens", 0) <= 0 \
            or rates.get("balanced_shape") != asdict(balanced_shape(rates)) \
            or rates.get("planned_parked_prefix_tokens") \
            != parked_prefix_tokens(rates) \
            or rates.get("plan_sha256") != plan_sha256 \
            or rates.get("runtime_identity_sha256") \
            != identity_sha(rates.get("runtime_identity", {})):
        raise ValueError("invalid exact-stack service normalization")
    validate_rates(rates)
    return rates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--model", choices=testbed.MODEL_SPECS,
                         default=testbed.MODEL)
    confirmation = sub.add_parser("prepare-confirmation")
    confirmation.add_argument("--plan", type=Path, required=True)
    confirmation.add_argument("--scout", type=Path, required=True)
    confirmation.add_argument("--hardware", choices=HARDWARE, required=True)
    confirmation.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-cell")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--cell-id", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--normalization", type=Path)
    testbed.add_common(run)
    run.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    calibration = sub.add_parser("reduce-calibration")
    calibration.add_argument("--plan", type=Path, required=True)
    calibration.add_argument("--hardware", choices=HARDWARE, required=True)
    calibration.add_argument("--runs", type=Path, required=True)
    calibration.add_argument("--out", type=Path, required=True)
    reduce = sub.add_parser("reduce")
    reduce.add_argument("--plan", type=Path, required=True)
    reduce.add_argument("--hardware", choices=HARDWARE, required=True)
    reduce.add_argument("--runs", type=Path, required=True)
    reduce.add_argument("--ttft-target-s", type=float, required=True)
    reduce.add_argument("--tpot-target-s", type=float, required=True)
    reduce.add_argument("--out", type=Path, required=True)
    final = sub.add_parser("reduce-confirmation")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--core-plan", type=Path, required=True)
    final.add_argument("--scout", type=Path, required=True)
    final.add_argument("--runs", type=Path, required=True)
    final.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(make_plan(args.model), indent=2) + "\n")
        return
    if args.command == "prepare-confirmation":
        plan = make_confirmation_plan(
            read_plan(args.plan), json.loads(args.scout.read_text()), args.hardware,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + "\n")
        return
    plan = read_plan(args.plan)
    if args.command == "run-cell":
        cell = next((row for row in plan["cells"] if row["cell_id"] == args.cell_id), None)
        if cell is None:
            raise ValueError("unknown service-headroom cell")
        cfg = testbed.config_from_args(args)
        if plan["model"] != testbed.MODEL:
            cfg = replace(cfg, service_campaign=True)
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] \
            else args.extra_vllm_args
        identity = collect_runtime_identity(
            plan, cfg, cell["hardware"], extra,
            runtime_provenance(
                cfg, args.out / f".{cell['hardware']}-image-sha256.json",
            ),
        )
        source_plan_sha = digest(plan) if plan["schema"] == SCHEMA \
            else plan["source_plan_sha256"]
        rates = None if cell["kind"] == "calibration" else read_rates(
            args.normalization, cell["hardware"], source_plan_sha,
        ) if args.normalization else None
        if cell["kind"] != "calibration" and rates is None:
            raise ValueError("headroom cells require --normalization")
        root = args.out / cell["cell_id"]
        if (root / "result.json").exists():
            result = json.loads((root / "result.json").read_text())
            validate_resume(result, plan, cell, identity,
                            rates["sha256"] if rates else None)
            if result.get("status") == "complete":
                return
            serving.archive_checkpoint(root / "result.json")
        if cell["kind"] == "calibration":
            run_calibration(plan, cell, cfg, root, extra, identity)
        else:
            run_headroom(plan, cell, rates, cfg, root, extra, identity)
        return
    if args.command == "reduce-calibration":
        result = reduce_calibration(plan, args.hardware, args.runs)
    elif args.command == "reduce":
        result = reduce_headroom(plan, args.hardware, args.runs,
                                 args.ttft_target_s, args.tpot_target_s)
    else:
        result = reduce_confirmation(
            plan, args.runs, read_plan(args.core_plan),
            json.loads(args.scout.read_text()),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
