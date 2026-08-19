"""Calibrate optimized-H100 prefill, RPS/SLO, and power evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import agentic_rps_sweep_campaign as rps
import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import power_model_campaign as power
import service_headroom_campaign as headroom
import single_gpu_capacity_campaign as capacity


SCHEMA = "queue-haul-h100-serving-calibration-v1"
MODELS = ("Qwen/Qwen3.8-27B", "google/gemma-4-26B-A4B-it")
PREFILL_MODELS = (*MODELS, "openai/gpt-oss-20b")
CONTEXTS = (256, 1024, 4096, 8192, 16384, 24576, 28672, 32767)
REPEATS = 3


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cells() -> list[dict]:
    return [
        {"cell_id": f"ctx{context}-c1-r{repeat}", "context_tokens": context,
         "concurrency": 1, "repeat": repeat}
        for context in CONTEXTS for repeat in range(REPEATS)
    ] + [
        {"cell_id": f"ctx8192-c16-r{repeat}", "context_tokens": 8192,
         "concurrency": 16, "repeat": repeat}
        for repeat in range(REPEATS)
    ]


def make_plan(seed: int = 1) -> dict:
    plan = make_plan_unchecked(seed)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    seed = plan.get("seed")
    if not isinstance(seed, int) or plan != make_plan_unchecked(seed):
        raise ValueError("invalid H100 serving campaign plan")


def make_plan_unchecked(seed: int) -> dict:
    return {
        "schema": SCHEMA, "hardware": "h100", "models": list(MODELS),
        "model_revisions": {model: testbed.model_spec(model).revision
                            for model in MODELS},
        "runtime": {
            "gpu_count": 1, "tensor_parallel_size": 1, "dtype": "bfloat16",
            "kv_cache_dtype": "auto", "max_model_len": 32768,
            "max_num_seqs": 256, "gpu_memory_utilization": .9,
            "chunked_prefill": True, "prefix_caching": True,
            "enforce_eager": False,
        },
        "prefill": {"output_tokens": 1, "cells": cells()},
        "rps_plan": rps.make_plan(seed, "h100"),
        "power_cells": [asdict(cell) for cell in power.cells(seed)],
        "power_max_ell": power.MAX_ELL,
        "seed": seed,
    }


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def config(model: str) -> testbed.Config:
    return rps.model_config(model, "h100")


def trace(plan: dict, model: str, cell: dict) -> list[dict]:
    return [{
        "offset_s": 0, "population": "prefill",
        "prepared": serving.prepare_issue(
            serving.Session(
                f"prefill-{cell['cell_id']}-{index}", 1,
                cell["context_tokens"] - 1, 1, 1024,
                plan["seed"] + cell["repeat"] * 100 + index,
            ), 0, model, bypass_lmcache=True,
        ),
    } for index in range(cell["concurrency"])]


def summarize(plan: dict, model: str, cell: dict, rows: list[dict],
              identity_sha256: str) -> dict:
    exact = [row for row in rows if serving.service_completion(row)
             and serving.exact_token_timing(row)]
    if len(exact) != cell["concurrency"] \
            or any(row.get("prompt_tokens") != cell["context_tokens"]
                   or row.get("output_tokens") != 1
                   or row.get("cached_tokens", 0) for row in exact):
        raise RuntimeError("prefill cell lacks complete uncached exact-token evidence")
    start, end = min(row["start_ns"] for row in exact), max(
        row["end_ns"] for row in exact)
    duration = (end - start) / 1e9
    if duration <= 0:
        raise RuntimeError("prefill cell has a nonpositive timing window")
    ttft = [float(row["ttft_s"]) for row in exact]
    return {
        "schema": SCHEMA, "plan_sha256": profiler.object_hash(plan),
        "model": model, "revision": testbed.model_spec(model).revision,
        **cell, "requests": len(exact), "window_s": duration,
        "prefill_tokens": sum(row["prompt_tokens"] for row in exact),
        "prefill_tps": sum(row["prompt_tokens"] for row in exact) / duration,
        "ttft_median_s": statistics.median(ttft),
        "ttft_p90_s": float(np.quantile(ttft, .9)),
        "runtime_identity_sha256": identity_sha256,
    }


def run_prefill(plan: dict, model: str, root: Path) -> None:
    if model not in PREFILL_MODELS:
        raise ValueError("unsupported calibration model")
    cfg = config(model)
    commands = capacity.stack_commands(cfg)
    identity = rps.runtime_identity(plan, cfg, commands)
    stack_root = root / "stack"
    with capacity.engine_stack(cfg, stack_root, identity, commands) as stack:
        testbed.validate_h100_optimized_runtime(
            testbed.shell(commands["vllm"]), testbed.read_text(stack.log))
        write_json(stack_root / "runtime-identity.json", identity)
        for cell in plan["prefill"]["cells"]:
            result = root / "cells" / cell["cell_id"] / "result.json"
            if result.exists():
                old = json.loads(result.read_text())
                if old.get("plan_sha256") != profiler.object_hash(plan) \
                        or old.get("model") != model \
                        or any(old.get(key) != value for key, value in cell.items()):
                    raise RuntimeError("stale prefill evidence")
                continue
            testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
            requests = trace(plan, model, cell)
            rows, error = headroom.issue_async_trace(
                cfg.host, stack.port, requests,
                time.monotonic_ns() + 1_000_000_000, 1800,
                min(16, len(requests)))
            if error:
                raise error
            write_json(result.parent / "requests.json", rows)
            write_json(result, summarize(plan, model, cell, rows,
                                         identity["sha256"]))


def reduce_prefill(plan: dict, model: str, root: Path) -> dict:
    rows = [json.loads((root / "cells" / cell["cell_id"] / "result.json").read_text())
            for cell in plan["prefill"]["cells"]]
    if any(row.get("plan_sha256") != profiler.object_hash(plan)
           or row.get("model") != model for row in rows):
        raise RuntimeError("stale prefill evidence")
    curve = []
    for context in CONTEXTS:
        selected = [row for row in rows if row["context_tokens"] == context
                    and row["concurrency"] == 1]
        if len(selected) != REPEATS:
            raise RuntimeError("incomplete prefill timing repeats")
        curve.append({
            "context_tokens": context,
            "prefill_tps_median": statistics.median(
                row["prefill_tps"] for row in selected),
            "ttft_median_s": statistics.median(
                row["ttft_median_s"] for row in selected),
            "ttft_p90_s": float(np.quantile(
                [row["ttft_p90_s"] for row in selected], .9)),
        })
    saturated = [row for row in rows if row["concurrency"] == 16]
    summary = {
        "schema": SCHEMA, "model": model,
        "revision": testbed.model_spec(model).revision,
        "plan_sha256": profiler.object_hash(plan), "curve": curve,
        "saturated_8192_prefill_tps_median": statistics.median(
            row["prefill_tps"] for row in saturated),
        "runtime_identity_sha256s": sorted({
            row["runtime_identity_sha256"] for row in rows}),
    }
    write_json(root / "summary.json", summary)
    with (root / "prefill.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0])
        writer.writeheader(); writer.writerows(curve)
    return summary


def validate_alignment(plan: dict, model: str, prefill: Path,
                       rps_summary: Path, power_root: Path) -> dict:
    prefill_value = json.loads(prefill.read_text())
    rps_value = json.loads(rps_summary.read_text())
    metadata = json.loads((power_root / "metadata.json").read_text())
    fit = json.loads((power_root / "fit.json").read_text())
    revision = plan["model_revisions"][model]
    if prefill_value.get("model") != model \
            or prefill_value.get("revision") != revision \
            or prefill_value.get("plan_sha256") != profiler.object_hash(plan) \
            or rps_value.get("schema") != rps.SCHEMA \
            or rps_value.get("hardware") != "h100" \
            or rps_value.get("plan_sha256") != profiler.object_hash(
                plan["rps_plan"]) \
            or model not in rps_value.get("models", {}) \
            or rps_value["models"][model].get("revision") != revision \
            or metadata.get("model") != model \
            or metadata.get("revision") != revision \
            or metadata.get("optimized_h100") is not True \
            or fit.get("max_ell") != power.MAX_ELL \
            or fit.get("power_curve", [{}])[-1].get("ell") != power.MAX_ELL \
            or "validation" not in fit:
        raise RuntimeError("serving evidence is not aligned")
    testbed.validate_h100_optimized_runtime(
        " ".join(metadata["server_command"]),
        (power_root / "server.log").read_text(errors="replace"))
    return {"schema": SCHEMA, "model": model, "revision": revision,
            "optimized_h100": True, "prefill": str(prefill),
            "rps": str(rps_summary), "power": str(power_root),
            "power_status": fit["status"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=1)
    for name in ("run-prefill", "reduce-prefill"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--model", choices=PREFILL_MODELS, required=True)
        command.add_argument("--root", type=Path, required=True)
    align = commands.add_parser("validate")
    align.add_argument("--plan", type=Path, required=True)
    align.add_argument("--model", choices=MODELS, required=True)
    align.add_argument("--prefill", type=Path, required=True)
    align.add_argument("--rps-summary", type=Path, required=True)
    align.add_argument("--power-root", type=Path, required=True)
    align.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        plan = make_plan(args.seed)
        write_json(args.out, plan)
        write_json(args.out.with_name("rps-plan.json"), plan["rps_plan"])
        return
    plan = read_plan(args.plan)
    if args.command == "run-prefill":
        run_prefill(plan, args.model, args.root)
    elif args.command == "reduce-prefill":
        reduce_prefill(plan, args.model, args.root)
    else:
        write_json(args.out, validate_alignment(
            plan, args.model, args.prefill, args.rps_summary, args.power_root))


if __name__ == "__main__":
    main()
