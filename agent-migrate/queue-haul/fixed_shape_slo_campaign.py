"""Measure fixed-shape service degradation against offered request rate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import service_headroom_campaign as service


SCHEMA = "queue-haul-fixed-shape-slo-v1"
RATES = (.125, .25, .5, 1., 2., 4., 8.)
INPUT_TOKENS = 3920
OUTPUT_TOKENS = 1024
REQUESTS = 32
BOUNDARY_REPEATS = 2
DEFAULT_SEED = 20260816


def rate_label(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def cell(rate: float, rate_index: int, replicate: int, seed: int) -> dict:
    return {"cell_id": f"rps{rate_label(rate)}-rep{replicate}",
            "offered_rps": rate, "replicate": replicate,
            "poisson_seed": seed + rate_index + replicate * len(RATES)}


def make_plan(model: str, ttft_slo_s: float = 1., tpot_slo_s: float = .1,
              seed: int = DEFAULT_SEED) -> dict:
    if model not in testbed.MODEL_SPECS or min(ttft_slo_s, tpot_slo_s) <= 0:
        raise ValueError("invalid fixed-shape campaign inputs")
    stack = service.make_plan(model)
    return {"schema": SCHEMA, "model": model, "hardware": "h100",
            "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
            "requests_per_point": REQUESTS, "rates_rps": list(RATES),
            "arrival": "open_loop_poisson", "max_concurrency": None,
            "seed": seed, "request_timeout_s": 1200, "drain_s": 1200,
            "ttft_slo_s": ttft_slo_s, "tpot_slo_s": tpot_slo_s,
            "boundary_repeats": BOUNDARY_REPEATS,
            "image_sha256": stack["image_sha256"], "stack": stack["stack"],
            "base_cells": [cell(rate, index, 0, seed)
                           for index, rate in enumerate(RATES)]}


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan != make_plan(plan.get("model", ""), plan.get("ttft_slo_s", 0),
                         plan.get("tpot_slo_s", 0), plan.get("seed", -1)):
        raise ValueError("fixed-shape plan is not canonical")
    return plan


def offered_trace(rate: float, seed: int) -> list[dict]:
    offsets = serving.poisson_schedule(rate, REQUESTS, seed)
    return [{"request_index": index, "offset_s": offset,
             "offered_rps": rate, "poisson_seed": seed}
            for index, offset in enumerate(offsets)]


def exact_sessions(seed: int) -> list[serving.Session]:
    return [serving.Session(f"fixed-shape-{index}", 1, INPUT_TOKENS - 1,
                            OUTPUT_TOKENS, 201088, seed)
            for index in range(REQUESTS)]


def summarize(plan: dict, spec: dict, requests: list[dict], metrics: list[dict],
              drained: bool, engine_exited: bool, failure_kind: str | None) -> dict:
    complete = [row for row in requests if serving.exact_token_timing(row)
                and row.get("prompt_tokens") == INPUT_TOKENS
                and row.get("output_tokens") == OUTPUT_TOKENS]
    ttft = [row["ttft_s"] for row in complete]
    tpot = [row["mean_tpot_s"] for row in complete]
    p90_ttft, p90_tpot = service.quantile(ttft, .9), service.quantile(tpot, .9)
    violation = len(complete) != REQUESTS or p90_ttft is None or p90_tpot is None \
        or p90_ttft > plan["ttft_slo_s"] or p90_tpot > plan["tpot_slo_s"]
    offsets = [row["offset_s"] for row in spec["offered_trace"]]
    queue = [row["vllm:num_requests_running"] + row["vllm:num_requests_waiting"]
             for row in metrics]
    return {**spec, "status": "complete", "offered_requests": REQUESTS,
            "realized_arrival_rps": ((REQUESTS - 1) / (offsets[-1] - offsets[0])
                                     if offsets[-1] > offsets[0] else math.inf),
            "exact_completions": len(complete),
            "exact_completion_rate": len(complete) / REQUESTS,
            "service_failure_rate": 1 - len(complete) / REQUESTS,
            "p90_ttft_s": p90_ttft, "p90_mean_tpot_s": p90_tpot,
            "ttft_slo_s": plan["ttft_slo_s"],
            "tpot_slo_s": plan["tpot_slo_s"], "slo_violation": violation,
            "drained": drained, "engine_exited": engine_exited,
            "engine_failure_kind": failure_kind,
            "cache_hit_requests": sum(row.get("cached_tokens", 0) > 0
                                      for row in complete),
            "max_send_lateness_s": max((row["send_lateness_s"]
                                         for row in requests), default=None),
            "max_in_system_requests": max(queue, default=None)}


def archive_partial(root: Path) -> None:
    if not root.exists():
        return
    interrupted = root.parent.parent / "interrupted"
    interrupted.mkdir(exist_ok=True)
    index = 1
    while (target := interrupted / f"{root.name}-attempt{index}").exists():
        index += 1
    root.rename(target)


def validate_resume(result: dict, plan: dict, expected: dict, identity: dict) -> None:
    if result.get("schema") != SCHEMA or result.get("plan_sha256") != service.digest(plan) \
            or any(result.get(key) != value for key, value in expected.items()
                   if key != "offered_trace") \
            or result.get("runtime_identity_sha256") != service.identity_sha(identity):
        raise RuntimeError("fixed-shape resume evidence changed")


def run_rate(plan: dict, expected: dict, cfg: testbed.Config, stack,
             identity: dict, root: Path, sessions: list[serving.Session]) -> dict:
    started_wall_ns = time.time_ns()
    root.mkdir(parents=True)
    trace = offered_trace(expected["offered_rps"], expected["poisson_seed"])
    spec = {**expected, "offered_trace": trace}
    (root / "offered.json").write_text(json.dumps(trace, indent=2) + "\n")
    prepared = [session.prompt(0) for session in sessions]
    bodies = [serving.completion_body(cfg.model, prompt, OUTPUT_TOKENS, forced, True)
              for prompt, forced in prepared]
    testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
    sampler = serving.MetricsSampler(cfg.host, stack.port, root / "engine.csv")
    power = profiler.PowerSampler(root / "power.csv")
    sampler.start(); power.start()
    requests, error, epoch = [], None, time.monotonic_ns() + 100_000_000
    engine_exited = False
    try:
        service.wait_sampler(sampler)
        def issue(index: int) -> dict:
            row = serving.issue(
                cfg.host, stack.port, cfg.model, sessions[index], index,
                epoch + int(trace[index]["offset_s"] * 1e9),
                plan["request_timeout_s"], True, prepared[index], bodies[index],
            )
            row.update({"input_tokens": INPUT_TOKENS, "prefix_tokens": 0,
                        "offset_s": trace[index]["offset_s"]})
            return row
        with ThreadPoolExecutor(max_workers=REQUESTS) as pool:
            futures = []
            for row in trace:
                scheduled = epoch / 1e9 + row["offset_s"]
                time.sleep(max(0, scheduled - time.monotonic()))
                futures.append(pool.submit(issue, row["request_index"]))
            requests, error = service.settle_futures(futures)
        drained = service.drain(sampler, plan["drain_s"]) if error is None else False
    except Exception as exc:
        error, drained = error or exc, False
    finally:
        try:
            engine_exited = service.close_samplers(sampler, power, stack.engine)
        except Exception as exc:
            error = error or exc
    (root / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    failure_kind = service.engine_failure_kind(stack.log, engine_exited)
    base = {"schema": SCHEMA, "plan_sha256": service.digest(plan), **spec,
            "runtime_identity": identity,
            "runtime_identity_sha256": service.identity_sha(identity),
            "started_wall_ns": started_wall_ns}
    if error or failure_kind == "infrastructure" or len(requests) != REQUESTS \
            or any(row.get("cached_tokens", 0) for row in requests
                   if serving.service_completion(row)):
        result = {**base, "status": "invalid",
                  "measurement_error": (f"{type(error).__name__}: {error}"
                                        if error else "invalid fixed-shape execution")}
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("fixed-shape measurement is invalid") from error
    result = {**base, **summarize(plan, spec, requests, sampler.rows, drained,
                                  engine_exited, failure_kind)}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def first_boundary(rows: list[dict]) -> tuple[float, float]:
    ordered = sorted((row for row in rows if row["replicate"] == 0),
                     key=lambda row: row["offered_rps"])
    index = next((index for index, row in enumerate(ordered)
                  if row["slo_violation"]), None)
    if index is None or index == 0:
        raise RuntimeError("SLO boundary is not bracketed by the offered rates")
    return ordered[index - 1]["offered_rps"], ordered[index]["offered_rps"]


def aggregate(rows: list[dict], plan: dict) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["offered_rps"], []).append(row)
    output = []
    for rate in plan["rates_rps"]:
        group = sorted(grouped[rate], key=lambda row: row["replicate"])
        item = {"model": plan["model"], "offered_rps": rate,
                "replicates": len(group),
                "exact_completion_rate_min": min(row["exact_completion_rate"]
                                                  for row in group)}
        for metric in ("p90_ttft_s", "p90_mean_tpot_s"):
            values = [row[metric] for row in group if row[metric] is not None]
            item[metric] = statistics.median(values) if values else None
            item[f"{metric}_min"] = min(values) if values else None
            item[f"{metric}_max"] = max(values) if values else None
        output.append(item)
    return output


def write_summary(root: Path, plan: dict, rows: list[dict], boundary: tuple[float, float],
                  identity: dict) -> dict:
    compact = [{key: value for key, value in row.items()
                if key not in ("offered_trace", "runtime_identity")}
               for row in sorted(rows, key=lambda row: (row["offered_rps"],
                                                         row["replicate"]))]
    summary = {"schema": SCHEMA, "model": plan["model"], "hardware": "h100",
               "plan_sha256": service.digest(plan), "runtime_identity": identity,
               "runtime_identity_sha256": service.identity_sha(identity),
               "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS,
               "requests_per_point": REQUESTS, "rates_rps": list(RATES),
               "ttft_slo_s": plan["ttft_slo_s"], "tpot_slo_s": plan["tpot_slo_s"],
               "boundary": {"predecessor_rps": boundary[0],
                            "first_violating_rps": boundary[1]},
               "runs": compact,
               "curve": aggregate(rows, plan)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, values in (("runs", summary["runs"]), ("curve", summary["curve"])):
        with (root / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=values[0], extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader(); writer.writerows(values)
    return summary


def run_model(plan: dict, cfg: testbed.Config, root: Path) -> None:
    if cfg.model != plan["model"]:
        raise ValueError("CLI model differs from fixed-shape plan")
    identity = service.collect_runtime_identity(
        plan, cfg, "h100", [], service.runtime_provenance(
            cfg, root / ".h100-image-sha256.json"))
    sessions = exact_sessions(plan["seed"])
    stack_index = len(list((root / "stacks").glob("attempt-*"))) + 1
    with service.destination_stack(cfg, root / "stacks" / f"attempt-{stack_index}",
                                   "h100", [], identity) as stack:
        rows = []
        for expected in plan["base_cells"]:
            path = root / "base" / expected["cell_id"]
            if (path / "result.json").exists():
                result = json.loads((path / "result.json").read_text())
                validate_resume(result, plan, expected, identity)
                if result["status"] == "complete":
                    rows.append(result); continue
                archive_partial(path)
            elif path.exists():
                archive_partial(path)
            rows.append(run_rate(plan, expected, cfg, stack, identity, path, sessions))
            if rows[-1]["engine_exited"] or not rows[-1]["drained"]:
                raise RuntimeError("service outcome requires a clean engine restart")
        boundary = first_boundary(rows)
        for replicate in range(1, BOUNDARY_REPEATS + 1):
            for rate in boundary:
                index = RATES.index(rate)
                expected = cell(rate, index, replicate, plan["seed"])
                path = root / "boundary" / expected["cell_id"]
                if (path / "result.json").exists():
                    result = json.loads((path / "result.json").read_text())
                    validate_resume(result, plan, expected, identity)
                    if result["status"] == "complete":
                        rows.append(result); continue
                    archive_partial(path)
                elif path.exists():
                    archive_partial(path)
                rows.append(run_rate(plan, expected, cfg, stack, identity, path, sessions))
                if rows[-1]["engine_exited"] or not rows[-1]["drained"]:
                    raise RuntimeError("service outcome requires a clean engine restart")
        write_summary(root, plan, rows, boundary, identity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--model", choices=testbed.MODEL_SPECS, required=True)
    prepare.add_argument("--ttft-slo-s", type=float, default=1.)
    prepare.add_argument("--tpot-slo-s", type=float, default=.1)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-model")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    testbed.add_common(run)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(make_plan(
            args.model, args.ttft_slo_s, args.tpot_slo_s, args.seed), indent=2) + "\n")
        return
    plan = read_plan(args.plan)
    cfg = replace(testbed.config_from_args(args), service_campaign=True)
    args.out.mkdir(parents=True, exist_ok=True)
    run_model(plan, cfg, args.out)


if __name__ == "__main__":
    main()
