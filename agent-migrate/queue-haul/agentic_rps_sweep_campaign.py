"""Collect simple single-A100 TTFT/TPOT curves under increasing agentic RPS.

The sweep is deliberately descriptive.  Every configured rate runs, and SLO
violations, request failures, and engine exits are recorded as outcomes rather
than used as campaign gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import service_headroom_campaign as headroom
import single_gpu_capacity_campaign as capacity


SCHEMA = "queue-haul-agentic-rps-sweep-v3"
PARENT_SCHEMA = "queue-haul-agentic-rps-sweep-v2"
PARENT_PLAN_SHA256 = "194ad7d6e376e903fb7ce3db7f40df925942f8cb21b91e2d6fb890a39825512d"
HISTORICAL_RESULT_IDENTITIES = {
    ("queue-haul-agentic-rps-sweep-v1",
     "4709014a6cbaa32104531be1c9e0482094a4f3ac6d155fb44d015f13473b67ed"),
    (PARENT_SCHEMA, PARENT_PLAN_SHA256),
}
MODELS = tuple(testbed.MODEL_SPECS)
BASE_RATES_RPS = (.125, .25, .5, 1.0, 2.0, 4.0, 8.0)
REFINEMENT_RATES_RPS = {
    "openai/gpt-oss-20b": (3.0, 5.0, 6.0, 7.0),
    "Qwen/Qwen3.8-27B": (.6, .7, .8, .9),
    "google/gemma-4-26B-A4B-it": (3.0, 5.0, 6.0, 7.0),
}
RATES_RPS_BY_MODEL = {
    model: tuple(sorted((*BASE_RATES_RPS, *REFINEMENT_RATES_RPS[model])))
    for model in MODELS
}
RATES_RPS = tuple(sorted({
    rate for rates in RATES_RPS_BY_MODEL.values() for rate in rates
}))
PROMPT_TOKENS = 3920
OUTPUT_TOKENS = 1024
REQUESTS_PER_POINT = 32
BOUNDARY_REPEATS = (1, 2)
REQUEST_TIMEOUT_S = 1800.0
FIXED_SLOS = {
    "openai/gpt-oss-20b": {"p90_ttft_s": 2.0,
                            "p90_tpot_s": .1},
    "google/gemma-4-26B-A4B-it": {"p90_ttft_s": 2.0,
                                   "p90_tpot_s": .2},
}


def digest(value) -> str:
    return profiler.object_hash(value)


def slug(value: str) -> str:
    return capacity.slug(value)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _plan(seed: int, hardware: str) -> dict:
    return {
        "schema": SCHEMA,
        "campaign": "agentic_rps_sweep",
        "hardware": hardware,
        "models": list(MODELS),
        "model_revisions": {
            model: testbed.model_spec(model).revision for model in MODELS
        },
        "request_shape": {
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "source": "fixed compact shape derived from the OpenHands coding trace",
        },
        "rates_rps": list(RATES_RPS),
        "rates_rps_by_model": {
            model: list(RATES_RPS_BY_MODEL[model]) for model in MODELS
        },
        "base_rates_rps": list(BASE_RATES_RPS),
        "refinement_rates_rps_by_model": {
            model: list(REFINEMENT_RATES_RPS[model]) for model in MODELS
        },
        "parent": {
            "schema": PARENT_SCHEMA,
            "plan_sha256": PARENT_PLAN_SHA256,
            "reusable_rates_rps": [],
            "relationship": (
                "same request and runtime contract; cells require re-reduction "
                "from token timestamps because TPOT aggregation changed"
            ),
        },
        "implementation": {
            "campaign_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
        "requests_per_point": REQUESTS_PER_POINT,
        "boundary_repeats": list(BOUNDARY_REPEATS),
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "slo": {
            "fixed": FIXED_SLOS,
            "relative_models": [model for model in MODELS
                                if model not in FIXED_SLOS],
            "relative_baseline_rps": BASE_RATES_RPS[0],
            "relative_multiplier": 2.0,
        },
        "runtime": capacity.make_runtime_contract(hardware == "a100"),
        "semantics": {
            "open_loop_poisson": True,
            "max_concurrency": None,
            "run_all_rates_after_violation": True,
            "slo_is_control_flow": False,
            "service_failures_are_outcomes": True,
            "unique_private_prompts": True,
            "forced_exact_output_length": True,
            "one_engine_per_model_unless_service_restart_is_needed": True,
            "refinement_points_predeclared": True,
            "tpot_definition": "p90_of_all_exact_post_first_token_intervals",
        },
        "seed": seed,
    }


def make_plan(seed: int = 1, hardware: str = "a100") -> dict:
    plan = _plan(seed, hardware)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    seed = plan.get("seed")
    hardware = plan.get("hardware")
    if not isinstance(seed, int) or hardware not in {"a100", "h100"}:
        raise ValueError("invalid agentic RPS sweep plan")
    if plan != _plan(seed, hardware):
        raise ValueError("invalid agentic RPS sweep plan")


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def stable_seed(plan: dict, rate: float, repeat: int, purpose: str) -> int:
    payload = f"{plan['seed']}:{rate:.8g}:{repeat}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def arrival_offsets(plan: dict, model: str, rate: float,
                    repeat: int) -> tuple[float, ...]:
    if model not in MODELS \
            or rate not in plan["rates_rps_by_model"][model] \
            or repeat < 0:
        raise ValueError("unsupported RPS cell")
    return serving.poisson_schedule(
        rate, plan["requests_per_point"],
        stable_seed(plan, rate, repeat, "arrivals"),
    )


def cell_id(model: str, rate: float, repeat: int) -> str:
    rate_text = f"{rate:g}".replace(".", "p")
    return f"{slug(model)}-rps{rate_text}-b{repeat}"


def cell_spec(model: str, rate: float, repeat: int) -> dict:
    return {
        "cell_id": cell_id(model, rate, repeat),
        "model": model,
        "revision": testbed.model_spec(model).revision,
        "offered_rps": rate,
        "repeat": repeat,
    }


def prepared_trace(plan: dict, model: str, rate: float,
                   repeat: int) -> list[dict]:
    rows = []
    for index, offset in enumerate(arrival_offsets(
            plan, model, rate, repeat)):
        session = serving.Session(
            session_id=f"agentic-{slug(model)}-{rate:g}-{repeat}-{index}",
            prefix_tokens=1,
            append_tokens=plan["request_shape"]["prompt_tokens"] - 1,
            output_tokens=plan["request_shape"]["output_tokens"],
            vocabulary=1024,
            seed=stable_seed(plan, rate, repeat, f"prompt-{index}"),
        )
        rows.append({
            "offset_s": offset,
            "population": "agentic",
            "prepared": serving.prepare_issue(
                session, 0, model, bypass_lmcache=True,
            ),
        })
    return rows


def model_config(model: str, hardware: str = "a100") -> testbed.Config:
    return replace(capacity.model_config(model), enforce_eager=hardware == "a100")


def runtime_identity(plan: dict, cfg: testbed.Config, commands: dict) -> dict:
    git_sha, dirty = profiler.git_state(True)
    identity = {
        "plan_sha256": digest(plan),
        "git_sha": git_sha,
        "git_dirty": dirty,
        "model": cfg.model,
        "revision": testbed.model_spec(cfg.model).revision,
        "hardware": capacity.gpu_snapshot(plan["hardware"].upper()),
        "runtime_mode": testbed.runtime_mode(),
        "runtime_versions": testbed.runtime_versions(cfg),
        "scheduler": plan["runtime"],
        "commands": commands,
    }
    identity["sha256"] = digest(identity)
    return identity


def wait_for_drain(sampler: serving.MetricsSampler, engine) -> bool:
    """Wait for natural drain without turning a wall-time budget into a gate."""
    while engine.poll() is None and not sampler.error:
        if sampler.rows and not any(
                sampler.rows[-1].get(key, 0) for key in
                ("vllm:num_requests_running", "vllm:num_requests_waiting")):
            return True
        time.sleep(.1)
    return False


def summarize_cell(plan: dict, cell: dict, requests: list[dict],
                   metrics: list[dict], drained: bool,
                   client_error: Exception | None, engine_exited: bool,
                   sampler_error: Exception | None) -> dict:
    completed = [row for row in requests if serving.service_completion(row)]
    exact = [row for row in completed if serving.exact_token_timing(row)]
    ttft = [float(row["ttft_s"]) for row in completed
            if row.get("ttft_s") is not None]
    tpot = [float(value) for row in exact
            for value in row.get("token_itls_s", [])]
    request_mean_tpot = [float(row["mean_tpot_s"]) for row in exact
                         if row.get("mean_tpot_s") is not None]
    scheduled = sorted(int(row["scheduled_ns"]) for row in requests)
    starts = sorted(int(row["start_ns"]) for row in requests)
    peak_running = max((row.get("vllm:num_requests_running", 0)
                        for row in metrics), default=None)
    peak_waiting = max((row.get("vllm:num_requests_waiting", 0)
                        for row in metrics), default=None)
    return {
        "schema": SCHEMA,
        "plan_sha256": digest(plan),
        **cell,
        "status": "recorded",
        "offered": plan["requests_per_point"],
        "completed": len(completed),
        "failed": plan["requests_per_point"] - len(completed),
        "exact_timing": len(exact),
        "p90_ttft_s": float(np.quantile(ttft, .9)) if ttft else None,
        "p90_tpot_s": float(np.quantile(tpot, .9)) if tpot else None,
        "diagnostic_p90_request_mean_tpot_s": (
            float(np.quantile(request_mean_tpot, .9))
            if request_mean_tpot else None
        ),
        "tpot_samples": len(tpot),
        "scheduled_span_s": ((scheduled[-1] - scheduled[0]) / 1e9
                             if len(scheduled) > 1 else 0.0),
        "actual_start_span_s": ((starts[-1] - starts[0]) / 1e9
                                if len(starts) > 1 else 0.0),
        "max_send_lateness_s": max(
            (float(row.get("send_lateness_s", 0)) for row in requests),
            default=None,
        ),
        "peak_running_requests": peak_running,
        "peak_waiting_requests": peak_waiting,
        "drained": drained,
        "engine_exited": engine_exited,
        "client_error": (f"{type(client_error).__name__}: {client_error}"
                         if client_error else None),
        "sampler_error": (f"{type(sampler_error).__name__}: {sampler_error}"
                          if sampler_error else None),
    }


def run_cell(plan: dict, cell: dict, cfg: testbed.Config, stack,
             root: Path) -> dict:
    trace = prepared_trace(plan, cfg.model, cell["offered_rps"], cell["repeat"])
    cell_root = root / "cells" / cell["cell_id"]
    cell_root.mkdir(parents=True, exist_ok=True)
    sampler = serving.MetricsSampler(
        cfg.host, stack.port, cell_root / "engine.csv", period_s=.1,
    )
    sampler.start()
    headroom.wait_sampler(sampler)
    epoch = time.monotonic_ns() + 1_000_000_000
    requests, client_error = headroom.issue_async_trace(
        cfg.host, stack.port, trace, epoch, plan["request_timeout_s"],
        shards=min(8, len(trace)),
    )
    drained = wait_for_drain(sampler, stack.engine)
    sampler_error = sampler.error
    try:
        sampler.close()
    except RuntimeError as exc:
        sampler_error = sampler_error or exc
    engine_exited = stack.engine.poll() is not None
    result = summarize_cell(
        plan, cell, requests, sampler.rows, drained, client_error,
        engine_exited, sampler_error,
    )
    write_json(cell_root / "requests.json", requests)
    write_json(cell_root / "result.json", result)
    return result


def result_path(root: Path, cell: dict) -> Path:
    return root / "cells" / cell["cell_id"] / "result.json"


def read_result(plan: dict, cell: dict, path: Path) -> dict:
    result = json.loads(path.read_text())
    same_cell = all(result.get(key) == value for key, value in cell.items())
    current = result.get("schema") == SCHEMA \
        and result.get("plan_sha256") == digest(plan)
    if not same_cell or not current:
        raise RuntimeError(f"stale or invalid sweep result: {cell['cell_id']}")
    return result


def rereduce_cell(plan: dict, source: Path, destination: Path,
                  source_label: str, source_origin: str) -> dict:
    """Recompute one historical cell from its retained token timestamps."""
    old_result_path = source / "result.json"
    requests_path = source / "requests.json"
    old = json.loads(old_result_path.read_text())
    requests = json.loads(requests_path.read_text())
    identity = (old.get("schema"), old.get("plan_sha256"))
    model = old.get("model")
    rate = old.get("offered_rps")
    repeat = old.get("repeat")
    if identity not in HISTORICAL_RESULT_IDENTITIES \
            or model not in MODELS \
            or rate not in model_rates(plan, model) \
            or repeat not in (0, *BOUNDARY_REPEATS):
        raise RuntimeError(f"invalid historical sweep cell: {source}")
    cell = cell_spec(model, rate, repeat)
    if any(old.get(key) != value for key, value in cell.items()):
        raise RuntimeError(f"historical sweep cell identity mismatch: {source}")
    derived = summarize_cell(
        plan, cell, requests, [], bool(old.get("drained")), None,
        bool(old.get("engine_exited")), None,
    )
    for field in (
        "peak_running_requests", "peak_waiting_requests", "drained",
        "engine_exited", "client_error", "sampler_error",
    ):
        derived[field] = old.get(field)
    derived.update({
        "source_schema": old["schema"],
        "source_plan_sha256": old["plan_sha256"],
        "source_result_sha256": hashlib.sha256(
            old_result_path.read_bytes()).hexdigest(),
        "source_requests_sha256": hashlib.sha256(
            requests_path.read_bytes()).hexdigest(),
        "source_label": source_label,
        "source_root": source_origin,
    })
    write_json(destination / cell["cell_id"] / "result.json", derived)
    return derived


def rereduce_sources(plan: dict, sources: list[Path], root: Path,
                     models: tuple[str, ...],
                     source_labels: list[str] | None = None,
                     source_origins: list[str] | None = None) -> list[dict]:
    """Pool historical raw cells into a new result root without rerunning."""
    selected = set(models)
    labels = source_labels or [source.name for source in sources]
    origins = source_origins or [str(source) for source in sources]
    if len(labels) != len(sources) or len(set(labels)) != len(labels):
        raise ValueError("source labels must be unique and match source roots")
    if len(origins) != len(sources):
        raise ValueError("source origins must match source roots")
    written = {}
    rows = []
    for source_root, source_label, source_origin in zip(
            sources, labels, origins):
        for requests_path in sorted(source_root.glob("*/requests.json")):
            source = requests_path.parent
            old = json.loads((source / "result.json").read_text())
            if old.get("model") not in selected:
                continue
            cell = old.get("cell_id")
            source_hash = hashlib.sha256(requests_path.read_bytes()).hexdigest()
            if cell in written:
                if written[cell] != source_hash:
                    raise RuntimeError(f"conflicting historical sweep cell: {cell}")
                continue
            result = rereduce_cell(
                plan, source, root / "cells", source_label, source_origin,
            )
            written[cell] = source_hash
            rows.append(result)
    write_json(root / "plan.json", plan)
    return rows


def run_specs(plan: dict, model: str, specs: list[dict], root: Path) -> None:
    pending = []
    for cell in specs:
        path = result_path(root, cell)
        if path.exists():
            read_result(plan, cell, path)
        else:
            pending.append(cell)
    if not pending:
        return
    cfg = model_config(model, plan["hardware"])
    restart = len(list((root / "stacks" / slug(model)).glob("restart-*")))
    while pending:
        commands = capacity.stack_commands(cfg)
        identity = runtime_identity(plan, cfg, commands)
        stack_root = root / "stacks" / slug(model) / f"restart-{restart:03d}"
        restart += 1
        with capacity.engine_stack(cfg, stack_root, identity, commands) as stack:
            write_json(stack_root / "runtime-identity.json", identity)
            for cell in list(pending):
                testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
                result = run_cell(plan, cell, cfg, stack, root)
                pending.remove(cell)
                if result["engine_exited"]:
                    break


def model_rates(plan: dict, model: str) -> tuple[float, ...]:
    return tuple(plan["rates_rps_by_model"][model])


def discovery_specs(plan: dict, model: str) -> list[dict]:
    return [cell_spec(model, rate, 0) for rate in model_rates(plan, model)]


def derive_slo(plan: dict, model: str, discovery: list[dict]) -> dict:
    if model in FIXED_SLOS:
        return {**FIXED_SLOS[model], "source": "fixed"}
    baseline_rate = plan["slo"]["relative_baseline_rps"]
    baseline = next(row for row in discovery
                    if row["offered_rps"] == baseline_rate)
    multiplier = plan["slo"]["relative_multiplier"]
    return {
        "p90_ttft_s": (baseline["p90_ttft_s"] * multiplier
                       if baseline.get("p90_ttft_s") is not None else None),
        "p90_tpot_s": (baseline["p90_tpot_s"] * multiplier
                       if baseline.get("p90_tpot_s") is not None else None),
        "source": f"{multiplier:g}x-{baseline_rate:g}-rps-baseline",
    }


def metric_violation(row: dict, slo: dict) -> bool:
    return any(
        row.get(field) is not None and slo.get(field) is not None
        and row[field] > slo[field]
        for field in ("p90_ttft_s", "p90_tpot_s")
    )


def boundary_rates(discovery: list[dict], slo: dict) -> tuple[float, ...]:
    ordered = sorted(discovery, key=lambda row: row["offered_rps"])
    for index, row in enumerate(ordered):
        if metric_violation(row, slo):
            rates = [row["offered_rps"]]
            if index:
                rates.insert(0, ordered[index - 1]["offered_rps"])
            return tuple(rates)
    return ()


def boundary_specs(model: str, rates: tuple[float, ...]) -> list[dict]:
    return [cell_spec(model, rate, repeat) for rate in rates
            for repeat in BOUNDARY_REPEATS]


def load_specs(plan: dict, root: Path, specs: list[dict]) -> list[dict]:
    return [read_result(plan, cell, result_path(root, cell)) for cell in specs]


def run_model(plan: dict, model: str, root: Path) -> None:
    discovery_cells = discovery_specs(plan, model)
    run_specs(plan, model, discovery_cells, root)
    discovery = load_specs(plan, root, discovery_cells)
    slo = derive_slo(plan, model, discovery)
    rates = boundary_rates(discovery, slo)
    write_json(root / "models" / slug(model) / "selection.json", {
        "model": model, "slo": slo, "repeated_boundary_rates": list(rates),
    })
    run_specs(plan, model, boundary_specs(model, rates), root)


def aggregate_rate(rows: list[dict], rate: float) -> dict:
    matches = [row for row in rows if row["offered_rps"] == rate]
    aggregate = {"offered_rps": rate, "repeats": len(matches)}
    for field in ("p90_ttft_s", "p90_tpot_s"):
        values = [float(row[field]) for row in matches
                  if row.get(field) is not None]
        aggregate.update({
            f"{field}_median": statistics.median(values) if values else None,
            f"{field}_minimum": min(values) if values else None,
            f"{field}_maximum": max(values) if values else None,
        })
    aggregate["completed_minimum"] = min(
        (row["completed"] for row in matches), default=0)
    aggregate["failed_maximum"] = max(
        (row["failed"] for row in matches), default=0)
    return aggregate


def reduce_model(plan: dict, model: str, root: Path) -> tuple[list[dict], dict]:
    discovery_cells = discovery_specs(plan, model)
    discovery = load_specs(plan, root, discovery_cells)
    slo = derive_slo(plan, model, discovery)
    rates = boundary_rates(discovery, slo)
    extra_cells = boundary_specs(model, rates)
    rows = discovery + load_specs(plan, root, extra_cells)
    aggregated = [aggregate_rate(rows, rate)
                  for rate in model_rates(plan, model)]
    confirmed = next((
        row["offered_rps"] for row in aggregated
        if row["repeats"] >= 3 and metric_violation({
            "p90_ttft_s": row["p90_ttft_s_median"],
            "p90_tpot_s": row["p90_tpot_s_median"],
        }, slo)
    ), None)
    return rows, {
        "model": model,
        "revision": testbed.model_spec(model).revision,
        "slo": slo,
        "repeated_boundary_rates": list(rates),
        "first_confirmed_violation_rps": confirmed,
        "curve": aggregated,
    }


def reduce(plan: dict, root: Path, models: tuple[str, ...] = MODELS) -> dict:
    rows, model_results = [], {}
    for model in models:
        model_rows, result = reduce_model(plan, model, root)
        rows.extend(model_rows)
        model_results[model] = result
    return {
        "schema": SCHEMA,
        "stage": "reduced",
        "plan_sha256": digest(plan),
        "hardware": plan["hardware"],
        "request_shape": plan["request_shape"],
        "campaign_gate": False,
        "source_plan_sha256s": sorted({row["plan_sha256"] for row in rows}),
        "rows": rows,
        "models": model_results,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(
        key for row in rows for key in row
        if key != "diagnostic_p90_request_mean_tpot_s"
    ))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(({key: row.get(key) for key in fields} for row in rows))


def run_campaign(plan: dict, root: Path, models: tuple[str, ...]) -> dict | None:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    for model in models:
        run_model(plan, model, root)
    if set(models) == set(MODELS):
        summary = reduce(plan, root)
        write_json(root / "summary.json", summary)
        write_csv(root / "rps-sweep.csv", summary["rows"])
        return summary
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=1)
    prepare.add_argument("--hardware", choices=("a100", "h100"), default="a100")
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--model", choices=MODELS, action="append")
    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("--plan", type=Path, required=True)
    reduce_parser.add_argument("--run-root", type=Path, required=True)
    reduce_parser.add_argument("--out", type=Path, required=True)
    reduce_parser.add_argument("--csv", type=Path)
    reduce_parser.add_argument("--model", choices=MODELS, action="append")
    rereduce = commands.add_parser("rereduce")
    rereduce.add_argument("--plan", type=Path, required=True)
    rereduce.add_argument("--source-root", type=Path, action="append",
                         required=True)
    rereduce.add_argument("--source-label", action="append")
    rereduce.add_argument("--source-origin", action="append")
    rereduce.add_argument("--run-root", type=Path, required=True)
    rereduce.add_argument("--model", choices=MODELS, action="append")
    rereduce.add_argument("--csv", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        write_json(args.out, make_plan(args.seed, args.hardware))
        return
    plan = read_plan(args.plan)
    if args.command == "run":
        models = tuple(args.model or MODELS)
        run_campaign(plan, args.run_root, models)
        return
    if args.command == "rereduce":
        models = tuple(args.model or MODELS)
        rows = rereduce_sources(
            plan, args.source_root, args.run_root, models, args.source_label,
            args.source_origin,
        )
        if args.csv:
            write_csv(args.csv, rows)
        return
    summary = reduce(plan, args.run_root, tuple(args.model or MODELS))
    write_json(args.out, summary)
    write_csv(args.csv or args.out.with_suffix(".csv"), summary["rows"])


if __name__ == "__main__":
    main()
