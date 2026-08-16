"""Confirm live cohort admission and its sustained incumbent service impact."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import service_headroom_campaign as headroom


SCHEMA = "queue-haul-service-admission-transition-v1"
HARDWARE = "a100"
DIRECTIONS = ("prefill_heavy", "balanced", "decode_heavy")
BLOCKS = (6, 7, 8)
TARGET_RHO = .50
BASELINE_S = 60
TRANSITION_S = 30
MEASUREMENT_S = 240
TARGETS = {"p90_ttft_s": 1.0, "p90_mean_tpot_s": .1}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cells() -> list[dict]:
    return [
        {"cell_id": f"a100-transition-{direction}-rho0.50-b{block}",
         "hardware": HARDWARE, "kind": "admission_transition",
         "direction": direction, "target_rho": TARGET_RHO,
         "block": block}
        for direction in DIRECTIONS for block in BLOCKS
    ]


def _discovery_half_load_passes(scout: dict) -> bool:
    for direction in ("prefill_heavy", "decode_heavy"):
        selected = [row for row in scout["rows"]
                    if row["direction"] == direction
                    and row["target_rho"] == TARGET_RHO]
        if len(selected) != 3 or not all(
                headroom.row_feasible(row, scout["targets"])
                for row in selected):
            return False
    return True


def make_plan(core: dict, rates: dict, scout: dict,
              confirmation_plan: dict, confirmed: dict) -> dict:
    headroom.validate_plan(core)
    headroom.validate_rates(rates)
    headroom.validate_scout_evidence(core, scout, HARDWARE)
    headroom.validate_confirmation_evidence(
        confirmed, confirmation_plan, core, scout,
    )
    if rates.get("hardware") != HARDWARE \
            or rates.get("plan_sha256") != headroom.digest(core) \
            or rates.get("sha256") != scout.get("normalization_sha256") \
            or confirmed.get("normalization_sha256") != rates.get("sha256") \
            or scout.get("targets") != TARGETS \
            or confirmed.get("targets") != TARGETS \
            or not _discovery_half_load_passes(scout):
        raise RuntimeError("source evidence does not support the frozen transition point")
    common = {key: core[key] for key in (
        "image_sha256", "model", "context_tokens", "drain_s",
        "request_timeout_s", "max_send_lateness_s", "max_client_tasks",
        "max_metric_gap_s", "min_exact_timing_coverage",
        "p99_min_incumbent_requests", "client", "stack", "shapes",
    )}
    planned_cells = cells()
    plan = {
        "schema": SCHEMA, **common, "hardware": HARDWARE,
        "base_rho": headroom.BASE_RHO, "target_rho": TARGET_RHO,
        "baseline_s": BASELINE_S, "transition_window_s": TRANSITION_S,
        "warmup_s": BASELINE_S + TRANSITION_S,
        "measurement_s": MEASUREMENT_S,
        "targets": TARGETS,
        "admission": {
            "initial_incumbent_sessions": headroom.SESSIONS_PER_SHAPE,
            "new_sessions": headroom.SESSIONS_PER_SHAPE,
            "start_offset_s": BASELINE_S,
            "ongoing_start_offset_s": BASELINE_S + TRANSITION_S,
            "launch_lead_s": .1,
            "claim": "three discrete resident-cohort recipes only",
        },
        "evidence_status": "transition_confirmation_only",
        "planner_usable": False,
        "source_plan_sha256": headroom.digest(core),
        "source_scout_sha256": headroom.digest(scout),
        "source_confirmation_plan_sha256": headroom.digest(confirmation_plan),
        "source_confirmed_sha256": headroom.digest(confirmed),
        "normalization_sha256": rates["sha256"],
        "service_runtime_identity_sha256":
        confirmed["service_runtime_identity_sha256"],
        "kv_capacity_tokens": rates["kv_capacity_tokens"],
        "phase_rates": {"prefill_tps": rates["prefill_tps"],
                        "decode_tps": rates["decode_tps"]},
        "balanced_shape": rates["balanced_shape"],
        "cells": planned_cells,
        "run_order": sorted(
            (cell["cell_id"] for cell in planned_cells),
            key=lambda value: headroom.digest(["transition-order", value]),
        ),
    }
    plan["recipes"] = {
        direction: phase_coordinates(plan, rates, direction)
        for direction in DIRECTIONS
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    core = headroom.make_plan()
    common = (
        "image_sha256", "model", "context_tokens", "drain_s",
        "request_timeout_s", "max_send_lateness_s", "max_client_tasks",
        "max_metric_gap_s", "min_exact_timing_coverage",
        "p99_min_incumbent_requests", "client", "stack", "shapes",
    )
    expected_cells = cells()
    expected_order = sorted(
        (cell["cell_id"] for cell in expected_cells),
        key=lambda value: headroom.digest(["transition-order", value]),
    )
    hashes = ("source_plan_sha256", "source_scout_sha256",
              "source_confirmation_plan_sha256", "source_confirmed_sha256",
              "normalization_sha256", "service_runtime_identity_sha256")
    admission = plan.get("admission", {})
    phase_rates = plan.get("phase_rates", {})
    expected_recipes = {
        direction: phase_coordinates(plan, phase_rates, direction)
        for direction in DIRECTIONS
    } if min(phase_rates.get("prefill_tps", 0),
             phase_rates.get("decode_tps", 0)) > 0 else None
    if plan.get("schema") != SCHEMA or plan.get("hardware") != HARDWARE \
            or any(plan.get(key) != core[key] for key in common) \
            or plan.get("base_rho") != headroom.BASE_RHO \
            or plan.get("target_rho") != TARGET_RHO \
            or plan.get("baseline_s") != BASELINE_S \
            or plan.get("transition_window_s") != TRANSITION_S \
            or plan.get("warmup_s") != BASELINE_S + TRANSITION_S \
            or plan.get("measurement_s") != MEASUREMENT_S \
            or plan.get("targets") != TARGETS \
            or admission != {
                "initial_incumbent_sessions": headroom.SESSIONS_PER_SHAPE,
                "new_sessions": headroom.SESSIONS_PER_SHAPE,
                "start_offset_s": BASELINE_S,
                "ongoing_start_offset_s": BASELINE_S + TRANSITION_S,
                "launch_lead_s": .1,
                "claim": "three discrete resident-cohort recipes only",
            } \
            or plan.get("evidence_status") != "transition_confirmation_only" \
            or plan.get("planner_usable") is not False \
            or any(len(plan.get(key, "")) != 64 for key in hashes) \
            or min(phase_rates.get("prefill_tps", 0),
                   phase_rates.get("decode_tps", 0),
                   plan.get("kv_capacity_tokens", 0)) <= 0 \
            or plan.get("recipes") != expected_recipes \
            or plan.get("cells") != expected_cells \
            or plan.get("run_order") != expected_order:
        raise ValueError("service-admission transition plan changed")


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def read_rates(path: Path, plan: dict) -> dict:
    rates = headroom.read_rates(path, HARDWARE, plan["source_plan_sha256"])
    if rates["sha256"] != plan["normalization_sha256"] \
            or rates["kv_capacity_tokens"] != plan["kv_capacity_tokens"] \
            or {"prefill_tps": rates["prefill_tps"],
                "decode_tps": rates["decode_tps"]} != plan["phase_rates"] \
            or rates["balanced_shape"] != plan["balanced_shape"]:
        raise RuntimeError("transition normalization differs from the frozen plan")
    return rates


def _stream(rates: dict, population: str, rho: float, start_s: float,
            duration_s: float) -> list[dict]:
    shape = headroom.shape_for(population, rates)
    work = sum(headroom.service_work(shape, rates))
    return [
        {"offset_s": start_s + offset, "population": population,
         "session_id": f"{population}-{index % headroom.SESSIONS_PER_SHAPE}",
         "request_index": index, "prefix_tokens": shape.prefix_tokens,
         "append_tokens": shape.append_tokens,
         "output_tokens": shape.output_tokens,
         "prefill_work_s": shape.append_tokens / rates["prefill_tps"],
         "decode_work_s": shape.output_tokens / rates["decode_tps"]}
        for index, offset in enumerate(
            headroom.uniform_offsets(rho / work, duration_s))
    ]


def offered_trace(plan: dict, rates: dict, direction: str) -> list[dict]:
    if direction not in DIRECTIONS:
        raise ValueError("unsupported transition mix")
    total_s = plan["warmup_s"] + plan["measurement_s"]
    rows = _stream(rates, "incumbent", plan["base_rho"], 0, total_s)
    rows.extend(_stream(
        rates, direction, plan["target_rho"] - plan["base_rho"],
        plan["warmup_s"], plan["measurement_s"],
    ))
    return sorted(rows, key=lambda row: (row["offset_s"], row["population"]))


def phase_coordinates(plan: dict, rates: dict, direction: str) -> dict[str, float]:
    phases = headroom.offered_phase_rho(
        plan, offered_trace(plan, rates, direction))
    return {**phases, "offered_rho": sum(phases.values())}


def window_plan(plan: dict, start_s: float, duration_s: float) -> dict:
    return {**plan, "warmup_s": start_s, "measurement_s": duration_s}


def cohort_summary(plan: dict, requests: list[dict], direction: str) -> dict:
    lo = plan["warmup_s"]
    hi = lo + plan["measurement_s"]
    selected = [row for row in requests if row["population"] == direction
                and lo <= row["offset_s"] < hi]
    completed = [row for row in selected if serving.service_completion(row)]
    exact = [row for row in completed if serving.exact_token_timing(row)]
    return {
        "offered": len(selected), "completed": len(completed),
        "completion_rate": len(completed) / len(selected) if selected else 0,
        "exact_timing_coverage": len(exact) / len(completed) if completed else 0,
        "p90_ttft_s": headroom.quantile(
            [row["ttft_s"] for row in completed], .9),
        "p90_mean_tpot_s": headroom.quantile(
            [row["mean_tpot_s"] for row in exact], .9),
    }


def trace_epoch_ns(trace: list[dict], requests: list[dict]) -> int:
    if len(trace) != len(requests) or not requests:
        raise RuntimeError("incomplete offered trace")
    epochs = [row["scheduled_ns"] - int(row["offset_s"] * 1e9)
              for row in requests]
    if max(epochs) - min(epochs) > 1:
        raise RuntimeError("request clocks do not share one offered-trace epoch")
    return round(statistics.median(epochs))


def validate_request_trace(trace: list[dict], requests: list[dict],
                           epoch_ns: int) -> None:
    observed = sorted(requests, key=lambda row: (
        row["offset_s"], row["population"]))
    for expected, actual in zip(trace, observed, strict=True):
        expected_fields = {
            "population": expected["population"],
            "offset_s": expected["offset_s"],
            "session_id": expected["session_id"],
            "request_index": expected["request_index"],
            "input_tokens": expected["append_tokens"],
            "prefix_tokens": expected["prefix_tokens"],
            "planned_output_tokens": expected["output_tokens"],
            "scheduled_ns": epoch_ns + int(expected["offset_s"] * 1e9),
        }
        if any(actual.get(key) != value
               for key, value in expected_fields.items()):
            raise RuntimeError("request trace differs from the frozen treatment")


def _admission_prewarm(host: str, port: int, model: str,
                       sessions: list[serving.Session], timeout_s: float,
                       target_ns: int, lead_s: float) -> tuple[list[dict], str, int]:
    time.sleep(max(0, (target_ns - int(lead_s * 1e9)
                       - time.monotonic_ns()) / 1e9))
    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures, launch_ns = headroom.submit_synchronized(
            executor, sessions,
            lambda session, _epoch: {
                **serving.prewarm(
                    host, port, model, [session], timeout_s, True,
                )[0],
                "session_id": session.session_id,
            },
            lead_s=lead_s,
        )
        rows, error = headroom.settle_futures(futures)
    return rows, (f"{type(error).__name__}: {error}" if error else ""), launch_ns


def _window_summaries(plan: dict, trace: list[dict], requests: list[dict],
                      metrics: list[dict], drained: bool) -> dict:
    return {
        "baseline": headroom.summarize(
            window_plan(plan, 0, plan["baseline_s"]),
            trace, requests, metrics, drained,
        ),
        "transition": headroom.summarize(
            window_plan(plan, plan["baseline_s"],
                        plan["transition_window_s"]),
            trace, requests, metrics, drained,
        ),
        "post_admission": headroom.summarize(
            plan, trace, requests, metrics, drained,
        ),
    }


def _invalid_reason(plan: dict, trace: list[dict], requests: list[dict],
                    summaries: dict, engine_failure: str | None) -> str | None:
    if engine_failure == "infrastructure":
        return "infrastructure engine failure"
    if len(requests) != len(trace) or not requests:
        return "incomplete offered trace"
    if max(row["send_lateness_s"] for row in requests) \
            > plan["max_send_lateness_s"]:
        return "offered trace schedule slip"
    if any(not row["telemetry_window_complete"]
           for row in summaries.values()) and engine_failure != "service":
        return "measurement telemetry is incomplete"
    return None


def run_cell(plan: dict, rates: dict, cell: dict, cfg: testbed.Config,
             root: Path, extra: list[str]) -> dict:
    identity = headroom.collect_runtime_identity(
        plan, cfg, HARDWARE, extra,
        headroom.cached_image_hash(
            cfg.sandbox, root.parent / ".a100-image-sha256.json",
        ) if testbed.runtime_mode() == "apptainer" else None,
    )
    headroom.validate_stage_inputs(plan, rates, identity)
    trace = offered_trace(plan, rates, cell["direction"])
    headroom.client_task_count(plan, trace)
    pool = headroom.sessions(cell["block"], rates)
    incumbents = [pool[f"incumbent-{index}"]
                  for index in range(headroom.SESSIONS_PER_SHAPE)]
    admitted = [pool[f"{cell['direction']}-{index}"]
                for index in range(headroom.SESSIONS_PER_SHAPE)]
    prepared = [{**row, "prepared": serving.prepare_issue(
        pool[row["session_id"]], row["request_index"], cfg.model, True,
    )} for row in trace]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "offered.json", trace)
    requests: list[dict] = []
    admission_rows: list[dict] = []
    acquisition_error: Exception | None = None
    admission_error = ""
    started_wall_ns = time.time_ns()
    epoch_ns = epoch_wall_ns = admission_scheduled_ns = 0
    admission_target_ns = admission_end_ns = 0
    drained = engine_exited = False
    engine_failure = None
    kv_capacity_tokens = None
    initial_prewarm_tokens = None
    try:
        with headroom.destination_stack(
                cfg, root / "stack", HARDWARE, extra, identity) as stack:
            kv_capacity_tokens = stack.kv_capacity_tokens
            if kv_capacity_tokens != plan["kv_capacity_tokens"]:
                raise RuntimeError("live KV capacity differs from the frozen plan")
            testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
            initial = serving.prewarm(
                cfg.host, stack.port, cfg.model, incumbents,
                plan["request_timeout_s"], True,
            )
            headroom.validate_prewarm(initial, incumbents)
            initial_prewarm_tokens = sum(row["prompt_tokens"] for row in initial)
            write_json(root / "incumbent-prewarm.json", initial)
            sampler = serving.MetricsSampler(
                cfg.host, stack.port, root / "engine.csv")
            power = profiler.PowerSampler(root / "power.csv")
            sampler.start()
            power.start()
            try:
                headroom.wait_sampler(sampler)
                lead_ns = int(plan["client"]["launch_lead_s"] * 1e9)
                epoch_ns = time.monotonic_ns() + lead_ns
                epoch_wall_ns = time.time_ns() + lead_ns
                with ThreadPoolExecutor(max_workers=1) as executor:
                    trace_future = executor.submit(
                        headroom.issue_async_trace,
                        cfg.host, stack.port, prepared, epoch_ns,
                        plan["request_timeout_s"],
                        plan["client"]["event_loop_shards"],
                    )
                    admission_target_ns = epoch_ns + int(
                        plan["admission"]["start_offset_s"] * 1e9)
                    admission_rows, admission_error, admission_scheduled_ns = \
                        _admission_prewarm(
                            cfg.host, stack.port, cfg.model, admitted,
                            plan["request_timeout_s"], admission_target_ns,
                            plan["admission"]["launch_lead_s"],
                        )
                    admission_end_ns = max(
                        (row["end_ns"] for row in admission_rows),
                        default=0,
                    )
                    requests, acquisition_error = trace_future.result()
                drained = headroom.drain(sampler, plan["drain_s"])
            except Exception as exc:
                acquisition_error = acquisition_error or exc
            finally:
                try:
                    engine_exited = headroom.close_samplers(
                        sampler, power, stack.engine)
                except Exception as exc:
                    acquisition_error = acquisition_error or exc
                engine_failure = headroom.engine_failure_kind(
                    stack.log, engine_exited)
    except Exception as exc:
        acquisition_error = acquisition_error or exc
    write_json(root / "admission-prewarm.json", admission_rows)
    write_json(root / "requests.json", requests)
    base = {
        "schema": SCHEMA, **cell, "plan_sha256": headroom.digest(plan),
        "runtime_identity": identity,
        "runtime_identity_sha256": headroom.identity_sha(identity),
        "normalization_sha256": rates["sha256"],
        "started_wall_ns": started_wall_ns,
        "epoch_monotonic_ns": epoch_ns, "epoch_wall_ns": epoch_wall_ns,
        "admission_target_ns": admission_target_ns,
        "admission_scheduled_ns": admission_scheduled_ns,
        "admission_end_ns": admission_end_ns,
        "admission_error": admission_error,
        "initial_prewarm_tokens": initial_prewarm_tokens,
        "admission_prewarm_tokens": sum(
            row.get("prompt_tokens", 0) for row in admission_rows),
        "kv_capacity_tokens": kv_capacity_tokens,
        "drained": drained, "engine_exited": engine_exited,
        "engine_failure_kind": engine_failure,
    }
    if acquisition_error:
        result = {**base, "status": "invalid",
                  "measurement_error":
                  f"{type(acquisition_error).__name__}: {acquisition_error}"}
        write_json(root / "result.json", result)
        raise RuntimeError("transition measurement is invalid") from acquisition_error
    summaries = _window_summaries(
        plan, trace, requests, sampler.rows, drained,
    )
    reason = _invalid_reason(
        plan, trace, requests, summaries, engine_failure,
    )
    result = {**base, "status": "invalid" if reason else "complete",
              "measurement_error": reason, "windows": summaries,
              "new_cohort": cohort_summary(
                  plan, requests, cell["direction"])}
    write_json(root / "result.json", result)
    if reason:
        raise RuntimeError(f"transition measurement is invalid: {reason}")
    return result


def _prewarm_contract(plan: dict, row: dict, prewarm: list[dict]) -> bool:
    shape = headroom.Shape(**(
        plan["balanced_shape"] if row["direction"] == "balanced"
        else plan["shapes"][row["direction"]]
    ))
    return len(prewarm) == plan["admission"]["new_sessions"] \
        and {item.get("session_id") for item in prewarm} == {
            f"{row['direction']}-{index}"
            for index in range(plan["admission"]["new_sessions"])
        } \
        and all(serving.service_completion(item)
                and item.get("prompt_tokens") == shape.prefix_tokens
                and item.get("cached_tokens") == 0 for item in prewarm)


def cell_decision(plan: dict, row: dict, prewarm: list[dict],
                  epoch_ns: int) -> dict:
    windows = row["windows"]
    cohort = row["new_cohort"]
    target_ns = epoch_ns + int(plan["admission"]["start_offset_s"] * 1e9)
    starts = [int(item["start_ns"]) for item in prewarm]
    ends = [int(item["end_ns"]) for item in prewarm]
    transition_s = ((max(ends) - target_ns) / 1e9 if ends else None)
    launch_lateness_s = (max(abs(start - target_ns) for start in starts) / 1e9
                         if starts else None)
    launch_spread_s = ((max(starts) - min(starts)) / 1e9 if starts else None)
    checks = {
        "baseline_incumbent_slo": headroom.row_feasible(
            windows["baseline"], plan["targets"]),
        "transition_incumbent_slo": headroom.row_feasible(
            windows["transition"], plan["targets"]),
        "post_admission_incumbent_slo": headroom.row_feasible(
            windows["post_admission"], plan["targets"]),
        "admission_materialized": not row["admission_error"]
        and _prewarm_contract(plan, row, prewarm),
        "admission_within_window": transition_s is not None
        and 0 <= transition_s <= plan["transition_window_s"],
        "admission_launch_on_time": launch_lateness_s is not None
        and launch_lateness_s <= plan["max_send_lateness_s"],
        "admission_launch_synchronized": launch_spread_s is not None
        and launch_spread_s <= plan["max_send_lateness_s"],
        "initial_incumbents_materialized": row["initial_prewarm_tokens"]
        == (plan["admission"]["initial_incumbent_sessions"]
            * plan["shapes"]["incumbent"]["prefix_tokens"]),
        "new_cohort_completed": cohort["offered"] > 0
        and cohort["completion_rate"] == 1
        and cohort["exact_timing_coverage"]
        >= plan["min_exact_timing_coverage"],
        "new_cohort_slo": cohort["p90_ttft_s"] is not None
        and cohort["p90_mean_tpot_s"] is not None
        and cohort["p90_ttft_s"] <= plan["targets"]["p90_ttft_s"]
        and cohort["p90_mean_tpot_s"]
        <= plan["targets"]["p90_mean_tpot_s"],
        "all_requests_cached": all(
            window["cache_mismatch_count"] == 0
            for window in windows.values()),
    }
    return {"cell_id": row["cell_id"], "direction": row["direction"],
            "block": row["block"], "transition_s": transition_s,
            "admission_launch_lateness_s": launch_lateness_s,
            "admission_launch_spread_s": launch_spread_s,
            "checks": checks, "pass": all(checks.values())}


def select_attempt(cell_root: Path, plan: dict, cell: dict,
                   attempt_root: Path) -> None:
    row = json.loads((attempt_root / "result.json").read_text())
    if row.get("status") != "complete" \
            or row.get("plan_sha256") != headroom.digest(plan) \
            or any(row.get(key) != value for key, value in cell.items()):
        raise RuntimeError("cannot select an incomplete transition attempt")
    write_json(cell_root / "selected.json", {
        "schema": SCHEMA, "plan_sha256": headroom.digest(plan),
        "cell_id": cell["cell_id"], "attempt": attempt_root.name,
        "result_sha256": headroom.digest(row),
    })


def selected_attempt(cell_root: Path, plan: dict, cell: dict) -> Path:
    selected_path = cell_root / "selected.json"
    if not selected_path.exists():
        raise RuntimeError(f"missing selected attempt: {cell['cell_id']}")
    selected = json.loads(selected_path.read_text())
    attempt_name = selected.get("attempt", "")
    if selected.get("schema") != SCHEMA \
            or selected.get("plan_sha256") != headroom.digest(plan) \
            or selected.get("cell_id") != cell["cell_id"] \
            or not re.fullmatch(r"attempt-\d{4}", attempt_name):
        raise RuntimeError(f"invalid selected attempt: {cell['cell_id']}")
    attempt = cell_root / attempt_name
    result_path = attempt / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"selected attempt has no result: {cell['cell_id']}")
    row = json.loads(result_path.read_text())
    if selected.get("result_sha256") != headroom.digest(row):
        raise RuntimeError(f"selected result changed: {cell['cell_id']}")
    return attempt


def reduce(plan: dict, rates: dict, root: Path) -> dict:
    rows, decisions = [], []
    for cell_id in plan["run_order"]:
        cell = next(row for row in plan["cells"] if row["cell_id"] == cell_id)
        attempt = selected_attempt(root / cell_id, plan, cell)
        path = attempt / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing transition result: {cell_id}")
        row = json.loads(path.read_text())
        if (row.get("status") != "complete"
                or row.get("schema") != SCHEMA
                or row.get("plan_sha256") != headroom.digest(plan)
                or row.get("normalization_sha256") != rates["sha256"]
                or row.get("kv_capacity_tokens") != plan["kv_capacity_tokens"]
                or row.get("runtime_identity_sha256")
                != headroom.identity_sha(row.get("runtime_identity", {}))
                or headroom.service_runtime_identity_sha(
                    row.get("runtime_identity", {}))
                != plan["service_runtime_identity_sha256"]
                or any(row.get(key) != value for key, value in cell.items())):
            raise RuntimeError(f"invalid transition result: {cell_id}")
        trace = json.loads((attempt / "offered.json").read_text())
        expected_trace = offered_trace(plan, rates, cell["direction"])
        if trace != expected_trace:
            raise RuntimeError(f"transition treatment changed: {cell_id}")
        requests = json.loads((attempt / "requests.json").read_text())
        epoch_ns = trace_epoch_ns(trace, requests)
        validate_request_trace(trace, requests, epoch_ns)
        metrics = headroom.read_engine_metrics(attempt / "engine.csv")
        row["windows"] = _window_summaries(
            plan, trace, requests, metrics, bool(row["drained"]),
        )
        reason = _invalid_reason(
            plan, trace, requests, row["windows"],
            row.get("engine_failure_kind"),
        )
        if reason:
            raise RuntimeError(f"selected transition evidence is invalid: {reason}")
        row["new_cohort"] = cohort_summary(
            plan, requests, cell["direction"])
        row["offered_coordinates"] = phase_coordinates(
            plan, rates, cell["direction"])
        if row["offered_coordinates"] != plan["recipes"][cell["direction"]]:
            raise RuntimeError(f"transition recipe changed: {cell_id}")
        prewarm = json.loads((attempt / "admission-prewarm.json").read_text())
        initial = json.loads((attempt / "incumbent-prewarm.json").read_text())
        pool = headroom.sessions(cell["block"], rates)
        incumbents = [pool[f"incumbent-{index}"]
                      for index in range(headroom.SESSIONS_PER_SHAPE)]
        try:
            headroom.validate_prewarm(initial, incumbents)
        except RuntimeError as exc:
            raise RuntimeError(
                f"initial transition state changed: {cell_id}") from exc
        raw_target_ns = epoch_ns + int(
            plan["admission"]["start_offset_s"] * 1e9)
        raw_end_ns = max((item["end_ns"] for item in prewarm), default=0)
        scheduled_ns = row.get("admission_scheduled_ns")
        if row.get("epoch_monotonic_ns") != epoch_ns \
                or row.get("admission_target_ns") != raw_target_ns \
                or row.get("admission_end_ns") != raw_end_ns \
                or not isinstance(scheduled_ns, int) \
                or abs(scheduled_ns - raw_target_ns) / 1e9 \
                > plan["max_send_lateness_s"] \
                or row.get("initial_prewarm_tokens") \
                != sum(item["prompt_tokens"] for item in initial) \
                or row.get("admission_prewarm_tokens") \
                != sum(item.get("prompt_tokens", 0) for item in prewarm):
            raise RuntimeError(f"transition timestamps changed: {cell_id}")
        decision = cell_decision(plan, row, prewarm, epoch_ns)
        rows.append(row)
        decisions.append(decision)
    observed_order = [row["cell_id"] for row in sorted(
        rows, key=lambda row: row["started_wall_ns"])]
    if observed_order != plan["run_order"]:
        raise RuntimeError("transition execution order differs from the frozen plan")
    if len({row["runtime_identity_sha256"] for row in rows}) != 1:
        raise RuntimeError("transition cells mix runtime identities")
    mix_checks = {
        direction: all(decision["pass"] for decision in decisions
                       if decision["direction"] == direction)
        for direction in DIRECTIONS
    }
    return {
        "schema": SCHEMA, "stage": "transition_confirmation",
        "plan_sha256": headroom.digest(plan),
        "normalization_sha256": rates["sha256"],
        "evidence_status": "three_discrete_transition_recipes",
        "planner_usable": False, "supported_envelope": None,
        "targets": plan["targets"], "mix_checks": mix_checks,
        "recipes": plan["recipes"],
        "campaign_pass": all(mix_checks.values()),
        "claim_scope": (
            "live materialization of eight sessions followed by 240 seconds "
            "of offered work at the plan's three exact normalized phase-work recipes"
        ),
        "decisions": decisions, "rows": rows,
    }


def attempt_roots(cell_root: Path) -> list[Path]:
    return sorted(path for path in cell_root.glob("attempt-*") if path.is_dir()
                  and re.fullmatch(r"attempt-\d{4}", path.name))


def _result_complete(path: Path, plan: dict, cell: dict) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return row.get("status") == "complete" \
        and row.get("plan_sha256") == headroom.digest(plan) \
        and row.get("normalization_sha256") == plan["normalization_sha256"] \
        and all(row.get(key) == value for key, value in cell.items())


def _cell_complete(cell_root: Path, plan: dict, cell: dict) -> bool:
    try:
        attempt = selected_attempt(cell_root, plan, cell)
        return _result_complete(attempt / "result.json", plan, cell)
    except (OSError, RuntimeError, json.JSONDecodeError):
        pass
    completed = [attempt for attempt in attempt_roots(cell_root)
                 if _result_complete(attempt / "result.json", plan, cell)]
    if len(completed) > 1:
        raise RuntimeError(f"multiple complete attempts: {cell['cell_id']}")
    if completed:
        select_attempt(cell_root, plan, cell, completed[0])
        return True
    return False


def run_all(plan: dict, rates: dict, cfg: testbed.Config, root: Path,
            summary_path: Path, extra: list[str], retry_delay_s: float,
            max_attempts: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    completed = 0
    for index, cell_id in enumerate(plan["run_order"]):
        cell, cell_root = by_id[cell_id], root / cell_id
        if _cell_complete(cell_root, plan, cell):
            completed += 1
            continue
        attempts = len(attempt_roots(cell_root))
        while True:
            attempts += 1
            attempt_root = cell_root / f"attempt-{attempts:04d}"
            if attempt_root.exists():
                raise RuntimeError(f"transition attempt already exists: {attempt_root}")
            write_json(status_path, {
                "state": "running", "cell_id": cell_id,
                "cell_index": index + 1, "cell_count": len(plan["run_order"]),
                "completed_cells": completed, "attempt": attempts,
                "updated_wall_ns": time.time_ns(),
                "plan_sha256": headroom.digest(plan),
            })
            try:
                run_cell(plan, rates, cell, cfg, attempt_root, extra)
                select_attempt(cell_root, plan, cell, attempt_root)
                break
            except RuntimeError:
                if max_attempts and attempts >= max_attempts:
                    write_json(status_path, {
                        "state": "failed", "cell_id": cell_id,
                        "completed_cells": completed, "attempt": attempts,
                        "updated_wall_ns": time.time_ns(),
                        "plan_sha256": headroom.digest(plan),
                    })
                    raise
                time.sleep(retry_delay_s)
        completed += 1
    result = reduce(plan, rates, root)
    write_json(summary_path, result)
    write_json(status_path, {
        "state": "complete", "completed_cells": completed,
        "cell_count": len(plan["run_order"]),
        "campaign_pass": result["campaign_pass"],
        "updated_wall_ns": time.time_ns(),
        "plan_sha256": headroom.digest(plan),
    })
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-plan", type=Path, required=True)
    prepare.add_argument("--normalization", type=Path, required=True)
    prepare.add_argument("--scout", type=Path, required=True)
    prepare.add_argument("--confirmation-plan", type=Path, required=True)
    prepare.add_argument("--confirmed", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("run-cell", "run"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--normalization", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        testbed.add_common(command)
        command.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
        if name == "run-cell":
            command.add_argument("--cell-id", required=True)
        else:
            command.add_argument("--summary", type=Path, required=True)
            command.add_argument("--retry-delay-s", type=float, default=30)
            command.add_argument("--max-attempts", type=int, default=3)
    final = commands.add_parser("reduce")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--normalization", type=Path, required=True)
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and (
            args.retry_delay_s < 0 or args.max_attempts < 0):
        parser.error("retry controls must be nonnegative")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        core = headroom.read_plan(args.source_plan)
        rates = headroom.read_rates(
            args.normalization, HARDWARE, headroom.digest(core))
        scout = json.loads(args.scout.read_text())
        confirmation_plan = headroom.read_plan(args.confirmation_plan)
        confirmed = json.loads(args.confirmed.read_text())
        write_json(args.out, make_plan(
            core, rates, scout, confirmation_plan, confirmed))
        return
    plan = read_plan(args.plan)
    rates = read_rates(args.normalization, plan)
    if args.command == "reduce":
        write_json(args.out, reduce(plan, rates, args.run_root))
        return
    cfg = testbed.config_from_args(args)
    extra = (args.extra_vllm_args[1:]
             if args.extra_vllm_args[:1] == ["--"]
             else args.extra_vllm_args)
    if args.command == "run-cell":
        cell = next((row for row in plan["cells"]
                     if row["cell_id"] == args.cell_id), None)
        if cell is None:
            raise ValueError("unknown transition cell")
        cell_root = args.run_root / cell["cell_id"]
        if _cell_complete(cell_root, plan, cell):
            return
        attempts = len(attempt_roots(cell_root)) + 1
        attempt_root = cell_root / f"attempt-{attempts:04d}"
        run_cell(plan, rates, cell, cfg, attempt_root, extra)
        select_attempt(cell_root, plan, cell, attempt_root)
        return
    run_all(plan, rates, cfg, args.run_root, args.summary, extra,
            args.retry_delay_s, args.max_attempts)


if __name__ == "__main__":
    main()
