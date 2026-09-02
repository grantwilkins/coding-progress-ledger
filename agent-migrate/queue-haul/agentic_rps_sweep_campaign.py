"""Collect legacy and replicated A100/H100 agentic TTFT/TPOT sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import service_headroom_campaign as headroom
import single_gpu_capacity_campaign as capacity


SCHEMA = "queue-haul-agentic-rps-sweep-v3"
TAIL_SCHEMA = "queue-haul-agentic-rps-tail-v2"
GPT_RETRY_SCHEMA = "queue-haul-gpt-oss-width128-retry-v1"
TAIL_ACQUISITION_SCHEMA = "queue-haul-agentic-rps-tail-v1"
TAIL_ACQUISITION_PLAN_SHA256 = \
    "511996215831bfecf1d950e318fcc1a3be09456c1303199178661b211290ec97"
PARENT_SCHEMA = "queue-haul-agentic-rps-sweep-v2"
PARENT_PLAN_SHA256 = "194ad7d6e376e903fb7ce3db7f40df925942f8cb21b91e2d6fb890a39825512d"
HISTORICAL_RESULT_IDENTITIES = {
    ("queue-haul-agentic-rps-sweep-v1",
     "4709014a6cbaa32104531be1c9e0482094a4f3ac6d155fb44d015f13473b67ed"),
    (PARENT_SCHEMA, PARENT_PLAN_SHA256),
    (TAIL_ACQUISITION_SCHEMA, TAIL_ACQUISITION_PLAN_SHA256),
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
TAIL_REQUESTS_PER_POINT = 128
TAIL_RATES_RPS_BY_MODEL = {
    "openai/gpt-oss-20b": (4.0, 6.0, 8.0),
    "Qwen/Qwen3.8-27B": (.7, 1.0, 2.0),
    "google/gemma-4-26B-A4B-it": (4.0, 6.0, 8.0),
}
FIXED_SLOS = {
    "openai/gpt-oss-20b": {"p90_ttft_s": 2.0,
                            "p90_tpot_s": .1},
    "google/gemma-4-26B-A4B-it": {"p90_ttft_s": 2.0,
                                   "p90_tpot_s": .2},
}
TAIL_SLOS = {
    **FIXED_SLOS,
    "Qwen/Qwen3.8-27B": {
        "p90_ttft_s": 6.964898975600001,
        "p90_tpot_s": .163794359,
    },
}
SLO_SCHEMA = "queue-haul-agentic-rps-sweep-v4"
SLO_MODEL = "openai/gpt-oss-20b"
SLO_TARGETS = {"p90_ttft_s": 1.0, "p90_tpot_s": .05}
REQUEST_TPOT_DEFINITION = \
    "p90_across_request_mean_post_first_token_latency"
SLO_HARDWARE = ("a100", "h100")
SLO_SCOUT_RATES_RPS = (
    .03125, .0625, .125, .25, .5, 1, 2, 4, 8, 10, 12, 16, 24, 32,
)
SLO_PRIMARY_BLOCKS, SLO_MAX_BLOCKS = 20, 30
SLO_WARMUP_RATE_RPS, SLO_MAX_SEND_LATENESS_S = 1, .05
SLO_MAX_METRIC_GAP_S = 1
SLO_METRICS_PERIOD_S = .25
SLO_MIN_EXACT_TPOT_INTERVAL_COVERAGE = .99
SLO_MAX_ATTEMPTS_PER_CELL = 2
SLO_DRAIN_TIMEOUT_S = 300
SLO_REFINEMENT_INTERVALS = 8
SLO_UPPER_SCOUT_GUARDS = 2
SLO_MAX_BOUNDARY_STEPS = 4


def digest(value) -> str:
    return profiler.object_hash(value)


def semantic_runtime_value(value):
    if isinstance(value, dict):
        return {key: semantic_runtime_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [semantic_runtime_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("CUDA_VISIBLE_DEVICES="):
            return "CUDA_VISIBLE_DEVICES=<allocated>"
        value = re.sub(r"(/tmp/qh-[^/\s\"']+)-\d+", r"\1-<pid>", value)
        return re.sub(
            r"(<[^<>]+ object at )0x[0-9a-fA-F]+(>)",
            r"\1<object-address>\2", value,
        )
    return value


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
        "client_shards": 8,
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


def slo_rate_order(seed: int, rates: tuple[float, ...],
                   block: int) -> tuple[float, ...]:
    plan = {"seed": seed}
    return tuple(sorted(
        rates,
        key=lambda rate: stable_seed(plan, rate, block, "rate-order"),
    ))


def _slo_plan(seed: int, hardware: str) -> dict:
    runtime = {
        **capacity.make_runtime_contract(False),
        "max_num_batched_tokens": testbed.model_spec(SLO_MODEL).batched_tokens,
        "block_size": 16,
        "mode": "native",
        "runtime_versions": list(testbed.NATIVE_RUNTIME_VERSIONS),
        "stream_interval": 1,
    }
    request_shape = {
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "source": "fixed compact shape derived from the OpenHands coding trace",
    }
    implementation = {"campaign_source_sha256": hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()}
    blocks = {
        "primary": SLO_PRIMARY_BLOCKS,
        "maximum": SLO_MAX_BLOCKS,
        "stopping": "reduce_at_20; extend_to_30_only_if_unresolved",
        "rate_order": "ascending_sha256(seed:rate:block:rate-order)",
    }
    warmup = {"rate_rps": SLO_WARMUP_RATE_RPS,
              "requests": REQUESTS_PER_POINT, "discard": True}
    collector = {
        "request_driver": "one_ready_blocking_thread_per_request",
        "dispatch_lead_s": 1,
        "metrics_period_s": SLO_METRICS_PERIOD_S,
    }
    validity = {
        "max_send_lateness_s": SLO_MAX_SEND_LATENESS_S,
        "max_metric_gap_s": SLO_MAX_METRIC_GAP_S,
        "required_completions": REQUESTS_PER_POINT,
        "minimum_exact_tpot_interval_coverage":
        SLO_MIN_EXACT_TPOT_INTERVAL_COVERAGE,
        "maximum_attempts_per_cell": SLO_MAX_ATTEMPTS_PER_CELL,
        "required_cached_tokens": 0,
        "require_telemetry": True,
        "require_drain": True,
    }
    statistics = {
        "cell": "p90",
        "rate": "median_across_blocks",
        "interval": "exact_binomial_order_statistic",
        "per_look_minimum_confidence": .975,
        "selected_interval_minimum_confidence": .95,
        "scope": "pointwise_rate_metric",
    }
    preflight = {
        "candidate_rates_rps": list(SLO_SCOUT_RATES_RPS),
        "requests_per_cell": REQUESTS_PER_POINT,
        "fresh_engine_per_cell": True,
        "required_consecutive_violations": SLO_UPPER_SCOUT_GUARDS + 1,
        "discard": True,
    }
    selection = {
        "refinement_intervals": SLO_REFINEMENT_INTERVALS,
        "lower_scout_guards": 1,
        "upper_scout_guards": SLO_UPPER_SCOUT_GUARDS,
        "maximum_clear_boundary_steps": SLO_MAX_BOUNDARY_STEPS,
    }
    semantics = {
        "open_loop_poisson": True,
        "max_concurrency": None,
        "fresh_engine_per_block": True,
        "randomized_complete_blocks": True,
        "service_failures_are_violations": True,
        "invalid_measurements_hard_fail": True,
        "tpot_definition": "p90_of_observable_exact_singleton_token_intervals",
        "finite_episode_claim": True,
        "boundary_rule": (
            "last_clear_pass_to_first_clear_fail; no_lower_fail; "
            "no_higher_pass; higher_clear_fail; width_within_tolerance"
        ),
    }
    comparison = {
        "model": SLO_MODEL,
        "revision": testbed.model_spec(SLO_MODEL).revision,
        "request_shape": request_shape,
        "requests_per_cell": REQUESTS_PER_POINT,
        "slo": SLO_TARGETS,
        "runtime": runtime,
        "blocks": blocks,
        "warmup": warmup,
        "collector": collector,
        "validity": validity,
        "statistics": statistics,
        "preflight": preflight,
        "selection": selection,
        "semantics": semantics,
        "implementation": implementation,
        "seed": seed,
        "statistical_unit": "fresh-engine-block-cell",
    }
    return {
        "schema": SLO_SCHEMA,
        "campaign": "agentic_rps_sweep",
        "study": "gpt_oss_slo_error_bars",
        "hardware": hardware,
        "models": [SLO_MODEL],
        "model_revisions": {SLO_MODEL: testbed.model_spec(SLO_MODEL).revision},
        "request_shape": request_shape,
        "requests_per_point": REQUESTS_PER_POINT,
        "blocks": blocks,
        "warmup": warmup,
        "collector": collector,
        "preflight": preflight,
        "selection": selection,
        "slo": {**SLO_TARGETS, "source": "fixed-paper-reference"},
        "runtime": runtime,
        "validity": validity,
        "statistics": statistics,
        "semantics": semantics,
        "comparison": comparison,
        "comparison_sha256": digest(comparison),
        "implementation": implementation,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "seed": seed,
    }


def make_slo_plan(seed: int = 20260901, hardware: str = "h100") -> dict:
    plan = _slo_plan(seed, hardware)
    validate_slo_plan(plan)
    return plan


def validate_slo_plan(plan: dict) -> None:
    seed, hardware = plan.get("seed"), plan.get("hardware")
    if not isinstance(seed, int) or hardware not in SLO_HARDWARE \
            or plan != _slo_plan(seed, hardware):
        raise ValueError("invalid agentic SLO error-bar plan")


def _tail_plan(seed: int) -> dict:
    rates = sorted({rate for values in TAIL_RATES_RPS_BY_MODEL.values()
                    for rate in values})
    return {
        "schema": TAIL_SCHEMA,
        "campaign": "agentic_rps_sustained_tail",
        "hardware": "a100",
        "models": list(MODELS),
        "model_revisions": {
            model: testbed.model_spec(model).revision for model in MODELS
        },
        "request_shape": {
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "source": "fixed compact shape derived from the OpenHands coding trace",
        },
        "rates_rps": rates,
        "rates_rps_by_model": {
            model: list(TAIL_RATES_RPS_BY_MODEL[model]) for model in MODELS
        },
        "parent": {
            "schema": TAIL_ACQUISITION_SCHEMA,
            "plan_sha256": TAIL_ACQUISITION_PLAN_SHA256,
            "relationship": (
                "same raw cells; retain every client-observed token interval "
                "when streamed token IDs are coalesced"
            ),
        },
        "implementation": {
            "campaign_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
        "requests_per_point": TAIL_REQUESTS_PER_POINT,
        "client_shards": TAIL_REQUESTS_PER_POINT,
        "boundary_repeats": [],
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "slo": {"fixed": TAIL_SLOS, "relative_models": []},
        "runtime": capacity.make_runtime_contract(),
        "semantics": {
            "open_loop_poisson": True,
            "max_concurrency": None,
            "run_all_rates_after_violation": True,
            "slo_is_control_flow": False,
            "service_failures_are_outcomes": True,
            "unique_private_prompts": True,
            "forced_exact_output_length": True,
            "one_engine_per_model_unless_service_restart_is_needed": True,
            "tpot_definition": (
                "p90_of_all_client_observed_post_first_token_intervals"
            ),
            "coalesced_tokens_share_arrival_timestamp": True,
            "sustained_tail_extension": True,
        },
        "seed": seed,
    }


def make_plan(seed: int = 1, hardware: str = "a100") -> dict:
    plan = _plan(seed, hardware)
    validate_plan(plan)
    return plan


def make_tail_plan(seed: int = 1) -> dict:
    plan = _tail_plan(seed)
    validate_plan(plan)
    return plan


def _gpt_retry_plan(seed: int = 1) -> dict:
    """Dense GPT tail used to check whether TPOT is curve-comparable."""
    plan = _tail_plan(seed)
    plan["schema"] = GPT_RETRY_SCHEMA
    plan["campaign"] = "gpt_oss_width128_tpot_retry"
    plan["rates_rps"] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    plan["rates_rps_by_model"] = {
        model: ([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                if model == "openai/gpt-oss-20b" else [])
        for model in MODELS
    }
    plan["parent"] = {
        "schema": TAIL_SCHEMA,
        "relationship": "width-128 GPT TPOT comparability retry",
    }
    plan["semantics"]["dense_retry"] = True
    return plan


def make_gpt_retry_plan(seed: int = 1) -> dict:
    plan = _gpt_retry_plan(seed)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    seed = plan.get("seed")
    hardware = plan.get("hardware")
    if not isinstance(seed, int) or hardware not in {"a100", "h100"}:
        raise ValueError("invalid agentic RPS sweep plan")
    if plan.get("schema") == TAIL_SCHEMA:
        expected = _tail_plan(seed)
    elif plan.get("schema") == GPT_RETRY_SCHEMA:
        expected = _gpt_retry_plan(seed)
    else:
        expected = _plan(seed, hardware)
    if plan != expected:
        raise ValueError("invalid agentic RPS sweep plan")


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    (validate_slo_plan if plan.get("schema") == SLO_SCHEMA
     else validate_plan)(plan)
    return plan


def stable_seed(plan: dict, rate: float, repeat: int, purpose: str) -> int:
    payload = f"{plan['seed']}:{rate:.8g}:{repeat}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def arrival_offsets(plan: dict, model: str, rate: float, repeat: int,
                    purpose: str = "",
                    allowed_rates: tuple[float, ...] | None = None
                    ) -> tuple[float, ...]:
    rates = (allowed_rates if plan.get("schema") == SLO_SCHEMA else
             tuple(plan["rates_rps_by_model"].get(model, ())))
    if model not in MODELS \
            or rates is None or rate not in rates \
            or repeat < 0:
        raise ValueError("unsupported RPS cell")
    return serving.poisson_schedule(
        rate, plan["requests_per_point"],
        stable_seed(plan, rate, repeat,
                    f"{purpose}-arrivals" if purpose else "arrivals"),
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


def prepared_trace(plan: dict, model: str, rate: float, repeat: int,
                   purpose: str = "",
                   allowed_rates: tuple[float, ...] | None = None) -> list[dict]:
    rows = []
    for index, offset in enumerate(arrival_offsets(
            plan, model, rate, repeat, purpose, allowed_rates)):
        session = serving.Session(
            session_id=(f"agentic-{purpose}-{slug(model)}-{rate:g}-{repeat}-{index}"
                        if purpose else
                        f"agentic-{slug(model)}-{rate:g}-{repeat}-{index}"),
            prefix_tokens=1,
            append_tokens=plan["request_shape"]["prompt_tokens"] - 1,
            output_tokens=plan["request_shape"]["output_tokens"],
            vocabulary=1024,
            seed=stable_seed(
                plan, rate, repeat,
                f"{purpose}-prompt-{index}" if purpose else f"prompt-{index}",
            ),
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
    formal = plan.get("schema") == SLO_SCHEMA
    git_sha, dirty = profiler.git_state(not formal)
    mode, versions = testbed.runtime_mode(), testbed.runtime_versions(cfg)
    if formal and (mode != plan["runtime"]["mode"]
                   or list(versions) != plan["runtime"]["runtime_versions"]):
        raise RuntimeError("runtime does not match the SLO plan")
    identity = {
        "plan_sha256": digest(plan),
        "git_sha": git_sha,
        "git_dirty": dirty,
        "model": cfg.model,
        "revision": testbed.model_spec(cfg.model).revision,
        "hardware": capacity.gpu_snapshot(plan["hardware"].upper()),
        "runtime_mode": mode,
        "runtime_versions": versions,
        "ambient_vllm_env": {key: value for key, value in sorted(os.environ.items())
                              if key.startswith("VLLM_")},
        "scheduler": plan["runtime"],
        "commands": commands,
    }
    shared = {key: identity[key] for key in (
        "git_sha", "model", "revision", "runtime_mode", "runtime_versions",
        "ambient_vllm_env", "scheduler",
    )}
    shared["commands"] = semantic_runtime_value(commands)
    identity["shared_fingerprint_sha256"] = digest(shared)
    identity["launch_fingerprint_sha256"] = digest({
        "shared": identity["shared_fingerprint_sha256"],
        "hardware": identity["hardware"],
    })
    identity["sha256"] = digest(identity)
    return identity


def finalize_runtime_identity(identity: dict, server_info: dict) -> dict:
    config = server_info.get("vllm_config")
    if not isinstance(config, dict):
        raise RuntimeError("vLLM did not expose its effective server config")
    semantic = semantic_runtime_value(config)
    semantic.pop("instance_id", None)
    result = {**identity, "server_config_sha256": digest(semantic)}
    result["fingerprint_sha256"] = digest({
        "launch": result["launch_fingerprint_sha256"],
        "server_config": result["server_config_sha256"],
    })
    result["sha256"] = digest({key: value for key, value in result.items()
                               if key != "sha256"})
    return result


def wait_for_drain(sampler: serving.MetricsSampler, engine,
                   timeout_s: float | None = None,
                   not_before_ns: int = 0) -> bool:
    """Wait for natural drain without turning a wall-time budget into a gate."""
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    while engine.poll() is None and not sampler.error \
            and (deadline is None or time.monotonic() < deadline):
        if sampler.rows and sampler.rows[-1]["monotonic_ns"] >= not_before_ns \
                and not any(
                sampler.rows[-1].get(key, 0) for key in
                ("vllm:num_requests_running", "vllm:num_requests_waiting")):
            return True
        time.sleep(.1)
    return False


def observed_token_intervals(row: dict) -> list[float]:
    """Return client-observed per-token gaps, including zero-gap bursts."""
    timestamps = [
        int(event["monotonic_ns"])
        for event in row.get("token_events", [])
        for _ in event.get("token_ids", [])
    ]
    return [
        (right - left) / 1e9
        for left, right in zip(timestamps, timestamps[1:])
    ]


def issue_threaded_trace(host: str, port: int, model: str,
                         trace: list[dict], timeout_s: float,
                         lead_s: float) -> tuple[list[dict], Exception | None]:
    def issue(item: dict, epoch_ns: int) -> dict:
        scheduled_ns = epoch_ns + int(item["offset_s"] * 1e9)
        time.sleep(max(0, (scheduled_ns - time.monotonic_ns()) / 1e9))
        row = serving.issue_prepared(
            host, port, model, item["prepared"], scheduled_ns, timeout_s, True,
        )
        return {**row, "population": item["population"],
                "offset_s": item["offset_s"]}

    with ThreadPoolExecutor(max_workers=len(trace)) as executor:
        futures, _ = headroom.submit_synchronized(
            executor, trace, issue, lead_s,
        )
        rows, error = headroom.settle_futures(futures)
    return sorted(rows, key=lambda row: row["scheduled_ns"]), error


def measure_trace(plan: dict, cell: dict, cfg: testbed.Config, stack,
                  root: Path, purpose: str = "",
                  drain_timeout_s: float | None = None,
                  allowed_rates: tuple[float, ...] | None = None) -> tuple:
    repeat = cell.get("block", cell.get("repeat"))
    trace = prepared_trace(
        plan, cfg.model, cell["offered_rps"], repeat, purpose, allowed_rates,
    )
    root.mkdir(parents=True, exist_ok=True)
    collector = plan.get("collector")
    sampler = serving.MetricsSampler(
        cfg.host, stack.port, root / "engine.csv",
        period_s=collector["metrics_period_s"] if collector else .1,
    )
    sampler.start()
    headroom.wait_sampler(sampler)
    if collector:
        requests, client_error = issue_threaded_trace(
            cfg.host, stack.port, cfg.model, trace,
            plan["request_timeout_s"], collector["dispatch_lead_s"],
        )
    else:
        requests, client_error = headroom.issue_async_trace(
            cfg.host, stack.port, trace, time.monotonic_ns() + 1_000_000_000,
            plan["request_timeout_s"],
            shards=min(plan.get("client_shards", 8), len(trace)),
        )
    drained = wait_for_drain(
        sampler, stack.engine, drain_timeout_s,
        max((int(row.get("end_ns", 0)) for row in requests), default=0),
    )
    sampler_error = sampler.error
    try:
        sampler.close()
    except RuntimeError as exc:
        sampler_error = sampler_error or exc
    return (requests, sampler.rows, drained, client_error,
            stack.engine.poll() is not None, sampler_error)


def summarize_cell(plan: dict, cell: dict, requests: list[dict],
                   metrics: list[dict], drained: bool,
                   client_error: Exception | None, engine_exited: bool,
                   sampler_error: Exception | None) -> dict:
    completed = [row for row in requests if serving.service_completion(row)]
    exact = [row for row in completed if serving.exact_token_timing(row)]
    ttft = [float(row["ttft_s"]) for row in completed
            if row.get("ttft_s") is not None]
    observed = plan["semantics"]["tpot_definition"] \
        == "p90_of_all_client_observed_post_first_token_intervals"
    metric_rows = completed if observed else exact
    intervals = observed_token_intervals if observed \
        else lambda row: row.get("token_itls_s", [])
    tpot = [float(value) for row in metric_rows for value in intervals(row)]
    request_mean_tpot = (
        [float(statistics.mean(values)) for row in metric_rows
         if (values := intervals(row))]
        if observed else
        [float(row["mean_tpot_s"]) for row in exact
         if row.get("mean_tpot_s") is not None]
    )
    scheduled = sorted(int(row["scheduled_ns"]) for row in requests)
    starts = sorted(int(row["start_ns"]) for row in requests)
    peak_running = max((row.get("vllm:num_requests_running", 0)
                        for row in metrics), default=None)
    peak_waiting = max((row.get("vllm:num_requests_waiting", 0)
                        for row in metrics), default=None)
    return {
        "schema": plan["schema"],
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


def observable_exact_token_itls(row: dict) -> list[float]:
    events = row.get("token_events")
    if events is None:
        return row.get("token_itls_s", []) \
            if serving.exact_token_timing(row) else []
    return [
        (int(right["monotonic_ns"]) - int(left["monotonic_ns"])) / 1e9
        for left, right in zip(events, events[1:])
        if len(left.get("token_ids", ())) == len(right.get("token_ids", ())) == 1
    ]


def request_tpot_s(row: dict) -> float | None:
    first, last, tokens = (row.get(key) for key in (
        "first_ns", "last_token_ns", "output_tokens"))
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (first, last, tokens)) \
            or tokens <= 1 or last < first:
        return None
    return (last - first) / 1e9 / (tokens - 1)


def run_cell(plan: dict, cell: dict, cfg: testbed.Config, stack,
             root: Path) -> dict:
    cell_root = root / "cells" / cell["cell_id"]
    requests, metrics, drained, client_error, engine_exited, sampler_error = \
        measure_trace(plan, cell, cfg, stack, cell_root)
    result = summarize_cell(
        plan, cell, requests, metrics, drained, client_error,
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
    current = result.get("schema") == plan["schema"] \
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
            if plan["hardware"] == "h100":
                testbed.validate_h100_optimized_runtime(
                    testbed.shell(commands["vllm"]),
                    testbed.read_text(stack.log),
                )
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
    fixed = plan["slo"]["fixed"]
    if model in fixed:
        return {**fixed[model], "source": "fixed"}
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


def boundary_specs(plan: dict, model: str,
                   rates: tuple[float, ...]) -> list[dict]:
    return [cell_spec(model, rate, repeat) for rate in rates
            for repeat in plan["boundary_repeats"]]


def load_specs(plan: dict, root: Path, specs: list[dict]) -> list[dict]:
    return [read_result(plan, cell, result_path(root, cell)) for cell in specs]


def run_model(plan: dict, model: str, root: Path) -> None:
    discovery_cells = discovery_specs(plan, model)
    run_specs(plan, model, discovery_cells, root)
    discovery = load_specs(plan, root, discovery_cells)
    slo = derive_slo(plan, model, discovery)
    rates = boundary_rates(discovery, slo) if plan["boundary_repeats"] else ()
    write_json(root / "models" / slug(model) / "selection.json", {
        "model": model, "slo": slo, "repeated_boundary_rates": list(rates),
    })
    run_specs(plan, model, boundary_specs(plan, model, rates), root)


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
    rates = boundary_rates(discovery, slo) if plan["boundary_repeats"] else ()
    extra_cells = boundary_specs(plan, model, rates)
    rows = discovery + load_specs(plan, root, extra_cells)
    aggregated = [aggregate_rate(rows, rate)
                  for rate in model_rates(plan, model)]
    minimum_repeats = 3 if plan["boundary_repeats"] else 1
    confirmed = next((
        row["offered_rps"] for row in aggregated
        if row["repeats"] >= minimum_repeats and metric_violation({
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
        "violation_confirmation": (
            "three_repeats" if plan["boundary_repeats"]
            else "single_predeclared_sustained_point"
        ),
        "curve": aggregated,
    }


def reduce(plan: dict, root: Path, models: tuple[str, ...] = MODELS) -> dict:
    rows, model_results = [], {}
    for model in models:
        model_rows, result = reduce_model(plan, model, root)
        rows.extend(model_rows)
        model_results[model] = result
    return {
        "schema": plan["schema"],
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


def slo_cell_spec(rate: float, block: int, stage: str = "formal") -> dict:
    rate_text = f"{rate:g}".replace(".", "p")
    prefix = "" if stage == "formal" else f"{stage}-"
    return {
        "cell_id": f"{prefix}{slug(SLO_MODEL)}-rps{rate_text}-b{block:02d}",
        "model": SLO_MODEL,
        "revision": testbed.model_spec(SLO_MODEL).revision,
        "offered_rps": rate,
        "block": block,
        "stage": stage,
    }


def slo_result_path(root: Path, cell: dict) -> Path:
    return root / "cells" / cell["cell_id"] / "result.json"


def valid_slo_outcome(plan: dict, result: dict) -> bool:
    if result.get("status") == "service_failure":
        return result.get("slo_violation") is True and all(
            result.get(field) is None
            for field in ("p90_ttft_s", "p90_tpot_s"))
    values = [result.get(field) for field in ("p90_ttft_s", "p90_tpot_s")]
    return all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in values) \
        and result.get("slo_violation") == metric_violation(result, plan["slo"])


def read_slo_result(plan: dict, cell: dict, path: Path,
                    selection_sha256: str | None = None) -> dict:
    result = json.loads(path.read_text())
    if result.get("schema") != SLO_SCHEMA \
            or result.get("plan_sha256") != digest(plan) \
            or any(result.get(key) != value for key, value in cell.items()) \
            or result.get("status") not in {"numeric", "service_failure"} \
            or not isinstance(result.get("slo_violation"), bool) \
            or not isinstance(result.get("evidence_path"), str) \
            or not isinstance(result.get("stack_path"), str) \
            or not valid_slo_outcome(plan, result) \
            or (cell["stage"] == "formal" and (
                not isinstance(selection_sha256, str)
                or result.get("selection_sha256") != selection_sha256)) \
            or any(not isinstance(result.get(key), str) for key in (
                "runtime_fingerprint_sha256", "shared_runtime_sha256",
                "launch_git_sha",
            )):
        raise RuntimeError(f"stale or invalid SLO result: {cell['cell_id']}")
    return result


def summarize_slo_cell(plan: dict, cell: dict, requests: list[dict],
                       metrics: list[dict], drained: bool,
                       client_error: Exception | None, engine_exited: bool,
                       sampler_error: Exception | None,
                       engine_failure_kind: str | None,
                       runtime: dict) -> dict:
    result = summarize_cell(
        plan, cell, requests, metrics, drained, client_error, engine_exited,
        sampler_error,
    )
    completed = [row for row in requests if serving.service_completion(row)]
    expected = plan["requests_per_point"]
    exact_itls = [value for row in completed
                  for value in observable_exact_token_itls(row)]
    expected_itls = expected * (plan["request_shape"]["output_tokens"] - 1)
    request_tpots = [value for row in completed
                     if (value := request_tpot_s(row)) is not None]
    request_metric = plan["semantics"].get("tpot_definition") == \
        REQUEST_TPOT_DEFINITION
    tpot = request_tpots if request_metric else exact_itls
    expected_tpot = expected if request_metric else expected_itls
    result.update({
        "p90_tpot_s": float(np.quantile(tpot, .9)) if tpot else None,
        "tpot_samples": len(tpot),
        "expected_tpot_samples": expected_tpot,
        "exact_tpot_interval_coverage": len(exact_itls) / expected_itls,
    })
    if request_metric:
        result["diagnostic_p90_request_mean_tpot_s"] = result["p90_tpot_s"]
    timestamps = sorted(int(row["monotonic_ns"]) for row in metrics)
    gaps = [(right - left) / 1e9
            for left, right in zip(timestamps, timestamps[1:])]
    max_gap = plan["validity"]["max_metric_gap_s"]
    telemetry_complete = bool(len(timestamps) >= 2 and requests
                              and timestamps[0] <= min(
                                  int(row["scheduled_ns"]) for row in requests)
                              and timestamps[-1] >= max(
                                  int(row["end_ns"]) for row in requests)
                              and max(gaps) <= max_gap)
    errors = []
    if engine_failure_kind in {"infrastructure", "runtime_contract"}:
        errors.append(engine_failure_kind)
    if (sampler_error or not telemetry_complete) and not engine_exited:
        errors.append("telemetry")
    if client_error or (len(requests) != expected and not engine_exited):
        errors.append("client_trace")
    if result["max_send_lateness_s"] is None or \
            result["max_send_lateness_s"] > \
            plan["validity"]["max_send_lateness_s"]:
        errors.append("send_lateness")
    if any(row.get("cached_tokens", 0) != 0 for row in completed):
        errors.append("cache_hit")
    if any(row.get("prompt_tokens") != plan["request_shape"]["prompt_tokens"]
           or row.get("planned_prompt_tokens") !=
           plan["request_shape"]["prompt_tokens"] for row in completed):
        errors.append("prompt_tokens")
    service_failure = engine_exited or not drained or len(completed) != expected
    if not service_failure:
        if request_metric and result["tpot_samples"] != \
                plan["validity"]["required_request_tpot_samples"]:
            errors.append("request_tpot")
        elif not request_metric and result["exact_tpot_interval_coverage"] < \
                plan["validity"]["minimum_exact_tpot_interval_coverage"]:
            errors.append("exact_tpot_interval_coverage")
    status = "invalid" if errors else (
        "service_failure" if service_failure else "numeric")
    if status != "numeric":
        result["p90_ttft_s"] = result["p90_tpot_s"] = None
    span = result["scheduled_span_s"]
    result.update({
        "schema": SLO_SCHEMA,
        "status": status,
        "validity_errors": errors,
        "runtime_fingerprint_sha256": runtime["fingerprint_sha256"],
        "shared_runtime_sha256": runtime["shared_fingerprint_sha256"],
        "launch_git_sha": runtime["git_sha"],
        "engine_failure_kind": engine_failure_kind,
        "telemetry_complete": telemetry_complete,
        "max_metric_gap_s": max(gaps, default=None),
        "realized_rps": ((len(requests) - 1) / span
                         if len(requests) > 1 and span > 0 else None),
        "slo_violation": status == "service_failure" or (
            status == "numeric" and metric_violation(result, plan["slo"])),
    })
    return result


def measure_slo_cell(plan: dict, cell: dict, cfg: testbed.Config, stack,
                     root: Path, runtime: dict,
                     purpose: str, group: str = "cells",
                     allowed_rates: tuple[float, ...] | None = None,
                     selection_sha256: str | None = None) -> dict:
    cell_root = root / group / cell["cell_id"]
    attempt = len(list(cell_root.glob("attempt-*")))
    if attempt >= plan["validity"]["maximum_attempts_per_cell"]:
        raise RuntimeError(f"SLO measurement retries exhausted: {cell['cell_id']}")
    attempt_root = cell_root / f"attempt-{attempt:03d}"
    measured = measure_trace(
        plan, cell, cfg, stack, attempt_root, purpose, SLO_DRAIN_TIMEOUT_S,
        allowed_rates,
    )
    failure_kind = (capacity.failure_kind(testbed.read_text(stack.log))
                    if measured[4] else None)
    result = summarize_slo_cell(
        plan, cell, *measured, failure_kind, runtime,
    )
    result["evidence_path"] = str(attempt_root.relative_to(root))
    result["stack_path"] = str(stack.log.parent.relative_to(root))
    if selection_sha256 is not None:
        result["selection_sha256"] = selection_sha256
    write_json(attempt_root / "requests.json", measured[0])
    write_json(attempt_root / "result.json", result)
    if result["status"] == "invalid":
        raise RuntimeError(
            f"invalid SLO measurement {cell['cell_id']}: "
            f"{', '.join(result['validity_errors'])}")
    write_json(cell_root / "result.json", result)
    return result


def freeze_plan(root: Path, plan: dict) -> None:
    path = root / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError(f"run root contains a different plan: {path}")
    if not path.exists():
        write_json(path, plan)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_slo_design(plan: dict, rows: list[dict]) -> dict:
    candidates = plan["preflight"]["candidate_rates_rps"]
    rates = [row["offered_rps"] for row in rows]
    if rates != candidates[:len(rates)]:
        raise RuntimeError("SLO scout results are not an ascending plan prefix")
    first = next((index for index, row in enumerate(rows)
                  if row["slo_violation"]), None)
    guards = plan["selection"]["upper_scout_guards"]
    if first is None or first == 0 or len(rows) != first + guards + 1:
        raise RuntimeError("SLO scout did not bracket the optimized-runtime knee")
    if any(row["status"] != "numeric" or row["slo_violation"]
           for row in rows[:first]) or any(
            not row["slo_violation"] for row in rows[first:]):
        raise RuntimeError("SLO scout evidence is not monotone")
    low, high = rates[first - 1:first + 1]
    intervals = plan["selection"]["refinement_intervals"]
    step = (high - low) / intervals
    refined = [low + index * step for index in range(intervals + 1)]
    lower_guards = plan["selection"]["lower_scout_guards"]
    lower = rates[max(0, first - 1 - lower_guards):first - 1]
    upper = rates[first + 1:first + guards + 1]
    if len(upper) != guards:
        raise RuntimeError("SLO scout bracket lacks upper guard anchors")
    formal = tuple(sorted(set((*lower, *refined, *upper))))
    return {
        "bracket": {
            "observed_pass_rps": low,
            "observed_violation_rps": high,
            "higher_observed_violation_rates_rps": upper,
        },
        "formal_rates_rps": list(formal),
        "refinement_step_rps": step,
        "maximum_clear_boundary_width_rps": (
            step * plan["selection"]["maximum_clear_boundary_steps"]),
        "block_orders": [
            {"block": block,
             "rates_rps": list(slo_rate_order(plan["seed"], formal, block))}
            for block in range(plan["blocks"]["maximum"])
        ],
    }


def scout_evidence_sha256(root: Path, row: dict) -> str:
    evidence = [root / row[key] for key in ("evidence_path", "stack_path")]
    if any(not path.is_dir() for path in evidence):
        raise RuntimeError("missing SLO scout evidence")
    return digest({str(path.relative_to(root)): file_sha256(path)
                   for directory in evidence
                   for path in sorted(directory.rglob("*")) if path.is_file()})


def make_slo_preflight_record(plan: dict, root: Path,
                              rows: list[dict]) -> dict:
    design = selected_slo_design(plan, rows)
    fingerprints = {row["runtime_fingerprint_sha256"] for row in rows}
    shared = {row["shared_runtime_sha256"] for row in rows}
    commits = {row["launch_git_sha"] for row in rows}
    if any(len(values) != 1 for values in (fingerprints, shared, commits)):
        raise RuntimeError("SLO preflight mixed runtime provenance")
    scout = []
    block = plan["blocks"]["maximum"]
    for row in rows:
        path = slo_result_path(
            root, slo_cell_spec(row["offered_rps"], block, "preflight"))
        scout.append({
            "offered_rps": row["offered_rps"],
            "status": row["status"],
            "slo_violation": row["slo_violation"],
            "result_sha256": file_sha256(path),
            "evidence_sha256": scout_evidence_sha256(root, row),
        })
    result = {
        "schema": SLO_SCHEMA,
        "status": "complete",
        "plan_sha256": digest(plan),
        "runtime_fingerprint_sha256": next(iter(fingerprints)),
        "shared_runtime_sha256": next(iter(shared)),
        "launch_git_sha": next(iter(commits)),
        "observed_rates_rps": [row["offered_rps"] for row in rows],
        "scout": scout,
        **design,
    }
    result["selection_sha256"] = digest(result)
    return result


def read_slo_preflight_record(plan: dict, root: Path, path: Path) -> dict:
    result = json.loads(path.read_text())
    rates = result.get("observed_rates_rps")
    candidates = plan["preflight"]["candidate_rates_rps"]
    if not isinstance(rates, list) or rates != candidates[:len(rates)]:
        raise RuntimeError("stale SLO preflight")
    block = plan["blocks"]["maximum"]
    rows = [read_slo_result(
        plan, slo_cell_spec(rate, block, "preflight"),
        slo_result_path(root, slo_cell_spec(rate, block, "preflight")),
    ) for rate in rates]
    if result != make_slo_preflight_record(plan, root, rows):
        raise RuntimeError("stale SLO preflight")
    return result


def slo_vllm_args(runtime: dict) -> list[str]:
    args = ["--stream-interval", str(runtime["stream_interval"])]
    if backend := runtime.get("attention_backend"):
        args += ["--attention-backend", backend]
    if (enabled := runtime.get("async_scheduling")) is not None:
        args.append("--async-scheduling" if enabled else "--no-async-scheduling")
    if runtime.get("server_info_system_probe") is False:
        args += ["--middleware", "server_info_middleware.ConfigOnly"]
    return args


def validate_slo_runtime_log(runtime: dict, text: str) -> None:
    backend = runtime.get("attention_backend")
    if backend and not re.search(
            rf"Using (?:AttentionBackendEnum\.)?{re.escape(backend)} "
            r"(?:attention )?backend", text, re.IGNORECASE):
        raise RuntimeError("SLO runtime did not prove its attention backend")
    if (enabled := runtime.get("async_scheduling")) is not None \
            and f"Asynchronous scheduling is " \
            f"{'enabled' if enabled else 'disabled'}." not in text:
        raise RuntimeError("SLO runtime did not prove its scheduling mode")


def validate_slo_server_info(runtime: dict, server_info: dict) -> None:
    if runtime.get("server_info_system_probe") is False \
            and server_info.get("system_env") != {}:
        raise RuntimeError("SLO server-info system probe was not disabled")


def run_slo_block(plan: dict, root: Path, block: int,
                  rates: tuple[float, ...], stage: str,
                  expected_fingerprint: str | None = None,
                  selection_sha256: str | None = None) -> str:
    if stage == "formal" and not isinstance(selection_sha256, str):
        raise ValueError("formal SLO cells require a frozen selection")
    cells = [slo_cell_spec(rate, block, stage) for rate in rates]
    pending = [cell for cell in cells
               if not slo_result_path(root, cell).exists()]
    recorded = [read_slo_result(
        plan, cell, slo_result_path(root, cell), selection_sha256)
                for cell in cells if cell not in pending]
    seen = {row["runtime_fingerprint_sha256"] for row in recorded}
    if len(seen) > 1 or seen and expected_fingerprint \
            and seen != {expected_fingerprint}:
        raise RuntimeError("SLO cells mix runtime fingerprints")
    cfg = replace(model_config(SLO_MODEL, plan["hardware"]),
                  enforce_eager=False)
    allowed_rates = tuple(set((*rates, plan["warmup"]["rate_rps"])))
    fingerprint = expected_fingerprint or next(iter(seen), None)
    launch_root = root / "stacks" / f"block-{block:03d}"
    while pending:
        launch = len(list(launch_root.glob("launch-*")))
        stack_root = launch_root / f"launch-{launch:03d}"
        commands = capacity.stack_commands(cfg, slo_vllm_args(plan["runtime"]))
        identity = runtime_identity(plan, cfg, commands)
        with capacity.engine_stack(cfg, stack_root, identity, commands) as stack:
            log = testbed.read_text(stack.log)
            testbed.validate_optimized_runtime(
                testbed.shell(commands["vllm"]), log,
            )
            validate_slo_runtime_log(plan["runtime"], log)
            validate_slo_server_info(plan["runtime"], stack.server_info)
            identity = finalize_runtime_identity(identity, stack.server_info)
            fingerprint = fingerprint or identity["fingerprint_sha256"]
            if identity["fingerprint_sha256"] != fingerprint:
                raise RuntimeError("SLO runtime changed between fresh-engine blocks")
            write_json(stack_root / "runtime-identity.json", identity)
            testbed.reset_vllm_caches(cfg, (stack.log,), ports=(stack.port,))
            warmup = slo_cell_spec(plan["warmup"]["rate_rps"], block, "warmup")
            warmup["cell_id"] += f"-l{launch:03d}"
            measured = measure_slo_cell(
                plan, warmup, cfg, stack, root, identity,
                f"warmup-{stage}", "warmups", allowed_rates,
            )
            if measured["status"] != "numeric":
                raise RuntimeError("discarded warmup did not complete numerically")
            for cell in list(pending):
                testbed.reset_vllm_caches(
                    cfg, (stack.log,), ports=(stack.port,))
                measured = measure_slo_cell(
                    plan, cell, cfg, stack, root, identity, stage, "cells",
                    allowed_rates, selection_sha256,
                )
                pending.remove(cell)
                if measured["engine_exited"] or not measured["drained"]:
                    break
    if fingerprint is None:
        raise RuntimeError("SLO block has no runtime fingerprint")
    return fingerprint


def preflight_slo(plan: dict, root: Path) -> dict:
    freeze_plan(root, plan)
    complete = root / "preflight" / "complete.json"
    preflight_root = root / "preflight"
    if complete.exists():
        return read_slo_preflight_record(plan, preflight_root, complete)
    block = plan["blocks"]["maximum"]
    rows, fingerprint, first = [], None, None
    guards = plan["selection"]["upper_scout_guards"]
    for rate in plan["preflight"]["candidate_rates_rps"]:
        fingerprint = run_slo_block(
            plan, preflight_root, block, (rate,), "preflight", fingerprint)
        cell = slo_cell_spec(rate, block, "preflight")
        row = read_slo_result(
            plan, cell, slo_result_path(preflight_root, cell))
        rows.append(row)
        if row["slo_violation"] and first is None:
            first = len(rows) - 1
            if first == 0:
                raise RuntimeError("lowest SLO scout rate did not pass")
        elif first is not None and not row["slo_violation"]:
            raise RuntimeError("SLO scout evidence is not monotone")
        if first is not None and len(rows) == first + guards + 1:
            break
    result = make_slo_preflight_record(plan, preflight_root, rows)
    write_json(complete, result)
    return result


def run_slo_campaign(plan: dict, root: Path, blocks: int) -> None:
    if blocks not in {plan["blocks"]["primary"], plan["blocks"]["maximum"]}:
        raise ValueError("SLO run must use the primary or maximum block count")
    if blocks == plan["blocks"]["maximum"] and reduce_slo(
            plan, root, plan["blocks"]["primary"]
    )["models"][SLO_MODEL]["decision"] != "extend_to_30":
        raise RuntimeError("30-block extension requires an unresolved 20-block look")
    freeze_plan(root, plan)
    preflight = preflight_slo(plan, root)
    for block in range(blocks):
        order = tuple(preflight["block_orders"][block]["rates_rps"])
        run_slo_block(
            plan, root, block, order, "formal",
            preflight["runtime_fingerprint_sha256"],
            preflight["selection_sha256"],
        )
        write_json(root / "progress.json", {
            "schema": SLO_SCHEMA, "plan_sha256": digest(plan),
            "selection_sha256": preflight["selection_sha256"],
            "completed_blocks": block + 1, "target_blocks": blocks,
        })


def order_statistic_interval(values: list[float | None],
                             minimum_confidence: float = .975) -> dict:
    count = len(values)
    candidates = [
        (rank, 1 - 2 * sum(math.comb(count, index)
                          for index in range(rank)) / 2 ** count)
        for rank in range(1, count // 2 + 1)
    ]
    rank, confidence = max(
        pair for pair in candidates if pair[1] >= minimum_confidence)
    ordered = sorted(math.inf if value is None else float(value)
                     for value in values)
    median = statistics.median(ordered)
    lower, upper = ordered[rank - 1], ordered[count - rank]
    finite = lambda value: value if math.isfinite(value) else None
    return {"median": finite(median), "lower": finite(lower),
            "upper": finite(upper), "lower_censored": not math.isfinite(lower),
            "upper_censored": not math.isfinite(upper),
            "confidence": confidence, "rank": rank}


def aggregate_slo_rate(rows: list[dict], rate: float, blocks: int,
                       slo: dict, minimum_confidence: float) -> dict:
    matches = sorted(
        (row for row in rows if row["offered_rps"] == rate),
        key=lambda row: row["block"],
    )
    if len(matches) != blocks or [row["block"] for row in matches] \
            != list(range(blocks)):
        raise RuntimeError(f"incomplete block evidence at {rate:g} RPS")
    realized = [row["realized_rps"] for row in matches
                if row["realized_rps"] is not None]
    aggregate = {
        "offered_rps": rate,
        "blocks": blocks,
        "numeric_cells": sum(row["status"] == "numeric" for row in matches),
        "service_failure_cells": sum(
            row["status"] == "service_failure" for row in matches),
        "realized_rps_median": statistics.median(realized) if realized else rate,
        "points": [{"block": row["block"],
                    "realized_rps": row["realized_rps"],
                    "status": row["status"],
                    "p90_ttft_s": row["p90_ttft_s"],
                    "p90_tpot_s": row["p90_tpot_s"]} for row in matches],
    }
    intervals = {}
    for field in ("p90_ttft_s", "p90_tpot_s"):
        interval = order_statistic_interval([
            row[field] if row["status"] == "numeric" else None
            for row in matches
        ], minimum_confidence)
        intervals[field] = interval
        aggregate.update({
            f"{field}_median": interval["median"],
            f"{field}_ci_low": interval["lower"],
            f"{field}_ci_high": interval["upper"],
            f"{field}_ci_confidence": interval["confidence"],
            f"{field}_ci_rank": interval["rank"],
        })
    clear_pass = all(
        intervals[field]["upper"] is not None
        and intervals[field]["upper"] < slo[field]
        for field in intervals
    )
    clear_fail = any(
        interval["lower_censored"]
        or interval["lower"] is not None and interval["lower"] > slo[field]
        for field, interval in intervals.items()
    )
    aggregate["classification"] = (
        "clear_pass" if clear_pass else "clear_fail" if clear_fail
        else "indeterminate")
    return aggregate


def consistent_slo_boundary(curve: list[dict]) -> tuple | None:
    passes = [index for index, row in enumerate(curve)
              if row["classification"] == "clear_pass"]
    failures = [index for index, row in enumerate(curve)
                if row["classification"] == "clear_fail"]
    if not passes or not failures or max(passes) >= min(failures):
        return None
    left, right = max(passes), min(failures)
    return (curve[left]["offered_rps"], curve[right]["offered_rps"], right)


def reduce_slo(plan: dict, root: Path, blocks: int) -> dict:
    if blocks not in {plan["blocks"]["primary"], plan["blocks"]["maximum"]}:
        raise ValueError("SLO reduction must use 20 or 30 blocks")
    preflight = preflight_slo(plan, root)
    rates = preflight["formal_rates_rps"]
    selection = preflight["selection_sha256"]
    cells = [slo_cell_spec(rate, block)
             for block in range(blocks)
             for rate in rates]
    rows = [read_slo_result(
        plan, cell, slo_result_path(root, cell), selection)
            for cell in cells]
    fingerprints = {row["runtime_fingerprint_sha256"] for row in rows}
    shared = {row["shared_runtime_sha256"] for row in rows}
    commits = {row["launch_git_sha"] for row in rows}
    if fingerprints != {preflight["runtime_fingerprint_sha256"]} \
            or shared != {preflight["shared_runtime_sha256"]} \
            or commits != {preflight["launch_git_sha"]}:
        raise RuntimeError("formal cells do not share the preflight runtime")
    curve = [aggregate_slo_rate(
        rows, rate, blocks, plan["slo"],
        plan["statistics"]["per_look_minimum_confidence"],
    )
             for rate in rates]
    boundary = consistent_slo_boundary(curve)
    width = boundary[1] - boundary[0] if boundary else None
    within_tolerance = bool(
        boundary and width <= preflight["maximum_clear_boundary_width_rps"])
    higher_failure = bool(boundary and any(
        row["classification"] == "clear_fail"
        for row in curve[boundary[2] + 1:]
    ))
    confirmed = bool(within_tolerance and higher_failure)
    decision = ("complete" if confirmed else
                "extend_to_30" if blocks == plan["blocks"]["primary"] else
                "unresolved_at_30")
    model = {
        "model": SLO_MODEL,
        "revision": plan["model_revisions"][SLO_MODEL],
        "slo": plan["slo"],
        "curve": curve,
        "last_clear_pass_rps": boundary[0] if boundary else None,
        "first_clear_violation_rps": boundary[1] if boundary else None,
        "clear_boundary_confirmed": bool(boundary),
        "clear_boundary_width_rps": width,
        "maximum_clear_boundary_width_rps": preflight[
            "maximum_clear_boundary_width_rps"],
        "boundary_within_tolerance": within_tolerance,
        "higher_clear_violation_confirmed": higher_failure,
        "decision": decision,
    }
    return {
        "schema": SLO_SCHEMA,
        "stage": "reduced",
        "plan_sha256": digest(plan),
        "comparison_sha256": plan["comparison_sha256"],
        "selection_sha256": selection,
        "formal_rates_rps": rates,
        "hardware": plan["hardware"],
        "request_shape": plan["request_shape"],
        "blocks": blocks,
        "runtime_fingerprint_sha256": next(iter(fingerprints)),
        "shared_runtime_sha256": next(iter(shared)),
        "launch_git_sha": next(iter(commits)),
        "finite_episode_claim": True,
        "rows": rows,
        "models": {SLO_MODEL: model},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--hardware", choices=("a100", "h100"), default="a100")
    prepare.add_argument("--error-bars", action="store_true")
    prepare.add_argument("--tail", action="store_true")
    prepare.add_argument("--gpt-retry", action="store_true")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--run-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--model", choices=MODELS, action="append")
    run.add_argument("--blocks", type=int, choices=(SLO_PRIMARY_BLOCKS,
                                                     SLO_MAX_BLOCKS),
                     default=SLO_PRIMARY_BLOCKS)
    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("--plan", type=Path, required=True)
    reduce_parser.add_argument("--run-root", type=Path, required=True)
    reduce_parser.add_argument("--out", type=Path, required=True)
    reduce_parser.add_argument("--csv", type=Path)
    reduce_parser.add_argument("--model", choices=MODELS, action="append")
    reduce_parser.add_argument("--blocks", type=int,
                               choices=(SLO_PRIMARY_BLOCKS, SLO_MAX_BLOCKS),
                               default=SLO_PRIMARY_BLOCKS)
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
        seed = args.seed if args.seed is not None else (
            20260901 if args.error_bars else 1)
        if args.gpt_retry:
            plan = make_gpt_retry_plan(seed)
        elif args.tail:
            plan = make_tail_plan(seed)
        elif args.error_bars:
            plan = make_slo_plan(seed, args.hardware)
        else:
            plan = make_plan(seed, args.hardware)
        write_json(args.out, plan)
        return
    plan = read_plan(args.plan)
    if plan["schema"] == SLO_SCHEMA:
        if args.command == "preflight":
            preflight_slo(plan, args.run_root)
            return
        if args.command == "run":
            if args.model and args.model != [SLO_MODEL]:
                raise ValueError("the SLO error-bar plan is GPT-OSS-only")
            run_slo_campaign(plan, args.run_root, args.blocks)
            return
        if args.command == "reduce":
            summary = reduce_slo(plan, args.run_root, args.blocks)
            write_json(args.out, summary)
            write_csv(args.csv or args.out.with_suffix(".csv"), summary["rows"])
            return
        raise ValueError("v4 SLO plans cannot re-reduce legacy cells")
    if args.command == "preflight":
        raise ValueError("preflight requires a v4 SLO error-bar plan")
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
