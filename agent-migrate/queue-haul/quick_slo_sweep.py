"""Measure a one-engine GPT-OSS SLO curve with request-block error bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import agentic_rps_sweep_campaign as campaign
import destination_runner as serving
import migration_profiler as profiler
import migration_testbed as testbed
import single_gpu_capacity_campaign as capacity


SCHEMA = "queue-haul-quick-slo-sweep-v2"
MODEL = campaign.SLO_MODEL
RATES = (.5, 1., 2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 20., 24.)
REQUESTS = 50
BOOTSTRAP_DRAWS = 10_000
BLOCK_LENGTHS = (5, 10)
DEFAULT_SEED = 20260902


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(seed: int = DEFAULT_SEED, hardware: str = "h100") -> dict:
    if hardware not in campaign.SLO_HARDWARE or not isinstance(seed, int):
        raise ValueError("invalid quick SLO inputs")
    runtime = {
        **capacity.make_runtime_contract(False),
        "max_num_batched_tokens": testbed.model_spec(MODEL).batched_tokens,
        "block_size": 16,
        "mode": "native",
        "runtime_versions": list(testbed.NATIVE_RUNTIME_VERSIONS),
        "stream_interval": 1,
        "attention_backend": "TRITON_ATTN",
        "async_scheduling": False,
    }
    common = {
        "campaign": "agentic_rps_sweep",
        "study": "quick_gpt_oss_slo_error_bars",
        "models": [MODEL],
        "model_revisions": {MODEL: testbed.model_spec(MODEL).revision},
        "request_shape": {"prompt_tokens": campaign.PROMPT_TOKENS,
                          "output_tokens": campaign.OUTPUT_TOKENS},
        "requests_per_point": REQUESTS,
        "rates_rps": list(RATES),
        "rates_rps_by_model": {MODEL: list(RATES)},
        "rate_order_rps": list(campaign.slo_rate_order(seed, RATES, 0)),
        "warmup": {"rate_rps": 1., "requests": REQUESTS, "discard": True},
        "slo": dict(campaign.SLO_TARGETS),
        "runtime": runtime,
        "collector": {"request_driver": "one_ready_blocking_thread_per_request",
                      "dispatch_lead_s": 1., "metrics_period_s": .25},
        "validity": {
            "max_send_lateness_s": .05,
            "max_metric_gap_s": 1.,
            "required_completions": REQUESTS,
            "minimum_exact_tpot_interval_coverage": .99,
            "maximum_attempts_per_cell": 2,
            "required_cached_tokens": 0,
            "require_telemetry": True,
            "require_drain": True,
        },
        "statistics": {
            "point": "empirical_p90",
            "interval": "circular_moving_request_block_bootstrap_envelope",
            "confidence": .95,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "request_block_lengths": list(BLOCK_LENGTHS),
            "scope": "pointwise_conditional_on_each_rate_episode",
            "classification": "point_estimate",
        },
        "semantics": {
            "open_loop_poisson": True,
            "max_concurrency": None,
            "engine_reuse": "one_warmed_launch_until_failure_or_resume",
            "reset_and_drain_between_rates": True,
            "run_all_predeclared_rates": True,
            "uncached_unique_prompts": True,
            "forced_exact_output_length": True,
            "service_failures_are_violations": True,
            "tpot_definition": "p90_of_observable_exact_singleton_token_intervals",
        },
        "implementation": {
            "quick_source_sha256": source_sha256(Path(__file__)),
            "collector_source_sha256": source_sha256(Path(campaign.__file__)),
        },
        "request_timeout_s": campaign.REQUEST_TIMEOUT_S,
        "seed": seed,
    }
    return {"schema": SCHEMA, "hardware": hardware, **common,
            "comparison_sha256": campaign.digest(common)}


def validate_plan(plan: dict) -> None:
    if plan != make_plan(plan.get("seed"), plan.get("hardware")):
        raise ValueError("invalid quick SLO plan")


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def moving_block_counts(size: int, block_length: int, draws: int,
                        rng: np.random.Generator) -> np.ndarray:
    if size <= 0 or draws <= 0 or block_length <= 0 or size % block_length:
        raise ValueError("bootstrap blocks must divide the request count")
    starts = rng.integers(0, size, size=(draws, size // block_length))
    indices = (starts[..., None] + np.arange(block_length)) % size
    counts = np.zeros((draws, size), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(draws), size), indices.reshape(-1)), 1)
    return counts


def weighted_p90(clusters: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    if not clusters or any(not len(cluster) for cluster in clusters) \
            or counts.shape[1] != len(clusters):
        raise ValueError("bootstrap requires one nonempty cluster per request")
    lengths = np.array([len(cluster) for cluster in clusters])
    values = np.concatenate(clusters)
    owners = np.repeat(np.arange(len(clusters)), lengths)
    order = np.argsort(values, kind="stable")
    values, owners = values[order], owners[order]
    output = np.empty(len(counts))
    for index, request_counts in enumerate(counts):
        weights = request_counts[owners]
        rank = .9 * (int(request_counts @ lengths) - 1)
        lower, upper = math.floor(rank), math.ceil(rank)
        cumulative = np.cumsum(weights)
        left = values[np.searchsorted(cumulative, lower + 1)]
        right = values[np.searchsorted(cumulative, upper + 1)]
        output[index] = left + (rank - lower) * (right - left)
    return output


def bootstrap_intervals(requests: list[dict], seed: int,
                        draws: int = BOOTSTRAP_DRAWS,
                        block_lengths: tuple[int, ...] = BLOCK_LENGTHS) -> dict:
    ordered = sorted((row for row in requests if serving.service_completion(row)),
                     key=lambda row: row["scheduled_ns"])
    clusters = {
        "p90_ttft_s": [np.array([float(row["ttft_s"])]) for row in ordered],
        "p90_tpot_s": [np.asarray(campaign.observable_exact_token_itls(row),
                                      dtype=float) for row in ordered],
    }
    if len(ordered) != REQUESTS:
        raise RuntimeError("quick SLO bootstrap requires every request")
    output = {metric: {
        "point": float(np.quantile(np.concatenate(values), .9)),
        "low": math.inf, "high": -math.inf, "by_block_length": {},
    } for metric, values in clusters.items()}
    rng = np.random.default_rng(seed)
    for block_length in block_lengths:
        counts = moving_block_counts(len(ordered), block_length, draws, rng)
        for metric, values in clusters.items():
            low, high = np.quantile(weighted_p90(values, counts), (.025, .975))
            output[metric]["low"] = min(output[metric]["low"], float(low))
            output[metric]["high"] = max(output[metric]["high"], float(high))
            output[metric]["by_block_length"][str(block_length)] = [
                float(low), float(high),
            ]
    for interval in output.values():
        interval["low"] = min(interval["low"], interval["point"])
        interval["high"] = max(interval["high"], interval["point"])
    return output


def reduce(plan: dict, root: Path) -> dict:
    results, curve = [], []
    for rate in plan["rates_rps"]:
        cell = campaign.slo_cell_spec(rate, 0, "quick")
        result = campaign.read_slo_result(
            plan, cell, campaign.slo_result_path(root, cell),
        )
        results.append(result)
        intervals = None
        if result["status"] == "numeric":
            requests = json.loads(
                (root / result["evidence_path"] / "requests.json").read_text())
            intervals = bootstrap_intervals(
                requests, campaign.stable_seed(plan, rate, 0, "quick-bootstrap"),
            )
            if any(not np.isclose(intervals[metric]["point"], result[metric])
                   for metric in intervals):
                raise RuntimeError("quick SLO P90 does not reproduce raw evidence")
        row = {
            "offered_rps": rate,
            "realized_rps_median": result.get("realized_rps"),
            "points": [{key: result.get(key) for key in (
                "block", "realized_rps", "status", "p90_ttft_s", "p90_tpot_s",
                "slo_violation",
            )}],
        }
        for metric in ("p90_ttft_s", "p90_tpot_s"):
            interval = intervals[metric] if intervals else None
            row.update({
                f"{metric}_median": result.get(metric),
                f"{metric}_ci_low": interval["low"] if interval else None,
                f"{metric}_ci_high": interval["high"] if interval else None,
            })
            for block_length in BLOCK_LENGTHS:
                bounds = interval["by_block_length"][str(block_length)] \
                    if interval else (None, None)
                row[f"{metric}_ci_block{block_length}_low"] = bounds[0]
                row[f"{metric}_ci_block{block_length}_high"] = bounds[1]
        curve.append(row)
    for key in ("runtime_fingerprint_sha256", "shared_runtime_sha256",
                "launch_git_sha"):
        if len({row[key] for row in results}) != 1:
            raise RuntimeError(f"quick SLO results mix {key}")
    violating = [row["offered_rps"] for row in results if row["slo_violation"]]
    summary = {
        "schema": SCHEMA,
        "stage": "reduced",
        "hardware": plan["hardware"],
        "plan_sha256": campaign.digest(plan),
        "comparison_sha256": plan["comparison_sha256"],
        "request_shape": plan["request_shape"],
        "statistics": plan["statistics"],
        "runtime_fingerprint_sha256": results[0]["runtime_fingerprint_sha256"],
        "shared_runtime_sha256": results[0]["shared_runtime_sha256"],
        "launch_git_sha": results[0]["launch_git_sha"],
        "rows": results,
        "models": {MODEL: {
            "slo": plan["slo"],
            "passing_rates_rps": [row["offered_rps"] for row in results
                                  if not row["slo_violation"]],
            "violating_rates_rps": violating,
            "first_observed_violation_rps": min(violating, default=None),
            "curve": curve,
        }},
    }
    campaign.write_json(root / "summary.json", summary)
    campaign.write_csv(root / "rps-sweep.csv", [
        {key: value for key, value in row.items() if key != "points"}
        for row in curve
    ])
    return summary


def run(plan: dict, root: Path) -> dict:
    profiler.git_state(False)
    cfg = replace(campaign.model_config(MODEL, plan["hardware"]),
                  enforce_eager=False)
    mode, versions = testbed.runtime_mode(), list(testbed.runtime_versions(cfg))
    if mode != plan["runtime"]["mode"] \
            or versions != plan["runtime"]["runtime_versions"]:
        raise RuntimeError("runtime does not match the quick SLO plan")
    campaign.freeze_plan(root, plan)
    campaign.run_slo_block(
        plan, root, 0, tuple(plan["rate_order_rps"]), "quick",
    )
    return reduce(plan, root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--hardware", choices=campaign.SLO_HARDWARE,
                         default="h100")
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        campaign.write_json(args.out, make_plan(args.seed, args.hardware))
        return
    run(read_plan(args.plan), args.run_root)


if __name__ == "__main__":
    main()
