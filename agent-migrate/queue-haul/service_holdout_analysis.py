"""Falsification analysis for mean service work versus decode hold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from profiles import ModelProfile
from simulate import (ExecutionScenario, PowerNode, ServingInstance, SimRequest,
                      SimSession, execute)


DIAGNOSTIC_THRESHOLDS = {"normal": (2_000, 100), "emergency": (10_000, 250)}


def _planned_gaps(row: dict) -> int:
    return row["n_requests"] * (row["output_len"] - 1)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"concurrency", "input_len", "output_len", "n_requests",
                "input_tps", "output_tps", "ttft_p95_ms", "tpot_p95_ms"}
    integers = {"concurrency", "input_len", "output_len", "n_requests"}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError(f"invalid service staircase {path}")
    for row in rows:
        for key in integers:
            row[key] = int(row[key])
        for key in required - integers:
            row[key] = float(row[key])
    return rows


def staircase_summary(prefill: list[dict], decode: list[dict]) -> dict:
    prefill_rows = [{"input_tokens": row["input_len"],
                     "n_requests": row["n_requests"],
                     "input_tps": row["input_tps"],
                     "p95_ttft_ms": row["ttft_p95_ms"]} for row in prefill]
    contexts = []
    for context in sorted({row["input_len"] for row in decode}):
        cells = [row for row in decode if row["input_len"] == context]
        peak = max(cells, key=lambda row: row["output_tps"])
        item = {"context_tokens": context, "peak_output_tps": peak["output_tps"],
                "peak_concurrency": peak["concurrency"],
                "peak_p95_ttft_ms": peak["ttft_p95_ms"],
                "peak_pooled_token_p95_itl_ms": peak["tpot_p95_ms"],
                "peak_n_requests": peak["n_requests"],
                "peak_planned_token_gaps_if_complete": _planned_gaps(peak)}
        for name, (ttft, itl) in DIAGNOSTIC_THRESHOLDS.items():
            for suffix, safe in (
                ("ttft", [row for row in cells if row["ttft_p95_ms"] <= ttft]),
                ("pooled_itl", [row for row in cells if row["tpot_p95_ms"] <= itl]),
                ("joint", [row for row in cells if row["ttft_p95_ms"] <= ttft
                            and row["tpot_p95_ms"] <= itl]),
            ):
                best = max(safe, key=lambda row: row["output_tps"], default=None)
                item[f"{name}_{suffix}"] = None if best is None else {
                    "safe_cell_concurrency_at_max_output_tps": best["concurrency"],
                    "safe_cell_output_tps": best["output_tps"],
                    "n_requests": best["n_requests"],
                    "planned_token_gaps_if_complete": _planned_gaps(best),
                }
        contexts.append(item)
    return {"prefill": prefill_rows, "decode": contexts}


def _errors(predictions: list[dict], threshold: float) -> dict:
    actual = np.asarray([row["actual_pooled_token_p95_itl_ms"] for row in predictions])
    predicted = np.asarray([row["predicted_pooled_token_p95_itl_ms"] for row in predictions])
    return {"mae_ms": float(np.mean(np.abs(actual - predicted))),
            "mape": float(np.mean(np.abs(actual - predicted) / actual)),
            "false_feasible": int(np.sum((actual > threshold) & (predicted <= threshold))),
            "false_infeasible": int(np.sum((actual <= threshold) & (predicted > threshold)))}


def decode_retrospective(rows: list[dict], profile: ModelProfile) -> dict:
    contexts = sorted({row["input_len"] for row in rows})
    if len(contexts) < 3:
        raise ValueError("decode retrospective needs at least three contexts")
    predictions = {"profile_work_leaky": [], "concurrency_throughput_proxy": []}
    curve = profile.case().decode
    for heldout in contexts:
        train = [row for row in rows if row["input_len"] != heldout]
        test = [row for row in rows if row["input_len"] == heldout]
        work_x = np.log([row["output_tps"] / curve.rate(row["input_len"], 1)
                         for row in train])
        beta = np.linalg.lstsq(np.column_stack((np.ones(len(train)), work_x)),
                               np.log([row["tpot_p95_ms"] for row in train]),
                               rcond=None)[0]
        proxy_x = np.asarray([1_000 * row["concurrency"] / row["output_tps"]
                              for row in train])
        scale = float(np.exp(np.mean(np.log(
            np.asarray([row["tpot_p95_ms"] for row in train]) / proxy_x))))
        for row in test:
            base = {"heldout_context_tokens": heldout,
                    "train_context_tokens": [value for value in contexts if value != heldout],
                    "concurrency": row["concurrency"],
                    "actual_pooled_token_p95_itl_ms": row["tpot_p95_ms"]}
            x = math.log(row["output_tps"] / curve.rate(heldout, 1))
            predictions["profile_work_leaky"].append({
                **base, "predicted_pooled_token_p95_itl_ms":
                float(math.exp(beta[0] + beta[1] * x))})
            predictions["concurrency_throughput_proxy"].append({
                **base, "predicted_pooled_token_p95_itl_ms":
                scale * 1_000 * row["concurrency"] / row["output_tps"]})
    return {"protocol": "leave one context bundle out",
            "inference_unit": "27 correlated staircase cells from three physical bundles",
            "limitation": "both features use achieved throughput; profile normalization also includes heldout context maxima",
            "models": {name: {"normal": _errors(values, DIAGNOSTIC_THRESHOLDS["normal"][1]),
                              "emergency": _errors(values, DIAGNOSTIC_THRESHOLDS["emergency"][1]),
                              "predictions": values}
                       for name, values in predictions.items()}}


def _decode_overlap(requests) -> int:
    points = sorted({value for row in requests for value in (row.prefill_end_s, row.end_s)})
    return max(sum(row.prefill_end_s <= point < row.end_s for row in requests)
               for point in points)


def _shape(profile: ModelProfile, name: str, arrivals, outputs,
           horizon_s: float = 30) -> dict:
    prompt = 256
    sessions = tuple(SimSession(
        f"s{i}", "instance", 4_096, prompt / horizon_s, output / horizon_s, 1,
        (SimRequest(float(arrival), prompt, int(output)),),
    ) for i, (arrival, output) in enumerate(zip(arrivals, outputs)))
    scenario = ExecutionScenario(
        25, horizon_s, 1e9, "awake", 0, (PowerNode("node", 1),),
        (ServingInstance("instance", ("node",)),), sessions, (),
    )
    result = execute(scenario, profile, ())
    ttft = np.asarray([row.prefill_end_s - row.arrival_s for row in result.requests]) * 1_000
    decode_ms = np.asarray([(row.end_s - row.prefill_end_s) / row.output_tokens
                            for row in result.requests]) * 1_000
    case = profile.case()
    prefill_tokens = sum(row.prompt_tokens for row in result.requests)
    decode_tokens = sum(row.output_tokens for row in result.requests)
    p, d = prefill_tokens / horizon_s / case.F, decode_tokens / horizon_s / case.G
    return {"name": name, "requests": len(result.requests),
            "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens,
            "p": p, "d": d, "ell": p + d,
            "p50_ttft_ms": float(np.quantile(ttft, .5)),
            "p95_ttft_ms": float(np.quantile(ttft, .95)),
            "p95_modeled_decode_ms_per_output_token": float(np.quantile(decode_ms, .95)),
            "peak_active_decode": _decode_overlap(result.requests),
            "source_power_at_deadline_w": result.modeled_source_power_at_deadline_w}


def simulation_ab(profile: ModelProfile) -> dict:
    count = 16
    uniform = [512] * count
    concentrated = [1_856] * 4 + [64] * 12
    rows = [_shape(profile, "smooth_uniform", range(count), uniform),
            _shape(profile, "burst_uniform", [0] * count, uniform),
            _shape(profile, "burst_long_first", [0] * count, concentrated),
            _shape(profile, "burst_long_last", [0] * count, concentrated[::-1])]
    work = {(round(row["p"], 12), round(row["d"], 12)) for row in rows}
    power = {round(row["source_power_at_deadline_w"], 12) for row in rows}
    if len(work) != 1 or len(power) != 1:
        raise RuntimeError("matched simulation changed aggregate work or power")
    return {"rows": rows, "matched_p_d": True, "matched_modeled_power": True,
            "boundary": "whole requests are FCFS-serialized; active decode cannot exceed one"}


def rate_mismatch(rows: list[dict], profile: ModelProfile) -> list[dict]:
    result = []
    for row in rows:
        if row["concurrency"] != 1:
            continue
        modeled = 1_000 / profile.case().decode.rate(row["input_len"], 1)
        result.append({"context_tokens": row["input_len"],
                       "profile_decode_ms_per_output_token": modeled,
                       "observed_pooled_token_p95_itl_ms": row["tpot_p95_ms"],
                       "observed_to_modeled_ratio": row["tpot_p95_ms"] / modeled})
    return result


def analyze(prefill_path: Path, decode_path: Path, profile_path: Path,
            h100_profile_path: Path) -> dict:
    prefill, decode = read_rows(prefill_path), read_rows(decode_path)
    profile = ModelProfile.load(profile_path)
    a100_raw, h100_raw = map(lambda path: json.loads(path.read_text()),
                             (profile_path, h100_profile_path))
    reused = all(a100_raw["cases"][case][phase] == h100_raw["cases"][case][phase]
                 for case in a100_raw["cases"] for phase in ("prefill_tps", "decode_tps"))
    paths = (prefill_path, decode_path, profile_path, h100_profile_path)
    return {"schema": "queue-haul-service-holdout-v2",
            "inputs": {"prefill": str(prefill_path), "decode": str(decode_path),
                       "profile": str(profile_path), "h100_profile": str(h100_profile_path)},
            "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths},
            "diagnostic_thresholds": {
                name: {"p95_ttft_ms": values[0], "pooled_token_p95_itl_ms": values[1]}
                for name, values in DIAGNOSTIC_THRESHOLDS.items()
            } | {"note": "descriptive sample thresholds, not an SLO or confidence bound"},
            "legacy_column_semantics": {
                "tpot_p95_ms": "p95 over finite ITLs pooled across requests; finite gap count was not retained"
            },
            "staircase": staircase_summary(prefill, decode),
            "decode_context_retrospective": decode_retrospective(decode, profile),
            "request_simulation_ab": simulation_ab(profile),
            "profile_rate_semantics": {"a100_h100_context_tables_identical": reused,
                                       "single_request_mismatch": rate_mismatch(decode, profile)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", type=Path, default=Path(
        "outputs/stage1_gpt_oss_20b_a100_tp1_eager_quick2_prefill_rho.csv"))
    parser.add_argument("--decode", type=Path, default=Path(
        "outputs/stage1_gpt_oss_20b_a100_tp1_eager_quick2_decode_context.csv"))
    parser.add_argument("--profile", type=Path,
                        default=Path("profiles/gpt_oss_20b_a100_tp1.json"))
    parser.add_argument("--h100-profile", type=Path,
                        default=Path("profiles/gpt_oss_20b_h100_tp1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.prefill, args.decode, args.profile, args.h100_profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
