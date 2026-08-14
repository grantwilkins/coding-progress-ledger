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


SLOS = {"normal": (2_000, 100), "emergency": (10_000, 250)}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"concurrency", "input_len", "input_tps", "output_tps",
                "ttft_p95_ms", "tpot_p95_ms"}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError(f"invalid service staircase {path}")
    for row in rows:
        for key in ("concurrency", "input_len"):
            row[key] = int(row[key])
        for key in required - {"concurrency", "input_len"}:
            row[key] = float(row[key])
    return rows


def staircase_summary(prefill: list[dict], decode: list[dict]) -> dict:
    prefill_rows = [{"input_tokens": row["input_len"],
                     "input_tps": row["input_tps"],
                     "p95_ttft_ms": row["ttft_p95_ms"]} for row in prefill]
    contexts = []
    for context in sorted({row["input_len"] for row in decode}):
        cells = [row for row in decode if row["input_len"] == context]
        peak = max(cells, key=lambda row: row["output_tps"])
        item = {"context_tokens": context, "peak_output_tps": peak["output_tps"],
                "peak_concurrency": peak["concurrency"],
                "peak_p95_ttft_ms": peak["ttft_p95_ms"],
                "peak_p95_tbt_ms": peak["tpot_p95_ms"]}
        for name, (ttft, tbt) in SLOS.items():
            for suffix, safe in (
                ("ttft", [row for row in cells if row["ttft_p95_ms"] <= ttft]),
                ("tbt", [row for row in cells if row["tpot_p95_ms"] <= tbt]),
                ("joint", [row for row in cells if row["ttft_p95_ms"] <= ttft
                            and row["tpot_p95_ms"] <= tbt]),
            ):
                best = max(safe, key=lambda row: row["output_tps"], default=None)
                item[f"{name}_{suffix}"] = None if best is None else {
                    "max_tested_concurrency": best["concurrency"],
                    "max_tested_output_tps": best["output_tps"],
                }
        contexts.append(item)
    return {"prefill": prefill_rows, "decode": contexts}


def _errors(predictions: list[dict], threshold: float) -> dict:
    actual = np.asarray([row["actual_p95_tbt_ms"] for row in predictions])
    predicted = np.asarray([row["predicted_p95_tbt_ms"] for row in predictions])
    return {"mae_ms": float(np.mean(np.abs(actual - predicted))),
            "mape": float(np.mean(np.abs(actual - predicted) / actual)),
            "false_feasible": int(np.sum((actual > threshold) & (predicted <= threshold))),
            "false_infeasible": int(np.sum((actual <= threshold) & (predicted > threshold)))}


def decode_holdout(rows: list[dict], profile: ModelProfile) -> dict:
    contexts = sorted({row["input_len"] for row in rows})
    if len(contexts) < 3:
        raise ValueError("decode holdout needs at least three contexts")
    predictions = {"work_only": [], "observed_iteration_proxy": []}
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
                    "actual_p95_tbt_ms": row["tpot_p95_ms"]}
            x = math.log(row["output_tps"] / curve.rate(heldout, 1))
            predictions["work_only"].append({
                **base, "predicted_p95_tbt_ms": float(math.exp(beta[0] + beta[1] * x))})
            predictions["observed_iteration_proxy"].append({
                **base, "predicted_p95_tbt_ms":
                scale * 1_000 * row["concurrency"] / row["output_tps"]})
    return {"protocol": "leave one complete context out",
            "limitation": "both features use achieved throughput and are diagnostic, not admission inputs",
            "models": {name: {"normal": _errors(values, SLOS["normal"][1]),
                              "emergency": _errors(values, SLOS["emergency"][1]),
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
    tbt = np.asarray([(row.end_s - row.prefill_end_s) / row.output_tokens
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
            "p95_modeled_tbt_ms": float(np.quantile(tbt, .95)),
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
        result.append({"context_tokens": row["input_len"], "modeled_tbt_ms": modeled,
                       "observed_p95_tbt_ms": row["tpot_p95_ms"],
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
    return {"schema": "queue-haul-service-holdout-v1",
            "inputs": {"prefill": str(prefill_path), "decode": str(decode_path),
                       "profile": str(profile_path), "h100_profile": str(h100_profile_path)},
            "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths},
            "diagnostic_slos": {
                name: {"p95_ttft_ms": values[0], "p95_tbt_ms": values[1]}
                for name, values in SLOS.items()
            } | {"note": "stricter diagnostic quantiles, not the legacy p90 policy"},
            "staircase": staircase_summary(prefill, decode),
            "decode_context_holdout": decode_holdout(decode, profile),
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
