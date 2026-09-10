"""Bounded, conditional GPU queue validation; never launches the fleet campaign."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pool_shed_resident_data import OUTPUT, digest, load_requests, write_csv
from pool_shed_resident_fit import UNLOADED, calibrate, prefill_seconds
from pool_shed_resident_queue import simulate


def comparison(observed, predicted, absolute, relative=.25):
    tolerance = max(absolute, relative * observed) if observed is not None else None
    return {"observed": observed, "predicted": predicted, "tolerance": tolerance,
            "pass": None if observed is None else predicted is not None and abs(predicted - observed) <= tolerance}


def window(rows, start, end):
    eligible = [r for r in rows if start <= r["scheduled_s"] < end]
    first = [r for r in eligible if r["first_s"] is not None and r["first_s"] <= end]
    done = [r for r in eligible if r["done"] and r["end_s"] <= end]
    tpot = [r["mean_tpot_s"] for r in done if r["mean_tpot_s"] is not None]
    quantile = lambda values: float(np.quantile(values, .9)) if values else None
    missing = [r for r in eligible if r["first_s"] is None or r["first_s"] > end]
    return {"arrivals": len(eligible), "first_tokens_observed": len(first), "completed": len(done),
            "unfinished": len(eligible) - len(done), "tpot_samples": len(tpot),
            "ttft_p90_s": quantile([r["first_s"] - r["scheduled_s"] for r in first]),
            "tpot_p90_s": quantile(tpot),
            "known_ttft_violations": sum(r["first_s"] - r["scheduled_s"] > 1 for r in first)
                + sum(end - r["scheduled_s"] > 1 for r in missing),
            "unobserved_first_tokens": len(missing),
            "all_prior_outstanding": sum(r["scheduled_s"] < end and (not r["done"] or r["end_s"] > end) for r in rows)}


def inputs(rows, unknown_cache):
    if unknown_cache not in ("miss", "hit"):
        raise ValueError("unknown cache sensitivity must be miss or hit")
    selected, requests = {}, []
    for row in rows:
        resident = row["cohort"] == "resident"
        if not resident and (row["serving_role"] != "destination" or row["client_dispatch_s"] is None):
            continue
        arrival = row["scheduled_s"] if resident else row["client_dispatch_s"]
        if arrival < 0:
            continue  # Prewarming finished before epoch; its observed cache state is an input.
        prompt = row["submitted_prompt_tokens"] or row["planned_prompt_tokens"]
        output = row["output_tokens"] if row["done"] else row["planned_output_tokens"]
        if prompt is None or output is None:
            raise ValueError(f"unknown request shape: {row['row_id']}")
        cached = row["effective_cached_tokens"]
        if cached is None:
            cached = 0 if unknown_cache == "miss" else prompt - 1
        history = (f"resident:{row['session']}" if resident else f"incoming:{row['session']}"
                   if row["cohort"] == "incoming" or row["phase"] in ("initial", "catch_up") else row["row_id"])
        selected[row["row_id"]] = row
        requests.append(dict(request_id=row["row_id"], gpu="destination", history=history,
                             arrival_s=arrival, prompt_tokens=prompt, cached_tokens=cached, output_tokens=output))
    requests.sort(key=lambda r: (r["arrival_s"], selected[r["request_id"]]["turn"] or 0))
    return requests, selected


def evaluate(rows, engine, coefficients, endpoint_before_fraction=0., unknown_cache="miss"):
    requests, selected = inputs(rows, unknown_cache)
    duration = rows[0]["duration_s"]
    # ponytail: no KV eviction model; measured preemptions require another execution model.
    preemptions = [r["num_preemptions_total"] for r in engine]
    if not preemptions or None in preemptions or max(preemptions) != min(preemptions):
        raise ValueError("queue pilot requires complete, zero-preemption episode evidence")
    result = simulate(requests, coefficients, duration, endpoint_before_fraction=endpoint_before_fraction)
    predicted = {r["request_id"]: r for r in result["requests"]}
    residents = [r for r in rows if r["cohort"] == "resident"]
    simulated = [{**predicted[r["row_id"]], "scheduled_s": r["scheduled_s"]} for r in residents]
    intervals = [(30, 90)] if rows[0]["arm"] == "resident" else [(0, 60), (60, 90), (90, 120), (120, 180), (180, 240), (240, 300), (60, 300)]
    windows = []
    for start, end in intervals:
        if end > duration:
            continue
        observed, forecast = window(residents, start, end), window(simulated, start, end)
        samples = [r for r in engine if start <= r["time_s"] < end]
        if not samples or any(r["num_requests_waiting"] is None for r in samples):
            raise ValueError("missing engine queue samples")
        waiting = [sum(r["eligible_s"] is not None and r["eligible_s"] <= s["time_s"]
                       and (r["admitted_s"] is None or r["admitted_s"] > s["time_s"]) for r in predicted.values()) for s in samples]
        metrics = {"ttft_p90_s": comparison(observed["ttft_p90_s"], forecast["ttft_p90_s"], .2),
                   "tpot_p90_s": comparison(observed["tpot_p90_s"], forecast["tpot_p90_s"], .005),
                   "engine_waiting_peak": comparison(max(r["num_requests_waiting"] for r in samples), max(waiting), 2, 0),
                   "resident_outstanding": comparison(observed["all_prior_outstanding"], forecast["all_prior_outstanding"], 1, 0),
                   "known_ttft_violations": comparison(observed["known_ttft_violations"], forecast["known_ttft_violations"], 1, 0),
                   "first_token_coverage": comparison(observed["first_tokens_observed"], forecast["first_tokens_observed"], 1, 0)}
        windows.append(dict(start_s=start, end_s=end, observed=observed, predicted=forecast, metrics=metrics))
    migration = [r for r in rows if r["phase"] == "initial" and r["serving_role"] == "destination"]
    initial = {}
    if migration:
        start = min(r["client_dispatch_s"] for r in migration)
        for field in ("first_s", "end_s"):
            observed = max(r[field] for r in migration) - start if all(r[field] is not None for r in migration) else None
            forecast = max(predicted[r["row_id"]][field] for r in migration) - start if all(predicted[r["row_id"]][field] is not None for r in migration) else None
            initial[field] = comparison(observed, forecast, 2.)
    burst = [r for r in residents if 60 <= r["scheduled_s"] < 90]
    observed = max(r["end_s"] for r in burst) - 60 if burst and all(r["completed_within_observation"] for r in burst) else None
    forecast = max(predicted[r["row_id"]]["end_s"] for r in burst) - 60 if burst and all(predicted[r["row_id"]]["done"] for r in burst) else None
    return {"episode": rows[0]["episode"], "workload": rows[0]["workload"], "arm": rows[0]["arm"], "seed": rows[0]["seed"],
            "split": "heldout" if rows[0]["seed"] == 7102 else "training_control" if rows[0]["arm"] == "control" else "diagnostic",
            "endpoint_before_fraction": endpoint_before_fraction, "unknown_cache": unknown_cache,
            "unknown_cache_inputs": sum(r["effective_cached_tokens"] is None for r in selected.values()),
            "observed_failed_requests": sum(r["status"] in ("failed", "dependency_failed") for r in selected.values()),
            "windows": windows, "initial_migration": initial, "burst_resident_drain_s": comparison(observed, forecast, 2.),
            "gpu": result["gpus"]["destination"], "predictions": list(predicted.values())}


def checks(episodes):
    rows = [{"episode": r["episode"], "start_s": w["start_s"], "end_s": w["end_s"], "metric": k, **v}
            for r in episodes for w in r["windows"] for k, v in w["metrics"].items() if v["pass"] is not None]
    rows += [{"episode": r["episode"], "metric": "initial_migration_" + k, **v} for r in episodes for k, v in r["initial_migration"].items()]
    rows += [{"episode": r["episode"], "metric": "burst_resident_drain_s", **r["burst_resident_drain_s"]}
             for r in episodes if r["arm"] != "resident"]
    return {"checks": len(rows), "failures": [r for r in rows if r["pass"] is not True],
            "gate_pass": bool(rows) and all(r["pass"] is True for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.data / "data-manifest.json").read_text())
    if (set(manifest["output_sha256"]) != {"requests.csv", "engine.csv", "migration-events.csv"}
            or any(digest(args.data / name) != sha for name, sha in manifest["output_sha256"].items())
            or digest(Path("pool_shed_resident_data.py")) != manifest["reducer_sha256"]):
        raise ValueError("compact trace artifacts or extractor changed; regenerate before validation")
    rows = load_requests(args.data / "requests.csv")
    expected = {r["episode"] for r in manifest["episodes"]}
    if (len(expected) != 20 or {r["episode"] for r in rows} != expected
            or len({r["episode"] for r in rows if r["seed"] == 7102}) != 6):
        raise ValueError("validation requires all20 episodes including six held-out episodes")
    with (args.data / "engine.csv").open() as stream:
        engine = [{k: v if k in ("episode", "serving_role") else float(v) if v else None for k, v in r.items()} for r in csv.DictReader(stream)]
    fit = calibrate(rows, request_sha256=digest(args.data / "requests.csv"))
    episodes, predictions = [], []
    for episode in sorted({r["episode"] for r in rows}):
        requests = [r for r in rows if r["episode"] == episode]
        samples = [r for r in engine if r["episode"] == episode and r["serving_role"] == "destination" and 0 <= r["time_s"] < requests[0]["duration_s"]]
        for before, unknown in ((0., "miss"), (1., "miss"), (0., "hit"), (1., "hit")):
            result = evaluate(requests, samples, fit["coefficients"], before, unknown)
            prediction = result.pop("predictions")
            if before == 0 and unknown == "miss":
                predictions.extend({"episode": episode, **r} for r in prediction)
            episodes.append(result)
    gates = [{"endpoint_before_fraction": before, "unknown_cache": unknown,
              **{split: checks([r for r in episodes if r["endpoint_before_fraction"] == before and r["unknown_cache"] == unknown
                               and (r["split"] == split if split == "heldout" else r["split"] != "training_control")])
                 for split in ("heldout", "stress_and_heldout")}}
             for before, unknown in ((0., "miss"), (1., "miss"), (0., "hit"), (1., "hit"))]
    lookup = {r["request_id"]: r for r in predictions}
    long_outputs = [{k: r[k] for k in ("episode", "row_id", "cohort", "prompt_tokens", "effective_cached_tokens", "output_tokens",
                                     "scheduled_s", "first_s", "end_s", "mean_tpot_s", "submillisecond_token_gaps", "token_gap_count")}
                    | {"predicted_end_s": lookup[r["row_id"]]["end_s"], "predicted_mean_tpot_s": lookup[r["row_id"]]["mean_tpot_s"]}
                    for r in rows if r["cohort"] == "resident" and r["output_tokens"] == 1233 and r["completed_within_observation"]]
    bursts = [{k: r[k] for k in ("episode", "row_id", "output_tokens", "mean_tpot_s", "submillisecond_token_gaps",
                                "token_gap_count", "median_token_gap_s", "max_token_gap_s")}
              for r in rows if r["episode"] == "episodes-coding_long-0-replay-7101" and r["cohort"] == "resident"
              and 60 <= r["scheduled_s"] < 90 and r["done"] and r["submillisecond_token_gaps"]]
    cold_errors = [prefill_seconds(json.loads(r["prompt_counts"])[0], r["request_prefill_kv_computed_tokens_sum"], fit["coefficients"])
                   - r["request_prefill_time_seconds_sum"] for r in json.loads(UNLOADED.read_text())["rows"]
                   if r["seed"] == 7102 and r["width"] == 1 and r["phase_valid"]]
    report = {"campaign_ready": False, "resident_latency_validated": False,
              "conditional_heldout_gate_pass": all(g["heldout"]["gate_pass"] for g in gates),
              "conditional_stress_gate_pass": all(g["stress_and_heldout"]["gate_pass"] for g in gates),
              "gates": gates, "fit": fit, "episodes": episodes,
              "singleton_prefill_heldout": {"requests": len(cold_errors), "rmse_s": float(np.sqrt(np.mean(np.square(cold_errors)))),
                                             "max_absolute_error_s": max(abs(e) for e in cold_errors)},
              "long_generation_evidence": long_outputs, "client_token_bursts": bursts,
              "remaining_work": ["Measure per-request server arrival, first scheduled, first generated and last generated timestamps alongside current client SSE events, plus iteration start/end, scheduled prefill tokens and active decode request IDs.",
                  "A bounded follow-up can test warm independent prefixes at8K/30K, decode concurrency1/8/16 and1536 fixed output tokens, two repeats; then retain one coding1233-token-tail episode and one width-eight long replay/resident burst as workload checks. Freeze runtime and separate fitting from checks. A broad policy campaign is unnecessary.",
                  "Validate loaded generation and variable token-delivery delay before integrating persistent weighted GPU groups, per-GPU KV capacity and causal source queues into the fleet executor and planner forecasts."],
              "input_sha256": {str(args.data / name): digest(args.data / name) for name in ("requests.csv", "engine.csv", "data-manifest.json")},
              "code_sha256": {name: digest(Path(name)) for name in ("pool_shed_resident_queue.py", "pool_shed_resident_fit.py", "pool_shed_resident_validation.py", "pool_shed_resident_data.py")},
              "scope": ["Resident releases use original offered arrivals and modeled prior completion, with stationary GPU history. Incoming work and probes use measured dispatch eligibility, native cache observations and actual generation; censored service uses planned generation.",
                  "This is a conditional destination scheduler check, not a prediction of source quiescence, KV transfer, handoff, prefix eviction or fleet placement. Full-context submitted prompts remain intact; cached tokens only decode measured native usage.",
                  "Unfinished requests remain in arrival counts; first-token waiting over one second counts as a known violation. TPOT excludes unfinished completions. Missing cache inputs are bracketed by full miss and maximal hit, not assigned a fitted reuse fraction.",
                  "Observed transport and dependency failures remain explicit. The pilot executes their planned demand without a transport-failure model, so those scout discrepancies do not identify GPU capacity or a replay slowdown.",
                  "TTFT and TPOT bands are max(0.2s,25%) and max(0.005s,25%); sampled queue-peak error at most2; resident outstanding, observed-first and known-violation count errors at most1; migration and burst-drain error max(2s,25%). These are pilot acceptance bands, not probabilistic confidence or production tail guarantees.",
                  "Window cohorts overlap; metric-check counts are not independent experimental samples. The adverse seed7101 replay episodes and loading scouts remain stress checks even though only seed7102 is the formal held-out seed.",
                  "Burst drain means completion of resident arrivals in60–90s, while later arrivals continue. It does not assert all subsequent queues vanish or repair an earlier SLO miss.",
                  "Client token intervals include buffering and stalls. The decode coefficient is an effective client-derived approximation, not a measured GPU iteration duration; isolated requests also contain submillisecond delivery bursts.",
                  "Endpoint before/after placement is structurally unidentifiable here: it shifts GPU events but gives the same client delivery times. The variants test absolute engine sample alignment, not whether actual dynamic endpoint delay is harmless.",
                  "Engine waiting includes remote-KV readiness as well as compute waiting. Observed dispatch eligibility and cache hits condition KV timing; elapsed initial KV spans do not independently validate transfer or queue prediction.",
                  "Unknown-cache limits are reported without refitting. Whole-site spare capacity never repays another GPU's work; this pilot is not installed in the fleet executor."]}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "predictions.csv", predictions)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("campaign_ready", "conditional_heldout_gate_pass", "conditional_stress_gate_pass")}))


if __name__ == "__main__":
    main()
