"""Run and fit live H100 migration timing measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

import migration_profiler as profiler
import network_campaign as network
from profiles import ModelProfile

SCHEMA = "queue-haul-live-timing-v1"
CONTEXTS = (8192, 16384, 24576, 31488)
METHODS = ("replay", "kv_transfer")
REGIONS = {"australiaeast": "Australia East",
           "southcentralus": "South Central US"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_plan(manifest_path: Path, cluster_path: Path, out: Path,
              stage: str, calibration_path: Path | None = None) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    cluster = network.Cluster.load(cluster_path)
    templates = sorted(manifest["sessions"], key=lambda row: row["rank"])
    scenarios = []

    def add(context, width, load, node, method, repeat, split,
            template_index=0, pair_id=None):
        selected = [templates[(template_index + index) % len(templates)]
                    for index in range(width)]
        sessions = [{"session_id": (f"timing-{context}-{repeat}-{order}"
                                    if pair_id else row["id"]),
                     **({"template_id": row["id"]} if pair_id else {}),
                     "initial_tokens": context,
                     "order": order}
                    for order, row in enumerate(selected)]
        identity = [stage, context, width, load, node.id, method, repeat]
        scenarios.append({
                                "scenario_id": profiler.object_hash(identity)[:16],
                                "cell_index": len(scenarios),
                                "design": "timing_live", "stage": stage,
                                "split": split,
                                "context_tokens": context, "concurrency": width,
                                "destination": node.id, "region": node.region,
                                "method": method, "destination_load": load,
                                "repeat": repeat, "deadline_s": 900,
                                **({"pair_id": pair_id,
                                    "template_index": template_index}
                                   if pair_id else {}),
                                "load_warmup_s": 10, "load_max_pending": 256,
                                "load_normalization": "destination_service",
                                "background": {item.id: [
                                    load if item.id == node.id else 0, 0]
                                    for item in cluster.destinations},
                                "sessions": sessions,
                                "moves": [{
                                    "session_id": row["session_id"],
                                    "destination_instance": node.id,
                                    "destination_pool": f"pool/{node.id}",
                                    "method": method, "order": row["order"],
                                    "path": [f"link/{node.id}"],
                                    "deadline_admitted": True,
                                } for row in sessions],
                            })

    if stage == "targeted":
        if calibration_path is None:
            raise ValueError("targeted timing requires a frozen calibration")
        blocks = [(context, repeat) for context in CONTEXTS
                  for repeat in range(3)]
        rng = random.Random(20260814)
        rng.shuffle(blocks)
        for context, repeat in blocks:
            paths = [(node, method) for node in cluster.destinations
                     for method in METHODS]
            rng.shuffle(paths)
            pair_id = profiler.object_hash([context, repeat])[:16]
            for node, method in paths:
                add(context, 1, 0, node, method, repeat,
                    "holdout" if repeat == 2 else "calibration",
                    CONTEXTS.index(context) * 3 + repeat, pair_id)
    else:
        widths, loads, repeats, contexts = ((1,), (0,), 1, (8192, 31488)) \
            if stage == "pilot" else ((1, 2, 4, 8), (0, .5, .8), 3, CONTEXTS)
        for context in contexts:
            for width in widths:
                for load in loads:
                    for node in cluster.destinations:
                        for method in METHODS:
                            for repeat in range(repeats):
                                add(context, width, load, node, method, repeat,
                                    "holdout" if stage == "pilot"
                                    or repeat == repeats - 1 else "train")
    profile_path = network.MODEL_PATH.resolve()
    profile = ModelProfile.load(profile_path)
    if stage == "targeted" and "H100" not in profile.hardware:
        raise ValueError("targeted timing requires the H100 model profile")
    plan = {
        "schema": SCHEMA, "stage": stage,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(profile_path),
                          "sha256": profiler.file_hash(profile_path),
                          "profile_id": profile.profile_id},
        **({"calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": profiler.file_hash(calibration_path.resolve())},
            "network_contract": network.freeze_contract(json.loads(
                calibration_path.resolve().read_text()))}
           if calibration_path else {}),
        "cluster": cluster.as_dict(), "scenarios": scenarios,
    }
    write_json(out, plan)
    return plan


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != SCHEMA or not plan.get("scenarios"):
        raise ValueError("invalid live timing plan")
    regions = {row["region"] for row in plan["cluster"]["destinations"]}
    if regions != set(REGIONS):
        raise ValueError("timing plan must use Australia East and South Central US")
    if any(row["method"] not in METHODS or row["context_tokens"] not in CONTEXTS
           or row["split"] not in {"train", "calibration", "holdout"}
           for row in plan["scenarios"]):
        raise ValueError("invalid timing cell")
    if plan.get("stage") == "targeted":
        if len(plan["scenarios"]) != 48 or not plan.get("calibration") \
                or "H100" not in plan["model_profile"]["profile_id"].upper():
            raise ValueError("invalid targeted timing contract")
        blocks = {}
        for row in plan["scenarios"]:
            key = row["context_tokens"], row["repeat"]
            blocks.setdefault(key, []).append(row)
        expected = {(context, repeat) for context in CONTEXTS
                    for repeat in range(3)}
        if set(blocks) != expected or any(
                len(rows) != 4
                or len({row["pair_id"] for row in rows}) != 1
                or len({row["sessions"][0]["session_id"] for row in rows}) != 1
                or {(row["region"], row["method"]) for row in rows}
                != {(region, method) for region in REGIONS for method in METHODS}
                or any(row["split"] != ("holdout" if repeat == 2
                                        else "calibration") for row in rows)
                for (_context, repeat), rows in blocks.items()):
            raise ValueError("targeted timing pairs or splits changed")
        if len({rows[0]["sessions"][0]["session_id"]
                for rows in blocks.values()}) != len(blocks):
            raise ValueError("targeted timing templates are not distinct")


def service_load(path: Path, start_ns: int, seconds: float,
                 prefill_tps: float, decode_tps: float) -> float:
    lo = start_ns - round(seconds * 1e9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if lo <= row["start_ns"] < start_ns]
    return sum(row["prompt_tokens"] for row in rows) / seconds / prefill_tps \
        + sum(row["output_tokens"] for row in rows) / seconds / decode_tps


def live_measurements(scenario: dict, result: dict) -> list[dict]:
    if result.get("status") != "complete":
        raise RuntimeError("timing result is incomplete")
    rows = [row for row in result.get("requests", []) if "request" in row]
    if len(rows) != len(scenario["sessions"]):
        raise RuntimeError("timing result has missing or extra requests")
    measured = []
    contexts = {row["session_id"]: row["initial_tokens"]
                for row in scenario["sessions"]}
    kv = ModelProfile.load(network.MODEL_PATH).case().kv_transfer
    for row in rows:
        request = row["request"]
        if request.get("status_code") != 200 \
                or not request.get("state_code_verified"):
            raise RuntimeError("timing request failed status or state validation")
        connections = [item for item in result.get("connections", [])
                       if item["route"] == f"api/{row['destination_instance']}"
                       and request["start_ns"] <= int(item["start_ns"])
                       <= request["end_ns"]]
        if len(connections) != 1:
            raise RuntimeError("timing request needs exactly one API connection")
        transfers = [item for item in result.get("resp_transfers", [])
                     if request["start_ns"] <= int(item["start_ns"])
                     <= request["end_ns"]]
        gets = [item for item in transfers if item["command"] == "GET"]
        cached = int(request.get("cached_tokens", 0))
        if row["method"] == "replay" and (cached or gets):
            raise RuntimeError("replay unexpectedly used cached KV")
        if row["method"] == "kv_transfer" and (not cached or not gets):
            raise RuntimeError("KV timing request lacked cache hits or GETs")
        get_payload = sum(int(item["payload_bytes"]) for item in gets)
        get_wire = sum(int(item["response_wire_bytes"]) for item in gets)
        get_window = ((max(int(item["end_ns"]) for item in gets)
                       - min(int(item["start_ns"]) for item in gets)) / 1e9
                      if gets else None)
        if gets and (get_payload <= 0 or get_wire <= get_payload
                     or not get_window or get_window <= 0):
            raise RuntimeError("invalid KV GET bytes or data window")
        context = contexts[row["session_id"]]
        if row["method"] == "kv_transfer" and (
                cached != context
                or get_payload != cached // kv.block_tokens * kv.block_bytes):
            raise RuntimeError("KV timing request did not reconstruct full state")
        first_response = profiler.first_stream_ns(request)
        timestamps = [request["start_ns"], connections[0]["client_first_byte_ns"],
                      connections[0]["client_last_byte_ns"],
                      connections[0]["target_first_byte_ns"], first_response,
                      connections[0]["target_last_byte_ns"], request["end_ns"]]
        if any(value is None for value in timestamps) \
                or list(map(int, timestamps)) != sorted(map(int, timestamps)):
            raise RuntimeError("timing stage timestamps are missing or unordered")
        measured.append({"row": row, "request": request,
                         "connection": connections[0], "transfers": transfers,
                         "gets": gets, "get_payload_bytes": get_payload,
                         "get_wire_bytes": get_wire,
                         "get_window_s": get_window})
    return measured


def run(plan_path: Path, cluster_path: Path, calibration_path: Path,
        run_root: Path, key: Path) -> None:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    cluster = network.Cluster.load(cluster_path)
    if network.Cluster.parse(plan["cluster"]) != cluster:
        raise ValueError("cluster differs from timing plan")
    manifest_path = Path(plan["manifest"]["path"])
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("timing manifest changed")
    profile_path = network.MODEL_PATH.resolve()
    profile = ModelProfile.load(profile_path)
    if profiler.file_hash(profile_path) != plan["model_profile"]["sha256"] \
            or profile.profile_id != plan["model_profile"]["profile_id"]:
        raise RuntimeError("timing model profile changed")
    calibration = json.loads(calibration_path.read_text())
    network.validate_calibration(calibration)
    if profiler.file_hash(calibration_path) != plan["calibration"]["sha256"]:
        raise RuntimeError("timing network calibration changed")
    contract = network.freeze_contract(calibration)
    if contract != plan["network_contract"]:
        raise RuntimeError("timing network contract changed")
    reports = network.host_check(cluster, key)
    signatures = {row["git_sha"] for row in reports.values()}
    if signatures != {plan["git_sha"]}:
        raise RuntimeError("timing plan and live hosts do not share one commit")
    run_root.mkdir(parents=True, exist_ok=True)
    frozen = run_root / "plan.json"
    if frozen.exists() and frozen.read_bytes() != plan_path.read_bytes():
        raise RuntimeError("run root contains a different timing plan")
    if not frozen.exists():
        shutil.copy2(plan_path, frozen)
    write_json(run_root / "run_metadata.json", {
        "schema": SCHEMA, "git_sha": plan["git_sha"], "hosts": reports,
        "manifest": plan["manifest"], "model_profile": plan["model_profile"],
        "calibration": plan["calibration"],
        "network_contract": plan["network_contract"]})
    manifest = json.loads(manifest_path.read_text())
    stack_index = len(list((run_root / "stacks").glob("stack-*"))) + 1
    stack = network.start_cluster(
        cluster, key, network.freeze_contract(calibration), "natural",
        run_root / "stacks" /
        f"{run_root.name}-stack-{stack_index:04d}")
    case = ModelProfile.load(network.MODEL_PATH).case()
    rates = (case.prefill.rate(network.SINK_LOAD_PREFILL_TOKENS, 1),
             case.decode.rate(network.SINK_LOAD_PREFILL_TOKENS, 1))
    try:
        for scenario in plan["scenarios"]:
            root = run_root / "scenarios" / scenario["scenario_id"]
            if (root / "gate.json").exists():
                continue
            attempt = len(list(root.glob("attempt-*"))) + 1
            attempt_root = root / f"attempt-{attempt:04d}"
            result = network.run_network_scenario(
                stack, manifest, scenario, attempt_root, case.F)
            if plan["stage"] == "targeted":
                live_measurements(scenario, result)
            path = attempt_root / f"sink_load_{scenario['destination']}.jsonl"
            achieved = service_load(path, result["started_ns"], 4, *rates) \
                if scenario["destination_load"] else 0
            gate = {"requested_load": scenario["destination_load"],
                    "achieved_load": achieved,
                    "passed": abs(achieved - scenario["destination_load"]) <= .05}
            write_json(attempt_root / "load_gate.json", gate)
            if not gate["passed"]:
                raise RuntimeError(
                    f"destination load gate failed: {achieved:.3f} versus "
                    f"{scenario['destination_load']:.3f}")
            write_json(root / "gate.json", {"attempt": attempt, **gate})
            write_json(run_root / "progress.json", {
                "complete": len(list(run_root.glob("scenarios/*/gate.json"))),
                "total": len(plan["scenarios"]),
                "last_scenario": scenario["scenario_id"],
            })
    finally:
        network.stop_cluster(stack)


def latest_result(root: Path, scenario_id: str) -> dict | None:
    gates = root / "scenarios" / scenario_id / "gate.json"
    if root.name.startswith("frontier-"):
        latest = network._latest_result(root / "scenarios" / scenario_id)
        return latest[1] if latest and latest[1].get("status") == "complete" else None
    if not gates.exists():
        return None
    attempt = json.loads(gates.read_text())["attempt"]
    return json.loads((gates.parent / f"attempt-{attempt:04d}/result.json").read_text())


def collect(run_root: Path) -> list[dict]:
    plan = json.loads((run_root / "plan.json").read_text())
    regions = {row["id"]: row["region"]
               for row in plan["cluster"]["destinations"]}
    rows = []
    for scenario in plan["scenarios"]:
        result = latest_result(run_root, scenario["scenario_id"])
        if not result:
            continue
        requests = [row for row in result.get("requests", []) if "request" in row]
        checked = live_measurements(scenario, result) \
            if plan.get("stage") == "targeted" else []
        counts = {(destination, method): sum(
            row["destination_instance"] == destination
            and row["method"] == method for row in requests)
            for destination in regions for method in METHODS}
        contexts = {row["session_id"]: row["initial_tokens"]
                    for row in scenario["sessions"]}
        for row in requests:
            destination, request = row["destination_instance"], row["request"]
            match = next((item for item in checked if item["row"] is row), None)
            connections = [item for item in result.get("connections", [])
                           if item["route"] == f"api/{destination}"
                           and request["start_ns"] <= int(item["start_ns"])
                           <= request["end_ns"]]
            connection = match["connection"] if match else min(
                connections, key=lambda item: abs(
                    int(item["start_ns"]) - request["start_ns"]), default={})
            transfers = match["transfers"] if match else [
                item for item in result.get("resp_transfers", [])
                if request["start_ns"] <= int(item["start_ns"])
                <= request["end_ns"]]
            gets = match["gets"] if match else [
                item for item in transfers if item["command"] == "GET"]
            span = lambda first, last: ((int(last) - int(first)) / 1e9
                                        if first and last else None)
            first_response = profiler.first_stream_ns(request)
            prompt = int(request.get("prompt_tokens", 0))
            cached = int(request.get("cached_tokens", 0))
            processed = int(request.get("processed_tokens", 0)) \
                or prompt - cached
            get_payload = sum(int(item["payload_bytes"]) for item in gets)
            get_wire = sum(int(item["response_wire_bytes"]) for item in gets)
            get_window = span(
                min((item["start_ns"] for item in gets), default=None),
                max((item["end_ns"] for item in gets), default=None))
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "condition_index": scenario.get(
                    "condition_index", scenario.get("cell_index", 0)),
                "split": scenario.get("split", "prior_live"),
                "destination": regions[destination], "method": row["method"],
                "path": f"{regions[destination]}:{row['method']}",
                "context_tokens": contexts[row["session_id"]],
                "width": len(requests), "same_path_width":
                    counts[destination, row["method"]],
                "destination_width": sum(counts[destination, value]
                                         for value in METHODS),
                "order_fraction": row["order"] / max(1, len(requests) - 1),
                "destination_load": scenario["background"][destination][0],
                "repeat": scenario.get("repeat", 0),
                "observed_s": (first_response - request["start_ns"]) / 1e9,
                "initial_time_to_first_response_s":
                    (first_response - request["start_ns"]) / 1e9,
                "measured_prompt_tokens": prompt,
                "measured_processed_tokens": processed,
                "decode_tail_s": (request["end_ns"] - first_response) / 1e9,
                "api_upload_s": span(connection.get("client_first_byte_ns"),
                                     connection.get("client_last_byte_ns")),
                "remote_response_start_s": span(
                    connection.get("client_last_byte_ns"),
                    connection.get("target_first_byte_ns")),
                "response_header_to_first_response_s": span(
                    connection.get("target_first_byte_ns"), first_response),
                "response_stream_s": span(
                    connection.get("target_first_byte_ns"),
                    connection.get("target_last_byte_ns")),
                "client_residual_s": span(
                    connection.get("target_last_byte_ns"), request["end_ns"]),
                "kv_ingest_envelope_s": span(
                    min((item["start_ns"] for item in transfers), default=None),
                    max((item["end_ns"] for item in transfers), default=None)),
                "kv_get_window_s": get_window,
                "measured_kv_bytes": get_payload,
                "kv_get_wire_bytes": get_wire,
                "kv_get_rate_Bps": get_payload / get_window
                    if get_window else None,
                "bandwidth_mbps": plan.get("network_contract", {})
                    .get("paths", {}).get(destination, {}).get("natural_mbps"),
            })
    return rows


def feature_names(paths: list[str]) -> list[str]:
    return [f"{path}:{name}" for path in paths for name in (
        "intercept", "log_context", "log_context_sq", "log_same_path_width",
        "log_destination_width", "destination_load", "order_fraction")]


def features(row: dict, paths: list[str]) -> list[float]:
    context = math.log(row["context_tokens"] / 8192)
    values = []
    for path in paths:
        on = float(row["path"] == path)
        values.extend((on, on * context, on * context ** 2,
                       on * math.log1p(row["same_path_width"]),
                       on * math.log1p(row["destination_width"]),
                       on * row["destination_load"],
                       on * row["order_fraction"]))
    return values


def metrics(rows: list[dict], predicted: np.ndarray) -> dict:
    observed = np.asarray([row["observed_s"] for row in rows])
    ratio, error = observed / predicted, abs(predicted / observed - 1)
    return {"n": len(rows), "median_actual_over_predicted":
            float(np.median(ratio)), "p90_actual_over_predicted":
            float(np.quantile(ratio, .9)), "median_absolute_percentage_error":
            float(np.median(error)), "within_20_percent":
            float(np.mean(error <= .2)), "max_actual_over_predicted":
            float(np.max(ratio))}


def fit(run_root: Path, out: Path, folds: int = 5) -> dict:
    rows = collect(run_root)
    if len(rows) < 100 or len({row["path"] for row in rows}) != 4:
        raise RuntimeError("timing fit requires 100 live samples across four paths")
    paths, predictions, validation = sorted({row["path"] for row in rows}), [], []
    for fold in range(folds):
        train = [row for row in rows if row["condition_index"] % folds != fold]
        holdout = [row for row in rows if row["condition_index"] % folds == fold]
        coefficient = np.linalg.lstsq(
            np.asarray([features(row, paths) for row in train]),
            np.log([row["observed_s"] for row in train]), rcond=None)[0]
        predicted = np.exp(np.asarray(
            [features(row, paths) for row in holdout]) @ coefficient)
        validation.append({"fold": fold, **metrics(holdout, predicted)})
        predictions.extend({**row, "fold": fold, "predicted_s": float(value)}
                           for row, value in zip(holdout, predicted))
    coefficient = np.linalg.lstsq(
        np.asarray([features(row, paths) for row in rows]),
        np.log([row["observed_s"] for row in rows]), rcond=None)[0]
    gate = (max(row["median_absolute_percentage_error"] for row in validation)
            <= .15 and max(row["p90_actual_over_predicted"]
                           for row in validation) <= 1.5
            and max(abs(math.log(row["median_actual_over_predicted"]))
                    for row in validation) <= math.log(1.1))
    model = {"schema": SCHEMA, "source": "measured_live_transfers",
             "run_root": str(run_root), "samples": len(rows), "paths": paths,
             "feature_names": feature_names(paths),
             "coefficients": coefficient.tolist(), "cross_validation": validation,
             "gate": {"passed": gate, "max_fold_mdape": .15,
                      "max_fold_p90_late_ratio": 1.5,
                      "median_bias_ratio": [1 / 1.1, 1.1]},
             "stage_semantics": {
                 "api_upload_s": "measured at the source proxy",
                 "remote_response_start_s":
                     "measured route plus queue plus replay/KV ingest envelope",
                 "response_header_to_first_response_s":
                     "measured response-header to first streamed response",
                 "kv_ingest_envelope_s": "measured EXISTS plus GET envelope",
                 "kv_get_window_s": "measured GET-only data window",
                 "measured_kv_bytes": "sum of RESP GET payload bytes",
                 "response_stream_s": "measured destination response stream",
                 "client_residual_s": "measured proxy-to-client completion",
                 "internal_queue_s": "not identifiable without server tracing",
             }}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "model.json", model)
    with (out / "holdout_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys(),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
    if not gate:
        raise RuntimeError("retrospective timing validation gate failed")
    return model


def validate(model_path: Path, run_root: Path, out: Path) -> dict:
    model, rows = json.loads(model_path.read_text()), collect(run_root)
    paths = model["paths"]
    if len(rows) < 8 or {row["path"] for row in rows} != set(paths):
        raise RuntimeError("live validation needs at least eight samples across four paths")
    predicted = np.exp(np.asarray(
        [features(row, paths) for row in rows]) @ model["coefficients"])
    overall = metrics(rows, predicted)
    by_path = {path: metrics(
        [row for row in rows if row["path"] == path],
        predicted[[row["path"] == path for row in rows]]) for path in paths}
    passed = overall["median_absolute_percentage_error"] <= .2 \
        and .9 <= overall["median_actual_over_predicted"] <= 1.1 \
        and overall["p90_actual_over_predicted"] <= 1.5 \
        and all(.75 <= row["median_actual_over_predicted"] <= 4 / 3
                for row in by_path.values())
    report = {"schema": SCHEMA, "source": "new_live_holdout",
              "model": str(model_path), "run_root": str(run_root),
              "overall": overall, "by_path": by_path, "passed": passed}
    write_json(out, report)
    if not passed:
        raise RuntimeError("new live timing holdout gate failed")
    return report


def refit_targeted(profile_path: Path, run_root: Path, out: Path) -> dict:
    plan = json.loads((run_root / "plan.json").read_text())
    raw, rows = json.loads(profile_path.read_text()), collect(run_root)
    if profiler.file_hash(profile_path) != plan["model_profile"]["sha256"] \
            or raw["profile_id"] != plan["model_profile"]["profile_id"]:
        raise RuntimeError("targeted fit profile differs from frozen plan")
    calibration_path = Path(plan["calibration"]["path"])
    if profiler.file_hash(calibration_path) != plan["calibration"]["sha256"]:
        raise RuntimeError("targeted fit calibration differs from frozen plan")
    calibration = [row for row in rows if row["split"] == "calibration"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    paths = [f"{region}:{method}" for region in REGIONS for method in METHODS]
    if len(calibration) != 32 or len(holdout) != 16 \
            or {row["path"] for row in calibration} != set(paths) \
            or {row["path"] for row in holdout} != set(paths):
        raise RuntimeError("targeted refit requires its exact live split contract")
    expected = {(path, context): (2, 1) for path in paths for context in CONTEXTS}
    actual = {(path, context): (
        sum(row["path"] == path and row["context_tokens"] == context
            for row in calibration),
        sum(row["path"] == path and row["context_tokens"] == context
            for row in holdout)) for path, context in expected}
    if actual != expected or {row["repeat"] for row in calibration} != {0, 1} \
            or {row["repeat"] for row in holdout} != {2}:
        raise RuntimeError("targeted repeats or contexts changed")

    replay_curve = []
    for context in CONTEXTS:
        selected = [row for row in calibration
                    if row["method"] == "replay"
                    and row["context_tokens"] == context]
        replay_curve.append([context, float(np.median([
            context / row["observed_s"]
            for row in selected]))])
    replay_times = np.maximum.accumulate([
        context / rate for context, rate in replay_curve])
    replay_curve = [[context, context / elapsed]
                    for (context, _rate), elapsed in zip(
                        replay_curve, replay_times)]

    kv = [row for row in calibration if row["method"] == "kv_transfer"]
    completions = np.linspace(0, min(row["observed_s"] for row in kv) * .8,
                              400)
    best = None
    for completion in completions:
        bottlenecks = {}
        for path in paths[1::2]:
            selected = [row for row in kv if row["path"] == path]
            size = np.asarray([row["measured_kv_bytes"] for row in selected],
                              dtype=float)
            elapsed = np.asarray([row["observed_s"] - completion
                                  for row in selected])
            iperf = selected[0]["bandwidth_mbps"] * 1e6 / 8
            bottlenecks[path] = min(float(size @ size / (size @ elapsed)),
                                    iperf)
        predicted = np.asarray([
            completion + row["measured_kv_bytes"] / bottlenecks[row["path"]]
            for row in kv])
        observed = np.asarray([row["observed_s"] for row in kv])
        loss = float(np.mean(np.log(predicted / observed) ** 2))
        if best is None or loss < best[0]:
            best = loss, float(completion), bottlenecks
    _, kv_completion, bottlenecks = best
    kv_rate = max(bottlenecks.values())
    nodes = {row["region"]: row["id"]
             for row in plan["cluster"]["destinations"]}
    effective_rates = {
        region: (plan["network_contract"]["paths"][nodes[region]]
                 ["natural_mbps"] * 1e6 / 8 if rate == kv_rate else rate)
        for path, rate in bottlenecks.items()
        for region in REGIONS if path.startswith(region)}

    def predict(row):
        context = row["context_tokens"]
        if not CONTEXTS[0] <= context <= CONTEXTS[-1]:
            raise ValueError("timing prediction outside measured context range")
        if row["method"] == "replay":
            rate = np.interp(context, *np.asarray(replay_curve).T)
            return context / rate
        network = row["measured_kv_bytes"] / effective_rates[
            row["path"].split(":")[0]]
        return max(network, row["measured_kv_bytes"] / kv_rate) + kv_completion

    predicted = np.asarray([predict(row) for row in holdout])
    overall = metrics(holdout, predicted)
    by_path = {value: metrics(
        [row for row in holdout if row["path"] == value],
        predicted[[row["path"] == value for row in holdout]])
        for value in paths}
    by_context = {str(context): metrics(
        [row for row in holdout if row["context_tokens"] == context],
        predicted[[row["context_tokens"] == context for row in holdout]])
        for context in CONTEXTS}

    def passed_group(row):
        return row["median_absolute_percentage_error"] <= .1 \
            and row["p90_actual_over_predicted"] <= 1.2 \
            and .95 <= row["median_actual_over_predicted"] <= 1.05 \
            and row["within_20_percent"] >= .9 \
            and row["max_actual_over_predicted"] <= 1.25

    passed = passed_group(overall) \
        and all(passed_group(row) for row in by_path.values()) \
        and all(passed_group(row) for row in by_context.values())
    replay_error = max(abs(value / row["observed_s"] - 1)
                       for row, value in zip(holdout, predicted)
                       if row["method"] == "replay")
    kv_error = max(abs(value / row["observed_s"] - 1)
                   for row, value in zip(holdout, predicted)
                   if row["method"] == "kv_transfer")
    fitted = {"schema": SCHEMA, "source": "measured_live_transfers",
              "target": "initial_time_to_first_response_s",
              "run_root": str(run_root), "git_sha": plan["git_sha"],
              "valid_context_tokens": [CONTEXTS[0], CONTEXTS[-1]],
              "replay_tps": replay_curve,
              "kv_destination_bytes_per_s": kv_rate,
              "kv_initial_completion_s": kv_completion,
              "kv_effective_path_bytes_per_s": effective_rates,
              "network_contract": plan["network_contract"],
              "train_repeats": [0, 1], "validation_repeat": 2}
    report = {"schema": SCHEMA, "source": "unseen_live_holdout",
              "target": "initial_time_to_first_response_s",
              "overall": overall, "by_path": by_path,
              "by_context": by_context, "passed": passed}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "model.json", fitted)
    write_json(out / "validation.json", report)
    with (out / "holdout_predictions.csv").open("w", newline="") as handle:
        predictions = [{**row, "predicted_s": float(value)}
                       for row, value in zip(holdout, predicted)]
        writer = csv.DictWriter(handle, fieldnames=predictions[0],
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(predictions)
    if passed:
        central = raw["cases"]["central"]
        central["replay_tps"]["1"] = replay_curve
        central["replay_completion_s"] = 0
        central["kv_transfer"]["destination_bytes_per_s"] = kv_rate
        central["kv_transfer"]["initial_completion_s"] = kv_completion
        for name, replay_scale, kv_scale, completion_scale in (
                ("faster", 1 / (1 - replay_error), 1 / (1 - kv_error), .9),
                ("slower", 1 / (1 + replay_error), 1 / (1 + kv_error), 1.1)):
            case = raw["cases"][name]
            case["replay_tps"]["1"] = [
                [context, rate * replay_scale] for context, rate in replay_curve]
            case["replay_completion_s"] = 0
            case["kv_transfer"]["destination_bytes_per_s"] = kv_rate * kv_scale
            case["kv_transfer"]["initial_completion_s"] = \
                kv_completion * completion_scale
        raw["profile_id"] = f"gpt-oss-20b-h100-nvl-tp1-live-{plan['git_sha'][:8]}"
        reference = f"{run_root} repeats 0-1; repeat 2 held out"
        raw["sources"]["replay"] = {
            "kind": "measured", "reference": reference,
            "valid_range": [CONTEXTS[0], CONTEXTS[-1]],
            "relative_error": replay_error}
        raw["sources"]["kv_transfer"] = {
            "kind": "measured", "reference": reference,
            "valid_range": [min(row["measured_kv_bytes"] for row in calibration
                                if row["method"] == "kv_transfer"),
                            max(row["measured_kv_bytes"] for row in calibration
                                if row["method"] == "kv_transfer")],
            "relative_error": kv_error}
        fitted_calibration = json.loads(calibration_path.read_text())
        for region, node in nodes.items():
            mbps = effective_rates[region] * 8 / 1e6
            fitted_calibration["paths"][node]["simultaneous_mbps"] = [mbps]
        aggregate = min(np.median(fitted_calibration[
            "aggregate_simultaneous_mbps"]),
            sum(effective_rates.values()) * 8 / 1e6)
        fitted_calibration["aggregate_simultaneous_mbps"] = [aggregate]
        fitted_calibration["timing_fit"] = {
            "run_root": str(run_root), "git_sha": plan["git_sha"],
            "target": "KV GET bytes to initial streamed response"}
        write_json(out / "profile.json", raw)
        write_json(out / "calibration.json", fitted_calibration)
        ModelProfile.load(out / "profile.json")
        network.validate_calibration(fitted_calibration)
    if not passed:
        raise RuntimeError("targeted unseen-context timing gate failed")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("make-plan")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--stage", choices=("pilot", "targeted", "full"),
                         required=True)
    command = sub.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command = sub.add_parser("fit")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("validate")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("refit-targeted")
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "make-plan":
        print(json.dumps(make_plan(
            args.manifest, args.cluster, args.out, args.stage,
            args.calibration), indent=2))
    elif args.command == "run":
        run(args.plan, args.cluster, args.calibration,
            args.run_root, args.ssh_key.expanduser())
    elif args.command == "fit":
        print(json.dumps(fit(args.run_root, args.out), indent=2))
    elif args.command == "validate":
        print(json.dumps(validate(
            args.model, args.run_root, args.out), indent=2))
    else:
        print(json.dumps(refit_targeted(
            args.profile, args.run_root, args.out), indent=2))


if __name__ == "__main__":
    main()
