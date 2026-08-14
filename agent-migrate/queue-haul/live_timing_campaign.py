"""Run and fit live H100 migration timing measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
              stage: str) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    cluster = network.Cluster.load(cluster_path)
    templates = sorted(manifest["sessions"], key=lambda row: row["rank"])
    scenarios = []

    def add(context, width, load, node, method, repeat, split):
        sessions = [{"session_id": row["id"], "initial_tokens": context,
                     "order": order}
                    for order, row in enumerate(templates[:width])]
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
        south = next(node for node in cluster.destinations
                     if node.region == "southcentralus")
        for context in (8192, 31488):
            for repeat in range(2):
                add(context, 1, 0, south, "kv_transfer", repeat, "calibration")
        for node in cluster.destinations:
            for method in METHODS:
                if node != south or method != "kv_transfer":
                    add(16384, 1, 0, node, method, 0, "holdout")
        for context in (16384, 24576):
            add(context, 1, 0, south, "kv_transfer", 0, "holdout")
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
    plan = {
        "schema": SCHEMA, "stage": stage,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
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


def service_load(path: Path, start_ns: int, seconds: float,
                 prefill_tps: float, decode_tps: float) -> float:
    lo = start_ns - round(seconds * 1e9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if lo <= row["start_ns"] < start_ns]
    return sum(row["prompt_tokens"] for row in rows) / seconds / prefill_tps \
        + sum(row["output_tokens"] for row in rows) / seconds / decode_tps


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
    calibration = json.loads(calibration_path.read_text())
    network.validate_calibration(calibration)
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
        counts = {(destination, method): sum(
            row["destination_instance"] == destination
            and row["method"] == method for row in requests)
            for destination in regions for method in METHODS}
        contexts = {row["session_id"]: row["initial_tokens"]
                    for row in scenario["sessions"]}
        for row in requests:
            destination, request = row["destination_instance"], row["request"]
            connections = [item for item in result.get("connections", [])
                           if item["route"] == f"api/{destination}"
                           and request["start_ns"] <= int(item["start_ns"])
                           <= request["end_ns"]]
            connection = min(connections, key=lambda item: abs(
                int(item["start_ns"]) - request["start_ns"]), default={})
            transfers = [item for item in result.get("resp_transfers", [])
                         if request["start_ns"] <= int(item["start_ns"])
                         <= request["end_ns"]]
            span = lambda first, last: ((int(last) - int(first)) / 1e9
                                        if first and last else None)
            first_response = request.get("first_byte_ns", request["end_ns"])
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
                "observed_s": (request["end_ns"] - result["started_ns"]) / 1e9,
                "time_to_first_token_s":
                    (first_response - request["start_ns"]) / 1e9,
                "decode_tail_s": (request["end_ns"] - first_response) / 1e9,
                "api_upload_s": span(connection.get("client_first_byte_ns"),
                                     connection.get("client_last_byte_ns")),
                "remote_response_start_s": span(
                    connection.get("client_last_byte_ns"),
                    connection.get("target_first_byte_ns")),
                "response_header_to_token_s": span(
                    connection.get("target_first_byte_ns"), first_response),
                "response_stream_s": span(
                    connection.get("target_first_byte_ns"),
                    connection.get("target_last_byte_ns")),
                "client_residual_s": span(
                    connection.get("target_last_byte_ns"), request["end_ns"]),
                "kv_transfer_window_s": span(
                    min((item["start_ns"] for item in transfers), default=None),
                    max((item["end_ns"] for item in transfers), default=None)),
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
            float(np.mean(error <= .2))}


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
                 "response_header_to_token_s":
                     "measured response-header to first generated token",
                 "kv_transfer_window_s": "measured RESP transfer envelope",
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


def refit_targeted(model_path: Path, run_root: Path, out: Path) -> dict:
    model, rows = json.loads(model_path.read_text()), collect(run_root)
    calibration = [row for row in rows if row["split"] == "calibration"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    path = "southcentralus:kv_transfer"
    if len(calibration) != 4 or len(holdout) != 5 \
            or {row["path"] for row in holdout} != set(model["paths"]) \
            or any(row["path"] != path for row in calibration):
        raise RuntimeError("targeted refit requires its exact live split contract")
    coefficient = np.asarray(model["coefficients"], dtype=float)
    old = np.exp(np.asarray(
        [features(row, model["paths"]) for row in calibration]) @ coefficient)
    design = np.asarray([[1, math.log(row["context_tokens"] / 8192)]
                         for row in calibration])
    correction = np.linalg.lstsq(
        design, np.log([row["observed_s"] for row in calibration])
        - np.log(old), rcond=None)[0]
    names = model["feature_names"]
    coefficient[names.index(f"{path}:intercept")] += correction[0]
    coefficient[names.index(f"{path}:log_context")] += correction[1]
    predicted = np.exp(np.asarray(
        [features(row, model["paths"]) for row in holdout]) @ coefficient)
    overall = metrics(holdout, predicted)
    by_path = {value: metrics(
        [row for row in holdout if row["path"] == value],
        predicted[[row["path"] == value for row in holdout]])
        for value in model["paths"]}
    passed = overall["median_absolute_percentage_error"] <= .2 \
        and .9 <= overall["median_actual_over_predicted"] <= 1.1 \
        and overall["p90_actual_over_predicted"] <= 1.5 \
        and all(.75 <= row["median_actual_over_predicted"] <= 4 / 3
                for row in by_path.values())
    fitted = {**model, "coefficients": coefficient.tolist(),
              "targeted_calibration": {
                  "source": "measured_live_transfers", "run_root": str(run_root),
                  "path": path, "samples": len(calibration),
                  "log_intercept_correction": float(correction[0]),
                  "log_context_correction": float(correction[1])}}
    report = {"schema": SCHEMA, "source": "unseen_live_holdout",
              "overall": overall, "by_path": by_path, "passed": passed}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "model.json", fitted)
    write_json(out / "validation.json", report)
    if not passed:
        raise RuntimeError("targeted unseen-context timing gate failed")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("make-plan")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--cluster", type=Path, required=True)
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
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "make-plan":
        print(json.dumps(make_plan(
            args.manifest, args.cluster, args.out, args.stage), indent=2))
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
            args.model, args.run_root, args.out), indent=2))


if __name__ == "__main__":
    main()
