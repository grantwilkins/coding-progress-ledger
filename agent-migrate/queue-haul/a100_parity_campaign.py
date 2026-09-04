"""Run balanced A100 queue timing parity and reduce A100 parity figures."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import migration_profiler as profiler
import network_campaign as network
import plot_style
from live_timing_campaign import live_measurements
from plot_h100_power_parity import load as load_power, write as write_power
from plot_live_timing_parity import write_queue
from profiles import ModelProfile


ROOT = Path(__file__).parent
PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
TIMING_MODEL = ROOT / "outputs/timing-power-validation-20260814/separation-regional-timing-v2.json"
ACTIONS = ("replay", "kv_transfer", "mixed")
LOADS = (0, .25, .5, .8)
SCHEMA = "queue-haul-a100-timing-parity-v1"
plot_style.apply()


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _timing_paths(model: dict, profile_path: Path, calibration_path: Path) -> dict:
    paths = model.get("network_contract", {}).get("paths", {})
    if model.get("schema") != network.PLAN_SCHEMA \
            or model.get("model_profile", {}).get("sha256") != profiler.file_hash(profile_path) \
            or model.get("calibration", {}).get("sha256") \
            != profiler.file_hash(calibration_path) \
            or set(paths) != {"east", "germany"}:
        raise ValueError("timing model is not the pinned East/Germany A100 fit")
    for path in paths.values():
        if path.get("natural_mbps", 0) <= 0 \
                or set(path.get("migration_components", {})) != {"replay", "kv_transfer"}:
            raise ValueError("timing model lacks regional migration components")
    return paths


def _move_time(profile: ModelProfile, paths: dict, context: int,
               destination: str, method: str) -> float:
    case, path = profile.case(), paths[destination]
    component = path["migration_components"][method]
    if not component["context_range"][0] <= context <= component["context_range"][1]:
        raise ValueError("timing context is outside the fitted range")
    if method == "replay":
        return component["compute_completion_factor"] * (
            context / case.replay.conservative_rate(context, 1)
            + case.replay_completion_s) + case.switch_s
    return case.kv_transfer.sealed_bytes(context) \
        / (path["natural_mbps"] * 125_000) + component["residual_s"]


def _prediction(profile: ModelProfile, paths: dict, sessions: list[dict],
                moves: list[dict]) -> dict:
    contexts = {row["session_id"]: row["initial_tokens"] for row in sessions}
    rows, totals = [], defaultdict(float)
    for move in moves:
        context = contexts[move["session_id"]]
        destination, method = move["destination_instance"], move["method"]
        duration = _move_time(profile, paths, context, destination, method)
        rows.append({"session_id": move["session_id"], "destination": destination,
                     "method": method, "context_tokens": context,
                     "predicted_s": duration})
        totals[destination, method] += duration
    return {"predicted_s": max(totals.values()),
            "path_s": {f"{destination}:{method}": value
                       for (destination, method), value in sorted(totals.items())},
            "moves": rows}


def make_timing_plan(manifest_path: Path, cluster_path: Path,
                     calibration_path: Path, profile_path: Path,
                     timing_model_path: Path, out: Path, seed: int = 20260903,
                     scenarios_per_action: int = 40) -> dict:
    if scenarios_per_action < 2:
        raise ValueError("timing parity needs at least two scenarios per action")
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    templates = sorted(manifest["sessions"], key=lambda row: row["id"])
    cluster = network.Cluster.load(cluster_path)
    if {node.id for node in cluster.destinations} != {"east", "germany"}:
        raise ValueError("A100 timing parity requires East and Germany destinations")
    calibration = json.loads(calibration_path.read_text())
    network.validate_calibration(calibration)
    contract = network.freeze_contract(calibration)
    profile = ModelProfile.load(profile_path)
    if "A100" not in profile.hardware:
        raise ValueError("timing parity requires an A100 profile")
    model = json.loads(timing_model_path.read_text())
    paths = _timing_paths(model, profile_path, calibration_path)
    block = profile.case().kv_transfer.block_tokens
    lower = max(8192, max(path["migration_components"][method]["context_range"][0]
                          for path in paths.values() for method in ("replay", "kv_transfer")))
    upper = min(31488, min(path["migration_components"][method]["context_range"][1]
                           for path in paths.values() for method in ("replay", "kv_transfer")))
    context_blocks = tuple(range((lower + block - 1) // block, upper // block + 1))
    if len(context_blocks) < 32:
        raise ValueError("timing model has insufficient context support")
    destinations = tuple(sorted(paths))
    scenarios = []
    for action_index, action in enumerate(ACTIONS):
        for index in range(scenarios_per_action):
            rng = random.Random(profiler.stable_seed(seed, action, index))
            width = rng.randint(4, 16)
            pairs = ([(destinations[i % 2], action) for i in range(width)]
                     if action != "mixed" else
                     [(destinations[i % 2], ("replay", "kv_transfer")[(i // 2) % 2])
                      for i in range(width)])
            rng.shuffle(pairs)
            contexts = [rng.choice(context_blocks) * block for _ in range(width)]
            sessions, moves = [], []
            for order, (context, (destination, method)) in enumerate(zip(contexts, pairs)):
                template = templates[(index + order) % len(templates)]
                session_id = f"{template['id']}-a100-parity-{action_index}-{index}-{order}"
                sessions.append({"session_id": session_id, "template_id": template["id"],
                                 "job_class": template["job_class"], "turn_index": 0,
                                 "initial_tokens": context, "order": order})
                moves.append({"session_id": session_id,
                              "destination_instance": destination,
                              "destination_pool": f"pool/{destination}",
                              "method": method, "order": order,
                              "path": [f"link/{destination}"],
                              "deadline_admitted": True})
            background = {destination: [rng.choice(LOADS), 0]
                          for destination in destinations}
            identity = [SCHEMA, seed, action, index, sessions, moves, background,
                        profiler.file_hash(timing_model_path)]
            scenarios.append({
                "scenario_id": profiler.object_hash(identity)[:16],
                "design": "calibration", "condition_id": action,
                "condition_index": action_index * scenarios_per_action + index,
                "repeat": 0, "policy": f"fixed_{action}",
                "workload": "agentic_tool_loop", "bandwidth": "natural",
                "bandwidth_mbps": {node: row["natural_mbps"]
                                   for node, row in contract["paths"].items()},
                "deadline_s": 300, "background": background,
                "source_load": .8, "load_normalization": "destination_service",
                "load_warmup_s": 10, "sessions": sessions, "moves": moves,
                "parity_prediction": {
                    "action": action, **_prediction(profile, paths, sessions, moves),
                },
            })
    random.Random(seed).shuffle(scenarios)
    plan = {
        "schema": network.PLAN_SCHEMA, "design": "calibration", "seed": seed,
        "manifest": {"path": str(manifest_path.resolve()),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(profile_path.resolve()),
                          "sha256": profiler.file_hash(profile_path)},
        "calibration": {"path": str(calibration_path.resolve()),
                        "sha256": profiler.file_hash(calibration_path)},
        "network_contract": contract, "cluster": cluster.as_dict(),
        "policies": [f"fixed_{action}" for action in ACTIONS],
        "conditions": [], "repeats": 1, "sessions_per_scenario": None,
        "parity": {
            "schema": SCHEMA, "scenarios_per_action": scenarios_per_action,
            "timing_model": {"path": str(timing_model_path.resolve()),
                             "sha256": profiler.file_hash(timing_model_path)},
            "prediction": "sum fitted isolated times per destination/action; take maximum",
            "gates": {"mae_s": 3, "r2": .8, "minimum_unique_fraction": .9},
        },
        "scenarios": scenarios,
    }
    validate_timing_plan(plan)
    _write_new(out, plan)
    return plan


def validate_timing_plan(plan: dict, check_files: bool = True) -> None:
    network.validate_plan(plan)
    parity, scenarios = plan.get("parity", {}), plan["scenarios"]
    per_action = parity.get("scenarios_per_action")
    if parity.get("schema") != SCHEMA or not isinstance(per_action, int) \
            or Counter(row["parity_prediction"]["action"] for row in scenarios) \
            != Counter({action: per_action for action in ACTIONS}):
        raise ValueError("timing parity action blocks are incomplete")
    for row in scenarios:
        methods = {move["method"] for move in row["moves"]}
        action = row["parity_prediction"]["action"]
        if len(row["moves"]) != len(row["sessions"]) \
                or {move["destination_instance"] for move in row["moves"]} \
                != {"east", "germany"} \
                or methods != ({"replay", "kv_transfer"} if action == "mixed" else {action}) \
                or row["parity_prediction"]["predicted_s"] <= 0:
            raise ValueError("invalid timing parity scenario")
    predictions = {round(row["parity_prediction"]["predicted_s"], 9)
                   for row in scenarios}
    if len(predictions) < parity["gates"]["minimum_unique_fraction"] * len(scenarios):
        raise ValueError("timing predictions are too tightly binned")
    if check_files:
        for key, value in (("manifest", plan["manifest"]),
                           ("model profile", plan["model_profile"]),
                           ("calibration", plan["calibration"]),
                           ("timing model", parity["timing_model"])):
            if profiler.file_hash(Path(value["path"])) != value["sha256"]:
                raise RuntimeError(f"pinned {key} changed")
        profile_path = Path(plan["model_profile"]["path"])
        profile = ModelProfile.load(profile_path)
        paths = _timing_paths(
            json.loads(Path(parity["timing_model"]["path"]).read_text()),
            profile_path, Path(plan["calibration"]["path"]))
        for row in scenarios:
            action = row["parity_prediction"]["action"]
            if row["parity_prediction"] != {
                    "action": action,
                    **_prediction(profile, paths, row["sessions"], row["moves"])}:
                raise RuntimeError("frozen timing prediction changed")


def run_timing(plan_path: Path, run_root: Path, ssh_key: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_timing_plan(plan)
    network.MODEL_PATH = Path(plan["model_profile"]["path"])
    return network.run_campaign(
        network.Cluster.parse(plan["cluster"]), ssh_key,
        Path(plan["calibration"]["path"]), plan_path, run_root)


def timing_rows(run_root: Path) -> tuple[list[dict], dict]:
    plan = json.loads((run_root / "plan.json").read_text())
    validate_timing_plan(plan)
    network.MODEL_PATH = Path(plan["model_profile"]["path"])
    rows = []
    for scenario in plan["scenarios"]:
        latest = network._latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest is None or latest[1].get("status") != "complete":
            raise RuntimeError(f"missing complete result for {scenario['scenario_id']}")
        result = latest[1]
        if result.get("request_failures"):
            raise RuntimeError(f"request failure in {scenario['scenario_id']}")
        checked = live_measurements(scenario, result)
        expected = [(row["session_id"], row["destination_instance"], row["method"], row["order"])
                    for row in scenario["moves"]]
        actual = [(row["row"]["session_id"], row["row"]["destination_instance"],
                   row["row"]["method"], row["row"]["order"]) for row in checked]
        if actual != expected:
            raise RuntimeError(f"executed moves differ from plan for {scenario['scenario_id']}")
        measured = max((profiler.first_stream_ns(row["request"])
                        - row["request"]["start_ns"]) / 1e9 for row in checked)
        rows.append({"scenario_id": scenario["scenario_id"],
                     "action": scenario["parity_prediction"]["action"],
                     "predicted_s": scenario["parity_prediction"]["predicted_s"],
                     "measured_s": measured})
    residuals = [row["measured_s"] - row["predicted_s"] for row in rows]
    mean = statistics.fmean(row["measured_s"] for row in rows)
    denominator = sum((row["measured_s"] - mean) ** 2 for row in rows)
    if not denominator:
        raise RuntimeError("timing measurements have zero variance")
    mae = statistics.fmean(map(abs, residuals))
    r2 = 1 - sum(value * value for value in residuals) / denominator
    gates = plan["parity"]["gates"]
    summary = {
        "schema": "queue-haul-a100-timing-parity-result-v1",
        "scenarios": len(rows), "action_counts": dict(Counter(
            row["action"] for row in rows)),
        "unique_predictions": len({round(row["predicted_s"], 9) for row in rows}),
        "mae_s": mae, "r2": r2,
        "gates": {"mae": mae <= gates["mae_s"], "r2": r2 >= gates["r2"]},
    }
    summary["passed"] = all(summary["gates"].values())
    return rows, summary


def reduce_timing(run_root: Path, out: Path) -> dict:
    rows, summary = timing_rows(run_root)
    write_queue(rows, out)
    summary_path = out.with_name(f"{out.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["passed"]:
        raise RuntimeError("A100 timing parity gates failed")
    return summary


def plot_power(run_root: Path, history: list[Path], out: Path) -> None:
    metadata = json.loads((run_root / "metadata.json").read_text())
    gpu = metadata["gpu"]
    if metadata.get("hardware") != "a100" \
            or (gpu["name"], gpu["power_limit_w"]) \
            != ("NVIDIA A100 80GB PCIe", 300.0):
        raise RuntimeError("power parity requires the 300 W A100 campaign")
    write_power(load_power(run_root, history), out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare-timing")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--profile", type=Path, default=PROFILE)
    command.add_argument("--timing-model", type=Path, default=TIMING_MODEL)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20260903)
    command.add_argument("--scenarios-per-action", type=int, default=40)
    command = commands.add_parser("run-timing")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/azrs").expanduser())
    command = commands.add_parser("reduce-timing")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path,
                         default=ROOT / "outputs/a100_live_queue_makespan_parity")
    command = commands.add_parser("plot-power")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--history-run-root", type=Path, action="append", default=[])
    command.add_argument("--out", type=Path,
                         default=ROOT / "outputs/a100_power_model_parity")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "prepare-timing":
        make_timing_plan(args.manifest, args.cluster, args.calibration, args.profile,
                         args.timing_model, args.out, args.seed,
                         args.scenarios_per_action)
    elif args.command == "run-timing":
        run_timing(args.plan, args.run_root, args.ssh_key.expanduser())
    elif args.command == "reduce-timing":
        reduce_timing(args.run_root, args.out)
    else:
        plot_power(args.run_root, args.history_run_root, args.out)


if __name__ == "__main__":
    main()
