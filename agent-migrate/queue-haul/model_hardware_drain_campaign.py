"""Run and reduce the matched eight-session Azure drain campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from profiles import ModelProfile


ROOT = Path(__file__).resolve().parent
MODELS = {
    "A100": set(plot_style.MODELS),
    "H100": {"openai/gpt-oss-20b"},
}
ACTIONS = {"replay": ("east_replay", "germany_replay"),
           "kv_transfer": ("east_kv_transfer", "germany_kv_transfer")}
RUNTIME_VERSIONS = {
    "A100": "0.24.0,0.5.1",
    "H100": "0.22.0,0.5.1",
}
NETWORK_GATE = "queue-haul-network-model-gate-v1"


def freeze_network_profile(timing_root: Path, out: Path) -> dict:
    import matched_action_campaign as matched
    import model_architecture_campaign as architecture
    import network_campaign as network

    report_path = timing_root / "report.json"
    report = json.loads(report_path.read_text())
    model, rows = report["model"], report["rows"]
    contexts = sorted(report["contexts"])
    smoke = report.get("concurrent_smoke") or {}
    expected = {(node, context, repeat, method)
                for node in ("east", "germany") for context in contexts
                for repeat in range(3) for method in ACTIONS}
    if report.get("schema") != network.MIGRATION_TIMING_SCHEMA \
            or report.get("status") != "complete" or not report.get("all_passed") \
            or not report.get("literal_token_timing") \
            or not report.get("source_sleep_wake_passed") \
            or report.get("bandwidth") != "controlled_40" \
            or report.get("repeats") != 3 or contexts[0] > 14042 or contexts[-1] != 32256 \
            or len(rows) != len(expected) or report.get("completed") != len(expected) \
            or {(row["destination"], row["context_tokens"], row["repeat"], row["method"])
                for row in rows} != expected \
            or not smoke.get("passed") or smoke.get("sessions") != 8 \
            or smoke.get("context_tokens") != 32256 or len(smoke.get("requests", [])) != 16:
        raise ValueError("incomplete two-route network readiness evidence")
    for node in ("east", "germany"):
        requests = [row for row in smoke["requests"] if row["destination"] == node]
        if len(requests) != 8 or len({row["session_id"] for row in requests}) != 8 \
                or sum(row["method"] == "replay" for row in requests) != 4 \
                or sum(row["method"] == "kv_transfer" for row in requests) != 4 \
                or smoke["wire_bytes"].get(f"kv/{node}/target_to_client", 0) <= 0 \
                or (max(row["request"]["start_ns"] for row in requests)
                    - min(row["request"]["start_ns"] for row in requests)) / 1e9 \
                > network.DRAIN_DISPATCH_SKEW_S \
                or any(not network.literal_timing_completion(row["request"])
                       or row["request"]["prompt_tokens"] != 32256
                       or row["request"].get("cached_tokens", 0) != (
                           32256 // rows[0]["chunk_tokens"] * rows[0]["chunk_tokens"]
                           if row["method"] == "kv_transfer" else 0)
                       for row in requests):
            raise ValueError(f"invalid eight-session operational gate: {node}")
    evidence = [report_path]
    for row in rows:
        path = timing_root / "requests" / (
            f"{row['destination']}-{row['context_tokens']}-{row['repeat']}-{row['method']}.json")
        request = json.loads(path.read_text())
        if request["measurement"] != row or not row["passed"] \
                or not network.literal_timing_completion(request["request"]) \
                or request["request"]["prompt_tokens"] != row["context_tokens"]:
            raise ValueError(f"invalid raw timing evidence: {path}")
        evidence.append(path)
    for node in ("east", "germany"):
        path = timing_root / "nodes" / node / "sink.log"
        registrations = architecture._json_markers(path, "QH_KV_GEOMETRY ")
        if not registrations:
            raise ValueError(f"missing live KV registration: {node}")
        for registration in registrations:
            architecture._validate_registration(registration)
        evidence.append(path)
    prefill_path, power_path = matched.CALIBRATIONS[model], matched.POWER_CALIBRATIONS[model]
    prefill, power = (json.loads(path.read_text()) for path in (prefill_path, power_path))
    if prefill["model"] != model or power["model"] != model \
            or not power["validation"]["gate_passed"]:
        raise ValueError("invalid measured model compute/power inputs")
    raw = json.loads((ROOT / "profiles/gpt_oss_20b_h100_tp1.json").read_text())
    case = raw["cases"]["central"]
    train = [row for row in rows if row["repeat"] < 2]
    replay = {context: statistics.median(row["destination_ready_s"] for row in train
              if row["context_tokens"] == context and row["method"] == "replay")
              for context in contexts}
    wire = {context: round(statistics.median(row["kv_wire_bytes"] for row in train
            if row["context_tokens"] == context and row["method"] == "kv_transfer"))
            for context in contexts}
    rates = {node: report["network_contract"]["paths"][node]["controlled_mbps"]["40"] * 125000
             for node in ("east", "germany")}
    residual = max(0., statistics.median(row["destination_ready_s"]
        - row["kv_wire_bytes"] / rates[row["destination"]]
        for row in train if row["method"] == "kv_transfer"))
    errors = [{"destination": row["destination"], "method": row["method"],
               "context_tokens": row["context_tokens"],
               "observed_s": row["destination_ready_s"],
               "predicted_s": replay[row["context_tokens"]] if row["method"] == "replay"
               else wire[row["context_tokens"]] / rates[row["destination"]] + residual}
              for row in rows if row["repeat"] == 2]
    for row in errors:
        row["relative_error"] = abs(row["predicted_s"] - row["observed_s"]) / row["observed_s"]
    case.update(F=power["F_prefill_tps"], G=power["G_decode_tps"],
                phase_power=power["phase_power"],
                prefill_tps={"1": [[row["context_tokens"], row["prefill_tps_median"]]
                                   for row in prefill["curve"]]},
                replay_tps={"1": [[context, context / replay[context]] for context in contexts]},
                replay_completion_s=0,
                decode_tps={"1": [[context, 1 / statistics.median(row["mean_tpot_s"]
                    for row in train if row["context_tokens"] == context)] for context in contexts]})
    case["kv_transfer"].update(block_tokens=1, block_bytes=1, setup_s=0,
        destination_bytes_per_s=1e12, initial_completion_s=residual,
        bytes_by_context=[[context, wire[context]] for context in contexts])
    raw.update(schema="queue-haul-model-profile-v5", profile_id=f"network-{model}",
               status="fitted", model=model, gpus_per_node=1,
               max_power_load=power["max_power_load"],
               kv_capacity_tokens=min(row["kv_capacity_tokens"] for row in report["node_reports"].values()),
               cases={"central": case})
    for name in ("replay", "kv_transfer", "service", "capacity"):
        raw["sources"][name] = {"kind": "measured", "reference": str(report_path.resolve()),
            "valid_range": [min(contexts), max(contexts)],
            "relative_error": float(np.quantile([row["relative_error"] for row in errors
                if row["method"] == name], .9)) if name in ACTIONS else 0.}
    raw["sources"]["power"]["reference"] = str(power_path)
    raw["sources"]["service"]["reference"] += f"; historical prefill: {prefill_path}"
    raw["sources"]["service"]["relative_error"] = .3
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=2) + "\n")
    ModelProfile.load(out)
    evidence.extend((prefill_path, power_path))
    gate = {"schema": NETWORK_GATE, "model": model, "hardware": "H100", "passed": True,
        "scope": "cross-host operational readiness; held-out timing errors are diagnostics",
        "launch": {"passed": True, "sessions": 8, "context_tokens": 32256},
        "timing": {"observations": len(rows), "held_out": errors},
        "runtime": report["runtime"], "calibration_sha256": report["calibration_sha256"],
        "network_contract": report["network_contract"],
        "profile_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "evidence": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in evidence},
        "limitations": ["wire-byte transfer curve; no architecture geometry-equivalence claim",
                        "nominal runtime token capacity; eight-session operation tested on each destination",
                        "historical prefill and phase power; inherited action-specific power overhead",
                        "ongoing service and power shedding are modeled, not measured drain outcomes"]}
    out.with_suffix(".gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def _gated_profile(path: Path, hardware: str) -> ModelProfile:
    profile = ModelProfile.load(path)
    gate = json.loads(path.with_suffix(".gate.json").read_text())
    if hardware.lower() not in profile.hardware.lower() \
            or profile.precision.lower() not in {"bf16", "bfloat16"} \
            or profile.tensor_parallel != 1 \
            or (gate.get("schema") != NETWORK_GATE and (
                profile.kv_geometry is None
                or gate.get("schema") != "queue-haul-model-architecture-gate-v1")) \
            or gate.get("model") != profile.model \
            or hardware.lower() not in gate.get("hardware", "").lower() \
            or not gate.get("passed") or not gate.get("launch", {}).get("passed") \
            or gate.get("profile_sha256") != hashlib.sha256(
                path.read_bytes()).hexdigest():
        raise ValueError(f"{path} is not a gated BF16 TP1 {hardware} profile")
    return profile


def _profile(path: Path, hardware: str) -> tuple[ModelProfile, Path]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("model profiles must be inside the repository") from exc
    profile = _gated_profile(resolved, hardware)
    return profile, relative


def _snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and source.read_bytes() != destination.read_bytes():
        raise ValueError(f"profile snapshot changed: {destination}")
    if not destination.exists():
        shutil.copy2(source, destination)


def run(hardware: str, profiles: list[Path], cluster: Path,
        calibration: Path, manifest: Path, run_root: Path,
        ssh_key: Path, deadline_s: float = 30, source_service: bool = False,
        deadlines_s: list[float] | None = None) -> dict:
    cluster, calibration, manifest, run_root, ssh_key = (
        path.resolve() for path in
        (cluster, calibration, manifest, run_root, ssh_key))
    loaded = sorted((_profile(path, hardware) for path in profiles),
                    key=lambda row: row[0].model)
    models = {profile.model for profile, _ in loaded}
    if models not in (MODELS[hardware], set(plot_style.MODELS)) \
            or len(loaded) != len(models):
        raise ValueError(f"{hardware} requires exactly {sorted(MODELS[hardware])}")
    prepared = []
    for index, (profile, relative) in enumerate(loaded):
        slug = f"m{index}"
        plan, arm = run_root / "plans" / f"{slug}.json", run_root / "arms" / slug
        source, snapshot = ROOT / relative, arm / "profile.json"
        _snapshot(source, snapshot)
        _snapshot(source.with_suffix(".gate.json"),
                  snapshot.with_suffix(".gate.json"))
        env = {**os.environ, "QH_MODEL_PROFILE": str(relative),
               "QH_RUNTIME": "native", "QH_LMCACHE_MODE": "mp",
               "QH_NATIVE_RUNTIME_VERSIONS": RUNTIME_VERSIONS[hardware]}
        gate_path = source.with_suffix(".gate.json")
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            if gate.get("schema") == NETWORK_GATE:
                import network_campaign as network
                calibrated = json.loads(calibration.read_text())
                if gate["calibration_sha256"] != network.profiler.object_hash(calibrated) \
                        or gate["network_contract"] != network.freeze_contract(calibrated):
                    raise ValueError("network gate does not match campaign calibration")
                env["QH_NATIVE_RUNTIME_VERSIONS"] = ",".join(gate["runtime"][name]
                    for name in ("vllm", "lmcache"))
        subprocess.run([
            sys.executable, str(ROOT / "network_campaign.py"), "prepare",
            "--design", "drain", "--cluster", str(cluster),
            "--calibration", str(calibration), "--manifest", str(manifest),
            "--out", str(plan),
            *(["--deadline-s", str(deadline_s)] if deadline_s != 30 else []),
            *(["--source-service"] if source_service else []),
            *(["--deadlines-s", *map(str, deadlines_s)] if deadlines_s else []),
        ], cwd=ROOT, env=env, check=True)
        prepared.append((plan, arm, env))
    for block in range(5):
        ordered = prepared[block % len(prepared):] \
            + prepared[:block % len(prepared)]
        for plan, arm, env in ordered:
            subprocess.run([
                sys.executable, str(ROOT / "network_campaign.py"), "run",
                "--cluster", str(cluster),
                "--current-calibration", str(calibration),
                "--plan", str(plan), "--run-root", str(arm),
                "--ssh-key", str(ssh_key), "--stack-block", str(block),
            ], cwd=ROOT, env=env, check=True)
    for plan, arm, env in prepared:
        subprocess.run([
            sys.executable, str(ROOT / "network_campaign.py"), "reduce",
            "--plan", str(plan), "--run-root", str(arm),
        ], cwd=ROOT, env=env, check=True)
    return reduce([run_root], run_root, {
        (model, hardware) for model in models})


def _rows(run_roots: list[Path]) -> list[dict]:
    output, arms, matrices = [], set(), set()
    for root in run_roots:
        for arm in sorted((root / "arms").glob("*")):
            plan_path = arm / "plan.json"
            plan = json.loads(plan_path.read_text())
            summary = json.loads((arm / "summary.json").read_text())
            metadata = json.loads((arm / "run_metadata.json").read_text())
            profile_path = arm / "profile.json"
            profile = ModelProfile.load(profile_path)
            hardware = "H100" if "h100" in profile.hardware.lower() else "A100"
            profile = _gated_profile(profile_path, hardware)
            key = profile.model, hardware
            profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            runtime = metadata.get("runtime_environment", {})
            regions = {node["id"]: node["region"]
                       for node in plan["cluster"]["destinations"]}
            with (arm / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            expected_rows = 50 * len(plan.get("drain_deadlines_s", [30]))
            if plan.get("design") != "drain" or summary.get("expected") != expected_rows \
                    or summary.get("completed", 0) + summary.get("failed", 0) != expected_rows \
                    or summary.get("missing") or summary.get("invalid_evidence") \
                    or metadata.get("plan_sha256") != plan_sha \
                    or profile_sha != plan["model_profile"]["sha256"] \
                    or runtime.get("QH_RUNTIME") != "native" \
                    or runtime.get("QH_LMCACHE_MODE") != "mp" \
                    or len(rows) != expected_rows or key in arms \
                    or sum(row["status"] == "complete" for row in rows) \
                    != summary.get("completed") \
                    or any(row["status"] not in {"complete", "failed"}
                           or int(row.get("attempt", 1)) != 1
                           or int(row.get("excluded_attempts", 0))
                           for row in rows):
                raise ValueError(f"invalid drain arm: {arm}")
            arms.add(key)
            matrices.add((plan["manifest"]["sha256"], plan.get("force_movement", False),
                plan.get("source_service_normalized", False),
                tuple(plan.get("drain_deadlines_s", [plan.get("drain_deadline_s", 30)])), tuple(sorted(
                (row["condition_index"], row["repeat"], tuple(
                    item["initial_tokens"] for item in row["sessions"]))
                for row in plan["scenarios"]))))
            output.extend({"model": profile.model, "hardware": hardware,
                           "arm_root": str(arm), "plan_sha256": plan_sha,
                           "profile_sha256": profile_sha,
                           "manifest_sha256": plan["manifest"]["sha256"],
                           "east_region": regions["east"],
                           "germany_region": regions["germany"],
                           **row} for row in rows)
    if not output or len(matrices) != 1:
        raise ValueError("drain arms are absent or unmatched")
    return output


def deadline_action_mix(rows: list[dict], out: Path) -> dict:
    plot_style.apply()
    models = sorted({row["model"] for row in rows})
    deadlines = sorted({float(row["deadline_s"]) for row in rows})
    columns = [column for values in ACTIONS.values() for column in values]
    cells = []
    for model in models:
        for deadline in deadlines:
            selected = [row for row in rows if row["model"] == model
                        and float(row["deadline_s"]) == deadline]
            completed = [row for row in selected if row["status"] == "complete"]
            cells.append({"model": model, "deadline_s": deadline,
                "episodes": len(selected), "completed": len(completed),
                "failed": len(selected) - len(completed),
                "attained": sum(row["target_met"] == "True" for row in completed),
                **{column: statistics.mean(int(row[column]) for row in completed)
                   if completed else None for column in columns}})
    with (out / "deadline_action_mix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, cells[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    figure, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4),
                                sharey=True, squeeze=False)
    for model, axis in zip(models, axes[0]):
        selected = [cell for cell in cells if cell["model"] == model]
        example = next(row for row in rows if row["model"] == model)
        bottom = np.zeros(len(deadlines))
        for column in columns:
            slot, action = column.split("_", 1)
            identity = f"{example[slot + '_region']}_{action}"
            values = np.array([cell[column] or 0 for cell in selected])
            axis.bar(range(len(deadlines)), values, bottom=bottom,
                     color=plot_style.ACTION_COLORS[identity],
                     hatch=plot_style.ACTION_HATCHES[identity],
                     label=plot_style.ACTION_NAMES[identity])
            bottom += values
        for index, cell in enumerate(selected):
            axis.text(index, 8.2, f"{cell['completed']}/{cell['episodes']}",
                      ha="center", fontsize=8)
        axis.set(title=plot_style.MODEL_NAMES[model], xlabel="Deadline (s)",
                 xticks=range(len(deadlines)), xticklabels=[f"{value:g}" for value in deadlines],
                 ylim=(0, 9))
    axes[0, 0].set_ylabel("Mean executed actions per completed drain")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, .17, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"deadline_action_mix.{suffix}")
    plt.close(figure)
    vectors = {(cell["model"], cell["deadline_s"]): tuple(cell[column] for column in columns)
               for cell in cells if cell["completed"]}
    changes = {model: len({value for (name, _), value in vectors.items() if name == model}) > 1
               for model in models}
    differences = {str(deadline): len({value for (_, value_deadline), value in vectors.items()
                                     if value_deadline == deadline}) > 1 for deadline in deadlines}
    report = {"within_model_changes": changes, "between_model_differences": differences,
              "all_episodes_completed": all(cell["completed"] == cell["episodes"] for cell in cells),
              "any_action_mix_change": any(changes.values()) or any(differences.values()),
              "cells": cells}
    (out / "action_mix_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def reduce(run_roots: list[Path], out: Path,
           expected: set[tuple[str, str]] | None = None) -> dict:
    rows = _rows(run_roots)
    arms = sorted({(row["model"], row["hardware"]) for row in rows})
    if set(arms) != (expected or {
            (model, hardware) for hardware, models in MODELS.items()
            for model in models}):
        raise ValueError("drain reduction has an incomplete arm set")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "drain_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_style.apply()
    figure, axis = plt.subplots()
    for model, hardware in arms:
        selected = [row for row in rows
                    if (row["model"], row["hardware"]) == (model, hardware)]
        values = sorted(float(row["time_to_target_s"]) for row in selected
                        if row["status"] == "complete"
                        and row["time_to_target_s"])
        axis.step([0, *values], [0, *[i / len(selected)
                  for i in range(1, len(values) + 1)]], where="post",
                  color=plot_style.MODEL_COLORS[model],
                  linestyle=plot_style.AGENTIC_HARDWARE_LINESTYLES[
                      hardware.lower()],
                  label=f"{plot_style.MODEL_NAMES[model]} / {hardware}")
    axis.axvline(30, color=plot_style.SLO_COLOR,
                 linestyle=plot_style.SLO_LINESTYLE)
    axis.set(xlabel="Time to full-drain attainment (s)",
             ylabel="Cumulative fraction of episodes", ylim=(0, 1.02))
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"drain_attainment_ecdf.{suffix}")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=plot_style.WIDE_FIGSIZE)
    bottom = [0.] * len(arms)
    labels = [f"{plot_style.MODEL_NAMES[model]}\n{hardware}"
              for model, hardware in arms]
    for action, columns in ACTIONS.items():
        values = []
        for arm in arms:
            known = [row for row in rows
                     if (row["model"], row["hardware"]) == arm
                     and all(row[column] for column in columns)]
            values.append(sum(int(row[column]) for row in known
                              for column in columns) / len(known)
                          if known else 0)
        axis.bar(labels, values, bottom=bottom,
                 label=plot_style.ACTION_NAMES[action],
                 color=plot_style.ACTION_COLORS[action],
                 hatch=plot_style.ACTION_HATCHES[action])
        bottom = [left + value for left, value in zip(bottom, values)]
    axis.set(ylabel="Mean actions per planned eight-session drain", ylim=(0, 8))
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(out / f"drain_action_mix.{suffix}")
    plt.close(figure)

    summary = {f"{model} / {hardware}": {
        "episodes": len(selected := [row for row in rows if
                        (row["model"], row["hardware"]) == (model, hardware)]),
        "completed_episodes": sum(row["status"] == "complete"
                                  for row in selected),
        "failed_episodes": sum(row["status"] == "failed" for row in selected),
        "action_mix_episodes": sum(
            all(row[column] for columns in ACTIONS.values()
                for column in columns) for row in selected),
        "retried_episodes": sum(int(row.get("attempt", 1)) > 1
                                for row in selected),
        "excluded_attempts": sum(int(row.get("excluded_attempts", 0))
                                 for row in selected),
        "drain_deadline_attainment": sum(
            row["status"] == "complete" and row["target_met"] == "True"
            for row in selected) / len(selected),
        "modeled_power_deadline_attainment": sum(
            row["status"] == "complete"
            and row["modeled_power_deadline_met"] == "True" for row in selected)
        / len(selected),
    } for model, hardware in arms}
    (out / "drain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if all("deadline_s" in row for row in rows) \
            and len({row["deadline_s"] for row in rows}) > 1:
        deadline_action_mix(rows, out)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("a100", "h100", "h100-sweep"):
        command = sub.add_parser(name)
        command.add_argument("--profiles" if name != "h100" else "--profile",
                             type=Path, nargs=3 if name != "h100" else None,
                             required=True)
        if name == "h100-sweep":
            command.add_argument("--deadlines-s", type=float, nargs="+", required=True)
        command.add_argument("--cluster", type=Path, required=True)
        command.add_argument("--calibration", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--ssh-key", type=Path,
                             default=Path("~/.ssh/azrs").expanduser())
    command = sub.add_parser("reduce")
    command.add_argument("--run-root", type=Path, action="append", required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("freeze-network-profile")
    command.add_argument("--timing-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "reduce":
        result = reduce(args.run_root, args.out)
    elif args.command == "freeze-network-profile":
        result = freeze_network_profile(args.timing_root, args.out)
    else:
        profiles = args.profiles if args.command != "h100" else [args.profile]
        result = run(args.command.split("-")[0].upper(), profiles, args.cluster,
                     args.calibration, args.manifest, args.run_root, args.ssh_key,
                     source_service=args.command == "h100-sweep",
                     deadlines_s=getattr(args, "deadlines_s", None))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
