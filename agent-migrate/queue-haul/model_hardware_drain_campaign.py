"""Run and reduce the matched eight-session Azure drain campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style
from profiles import ModelProfile


ROOT = Path(__file__).resolve().parent
MODELS = {
    "A100": set(plot_style.MODELS),
    "H100": {"openai/gpt-oss-20b"},
}
ACTIONS = {"replay": ("east_replay", "germany_replay"),
           "kv_transfer": ("east_kv_transfer", "germany_kv_transfer")}


def _gated_profile(path: Path, hardware: str) -> ModelProfile:
    profile = ModelProfile.load(path)
    gate = json.loads(path.with_suffix(".gate.json").read_text())
    if hardware.lower() not in profile.hardware.lower() \
            or profile.precision.lower() not in {"bf16", "bfloat16"} \
            or profile.tensor_parallel != 1 or profile.kv_geometry is None \
            or gate.get("schema") != "queue-haul-model-architecture-gate-v1" \
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
        ssh_key: Path) -> dict:
    cluster, calibration, manifest, run_root, ssh_key = (
        path.resolve() for path in
        (cluster, calibration, manifest, run_root, ssh_key))
    loaded = sorted((_profile(path, hardware) for path in profiles),
                    key=lambda row: row[0].model)
    if {profile.model for profile, _ in loaded} != MODELS[hardware] \
            or len(loaded) != len(MODELS[hardware]):
        raise ValueError(f"{hardware} requires exactly {sorted(MODELS[hardware])}")
    prepared = []
    for profile, relative in loaded:
        slug = re.sub(r"[^a-z0-9]+", "-", profile.model.lower()).strip("-")
        plan, arm = run_root / "plans" / f"{slug}.json", run_root / "arms" / slug
        source, snapshot = ROOT / relative, arm / "profile.json"
        _snapshot(source, snapshot)
        _snapshot(source.with_suffix(".gate.json"),
                  snapshot.with_suffix(".gate.json"))
        env = {**os.environ, "QH_MODEL_PROFILE": str(relative),
               "QH_RUNTIME": "native", "QH_LMCACHE_MODE": "mp"}
        subprocess.run([
            sys.executable, str(ROOT / "network_campaign.py"), "prepare",
            "--design", "drain", "--cluster", str(cluster),
            "--calibration", str(calibration), "--manifest", str(manifest),
            "--out", str(plan),
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
        (model, hardware) for model in MODELS[hardware]})


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
            if plan.get("design") != "drain" or summary.get("expected") != 50 \
                    or summary.get("completed", 0) + summary.get("failed", 0) != 50 \
                    or summary.get("missing") or summary.get("invalid_evidence") \
                    or metadata.get("plan_sha256") != plan_sha \
                    or profile_sha != plan["model_profile"]["sha256"] \
                    or runtime.get("QH_RUNTIME") != "native" \
                    or runtime.get("QH_LMCACHE_MODE") != "mp" \
                    or len(rows) != 50 or key in arms \
                    or sum(row["status"] == "complete" for row in rows) \
                    != summary.get("completed") \
                    or any(row["status"] not in {"complete", "failed"}
                           or int(row.get("attempt", 1)) != 1
                           or int(row.get("excluded_attempts", 0))
                           for row in rows):
                raise ValueError(f"invalid drain arm: {arm}")
            arms.add(key)
            matrices.add((plan["manifest"]["sha256"], tuple(sorted(
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
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("a100", "h100"):
        command = sub.add_parser(name)
        command.add_argument("--profiles" if name == "a100" else "--profile",
                             type=Path, nargs=3 if name == "a100" else None,
                             required=True)
        command.add_argument("--cluster", type=Path, required=True)
        command.add_argument("--calibration", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--ssh-key", type=Path,
                             default=Path("~/.ssh/azrs").expanduser())
    command = sub.add_parser("reduce")
    command.add_argument("--run-root", type=Path, action="append", required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "reduce":
        result = reduce(args.run_root, args.out)
    else:
        profiles = args.profiles if args.command == "a100" else [args.profile]
        result = run(args.command.upper(), profiles, args.cluster,
                     args.calibration, args.manifest, args.run_root, args.ssh_key)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
