"""Run randomized migrations against five-second source-power windows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import migration_profiler as profiler
import plot_hardware_power_parity as parity
import policy_hardware_campaign as policy
from profiles import ModelProfile


ROOT = Path(__file__).parent
POLICIES = parity.METHODS
PRE_LOAD = .4
WINDOW_S = SETTLE_S = 5


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT.parent / path


def migration_counts(episodes: int, seed: int) -> list[int]:
    if episodes < 8:
        raise ValueError("at least eight episodes are required")
    counts = [index % 8 + 1 for index in range(episodes)]
    random.Random(seed).shuffle(counts)
    return counts


def make_plan(source_path: Path, episodes: int = 50, seed: int = 20260812) -> dict:
    source = json.loads(source_path.read_text())
    manifest_path, model_path = _path(source["manifest"]["path"]), \
        _path(source["model_profile"]["path"])
    if profiler.file_hash(manifest_path) != source["manifest"]["sha256"] \
            or profiler.file_hash(model_path) != source["model_profile"]["sha256"]:
        raise RuntimeError("source plan inputs changed")
    manifest, model = json.loads(manifest_path.read_text()), ModelProfile.load(model_path)
    controls = [row for row in source["scenarios"] if row["policy"] == "control"]
    if episodes > len(controls):
        raise ValueError("not enough frozen source episodes")
    rng, scenarios = random.Random(seed), []
    selected = rng.sample(controls, episodes)
    for repeat, (control, count) in enumerate(zip(selected, migration_counts(episodes, seed))):
        problem, routes = policy._problem(
            model, control["sessions"], 10_000, control["required_deadline_s"],
        )
        match_id = f"power-{repeat:02d}-{control['sample_id']}-n{count}"
        block = []
        for name in POLICIES:
            planned = policy._moves(
                name, problem, routes, model,
                profiler.stable_seed(seed, repeat, name),
            )[:count]
            sessions = {row["session_id"]: row for row in control["sessions"]}
            moves = [{**sessions[row["session_id"]], **row} for row in planned]
            scenario_id = "pv-" + profiler.object_hash([match_id, name, moves])[:16]
            post_load = PRE_LOAD * (1 - count / len(sessions))
            block.append({
                **control, "scenario_id": scenario_id, "match_id": match_id,
                "episode": repeat, "repeat": repeat, "policy": name,
                "condition": f"{control['condition']}-n{count}",
                "kind": "migration", "method": moves[0]["method"]
                if len({row["method"] for row in moves}) == 1 else "mixed",
                "moves": moves, "allow_partial_moves": True,
                "bandwidth_mbps": 10_000, "concurrency": count,
                "move_concurrency": count, "serving_concurrency": count,
                "deadline_s": 600, "verify_continuations": False,
                "sample_power": True, "power_interval_s": .25,
                "power_target_fraction": count / len(sessions),
                "power_validation": {
                    "pre_load": PRE_LOAD, "post_load": post_load,
                    "prefill_tps": model.case().F, "settle_s": SETTLE_S,
                    "window_s": WINDOW_S, "sample_interval_s": .25,
                },
            })
        rng.shuffle(block)
        scenarios.extend(block)
    output = {
        "schema": profiler.PLAN_SCHEMA, "campaign": "power_parity_random",
        "seed": seed, "episodes": episodes, "policies": list(POLICIES),
        "sessions_per_episode": 8, "manifest": source["manifest"],
        "model_profile": source["model_profile"], "scenarios": scenarios,
        "source_plan": {"path": policy._portable_path(source_path),
                        "sha256": profiler.file_hash(source_path)},
        "power_validation": {"pre_load": PRE_LOAD, "settle_s": SETTLE_S,
                             "window_s": WINDOW_S},
    }
    profiler.validate_plan(output, manifest)
    validate_plan(output)
    return output


def validate_plan(plan: dict) -> None:
    policies = set(POLICIES)
    if set(plan["policies"]) != policies \
            or len(plan["scenarios"]) != plan["episodes"] * len(policies):
        raise ValueError("power-parity matrix is incomplete")
    for repeat in range(plan["episodes"]):
        rows = [row for row in plan["scenarios"] if row["repeat"] == repeat]
        if {row["policy"] for row in rows} != policies \
                or len({(row["match_id"], len(row["moves"])) for row in rows}) != 1:
            raise ValueError("power-parity arms are not matched")
    counts = [len(next(row for row in plan["scenarios"]
                       if row["repeat"] == repeat)["moves"])
              for repeat in range(plan["episodes"])]
    if set(counts) != set(range(1, 9)):
        raise ValueError("power-parity plan must cover every migration count")


def prepare(source: Path, out: Path, episodes=50, seed=20260812) -> dict:
    plan = make_plan(source, episodes, seed)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", plan)
    run = out / "run.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
: "${QH_POWER_PARITY_RUN_ROOT:?set QH_POWER_PARITY_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
uv run python queue-haul/power_parity_experiment.py run --plan "$script_dir/plan.json" --run-root "$QH_POWER_PARITY_RUN_ROOT" --stack-scenarios 28 --fail-fast
uv run python queue-haul/power_parity_experiment.py reduce --run-root "$QH_POWER_PARITY_RUN_ROOT"
""")
    run.chmod(0o755)
    (out / "run.sbatch").write_text("""#!/bin/bash
#SBATCH --job-name=qh-power-parity
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=power-parity-%j.out
#SBATCH --error=power-parity-%j.err
set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
script_path="$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for (i=1;i<=NF;i++) if ($i ~ /^Command=/) {sub(/^Command=/,"",$i); print $i}}')"
script_dir="$(dirname "$script_path")"
export QH_POWER_PARITY_RUN_ROOT="${QH_POWER_PARITY_RUN_ROOT:-/scratch/users/$USER/qh-power-parity-random-20260812}"
export QH_APPTAINER_IMAGE="${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}"
test "$(sha256sum "$QH_APPTAINER_IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
export QH_PORT_OFFSET="${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}"
bash "$script_dir/run.sh"
""")
    return plan


def _source_mean(path: Path, window: list[int]) -> float:
    with path.open() as handle:
        rows = [{"monotonic_ns": int(row["monotonic_ns"]),
                 "gpu": int(row["gpu"]), "power_w": float(row["power_w"])}
                for row in csv.DictReader(handle) if row["valid"] == "1"]
    if sorted({row["gpu"] for row in rows}) != [0, 1]:
        raise RuntimeError(f"expected source and destination GPU traces in {path}")
    source = sorted((row for row in rows if row["gpu"] == 0),
                    key=lambda row: row["monotonic_ns"])
    return parity._mean(source, *window)


def reduce(run_root: Path, out: Path | None = None) -> list[dict]:
    plan = json.loads((run_root / "plan.json").read_text())
    validate_plan(plan)
    model_path = _path(plan["model_profile"]["path"])
    if profiler.file_hash(model_path) != plan["model_profile"]["sha256"]:
        raise RuntimeError("model profile changed after planning")
    curve, rows = ModelProfile.load(model_path).case().power_curve, []
    for scenario in plan["scenarios"]:
        root = run_root / "scenarios" / scenario["scenario_id"]
        result = json.loads((root / "result.json").read_text())
        windows, spec = result.get("power_validation"), scenario["power_validation"]
        if result.get("status") != "complete" or not windows:
            raise RuntimeError(f"incomplete power validation {scenario['scenario_id']}")
        before = _source_mean(root / "power.csv", windows["pre_window_ns"])
        after = _source_mean(root / "power.csv", windows["post_window_ns"])
        expected = curve.power(spec["pre_load"]) - curve.power(spec["post_load"])
        rows.append({
            "scenario_id": scenario["scenario_id"], "match_id": scenario["match_id"],
            "method": scenario["policy"], "repeat": scenario["repeat"],
            "migrated_sessions": len(scenario["moves"]),
            "pre_load": spec["pre_load"], "post_load": spec["post_load"],
            "pre_window_power_w": before, "post_window_power_w": after,
            "requested_shed_w": expected, "predicted_shed_w": expected,
            "measured_shed_w": before - after,
        })
    normalized, scale = parity.normalize(rows)
    parity.write_plot(normalized, scale, out or run_root / "power_parity")
    return normalized


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        args = profiler.parse_args(argv)
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] \
            else args.extra_vllm_args
        profiler.run_plan(
            args.plan, args.run_root, profiler.b.config_from_args(args),
            args.allow_dirty, extra, args.resume_from_git_sha,
            args.power_state_cycles, args.power_state_window_s,
            args.node_power, args.fail_fast, args.stack_scenarios,
        )
        return
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--source-plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=50)
    command.add_argument("--seed", type=int, default=20260812)
    command = commands.add_parser("reduce")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.source_plan, args.out, args.episodes, args.seed)
    else:
        reduce(args.run_root, args.out)


if __name__ == "__main__":
    main()
