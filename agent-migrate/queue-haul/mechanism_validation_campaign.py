"""Prepare and reduce the current-stack two-A100 mechanism campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import migration_profiler as profiler


DEFAULT_MANIFEST = Path("queue-haul/outputs/coding-manifest.json")
DEFAULT_SESSION = "claude:d822a508-dc39-797d-a37d-fa6552fdb8bb"
SCHEDULE = tuple({"at_s": second, "append_tokens": 512}
                 for second in (2, 4, 6, 8))


def make_plan(manifest_path: Path, repeats: int = 5, seed: int = 1,
              session_id: str = DEFAULT_SESSION) -> dict:
    plan = profiler.make_plan(
        manifest_path, [28_000], [1], [10_000],
        ["kv_transfer", "replay"], ["one_turn"], repeats, seed,
        deadline_s=180, session_ids=[session_id], activity_tokens=[2048],
        serving_concurrency=[1], final_state="awake",
    )
    for scenario in plan["scenarios"]:
        scenario.update(
            campaign="mechanism_validation", split="validation",
            request_schedule=list(SCHEDULE), activity_tokens=2048,
        )
        scenario["sessions"][0]["initial_tokens"] = 28_000
    profiler.validate_plan(plan, json.loads(manifest_path.read_text()))
    return plan


def prepare(manifest: Path, out: Path, repeats: int = 5, seed: int = 1,
            session_id: str = DEFAULT_SESSION) -> dict:
    plan = make_plan(manifest, repeats, seed, session_id)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", plan)
    run = out / "run.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
: "${QH_MECHANISM_RUN_ROOT:?set QH_MECHANISM_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
status=0
command=(uv run python queue-haul/migration_profiler.py run --plan "$script_dir/plan.json" --run-root "$QH_MECHANISM_RUN_ROOT" --stack-scenarios 15)
[[ -z "${QH_RESUME_FROM_GIT_SHA:-}" ]] || command+=(--resume-from-git-sha "$QH_RESUME_FROM_GIT_SHA")
"${command[@]}" || status=$?
[[ -f "$QH_MECHANISM_RUN_ROOT/plan.json" ]] || exit "$status"
uv run python queue-haul/mechanism_validation_campaign.py reduce --run-root "$QH_MECHANISM_RUN_ROOT" --out queue-haul/outputs/mechanism-validation
exit "$status"
""")
    run.chmod(0o755)
    (out / "run.sbatch").write_text("""#!/bin/bash
#SBATCH --job-name=qh-mechanisms
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=mechanism-validation-%j.out
#SBATCH --error=mechanism-validation-%j.err
set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
script_path="$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for (i=1;i<=NF;i++) if ($i ~ /^Command=/) {sub(/^Command=/,"",$i); print $i}}')"
script_dir="$(dirname "$script_path")"
export QH_MECHANISM_RUN_ROOT="${QH_MECHANISM_RUN_ROOT:-/scratch/users/$USER/qh-mechanism-validation-20260731}"
export QH_APPTAINER_IMAGE="${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}"
test "$(sha256sum "$QH_APPTAINER_IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
port_file="$QH_MECHANISM_RUN_ROOT/port-offset"
if [[ -f "$port_file" ]]; then
  read -r QH_PORT_OFFSET < "$port_file"
else
  QH_PORT_OFFSET="${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}"
  mkdir -p "$QH_MECHANISM_RUN_ROOT"
  printf '%s\n' "$QH_PORT_OFFSET" > "$port_file"
fi
export QH_PORT_OFFSET
bash "$script_dir/run.sh"
""")
    return plan


def representatives(rows: list[dict[str, str]], repeats: int) -> dict[str, str]:
    selected = {}
    for method in ("kv_transfer", "replay"):
        candidates = [row for row in rows if row["method"] == method]
        if len(candidates) != repeats or any(row["success"] != "True" for row in candidates):
            raise RuntimeError(f"expected {repeats} successful {method} trials")
        median = statistics.median(float(row["request_wait_s"])
                                   for row in candidates)
        selected[method] = min(
            candidates,
            key=lambda row: (abs(float(row["request_wait_s"]) - median),
                             row["scenario_id"]),
        )["scenario_id"]
    return selected


def reduce(run_root: Path, out: Path) -> dict[str, str]:
    import plot_testbed_kv_timeline as timeline

    profiler.reduce_run(run_root)
    with (run_root / "scenarios.csv").open() as stream:
        scenarios = list(csv.DictReader(stream))
    if any(row["status"] != "complete" for row in scenarios):
        raise RuntimeError("every mechanism-validation scenario must complete")
    with (run_root / "migrations.csv").open() as stream:
        migrations = list(csv.DictReader(stream))
    plan = json.loads((run_root / "plan.json").read_text())
    repeats = len({row["repeat"] for row in plan["scenarios"]
                   if row["kind"] == "migration"})
    selected = representatives(migrations, repeats)
    for scenario_id in selected.values():
        timeline.write(run_root, scenario_id, out)
    profiler.write_csv(out / "representative_traces.csv", [
        {"method": method, "scenario_id": scenario_id,
         "selection": "nearest per-method median request_wait_s",
         "run_root": str(run_root)}
        for method, scenario_id in selected.items()
    ])
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--repeats", type=int, default=5)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument("--session-id", default=DEFAULT_SESSION)
    command = sub.add_parser("reduce")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.manifest, args.out, args.repeats, args.seed,
                args.session_id)
    else:
        reduce(args.run_root, args.out)


if __name__ == "__main__":
    main()
