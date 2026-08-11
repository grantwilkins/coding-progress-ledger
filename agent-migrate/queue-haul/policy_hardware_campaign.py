"""Build and reduce a paired, ungated hardware policy campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import migration_profiler as profiler
import plot_style
from destination import dedicated_sink_architecture
from migration import ORDERED_EAGER_PARALLEL_V1
from planner import _duration, plan
from profiles import ModelProfile, WorkloadProfile
from simulate import ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession


ROOT = Path(__file__).parent
EXECUTION_CONTRACT = ORDERED_EAGER_PARALLEL_V1
DEFAULT_MANIFEST = Path("queue-haul/outputs/coding-manifest.json")
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
DEFAULT_WORKLOADS = tuple(ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
))
TOKEN_DISTRIBUTIONS = ("uniform_support", "uniform_range")
REQUIRED_DEADLINES_S = (30,)
CONTEXT_PACKS = {
    "tiny": (2048,) * 8,
    "small": (4096,) * 8,
    "medium": (8192,) * 8,
    "mixed": (2048, 4096, 4096, 8192, 8192, 12288, 12288, 14336),
    "large": (16384,) * 8,
}
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "isolated_fastest",
    "kv_only", "replay_only",
)
DEFAULT_POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "kv_only", "replay_only",
)
NETWORK_BASELINES = (
    "isolated_fastest", "queue_haul_power_blind",
    "queue_haul_deadline_blind",
)
RERUN_POLICIES = ("queue_haul", "greedy", *NETWORK_BASELINES)
DEADLINE_BLIND_HORIZON_S = 600
CDF_POLICIES = plot_style.POLICIES
TAB10_COLORS = CDF_COLORS = plot_style.POLICY_COLORS
PACKING_POLICIES = (
    "queue_haul", "greedy", "isolated_fastest", "kv_only", "replay_only",
)
LABELS = CDF_LABELS = plot_style.POLICY_NAMES
CDF_LINESTYLES = plot_style.POLICY_LINESTYLES
CDF_FIGSIZE = plot_style.COMPACT_FIGSIZE
plot_style.apply()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.parent.resolve()))
    except ValueError:
        return str(resolved)


def _problem(profile, sessions, bandwidth_mbps, deadline_s):
    case, n = profile.case(), len(sessions)
    load = .4 / n
    simulated = tuple(
        SimSession(
            row["session_id"], "source", row["initial_tokens"],
            case.F * load, 0, 2 * row["initial_tokens"],
        )
        for row in sessions
    )
    scenario = ExecutionScenario(
        deadline_s, deadline_s, case.power_curve.power(0), "awake", 0,
        (PowerNode("source-node", 1, True),
         PowerNode("destination-node", 1, False)),
        (ServingInstance("source", ("source-node",)),
         ServingInstance("destination", ("destination-node",))),
        simulated, (NetworkLink("link", bandwidth_mbps * 125_000),),
    )
    return scenario, {("source", "destination"): ("link",)}


def _ranked_moves(sessions, methods, scenario, profile, offset=0):
    case = profile.case()
    links = {row.link_id: row.bytes_per_s for row in scenario.links}
    choices = [min(
        (_duration(row, method, case, ("link",), links), row, method)
        for method in methods
    ) for row in sessions]
    return tuple(
        (row.session_id, method, order)
        for order, (_, row, method) in enumerate(
            sorted(choices, key=lambda choice: (choice[0], choice[1].session_id)),
            start=offset,
        )
    )


def _moves(policy, scenario, routes, profile, seed):
    sessions = list(scenario.sessions)
    if policy in {"queue_haul", "greedy", "greedy_lagrangian", "random"}:
        solver = {"queue_haul": "lp_work_first", "greedy": "greedy",
                  "greedy_lagrangian": "greedy_lagrangian",
                  "random": "random"}[policy]
        destination = dedicated_sink_architecture(
            profile, "destination", ("link",),
        ) if policy == "greedy_lagrangian" else None
        result = plan(
            scenario, profile, routes, solver, seed=seed, destination=destination,
        )
        moves = result.moves
    else:
        fixed_method = None if policy == "isolated_fastest" else {
            "kv_only": "kv_transfer", "replay_only": "replay",
        }[policy]
        moves = _ranked_moves(
            sessions, (fixed_method,) if fixed_method
            else ("replay", "kv_transfer"), scenario, profile,
        )
    admitted = {
        move.session_id if hasattr(move, "session_id") else move[0]
        for move in moves
    }
    moves = tuple(moves) + _ranked_moves(
        [row for row in sessions if row.session_id not in admitted],
        ("replay", "kv_transfer"), scenario, profile, len(moves),
    )
    normalized = [
        {
            "session_id": move.session_id, "method": move.method,
            "order": move.order,
            "planned_rate_limit_bytes_per_s":
                move.rate_limit_bytes_per_s,
            "planned_quiesce_s": move.quiesce_s,
            "deadline_admitted": move.session_id in admitted,
        } if hasattr(move, "session_id") else {
            "session_id": move[0], "method": move[1], "order": move[2],
            "planned_rate_limit_bytes_per_s": None,
            "planned_quiesce_s": None,
            "deadline_admitted": move[0] in admitted,
        }
        for move in moves
    ]
    if {row["session_id"] for row in normalized} \
            != {row.session_id for row in sessions}:
        raise RuntimeError(f"{policy} did not plan the complete episode")
    return normalized


def _context_tokens(workload, distribution, count, rng):
    support = tuple(sorted({
        round(row.context_tokens / 256) * 256 for row in workload.records
    }))
    if distribution == "uniform_support":
        return [rng.choice(support) for _ in range(count)]
    if distribution == "uniform_range":
        return [rng.randrange(support[0], support[-1] + 256, 256)
                for _ in range(count)]
    raise ValueError(f"unknown token distribution {distribution}")


def make_plan(manifest_path: Path, model_path: Path = DEFAULT_MODEL,
              episodes: int = 2, sessions: int = 8, seed: int = 0,
              bandwidth_mbps: float = 10_000, deadline_s: float = 180,
              workload_paths=(DEFAULT_WORKLOADS[0],),
              token_distributions=("uniform_support",),
              required_deadlines_s=REQUIRED_DEADLINES_S,
              bandwidths_mbps=None, context_packs=(),
              policies=DEFAULT_POLICIES) -> dict:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    profile = ModelProfile.load(model_path)
    pack_names = tuple(context_packs)
    if len(set(pack_names)) != len(pack_names) \
            or not set(pack_names) <= CONTEXT_PACKS.keys() \
            or pack_names and any(len(CONTEXT_PACKS[name]) != sessions
                                  for name in pack_names):
        raise ValueError("invalid context packs for campaign width")
    workloads = [] if pack_names else [
        WorkloadProfile.load(path) for path in workload_paths
    ]
    bandwidths = (bandwidth_mbps,) if bandwidths_mbps is None \
        else tuple(bandwidths_mbps)
    policies = tuple(policies)
    if episodes < 1 or not 1 <= sessions <= len(manifest["sessions"]) \
            or not bandwidths or min(bandwidths) <= 0 \
            or len(set(bandwidths)) != len(bandwidths) \
            or deadline_s <= profile.power_window_s \
            or not (workloads or pack_names) \
            or not token_distributions or not required_deadlines_s \
            or min(required_deadlines_s) <= profile.power_window_s \
            or not policies or len(set(policies)) != len(policies) \
            or not set(policies) <= set(POLICIES):
        raise ValueError("invalid policy campaign dimensions")
    replay_contexts = profile.case().replay.by_concurrency[1][0]

    rng, scenarios = random.Random(seed), []
    available = sorted(manifest["sessions"], key=lambda row: row["id"])
    sources = [(
        workload.profile_id, workload.records[0].job_type, distribution, workload
    ) for workload in workloads for distribution in token_distributions]
    sources.extend((name, name, "fixed", CONTEXT_PACKS[name])
                   for name in pack_names)
    cells = [
        (profile_id, job_type, distribution, source, bandwidth,
         required_deadline, repeat)
        for profile_id, job_type, distribution, source in sources
        for bandwidth in bandwidths
        for required_deadline in required_deadlines_s
        for repeat in range(episodes)
    ]
    for episode, (profile_id, job_type, distribution, source, bandwidth,
                  required_deadline, repeat) in enumerate(cells):
        sample_rng = random.Random(profiler.stable_seed(
            seed, profile_id, distribution, repeat
        ))
        chosen = sample_rng.sample(available, sessions)
        contexts = list(source) if distribution == "fixed" else _context_tokens(
            source, distribution, sessions, sample_rng
        )
        if min(contexts) < replay_contexts[0] \
                or max(contexts) > replay_contexts[-1]:
            raise ValueError("workload contexts exceed measured replay range")
        session_rows = [{
            "session_id": row["id"], "job_class": row["job_class"],
            "turn_index": 0,
            "initial_tokens": contexts[order],
            "order": order,
        } for order, row in enumerate(chosen)]
        condition = f"{job_type}-{distribution}-{bandwidth:g}mbps-" \
            f"{required_deadline:g}s"
        sample_id = profiler.object_hash(
            [seed, profile_id, distribution, repeat, session_rows]
        )[:16]
        match_id = profiler.object_hash(
            [sample_id, bandwidth, required_deadline]
        )[:16]
        base = {
            "match_id": match_id, "sample_id": sample_id, "episode": episode,
            "campaign": "policy_hardware", "split": "measurement",
            "condition": condition, "context_profile": job_type,
            "token_distribution": distribution,
            "required_deadline_s": required_deadline,
            "power_target_fraction": 1.0,
            "activity": "none", "activity_tokens": 0,
            "request_schedule": [], "repeat": repeat,
            "deadline_s": deadline_s, "sessions": session_rows,
            "serving_concurrency": 1, "concurrency": sessions,
            "move_concurrency": sessions, "copy_policy": "initial_final",
            "final_state": "awake", "bandwidth_mbps": bandwidth,
        }
        scenarios.append({
            **base, "scenario_id": f"c-{match_id}", "kind": "control",
            "method": "replay", "policy": "control", "moves": [],
        })
        problem, routes = _problem(
            profile, session_rows, bandwidth, required_deadline
        )
        for policy in policies:
            moves = _moves(
                policy, problem, routes, profile,
                profiler.stable_seed(seed, episode, policy),
            )
            move_rows = [{
                **next(row for row in session_rows
                       if row["session_id"] == move["session_id"]),
                **move,
            } for move in moves]
            scenario_id = profiler.object_hash([match_id, policy, move_rows])[:16]
            scenarios.append({
                **base, "scenario_id": f"p-{scenario_id}",
                "kind": "migration", "method":
                    move_rows[0]["method"]
                    if len({row["method"] for row in move_rows}) == 1
                    else "mixed",
                "policy": policy, "moves": move_rows,
            })
    blocks = [[row for row in scenarios if row["episode"] == episode]
              for episode in range(len(cells))]
    for block in blocks:
        rng.shuffle(block)
    rng.shuffle(blocks)
    blocks.sort(key=lambda block: block[0]["bandwidth_mbps"])
    scenarios = [row for block in blocks for row in block]
    output = {
        "schema": profiler.PLAN_SCHEMA,
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "seed": seed, "campaign": "policy_hardware",
        "execution_contract": ORDERED_EAGER_PARALLEL_V1,
        "model_profile": {
            "path": _portable_path(model_path),
            "sha256": profiler.file_hash(model_path),
            "profile_id": profile.profile_id,
        },
        "policies": list(policies), "episodes": len(cells),
        "episodes_per_cell": episodes,
        "workload_profiles": [] if pack_names else [
            {"path": _portable_path(path), "sha256": profiler.file_hash(path)}
            for path in workload_paths
        ],
        "token_distributions": ["fixed"] if pack_names
            else list(token_distributions),
        "context_packs": {name: list(CONTEXT_PACKS[name])
                          for name in pack_names},
        "bandwidths_mbps": list(bandwidths),
        "required_deadlines_s": list(required_deadlines_s),
        "power_target_fraction": 1.0,
        "sessions_per_episode": sessions, "scenarios": scenarios,
    }
    profiler.validate_plan(output, manifest)
    validate_policy_plan(output)
    return output


def validate_policy_plan(plan_: dict) -> None:
    if plan_.get("execution_contract") != ORDERED_EAGER_PARALLEL_V1:
        raise ValueError("unsupported policy hardware execution contract")
    policies = set(plan_["policies"])
    episode_order = [row["episode"] for row in plan_["scenarios"]]
    if sum(
        index == 0 or episode != episode_order[index - 1]
        for index, episode in enumerate(episode_order)
    ) != plan_["episodes"]:
        raise ValueError("matched episodes must form contiguous blocks")
    for episode in range(plan_["episodes"]):
        rows = [row for row in plan_["scenarios"]
                if row["episode"] == episode]
        if {row["policy"] for row in rows} != policies | {"control"}:
            raise ValueError("every episode must contain every policy and one control")
        if any(
            row["kind"] == "migration"
            and (row["move_concurrency"] != len(row["sessions"])
                 or sorted(move["order"] for move in row["moves"])
                 != list(range(len(row["moves"]))))
            for row in rows
        ):
            raise ValueError("policy migrations must launch the complete episode")
        signatures = {
            tuple(sorted(
                (row["session_id"], row["initial_tokens"])
                for row in scenario["sessions"]
            ))
            for scenario in rows
        }
        if len(signatures) != 1 or any(
            scenario["kind"] == "migration"
            and {row["session_id"] for row in scenario["moves"]}
            != {row["session_id"] for row in scenario["sessions"]}
            for scenario in rows
        ):
            raise ValueError("policies must consume the same complete episode")


def matched_baseline_plan(source_path: Path, policies=NETWORK_BASELINES,
                          model_path: Path | None = None,
                          condition_shard=None) -> dict:
    source = json.loads(source_path.read_text())
    manifest_path = Path(source["manifest"]["path"])
    source_model_path = Path(source["model_profile"]["path"])
    manifest_path = manifest_path if manifest_path.is_absolute() \
        else ROOT.parent / manifest_path
    source_model_path = source_model_path if source_model_path.is_absolute() \
        else ROOT.parent / source_model_path
    model_path = model_path or source_model_path
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_plan(source, manifest)
    validate_policy_plan(source)
    if profiler.file_hash(manifest_path) != source["manifest"]["sha256"] \
            or profiler.file_hash(model_path) != source["model_profile"]["sha256"]:
        raise ValueError("source plan inputs changed")
    policies = tuple(policies)
    if not policies or len(set(policies)) != len(policies) \
            or not set(policies) <= set(RERUN_POLICIES):
        raise ValueError("invalid matched baseline policies")
    profile, rng, scenarios = ModelProfile.load(model_path), \
        random.Random(source["seed"]), []
    episode_order = list(dict.fromkeys(
        row["episode"] for row in source["scenarios"]
    ))
    if condition_shard:
        index, count = condition_shard
        conditions = sorted({row["condition"] for row in source["scenarios"]})
        if not 0 <= index < count or not conditions[index::count]:
            raise ValueError("invalid matched condition shard")
        selected = set(conditions[index::count])
        episode_order = [source_episode for source_episode in episode_order
                         if next(row["condition"] for row in source["scenarios"]
                                 if row["episode"] == source_episode) in selected]
    for episode, source_episode in enumerate(episode_order):
        rows = [row for row in source["scenarios"]
                if row["episode"] == source_episode]
        control = {**next(row for row in rows if row["policy"] == "control"),
                   "episode": episode}
        block = [control]
        problem, routes = _problem(
            profile, control["sessions"], control["bandwidth_mbps"],
            control["required_deadline_s"],
        )
        for policy in policies:
            moves = _moves(
                policy, problem, routes, profile,
                profiler.stable_seed(source["seed"], source_episode, policy),
            )
            move_rows = [{
                **next(row for row in control["sessions"]
                       if row["session_id"] == move["session_id"]),
                **move,
            } for move in moves]
            scenario_id = profiler.object_hash([
                control["match_id"], policy, move_rows,
            ])[:16]
            block.append({
                **control, "scenario_id": f"p-{scenario_id}",
                "kind": "migration", "method": move_rows[0]["method"]
                if len({row["method"] for row in move_rows}) == 1 else "mixed",
                "policy": policy, "moves": move_rows,
            })
        rng.shuffle(block)
        scenarios.extend(block)
    output = {
        **source, "episodes": len(episode_order), "policies": list(policies),
        "scenarios": scenarios,
        "model_profile": {
            **source["model_profile"], "path": _portable_path(model_path),
        },
        "baseline_source_plan": {
            "path": _portable_path(source_path),
            "sha256": profiler.file_hash(source_path),
        },
        **({"baseline_condition_shard": list(condition_shard)}
           if condition_shard else {}),
    }
    profiler.validate_plan(output, manifest)
    validate_policy_plan(output)
    return output


def _prepare(plan_: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    profiler.write_json(plan_path, plan_)
    job = out / "run.sh"
    job.write_text("""#!/usr/bin/env bash
set -euo pipefail
: "${QH_POLICY_RUN_ROOT:?set QH_POLICY_RUN_ROOT}"
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
status=0
run=(uv run python queue-haul/migration_profiler.py run --plan "$script_dir/plan.json" --run-root "$QH_POLICY_RUN_ROOT" --stack-scenarios 30)
[[ -z "${QH_RESUME_FROM_GIT_SHA:-}" ]] || run+=(--resume-from-git-sha "$QH_RESUME_FROM_GIT_SHA")
"${run[@]}" || status=$?
[[ -f "$QH_POLICY_RUN_ROOT/plan.json" ]] || exit "$status"
uv run python queue-haul/policy_hardware_campaign.py reduce \
  --run-root "$QH_POLICY_RUN_ROOT"
exit "$status"
""")
    job.chmod(0o755)
    (out / "run.sbatch").write_text("""#!/bin/bash
#SBATCH --job-name=qh-policy-parallel
#SBATCH --partition=ramr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:A100_SXM4&GPU_MEM:80GB
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=policy-hardware-%j.out
#SBATCH --error=policy-hardware-%j.err
set -euo pipefail
export LC_ALL=C
module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4
script_path="$(scontrol show job "$SLURM_JOB_ID" -o | awk '{for (i=1;i<=NF;i++) if ($i ~ /^Command=/) {sub(/^Command=/,"",$i); print $i}}')"
script_dir="$(dirname "$script_path")"
export QH_POLICY_RUN_ROOT="${QH_POLICY_RUN_ROOT:-/scratch/users/$USER/qh-policy-run-width8-pilot}"
export QH_APPTAINER_IMAGE="${QH_APPTAINER_IMAGE:-/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif}"
test "$(sha256sum "$QH_APPTAINER_IMAGE" | cut -d' ' -f1)" = 50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab
export QH_PORT_OFFSET="${QH_PORT_OFFSET:-$((SLURM_JOB_ID % 40000 + 1000))}"
bash "$script_dir/run.sh"
""")
    return plan_


def prepare(manifest: Path, out: Path, **kwargs) -> dict:
    return _prepare(make_plan(manifest, **kwargs), out)


def prepare_baselines(source_plan: Path, out: Path,
                      policies=NETWORK_BASELINES,
                      model_path: Path | None = None,
                      condition_shard=None) -> dict:
    return _prepare(matched_baseline_plan(
        source_plan, policies, model_path, condition_shard), out)


def _time(start, end):
    return (end - start) / 1e9


def _threshold(values, planned, fraction):
    rank = math.ceil(planned * fraction)
    return sorted(values)[rank - 1] if len(values) >= rank else None


def deadline_attainment(commits, sessions, deadlines, power_curve,
                         power_window_s):
    before, target = power_curve.power(.4), power_curve.power(.4) \
        - power_curve.power(0)
    output = []
    for deadline in deadlines:
        start, completed, cursor, area = deadline - power_window_s, 0, \
            deadline - power_window_s, 0
        for commit in sorted(commits):
            if commit <= start:
                completed += 1
            elif commit <= deadline:
                area += (commit - cursor) * power_curve.power(
                    .4 * (1 - completed / sessions)
                )
                cursor, completed = commit, completed + 1
        area += (deadline - cursor) * power_curve.power(
            .4 * (1 - completed / sessions)
        )
        output.append({
            "required_deadline_s": deadline,
            "committed_by_deadline":
                sum(value <= deadline for value in commits),
            "committed_before_power_window":
                sum(value <= start for value in commits),
            "power_attainment_fraction": min(
                1, max(0, (before - area / power_window_s) / target)
            ),
        })
    return output


def reduce_run(run_root: Path, out: Path | None = None):
    plan_ = json.loads((run_root / "plan.json").read_text())
    validate_policy_plan(plan_)
    power_curve, power_window_s = None, 0
    if plan_.get("power_target_fraction"):
        model_path = ROOT.parent / plan_["model_profile"]["path"]
        if profiler.file_hash(model_path) != plan_["model_profile"]["sha256"]:
            raise RuntimeError("model profile changed after planning")
        model = ModelProfile.load(model_path)
        power_curve, power_window_s = model.case().power_curve, model.power_window_s
    out = out or run_root
    controls = {}
    for scenario in plan_["scenarios"]:
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        if scenario["policy"] != "control" or not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            controls[(scenario["match_id"], result.get("allocation_id"))] = {
                row["session_id"]: _time(row["start_ns"], row["first_byte_ns"])
                for row in result["continuations"]
            }

    migrations, summaries, attainment = [], [], []
    for scenario in plan_["scenarios"]:
        if scenario["policy"] == "control":
            continue
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        raw = result.get("migrations", []) if result.get("status") == "complete" else []
        epoch = min((row["queued_ns"] for row in raw), default=None)
        contexts = {row["session_id"]: row["initial_tokens"]
                    for row in scenario["sessions"]}
        control = controls.get(
            (scenario["match_id"], result.get("allocation_id")), {}
        )
        ready, committed = [], []
        for row in raw:
            initial = row["initial"]
            continuation = next(
                item for item in result["continuations"]
                if item["session_id"] == row["move"]["session_id"]
            )
            readiness = _time(epoch, initial["first_byte_ns"])
            commit = _time(epoch, row["switch_end_ns"])
            ready.append(readiness)
            committed.append(commit)
            migrations.append({
                "scenario_id": scenario["scenario_id"],
                "match_id": scenario["match_id"],
                "sample_id": scenario.get("sample_id", scenario["match_id"]),
                "episode": scenario["episode"], "policy": scenario["policy"],
                "repeat": scenario.get("repeat", scenario["episode"]),
                "condition": scenario.get("condition", "default"),
                "context_profile": scenario.get("context_profile", "coding"),
                "token_distribution":
                    scenario.get("token_distribution", "uniform_support"),
                "required_deadline_s":
                    scenario.get("required_deadline_s", scenario["deadline_s"]),
                "session_id": row["move"]["session_id"],
                "method": row["move"]["method"], "order": row["move"]["order"],
                "context_tokens": contexts[row["move"]["session_id"]],
                "reaction_readiness_s": readiness,
                "reaction_commit_s": commit,
                "migration_start_s": _time(epoch, row["initial_start_ns"]),
                "migration_finish_s": _time(epoch, row["initial_end_ns"]),
                "quiesce_s": _time(epoch, row["pause_start_ns"]),
                "service_pause_s": _time(row["pause_start_ns"], row["switch_end_ns"]),
                "catch_up_start_s": _time(epoch, row.get("catch_up_start_ns"))
                if row.get("catch_up_start_ns") is not None else None,
                "catch_up_finish_s": _time(epoch, row.get("catch_up_end_ns"))
                if row.get("catch_up_end_ns") is not None else None,
                "continuation_start_s": _time(epoch, continuation["start_ns"]),
                "first_token_s": _time(epoch, continuation["first_byte_ns"]),
                "scheduler_wait_s": _time(epoch, row["initial_start_ns"]),
                "migration_ttft_s":
                    _time(row["initial_start_ns"], initial["first_byte_ns"]),
                "continuation_ttft_s":
                    _time(continuation["start_ns"], continuation["first_byte_ns"]),
                "continuation_ttft_delta_s": (
                    _time(continuation["start_ns"],
                          continuation["first_byte_ns"])
                    - control[row["move"]["session_id"]]
                    if row["move"]["session_id"] in control else float("nan")
                ),
            })
        planned = len(scenario["moves"])
        for row in deadline_attainment(
            committed, len(scenario["sessions"]),
            [scenario["required_deadline_s"]], power_curve,
            power_window_s,
        ) if power_curve else ():
            attainment.append({
                "scenario_id": scenario["scenario_id"],
                "match_id": scenario["match_id"],
                "sample_id": scenario.get("sample_id", scenario["match_id"]),
                "episode": scenario["episode"], "policy": scenario["policy"],
                "repeat": scenario.get("repeat", scenario["episode"]),
                "condition": scenario["condition"],
                "context_profile": scenario.get("context_profile", "coding"),
                "token_distribution": scenario["token_distribution"],
                "power_target_fraction": scenario["power_target_fraction"],
                **row,
                "hit_power_target":
                    row["power_attainment_fraction"] >= 1,
            })
        summaries.append({
            "scenario_id": scenario["scenario_id"],
            "match_id": scenario["match_id"],
            "sample_id": scenario.get("sample_id", scenario["match_id"]),
            "episode": scenario["episode"],
            "repeat": scenario.get("repeat", scenario["episode"]),
            "condition": scenario.get("condition", "default"),
            "context_profile": scenario.get("context_profile", "coding"),
            "token_distribution":
                scenario.get("token_distribution", "uniform_support"),
            "required_deadline_s":
                scenario.get("required_deadline_s", scenario["deadline_s"]),
            "policy": scenario["policy"], "status":
                result.get("status", "missing"),
            "planned_migrations": planned,
            "completed_migrations": len(raw),
            "readiness_50_s": _threshold(ready, planned, .5),
            "readiness_90_s": _threshold(ready, planned, .9),
            "readiness_100_s": _threshold(ready, planned, 1),
            "commit_50_s": _threshold(committed, planned, .5),
            "commit_90_s": _threshold(committed, planned, .9),
            "commit_100_s": _threshold(committed, planned, 1),
            "deadline_s": scenario["deadline_s"],
            "matched_control_complete": len(control) == planned,
        })
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out / "policy_migrations.csv", migrations)
    profiler.write_csv(out / "policy_episodes.csv", summaries)
    profiler.write_csv(out / "policy_attainment.csv", attainment)
    plot(migrations, summaries, out, "pooled")
    plot_destination_ttft(migrations, summaries, out)
    plot_max_session_ttft(migrations, summaries, out)
    if power_curve:
        plot_power_shed(migrations, summaries, power_curve, out)
        plot_hardware_pareto(attainment, summaries, out)
        plot_disruption(migrations, summaries, power_curve, out)
        plot_migration_time_per_watt(summaries, power_curve, out)
        plot_max_session_ttft_per_watt(
            migrations, summaries, power_curve, out
        )
        plot_full_power_attainment(summaries, power_window_s, out)
    for condition in sorted({row["condition"] for row in summaries}):
        plot(
            [row for row in migrations if row["condition"] == condition],
            [row for row in summaries if row["condition"] == condition],
            out, condition,
        )
    if attainment:
        plot_attainment(attainment, out, "pooled")
        for condition in sorted({row["condition"] for row in attainment}):
            plot_attainment(
                [row for row in attainment if row["condition"] == condition],
                out, condition,
            )
    timeline = representative_timeline(migrations, summaries)
    if timeline:
        profiler.write_csv(out / "policy_gantt.csv", timeline)
        plot_timeline(timeline, out)
    return migrations, summaries


def completion_curve(rows, summaries, policy, field):
    total = sum(int(row["planned_migrations"]) for row in summaries
                if row["policy"] == policy)
    values = sorted(float(row[field]) for row in rows
                    if row["policy"] == policy)
    return np.asarray(values), np.arange(1, len(values) + 1) / total


def full_power_attainment_curve(summaries, policy, deadline_s,
                                power_window_s):
    selected = [row for row in summaries if row["policy"] == policy
                and float(row["required_deadline_s"]) == deadline_s]
    if not selected:
        return np.asarray([]), np.asarray([])
    events = sorted(
        float(row["commit_100_s"]) + power_window_s for row in selected
        if row["commit_100_s"] not in (None, "")
    )
    x = np.r_[0, events]
    y = np.r_[0, np.arange(1, len(events) + 1) / len(selected)]
    return x, y


def power_shed_quantiles(rows, summaries, policy, power_curve, times):
    commits = {}
    for row in rows:
        if row["policy"] == policy:
            commits.setdefault(row["scenario_id"], []).append(
                float(row["reaction_commit_s"])
            )
    before, target = power_curve.power(.4), \
        power_curve.power(.4) - power_curve.power(0)
    curves = []
    for summary in summaries:
        if summary["policy"] != policy:
            continue
        planned = int(summary["planned_migrations"])
        completed = np.searchsorted(
            sorted(commits.get(summary["scenario_id"], ())), times,
            side="right",
        )
        load = .4 * (1 - completed / planned)
        curves.append(100 * (before - np.asarray([
            power_curve.power(value) for value in load
        ])) / target)
    return np.percentile(curves, (25, 50, 75), axis=0)


def plot_destination_ttft(rows, summaries, out):
    fig, ax = plt.subplots(figsize=CDF_FIGSIZE)
    for policy in POLICIES:
        x, y = completion_curve(
            rows, summaries, policy, "migration_ttft_s"
        )
        if len(x):
            ax.step(
                np.r_[0, x], np.r_[0, y], where="post",
                color=CDF_COLORS[policy], linestyle=CDF_LINESTYLES[policy],
                linewidth=3,
                label=CDF_LABELS[policy],
            )
    ax.set(
        xlabel="Migration + Destination TTFT (s)",
        ylabel="Cumulative Distribution", ylim=(0, 1.02),
    )
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"policy_hardware_destination_ttft_cdf.{suffix}", dpi=220
        )
    plt.close(fig)


def max_session_ttft_points(rows, summaries):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(
            float(row["migration_ttft_s"])
        )
    return [{"policy": row["policy"],
             "max_session_ttft_s": max(grouped[row["scenario_id"]])}
            for row in summaries
            if len(grouped.get(row["scenario_id"], ()))
            == int(row["planned_migrations"])]


def _plot_max_session_cdf(points, key, xlabel, filename, out):
    fig, ax = plt.subplots(figsize=CDF_FIGSIZE)
    for policy in POLICIES:
        values = sorted(row[key] for row in points
                        if row["policy"] == policy)
        if values:
            ax.step(np.r_[0, values], np.r_[0, np.arange(1, len(values) + 1)
                                           / len(values)], where="post",
                    color=CDF_COLORS[policy],
                    linestyle=CDF_LINESTYLES[policy], linewidth=3,
                    label=CDF_LABELS[policy])
    ax.set(xlabel=xlabel,
           ylabel="Cumulative Distribution", ylim=(0, 1.02))
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{filename}.{suffix}", dpi=220)
    plt.close(fig)


def plot_max_session_ttft(rows, summaries, out):
    _plot_max_session_cdf(
        max_session_ttft_points(rows, summaries), "max_session_ttft_s",
        "Maximum Session Migration + TTFT (s)",
        "policy_hardware_max_session_ttft_cdf", out,
    )


def max_session_ttft_per_watt_points(rows, summaries, power_curve):
    saved = power_curve.power(.4) - power_curve.power(0)
    return [{"policy": row["policy"],
             "max_session_s_per_w": row["max_session_ttft_s"] / saved}
            for row in max_session_ttft_points(rows, summaries)]


def plot_max_session_ttft_per_watt(rows, summaries, power_curve, out):
    _plot_max_session_cdf(
        max_session_ttft_per_watt_points(rows, summaries, power_curve),
        "max_session_s_per_w", "Max Session Migration + TTFT (s/W)",
        "policy_hardware_max_session_ttft_per_watt_cdf", out,
    )


def plot_full_power_attainment(summaries, power_window_s, out,
                               deadline_s=30):
    fig, ax = plt.subplots(figsize=plot_style.WIDE_FIGSIZE)
    for policy in CDF_POLICIES:
        x, y = full_power_attainment_curve(
            summaries, policy, deadline_s, power_window_s
        )
        if len(x):
            ax.step(x, y, where="post", **plot_style.policy_style(policy))
    ax.axvline(deadline_s, color="black", linestyle="--", linewidth=1.5)
    ax.text(
        deadline_s, .4, f"{deadline_s:g} s deadline",
        transform=ax.get_xaxis_transform(), ha="center", va="center",
        rotation=90, fontstyle="italic",
        fontsize=plot_style.LARGE_ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.set(
        xlabel="Time to Full Power-Shed Attainment (s)",
        ylabel="Cumulative Distribution", ylim=(0, 1.02),
    )
    ax.set_xlim(left=0)
    ax.tick_params(labelsize=plot_style.LARGE_FONT_SIZE)
    ax.xaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    ax.yaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    ax.grid(alpha=.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", framealpha=1,
                  facecolor="white", edgecolor="none",
                  fontsize=plot_style.LARGE_LEGEND_FONT_SIZE)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"policy_hardware_{deadline_s:g}s_full_power_attainment_cdf.{suffix}",
            dpi=plot_style.SAVE_DPI,
        )
    plt.close(fig)


def plot_destination_ttft_by_bandwidth(rows, summaries, scenarios, out):
    bandwidth = {
        row["scenario_id"]: float(row["bandwidth_mbps"]) for row in scenarios
    }
    values = sorted(set(bandwidth.values()))
    fig, axes = plt.subplots(
        math.ceil(len(values) / 2), 2, figsize=(12, 3.8 * math.ceil(len(values) / 2)),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax, value in zip(axes.flat, values):
        ids = {key for key, item in bandwidth.items() if item == value}
        selected_rows = [row for row in rows if row["scenario_id"] in ids]
        selected_summaries = [
            row for row in summaries if row["scenario_id"] in ids
        ]
        for policy in POLICIES:
            x, y = completion_curve(
                selected_rows, selected_summaries, policy, "migration_ttft_s"
            )
            if len(x):
                ax.step(
                    np.r_[0, x], np.r_[0, y], where="post",
                    color=TAB10_COLORS[policy], label=LABELS[policy],
                )
        ax.set(
            title=f"{value / 1000:g} Gbit/s",
            xlabel="Migration start → destination first token (s)",
            ylabel="Fraction of planned migrations", ylim=(0, 1.02),
        )
        ax.grid(alpha=.25)
    for ax in axes.flat[len(values):]:
        ax.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(POLICIES),
               frameon=False)
    fig.suptitle("Full width-8 hardware campaign by bandwidth", y=.94)
    fig.tight_layout(rect=(0, 0, 1, .88))
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"policy_hardware_destination_ttft_cdf_by_bandwidth.{suffix}",
            dpi=220,
        )
    plt.close(fig)


def plot_power_shed(rows, summaries, power_curve, out):
    times = np.asarray(sorted({
        0, *(float(row["reaction_commit_s"]) for row in rows)
    }))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for policy in POLICIES:
        if not any(row["policy"] == policy for row in summaries):
            continue
        low, median, high = power_shed_quantiles(
            rows, summaries, policy, power_curve, times
        )
        ax.step(
            times, median, where="post", color=TAB10_COLORS[policy],
            label=LABELS[policy],
        )
        ax.fill_between(
            times, low, high, step="post", color=TAB10_COLORS[policy], alpha=.12
        )
    ax.set(
        title="Pooled episodes (median; shaded IQR)",
        xlabel="Elapsed migration time (s)",
        ylabel="Modeled source-power shed (% of maximum)",
        xlim=(0, times[-1]), ylim=(0, 102),
    )
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"policy_hardware_power_shed_over_time.{suffix}", dpi=220
        )
    plt.close(fig)


def disruption_points(rows, summaries, power_curve):
    migrations = {}
    for row in rows:
        migrations.setdefault(row["scenario_id"], []).append(row)
    points, before = [], power_curve.power(.4)
    for summary in summaries:
        selected = migrations.get(summary["scenario_id"], [])
        planned = int(summary["planned_migrations"])
        saved = before - power_curve.power(.4 * (1 - len(selected) / planned))
        if saved > 0:
            pause = sum(float(row.get("service_pause_s") or
                              (float(row["reaction_commit_s"])
                               - float(row["quiesce_s"]))) for row in selected)
            points.append({
                "policy": summary["policy"],
                "session_s_per_w": pause / saved,
            })
    return points


def plot_disruption(rows, summaries, power_curve, out):
    points = disruption_points(rows, summaries, power_curve)
    fig, ax = plt.subplots(figsize=CDF_FIGSIZE)
    for policy in POLICIES:
        values = sorted(row["session_s_per_w"] for row in points
                        if row["policy"] == policy)
        if values:
            ax.step(values, np.arange(1, len(values) + 1) / len(values),
                    where="post", color=CDF_COLORS[policy],
                    linestyle=CDF_LINESTYLES[policy], linewidth=3,
                    label=CDF_LABELS[policy])
    ax.set(
        xscale="log", xlabel="Session Downtime / Power Shed (s/W)",
        ylabel="Cumulative Distribution", ylim=(0, 1.02),
    )
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=13, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_disruption_cdf.{suffix}", dpi=220)
    plt.close(fig)


def migration_time_per_watt_points(summaries, power_curve):
    saved = power_curve.power(.4) - power_curve.power(0)
    return [{"policy": row["policy"],
             "migration_s_per_w": float(row["commit_100_s"]) / saved}
            for row in summaries if row["commit_100_s"] not in (None, "")]


def plot_migration_time_per_watt(summaries, power_curve, out):
    points = migration_time_per_watt_points(summaries, power_curve)
    fig, ax = plt.subplots(figsize=CDF_FIGSIZE)
    for policy in POLICIES:
        values = sorted(row["migration_s_per_w"] for row in points
                        if row["policy"] == policy)
        if values:
            ax.step(values, np.arange(1, len(values) + 1) / len(values),
                    where="post", color=CDF_COLORS[policy],
                    linestyle=CDF_LINESTYLES[policy], linewidth=3,
                    label=CDF_LABELS[policy])
    ax.set(
        xlabel="E2E Migration / Power Shed (s/W)",
        ylabel="Cumulative Distribution", ylim=(0, 1.02),
    )
    ax.tick_params(labelsize=15)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=13, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_migration_time_per_watt_cdf.{suffix}",
                    dpi=220)
    plt.close(fig)


def pareto_points(attainment, summaries):
    achieved = {row["scenario_id"]: row for row in attainment}
    points = []
    for summary in summaries:
        if summary["scenario_id"] not in achieved:
            continue
        commit = summary["commit_100_s"]
        elapsed = float(commit) if commit not in (None, "") \
            else float(summary["deadline_s"])
        points.append({
            "match_id": summary["match_id"], "policy": summary["policy"],
            "shed_percent":
                100 * float(achieved[summary["scenario_id"]]
                            ["power_attainment_fraction"]),
            "deadline_fraction":
                elapsed / float(summary["required_deadline_s"]),
            "censored": commit in (None, ""), "pareto": False,
        })
    for point in points:
        peers = [row for row in points
                 if row["match_id"] == point["match_id"]]
        point["pareto"] = not any(
            row["shed_percent"] >= point["shed_percent"]
            and row["deadline_fraction"] <= point["deadline_fraction"]
            and (row["shed_percent"] > point["shed_percent"]
                 or row["deadline_fraction"] < point["deadline_fraction"])
            for row in peers
        )
    return points


def plot_hardware_pareto(attainment, summaries, out):
    points = pareto_points(attainment, summaries)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for policy in POLICIES:
        selected = [row for row in points if row["policy"] == policy]
        if not selected:
            continue
        count = sum(row["pareto"] for row in selected)
        ax.scatter(
            [row["shed_percent"] for row in selected],
            [row["deadline_fraction"] for row in selected],
            color=TAB10_COLORS[policy], alpha=.45, s=28,
            label=f"{LABELS[policy]} ({count}/{len(selected)} frontier)",
            zorder=len(POLICIES) - POLICIES.index(policy),
        )
        frontier = [row for row in selected if row["pareto"]]
        ax.scatter(
            [row["shed_percent"] for row in frontier],
            [row["deadline_fraction"] for row in frontier],
            facecolors="none", edgecolors="black", linewidths=.6, s=48,
            zorder=5,
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set(
        title="Measured paired hardware operating points (100% target)",
        xlabel="Modeled source-power shed by deadline (% of maximum)",
        ylabel="Full-width completion time / required deadline",
        xlim=(-2, 102), ylim=(0, None),
    )
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        .5, .01, "Black outline: paired nondominated; "
        "19 s optimizer policies include fastest-tail moves",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_measured_pareto.{suffix}", dpi=220)
    plt.close(fig)


def _pooled_results(paths):
    rows, summaries = [], []
    for path in paths:
        with (path / "policy_migrations.csv").open() as stream:
            rows.extend(csv.DictReader(stream))
        with (path / "policy_episodes.csv").open() as stream:
            summaries.extend(csv.DictReader(stream))
    return rows, summaries



def validate_run(run_root: Path, expected_episodes=120,
                 policy="isolated_fastest"):
    plan_path = run_root / "plan.json"
    plan_ = json.loads(plan_path.read_text())
    validate_policy_plan(plan_)
    metadata = json.loads((run_root / "run_metadata.json").read_text())
    model_path = ROOT.parent / plan_["model_profile"]["path"]
    manifest_path = ROOT.parent / plan_["manifest"]["path"]
    if plan_["policies"] != [policy] or plan_["episodes"] != expected_episodes \
            or metadata.get("dirty") \
            or metadata.get("plan_sha256") != profiler.file_hash(plan_path) \
            or metadata.get("plan_object_sha256") != profiler.object_hash(plan_) \
            or profiler.file_hash(model_path) \
            != plan_["model_profile"]["sha256"] \
            or profiler.file_hash(manifest_path) != plan_["manifest"]["sha256"]:
        raise RuntimeError("baseline plan or provenance is not frozen")
    for scenario in plan_["scenarios"]:
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        if result.get("status") != "complete":
            raise RuntimeError("incomplete scenario {}".format(scenario["scenario_id"]))
    with (run_root / "policy_episodes.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    with (run_root / "policy_attainment.csv").open() as stream:
        attainment = list(csv.DictReader(stream))
    scenario_ids = {
        row["scenario_id"] for row in plan_["scenarios"]
        if row["policy"] == policy
    }
    if len(summaries) != expected_episodes \
            or {row["scenario_id"] for row in summaries} != scenario_ids \
            or len(attainment) != expected_episodes \
            or {row["scenario_id"] for row in attainment} != scenario_ids \
            or any(row["status"] != "complete"
                   or row["completed_migrations"] != row["planned_migrations"]
                   or row["matched_control_complete"] != "True"
                   for row in summaries):
        raise RuntimeError("baseline reduction is incomplete")
    cells = {}
    for row in summaries:
        cells.setdefault(row["condition"], set()).add(int(row["repeat"]))
    expected_repeats = set(range(plan_["episodes_per_cell"]))
    if not cells or any(repeats != expected_repeats
                        for repeats in cells.values()):
        raise RuntimeError("baseline conditions do not have exact repetitions")
    report = {
        "schema": "queue-haul-policy-baseline-validation-v1",
        "valid": True, "policy": policy, "episodes": len(summaries),
        "attainment_rows": len(attainment), "conditions": len(cells),
        "repetitions_per_condition": plan_["episodes_per_cell"],
        "executed_scenarios": len(plan_["scenarios"]),
        "git_sha": metadata["git_sha"],
        "plan_sha256": profiler.file_hash(plan_path),
        "model_profile_sha256": plan_["model_profile"]["sha256"],
    }
    profiler.write_json(run_root / "policy_validation.json", report)
    files = sorted(path for path in run_root.rglob("*")
                   if path.is_file() and path.name != "SHA256SUMS")
    (run_root / "SHA256SUMS").write_text("".join(
        f"{profiler.file_hash(path)}  {path.relative_to(run_root)}\n"
        for path in files
    ))
    return report


def common_packing_comparison(packing: Path, baseline: Path, out: Path):
    plans = [json.loads((root / "plan.json").read_text())
             for root in (packing, baseline)]
    matches = [{row["match_id"] for row in plan_["scenarios"]}
               for plan_ in plans]
    if matches[0] != matches[1] or plans[0]["episodes"] != 120 \
            or plans[1]["episodes"] != 120 \
            or set(plans[0]["policies"]) \
            != set(PACKING_POLICIES) - {"isolated_fastest"} \
            or plans[1]["policies"] != ["isolated_fastest"]:
        raise RuntimeError("comparison inputs are not the common packing cohort")
    attainment, summaries = [], []
    for root in (packing, baseline):
        with (root / "policy_attainment.csv").open() as stream:
            attainment.extend(csv.DictReader(stream))
        with (root / "policy_episodes.csv").open() as stream:
            summaries.extend(csv.DictReader(stream))
    by_policy = {
        policy: {row["match_id"]: float(row["power_attainment_fraction"])
                 for row in attainment if row["policy"] == policy}
        for policy in PACKING_POLICIES
    }
    if len(attainment) != 120 * len(PACKING_POLICIES) \
            or len(summaries) != 120 * len(PACKING_POLICIES) \
            or any(set(rows) != matches[0] for rows in by_policy.values()) \
            or any({row["match_id"] for row in summaries
                    if row["policy"] == policy} != matches[0]
                   for policy in PACKING_POLICIES) \
            or any(row["status"] != "complete" for row in summaries):
        raise RuntimeError("comparison inputs are incomplete or unmatched")
    reference = by_policy["queue_haul"]
    classified = []
    for policy in PACKING_POLICIES:
        labels = ["better" if value - reference[match] > .05
                  else "worse" if value - reference[match] < -.05
                  else "similar"
                  for match, value in by_policy[policy].items()]
        classified.append({
            "policy": policy, "observations": len(labels),
            **{label: labels.count(label) for label in
               ("better", "similar", "worse")},
        })
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out / "policy_common_packing_classification.csv",
                       classified)
    fig, ax = plt.subplots(figsize=(7, 4))
    left = np.zeros(len(classified))
    for label, color in (("worse", "#D55E00"), ("similar", "#999999"),
                         ("better", "#009E73")):
        values = np.asarray([100 * row[label] / row["observations"]
                             for row in classified])
        ax.barh(range(len(classified)), values, left=left, label=label.title(),
                color=color)
        left += values
    ax.set(
        xlabel="Share of matched packing episodes (%)", xlim=(0, 100),
        yticks=range(len(classified)),
        yticklabels=["{} (n={})".format(CDF_LABELS[row["policy"]], row["observations"])
                     for row in classified],
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(.5, 1))
    fig.text(.5, .01,
             "Relative to Queue-Haul LP; similar is within ±5 percentage points",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_common_packing.{suffix}", dpi=220)
    plt.close(fig)
    return classified


def plot_reduced(out, model_path=DEFAULT_MODEL, pooled_with=()):
    plan_ = json.loads((out / "plan.json").read_text())
    rows, summaries = _pooled_results((out,))
    with (out / "policy_attainment.csv").open() as stream:
        attainment = list(csv.DictReader(stream))
    pooled_rows, pooled_summaries = _pooled_results(pooled_with)
    plot_destination_ttft(rows + pooled_rows, summaries + pooled_summaries, out)
    plot_max_session_ttft(
        rows + pooled_rows, summaries + pooled_summaries, out
    )
    plot_destination_ttft_by_bandwidth(
        rows, summaries, plan_["scenarios"], out
    )
    model = ModelProfile.load(model_path)
    power_curve = model.case().power_curve
    plot_power_shed(rows, summaries, power_curve, out)
    plot_hardware_pareto(attainment, summaries, out)
    plot_disruption(rows + pooled_rows, summaries + pooled_summaries,
                    power_curve, out)
    plot_migration_time_per_watt(summaries + pooled_summaries, power_curve, out)
    plot_max_session_ttft_per_watt(
        rows + pooled_rows, summaries + pooled_summaries, power_curve, out
    )
    plot_full_power_attainment(
        summaries + pooled_summaries, model.power_window_s, out
    )


def representative_timeline(rows, summaries):
    complete = {
        row["scenario_id"] for row in summaries
        if row["policy"] == "queue_haul"
        and row["completed_migrations"] == row["planned_migrations"]
    }
    grouped = {
        scenario: sorted(
            (row for row in rows if row["scenario_id"] == scenario),
            key=lambda row: row["order"],
        )
        for scenario in complete
    }
    mixed = [value for value in grouped.values()
             if {row["method"] for row in value} == {"replay", "kv_transfer"}]
    return min(mixed, key=lambda value: (value[0]["episode"], value[0]["scenario_id"])) \
        if mixed else []


def plot_timeline(rows, out):
    colors = {"replay": "#4C78A8", "kv_transfer": "#F58518"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for y, row in enumerate(rows):
        start, commit = row["migration_start_s"], row["reaction_commit_s"]
        ax.barh(y, commit - start, left=start, color=colors[row["method"]])
        ax.scatter(row["reaction_readiness_s"], y, marker="o", color="black", s=24)
        ax.scatter(commit, y, marker="D", color="#A11919", s=34)
        ax.scatter(row["first_token_s"], y, marker="*", color="#176B52", s=70)
    ax.set(
        xlabel="Time from policy epoch (s)", ylabel="Submission rank",
        yticks=range(len(rows)),
        yticklabels=[f"{row['order']}: {row['method'].replace('_', ' ')}"
                     for row in rows],
        title=f"Measured Queue-Haul parallel episode {rows[0]['episode']}",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.25)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        *(Line2D([0], [0], color=color, lw=8, label=method.replace("_", " "))
          for method, color in colors.items()),
        Line2D([0], [0], marker="o", color="black", lw=0,
               label="Destination ready"),
        Line2D([0], [0], marker="D", color="#A11919", lw=0, label="Commit"),
        Line2D([0], [0], marker="*", color="#176B52", lw=0,
               markersize=10, label="First token"),
    ], ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_gantt.{suffix}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)


def plot(rows, summaries, out, cohort=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    policies = [policy for policy in POLICIES
                if any(row["policy"] == policy for row in summaries)]
    for policy in policies:
        for ax, field in zip(
            axes[:3], ("reaction_readiness_s", "migration_ttft_s",
                       "reaction_commit_s")
        ):
            x, y = completion_curve(rows, summaries, policy, field)
            ax.step(
                np.r_[0, x], np.r_[0, y],
                where="post", color=TAB10_COLORS[policy], label=LABELS[policy],
            )
    axes[0].set_title("Controller queue → first token")
    axes[1].set_title("Destination TTFT")
    axes[2].set_title("Controller queue → route commit")
    for i, ax in enumerate(axes):
        ax.set_xlabel(
            "Transfer/replay + destination prefill (s)" if i == 1
            else "Time from common policy epoch (s)"
        )
        ax.set_ylabel("Fraction of planned migrations")
        ax.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(policies),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for suffix in ("png", "pdf"):
        name = f"policy_hardware_{cohort}_cdf" if cohort \
            else "policy_hardware_cdf"
        fig.savefig(out / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot_attainment(rows, out, condition=None):
    deadlines = sorted({row["required_deadline_s"] for row in rows})
    fig, axes = plt.subplots(
        1, len(deadlines), figsize=(4.4 * len(deadlines), 4), squeeze=False
    )
    axes = axes[0]
    for ax, deadline in zip(axes, deadlines):
        for policy in POLICIES:
            values = sorted(
                100 * row["power_attainment_fraction"] for row in rows
                if row["policy"] == policy
                and row["required_deadline_s"] == deadline
            )
            if values:
                ax.step(
                    values, np.arange(1, len(values) + 1) / len(values),
                    where="post", color=TAB10_COLORS[policy], label=LABELS[policy],
                )
        ax.set(
            title=f"{deadline:g} s requirement",
            xlabel="Power-target attainment by deadline (%)",
            ylabel="Fraction of episodes", xlim=(0, 102), ylim=(0, 1.02),
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(POLICIES),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    for suffix in ("png", "pdf"):
        name = f"policy_hardware_{condition}_attainment_cdf" if condition \
            else "policy_hardware_attainment_cdf"
        fig.savefig(out / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    command.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=2)
    command.add_argument("--sessions", type=int, default=8)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--bandwidth-mbps", type=float, default=10_000)
    command.add_argument("--bandwidths-mbps", type=float, nargs="+")
    command.add_argument("--deadline-s", type=float, default=180)
    command.add_argument("--workload-profiles", type=Path, nargs="+",
                         default=DEFAULT_WORKLOADS)
    command.add_argument("--token-distributions", nargs="+",
                         default=TOKEN_DISTRIBUTIONS)
    command.add_argument("--required-deadlines-s", type=float, nargs="+",
                         default=REQUIRED_DEADLINES_S)
    command.add_argument("--context-packs", nargs="+",
                         choices=tuple(CONTEXT_PACKS))
    command.add_argument("--policies", nargs="+", choices=POLICIES,
                         default=DEFAULT_POLICIES)
    command = sub.add_parser("prepare-baselines")
    command.add_argument("--source-plan", type=Path, required=True)
    command.add_argument("--model-profile", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--policies", nargs="+", choices=RERUN_POLICIES,
                         default=NETWORK_BASELINES)
    command.add_argument("--condition-shard", type=int, nargs=2)
    command = sub.add_parser("reduce")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path)
    command = sub.add_parser("plot-reduced")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    command.add_argument("--pooled-with", type=Path, nargs="*", default=())
    command = sub.add_parser("validate")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--expected-episodes", type=int, default=120)
    command.add_argument("--policy", choices=POLICIES,
                         default="isolated_fastest")
    command = sub.add_parser("plot-common-packing")
    command.add_argument("--packing-run", type=Path, required=True)
    command.add_argument("--baseline-run", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.manifest, args.out, model_path=args.model_profile,
            episodes=args.episodes, sessions=args.sessions, seed=args.seed,
            bandwidth_mbps=args.bandwidth_mbps, deadline_s=args.deadline_s,
            bandwidths_mbps=args.bandwidths_mbps,
            workload_paths=args.workload_profiles,
            token_distributions=args.token_distributions,
            required_deadlines_s=args.required_deadlines_s,
            context_packs=args.context_packs or (), policies=args.policies,
        )
    elif args.command == "prepare-baselines":
        prepare_baselines(
            args.source_plan, args.out, args.policies, args.model_profile,
            args.condition_shard,
        )
    elif args.command == "reduce":
        reduce_run(args.run_root, args.out)
    elif args.command == "plot-reduced":
        plot_reduced(args.out, args.model_profile, args.pooled_with)
    elif args.command == "validate":
        validate_run(args.run_root, args.expected_episodes, args.policy)
    else:
        common_packing_comparison(args.packing_run, args.baseline_run, args.out)


if __name__ == "__main__":
    main()
