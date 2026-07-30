"""Build and reduce a paired, ungated hardware policy campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import migration_profiler as profiler
from planner import _duration, plan
from profiles import ModelProfile, WorkloadProfile
from simulate import ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession


ROOT = Path(__file__).parent
DEFAULT_MANIFEST = Path("queue-haul/outputs/coding-manifest.json")
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
DEFAULT_WORKLOADS = tuple(ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
))
TOKEN_DISTRIBUTIONS = ("uniform_support", "uniform_range")
REQUIRED_DEADLINES_S = (30,)
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only")
LABELS = {
    "queue_haul": "QH choice/order", "greedy": "Greedy choice/order",
    "isolated_fastest": "Per-session fastest", "random": "Random choice/order",
    "kv_only": "KV only", "replay_only": "Replay only",
}
EXECUTION_CONTRACT = "eager_parallel_all_sessions"


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
    if policy in {"queue_haul", "greedy", "random"}:
        solver = {"queue_haul": "lp_work_first", "greedy": "greedy",
                  "random": "random"}[policy]
        result = plan(scenario, profile, routes, solver, seed=seed)
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
              bandwidths_mbps=None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    profile = ModelProfile.load(model_path)
    workloads = [WorkloadProfile.load(path) for path in workload_paths]
    bandwidths = (bandwidth_mbps,) if bandwidths_mbps is None \
        else tuple(bandwidths_mbps)
    if episodes < 1 or not 1 <= sessions <= len(manifest["sessions"]) \
            or not bandwidths or min(bandwidths) <= 0 \
            or len(set(bandwidths)) != len(bandwidths) \
            or deadline_s <= profile.power_window_s \
            or not workloads or not token_distributions \
            or not required_deadlines_s \
            or min(required_deadlines_s) <= profile.power_window_s:
        raise ValueError("invalid policy campaign dimensions")
    replay_contexts = profile.case().replay.by_concurrency[1][0]

    rng, scenarios = random.Random(seed), []
    available = sorted(manifest["sessions"], key=lambda row: row["id"])
    cells = [
        (workload, distribution, bandwidth, required_deadline, repeat)
        for workload in workloads for distribution in token_distributions
        for bandwidth in bandwidths
        for required_deadline in required_deadlines_s
        for repeat in range(episodes)
    ]
    for episode, (workload, distribution, bandwidth, required_deadline, repeat) \
            in enumerate(cells):
        sample_rng = random.Random(profiler.stable_seed(
            seed, workload.profile_id, distribution, repeat
        ))
        chosen = sample_rng.sample(available, sessions)
        contexts = _context_tokens(
            workload, distribution, sessions, sample_rng
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
        job_type = workload.records[0].job_type
        condition = f"{job_type}-{distribution}-{bandwidth:g}mbps-" \
            f"{required_deadline:g}s"
        sample_id = profiler.object_hash(
            [seed, workload.profile_id, distribution, repeat, session_rows]
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
            "request_schedule": [], "repeat": episode,
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
            profile, session_rows, bandwidth_mbps, required_deadline
        )
        for policy in POLICIES:
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
    scenarios = [row for block in blocks for row in block]
    output = {
        "schema": profiler.PLAN_SCHEMA,
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "seed": seed, "campaign": "policy_hardware",
        "execution_contract": EXECUTION_CONTRACT,
        "model_profile": {
            "path": _portable_path(model_path),
            "sha256": profiler.file_hash(model_path),
            "profile_id": profile.profile_id,
        },
        "policies": list(POLICIES), "episodes": len(cells),
        "episodes_per_cell": episodes,
        "workload_profiles": [
            {"path": _portable_path(path), "sha256": profiler.file_hash(path)}
            for path in workload_paths
        ],
        "token_distributions": list(token_distributions),
        "bandwidths_mbps": list(bandwidths),
        "required_deadlines_s": list(required_deadlines_s),
        "power_target_fraction": 1.0,
        "sessions_per_episode": sessions, "scenarios": scenarios,
    }
    profiler.validate_plan(output, manifest)
    validate_policy_plan(output)
    return output


def validate_policy_plan(plan_: dict) -> None:
    if plan_.get("execution_contract") != EXECUTION_CONTRACT:
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


def prepare(manifest: Path, out: Path, **kwargs) -> dict:
    plan_ = make_plan(manifest, **kwargs)
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
#SBATCH --mem=256G
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
    total = sum(row["planned_migrations"] for row in summaries
                if row["policy"] == policy)
    values = sorted(row[field] for row in rows if row["policy"] == policy)
    return np.asarray(values), np.arange(1, len(values) + 1) / total


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
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
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
                where="post", color=colors[policy], label=LABELS[policy],
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
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
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
                    where="post", color=colors[policy], label=LABELS[policy],
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
    command = sub.add_parser("reduce")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.manifest, args.out, episodes=args.episodes,
            sessions=args.sessions, seed=args.seed,
            bandwidth_mbps=args.bandwidth_mbps, deadline_s=args.deadline_s,
            bandwidths_mbps=args.bandwidths_mbps,
            workload_paths=args.workload_profiles,
            token_distributions=args.token_distributions,
            required_deadlines_s=args.required_deadlines_s,
        )
    else:
        reduce_run(args.run_root, args.out)


if __name__ == "__main__":
    main()
