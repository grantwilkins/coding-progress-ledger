"""Build and reduce a paired, ungated hardware policy campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import migration_profiler as profiler
from planner import _duration, plan
from profiles import ModelProfile
from simulate import ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession


ROOT = Path(__file__).parent
DEFAULT_MANIFEST = Path("queue-haul/outputs/coding-manifest.json")
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
POLICIES = ("queue_haul", "greedy", "random", "kv_only", "replay_only")
LABELS = {
    "queue_haul": "Queue-Haul", "greedy": "Greedy",
    "random": "Random feasible", "kv_only": "KV only",
    "replay_only": "Replay only",
}


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


def _moves(policy, scenario, routes, profile, seed):
    sessions = list(scenario.sessions)
    if policy in {"queue_haul", "greedy", "random"}:
        solver = {"queue_haul": "lp_work_first", "greedy": "greedy",
                  "random": "random"}[policy]
        result = plan(scenario, profile, routes, solver, seed=seed)
        moves = result.moves
        solve_s = result.solve_s
    else:
        method = "kv_transfer" if policy == "kv_only" else "replay"
        case = profile.case()
        links = {row.link_id: row.bytes_per_s for row in scenario.links}
        ordered = sorted(
            sessions,
            key=lambda row: (
                _duration(row, method, case, ("link",), links),
                row.session_id,
            ),
        )
        moves = tuple(
            (row.session_id, method, order)
            for order, row in enumerate(ordered)
        )
        solve_s = 0.0
    normalized = [
        (move.session_id, move.method, move.order)
        if hasattr(move, "session_id") else move
        for move in moves
    ]
    if {row[0] for row in normalized} != {row.session_id for row in sessions}:
        raise RuntimeError(f"{policy} did not plan the complete episode")
    return normalized, solve_s


def make_plan(manifest_path: Path, model_path: Path = DEFAULT_MODEL,
              episodes: int = 50, sessions: int = 8, seed: int = 0,
              bandwidth_mbps: float = 10_000, deadline_s: float = 180,
              context_min: int = 4096, context_max: int = 30_464) -> dict:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    profile = ModelProfile.load(model_path)
    if episodes < 1 or not 1 <= sessions <= len(manifest["sessions"]) \
            or bandwidth_mbps <= 0 or deadline_s <= profile.power_window_s \
            or context_min < 1 or context_max < context_min \
            or context_min % 256 or context_max % 256:
        raise ValueError("invalid policy campaign dimensions")
    replay_contexts = profile.case().replay.by_concurrency[1][0]
    if context_min < replay_contexts[0] or context_max > replay_contexts[-1]:
        raise ValueError("policy campaign contexts exceed measured replay range")

    rng, scenarios = random.Random(seed), []
    available = sorted(manifest["sessions"], key=lambda row: row["id"])
    for episode in range(episodes):
        chosen = rng.sample(available, sessions)
        session_rows = [{
            "session_id": row["id"], "job_class": row["job_class"],
            "turn_index": 0,
            "initial_tokens": rng.randrange(context_min, context_max + 256, 256),
            "order": order,
        } for order, row in enumerate(chosen)]
        match_id = profiler.object_hash([seed, episode, session_rows])[:16]
        base = {
            "match_id": match_id, "episode": episode,
            "campaign": "policy_hardware", "split": "measurement",
            "activity": "none", "activity_tokens": 0,
            "request_schedule": [], "repeat": episode,
            "deadline_s": deadline_s, "sessions": session_rows,
            "serving_concurrency": 1, "concurrency": 1,
            "move_concurrency": 1, "copy_policy": "initial_final",
            "final_state": "awake", "bandwidth_mbps": bandwidth_mbps,
        }
        scenarios.append({
            **base, "scenario_id": f"c-{match_id}", "kind": "control",
            "method": "replay", "policy": "control", "moves": [],
        })
        problem, routes = _problem(
            profile, session_rows, bandwidth_mbps, deadline_s
        )
        for policy in POLICIES:
            moves, solve_s = _moves(
                policy, problem, routes, profile,
                profiler.stable_seed(seed, episode, policy),
            )
            move_rows = [{
                **next(row for row in session_rows if row["session_id"] == session),
                "method": method, "order": order,
            } for session, method, order in moves]
            scenario_id = profiler.object_hash([match_id, policy, move_rows])[:16]
            scenarios.append({
                **base, "scenario_id": f"p-{scenario_id}",
                "kind": "migration", "method":
                    move_rows[0]["method"]
                    if len({row["method"] for row in move_rows}) == 1
                    else "mixed",
                "policy": policy, "planner_s": solve_s, "moves": move_rows,
            })
    rng.shuffle(scenarios)
    output = {
        "schema": profiler.PLAN_SCHEMA,
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "seed": seed, "campaign": "policy_hardware",
        "policies": list(POLICIES), "episodes": episodes,
        "sessions_per_episode": sessions, "scenarios": scenarios,
    }
    profiler.validate_plan(output, manifest)
    validate_policy_plan(output)
    return output


def validate_policy_plan(plan_: dict) -> None:
    policies = set(plan_["policies"])
    for episode in range(plan_["episodes"]):
        rows = [row for row in plan_["scenarios"]
                if row["episode"] == episode]
        if {row["policy"] for row in rows} != policies | {"control"}:
            raise ValueError("every episode must contain every policy and one control")
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
    job.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
: "${{QH_POLICY_RUN_ROOT:?set QH_POLICY_RUN_ROOT}}"
export QH_LMCACHE_MODE="${{QH_LMCACHE_MODE:-mp}}"
status=0
uv run python queue-haul/migration_profiler.py run --plan {shlex.quote(str(plan_path))} --run-root "$QH_POLICY_RUN_ROOT" || status=$?
uv run python queue-haul/policy_hardware_campaign.py reduce --run-root "$QH_POLICY_RUN_ROOT"
exit "$status"
""")
    job.chmod(0o755)
    return plan_


def _time(start, end):
    return (end - start) / 1e9


def _threshold(values, planned, fraction):
    rank = math.ceil(planned * fraction)
    return sorted(values)[rank - 1] if len(values) >= rank else None


def reduce_run(run_root: Path, out: Path | None = None):
    plan_ = json.loads((run_root / "plan.json").read_text())
    validate_policy_plan(plan_)
    out = out or run_root
    controls = {}
    for scenario in plan_["scenarios"]:
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        if scenario["policy"] != "control" or not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            controls[scenario["match_id"]] = {
                row["session_id"]: _time(row["start_ns"], row["first_byte_ns"])
                for row in result["continuations"]
            }

    migrations, summaries = [], []
    for scenario in plan_["scenarios"]:
        if scenario["policy"] == "control":
            continue
        path = run_root / "scenarios" / scenario["scenario_id"] / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        raw = result.get("migrations", []) if result.get("status") == "complete" else []
        epoch = min((row["queued_ns"] for row in raw), default=None)
        contexts = {row["session_id"]: row["initial_tokens"]
                    for row in scenario["sessions"]}
        control = controls.get(scenario["match_id"], {})
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
                "episode": scenario["episode"], "policy": scenario["policy"],
                "session_id": row["move"]["session_id"],
                "method": row["move"]["method"], "order": row["move"]["order"],
                "context_tokens": contexts[row["move"]["session_id"]],
                "reaction_readiness_s": readiness,
                "reaction_commit_s": commit,
                "scheduler_wait_s": _time(epoch, row["initial_start_ns"]),
                "migration_ttft_s":
                    _time(row["initial_start_ns"], initial["first_byte_ns"]),
                "continuation_ttft_s":
                    _time(continuation["start_ns"], continuation["first_byte_ns"]),
                "continuation_ttft_delta_s":
                    _time(continuation["start_ns"], continuation["first_byte_ns"])
                    - control.get(row["move"]["session_id"], float("nan")),
            })
        planned = len(scenario["moves"])
        summaries.append({
            "scenario_id": scenario["scenario_id"],
            "match_id": scenario["match_id"], "episode": scenario["episode"],
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
            "planner_s": scenario.get("planner_s", 0),
        })
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out / "policy_migrations.csv", migrations)
    profiler.write_csv(out / "policy_episodes.csv", summaries)
    plot(migrations, summaries, out)
    return migrations, summaries


def completion_curve(rows, summaries, policy, field):
    total = sum(row["planned_migrations"] for row in summaries
                if row["policy"] == policy)
    values = sorted(row[field] for row in rows if row["policy"] == policy)
    return np.asarray(values), np.arange(1, len(values) + 1) / total


def plot(rows, summaries, out):
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
    horizon = max(
        [row["deadline_s"] for row in summaries]
        + [row[field] for row in rows for field in
           ("reaction_readiness_s", "migration_ttft_s",
            "reaction_commit_s")]
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.ravel()
    policies = [policy for policy in POLICIES
                if any(row["policy"] == policy for row in summaries)]
    for policy in policies:
        for ax, field in zip(
            axes[:3], ("reaction_readiness_s", "migration_ttft_s",
                       "reaction_commit_s")
        ):
            x, y = completion_curve(rows, summaries, policy, field)
            ax.step(
                np.r_[0, x, horizon],
                np.r_[0, y, y[-1] if len(y) else 0],
                where="post", color=colors[policy], label=LABELS[policy],
            )
        delta = sorted(
            row["continuation_ttft_delta_s"] for row in rows
            if row["policy"] == policy
            and math.isfinite(row["continuation_ttft_delta_s"])
        )
        if delta:
            total = sum(
                row["planned_migrations"] for row in summaries
                if row["policy"] == policy
            )
            axes[3].step(
                delta, np.arange(1, len(delta) + 1) / total,
                where="post", color=colors[policy], label=LABELS[policy],
            )
    axes[0].set_title("Reaction → first token")
    axes[1].set_title("Destination TTFT")
    axes[2].set_title("Reaction → route commit")
    axes[3].set_title("Next-request TTFT inflation")
    for i, ax in enumerate(axes[:3]):
        ax.set_xlabel(
            "Transfer/replay + destination prefill (s)" if i == 1
            else "Time from common policy epoch (s)"
        )
        ax.set_ylabel("Fraction of planned migrations")
        ax.set_ylim(0, 1.02)
    axes[3].set_xlabel("Treatment − matched control (s)")
    axes[3].set_ylabel("Fraction of planned migrations")
    axes[3].set_ylim(0, 1.02)
    fig.legend(loc="upper center", ncol=len(policies), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"policy_hardware_cdf.{suffix}", dpi=220)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--episodes", type=int, default=50)
    command.add_argument("--sessions", type=int, default=8)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--bandwidth-mbps", type=float, default=10_000)
    command.add_argument("--deadline-s", type=float, default=180)
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
        )
    else:
        reduce_run(args.run_root, args.out)


if __name__ == "__main__":
    main()
