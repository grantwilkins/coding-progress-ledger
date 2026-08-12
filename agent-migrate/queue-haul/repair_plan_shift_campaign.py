"""Simulate three planner snapshots and plot their action mix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import migration_profiler as profiler
import network_campaign as network
from planner import _expected_scenario
from profiles import ModelProfile
from simulate import predict


ROOT = Path(__file__).parent
DEFAULT_PARENT = ROOT / "outputs/east-germany-separation-20260809/plan.json"
STATES = (
    ("original", "Original", .25, "natural"),
    ("bandwidth_drop", "Bandwidth drop", .25, "controlled_40"),
    ("prefill_drop", "Prefill drop", .90, "natural"),
)
MIX = (
    ("east_replay", "East replay", "#006CB8"),
    ("east_kv_transfer", "East KV", "#6FC3DF"),
    ("germany_replay", "Germany replay", "#B1040E"),
    ("germany_kv_transfer", "Germany KV", "#E98300"),
)


def _scenario(template, parent, condition, germany_rho, bandwidth):
    return {
        **template,
        "design": "repair_plan_shift_simulation",
        "condition_id": condition,
        "background": {"east": (.25, 0), "germany": (germany_rho, 0)},
        "bandwidth": bandwidth,
        "bandwidth_mbps": network._bandwidths(
            parent["network_contract"], bandwidth),
        "requested_shed_fraction": .5,
        "admission_mode": "normal",
        "full_horizon_s": network.ORACLE_STALE_HORIZON_S,
        "background_kv_headroom_tokens": {
            "east": network.HARDWARE_GAP_BACKGROUND_KV_TOKENS,
            "germany": network.HARDWARE_GAP_BACKGROUND_KV_TOKENS,
        },
    }


def _diff(original, moves):
    signature = lambda move: (
        move.method, move.destination_pool, move.destination_instance)
    before = {move.session_id: signature(move) for move in original}
    after = {move.session_id: signature(move) for move in moves}
    changed = {session for session in before.keys() | after.keys()
               if before.get(session) != after.get(session)}
    return {
        "changed_sessions": len(changed),
        "redirected_sessions": sum(session in before and session in after
                                   for session in changed),
        "added_sessions": len(after.keys() - before.keys()),
        "removed_sessions": len(before.keys() - after.keys()),
        "unchanged_sessions": sum(before.get(session) == action
                                  for session, action in after.items()),
    }


def _plot(rows, path):
    bottom = [0] * len(rows)
    fig, axis = plt.subplots(figsize=(6, 3.5))
    for key, label, color in MIX:
        values = [row[key] for row in rows]
        axis.bar([row["label"] for row in rows], values, bottom=bottom,
                 label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axis.set_ylabel("Planned migrations")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(out: Path, parent_path: Path = DEFAULT_PARENT):
    out.mkdir(parents=True, exist_ok=True)
    parent = json.loads(parent_path.read_text())
    manifest_path = Path(parent["manifest"]["path"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT.parent / manifest_path
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(network.MODEL_PATH)
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped"
                    and row["repeat"] == 0 and row["policy"] == "queue_haul")
    plans, mixes, diffs, original = [], [], [], None
    for condition, label, germany_rho, bandwidth in STATES:
        scenario = _scenario(
            template, parent, condition, germany_rho, bandwidth)
        problem, architecture, routes, target, _ = network._scenario_problem(
            scenario, manifest, profile)
        result = network.solve(
            problem, profile, routes, "lp_work_first", destination=architecture,
            admission_mode="normal")
        execution = predict(
            _expected_scenario(problem, result.moves), profile, result.moves,
            destination=architecture)
        counts = network._constraint_action_counts(result.moves)
        violations = () if original is None else network._plan_violations(
            problem, profile, architecture, original)
        if original is None:
            original = result.moves
        diff = {"condition": condition, "label": label,
                **_diff(original, result.moves)}
        mix = {"condition": condition, "label": label,
               "planned_sessions": len(result.moves), **counts}
        plans.append({
            "condition": condition,
            "label": label,
            "bandwidth": bandwidth,
            "bandwidth_mbps": scenario["bandwidth_mbps"],
            "germany_prefill_rho": germany_rho,
            "requested_shed_w": float(target),
            "planned_shed_w": float(result.initial_source_power_w
                                     - result.planned_source_power_w),
            "simulated_shed_w": float(result.initial_source_power_w
                                       - execution.modeled_source_power_at_deadline_w),
            "planner_feasible": bool(result.feasible),
            "simulated_deadline_met": bool(execution.deadline_met),
            "simulated_makespan_s": float(execution.migration_makespan_s),
            "original_plan_violations": list(violations),
            "action_mix": counts,
            "moves": [asdict(move) for move in result.moves],
        })
        mixes.append(mix)
        diffs.append(diff)
    passed = all(plan["planner_feasible"] and plan["simulated_deadline_met"]
                 for plan in plans) \
        and all(plan["original_plan_violations"] for plan in plans[1:]) \
        and all(row["changed_sessions"] for row in diffs[1:]) \
        and len({tuple(row[key] for key, _, _ in MIX) for row in mixes}) == 3
    manifest_out = {
        "schema": "queue-haul-repair-plan-shift-simulation-v1",
        "semantics": "independent full snapshot replans",
        "parent_plan": str(parent_path),
        "parent_plan_sha256": profiler.file_hash(parent_path),
        "target_fraction": .5,
        "plans": plans,
    }
    profiler.write_json(out / "plans.json", manifest_out)
    profiler.write_csv(out / "action_mix.csv", mixes)
    profiler.write_csv(out / "plan_diffs.csv", diffs)
    _plot(mixes, out / "action_mix.png")
    report = {"schema": "queue-haul-repair-plan-shift-validation-v1",
              "plans": len(plans), "passed": passed}
    profiler.write_json(out / "validation.json", report)
    if not passed:
        raise RuntimeError("repair plan shift simulation failed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args(argv)
    run(args.out, args.parent)


if __name__ == "__main__":
    main()
