"""Simulate three planner snapshots and plot their action mix."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import migration_profiler as profiler
import network_campaign as network
import plot_style
from planner import _expected_scenario
from profiles import ModelProfile
from simulate import predict


ROOT = Path(__file__).parent
DEFAULT_PARENT = ROOT / "outputs/east-germany-separation-20260809/plan.json"
GERMANY_BANDWIDTH_MBPS = 2200
STATES = (
    ("original", "Original", .25, .25, "natural"),
    ("germany_bandwidth_drop", "Germany bandwidth", .25, .25, "germany_drop"),
    ("east_prefill_drop", "East prefill", .976, .25, "natural"),
)
MIX = tuple(plot_style.ACTION_NAMES)[2:6]
plot_style.apply()


def _scenario(template, parent, condition, east_rho, germany_rho, bandwidth):
    rates = network._bandwidths(parent["network_contract"], "natural")
    if bandwidth == "germany_drop":
        rates["germany"] = GERMANY_BANDWIDTH_MBPS
    return {
        **template,
        "design": "repair_plan_shift_simulation",
        "condition_id": condition,
        "background": {"east": (east_rho, 0), "germany": (germany_rho, 0)},
        "bandwidth": bandwidth,
        "bandwidth_mbps": rates,
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
    labels = {
        "east_replay": "Replay → eastus-2",
        "east_kv_transfer": "KV → eastus-2",
        "germany_replay": "Replay → germany-west-central",
        "germany_kv_transfer": "KV → germany-west-central",
    }
    fig, axis = plt.subplots(figsize=(7, 3))
    left = np.zeros(len(rows))
    for action in MIX:
        values = np.array([row[action] / row["planned_sessions"]
                           for row in rows]) * 100
        axis.barh(range(len(rows)), values, left=left,
                  color=plot_style.ACTION_COLORS[action],
                  hatch=plot_style.ACTION_HATCHES[action], edgecolor="white",
                  linewidth=1.2, label=labels[action])
        left += values
    axis.set(yticks=range(len(rows)),
             yticklabels=[row["label"] for row in rows], xlim=(0, 100),
             xlabel="Selected-action share (%)")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(.6, -.01), fontsize=10, handlelength=1.8)
    fig.subplots_adjust(left=.23, right=.97, bottom=.42, top=.96)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
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
    for condition, label, east_rho, germany_rho, bandwidth in STATES:
        scenario = _scenario(
            template, parent, condition, east_rho, germany_rho, bandwidth)
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
            "east_prefill_rho": east_rho,
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
        and len({tuple(row[key] for key in MIX) for row in mixes}) == 3
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
