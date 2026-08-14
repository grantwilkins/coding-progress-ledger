"""Bootstrap Queue-Haul action mixes over calibrated timing and power."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import network_campaign as campaign
import plot_style
import testbed_calibration_campaign as calibration
from plot_pooled_action_adaptation import (
    ACTION_MIX_CASES, ACTION_MIX_LABEL_SIZE, ACTION_MIX_LEGEND_SIZE,
    ACTION_MIX_TICK_SIZE, constraint_scenarios, pooled_cases, solve_case,
)
from plot_pooled_shed_frontier import write_csv
from profiles import ModelProfile
from stress_frontier_campaign import state_profile


ACTIONS = ("replay", "kv_transfer", "not_moved")
MODES = ("timing", "power", "joint")
PROFILE = Path("outputs/azure-compact-calibration-20260813/") / \
    "gpt_oss_20b_a100_tp1_azure_300w_phase.json"
TIMING = Path("outputs/timing-power-validation-20260814/migrations.csv")
TIMING_SUMMARY = Path("outputs/timing-power-validation-20260814/timing-summary.json")
FIGSIZE = (5.5, 3)
plot_style.apply()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_timing_bootstrap(rows: list[dict], rng) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in (
            "destination", "method", "bandwidth", "context_tokens"))
        groups.setdefault(key, []).append(row)
    if len(rows) != 108 or len(groups) != 36 or any(len(group) != 3
                                                    for group in groups.values()):
        raise ValueError("timing bootstrap requires 36 balanced three-repeat cells")
    return [group[index] for group in groups.values()
            for index in rng.integers(0, 3, 3)]


def timing_fit(profile, parent, rows, provenance):
    fits, _, contexts = calibration.regional_timing_model(
        profile, parent, rows, provenance)
    if contexts != [1536, 7680, 32256]:
        raise ValueError("timing support changed")
    return fits


def calibrated_scenario(scenario: dict, fits: dict) -> dict:
    label = scenario["bandwidth"]
    return {
        **scenario,
        "migration_components": {
            node: fit["migration_components"] for node, fit in fits.items()},
        "bandwidth_mbps": {
            node: fit["effective_pipeline_mbps"][label]
            for node, fit in fits.items()},
    }


def power_draw(profile: ModelProfile, index: int) -> ModelProfile:
    return state_profile(profile, {
        "power_bootstrap_index": index, "service_multiplier": 1,
        "replay_multiplier": 1, "kv_multiplier": 1,
    })


def action_row(case, label, replicate, mode, result, sessions, target_w,
               power_index):
    counts = campaign._constraint_action_counts(result.moves)
    replay = counts["east_replay"] + counts["germany_replay"]
    kv = counts["east_kv_transfer"] + counts["germany_kv_transfer"]
    moved = len(result.moves)
    if replay + kv != moved or not 0 <= moved <= sessions:
        raise RuntimeError("bootstrap action counts do not conserve sessions")
    return {
        "case_id": case, "bound_constraint": label, "replicate": replicate,
        "mode": mode, "power_bootstrap_index": power_index,
        "target_w": target_w, "target_met": result.power_shortfall_w == 0,
        "replay_count": replay, "kv_transfer_count": kv,
        "not_moved_count": sessions - moved,
        "replay": replay / sessions, "kv_transfer": kv / sessions,
        "not_moved": 1 - moved / sessions,
    }


def validate_support(problem, result, profile, fits):
    horizon = problem.deadline_s - problem.controller_delay_s - profile.power_window_s
    contexts = [int(np.ceil(session.context_tokens
                            + session.expected_growth_tokens_per_s * horizon))
                for session in problem.sessions]
    bounds = next(iter(fits.values()))["migration_components"]["replay"][
        "context_range"]
    moved = {move.session_id for move in result.moves}
    remaining = [session for session in problem.sessions
                 if session.session_id not in moved]
    phase = profile.case().phase_power
    if min(contexts) < bounds[0] or max(contexts) > bounds[1] \
            or phase is None or not phase.contains(
                sum(session.expected_f for session in problem.sessions),
                sum(session.expected_g for session in problem.sessions)) \
            or not phase.contains(sum(session.expected_f for session in remaining),
                                  sum(session.expected_g for session in remaining)):
        raise RuntimeError("bootstrap draw extrapolates beyond calibration support")


def simulate(plan_paths, profile_path=PROFILE, timing_path=TIMING,
             samples=1000, seed=1):
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    profile, timing_rows = ModelProfile.load(profile_path), read_csv(timing_path)
    cases = {case: (scenario, manifest)
             for case, scenario, manifest in pooled_cases(plan_paths)}
    bound, manifest = cases["hardware_gap/all-bind"]
    released, released_manifest = cases["hardware_gap/all-release"]
    if manifest != released_manifest:
        raise RuntimeError("bootstrap cases require one matched pack")
    scenarios = constraint_scenarios(bound, released)
    if tuple(scenarios) != tuple(case for case, _ in ACTION_MIX_CASES):
        raise RuntimeError("constraint truth table changed")
    parent = json.loads(next(path for path in plan_paths
                             if "separation" in str(path)).read_text())
    central = timing_fit(profile, parent, timing_rows, str(timing_path))
    expected = json.loads(TIMING_SUMMARY.read_text())["fits"]
    if any(not np.isclose(central[node][key], expected[node][key])
           for node in central for key in (
               "replay_compute_completion_factor", "kv_residual_s",
               "kv_ingest_lower_bound_bytes_per_s")) or any(
                   not np.isclose(rate, expected[node][
                       "effective_pipeline_mbps"][label])
                   for node in central for label, rate in central[node][
                       "effective_pipeline_mbps"].items()):
        raise RuntimeError("central timing refit does not reproduce summary")
    rng = np.random.default_rng(seed)
    draws = [(timing_fit(profile, parent,
                         stratified_timing_bootstrap(timing_rows, rng),
                         str(timing_path)),
              int(rng.integers(len(profile.case().phase_power.bootstrap))))
             for _ in range(samples)]
    labels, rows = dict(ACTION_MIX_CASES), []
    for mode in (*MODES, "central"):
        targets = {}
        selected_draws = draws if mode != "central" else [(central, -1)]
        for replicate, (sampled_timing, power_index) in enumerate(selected_draws):
            fit = sampled_timing if mode in {"timing", "joint"} else central
            sampled_profile = power_draw(profile, power_index) \
                if mode in {"power", "joint"} else profile
            for case, scenario in scenarios.items():
                calibrated = calibrated_scenario(scenario, fit)
                problem, result = solve_case(calibrated, manifest,
                                             sampled_profile, 2 / 3)
                initial = campaign.source_power(problem, sampled_profile)
                minimum = campaign.source_power(
                    problem, sampled_profile,
                    (session.session_id for session in problem.sessions))
                target = 2 / 3 * (initial - minimum)
                targets.setdefault(replicate, target)
                if not np.isclose(targets[replicate], target):
                    raise RuntimeError("paired cases do not share one target")
                validate_support(problem, result, sampled_profile, fit)
                rows.append(action_row(
                    case, labels[case], replicate, mode, result,
                    len(problem.sessions), target,
                    power_index if mode in {"power", "joint"} else -1))
    if len(rows) != len(ACTION_MIX_CASES) * (samples * len(MODES) + 1):
        raise RuntimeError("bootstrap campaign is incomplete")
    return rows


def summarize(rows):
    output = []
    for mode in MODES:
        for case, label in ACTION_MIX_CASES:
            selected = [row for row in rows if row["mode"] == mode
                        and row["case_id"] == case]
            for action in ACTIONS:
                values = np.asarray([row[action] for row in selected])
                q = np.quantile(values, (.05, .25, .5, .75, .95))
                output.append({
                    "mode": mode, "bound_constraint": label, "action": action,
                    "mean": values.mean(), "p05": q[0], "p25": q[1],
                    "median": q[2], "p75": q[3], "p95": q[4],
                    "target_met_rate": np.mean([row["target_met"]
                                                for row in selected]),
                })
    return output


def plot_intervals(summary, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    joint = [row for row in summary if row["mode"] == "joint"]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)
    for axis, action in zip(axes, ACTIONS):
        selected = {row["bound_constraint"]: row for row in joint
                    if row["action"] == action}
        color = plot_style.ACTION_COLORS[action]
        for y, (_, label) in enumerate(ACTION_MIX_CASES):
            row = selected[label]
            axis.hlines(y, 100 * float(row["p05"]), 100 * float(row["p95"]),
                        color=color, linewidth=1)
            axis.hlines(y, 100 * float(row["p25"]), 100 * float(row["p75"]),
                        color=color, linewidth=4)
            axis.scatter(100 * float(row["median"]), y,
                         color=color, s=18, zorder=3)
        axis.set(title=plot_style.ACTION_NAMES[action], xlim=(0, 100),
                 xticks=(0, 50, 100), ylim=(-.5, len(ACTION_MIX_CASES) - .5))
        axis.title.set_size(10)
        axis.xaxis.set_major_formatter(PercentFormatter())
        axis.grid(axis="x", alpha=.2)
        axis.invert_yaxis()
        axis.tick_params(labelsize=9)
    axes[0].set_yticks(range(len(ACTION_MIX_CASES)),
                       [label for _, label in ACTION_MIX_CASES])
    fig.supxlabel("Source-session share", y=.03, fontsize=12)
    fig.subplots_adjust(left=.36, right=.99, bottom=.19, top=.87, wspace=.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def plot(rows, out):
    import matplotlib.pyplot as plt

    joint = [row for row in rows if row["mode"] == "joint"]
    fig, axis = plt.subplots(figsize=FIGSIZE)
    left = np.zeros(len(ACTION_MIX_CASES))
    for action in ACTIONS:
        values = np.asarray([np.mean([
            float(row[action]) for row in joint if row["case_id"] == case])
            for case, _ in ACTION_MIX_CASES]) * 100
        axis.barh(range(len(values)), values, left=left,
                  color=plot_style.ACTION_COLORS[action],
                  hatch=plot_style.ACTION_HATCHES[action], edgecolor="white",
                  linewidth=1.2, label=plot_style.ACTION_NAMES[action])
        left += values
    for y, (case, _) in enumerate(ACTION_MIX_CASES):
        selected = [row for row in joint if row["case_id"] == case]
        for values in ([float(row["replay"]) for row in selected],
                       [float(row["replay"]) + float(row["kv_transfer"])
                        for row in selected]):
            low, center, high = 100 * np.quantile(values, (.05, .5, .95))
            axis.errorbar(center, y, xerr=[[center - low], [high - center]],
                          fmt="|", color="black", linewidth=.8,
                          capsize=2, markersize=5, zorder=4)
    axis.set(yticks=range(len(ACTION_MIX_CASES)),
             yticklabels=[label for _, label in ACTION_MIX_CASES],
             xlim=(0, 100), xlabel="Source-session share (%)")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(labelsize=ACTION_MIX_TICK_SIZE)
    axis.xaxis.label.set_size(ACTION_MIX_LABEL_SIZE)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(.58, .01),
               fontsize=ACTION_MIX_LEGEND_SIZE, handlelength=1.8)
    fig.subplots_adjust(left=.36, right=.97, bottom=.37, top=.96)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--timing", type=Path, default=TIMING)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = simulate(args.plan, args.profile, args.timing,
                    args.samples, args.seed)
    summary = summarize(rows)
    write_csv(rows, args.out_dir / "bootstrapped_action_mix.csv")
    write_csv(summary, args.out_dir / "bootstrapped_action_mix_summary.csv")
    plot(rows, args.out_dir / "bootstrapped_action_mix")
    plot_intervals(summary, args.out_dir / "bootstrapped_action_mix_intervals")
    phase = ModelProfile.load(args.profile).case().phase_power
    metadata = {
        "schema": "queue-haul-action-calibration-bootstrap-v1",
        "claim": "modeled calibration-sensitivity distribution",
        "samples_per_case": args.samples, "seed": args.seed,
        "joint_rows": args.samples * len(ACTION_MIX_CASES),
        "power_model": "ell = 0.0001783544319906395 f + 0.009287706208604906 g",
        "power_bootstrap_near_zero_prefill_fraction": float(np.mean(
            [row[2] < 1e-8 for row in phase.bootstrap])),
        "joint_target_met_rate": {
            label: float(next(row["target_met_rate"] for row in summary
                              if row["mode"] == "joint"
                              and row["bound_constraint"] == label))
            for _, label in ACTION_MIX_CASES},
        "limitations": [
            "fixed 28-session pack; not independent workload observations",
            "timing telemetry covers concurrency one and zero destination prefill load",
            "constraint combinations are modeled hybrids, not eight hardware runs",
            "phase-power calibration is fitted sensitivity evidence, not promotion-ready",
        ],
    }
    (args.out_dir / "bootstrapped_action_mix_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
