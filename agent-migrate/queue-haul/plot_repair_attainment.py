"""Reduce the prospective repair sweep to a paired attainment CDF.

Every selected cell holds the generated workload fixed and evaluates both the
original schedule and its residual repair under the same 10x-degraded timing.
Missing target times remain in the denominator as right-censored failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

import plot_style


SCHEMA = "queue-haul-repair-attainment-cdf-v1"
SOURCE_SCHEMA = "queue-haul-repair-stress-multiaxis-preflight-v3"
POPULATION_SCHEMA = "queue-haul-repair-attainment-population-v1"
COMMON_PHASE = "common_workload_multiaxis"
POLICIES = ("replan", "no_replan")
PANEL_FIGSIZE = (3.25, 2.25)
PANEL_FONT_SIZE = 9
PANEL_LEGEND_FONT_SIZE = 7.5
DEFAULT_SOURCE = Path(
    "outputs/repair-stress-multiaxis-hardware-20260814/preflight.json")
DEFAULT_OUT = Path("outputs/repair-attainment-simulation-20260817")

plot_style.apply()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(bundle: dict, target_fraction: float,
                population: str = "all") -> list[dict]:
    if bundle.get("schema") not in {SOURCE_SCHEMA, POPULATION_SCHEMA}:
        raise RuntimeError("unexpected repair preflight schema")
    axes = tuple(bundle["fault_axes"])
    seeds = tuple(bundle["context_seeds"])
    phase = COMMON_PHASE if bundle["schema"] == SOURCE_SCHEMA \
        else "attainment_population"
    selected = [row for row in bundle["cells"] if
                row["sweep_phase"] == phase and
                row["target_fraction"] == target_fraction]
    keys = [(row["context_seed"], row["fault_axis"]) for row in selected]
    expected = {(seed, axis) for seed in seeds for axis in axes}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("paired sweep is incomplete or duplicated")
    fixed = {
        (row["fault_at_s"], row["detection_at_s"],
         row["migration_cutoff_s"], row["observation_horizon_s"],
         row["healthy_east_load"], row["move_concurrency"])
        for row in selected
    }
    if len(fixed) != 1:
        raise RuntimeError("paired sweep mixes experiment settings")
    if population == "applied":
        selected = [row for row in selected if row["outcome"] == "applied"]
    elif population != "all":
        raise ValueError("population must be 'applied' or 'all'")
    rows = []
    for row in sorted(selected, key=lambda value: (
            value["context_seed"], value["fault_axis"])):
        rows.append({
            "context_seed": row["context_seed"],
            "fault_axis": row["fault_axis"],
            "target_fraction": row["target_fraction"],
            "requested_shed_w": row["requested_shed_w"],
            "repair_outcome": row["outcome"],
            "changed_sessions": row["diff"]["changed_sessions"],
            "repair_target_s": row["repair_target_s"],
            "no_replan_target_s": row["control_target_s"],
            "repair_by_cutoff": bool(row["repair_target_s"] is not None and
                                     row["repair_target_s"] <=
                                     row["migration_cutoff_s"]),
            "no_replan_by_cutoff": bool(
                row["control_target_s"] is not None and
                row["control_target_s"] <= row["migration_cutoff_s"]),
        })
    return rows


def attainment_curve(rows: list[dict], horizon_s: float,
                      cutoff_s: float) -> list[dict]:
    times = {0.0, float(cutoff_s), float(horizon_s)}
    for row in rows:
        for field in ("repair_target_s", "no_replan_target_s"):
            value = row[field]
            if value is not None and 0 <= value <= horizon_s:
                times.add(float(value))
    output = []
    for time_s in sorted(times):
        result = {"time_s": time_s}
        for policy, field in (("replan", "repair_target_s"),
                              ("no_replan", "no_replan_target_s")):
            attained = sum(row[field] is not None and row[field] <= time_s
                           for row in rows)
            result[f"{policy}_attained"] = attained
            result[f"{policy}_fraction"] = attained / len(rows)
        output.append(result)
    return output


def summarize(rows: list[dict], bundle: dict, target_fraction: float,
              population: str) -> dict:
    cutoff = float(rows and next(
        row["migration_cutoff_s"] for row in bundle["cells"]
        if row["target_fraction"] == target_fraction))
    horizon = float(next(
        row["observation_horizon_s"] for row in bundle["cells"]
        if row["target_fraction"] == target_fraction))

    def policy_summary(field: str) -> dict:
        return {
            "attained_by_25s": sum(row[field] is not None and
                                   row[field] <= cutoff for row in rows),
            "attainment_rate_by_25s": sum(
                row[field] is not None and row[field] <= cutoff
                for row in rows) / len(rows),
            "attained_by_120s": sum(row[field] is not None and
                                    row[field] <= horizon for row in rows),
            "attainment_rate_by_120s": sum(
                row[field] is not None and row[field] <= horizon
                for row in rows) / len(rows),
        }

    by_axis = {}
    for axis in bundle["fault_axes"]:
        axis_rows = [row for row in rows if row["fault_axis"] == axis]
        by_axis[axis] = {
            "paired_interventions": len(axis_rows),
            "replan_attained_by_25s": sum(row["repair_by_cutoff"]
                                           for row in axis_rows),
            "no_replan_attained_by_25s": sum(row["no_replan_by_cutoff"]
                                              for row in axis_rows),
        }
    return {
        "schema": SCHEMA,
        "claim_scope": (
            "simulator-generated interventions where repair produced a "
            "different feasible residual plan, selected without reference to "
            "completion time; both schedules use identical degraded timing; "
            "target nonattainment is right-censored and retained"),
        "target_fraction": target_fraction,
        "population": population,
        "workload_packs": len(bundle["context_seeds"]),
        "fault_axes": list(bundle["fault_axes"]),
        "paired_interventions": len(rows),
        "migration_cutoff_s": cutoff,
        "observation_horizon_s": horizon,
        "replan": policy_summary("repair_target_s"),
        "no_replan": policy_summary("no_replan_target_s"),
        "by_axis": by_axis,
        "repair_outcomes": {
            outcome: sum(row["repair_outcome"] == outcome for row in rows)
            for outcome in ("applied", "unchanged", "revised_maximum")
        },
    }


def plot(curve: list[dict], cutoff_s: float, horizon_s: float,
         output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 3))
    for policy in POLICIES:
        axis.step(
            [row["time_s"] for row in curve],
            [100 * row[f"{policy}_fraction"] for row in curve],
            where="post",
            color=plot_style.SCHEDULE_COMPARISON_COLORS[policy],
            linestyle=plot_style.SCHEDULE_COMPARISON_LINESTYLES[policy],
            label=plot_style.SCHEDULE_COMPARISON_NAMES[policy],
        )
    axis.axvline(
        cutoff_s, color=plot_style.EVENT_COLORS["shed_target"],
        linestyle=plot_style.EVENT_LINESTYLES["repair_decision"],
        linewidth=1.5, label="25 s cutoff",
    )
    axis.set(xlim=(0, horizon_s), ylim=(0, 100),
             xlabel="Time since migration start (s)",
             ylabel="Target attainment (%)")
    axis.grid(axis="y", alpha=.2)
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.legend(frameon=False, loc="lower right", fontsize=10)
    fig.subplots_adjust(left=.18, right=.98, bottom=.2, top=.97)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"),
                    dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def transition_summary(bundle: dict, target_fraction: float) -> list[dict]:
    actions = tuple(plot_style.REPAIR_ACTION_NAMES)
    rows = [row for row in bundle["cells"]
            if row["target_fraction"] == target_fraction
            and row["outcome"] == "applied"]
    output = []
    for axis in bundle["fault_axes"]:
        selected = [row for row in rows if row["fault_axis"] == axis]
        if not selected or any("transition_counts" not in row
                               for row in selected):
            raise RuntimeError("source lacks exact transition counts")
        counts = {action: sum(row["transition_counts"][action]
                              for row in selected) for action in actions}
        pending = sum(row["transition_counts"]["pending"] for row in selected)
        if sum(counts.values()) != pending:
            raise RuntimeError("transition counts do not conserve pending actions")
        output.append({
            "fault_axis": axis, "replans": len(selected),
            "pending_actions": pending,
            **{action: counts[action] for action in actions},
            **{f"{action}_fraction": counts[action] / pending
               for action in actions},
        })
    return output


def plot_response(curve: list[dict], transitions: list[dict], cutoff_s: float,
                  horizon_s: float, output: Path) -> None:
    actions = tuple(plot_style.REPAIR_ACTION_NAMES)
    fig, cdf_axis = plt.subplots(figsize=PANEL_FIGSIZE)
    for policy in POLICIES:
        cdf_axis.step(
            [row["time_s"] for row in curve],
            [100 * row[f"{policy}_fraction"] for row in curve],
            where="post", linewidth=1.4,
            color=plot_style.SCHEDULE_COMPARISON_COLORS[policy],
            linestyle=plot_style.SCHEDULE_COMPARISON_LINESTYLES[policy],
            label=plot_style.SCHEDULE_COMPARISON_NAMES[policy])
    cdf_axis.axvline(
        cutoff_s, color=plot_style.EVENT_COLORS["shed_target"],
        linestyle=plot_style.EVENT_LINESTYLES["repair_decision"],
        linewidth=1, label="25 s cutoff")
    cdf_axis.set(xlim=(0, horizon_s), ylim=(0, 100),
                 xticks=(0, 25, 60, 120), yticks=(0, 50, 100),
                 xlabel="Time (s)", ylabel="Target attainment (%)")
    cdf_axis.grid(axis="y", alpha=.2)
    cdf_axis.tick_params(labelsize=PANEL_FONT_SIZE)
    cdf_axis.xaxis.label.set_size(PANEL_FONT_SIZE)
    cdf_axis.yaxis.label.set_size(PANEL_FONT_SIZE)
    cdf_axis.legend(frameon=False, loc="lower right",
                    fontsize=PANEL_LEGEND_FONT_SIZE,
                    handlelength=1.5, borderpad=.1, labelspacing=.25)
    fig.subplots_adjust(left=.2, right=.97, bottom=.22, top=.97)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)

    fig, mix_axis = plt.subplots(figsize=PANEL_FIGSIZE)
    ordered = [next(row for row in transitions
                    if row["fault_axis"] == axis)
               for axis in ("bandwidth", "prefill", "joint")]
    left = [0.0] * len(ordered)
    for action in actions:
        values = [100 * row[f"{action}_fraction"] for row in ordered]
        mix_axis.barh(
            range(len(ordered)), values, left=left, height=.68,
            color=plot_style.REPAIR_ACTION_COLORS[action],
            hatch=plot_style.REPAIR_ACTION_HATCHES[action],
            edgecolor="white", linewidth=.4,
            label=plot_style.REPAIR_ACTION_SHORT_NAMES[action])
        left = [old + value for old, value in zip(left, values)]
    mix_axis.set(xlim=(0, 100), xticks=(0, 50, 100), xlabel="Actions (%)",
                 yticks=range(len(ordered)),
                 yticklabels=[plot_style.RESOURCE_FAULT_NAMES[
                     row["fault_axis"]] for row in ordered])
    mix_axis.invert_yaxis()
    mix_axis.grid(axis="x", alpha=.2)
    mix_axis.tick_params(labelsize=PANEL_FONT_SIZE)
    mix_axis.xaxis.label.set_size(PANEL_FONT_SIZE)
    mix_axis.yaxis.label.set_size(PANEL_FONT_SIZE)
    mix_axis.legend(frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(.5, -.62),
               fontsize=PANEL_LEGEND_FONT_SIZE, handlelength=1.5,
               columnspacing=.9,
               labelspacing=.25)
    fig.subplots_adjust(left=.24, right=.97, bottom=.42, top=.97)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_name("repair_actions").with_suffix(f".{suffix}"),
                    dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def run(source: Path, out: Path, target_fraction: float,
        population: str = "applied") -> dict:
    bundle = json.loads(source.read_text())
    rows = paired_rows(bundle, target_fraction, population)
    summary = summarize(rows, bundle, target_fraction, population)
    curve = attainment_curve(
        rows, summary["observation_horizon_s"],
        summary["migration_cutoff_s"])
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "paired_attainment.csv", rows)
    _write_csv(out / "attainment_cdf.csv", curve)
    summary["source"] = {"path": str(source), "sha256": _sha256(source),
                         "schema": bundle["schema"]}
    summary["source_interventions"] = len(bundle["cells"])
    summary["source_outcomes"] = {
        outcome: sum(row["outcome"] == outcome for row in bundle["cells"])
        for outcome in ("applied", "unchanged", "revised_maximum")
    }
    summary["inputs"] = {key: bundle[key] for key in
                         ("base_plan", "timing", "parent", "manifest",
                          "model_profile")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(curve, summary["migration_cutoff_s"],
         summary["observation_horizon_s"], out / "attainment_cdf")
    if all("transition_counts" in row for row in bundle["cells"]):
        transitions = transition_summary(bundle, target_fraction)
        _write_csv(out / "plan_changes.csv", transitions)
        plot_response(curve, transitions, summary["migration_cutoff_s"],
                      summary["observation_horizon_s"], out / "repair_response")
    (out / "README.md").write_text(
        "# Paired repair-attainment simulation\n\n"
        f"This artifact contains {summary['paired_interventions']:,} paired "
        f"actual plan shifts drawn from {summary['workload_packs']:,} "
        "independently "
        "generated workload packs crossed with bandwidth, prefill, and joint "
        "10x resource drops. The original and repaired residual schedules are "
        "both replayed under the same degraded timing. Curves retain failures "
        "to attain the target by 120 s in the denominator; they are not "
        "bootstrap replicas of the hardware cases.\n\n"
        "`repair_response.pdf` shows the paired attainment CDF. "
        "`repair_actions.pdf` classifies each original pending action as "
        "retained, changed, redirected, or removed. Exact pooled counts are "
        "in `plan_changes.csv`.\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-fraction", type=float, default=.50)
    parser.add_argument("--population", choices=("applied", "all"),
                        default="applied")
    args = parser.parse_args()
    run(args.source, args.out, args.target_fraction, args.population)


if __name__ == "__main__":
    main()
