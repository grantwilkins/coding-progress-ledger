"""Plot held-out H100 timing predictions against live measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style
from live_timing_campaign import collect
from profiles import ModelProfile


ACTIONS = ("replay", "kv_transfer")
REGIONS = {"australiaeast": ("Australia East", "o"),
           "southcentralus": ("South Central US", "s")}
plot_style.apply()


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    required = {(action, region) for action in ACTIONS for region in REGIONS}
    if not rows or {row["split"] for row in rows} != {"holdout"} \
            or {(row["method"], row["destination"]) for row in rows} != required:
        raise ValueError("parity plot requires the complete unseen four-path holdout")
    return [{"action": row["method"], "region": row["destination"],
             "cohort": "new_holdout",
             "scenario_id": row["scenario_id"],
             "context_tokens": int(row["context_tokens"]),
             "predicted_s": float(row["predicted_s"]),
             "measured_s": float(row["initial_time_to_first_response_s"])}
            for row in rows]


def load_history(run_root: Path, model_path: Path, profile_path: Path) -> list[dict]:
    model = json.loads(model_path.read_text())
    curve = np.asarray(model["replay_tps"]).T
    kv = ModelProfile.load(profile_path).case().kv_transfer
    rows = collect(run_root)
    if not rows or any(not model["valid_context_tokens"][0]
                       <= row["context_tokens"]
                       <= model["valid_context_tokens"][1] for row in rows):
        raise ValueError("historical timing lies outside the fitted domain")
    out = []
    for row in rows:
        if row["method"] == "replay":
            predicted = row["context_tokens"] / np.interp(
                row["context_tokens"], *curve)
        else:
            size = kv.sealed_bytes(row["context_tokens"])
            predicted = model["kv_initial_completion_s"] + max(
                size / model["kv_effective_path_bytes_per_s"][row["destination"]],
                size / model["kv_destination_bytes_per_s"])
        out.append({"action": row["method"], "region": row["destination"],
                    "cohort": "historical", "scenario_id": row["scenario_id"],
                    "context_tokens": row["context_tokens"],
                    "predicted_s": float(predicted),
                    "measured_s": row["initial_time_to_first_response_s"]})
    return out


def queue_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["scenario_id"], []).append(row)
    out = []
    for scenario_id, group in groups.items():
        paths = {}
        for row in group:
            key = row["region"], row["action"]
            paths[key] = paths.get(key, 0) + row["predicted_s"]
        actions = {row["action"] for row in group}
        out.append({"scenario_id": scenario_id,
                    "action": next(iter(actions)) if len(actions) == 1 else "mixed",
                    "predicted_s": max(paths.values()),
                    "measured_s": max(row["measured_s"] for row in group)})
    return out


def write(rows: list[dict], out: Path, log_scale: bool = False) -> None:
    limit = 1.04 * max(row[key] for row in rows
                       for key in ("predicted_s", "measured_s"))
    fig, axis = plt.subplots(figsize=plot_style.COMPACT_FIGSIZE)
    axis.plot([0, limit], [0, limit], color="black", linestyle="--",
              linewidth=1.5, label="Prediction = measurement")
    has_history = any(row["cohort"] == "historical" for row in rows)
    for action in ACTIONS:
        for region, (region_name, marker) in REGIONS.items():
            selected = [row for row in rows
                        if row["action"] == action and row["region"] == region]
            history = [row for row in selected if row["cohort"] == "historical"]
            shown = history or selected
            axis.scatter([row["predicted_s"] for row in shown],
                         [row["measured_s"] for row in shown],
                         color=plot_style.ACTION_COLORS[action], marker=marker,
                         s=14 if history else 45, alpha=.22 if history else .8,
                         label=f"{plot_style.ACTION_NAMES[action]} — {region_name}")
            if history:
                holdout = [row for row in selected if row["cohort"] == "new_holdout"]
                axis.scatter([row["predicted_s"] for row in holdout],
                             [row["measured_s"] for row in holdout],
                             color=plot_style.ACTION_COLORS[action], marker=marker,
                             edgecolor="black", linewidth=.5, s=45, alpha=.9)
    axis.set(xlabel="Predicted TTFT (s)", ylabel="Measured TTFT (s)",
             xlim=(0, limit), ylim=(0, limit),
             title=("All live H100 transfers" if has_history
                    else "Unseen live holdout") + f" (n={len(rows):,})")
    if log_scale:
        lower = .9 * min(row[key] for row in rows
                         for key in ("predicted_s", "measured_s"))
        axis.set(xscale="log", yscale="log", xlim=(lower, limit),
                 ylim=(lower, limit))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_queue(rows: list[dict], out: Path) -> None:
    limits = (1, 200)
    fig, axis = plt.subplots(figsize=plot_style.HALF_COLUMN_FIGSIZE)
    axis.plot(limits, limits, color="black", linestyle="--",
              linewidth=1.5, label="Ideal")
    for action in ("replay", "kv_transfer", "mixed"):
        selected = [row for row in rows if row["action"] == action]
        if selected:
            axis.scatter([row["predicted_s"] for row in selected],
                         [row["measured_s"] for row in selected],
                         color=plot_style.TIMING_ACTION_COLORS[action], s=9,
                         alpha=.58, label=plot_style.TIMING_ACTION_NAMES[action])
    measured = np.array([row["measured_s"] for row in rows])
    residuals = measured - np.array([row["predicted_s"] for row in rows])
    r2 = 1 - residuals @ residuals / sum((measured - measured.mean()) ** 2)
    axis.set(xlabel="Predicted Full\nMigration (s)",
             ylabel="Measured Full Migration (s)",
             xscale="log", yscale="log", xlim=limits, ylim=limits)
    axis.text(.98, .03,
              f"MAE {np.mean(abs(residuals)):.2f} s\n$R^2$ {r2:.3f}",
              ha="right", va="bottom", transform=axis.transAxes,
              fontsize=plot_style.HALF_COLUMN_ANNOTATION_FONT_SIZE)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2, linewidth=.4)
    plot_style.half_column(axis)
    axis.legend(frameon=False, fontsize=plot_style.HALF_COLUMN_LEGEND_FONT_SIZE,
                loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2,
                columnspacing=.2, handlelength=1, handletextpad=.25,
                labelspacing=.2, borderaxespad=0)
    fig.tight_layout(pad=.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--history-run-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--queue-out", type=Path)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len([value for value in (args.history_run_root, args.model, args.profile)
            if value]) not in (0, 3):
        parser.error("--history-run-root, --model, and --profile must be provided together")
    rows = load(args.source)
    if args.history_run_root:
        history = load_history(args.history_run_root, args.model, args.profile)
        rows += history
        if args.queue_out:
            write_queue(queue_rows(history), args.queue_out)
    elif args.queue_out:
        parser.error("--queue-out requires historical inputs")
    write(rows, args.out, args.log)


if __name__ == "__main__":
    main()
