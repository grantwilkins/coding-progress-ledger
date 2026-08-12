"""Plot predicted versus measured source-power shed across hardware runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style


ROOT = Path(__file__).parent
SOURCE = ROOT / "outputs/power_drain_live_20260714/scenario_summary.csv"
OUTPUT = ROOT / "outputs/hardware_power_shed_parity"
METHODS = ("lp", "greedy")
plot_style.apply()


def load_points(path: Path) -> tuple[list[dict], float]:
    with path.open() as handle:
        source = list(csv.DictReader(handle))
    scale = max(float(row["target_w"]) for row in source)
    if not source or scale <= 0 or not set(METHODS) <= {
            row["policy"] for row in source}:
        raise ValueError("power validation requires both methods and a positive request")
    rows = [{
        "workload": row["workload"], "method": row["policy"],
        "seed": int(row["seed"]), "requested_shed_w": float(row["target_w"]),
        "predicted_shed_w": float(row["planned_source_drop_w"]),
        "measured_shed_w": float(row["measured_source_drop_w"]),
        "predicted_percent": 100 * float(row["planned_source_drop_w"]) / scale,
        "measured_percent": 100 * float(row["measured_source_drop_w"]) / scale,
    } for row in source if row["policy"] in METHODS]
    return rows, scale


def write_plot(rows: list[dict], scale: float, out: Path) -> None:
    values = [row[key] for row in rows
              for key in ("predicted_percent", "measured_percent")]
    lower, upper = min(-5, min(values)), max(105, max(values))
    padding = .03 * (upper - lower)
    limits = lower - padding, upper + padding
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.5,
              label="Prediction = measurement", zorder=1)
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        axis.scatter(
            [row["predicted_percent"] for row in selected],
            [row["measured_percent"] for row in selected],
            color=plot_style.POWER_VALIDATION_COLORS[method],
            marker=plot_style.POWER_VALIDATION_MARKERS[method], s=34,
            alpha=.7, linewidths=1,
            label=plot_style.POWER_VALIDATION_NAMES[method], zorder=2,
        )
    axis.set(xlabel="Predicted shed (% of max request)",
             ylabel="Measured shed (% of max request)",
             xlim=limits, ylim=limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=1, fontsize=9,
               loc="center left", bbox_to_anchor=(.65, .57))
    fig.text(.98, .02, f"Maximum request = {scale:.1f} W; {len(rows)} runs",
             ha="right", fontsize=9)
    fig.subplots_adjust(left=.13, right=.63, bottom=.17, top=.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    write_plot(*load_points(args.source), args.out)


if __name__ == "__main__":
    main()
