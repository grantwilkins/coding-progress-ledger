"""Plot pre-run modeled versus measured A100 migration makespan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import plot_style


METHODS = ("queue_haul", "greedy")
MARKERS = {"queue_haul": "o", "greedy": "s"}
plot_style.apply()


def paired_points(predictions: Path, measurements: Path):
    with predictions.open() as stream:
        predicted = list(csv.DictReader(stream))
    index = {(row["condition_index"], row["repeat"], row["policy"]): row
             for row in predicted if row["policy"] in METHODS}
    if len(index) != sum(row["policy"] in METHODS for row in predicted):
        raise RuntimeError("duplicate modeled timing key")
    with measurements.open() as stream:
        measured = [row for row in csv.DictReader(stream)
                    if row["policy"] in METHODS]
    keys = {(row["condition_index"], row["repeat"], row["policy"])
            for row in measured}
    if keys != set(index) or len(measured) != len(keys):
        raise RuntimeError("modeled and measured timing episodes are not matched")
    return [{
        "condition_index": row["condition_index"], "repeat": row["repeat"],
        "policy": row["policy"],
        "predicted_makespan_s": float(index[
            row["condition_index"], row["repeat"], row["policy"]
        ]["predicted_makespan_s"]),
        "measured_makespan_s": float(row["migration_s"]),
    } for row in measured]


def metrics(rows):
    summary = []
    for policy in METHODS:
        selected = [row for row in rows if row["policy"] == policy]
        errors = np.asarray([row["measured_makespan_s"]
                             - row["predicted_makespan_s"]
                             for row in selected])
        summary.append({
            "policy": policy, "episodes": len(selected),
            "bias_s": float(errors.mean()),
            "mae_s": float(np.abs(errors).mean()),
            "rmse_s": float(np.sqrt(np.mean(errors ** 2))),
        })
    return summary


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [row[key] for row in rows
              for key in ("predicted_makespan_s", "measured_makespan_s")]
    lower, upper = min(values), max(values)
    pad = .06 * (upper - lower)
    limits = lower - pad, upper + pad
    fig, axis = plt.subplots(figsize=plot_style.COMPACT_FIGSIZE)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.5)
    for policy in METHODS:
        selected = [row for row in rows if row["policy"] == policy]
        axis.scatter(
            [row["predicted_makespan_s"] for row in selected],
            [row["measured_makespan_s"] for row in selected],
            color=plot_style.POLICY_COLORS[policy], marker=MARKERS[policy],
            s=55, alpha=.65, linewidths=.8,
            label=plot_style.POLICY_NAMES[policy],
        )
    axis.set(xlim=limits, ylim=limits,
             xlabel="Predicted Migration Time (s)",
             ylabel="Measured Migration Time (s)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2)
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, .5),
                borderaxespad=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = paired_points(args.predictions, args.measurements)
    write_csv(rows, args.out.with_suffix(".csv"))
    write_csv(metrics(rows), args.out.with_name(f"{args.out.name}_summary.csv"))
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
