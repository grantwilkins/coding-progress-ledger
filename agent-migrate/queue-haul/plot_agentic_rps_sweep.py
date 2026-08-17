"""Plot the two-panel, three-model agentic RPS sweep figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import agentic_rps_sweep_campaign as campaign
import plot_style


plot_style.apply()


METRICS = (
    ("p90_ttft_s", "P90 TTFT (s)"),
    ("p90_mean_tpot_s", "P90 mean TPOT (s)"),
)


def error_range(row: dict, field: str) -> np.ndarray | None:
    center = row.get(f"{field}_median")
    low = row.get(f"{field}_minimum")
    high = row.get(f"{field}_maximum")
    if row.get("repeats", 0) < 3 or None in (center, low, high):
        return None
    return np.asarray([[center - low], [high - center]])


def plot_summary(summary: dict, output: Path) -> None:
    figure, axes = plt.subplots(
        1, 2,
        figsize=(plot_style.WIDE_FIGSIZE[0], 4.8),
        sharex=True,
    )
    for axis, (field, ylabel) in zip(axes, METRICS):
        for model in campaign.MODELS:
            result = summary["models"][model]
            rows = sorted(result["curve"],
                          key=lambda row: row["offered_rps"])
            usable = [row for row in rows
                      if row.get(f"{field}_median") is not None]
            color = plot_style.MODEL_COLORS[model]
            axis.plot(
                [row["offered_rps"] for row in usable],
                [row[f"{field}_median"] for row in usable],
                color=color,
                linestyle=plot_style.MODEL_LINESTYLES[model],
                marker=plot_style.MODEL_MARKERS[model],
                label=plot_style.MODEL_NAMES[model],
            )
            for row in usable:
                error = error_range(row, field)
                if error is not None:
                    axis.errorbar(
                        [row["offered_rps"]], [row[f"{field}_median"]],
                        yerr=error, fmt="none", ecolor=color,
                        elinewidth=1.4, capsize=3, zorder=4,
                    )
            target = result["slo"].get(field)
            if target is not None:
                axis.axhline(
                    target, color=color, linestyle=(0, (4, 2, 1, 2)),
                    linewidth=1.2, alpha=.7,
                )
            violation = result.get("first_confirmed_violation_rps")
            violated_row = next((row for row in usable
                                 if row["offered_rps"] == violation), None)
            if violated_row is not None and target is not None \
                    and violated_row[f"{field}_median"] > target:
                axis.scatter(
                    [violation], [violated_row[f"{field}_median"]],
                    marker="X", s=72, color=color, edgecolors="white",
                    linewidths=.8, zorder=5,
                )
        axis.set_xlabel("Total offered RPS")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, max(campaign.RATES_RPS) + .15)
        axis.set_xticks(range(0, int(max(campaign.RATES_RPS)) + 1))
        axis.grid(alpha=.22)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color="#555555", linestyle=(0, (4, 2, 1, 2)),
               linewidth=1.2, label="Model SLO"),
        Line2D([0], [0], marker="X", linestyle="none", color="#555555",
               markeredgecolor="white", markersize=7,
               label="First confirmed violation"),
    ])
    labels.extend(["Model SLO", "First confirmed violation"])
    figure.legend(
        handles, labels, frameon=False,
        fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE, ncol=3,
        loc="upper center", bbox_to_anchor=(.5, .99),
    )
    figure.subplots_adjust(
        left=.11, right=.98, bottom=.16, top=.82, wspace=.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix(f".{suffix}"),
                       dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot(summary: dict, output_dir: Path) -> Path:
    if summary.get("schema") != campaign.SCHEMA \
            or summary.get("stage") != "reduced":
        raise ValueError("agentic RPS summary is not reduced evidence")
    output = output_dir / "agentic-rps-sweep"
    plot_summary(summary, output)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    plot(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
