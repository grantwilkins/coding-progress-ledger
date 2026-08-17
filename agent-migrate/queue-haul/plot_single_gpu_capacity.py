"""Plot non-gating single-A100 model/context capacity discovery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

import plot_style


plot_style.apply()


def plot(summary: dict, output: Path) -> None:
    rows = summary["rows"]
    maximum = max((row[field] for row in rows for field in (
        "max_peak_running_requests", "max_completed_burst_width")), default=1)
    upper = 2 ** math.ceil(math.log2(maximum * 1.15))
    ticks = [2 ** exponent for exponent in range(int(math.log2(upper)) + 1)]
    fig, axes = plt.subplots(1, 2, figsize=plot_style.WIDE_FIGSIZE,
                             sharex=True, sharey=True)
    fields = (
        ("max_peak_running_requests", "Maximum simultaneous\nrunning requests"),
        ("max_completed_burst_width",
         "Largest tested completed burst\n(lower bound)"),
    )
    for axis, (field, title) in zip(axes, fields):
        for model in plot_style.MODELS:
            model_rows = sorted(
                (row for row in rows if row["model"] == model),
                key=lambda row: row["context_tokens"],
            )
            launched = [row for row in model_rows if row["launchable"]]
            if launched:
                axis.plot(
                    [row["context_tokens"] / 1000 for row in launched],
                    [max(1, row[field]) for row in launched],
                    color=plot_style.MODEL_COLORS[model],
                    linestyle=plot_style.MODEL_LINESTYLES[model],
                    marker=plot_style.MODEL_MARKERS[model],
                    label=plot_style.MODEL_NAMES[model],
                )
                right_censored = [
                    row for row in launched
                    if field == "max_completed_burst_width"
                    and row["right_censored"]
                ]
                if right_censored:
                    axis.scatter(
                        [row["context_tokens"] / 1000 for row in right_censored],
                        [max(1, row[field]) for row in right_censored],
                        marker=plot_style.MODEL_MARKERS[model], s=92,
                        facecolors="white",
                        edgecolors=plot_style.MODEL_COLORS[model],
                        linewidths=1.7, zorder=4,
                    )
            failed = [row for row in model_rows if not row["launchable"]]
            if failed:
                axis.scatter(
                    [row["context_tokens"] / 1000 for row in failed],
                    [1 for _ in failed], marker="x", s=72,
                    color=plot_style.MODEL_COLORS[model], linewidths=2,
                    zorder=5,
                )
        axis.set_title(title)
        axis.set_yscale("log", base=2)
        axis.set_yticks(ticks, labels=[str(tick) for tick in ticks])
        axis.grid(True, which="both", alpha=.22)
    axes[0].set_ylim(.8, upper)
    axes[0].set_ylabel("Requests")
    axes[0].legend(frameon=False, fontsize=plot_style.LEGEND_FONT_SIZE)
    fig.supxlabel("Prompt context (thousand tokens)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    plot(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
