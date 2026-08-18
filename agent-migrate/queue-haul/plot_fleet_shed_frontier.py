"""Plot the fleet-scale deadline-to-power-shed frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

import plot_style

plot_style.apply()

POLICIES = ("queue_haul", "greedy", "replay_only", "isolated_fastest", "kv_only")


def read(path: Path, mode: str = "normal") -> dict:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["mode"] == mode]
    if not rows:
        raise RuntimeError(f"no {mode!r} rows in {path}")
    curves = {}
    for policy in POLICIES:
        selected = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: float(row["deadline_s"]))
        if not selected:
            raise RuntimeError(f"frontier is missing policy {policy!r}")
        curves[policy] = (
            np.array([float(row["deadline_s"]) for row in selected]),
            np.array([float(row["median_realized_shed_kw"]) for row in selected]),
        )
    return curves


def write(curves: dict, out: Path, removable_kw: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=plot_style.COLUMN_FIGSIZE)
    for policy, (deadlines, shed) in curves.items():
        axis.plot(deadlines, shed, **plot_style.policy_style(policy))
    if removable_kw is not None:
        axis.axhline(removable_kw, color="black", linestyle="--", linewidth=1.2)
        axis.text(axis.get_xlim()[1], removable_kw, " full shed", va="center",
                  fontstyle="italic", fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE)
    axis.set(xscale="log", xlabel="Demand-Response Notice (s)",
             ylabel="Accelerator Power Shed (kW)")
    ticks = next(iter(curves.values()))[0]
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{value:g}" for value in ticks])
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
    axis.xaxis.label.set_size(plot_style.COLUMN_FONT_SIZE)
    axis.yaxis.label.set_size(plot_style.COLUMN_FONT_SIZE)
    axis.grid(alpha=.25)
    axis.legend(loc="upper left", frameon=False,
                fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                       bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--mode", default="normal",
                        choices=("normal", "emergency"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    curves = read(args.campaign / "frontier.csv", args.mode)
    summary = json.loads((args.campaign / "summary.json").read_text())
    if summary["schema"] != "queue-haul-fleet-shed-frontier-v1":
        raise RuntimeError("unexpected frontier schema")
    write(curves, args.out or args.campaign / "fleet-shed-frontier")


if __name__ == "__main__":
    main()
