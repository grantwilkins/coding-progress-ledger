"""Plot the fleet-scale deadline-to-power-shed frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, PercentFormatter
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
            np.array([float(row["attainment_rate"]) for row in selected]),
        )
    return curves


def write(curves: dict, out: Path) -> None:
    figure, axis = plt.subplots(figsize=plot_style.COLUMN_FIGSIZE)
    for policy, (deadlines, shed) in curves.items():
        axis.plot(deadlines, shed, **plot_style.policy_style(policy))
    axis.set(xscale="log", ylim=(-.02, 1.02),
             xlabel="Demand-Response Notice (s)",
             ylabel="Requests Attained by Deadline")
    ticks = next(iter(curves.values()))[0]
    axis.set_xticks(ticks)
    # Label a readable subset: adjacent log-spaced deadlines collide otherwise.
    keep, last = [], 0.0
    for value in ticks:
        keep.append(f"{value:g}" if value / max(last, 1e-9) >= 1.7 else "")
        if keep[-1]:
            last = value
    axis.set_xticklabels(keep)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
    axis.xaxis.label.set_size(plot_style.COLUMN_FONT_SIZE)
    axis.yaxis.label.set_size(plot_style.COLUMN_FONT_SIZE)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
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
