"""Plot the raw per-GPU power trace from a measured power sample log."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"0": "#8C1515", "1": "#008566", "text": "#2E2D29", "grid": "#DAD7CB"}


def read_power(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    rows = list(csv.DictReader(path.open()))
    if any(row["valid"] != "1" for row in rows):
        raise ValueError("power trace contains invalid samples")
    base = min(int(row["monotonic_ns"]) for row in rows)
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        samples[row["gpu"]].append(
            ((int(row["monotonic_ns"]) - base) / 1e9, float(row["power_w"]))
        )
    return {
        gpu: tuple(zip(*sorted(points)))
        for gpu, points in samples.items()
    }


def plot(series: dict[str, tuple[list[float], list[float]]], out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 12,
    })
    fig, axes = plt.subplots(figsize=(10, 4))
    for gpu, (times, watts) in sorted(series.items()):
        axes.plot(times, watts, color=COLORS[gpu], linewidth=2, label=f"GPU {gpu}")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Power per GPU (W)")
    axes.set_xlim(0, max(max(times) for times, _ in series.values()))
    axes.legend(frameon=False, ncol=len(series), loc="upper right")
    axes.grid(color=COLORS["grid"], linewidth=.8, alpha=.7)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(COLORS["text"])
    axes.tick_params(colors=COLORS["text"])
    axes.set_facecolor("#FFFFFF")
    fig.set_facecolor("#FFFFFF")
    fig.tight_layout(pad=.25)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", type=Path, default=Path("data/power.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/power-trace/power_per_gpu"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(read_power(args.power), args.out)


if __name__ == "__main__":
    main()
