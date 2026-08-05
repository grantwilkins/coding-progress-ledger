"""Reduce and plot synchronized three-region handoff power."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"sweden": "#8C1515", "east": "#006CB8",
          "west": "#008566", "germany": "#008566"}
REGIONS = {"sweden": "sweden-central", "east": "eastus-2",
           "west": "west-europe", "germany": "germany-west-central"}
SPANS = (("Migration", "handoff_start", "handoff_end", "#DAD7CB", .5),
         ("Switch", "pre_end", "handoff_start", "#007C92", .12),
         ("Shutdown", "sleep_start", "sleep_ready", "#6F4E7C", .12))
PLOT_START_S = 30


def read_power(path: Path, base_ns: int) -> list[tuple[float, float]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["valid"] != "1" for row in rows):
        raise ValueError(f"invalid power samples: {path}")
    return [((int(row["wall_ns"]) - base_ns) / 1e9, float(row["power_w"]))
            for row in rows]


def bin_mean(points: list[tuple[float, float]],
             start: float = PLOT_START_S) -> tuple[list[float], list[float]]:
    bins = {}
    for seconds, value in points:
        if seconds >= start:
            bins.setdefault(int(seconds), []).append(value)
    return ([index + .5 for index in sorted(bins)],
            [statistics.fmean(bins[index]) for index in sorted(bins)])


def style(axis, xlim: tuple[float, float], ylabel: str) -> None:
    axis.set_xlim(*xlim)
    axis.set_ylabel(ylabel, size=16)
    axis.tick_params(labelsize=14)
    axis.grid(alpha=.25)
    for spine in axis.spines.values():
        spine.set_color("black")


def reduce(run_root: Path) -> list[dict]:
    result = json.loads((run_root / "result.json").read_text())
    phases, base = result["phases"], result["phases"]["pre_start"]["wall_ns"]
    names = {"pre": ("pre_start", "pre_end"),
             "post": ("post_start", "post_end")}
    if "traffic_switched" in phases:
        names["post"] = ("sleep_ready", "post_end")
        names |= {"migration": ("handoff_start", "handoff_end"),
                  "source_fall": ("traffic_switched", "sleep_ready")}
    windows = {name: ((phases[start]["wall_ns"] - base) / 1e9,
                      (phases[end]["wall_ns"] - base) / 1e9)
               for name, (start, end) in names.items()}
    nodes = ("sweden", *sorted(result["scenario"]["background"]))
    paths = {node: run_root / "nodes" / node / "power.csv" for node in nodes}
    paths["sweden"] = run_root / "power.csv"
    rows = []
    for node, path in paths.items():
        points = read_power(path, base)
        for phase, (start, end) in windows.items():
            values = [watts for seconds, watts in points if start <= seconds <= end]
            if not values:
                raise ValueError(f"{node} power does not cover {phase}")
            rows.append({"node": node, "phase": phase, "samples": len(values),
                         "mean_power_w": statistics.fmean(values),
                         "median_power_w": statistics.median(values)})
    with (run_root / "power_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if result.get("schema") == "queue-haul-three-node-handoff-v2":
        means = {(row["node"], row["phase"]): row["mean_power_w"] for row in rows}
        if means["sweden", "pre"] < 200 or \
                means["sweden", "pre"] - means["sweden", "post"] < 50:
            raise ValueError("source power did not show a high-to-low handoff")
        queue = []
        for node in paths:
            with (run_root / f"metrics_{node}.csv").open() as handle:
                samples = list(csv.DictReader(handle))
            for phase, (start, end) in windows.items():
                selected = [row for row in samples
                            if start <= (int(row["wall_ns"]) - base) / 1e9 <= end]
                if not selected:
                    raise ValueError(f"{node} queue metrics do not cover {phase}")
                queue.append({
                    "node": node, "phase": phase, "samples": len(selected),
                    "mean_running": statistics.fmean(float(row["vllm:num_requests_running"]) for row in selected),
                    "max_waiting": max(float(row["vllm:num_requests_waiting"]) for row in selected),
                })
        with (run_root / "queue_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=queue[0])
            writer.writeheader()
            writer.writerows(queue)

    marker = {name: (phase["wall_ns"] - base) / 1e9
              for name, phase in phases.items()}
    xlim = (PLOT_START_S, marker["post_end"])
    plt.style.use("default")
    figure, axis = plt.subplots(figsize=(9, 4))
    for node, path in paths.items():
        axis.plot(*bin_mean(read_power(path, base)), lw=1.5,
                  color=COLORS[node], label=REGIONS[node])
    for label, start, end, color, alpha in SPANS:
        if start in marker and end in marker:
            axis.axvspan(marker[start], marker[end], color=color, alpha=alpha,
                         label=label)
    for name in ("pre_end", "handoff_start", "handoff_end", "post_start"):
        if name in marker:
            axis.axvline(marker[name], color="black", lw=.7, ls=":")
    style(axis, xlim, "Power per GPU (W)")
    axis.set_xlabel("Time (s)", size=16)
    axis.legend(frameon=False, fontsize=13, loc="upper center",
                bbox_to_anchor=(.5, -.2), ncol=3)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(run_root / f"power_handoff.{suffix}", dpi=220,
                       bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(reduce(parser.parse_args().run_root), indent=2))


if __name__ == "__main__":
    main()
