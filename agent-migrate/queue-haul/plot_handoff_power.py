"""Reduce and plot synchronized three-node handoff power and KV movement."""

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
SPANS = (("KV transfer", "handoff_start", "handoff_end", "#DAD7CB", .5),
         ("Sweden sleep", "sleep_start", "sleep_ready", "#007C92", .12),
         ("Destination steady", "post_start", "post_end", "#6F4E7C", .12))
PLOT_START_S = 30


def read_power(path: Path, base_ns: int) -> list[tuple[float, float]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["valid"] != "1" for row in rows):
        raise ValueError(f"invalid power samples: {path}")
    return [((int(row["wall_ns"]) - base_ns) / 1e9, float(row["power_w"]))
            for row in rows]


def read_fetches(run_root: Path, base_ns: int) -> dict[str, list[tuple[float, float]]]:
    """Cumulative-ready first fetch of each KV chunk, by destination region."""
    route = {row["connection_id"]: row["route"] for row
             in csv.DictReader((run_root / "proxy_connections.csv").open())}
    rows = sorted((row for row
                   in csv.DictReader((run_root / "resp_transfers.csv").open())
                   if row["command"] == "GET"), key=lambda row: int(row["end_ns"]))
    series, seen = {}, set()
    for row in rows:
        if row["key_hashes"] in seen:
            continue
        seen.add(row["key_hashes"])
        pool, _, node = route[row["connection_id"]].partition("/")
        if pool != "kv":
            continue
        series.setdefault(node, []).append(
            ((int(row["end_ns"]) - base_ns) / 1e9, int(row["payload_bytes"]) / 1e6))
    if not series:
        raise ValueError(f"no kv fetches in {run_root}")
    return series


def bin_mean(points: list[tuple[float, float]],
             start: float = PLOT_START_S) -> tuple[list[float], list[float]]:
    bins = {}
    for seconds, value in points:
        if seconds >= start:
            bins.setdefault(int(seconds), []).append(value)
    return ([index + .5 for index in sorted(bins)],
            [statistics.fmean(bins[index]) for index in sorted(bins)])


def cumulative(points: list[tuple[float, float]], stop: float,
               start: float = PLOT_START_S) -> tuple[list[float], list[float]]:
    bins = {index: 0. for index in range(int(start), int(stop) + 1)}
    for seconds, value in points:
        if start <= seconds <= stop:
            bins[int(seconds)] += value
    total, series = 0., []
    for index in sorted(bins):
        total += bins[index]
        series.append(total / 1000)
    return [index + .5 for index in sorted(bins)], series


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
    kv = read_fetches(run_root, phases["pre_start"]["monotonic_ns"])
    plt.style.use("default")
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, height_ratios=(2.4, 1),
        gridspec_kw={"hspace": .12})
    for node, path in paths.items():
        top.plot(*bin_mean(read_power(path, base)), lw=1.5,
                 color=COLORS[node], label=node.title())
    for node in paths.keys() - {"sweden"}:
        bottom.plot(*cumulative(kv[node], xlim[1]), lw=1.5,
                    color=COLORS[node], label=f"{node.title()} fetches KV")
    for axis in (top, bottom):
        for label, start, end, color, alpha in SPANS:
            if start in marker and end in marker:
                axis.axvspan(marker[start], marker[end], color=color, alpha=alpha,
                             label=label if axis is top else None)
        for name in ("pre_end", "handoff_start", "handoff_end", "post_start"):
            if name in marker:
                axis.axvline(marker[name], color="black", lw=.7, ls=":")
    style(top, xlim, "Power per GPU (W)")
    style(bottom, xlim, "KV fetched (GB)")
    bottom.set_xlabel("Time (s)", size=16)
    top.legend(frameon=False, fontsize=13, loc="upper center",
               bbox_to_anchor=(.5, 1.28), ncol=3)
    bottom.legend(frameon=False, fontsize=12, loc="upper left", ncol=3)
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
