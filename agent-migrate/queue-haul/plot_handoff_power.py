"""Reduce and plot synchronized three-node handoff power."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"sweden": "#8C1515", "east": "#006CB8", "west": "#008566"}


def read_power(path: Path, base_ns: int) -> list[tuple[float, float]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["valid"] != "1" for row in rows):
        raise ValueError(f"invalid power samples: {path}")
    return [((int(row["wall_ns"]) - base_ns) / 1e9, float(row["power_w"]))
            for row in rows]


def reduce(run_root: Path) -> list[dict]:
    result = json.loads((run_root / "result.json").read_text())
    phases, base = result["phases"], result["phases"]["pre_start"]["wall_ns"]
    windows = {name: ((phases[f"{name}_start"]["wall_ns"] - base) / 1e9,
                      (phases[f"{name}_end"]["wall_ns"] - base) / 1e9)
               for name in ("pre", "post")}
    paths = {"sweden": run_root / "power.csv",
             "east": run_root / "nodes/east/power.csv",
             "west": run_root / "nodes/west/power.csv"}
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
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for node, path in paths.items():
        points = read_power(path, base)
        axis.plot(*zip(*points), color=COLORS[node], linewidth=1,
                  alpha=.8, label=node.title())
    handoff = (phases["handoff_start"]["wall_ns"] - base) / 1e9
    sleep = (phases["sleep_ready"]["wall_ns"] - base) / 1e9
    axis.axvline(handoff, color="#2E2D29", linestyle="--", label="Handoff")
    axis.axvline(sleep, color="#E98300", linestyle=":", label="Sweden sleep")
    axis.set(xlabel="Seconds from inference window start", ylabel="GPU power (W)")
    axis.legend(frameon=False, ncol=5)
    figure.tight_layout()
    figure.savefig(run_root / "power_handoff.png", dpi=200)
    figure.savefig(run_root / "power_handoff.pdf")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(reduce(parser.parse_args().run_root), indent=2))


if __name__ == "__main__":
    main()
