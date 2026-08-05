"""Plot measured migration-to-first-token ECDFs by movement method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {"kv_transfer": "KV Migrate Only", "replay": "Replay Context Only"}
COLORS = {"kv_transfer": "#006CB8", "replay": "#E98300"}
LINESTYLES = {"kv_transfer": "-.", "replay": ":"}
FIGSIZE = (5, 4)


def attempts(run_root: Path) -> list[Path]:
    """Last completed attempt per scenario; retried attempts supersede failures."""
    paths = {}
    for path in sorted(run_root.glob("scenarios/*/attempt-*/result.json")):
        if json.loads(path.read_text())["status"] == "complete":
            paths[path.parent.parent] = path
    missing = sorted({path for path in run_root.glob("scenarios/*")}
                     - set(paths))
    if missing:
        raise ValueError(f"scenarios without a complete attempt: {missing}")
    return [paths[key] for key in sorted(paths)]


def extract(run_root: Path) -> list[dict]:
    rows = []
    for path in attempts(run_root):
        result = json.loads(path.read_text())
        scenario = json.loads((path.parent / "scenario.json").read_text())
        if len(result["requests"]) != 1:
            raise ValueError(f"CDF requires one migration per scenario: {path}")
        move = result["requests"][0]
        request, start = move["request"], result["started_ns"]
        if not start <= request["start_ns"] <= request["first_byte_ns"]:
            raise ValueError(f"invalid migration timing order: {path}")
        rows.append({
            "scenario_id": result["scenario_id"],
            "method": move["method"],
            "bandwidth": scenario["bandwidth"],
            "bandwidth_mbps": scenario["bandwidth_mbps"],
            "workload": scenario["workload"],
            "context_size": scenario["context_size"],
            "migration_s": result["migration_s"],
            "migration_ttft_s": (request["first_byte_ns"] - start) / 1e9,
            "destination_request_ttft_s":
                (request["first_byte_ns"] - request["start_ns"]) / 1e9,
        })
    if not rows:
        raise ValueError(f"no migrations in {run_root}")
    return rows


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), float))
    return x, np.arange(1, len(x) + 1) / len(x)


def write(run_root: Path) -> list[dict]:
    rows = extract(run_root)
    with (run_root / "migration_ttft_cdf.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.style.use("default")
    figure, axis = plt.subplots(figsize=FIGSIZE)
    for method, label in METHODS.items():
        x, y = ecdf(row["migration_ttft_s"] for row in rows
                    if row["method"] == method)
        if not len(x):
            continue
        axis.step(np.r_[0, x], np.r_[0, y], where="post", linewidth=3,
                  color=COLORS[method], linestyle=LINESTYLES[method],
                  label=label)
    axis.set(xlabel="Migration + Destination TTFT (s)",
             ylabel="Cumulative Distribution", ylim=(0, 1.02))
    axis.tick_params(labelsize=15)
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.grid(alpha=.25)
    axis.legend(frameon=False, fontsize=13)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(run_root / f"migration_ttft_cdf.{suffix}", dpi=220)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    rows = write(parser.parse_args().run_root)
    print(json.dumps({method: len([row for row in rows
                                   if row["method"] == method])
                      for method in METHODS}, indent=2))


if __name__ == "__main__":
    main()
