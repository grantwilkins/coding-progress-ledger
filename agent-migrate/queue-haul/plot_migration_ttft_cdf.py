"""Plot measured semi-live migration-to-first-token ECDFs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).parent
DEFAULT_RUN = ROOT / "data/qh-destination-v7-run-20260722/loaded"
DEFAULT_OUT = ROOT / "outputs/mechanism-validation"
METHODS = {"kv_transfer": "KV transfer", "replay": "Context replay"}


def extract(root: Path) -> list[dict]:
    rows = []
    for run in sorted(root.glob("rho*-t*-b*-r*")):
        match = re.fullmatch(r"rho([0-9.]+)-t(\d+)-b(\d+)-r(\d+)", run.name)
        if not match:
            continue
        rho, context, bandwidth, repeat = match.groups()
        for method in METHODS:
            path = run / method
            result = json.loads((path / "result.json").read_text())
            scenario = json.loads((path / "scenario.json").read_text())
            moves = result["migrations"]
            if scenario["concurrency"] != 1 or len(moves) != 1 \
                    or moves[0]["move"]["method"] != method \
                    or moves[0].get("error"):
                raise ValueError("CDF requires one successful concurrency-one migration")
            move, request = moves[0], moves[0]["initial"]
            start, first, commit = (
                move["initial_start_ns"], request["first_byte_ns"],
                move["switch_end_ns"],
            )
            if not start <= request["start_ns"] <= first <= request["end_ns"] \
                    or commit < start:
                raise ValueError("invalid migration timing order")
            foreground = json.loads(
                (path / "foreground/requests.json").read_text()
            )
            active = any(
                item["start_ns"] <= start < item["end_ns"]
                for item in foreground
            )
            arriving = any(
                start < item["start_ns"] < commit for item in foreground
            )
            rows.append({
                "method": method,
                "target_rho": float(rho),
                "context_tokens": int(context),
                "bandwidth_mbps": int(bandwidth),
                "repeat": int(repeat),
                "migration_ttft_s": (first - start) / 1e9,
                "transfer_or_replay_s":
                    (request["start_ns"] - start) / 1e9,
                "destination_request_ttft_s":
                    (first - request["start_ns"]) / 1e9,
                "foreground_overlap": "active_at_start" if active else
                    "arrived_during" if arriving else "none",
                "provenance": str(
                    (path / "result.json").relative_to(ROOT)
                    if path.is_relative_to(ROOT) else path / "result.json"
                ),
            })
    if not rows:
        raise ValueError(f"no semi-live migrations in {root}")
    return rows


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), float))
    return x, np.arange(1, len(x) + 1) / len(x)


def write(root: Path, out: Path) -> None:
    rows = extract(root)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "migration_ttft_cdf.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    cells = sorted({
        (row["context_tokens"], row["bandwidth_mbps"]) for row in rows
    })
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})
    colors = {"kv_transfer": "#0072B2", "replay": "#D55E00"}
    fig, axes = plt.subplots(
        1, len(cells), figsize=(4.1 * len(cells), 3.2),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax, (context, bandwidth) in zip(axes[0], cells):
        selected = [
            row for row in rows
            if (row["context_tokens"], row["bandwidth_mbps"])
            == (context, bandwidth)
        ]
        for method in METHODS:
            method_rows = [
                row for row in selected if row["method"] == method
            ]
            x, y = ecdf(row["migration_ttft_s"] for row in method_rows)
            ax.step(
                x, y, where="post", color=colors[method], linewidth=2,
                label=f"{METHODS[method]} (n={len(x)})",
            )
            for point, fraction, row in zip(
                x, y, sorted(
                    method_rows, key=lambda item: item["migration_ttft_s"]
                ),
            ):
                overlap = row["foreground_overlap"] != "none"
                ax.scatter(point, fraction, s=30, marker="o" if overlap else "x",
                           color=colors[method], zorder=3)
        ax.set_title(
            f"{context // 1024}K context · {bandwidth / 1000:g} Gbps"
            f"\nn={len(selected) // len(METHODS)} per method"
        )
        ax.set_xlabel("Migration start to first destination token (s)")
        ax.set_ylim(0, 1.04)
    axes[0][0].set_ylabel("Fraction of migrations")
    method_legend = [
        Line2D([0], [0], color=colors[key], linewidth=2, label=value)
        for key, value in METHODS.items()
    ]
    overlap_legend = [
        Line2D([0], [0], color="#444444", marker="o", linestyle="",
               label="Foreground overlap"),
        Line2D([0], [0], color="#444444", marker="x", linestyle="",
               label="No overlap"),
    ]
    fig.legend(
        handles=method_legend + overlap_legend, loc="upper center",
        ncol=4, frameon=False, bbox_to_anchor=(.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, .9))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"migration_ttft_cdf.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write(args.run_root, args.out)


if __name__ == "__main__":
    main()
