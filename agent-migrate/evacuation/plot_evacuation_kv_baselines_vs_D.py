"""Sweep deadline D and plot KV-weighted % evacuated: the proportional-fairness
optimizer vs heuristic baselines (8-rack pod at fitted occupancy, per-job classes).

KV-weighted because count-greedy/cost-greedy heuristics hoard small-KV jobs and
abandon large ones; weighting by eta_q*T_q surfaces that. prop-fair refuses to
starve any class, so it keeps large-KV jobs in play.

Usage:
    cd evacuation && uv run python plot_evacuation_kv_baselines_vs_D.py [--recompute]

Writes outputs/kv_evacuated_baselines_vs_D.{pdf,png} and outputs/kv_baselines_D.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines import allocate
from instance import build_instance
from objective_metrics import evac_summary
from stage1 import solve_stage1

OUT = Path(__file__).resolve().parent / "outputs"
CSV = OUT / "kv_baselines_D.csv"
D_SWEEP_S = (15, 30, 60, 90, 120, 180, 240, 360, 600, 1200)
RANDOM_SEEDS = 30
DET = ("greedy", "replay_only", "state_only")

# (label, color, marker); lines drawn ours -> deterministic baselines -> random.
STYLE = {
    "ours":         ("Ours (prop-fair)",  "#3a7ca5", "o"),
    "greedy":       ("greedy (cheapest)", "#c44536", "s"),
    "replay_only":  ("replay-only",       "#4a9b54", "v"),
    "state_only":   ("state-only",        "#6a4c93", "D"),
    "random":       ("random (mean±1sd)", "0.45", "x"),
}


def _kv(inst, z):
    return 100.0 * evac_summary(inst, z)["kv_weighted_evacuation"]


def compute():
    rows = []
    for D in D_SWEEP_S:
        inst = build_instance(D=float(D))
        ours = _kv(inst, solve_stage1(inst, "prop_fair").z)
        rows.append(("ours", -1, D, ours))
        for name in DET:
            rows.append((name, 0, D, _kv(inst, allocate(inst, name).z)))
        for seed in range(RANDOM_SEEDS):
            rows.append(("random", seed, D, _kv(inst, allocate(inst, "random", seed=seed).z)))
        print(f"D={D:4d}s  ours(prop-fair) kv={ours:5.1f}%")
    OUT.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "D_s", "kv_evac_pct"])
        w.writerows(rows)
    return rows


def load():
    with CSV.open() as fh:
        return [(r["method"], int(r["seed"]), float(r["D_s"]), float(r["kv_evac_pct"]))
                for r in csv.DictReader(fh)]


def plot(rows):
    data: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for method, _, D, pct in rows:
        data[method][D].append(pct)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method in ("ours", *DET, "random"):
        label, color, marker = STYLE[method]
        y = np.array([np.mean(data[method][D]) for D in D_SWEEP_S])
        if method == "random":
            sd = np.array([np.std(data[method][D]) for D in D_SWEEP_S])
            ax.fill_between(D_SWEEP_S, y - sd, y + sd, color=color, alpha=0.2)
        ax.plot(D_SWEEP_S, y, marker=marker, color=color, ms=6,
                lw=2.6 if method == "ours" else 1.6,
                label=label, zorder=5 if method == "ours" else 2)

    ax.set_xscale("log")
    ax.set_xlabel("Deadline $D$ (s)", fontsize=17)
    ax.set_ylabel("KV Cache Bytes Migrated Successfully (%)", fontsize=13)
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=14, loc="upper left")
    fig.savefig(OUT / "kv_evacuated_baselines_vs_D.pdf", bbox_inches="tight")
    fig.savefig(OUT / "kv_evacuated_baselines_vs_D.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'kv_evacuated_baselines_vs_D.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    rows = compute() if ap.parse_args().recompute or not CSV.exists() else load()
    plot(rows)


if __name__ == "__main__":
    main()
