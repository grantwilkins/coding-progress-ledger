"""Per-token-bucket action mix (replay/state-transfer split), Stage 1 vs
Stage 2, as a distribution over random instances.

For each log-T bucket b, aggregate over its jobs and destinations:
    R_b = sum_{q in b, l} x_R[q, l]   (replay)
    S_b = sum_{q in b, l} x_S[q, l]   (state transfer)
and report replay's share of moved jobs, R_b / (R_b + S_b).

Run N_SEEDS random instances in parallel (prop-fair objective, per-job
classes) at a deadline near the o=1 evacuation frontier, where the
prefill-vs-WAN tradeoff is active. A shifted Stage 1 -> Stage 2 pair means
the peak-pressure stage traded state-ingest for prefill (or vice versa).

Compute is cached to outputs/action_mix_dist.json. Re-run with --recompute.

Usage:
    cd evacuation && uv run python plot_stage2_action_mix.py [--recompute]
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from objective_metrics import BUCKET_LABELS, bucket_idx
from stage1 import solve_stage1
from stage2 import solve_stage2

OUT = Path(__file__).resolve().parent / "outputs"
JSON = OUT / "action_mix_dist.json"
N_SEEDS = 50
N_WORKERS = 8
D_MIX = 300.0  # near the o=1 frontier: both resources active


def per_bucket_split(x_R, x_S, b, B):
    R = np.array([x_R[b == k].sum() for k in range(B)])
    S = np.array([x_S[b == k].sum() for k in range(B)])
    moved = R + S
    return np.where(moved > 0, 100.0 * R / moved, np.nan)  # state share is 100 - R


def _run_seed(seed: int):
    # The prop-fair conic solve is occasionally brittle on a random draw; drop
    # (and report) those seeds rather than letting one kill the distribution.
    inst = build_instance(D=D_MIX, seed=seed)
    try:
        s1 = solve_stage1(inst, "prop_fair")
        s2 = solve_stage2(inst, s1)
    except (cp.error.SolverError, RuntimeError):
        return None
    b, B = bucket_idx(inst), len(BUCKET_LABELS)
    return (per_bucket_split(s1.x_R, s1.x_S, b, B).tolist(),
            per_bucket_split(s2.x_R, s2.x_S, b, B).tolist())


def compute() -> dict:
    with Pool(N_WORKERS) as pool:
        res = [r for r in pool.map(_run_seed, range(N_SEEDS)) if r is not None]
    print(f"aggregated {len(res)}/{N_SEEDS} instances (dropped prop-fair solver failures)")
    R1 = np.array([r[0] for r in res])
    R2 = np.array([r[1] for r in res])
    data = {
        "buckets": list(BUCKET_LABELS),
        "R1_mean": np.nanmean(R1, 0).tolist(), "R1_std": np.nanstd(R1, 0).tolist(),
        "R2_mean": np.nanmean(R2, 0).tolist(), "R2_std": np.nanstd(R2, 0).tolist(),
        "n_seeds": len(res),
    }
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(data))
    for k, name in enumerate(data["buckets"]):
        print(f"{name:>10s}  S1 replay {data['R1_mean'][k]:5.1f}+/-{data['R1_std'][k]:.1f}%   "
              f"S2 replay {data['R2_mean'][k]:5.1f}+/-{data['R2_std'][k]:.1f}%")
    return data


def main() -> None:
    data = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())
    buckets = data["buckets"]
    R1m, R1s = np.array(data["R1_mean"]), np.array(data["R1_std"])
    R2m, R2s = np.array(data["R2_mean"]), np.array(data["R2_std"])
    B = len(buckets)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(B)
    w = 0.38
    ax.bar(x - w/2, R1m, w, color="#3a7ca5", label="Stage 1: replay")
    ax.bar(x - w/2, 100 - R1m, w, bottom=R1m, color="#a5c8e1", label="Stage 1: state transfer")
    ax.bar(x + w/2, R2m, w, color="#c44536", label="Stage 2: replay")
    ax.bar(x + w/2, 100 - R2m, w, bottom=R2m, color="#edaaa0", label="Stage 2: state transfer")
    # Error bars (+/-1 sd over instances) on the replay/state boundary of each bar.
    ax.errorbar(x - w/2, R1m, yerr=R1s, fmt="none", ecolor="0.15", capsize=4, lw=1.4)
    ax.errorbar(x + w/2, R2m, yerr=R2s, fmt="none", ecolor="0.15", capsize=4, lw=1.4)
    # Replay share, centred inside each replay (lower) segment.
    for i in range(B):
        ax.text(x[i] - w/2, R1m[i] / 2, f"{R1m[i]:.0f}%", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")
        ax.text(x[i] + w/2, R2m[i] / 2, f"{R2m[i]:.0f}%", ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=15)
    ax.set_xlabel("Context length bucket (tokens)", fontsize=18)
    ax.set_ylabel("Share of migrated jobs (%)", fontsize=18)
    ax.set_ylim(0, 104)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=14, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, columnspacing=3.0, handlelength=1.4)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "stage2_action_mix.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stage2_action_mix.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'stage2_action_mix.pdf'}  (error bars = +/-1 sd over {data['n_seeds']} instances)")


if __name__ == "__main__":
    main()
