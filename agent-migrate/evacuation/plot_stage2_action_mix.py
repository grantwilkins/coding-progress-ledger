"""Per-model action mix (replay/state-transfer split), Stage 1 vs Stage 2, as a
distribution over random instances.

For each model m, aggregate over its classes and destinations:
    R_m = sum_{q in m, l} x_R[q, l]   (replay)
    S_m = sum_{q in m, l} x_S[q, l]   (state transfer)
and report replay's share of moved jobs, R_m / (R_m + S_m).

Run N_SEEDS random instances in parallel (prop-fair objective on the
(model, token-bucket) grid, matching the poster) and draw the mean replay share
per model with error bars = +/-1 sd across instances. A shifted Stage 1 -> Stage 2
pair means the peak-pressure stage traded state-ingest for prefill (or vice versa).

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
from stage1 import solve_stage1
from stage2 import solve_stage2

OUT = Path(__file__).resolve().parent / "outputs"
JSON = OUT / "action_mix_dist.json"
N_SEEDS = 50
N_WORKERS = 8


def per_model_split(x_R, x_S, model_idx, M):
    R = np.array([x_R[model_idx == m].sum() for m in range(M)])
    S = np.array([x_S[model_idx == m].sum() for m in range(M)])
    moved = R + S
    R_pct = 100.0 * np.where(moved > 0, R / moved, 0.0)
    return R_pct  # state share is 100 - R_pct


def _run_seed(seed: int):
    # The prop-fair conic solve is occasionally brittle on a random draw; drop
    # (and report) those seeds rather than letting one kill the distribution.
    inst = build_instance(total_jobs=10_000, n_bins=5, seed=seed)
    try:
        s1 = solve_stage1(inst, "prop_fair")
        s2 = solve_stage2(inst, s1)
    except (cp.error.SolverError, RuntimeError):
        return None
    M = len(inst.M_names)
    return (per_model_split(s1.x_R, s1.x_S, inst.model_idx, M).tolist(),
            per_model_split(s2.x_R, s2.x_S, inst.model_idx, M).tolist())


def compute() -> dict:
    with Pool(N_WORKERS) as pool:
        res = [r for r in pool.map(_run_seed, range(N_SEEDS)) if r is not None]
    print(f"aggregated {len(res)}/{N_SEEDS} instances (dropped prop-fair solver failures)")
    R1 = np.array([r[0] for r in res])
    R2 = np.array([r[1] for r in res])
    data = {
        "models": list(build_instance(total_jobs=10_000, n_bins=5).M_names),
        "R1_mean": R1.mean(0).tolist(), "R1_std": R1.std(0).tolist(),
        "R2_mean": R2.mean(0).tolist(), "R2_std": R2.std(0).tolist(),
        "n_seeds": len(res),
    }
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(data))
    for m, name in enumerate(data["models"]):
        print(f"{name:>20s}  S1 replay {data['R1_mean'][m]:5.1f}+/-{data['R1_std'][m]:.1f}%   "
              f"S2 replay {data['R2_mean'][m]:5.1f}+/-{data['R2_std'][m]:.1f}%")
    return data


def main() -> None:
    data = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())
    models = data["models"]
    R1m, R1s = np.array(data["R1_mean"]), np.array(data["R1_std"])
    R2m, R2s = np.array(data["R2_mean"]), np.array(data["R2_std"])
    M = len(models)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(M)
    w = 0.38
    ax.bar(x - w/2, R1m, w, color="#3a7ca5", label="Stage 1 — replay")
    ax.bar(x - w/2, 100 - R1m, w, bottom=R1m, color="#a5c8e1", label="Stage 1 — state")
    ax.bar(x + w/2, R2m, w, color="#c44536", label="Stage 2 — replay")
    ax.bar(x + w/2, 100 - R2m, w, bottom=R2m, color="#edaaa0", label="Stage 2 — state")
    # Error bars (+/-1 sd over instances) on the replay/state boundary of each bar.
    ax.errorbar(x - w/2, R1m, yerr=R1s, fmt="none", ecolor="black", capsize=4, lw=1.4)
    ax.errorbar(x + w/2, R2m, yerr=R2s, fmt="none", ecolor="black", capsize=4, lw=1.4)
    for i in range(M):
        ax.text(x[i] - w/2, R1m[i] + R1s[i] + 1, f"{R1m[i]:.0f}%",
                ha="center", va="bottom", fontsize=12, color="#3a7ca5", fontweight="bold")
        ax.text(x[i] + w/2, R2m[i] + R2s[i] + 1, f"{R2m[i]:.0f}%",
                ha="center", va="bottom", fontsize=12, color="#c44536", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=14)
    ax.set_ylabel("share of moved jobs (%)", fontsize=17)
    ax.set_ylim(0, 112)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=14, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUT / "stage2_action_mix.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stage2_action_mix.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'stage2_action_mix.pdf'}  (error bars = +/-1 sd over {data['n_seeds']} instances)")


if __name__ == "__main__":
    main()
