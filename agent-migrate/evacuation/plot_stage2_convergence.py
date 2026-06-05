"""Convergence of three first-order Stage 2 solvers vs CVXPY truth, as a
distribution over random instances.

Run N_SEEDS instances in parallel; for each, solve subgradient / mirror descent
/ ADMM against the CVXPY ground-truth phi* and record the RELATIVE primal gap
(phi_k - phi*)/phi*. Relative (not absolute) gap is the right aggregation because
phi* differs per random instance. We plot the median trajectory and a 25-75
percentile band per solver.

Compute is cached to outputs/stage2_convergence_dist.json (ADMM is the
bottleneck: Q tiny QPs per iteration). Re-run with --recompute.

Usage:
    cd evacuation && uv run python plot_stage2_convergence.py [--recompute]
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2_dual import admm, mirror_descent, subgradient

OUT = Path(__file__).resolve().parent / "outputs"
JSON = OUT / "stage2_convergence_dist.json"

N_SEEDS = 50
N_WORKERS = 10
SUBGRAD_ITERS = 800
ADMM_ITERS = 800  # full budget so every method spans the same iteration axis
ADMM_RHO = 150.0
COLORS = {"subgradient": "#3a7ca5", "mirror descent": "#e8943a", "ADMM": "#4a9b54"}


def _run_seed(seed: int):
    """One random instance -> relative primal-gap trajectory per solver.

    Returns None if Z* > 0 (ADMM's precondition fails); such seeds are dropped
    from the aggregate, mirroring plot_decomp_convergence_scaling.
    """
    # total_jobs=200 (n_q ~ O(1)) is the regime ADMM's rho is tuned for; larger
    # n rescales the per-class loads and detunes ADMM. Relative gap below makes
    # the comparison scale-free across the random instances.
    inst = build_instance(D=300.0, total_jobs=200, seed=seed)
    s1 = solve_stage1(inst)
    if s1.Z_star > 1e-6:
        return None
    phi_star = solve_stage2(inst, s1).phi_star
    out = {}
    for name, traj in (
        ("subgradient", subgradient(inst, s1, phi_star=phi_star, max_iter=SUBGRAD_ITERS)),
        ("mirror descent", mirror_descent(inst, s1, phi_star=phi_star, max_iter=SUBGRAD_ITERS)),
        ("ADMM", admm(inst, s1, max_iter=ADMM_ITERS, rho=ADMM_RHO)),
    ):
        out[name] = np.maximum((traj.primal - phi_star) / phi_star, 1e-9).tolist()
    return out


def _pad(gaps: list[np.ndarray]) -> np.ndarray:
    """Stack early-terminating runs, padding each with its last (converged) value."""
    n = max(len(g) for g in gaps)
    return np.vstack([np.concatenate([g, np.full(n - len(g), g[-1])]) for g in gaps])


def compute() -> dict:
    with Pool(N_WORKERS) as pool:
        runs = [r for r in pool.map(_run_seed, range(N_SEEDS)) if r is not None]
    print(f"aggregated {len(runs)}/{N_SEEDS} instances (dropped Z*>0 seeds)")
    agg = {"n_seeds": len(runs)}
    for name in COLORS:
        g = _pad([np.array(r[name]) for r in runs])
        agg[name] = {
            "median": np.median(g, axis=0).tolist(),
            "lo": np.percentile(g, 25, axis=0).tolist(),
            "hi": np.percentile(g, 75, axis=0).tolist(),
        }
        print(f"{name:>16s}: median final rel-gap {agg[name]['median'][-1]:.2e}")
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(agg))
    return agg


def main() -> None:
    agg = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, c in COLORS.items():
        med = np.array(agg[name]["median"])
        lo, hi = np.array(agg[name]["lo"]), np.array(agg[name]["hi"])
        it = np.arange(1, len(med) + 1)
        ax.semilogy(it, med, color=c, lw=2.8, label=name)
        ax.fill_between(it, lo, hi, color=c, alpha=0.2, lw=0)
    ax.set_xlabel("iteration", fontsize=20)
    ax.set_ylabel(r"relative primal gap  $(\phi_k - \phi^\star)/\phi^\star$", fontsize=20)
    ax.tick_params(labelsize=17)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=16, loc="upper right")
    fig.savefig(OUT / "stage2_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stage2_convergence.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'stage2_convergence.pdf'}  (band = 25-75 pct over {agg['n_seeds']} instances)")


if __name__ == "__main__":
    main()
