"""Stage 2 decomposition convergence vs the coupling dimension (number of
pressure constraints), across destination counts.

For n_dest in {3, 6, 10}, over many seeds, run subgradient / mirror descent /
ADMM against the CVXPY ground truth phi* and aggregate the RELATIVE primal gap
(phi_k - phi*)/phi*. Relative (not absolute) gap is used because the synthetic
destinations add capacity, so phi* shrinks as n_dest grows; normalizing makes
the convergence rate comparable across the three coupling dimensions.

Compute is cached to outputs/convergence_scaling.json (ADMM is the bottleneck:
Q tiny QPs per iteration). Re-run with --recompute to regenerate.

Usage:
    cd evacuation && uv run python plot_decomp_convergence_scaling.py [--recompute]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2_dual import admm, build_dual_structure, mirror_descent, subgradient

OUT = Path(__file__).resolve().parent / "outputs"
JSON = OUT / "convergence_scaling.json"

N_DEST = [3, 6, 10]
SEEDS = range(30)
SUBGRAD_ITERS = 800
ADMM_ITERS = 100
ADMM_RHO = 150.0
COLORS = {"subgradient": "#3a7ca5", "mirror descent": "#e8943a", "ADMM": "#4a9b54"}


def _pad(gaps: list[np.ndarray]) -> np.ndarray:
    """Stack early-terminating runs, padding each with its last value."""
    n = max(len(g) for g in gaps)
    return np.vstack([np.concatenate([g, np.full(n - len(g), g[-1])]) for g in gaps])


def compute() -> dict:
    data: dict = {"n_I": {}, "iters": {}}
    for nd in N_DEST:
        per_solver = {"subgradient": [], "mirror descent": [], "ADMM": []}
        n_I = build_dual_structure(build_instance(total_jobs=200, seed=0, n_dest=nd))[0].shape[0]
        for seed in SEEDS:
            inst = build_instance(D=300.0, total_jobs=200, seed=seed, n_dest=nd)
            s1 = solve_stage1(inst)
            if s1.Z_star > 1e-6:        # ADMM precondition; synthetic dests should keep Z*=0
                continue
            phi_star = solve_stage2(inst, s1).phi_star
            for name, traj in (
                ("subgradient", subgradient(inst, s1, phi_star=phi_star, max_iter=SUBGRAD_ITERS)),
                ("mirror descent", mirror_descent(inst, s1, phi_star=phi_star, max_iter=SUBGRAD_ITERS)),
                ("ADMM", admm(inst, s1, max_iter=ADMM_ITERS, rho=ADMM_RHO)),
            ):
                per_solver[name].append(np.maximum((traj.primal - phi_star) / phi_star, 1e-9))
        agg = {}
        for name, gaps in per_solver.items():
            g = _pad(gaps)
            agg[name] = {
                "median": np.median(g, axis=0).tolist(),
                "lo": np.percentile(g, 25, axis=0).tolist(),
                "hi": np.percentile(g, 75, axis=0).tolist(),
            }
        data[str(nd)] = agg
        data["n_I"][str(nd)] = int(n_I)
        print(f"n_dest={nd}: |I|={n_I}  seeds={len(per_solver['ADMM'])}")
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(data))
    return data


def main() -> None:
    if "--recompute" in sys.argv or not JSON.exists():
        data = compute()
    else:
        data = json.loads(JSON.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, nd in zip(axes, N_DEST):
        agg = data[str(nd)]
        for name, c in COLORS.items():
            med = np.array(agg[name]["median"])
            lo, hi = np.array(agg[name]["lo"]), np.array(agg[name]["hi"])
            it = np.arange(1, len(med) + 1)
            ax.semilogy(it, med, color=c, lw=1.7, label=name)
            ax.fill_between(it, lo, hi, color=c, alpha=0.18, lw=0)
        ax.set_xlabel(f"iteration   ($L={nd}$,  $|\\mathcal{{I}}|={data['n_I'][str(nd)]}$)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"relative primal gap  $(\phi_k - \phi^\star)/\phi^\star$")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "decomp_convergence_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "decomp_convergence_scaling.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'decomp_convergence_scaling.pdf'}")


if __name__ == "__main__":
    main()
