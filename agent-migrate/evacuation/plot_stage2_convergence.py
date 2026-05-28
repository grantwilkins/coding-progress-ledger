"""Convergence comparison of three first-order Stage 2 solvers vs CVXPY truth.

Usage:
    cd evacuation && uv run python plot_stage2_convergence.py

Writes outputs/stage2_convergence.{pdf,png}.

Smaller instance (total_jobs=200) keeps per-class ADMM updates tractable: each
iteration solves Q tiny K=6 simplex-QPs via CVXPY (Parameters).
"""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from stage1 import solve_stage1
from stage2 import solve_stage2
from stage2_dual import subgradient, mirror_descent, admm, build_dual_structure


def main() -> None:
    inst = build_instance(D=300.0, total_jobs=200, seed=0)
    s1 = solve_stage1(inst)
    truth = solve_stage2(inst, s1)
    phi_star = truth.phi_star
    print(f"CVXPY ground truth: Z*={s1.Z_star:.4f}  phi*={phi_star:.6f}")

    runs = []
    for name, fn, kwargs in [
        ("subgradient (Polyak)",    subgradient,    dict(max_iter=800, phi_star=phi_star)),
        ("mirror descent (Polyak)", mirror_descent, dict(max_iter=800, phi_star=phi_star)),
        ("ADMM",                    admm,           dict(max_iter=200, rho=150.0)),
    ]:
        t0 = time.perf_counter()
        traj = fn(inst, s1, **kwargs)
        dt = time.perf_counter() - t0
        primal_gap = np.maximum(traj.primal - phi_star, 1e-12)
        print(f"{name:>26s}: {len(traj.iters):4d} iter in {dt:6.2f}s  "
              f"final primal_gap={primal_gap[-1]:.3e}")
        runs.append((name, traj, primal_gap, dt))

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    colors = {"subgradient (Polyak)": "C0", "mirror descent (Polyak)": "C1", "ADMM": "C2"}
    for name, traj, pgap, dt in runs:
        ax.semilogy(traj.iters, pgap, label=f"{name}   ({dt:.1f}s)", color=colors[name], lw=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"primal gap  $\phi_k - \phi^\star$")
    n_I = build_dual_structure(inst)[0].shape[0]
    ax.set_title("Stage 2 dual decomposition: per-class subproblems at given prices\n"
                 f"$Q={inst.T.size}$ classes, $|\\mathcal{{I}}|={n_I}$ pressure indices, "
                 f"$\\phi^\\star={phi_star:.4f}$ (CVXPY ground truth)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage2_convergence.pdf")
    fig.savefig(out / "stage2_convergence.png", dpi=150)
    print(f"wrote {out / 'stage2_convergence.pdf'}")


if __name__ == "__main__":
    main()
