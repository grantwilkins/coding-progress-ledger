"""How the saddle-point methods (PDHG, proximal bundle) converge on the Stage 2
dual, vs the dual-ascent / splitting methods, as a distribution over instances.

Runs N_SEEDS instances in parallel; for PDHG and the proximal bundle method we
record the RELATIVE primal gap (phi_k - phi*)/phi* and draw median + 25-75 pct
band. The three established solvers (subgradient, mirror descent, ADMM) are
overlaid as thin reference medians from stage2_convergence_dist.json (run
plot_stage2_convergence.py first to populate it).

Compute is cached to outputs/pdhg_bundle_dist.json. Re-run with --recompute.

Usage:
    cd evacuation && uv run python plot_stage2_pdhg_bundle.py [--recompute]
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
from stage2_dual import bundle, pdhg

OUT = Path(__file__).resolve().parent / "outputs"
JSON = OUT / "pdhg_bundle_dist.json"
REF_JSON = OUT / "stage2_convergence_dist.json"

N_SEEDS = 50
N_WORKERS = 8
PDHG_ITERS = 800
BUNDLE_ITERS = 120
NEW = {"PDHG": "#6a4c93", "proximal bundle": "#c44536"}
REF = {"subgradient": "#3a7ca5", "mirror descent": "#e8943a", "ADMM": "#4a9b54"}


def _run_seed(seed: int):
    inst = build_instance(D=300.0, total_jobs=200, seed=seed)
    s1 = solve_stage1(inst)
    if s1.Z_star > 1e-6:
        return None
    phi = solve_stage2(inst, s1).phi_star
    tp = pdhg(inst, s1, phi_star=phi, max_iter=PDHG_ITERS)
    tb = bundle(inst, s1, phi_star=phi, max_iter=BUNDLE_ITERS)
    return {
        "PDHG": np.maximum((tp.primal - phi) / phi, 1e-9).tolist(),
        "proximal bundle": np.maximum((tb.primal - phi) / phi, 1e-9).tolist(),
    }


def _pad(gaps: list[np.ndarray]) -> np.ndarray:
    n = max(len(g) for g in gaps)
    return np.vstack([np.concatenate([g, np.full(n - len(g), g[-1])]) for g in gaps])


def compute() -> dict:
    with Pool(N_WORKERS) as pool:
        runs = [r for r in pool.map(_run_seed, range(N_SEEDS)) if r is not None]
    print(f"aggregated {len(runs)}/{N_SEEDS} instances")
    agg = {"n_seeds": len(runs)}
    for name in NEW:
        g = _pad([np.array(r[name]) for r in runs])
        agg[name] = {"median": np.median(g, axis=0).tolist(),
                     "lo": np.percentile(g, 25, axis=0).tolist(),
                     "hi": np.percentile(g, 75, axis=0).tolist()}
        print(f"{name:>16s}: median final rel-gap {agg[name]['median'][-1]:.2e}")
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(agg))
    return agg


def main() -> None:
    agg = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())

    fig, ax = plt.subplots(figsize=(12, 5.6))
    if REF_JSON.exists():
        ref = json.loads(REF_JSON.read_text())
        for name, c in REF.items():
            med = np.array(ref[name]["median"])
            ax.semilogy(np.arange(1, len(med) + 1), med, color=c, lw=1.3, ls="--",
                        alpha=0.7, label=f"{name} (ref)")
    for name, c in NEW.items():
        med = np.array(agg[name]["median"])
        lo, hi = np.array(agg[name]["lo"]), np.array(agg[name]["hi"])
        it = np.arange(1, len(med) + 1)
        ax.semilogy(it, med, color=c, lw=2.8, label=name, zorder=5)
        ax.fill_between(it, lo, hi, color=c, alpha=0.2, lw=0, zorder=4)

    ax.set_xlabel("iteration", fontsize=17)
    ax.set_ylabel(r"relative primal gap  $(\phi_k - \phi^\star)/\phi^\star$", fontsize=17)
    ax.tick_params(labelsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=13, loc="upper right", ncol=2)
    fig.savefig(OUT / "stage2_pdhg_bundle.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stage2_pdhg_bundle.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'stage2_pdhg_bundle.pdf'}  (band = 25-75 pct over {agg['n_seeds']} instances)")


if __name__ == "__main__":
    main()
