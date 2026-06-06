"""Mix sensitivity: sweep the flagship's token share (the provider's primary
workload knob) and show how the rebuild plan and the evacuation frontier respond.

Left  — aggregate replay vs state-transfer share of migrated jobs at D=300s
         (prop-fair Stage 1 -> peak-pressure Stage 2), mean +/-1 sd over seeds.
Right — minimum deadline to evacuate every session (Stage-1 throughput, the
         feasibility frontier: smallest D with Z* ~ 0), mean +/-1 sd over seeds.

As the heavy-eta flagship takes more of the token mass, both the transfer band
and the deadline to clear the site grow. Compute is cached to
outputs/flagship_sensitivity.json. Re-run with --recompute.

Usage:
    cd evacuation && uv run python plot_flagship_sensitivity.py [--recompute]
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
JSON = OUT / "flagship_sensitivity.json"
SHARES = [0.30, 0.40, 0.50, 0.55, 0.70, 0.80, 0.90]
SEEDS = range(8)
D_MIX = 300.0
D_LO, D_HI = 5.0, 1200.0   # bisection bracket for the full-evac deadline
N_WORKERS = 8


def _replay_share(s, seed):
    inst = build_instance(D=D_MIX, total_jobs=10_000, n_bins=5, seed=seed, flagship_share=s)
    s1 = solve_stage1(inst, "prop_fair")
    s2 = solve_stage2(inst, s1)
    R, S = s2.x_R.sum(), s2.x_S.sum()
    return 100.0 * R / (R + S)


def _min_full_evac_D(s, seed):
    # Bisect the smallest deadline at which throughput Stage 1 strands nothing.
    def stranded(D):
        inst = build_instance(D=D, total_jobs=10_000, n_bins=5, seed=seed, flagship_share=s)
        return solve_stage1(inst, "throughput").Z_star
    lo, hi = D_LO, D_HI
    if stranded(hi) > 1e-3:
        return D_HI  # not fully evacuable within the bracket
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if stranded(mid) > 1e-3:
            lo = mid
        else:
            hi = mid
    return hi


def _run(args):
    s, seed = args
    try:
        return s, _replay_share(s, seed), _min_full_evac_D(s, seed)
    except (cp.error.SolverError, RuntimeError):
        return None


def compute() -> dict:
    tasks = [(s, seed) for s in SHARES for seed in SEEDS]
    with Pool(N_WORKERS) as pool:
        res = [r for r in pool.map(_run, tasks) if r is not None]
    out = {"shares": SHARES, "replay_mean": [], "replay_sd": [], "D_mean": [], "D_sd": []}
    for s in SHARES:
        rep = np.array([r[1] for r in res if r[0] == s])
        Dd = np.array([r[2] for r in res if r[0] == s])
        out["replay_mean"].append(float(rep.mean())); out["replay_sd"].append(float(rep.std()))
        out["D_mean"].append(float(Dd.mean())); out["D_sd"].append(float(Dd.std()))
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(out))
    for i, s in enumerate(SHARES):
        print(f"flagship {s:.2f}:  replay {out['replay_mean'][i]:5.1f}%   "
              f"min-D {out['D_mean'][i]:6.1f}s")
    return out


def main() -> None:
    d = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())
    s = np.array(d["shares"]) * 100
    rm, rs = np.array(d["replay_mean"]), np.array(d["replay_sd"])
    Dm, Ds = np.array(d["D_mean"]), np.array(d["D_sd"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    axL.fill_between(s, rm, 100, color="#edaaa0", alpha=0.6, label="state transfer")
    axL.fill_between(s, 0, rm, color="#c44536", alpha=0.85, label="replay")
    axL.errorbar(s, rm, yerr=rs, fmt="o-", color="0.1", capsize=4, lw=1.6, ms=5)
    axL.set_ylim(0, 100); axL.set_xlim(s[0], s[-1])
    axL.set_xlabel("Flagship token share (%)", fontsize=15)
    axL.set_ylabel("Share of migrated jobs (%)", fontsize=15)
    axL.legend(fontsize=12, loc="center left")
    axL.tick_params(labelsize=12); axL.grid(True, alpha=0.3); axL.set_axisbelow(True)

    axR.errorbar(s, Dm, yerr=Ds, fmt="o-", color="#3a7ca5", capsize=4, lw=2.0, ms=6)
    axR.set_xlim(s[0], s[-1]); axR.set_ylim(bottom=0)
    axR.set_xlabel("Flagship token share (%)", fontsize=15)
    axR.set_ylabel("Min deadline for full evacuation (s)", fontsize=15)
    axR.tick_params(labelsize=12); axR.grid(True, alpha=0.3); axR.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "flagship_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "flagship_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'flagship_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
