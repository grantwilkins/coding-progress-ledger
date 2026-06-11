"""Occupancy sensitivity: sweep per-rack session occupancy (+/-50% around the
HBM-derived fit) and show the evacuation frontier and the rebuild plan.

Left  — minimum deadline to evacuate everything the destination can hold
         (Stage-1 throughput; smallest D with Z* at the residency floor
         Z_min(o) = Z* as D -> inf, which is > 0 for o > 1: the decode-HBM
         wall), mean +/-1 sd over seeds.
Right — KV-weighted disposition of the pod at that deadline: replay vs state
         transfer vs stranded (residency-infeasible), mean over seeds.

Compute is cached to outputs/occupancy_deadline.json. Re-run with --recompute.

Usage:
    cd evacuation && uv run python plot_occupancy_deadline.py [--recompute]
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
JSON = OUT / "occupancy_deadline.json"
OCCUPANCIES = [0.50, 0.75, 1.00, 1.25, 1.50]
SEEDS = range(8)
D_HUGE = 50_000.0          # residency floor probe (deadline never binds)
D_LO, D_HI = 10.0, 5_000.0  # bisection bracket for the frontier
N_WORKERS = 8


def _point(args):
    o, seed = args
    try:
        z_min = solve_stage1(build_instance(D=D_HUGE, occupancy=o, seed=seed)).Z_star
        lo, hi = D_LO, D_HI
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            inst = build_instance(D=mid, occupancy=o, seed=seed)
            if solve_stage1(inst).Z_star > z_min + 1e-2:
                lo = mid
            else:
                hi = mid
        inst = build_instance(D=hi, occupancy=o, seed=seed)
        s2 = solve_stage2(inst, solve_stage1(inst))
        kv = inst.eta * inst.T
        kv_tot = float(kv @ inst.n)
        return {"o": o, "minD": hi, "z_min": z_min,
                "replay": float(kv @ s2.x_R.sum(axis=1)) / kv_tot,
                "state": float(kv @ s2.x_S.sum(axis=1)) / kv_tot,
                "stranded": float(kv @ s2.z) / kv_tot}
    except (cp.error.SolverError, RuntimeError):
        return None


def compute() -> dict:
    with Pool(N_WORKERS) as pool:
        res = [r for r in pool.map(_point, [(o, s) for o in OCCUPANCIES for s in SEEDS]) if r]
    out = {"occ": OCCUPANCIES}
    for key in ("minD", "replay", "state", "stranded"):
        vals = [np.array([r[key] for r in res if r["o"] == o]) for o in OCCUPANCIES]
        out[f"{key}_mean"] = [float(v.mean()) for v in vals]
        out[f"{key}_sd"] = [float(v.std()) for v in vals]
    OUT.mkdir(exist_ok=True)
    JSON.write_text(json.dumps(out))
    for i, o in enumerate(OCCUPANCIES):
        print(f"o={o:.2f}:  min-D {out['minD_mean'][i]:7.1f}s   replay {out['replay_mean'][i]:.2f}"
              f"   state {out['state_mean'][i]:.2f}   stranded {out['stranded_mean'][i]:.2f}")
    return out


def main() -> None:
    d = compute() if "--recompute" in sys.argv or not JSON.exists() else json.loads(JSON.read_text())
    occ = np.array(d["occ"])
    Dm, Ds = np.array(d["minD_mean"]), np.array(d["minD_sd"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    axL.errorbar(occ, Dm, yerr=Ds, fmt="o-", color="#3a7ca5", capsize=4, lw=2.0, ms=6)
    axL.axvspan(1.0, occ[-1], color="0.85", alpha=0.5)
    axL.text(1.02, 0.05, "decode-HBM wall:\nfull evacuation infeasible",
             transform=axL.get_xaxis_transform(), fontsize=11, color="0.3")
    axL.set_xlabel("Per-rack occupancy (x fitted)", fontsize=15)
    axL.set_ylabel("Min deadline to clear the pod (s)", fontsize=15)
    axL.set_xlim(occ[0], occ[-1]); axL.set_ylim(bottom=0)
    axL.tick_params(labelsize=12); axL.grid(True, alpha=0.3); axL.set_axisbelow(True)

    rep, st, strd = (np.array(d[f"{k}_mean"]) for k in ("replay", "state", "stranded"))
    w = 0.16
    axR.bar(occ, rep, w, color="#c44536", label="replay")
    axR.bar(occ, st, w, bottom=rep, color="#edaaa0", label="state transfer")
    axR.bar(occ, strd, w, bottom=rep + st, color="0.6", label="stranded")
    axR.set_xlabel("Per-rack occupancy (x fitted)", fontsize=15)
    axR.set_ylabel("Share of pod KV bytes", fontsize=15)
    axR.set_ylim(0, 1.0); axR.set_xticks(occ)
    axR.legend(fontsize=11, loc="lower left")
    axR.tick_params(labelsize=12); axR.grid(True, alpha=0.3, axis="y"); axR.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "occupancy_deadline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "occupancy_deadline.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'occupancy_deadline.pdf'}")


if __name__ == "__main__":
    main()
