"""T14 gate: θ_egress K-sweep (formulation.md §4 — the one shared egress row is the whole story).

Force transfer (only η·T bytes bind a WAN uplink; replay's β·T can't), grade spare_ℓ across K
destinations at a fixed per-dest mean, and sweep K so Σspare climbs through the saturation band.
As admission slackens the binding constraint migrates admission→egress: θ_admit→0, θ_egress 0→+.
Duals are read on the max-shed LP (huge S* ⇒ the re-solve path), the S*-independent capacity price.

A. migration vs K: θ_egress rises, θ_admit falls, band shaded — the gate.
B. admission-limited snapshot (small K): routing pinned to graded spare_ℓ, θ_admit>0 across.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import DestFleet, Event, Movement, solve
from impact import compute
from instance import Workload, generate
from power import PoolPower

POOL = replace(PoolPower(), mean_context_tokens=163000)  # long context ⇒ huge η·T, memory regime
pop = generate(POOL, Workload(state_mix=(1.0, 0.0, 0.0), t_mix=((1.0, 12.0, 0.2),)), n_nodes=8)
EV = Event(D=300, tau_pre=299, W=8)        # prefill window ~1s ⇒ replay infeasible ⇒ forced transfer
MV = Movement(lambda_src=1e9, mu_in=1e13)  # WAN uplink; slack ingest ⇒ held is the per-ℓ wall
imp = compute(pop, POOL, MV)

# Size so the admission↔egress crossover lands at K_cross: demand = uplink-feedable sessions,
# per-dest mean spare fixed ⇒ Σspare·S_node = K·demand/K_cross sweeps the band over K.
demand = MV.lambda_src * (EV.D - EV.tau_src) / imp.b_transfer.mean()
K_cross = 5
spare_bar = demand / (K_cross * POOL.s_node)


def fleet(K):
    g = np.linspace(0.6, 1.4, K)
    g *= K / g.sum()  # graded, mean 1 ⇒ Σspare = K·spare_bar (band traversal stays exact)
    return DestFleet(np.full(K, EV.W), spare_bar * g, np.full(K, pop.mfu), np.full(K, 0.6))


Ks = np.arange(1, 13)
plans = [solve(pop, POOL, imp, 1e15, EV, MV, fleet=fleet(K)) for K in Ks]
te = np.array([p.theta_egress for p in plans])
ta = np.array([p.theta_admit.max() for p in plans])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- A. admission→egress migration (the gate) ---
band = (1.1 * K_cross, 1.5 * K_cross)
axA.axvspan(*band, color="orange", alpha=0.15, label="saturation band")
axA.plot(Ks, te, "o-", color="tab:red", label=r"$\theta_{egress}$")
axA.set(xlabel="number of destinations $K$", ylabel=r"$\theta_{egress}$ (W/byte)",
        title="A. binding constraint migrates admission$\\to$egress")
axA.tick_params(axis="y", labelcolor="tab:red")
axA.legend(loc="upper left", fontsize=8)
axA2 = axA.twinx()
axA2.plot(Ks, ta, "s--", color="tab:blue", label=r"$\max_\ell\,\theta_{admit,\ell}$")
axA2.set_ylabel(r"$\max_\ell\,\theta_{admit,\ell}$", color="tab:blue")
axA2.tick_params(axis="y", labelcolor="tab:blue")
axA2.legend(loc="upper right", fontsize=8)

# --- B. admission-limited snapshot: routing pinned to graded spare_ℓ ---
K_B = 3
fB = fleet(K_B)
pB = solve(pop, POOL, imp, 1e15, EV, MV, fleet=fB)
order = np.argsort(-fB.spare)
routed = (pB.Y_R + pB.Y_S).sum(0)[order]
cap = (fB.spare * POOL.s_node)[order]
x = np.arange(K_B)
axB.bar(x, routed, color="tab:green", alpha=0.8, label=r"routed $\Sigma_j y_{j\ell}$")
axB.plot(x, cap, "k_", ms=22, mew=2, label=r"held cap $spare_\ell\!\cdot\!S_{node}$")
axB.set(xlabel=f"destination $\\ell$ (sorted by spare, $K={K_B}$)", ylabel="sessions",
        title="B. admission-limited: routing pinned to graded spare")
axB.set_xticks(x)
axB.legend(loc="upper right", fontsize=8)
axBt = axB.twinx()
axBt.plot(x, pB.theta_admit[order], "D:", color="0.4", label=r"$\theta_{admit,\ell}$")
axBt.set_ylabel(r"$\theta_{admit,\ell}$", color="0.4")
axBt.tick_params(axis="y", labelcolor="0.4")
axBt.legend(loc="lower left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/multidest_dual.{ext}", dpi=150)

monotone = bool(np.all(np.diff(te) >= -1e-15))
gate = te[0] < 1e-12 < te[-1]
print(f"demand={demand:.1f} sessions  K_cross={K_cross}  band K∈[{band[0]:.1f},{band[1]:.1f}]  jobs={len(pop)}")
print(f"θ_egress: {te[0]:.2e} (K=1, slack) → {te[-1]:.2e} (K=12, bind)  monotone={monotone}")
print(f"GATE {'PASS' if gate and monotone else 'FAIL — resize fleet'}")
print(f"B K={K_B}: routed={routed.round(2)} cap={cap.round(2)} "
      f"pinned={np.allclose(routed, cap, rtol=1e-6)} θ_admit>0={bool(np.all(pB.theta_admit > 0))}")
