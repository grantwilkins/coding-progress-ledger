"""T13 validation: multi-destination dispatch (formulation.md §4 destination index).

The one shared source-egress row IS the entire multi-destination structure; everything
else is K independent per-destination blocks. Four panels, one per T13 success criterion:

A. K=1 reduces to the single-dest solve exactly (fleet=None == explicit K=1 fleet).
B. K identical destinations add admission capacity linearly — any split is cost-tied.
C. The shared uplink is the coupling: as λ_src tightens, θ_egress rises 0→+ and max-shed
   collapses (drop that one row and K independent dispatches separate).
D. Heterogeneous destinations: routing concentrates on the cheapest-reachable site until
   its admission price θ_admit,ℓ turns positive, then spreads to the next.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import DestFleet, Event, Movement, bind_dp, solve
from impact import compute
from instance import Workload, generate
from power import PoolPower

POOL = PoolPower()
kW = 1e3
fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(12, 9))

# --- A. K=1 reduces to the single-dest program exactly ---
popA = generate(POOL, n_nodes=8)
impA = compute(popA, POOL)
ceilA = solve(popA, POOL, impA, 2 * bind_dp(impA).sum()).shed_guaranteed
S = np.linspace(0.05, 0.95, 16) * ceilA
fK1 = DestFleet.from_event(Event(), Movement(), POOL, popA)
none_ = np.array([solve(popA, POOL, impA, s).shed_guaranteed for s in S])
k1_ = np.array([solve(popA, POOL, impA, s, fleet=fK1).shed_guaranteed for s in S])
axA.plot(S / kW, none_ / kW, lw=4, alpha=0.45, color="tab:blue", label="fleet=None (single-dest)")
axA.plot(S / kW, k1_ / kW, lw=1.3, ls="--", color="tab:red", label="explicit K=1 DestFleet")
axA.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="achieved shed (kW)",
        title=f"A. K=1 reduces exactly  (max |Δ| = {np.abs(none_ - k1_).max():.1e} W)")
axA.legend(loc="upper left", fontsize=8)

# --- B. homogeneous K-sweep: K identical dests ≡ one K×-capacity dest (any split optimal) ---
popB = generate(POOL, Workload(state_mix=(0.8, 0.1, 0.1)), n_nodes=8)  # many active so admission binds
impB = compute(popB, POOL)
MVB = Movement(lambda_src=1e12)  # uplink slack ⇒ per-dest admission sets the ceiling
Ks = list(range(1, 7))
split = np.array([solve(popB, POOL, impB, 1e12, move=MVB,
                  fleet=DestFleet(np.full(K, 8), np.full(K, 1.0), np.full(K, popB.mfu), np.full(K, 0.6))
                  ).shed_guaranteed for K in Ks]) / kW
merged = np.array([solve(popB, POOL, impB, 1e12, move=MVB,
                   fleet=DestFleet(np.array([8 * K]), np.array([1.0 * K]), np.array([popB.mfu]), np.array([0.6]))
                   ).shed_guaranteed for K in Ks]) / kW
axB.plot(Ks, merged, lw=4, alpha=0.45, color="tab:blue", label="1 dest, $K\\times$ capacity")
axB.plot(Ks, split, "o--", color="tab:red", ms=6, label="$K$ identical dests")
axB.set(xlabel="number of identical destinations $K$", ylabel="max achievable shed (kW)",
        title=f"B. split $\\equiv$ merged  (max |Δ| = {np.abs(split - merged).max():.1e} kW)")
axB.legend(loc="upper left", fontsize=8)

# --- C. the shared uplink is the coupling: θ_egress rises as λ_src tightens ---
POOLc = replace(POOL, mean_context_tokens=163000)  # track the long-context E[T]
popC = generate(POOLc, Workload(state_mix=(1.0, 0.0, 0.0), t_mix=((1.0, 12.0, 0.2),)), n_nodes=6)
EVC = Event(D=300, tau_pre=299, W=8)  # prefill window ~0 ⇒ every move must transfer the shared link
lams = np.logspace(8.4, 12, 14)
fC = lambda: DestFleet(np.array([8, 8]), np.array([60.0, 60.0]), np.array([popC.mfu] * 2), np.array([0.6, 0.6]))
shedC, teC = [], []
for lam in lams:
    MVc = Movement(lambda_src=lam, mu_in=1e13)  # slack ingest ⇒ the uplink is the sole bottleneck
    p = solve(popC, POOLc, compute(popC, POOLc, MVc), 1e12, EVC, MVc, fleet=fC())
    shedC.append(p.shed_guaranteed / kW)
    teC.append(p.theta_egress)
axC.plot(lams, shedC, "o-", color="tab:blue", label="max shed")
axC.set(xscale="log", xlabel="shared uplink $\\Lambda_{src}$ (B/s)", ylabel="max shed (kW)",
        title="C. shared uplink binds → shed collapses, $\\theta_{egress}>0$")
axC2 = axC.twinx()
axC2.plot(lams, teC, "s-", color="tab:red", label="$\\theta_{egress}$ (W/byte)")
axC2.set_ylabel("$\\theta_{egress}$ (W/byte)", color="tab:red")
axC2.tick_params(axis="y", labelcolor="tab:red")
axC.legend(loc="center left", fontsize=8)
axC2.legend(loc="upper right", fontsize=8)

# --- D. heterogeneous dests: concentrate on cheapest-reachable, then spread ---
popD = generate(POOL, n_nodes=8)
MVD = Movement(lambda_src=1e10, mu_in=1e9)  # pricey transfer ⇒ jobs replay ⇒ per-dest ρ_ℓ drives routing
impD = compute(popD, POOL, MVD)
fD = lambda: DestFleet(np.array([8, 8]), np.array([1.5, 1.5]), np.array([0.5, 0.3]), np.array([0.6, 0.6]))
ceilD = solve(popD, POOL, impD, 2 * bind_dp(impD).sum(), move=MVD, fleet=fD()).shed_guaranteed
Sd = np.linspace(0.05, 1.0, 18) * ceilD
m0, m1, ta0 = [], [], []
for s in Sd:
    p = solve(popD, POOL, impD, s, move=MVD, fleet=fD())
    moved = (p.Y_R + p.Y_S).sum(0)
    m0.append(moved[0]); m1.append(moved[1]); ta0.append(p.theta_admit[0])
axD.plot(Sd / kW, m0, "o-", color="tab:green", label="dest 0 (fast rebuild, mfu 0.5)")
axD.plot(Sd / kW, m1, "s-", color="tab:purple", label="dest 1 (slow, mfu 0.3)")
axD.set(xlabel="requested shed $S^\\star$ (kW)", ylabel="sessions routed (Σ y)",
        title="D. concentrate on cheapest-reachable, then spread")
axD.legend(loc="upper left", fontsize=8)
axDt = axD.twinx()
axDt.plot(Sd / kW, ta0, color="0.5", lw=1, ls=":", label="$\\theta_{admit,0}$")
axDt.set_ylabel("$\\theta_{admit,0}$", color="0.5")
axDt.legend(loc="lower right", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/multidest_t13.{ext}", dpi=150)

print(f"A K=1 reduction: max|Δshed|={np.abs(none_ - k1_).max():.2e} W")
print(f"B homogeneous split≡merged: max|Δ|={np.abs(split - merged).max():.2e} kW")
print(f"C uplink: θ_egress {teC[-1]:.2e} (slack) → {teC[0]:.2e} (bind); shed {shedC[-1]:.0f}→{shedC[0]:.0f} kW")
print(f"D routing: dest0 saturates at Σy≈{max(m0):.1f}, θ_admit,0 0→{max(ta0):.1f}; dest1 fills after")
