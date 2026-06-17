"""T16 — Figure 2: realized vs certified routing under uplink contention (the headline).

Replays the T14 saturation-band plans through the per-ℓ DES (T15). PURE transfer (prefill budget
W·(D−τ_pre)=0 ⇒ Y_R≡0), so one shared egress link serializes every (j,ℓ) shipment and K clusters
ingest in parallel; under Λ_ℓ≥λ_src rebuild≈egress. The reconstruction gap is then the executable
shadow of θ_egress: where the uplink is slack (θ_egress=0) the DES reconstructs exactly what the LP
certified (§4 routing execution-exact); where it binds (θ_egress>0) egress saturates to D, so the
last shipments can't rebuild in time and their destinations under-fill — realized < certified, the
multi-dest "certify low, report high." Tightening the execution window amplifies the same tail.

A · across the K-band: certified shed vs DES reconstruction at the planned D and a tight 0.5D, with
    θ_egress on a twin axis — the gap opens exactly where θ_egress turns positive (uplink binds).
B · per-destination held admission at the most-binding K + tight D: realized (resident by D) vs
    certified sessions vs the held cap spare_ℓ·S_node — under-fill lands on the last-egressed ℓ,
    and no destination is ever over-admitted (realized ≤ certified ≤ cap).
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
from simulate import simulate

POOL = replace(PoolPower(), mean_context_tokens=163000)  # long context ⇒ huge η·T, memory regime
pop = generate(POOL, Workload(state_mix=(1.0, 0.0, 0.0), t_mix=((1.0, 12.0, 0.2),)), n_nodes=8)
EV = Event(D=300, tau_pre=300, W=8)        # prefill budget W·(D−τ_pre)=0 ⇒ Y_R≡0 ⇒ PURE transfer
MV = Movement(lambda_src=1e9, mu_in=1e13)  # WAN uplink binds; slack ingest ⇒ gap is pure uplink tail
imp = compute(pop, POOL, MV)
kW = 1e3

# Band sizing (identical to the T14 gate): per-dest mean spare fixed, K sweeps Σspare through
# [1.1,1.5]×demand with max_ℓ spare_ℓ < demand, so the uplink crossover lands at K_cross.
demand = MV.lambda_src * (EV.D - EV.tau_src) / imp.b_transfer.mean()
K_cross = 5
spare_bar = demand / (K_cross * POOL.s_node)


def fleet(K):
    g = np.linspace(0.6, 1.4, K)
    g *= K / g.sum()  # graded, mean 1 ⇒ Σspare = K·spare_bar (band traversal stays exact)
    return DestFleet(np.full(K, EV.W), spare_bar * g, np.full(K, pop.mfu), np.full(K, 0.6))


Ks = np.arange(1, 13)
fleets = [fleet(K) for K in Ks]
plans = [solve(pop, POOL, imp, 1e15, EV, MV, fleet=f) for f in fleets]  # max-shed: the capacity price
certified = np.array([p.shed_guaranteed for p in plans])
te = np.array([p.theta_egress for p in plans])


def recon_shed(plan, fl, D):  # shed reconstructed (rebuild ≤ D) when executed under window D
    return simulate(pop, POOL, imp, plan, replace(EV, D=D), MV, fleet=fl).reconstruction_shed


PLANNED, TIGHT = EV.D, 0.5 * EV.D
recon_D = np.array([recon_shed(p, f, PLANNED) for p, f in zip(plans, fleets)])
recon_t = np.array([recon_shed(p, f, TIGHT) for p, f in zip(plans, fleets)])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- A. across the K-band: certified vs reconstructed shed, gap = executable shadow of θ_egress ---
band = (1.1 * K_cross, 1.5 * K_cross)
axA.axvspan(*band, color="orange", alpha=0.12, label="saturation band")
axA.fill_between(Ks, recon_D / kW, certified / kW, color="tab:red", alpha=0.12, label="contention tail")
axA.plot(Ks, certified / kW, "k-", lw=2.5, label=r"certified $S^\star$ (LP)")
axA.plot(Ks, recon_D / kW, "o-", color="tab:green", lw=1.8, ms=4, label="reconstructed @ planned $D$")
axA.plot(Ks, recon_t / kW, "^--", color="tab:red", lw=1.8, ms=4, label="reconstructed @ tight $0.5D$")
axA.set(xlabel="number of destinations $K$", ylabel="shed (kW)",
        title="A · reconstruction tracks certified until the uplink binds")
axA.legend(loc="center left", fontsize=8)
axA2 = axA.twinx()
axA2.plot(Ks, te, "x:", color="0.5", lw=1, ms=5, label=r"$\theta_{egress}$")
axA2.set_ylabel(r"$\theta_{egress}$ (W/byte)", color="0.5")
axA2.tick_params(axis="y", labelcolor="0.5")
axA2.legend(loc="lower right", fontsize=8)

# --- B. per-ℓ held admission at the binding onset + tight D: realized vs certified vs held cap ---
idx = int(np.argmax(te > 1e-12))        # first K where the uplink binds; admission still fills every ℓ to cap
KB, fB, pB = int(Ks[idx]), fleets[idx], plans[idx]
DB = 0.5 * EV.D
rB = simulate(pop, POOL, imp, pB, replace(EV, D=DB), MV, fleet=fB)
Y = pB.Y_R + pB.Y_S
order = np.argsort(-fB.spare)           # destinations sorted by spare (held cap)
cert_sess = Y.sum(0)[order]
resident = (np.where(np.isfinite(rB.rebuild_done), rB.rebuild_done, np.inf) <= DB)
real_sess = (Y * resident).sum(0)[order]
held_cap = (fB.spare * POOL.s_node)[order]
x = np.arange(KB)
axB.bar(x - 0.2, cert_sess, 0.4, color="tab:blue", alpha=0.55, label="certified (LP)")
axB.bar(x + 0.2, real_sess, 0.4, color="tab:green", alpha=0.85, label=f"realized @ $0.5D$ (resident)")
axB.plot(x, held_cap, "k_", ms=18, mew=2, label=r"held cap $spare_\ell\!\cdot\!S_{node}$")
axB.set(xlabel=f"destination $\\ell$ (sorted by spare, $K={KB}$, tight $0.5D$)", ylabel="sessions",
        title="B · per-ℓ admission: under-fill, never over-admit")
axB.set_xticks(x)
axB.legend(loc="upper right", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/multidest_validation.{ext}", dpi=150)

# --- diagnostics: exact where uplink slack, contention tail where it binds, no over-admission ---
slack, bind = te < 1e-12, te > 1e-12
exact_slack = bool(np.allclose(recon_D[slack], certified[slack], rtol=1e-9))   # θ_egress=0 ⇒ exact
tail_bind = bool(np.any(recon_D[bind] < certified[bind] - 1e-6)) if bind.any() else False  # θ_egress>0 ⇒ tail
onset = float(np.max((certified - recon_t) / np.maximum(certified, 1e-300)))
no_over = bool(np.all(real_sess <= cert_sess + 1e-6) and np.all(cert_sess <= held_cap + 1e-6))
# realized_load ≤ load_cap (the load-regime admission check) across the whole band, every ℓ
load_ok = all(bool(np.all(simulate(pop, POOL, imp, p, replace(EV, D=TIGHT), MV, fleet=f).realized_load
                          <= f.spare * POOL.rho_star + 1e-6)) for p, f in zip(plans, fleets))
gate = te[0] < 1e-12 < te[-1]
print(f"regime={imp.regime}  jobs={len(pop)}  pure-transfer(Y_R≡0)={max(float(np.nansum(p.Y_R)) for p in plans)==0}"
      f"  demand≈{demand:.1f} sessions  band K∈[{band[0]:.1f},{band[1]:.1f}]")
print(f"θ_egress: {te[0]:.2e}(K=1) → {te[-1]:.2e}(K=12)  uplink-binds-in-band={gate}")
print(f"EXACT where uplink slack (θ_egress=0 ⇒ reconstruction==certified, rel 1e-9): {exact_slack}")
print(f"CONTENTION TAIL where uplink binds (θ_egress>0 ⇒ realized<certified): {tail_bind}  "
      f"| amplified to {onset:.0%} under-fill at tight 0.5D")
print(f"NO over-admission (realized ≤ certified ≤ held cap, all ℓ): {no_over}  "
      f"| load admission realized_load ≤ L̄ across band: {load_ok}")
print(f"Panel B K={KB} @0.5D: certified={cert_sess.round(1)} realized={real_sess.round(1)} cap={held_cap.round(1)}")
# ρ_ℓ heterogeneity (Hopper/Blackwell) sweep: inert here because rebuild is ingest (μ_in), not prefill.
flh = DestFleet(np.full(KB, EV.W), fB.spare, np.linspace(0.30, 0.50, KB), np.full(KB, 0.6))
rh = simulate(pop, POOL, imp, solve(pop, POOL, imp, 1e15, EV, MV, fleet=flh), replace(EV, D=DB), MV, fleet=flh)
print(f"ρ_ℓ mix (mfu 0.30→0.50): reconstruction {rh.reconstruction_shed/kW:.1f}kW vs uniform "
      f"{rB.reconstruction_shed/kW:.1f}kW — NULL: forced-transfer rebuild is ingest, so ρ_ℓ doesn't bite")
ok = exact_slack and tail_bind and no_over and load_ok and gate
print(f"RESULT: {'§4 routing execution-exact where uplink slack; certify-low-report-high tail where it binds; no over-admission' if ok else 'CHECK — see above'}")
