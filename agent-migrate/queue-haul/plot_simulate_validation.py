"""Validation plot for simulate.py (formulation.md §10.2 reconstruction DES).

Replays a solved LP plan through the deterministic flow shop and points the sweeps at
the gaps that actually exist here — the precedence/pipeline-fill gap is near-null
(ingest non-binding, replay stage-1 tiny), so the store-and-forward↔cut-through spread
is ≈0 (printed). Two real gaps remain:

A — Grid relief: realized shed vs deadline D for four link disciplines. The serial
egress link clears only a prefix by D, and a few giant KV transfers dominate it. LPT
(longest first) ships those first and clears almost nothing; FIFO is arbitrary;
power-density (PD) and Johnson both push the transfers last and bank the most watts —
the cost of the LP being execution-order-blind.
B — Service continuity: the prefill stage's P‖Cmax packing gap. The LP's prefill row budgets
τ_pre+Σp2/W (perfect packing); discrete jobs don't divide evenly across W servers, so the DES
makespan exceeds it. Across the PHYSICAL pool W∈{4,8,16} (assumptions §5, center 8) the LP
over-promises ~3–13% — small but NOT a null: the single-largest-job floor W*=Σp2/max_p2 (≈49
here) sits far past the physical range, so this is pure list-scheduling loss (not a max-job
effect) and it keeps growing past the pool (24% at W=32). Envelope containment lb≤makespan≤ub
is enforced in the tests, not re-drawn here.
C — The lever that worsens the gap WITHIN the physical pool is the context tail: at W=8 the
packing gap rises with the CoV of prefill time T/ρ_dest(T). (The gap is NOT a function of W/W*
alone — it depends on the full prefill-time distribution, so B and C don't collapse.)

Note: raising transfer-fraction shifts the bottleneck from prefill-packing (B/C) to
link-serialization (A) rather than cleanly shrinking either — the two gaps trade off.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import Event, bind_dp, solve
from impact import Movement, compute
from instance import Workload, generate
from power import PoolPower, rho_dest
from simulate import simulate

POOL, MOVE = PoolPower(), Movement()
MOVE_LS = replace(MOVE, lambda_src=1e18)  # slack link to isolate the prefill stage
POP = generate(POOL, n_nodes=16)
IMP = compute(POP, POOL)
EV = Event(dest_nodes=48, W=16)
PLAN = solve(POP, POOL, IMP, 0.5 * bind_dp(IMP).sum(), EV, MOVE)
certified = PLAN.shed_guaranteed
egress = (PLAN.y_R @ IMP.b_replay + PLAN.y_S @ IMP.b_transfer) / MOVE.lambda_src
kW = 1e3
DISC = {"fifo": "tab:gray", "lpt": "tab:orange", "johnson": "tab:red", "pd": "tab:blue"}

# --- Panel A: realized shed vs deadline, by discipline ---
Dgrid = EV.tau_src + np.linspace(0.05, 1.2, 24) * egress
A = {d: np.array([simulate(POP, POOL, IMP, PLAN, replace(EV, D=D), MOVE, discipline=d).realized_shed
                  for D in Dgrid]) for d in DISC}

# --- Panel B: prefill packing gap (LP over-promise) vs W; physical pool {4,8,16} (§5) ---
reb = POP.T / rho_dest(POP.T, POP.mfu)
rmask = (PLAN.action == "R") & (PLAN.y > 1e-9)
p2R = (PLAN.y_R * reb)[rmask]
Wstar = p2R.sum() / p2R.max()  # W past which the single largest prefill becomes the binding floor
Wgrid = np.array([1, 2, 4, 8, 16, 32, 64])
gapB = np.array([simulate(POP, POOL, IMP, PLAN, replace(EV, W=int(w)), MOVE_LS, discipline="johnson").makespan
                 / (EV.tau_pre + p2R.sum() / w) - 1 for w in Wgrid])  # DES finish vs LP volume-row promise

# --- Panel C: packing gap vs prefill-time CoV (regenerate at fixed E[T], W=8) ---
mu0 = np.log(POOL.mean_context_tokens)
covs, gapC = [], []
for sg in np.linspace(0.3, 1.4, 7):
    wl = replace(Workload(), t_mix=((1.0, mu0 - sg**2 / 2, sg),))
    pop = generate(POOL, wl, n_nodes=16); imp = compute(pop, POOL)
    plan = solve(pop, POOL, imp, 0.5 * bind_dp(imp).sum(), EV, MOVE)
    rm = (plan.action == "R") & (plan.y > 1e-9)
    if rm.sum() < 2:
        continue
    r = pop.T[rm] / rho_dest(pop.T[rm], pop.mfu)
    s = simulate(pop, POOL, imp, plan, replace(EV, W=8), MOVE_LS, discipline="johnson")
    covs.append(r.std() / r.mean())
    gapC.append(s.makespan / (EV.tau_pre + (plan.y_R[rm] * r).sum() / 8) - 1)

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))

axA.axhline(certified / kW, color="k", ls="--", lw=1, label="certified $S^\\star$")
for d, c in DISC.items():
    axA.plot(Dgrid, A[d] / kW, color=c, lw=2, label=d)
axA.set(xlabel="deadline $D$ (s)", ylabel="realized shed (kW)",
        title="A · grid relief: order-blindness wastes shed")
axA.legend(loc="lower right", fontsize=8)

axB.axvspan(4, 16, color="tab:green", alpha=0.10, label="physical pool $W\\in\\{4,8,16\\}$")
axB.axvline(Wstar, color="0.5", ls=":", lw=1)
axB.text(Wstar, gapB.max() * 95, f"  $W^*$≈{Wstar:.0f}\n  (unphysical)", fontsize=7, va="top", color="0.4")
axB.plot(Wgrid, gapB * 100, "o-", color="tab:red", lw=2)
axB.set(xscale="log", xlabel="rebuild servers $W$", ylabel="LP prefill-row over-promise (%)",
        title="B · packing gap: LP over-promises even in the physical pool")
axB.legend(loc="upper left", fontsize=8)

axC.plot(covs, np.array(gapC) * 100, "o-", color="tab:purple", lw=2)
axC.set(xlabel="CoV of prefill time $T/\\rho_{dest}(T)$", ylabel="packing gap at $W$=8 (%)",
        title="C · tail heaviness worsens the gap at physical $W$=8")

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/simulate_validation.{ext}", dpi=150)

# --- diagnostics: the informative null + the two gaps ---
sf = simulate(POP, POOL, IMP, PLAN, EV, MOVE, mode="sf", discipline="johnson")
ct = simulate(POP, POOL, IMP, PLAN, EV, MOVE, mode="cutthrough", discipline="johnson")
print(f"regime={IMP.regime}  jobs={len(POP)}  transfer_frac={PLAN.y_S.sum()/PLAN.y.sum():.2f}  certified={certified/kW:.1f}kW")
split = (PLAN.y_R > 1e-9) & (PLAN.y_S > 1e-9)  # a split job needs BOTH limbs ≤ D to reconstruct
print(f"split jobs (both limbs must land by D): {int(split.sum())}  "
      f"power share {(bind_dp(IMP)*PLAN.y)[split].sum()/certified:.1%}  "
      f"— any realized↔reconstruction gap from these is split-penalty, not queueing")
print(f"NULL pipeline-fill: S&F vs cut-through makespan spread = {abs(ct.makespan-sf.makespan)/sf.makespan:.1e} "
      f"(≈0: no two comparable sequential stages)")
tight = EV.tau_src + 0.3 * egress
gr = {d: simulate(POP, POOL, IMP, PLAN, replace(EV, D=tight), MOVE, discipline=d).realized_shed / kW for d in DISC}
print(f"@D={tight:.0f}s realized kW: " + "  ".join(f"{d}={v:.1f}" for d, v in gr.items())
      + f"  (certified {certified/kW:.1f}; LPT order-blindness costs {gr['pd']-gr['lpt']:.1f}kW vs PD)")
g = dict(zip(Wgrid.tolist(), gapB))
print(f"prefill packing gap (link slacked, Johnson): physical pool W=4:{g[4]:+.0%} W=8:{g[8]:+.0%} W=16:{g[16]:+.0%}"
      f"  | W*≈{Wstar:.0f} (unphysical) so this is packing loss, not the max-job floor"
      f"  | beyond pool W=32:{g[32]:+.0%} W=64:{g[64]:+.0%}")
print(f"  Panel C (W=8): CoV {covs[0]:.2f}→{covs[-1]:.2f} ⇒ gap {gapC[0]:.0%}→{gapC[-1]:.0%}  (tail-driven, not W/W*-collapsible)")
