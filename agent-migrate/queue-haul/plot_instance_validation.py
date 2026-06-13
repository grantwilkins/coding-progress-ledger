"""Validation plot for instance.py (formulation.md §Job model).

Left: the generated population in (load ℓ, memory m) space. Active jobs spread along
ℓ>0 and sit in HBM; idle/cold pin to ℓ=0 yet still carry KV — the two axes the dispatch
trades off. Right: the regime walk. Sweeping context short→long, the memory term
S_held/S_node is pinned at α·N (we always pack α× capacity), while the *measured* load
term L/ρ* falls, so the binding constraint crosses load→memory. The crossover is an
OUTPUT (L is measured from the drawn jobs), not something packed into the setup.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from instance import Workload, generate
from power import CAP_FP8_GB, GB, PoolPower

N_NODES = 32
ALPHA = Workload().alpha
MEM_TERM = ALPHA * N_NODES  # S_held/S_node = α·N, precision-independent

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# --- Panel A: population structure at BF16 center ---
pop = generate(PoolPower(), n_nodes=N_NODES)
for st, color in [("cold", "0.6"), ("idle", "tab:orange"), ("active", "tab:red")]:
    sel = pop.state == st
    axA.scatter(pop.ell[sel], pop.m[sel] / GB, s=9, alpha=0.45, color=color,
                label=f"{st} ({sel.sum()})")
axA.set(xlabel="job load $\\ell_j$", ylabel="KV footprint $m_j$ (GB)", yscale="log",
        title=f"Population in (load, memory): N={len(pop)}, E[T]={pop.T.mean():,.0f}")
axA.legend(loc="lower right", fontsize=8, title=f"BF16, α={ALPHA}")

# --- Panel B: regime walk over context ---
ET, SIG = np.geomspace(3e3, 2e5, 25), 0.9
axB.axhline(MEM_TERM, color="k", lw=1.3, ls="--",
            label=f"memory-bound  $S_{{held}}/S_{{node}}=\\alpha N={MEM_TERM:.0f}$")
for base, name, color in [(PoolPower(), "BF16", "tab:blue"),
                          (replace(PoolPower(), cap_gb=CAP_FP8_GB), "FP8", "tab:orange")]:
    load = []
    for et in ET:
        wl = replace(Workload(), t_mix=((1.0, np.log(et) - SIG**2 / 2, SIG),))
        p = replace(base, mean_context_tokens=et)
        load.append(generate(p, wl, n_nodes=N_NODES).ell.sum() / p.rho_star)
    load = np.array(load)
    axB.plot(ET / 1e3, load, color=color, lw=2, label=f"{name} load-bound $L/\\rho^\\star$")
    cross = ET[np.argmin(np.abs(load - MEM_TERM))] / 1e3
    axB.plot(cross, MEM_TERM, "o", color=color)
    axB.annotate(f"{name} crossover\n{cross:.0f}k", (cross, MEM_TERM), fontsize=7,
                 color=color, xytext=(cross, MEM_TERM * 2.4), ha="center")
axB.axvline(65.8, color="gray", lw=0.8, alpha=0.6)
axB.text(65.8, MEM_TERM / 2.2, "center E[T]", rotation=90, fontsize=8, color="gray", ha="right")
axB.set(xlabel="context scale E[T] (k tokens)", ylabel="nodes (binding term)",
        xscale="log", yscale="log", title="Regime walk: load→memory crossover (measured)")
axB.legend(loc="upper right", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/instance_validation.{ext}", dpi=150)

print(f"N={len(pop)}  E[T]={pop.T.mean():,.0f}  mean ℓ(active)="
      f"{pop.ell[pop.state == 'active'].mean():.4f}  "
      f"regime={'memory' if PoolPower().memory_bound(pop.ell.sum(), len(pop)) else 'load'}-bound")
print(f"states: " + "  ".join(f"{s}={(pop.state == s).sum()}" for s in ('active', 'idle', 'cold')))
