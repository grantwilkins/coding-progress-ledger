"""Validation plot for power.py (formulation.md §Pool power model).

Left: ramp-plateau node curve — the dispatch never evaluates it; the plot checks
that the prices power.py *does* export are consistent with it: the curve passes
through (ρ*, P_busy), so the origin secant has slope p̄ and the plateau has slope
s_plat. Right: regime test N = max(L/ρ*, S_held/S_node) with the BF16/FP8 crossover.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from power import CAP_FP8_GB, PoolPower

POWER_KNEE = 0.10  # §2 center, plot-only
LATENCY_KNEE = 0.85  # §2 center, plot-only

p = PoolPower()
pi_knee = p.p_busy_w - p.s_plat * (p.rho_star - POWER_KNEE)  # anchors pi(rho*) = P_busy


def pi(ell):
    ramp = p.p_idle_w + (pi_knee - p.p_idle_w) * ell / POWER_KNEE
    return np.where(ell <= POWER_KNEE, ramp, pi_knee + p.s_plat * (ell - POWER_KNEE))


assert np.isclose(pi(p.rho_star), p.p_busy_w)
assert np.isclose(pi(p.rho_star) / p.rho_star, p.p_bar)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ell = np.linspace(0, 1, 500)
ax1.plot(ell, pi(ell) / 1e3, "k", lw=2, label="node curve $\\pi(\\ell)$")
ax1.plot([0, p.rho_star], [0, p.p_busy_w / 1e3], "--", color="tab:red",
         label=f"amortized $\\bar p$ = {p.p_bar / 1e3:.1f} kW/unit (origin secant)")
ax1.plot([POWER_KNEE, 1], [pi_knee / 1e3, (pi_knee + p.s_plat * (1 - POWER_KNEE)) / 1e3],
         ":", color="tab:blue", lw=3,
         label=f"guaranteed $s_{{plat}}$ = {p.s_plat:.0f} W/unit (plateau slope)")
for x, name in [(POWER_KNEE, "power knee"), (p.rho_star, "$\\rho^\\star$"),
                (LATENCY_KNEE, "latency knee")]:
    ax1.axvline(x, color="gray", lw=0.8, alpha=0.6)
    ax1.text(x, 11.3, name, rotation=90, ha="right", va="top", fontsize=8, color="gray")
ax1.plot(p.rho_star, p.p_busy_w / 1e3, "o", color="tab:red")
ax1.set(xlabel="node load $\\ell$", ylabel="node power (kW)", xlim=(0, 1), ylim=(0, 11.5),
        title=f"Ramp–plateau curve and the two prices (bracket {p.bracket_ratio:.0f}$\\times$)")
ax1.legend(loc="lower right", fontsize=8)

fp8 = replace(p, cap_gb=CAP_FP8_GB)
L = 8.0
s_held = np.linspace(0, 700, 500)
for pool, name, color in [(p, "BF16", "tab:blue"), (fp8, "FP8", "tab:orange")]:
    cross = (L / pool.rho_star) * pool.s_node
    assert not pool.memory_bound(L, 0.999 * cross) and pool.memory_bound(L, 1.001 * cross)
    ax2.plot(s_held, [pool.node_count(L, s) for s in s_held], color=color,
             label=f"{name}: $\\mu$ = {pool.mu:.0f} W/session, crossover {cross:.0f}")
    ax2.axvline(cross, color=color, lw=0.8, ls="--", alpha=0.6)
ax2.axhline(L / p.rho_star, color="gray", lw=0.8, alpha=0.6)
ax2.text(5, L / p.rho_star + 0.15, "load-bound: $N = L/\\rho^\\star$", fontsize=8, color="gray")
ax2.set(xlabel="held sessions $S_{held}$", ylabel="nodes $N$",
        title=f"Regime test $N = \\max(L/\\rho^\\star,\\ S_{{held}}/S_{{node}})$ at $L$ = {L:.0f}")
ax2.legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/power_validation.{ext}", dpi=150)

print(f"p_bar={p.p_bar:.0f} W/unit  s_plat={p.s_plat:.0f} W/unit  "
      f"mu(BF16)={p.mu:.1f}  mu(FP8)={fp8.mu:.1f} W/session")
print("curve passes (rho*, P_busy); secant slope == p_bar; memory_bound flips at crossover: OK")
