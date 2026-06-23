"""Validation plot for impact.py (formulation.md §Per-job impact / §Dispatch).

Left: replay rebuild cost vs context T. Full replay uses the average rate ρ_dest(T/2),
so the cost is near-flat-rate for short jobs then steepens away from a constant-F line.
Center: the code currently reports the single-price future proxy p̄·ℓ for every class;
raw f/g are stored, but token-energy work power is not calibrated yet. Right: the
memory-regime score μ·T/E[T] spreads widely around μ because context is tail-heavy.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from impact import Movement, compute

from instance import JobPopulation, Workload, _draw, _mean_T, class_workload
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_dest

POOL, MOVE = PoolPower(), Movement()
T_STAR = 29_000


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))

# --- Panel A: replay cost, average full-context replay rate vs constant-F ---
T = np.geomspace(1e3, 5e5, 200)
syn = JobPopulation(
    np.full(len(T), "agentic"),
    np.full(len(T), "agentic_tool_loop"),
    np.full(len(T), "active"),
    np.zeros(len(T), bool),
    T,
    np.zeros(len(T)),
    np.zeros(len(T)),
    np.zeros(len(T)),
    np.zeros(len(T)),
    np.zeros(len(T)),
    np.ones(len(T), bool),
    np.zeros(len(T)),
    np.zeros(len(T)),
    ETA_BYTES_PER_TOK * T,
    "bf16",
    0.35,
)
imp = compute(syn, POOL, MOVE)
ship = BETA_BYTES_PER_TOK * T / MOVE.lambda_src
const_f = ship + (1 + congestion(MOVE.dest_prefill_util)) * T / rho_dest(
    0.0
)  # flat 63k rate
axA.plot(T / 1e3, imp.c_replay, color="tab:red", lw=2, label="ρ_dest(T/2) replay")
axA.plot(T / 1e3, const_f, color="0.5", lw=1.6, ls="--", label="constant-F (flat 63k)")
axA.axvline(T_STAR / 1e3, color="gray", lw=0.8, alpha=0.7)
axA.text(
    T_STAR / 1e3,
    imp.c_replay.min() * 1.4,
    "T*≈29k",
    rotation=90,
    fontsize=8,
    color="gray",
)
axA.set(
    xlabel="context T (k tokens)",
    ylabel="replay cost $c_j(R)$ (s)",
    xscale="log",
    yscale="log",
    title="Replay rebuild: flat-rate → full-context average",
)
axA.legend(loc="upper left", fontsize=8)

classes = [
    ("ordinary chat", "ordinary_chat", "tab:blue"),
    ("long chat/code", "long_chat_code", "tab:green"),
    ("reasoning chat", "reasoning_chat", "tab:purple"),
    ("agentic loop", "agentic_tool_loop", "tab:red"),
]

# --- Panel B: current future-impact proxy by class ---
proxy_max = 0.0
for name, cls, color in classes:
    wl = class_workload(cls, state_mix=(1.0, 0.0, 0.0))
    pool = replace(POOL, mean_context_tokens=_mean_T(wl))
    pop = _draw(np.random.default_rng(0), 3000, wl, "bf16")
    imp_pop = compute(pop, pool)
    proxy_max = max(proxy_max, imp_pop.dp_expected_single.max())
    axB.scatter(
        imp_pop.dp_expected_single,
        imp_pop.dp_expected,
        s=7,
        alpha=0.35,
        color=color,
        label=name,
    )
lim = [0, max(proxy_max, 1)]
axB.plot(lim, lim, "k-", lw=1, label="future proxy = p̄·ℓ")
axB.set(
    xlabel="single-price $\\bar p\\,\\ell_j$ (W)",
    ylabel="reported future impact (W)",
    xlim=lim,
    ylim=lim,
    title="Current code reports the load-based future proxy",
)
axB.legend(loc="upper left", fontsize=8)

pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
imp_pop = compute(pop, POOL)
act = pop.state == "active"

# --- Panel C: memory score spread around μ ---
dm = imp_pop.dp_memory
axC.hist(dm[act], bins=60, color="tab:green", alpha=0.7)
axC.axvline(
    POOL.mu, color="k", lw=1.4, ls="--", label=f"μ = {POOL.mu:.0f} W (job at E[T])"
)
rho = _spearman(imp_pop.dp_expected[act], dm[act])
axC.set(
    xlabel="memory score $\\mu\\,T_j/E[T]$ (W)",
    ylabel="active jobs",
    title=f"Memory score spreads around μ  (CoV={dm[act].std()/dm[act].mean():.2f})",
)
axC.legend(loc="upper right", fontsize=8, title=f"load↔mem rank ρ = {rho:+.2f}")

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/impact_validation.{ext}", dpi=150)

print(
    f"regime={imp_pop.regime}  μ={POOL.mu:.0f} W  φ_pre={congestion(MOVE.dest_prefill_util):.2f}  "
    f"φ_in={congestion(MOVE.dest_ingest_util):.2f}"
)
print("future proxy: dp_expected == p_bar * ell for every class; raw f/g are stored for later calibration")
print(
    f"load↔memory Spearman (active) = {rho:+.3f}  memory-score CoV = {dm[act].std()/dm[act].mean():.2f}"
)
