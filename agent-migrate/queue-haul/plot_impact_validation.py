"""Validation plot for impact.py (formulation.md §Per-job impact / §Dispatch).

Left: replay rebuild cost vs context T. Because ρ_dest(T) is a *function* (flat ≈63k
below T*≈29k, ~1/T above), the cost is near-flat-rate for short jobs then steepens —
diverging from the constant-F line a fixed prefill rate would predict. Center: two-price
vs single-price expected ΔP. Agentic (prefill-skewed) falls *below* the diagonal (prefill
discount); chat and reasoning (decode-skewed) rise *above* it (decode premium) — both
deviate, in opposite directions; only a phase-balanced job sits on the line. Right: the
memory-regime score μ·T/E[T] spreads widely around μ because context is tail-heavy; the
annotation is the measured load-vs-memory rank correlation (near-zero ⇒ the regimes reorder
shed priorities, the T8 finding).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from impact import Movement, compute
from instance import JobPopulation, Workload, _draw, generate
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_dest

POOL, MOVE = PoolPower(), Movement()
T_STAR = 29_000


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))

# --- Panel A: replay cost, ρ_dest(T) function vs constant-F ---
T = np.geomspace(1e3, 5e5, 200)
syn = JobPopulation(
    np.full(len(T), "agentic"),
    np.full(len(T), "active"),
    np.zeros(len(T), bool),
    T,
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
axA.plot(T / 1e3, imp.c_replay, color="tab:red", lw=2, label="ρ_dest(T) (actual)")
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
    title="Replay rebuild: flat-rate → 1/ρ_dest steepening",
)
axA.legend(loc="upper left", fontsize=8)

# --- Panel B: two-price vs single-price expected ΔP, by class ---
pop = _draw(np.random.default_rng(0), 40000, Workload(), "bf16")
imp_pop = compute(pop, POOL)
act = pop.state == "active"
groups = [
    ("agentic", act & (pop.job_type == "agentic") & ~pop.is_reasoning, "tab:red"),
    ("reasoning", act & pop.is_reasoning, "tab:purple"),
    ("chat", act & (pop.job_type == "chat"), "tab:blue"),
]
for name, sel, color in groups:
    axB.scatter(
        imp_pop.dp_expected_single[sel],
        imp_pop.dp_expected[sel],
        s=7,
        alpha=0.35,
        color=color,
        label=name,
    )
lim = [0, max(imp_pop.dp_expected_single[act].max(), 1)]
axB.plot(lim, lim, "k-", lw=1, label="two-price = single")
axB.set(
    xlabel="single-price $\\bar p\\,\\ell_j$ (W)",
    ylabel="two-price $\\Delta P_j$ (W)",
    xlim=lim,
    ylim=lim,
    title="Two-price vs single: opposite-sign skew by class",
)
axB.legend(loc="upper left", fontsize=8)

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
print(
    f"two-price gap: agentic={np.mean(imp_pop.dp_expected[groups[0][1]] - imp_pop.dp_expected_single[groups[0][1]]):+.0f} W  "
    f"chat={np.mean(imp_pop.dp_expected[groups[2][1]] - imp_pop.dp_expected_single[groups[2][1]]):+.1f} W"
)
print(
    f"load↔memory Spearman (active) = {rho:+.3f}  memory-score CoV = {dm[act].std()/dm[act].mean():.2f}"
)
