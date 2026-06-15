"""Sensitivity sweeps (T7; formulation.md §"Sensitivity claims").

Sweep three uncertain parameters — operating utilization, compute efficiency
(MFU), and the price ratio between the expected and guaranteed prices — and show
the two halves of the claim. WHICH jobs to move is unchanged: each sweep leaves
the job ordering essentially identical (selection is robust). HOW MUCH power can
be cut moves smoothly with every parameter (the absolute amount is sensitive).

Load-bound pool only: the three parameters enter a job's score solely in the load
regime (in the memory regime the score is the memory price times context, with
none of the three present), so this is where the invariance is a real result and
not a triviality.

Budgets are left slack so the largest feasible cut is set by the source pool's
price-times-load, not by a destination limit — the same isolation the certify
experiment uses. The deadline is therefore non-binding here (short-context replay
ships only a few KB per job), so it is not a margin axis in this regime; omitted.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from dispatch import Event, bind_dp, solve
from impact import Movement, compute
from instance import Workload, generate
from power import PoolPower

LOAD_POOL = replace(PoolPower(), mean_context_tokens=3378)  # short context → load-bound
LOAD_WL = replace(Workload(), t_mix=((1.0, 8.0, 0.5),))
N_NODES, SEEDS, kW = 4, range(8), 1e3
# Slack budgets: the ceiling is the population's price×load, not a destination cap.
SLACK_E = Event(D=1e9, W=10**7, dest_nodes=10**7)
SLACK_M = replace(Movement(), lambda_src=1e18, mu_in=1e18)


def evaluate(pool, wl, seed):
    """One config × seed → the two job orderings (over movable jobs) and the
    largest feasible power reduction."""
    pop = generate(pool, wl, n_nodes=N_NODES, seed=seed)
    imp = compute(pop, pool)
    assert imp.regime == "load", f"expected load-bound pool, got {imp.regime}"
    dp = bind_dp(imp)
    mv = dp > 0
    by_power = imp.dp_expected[mv]
    by_downtime = np.minimum(imp.c_replay, imp.c_transfer)[mv] / dp[mv]
    ceil = solve(pop, pool, imp, 2 * dp.sum(), SLACK_E, SLACK_M).shed_guaranteed
    return by_power, by_downtime, ceil


def sweep(values, center, build):
    """build(v)->(pool,wl). Per (value, seed): agreement of each ordering with the
    same seed's center ordering, plus the largest feasible reduction (watts)."""
    agree_power = np.zeros((len(values), len(SEEDS)))
    agree_downtime = np.zeros((len(values), len(SEEDS)))
    cut = np.zeros((len(values), len(SEEDS)))
    for j, sd in enumerate(SEEDS):
        p0, d0, _ = evaluate(*build(center), sd)
        for i, v in enumerate(values):
            p, d, k = evaluate(*build(v), sd)
            agree_power[i, j] = spearmanr(p0, p).correlation
            agree_downtime[i, j] = spearmanr(d0, d).correlation
            cut[i, j] = k
    return agree_power, agree_downtime, cut


AXES = [
    ("operating utilization", np.linspace(0.55, 0.90, 11), 0.80,
     lambda v: (replace(LOAD_POOL, rho_star=v), LOAD_WL), "tab:blue",
     "Higher utilization, smaller reduction"),
    ("compute efficiency (fraction of peak)", np.linspace(0.30, 0.50, 11), 0.35,
     lambda v: (LOAD_POOL, replace(LOAD_WL, mfu=v)), "tab:orange",
     "Higher efficiency, smaller reduction"),
    ("price ratio (expected ÷ guaranteed)", np.linspace(17, 58, 11), 30.0,
     lambda v: (replace(LOAD_POOL, bracket_ratio=v), LOAD_WL), "tab:green",
     "Wider price ratio, smaller reduction"),
]

results = [sweep(vals, ctr, bld) for _, vals, ctr, bld, _, _ in AXES]
center_cut = np.mean([evaluate(LOAD_POOL, LOAD_WL, sd)[2] for sd in SEEDS]) / kW
request = 0.5 * center_cut  # an example grid request, for the reference line

fig, ax = plt.subplot_mosaic(
    [["order", "order", "order"], ["util", "mfu", "bracket"]], figsize=(13, 8)
)

# --- top: the job ordering is unchanged by any sweep (both ways of ordering) ---
x = np.linspace(0, 1, 11)
for (name, _, _, _, color, _), (ap, ad, _) in zip(AXES, results):
    ax["order"].plot(x, ap.mean(1), color=color, lw=2.2, label=name)
    ax["order"].plot(x, ad.mean(1), color=color, lw=2.2, ls="--")
floor = min(min(r[0].min(), r[1].min()) for r in results)
ax["order"].set_ylim(min(floor, 0.999) - 0.0003, 1.0004)
ax["order"].set(
    xlabel="position within each parameter's swept range (low → high)",
    ylabel="agreement with the baseline ordering\n(1.0 = identical jobs in identical order)",
    title="Which jobs to move: the ordering barely changes across all three sweeps",
)
params = ax["order"].legend(loc="lower left", fontsize=9, title="swept parameter")
ax["order"].add_artist(params)
ax["order"].legend(
    [Line2D([0], [0], color="0.3", lw=2.2), Line2D([0], [0], color="0.3", lw=2.2, ls="--")],
    ["ordered by power freed", "ordered by downtime per watt"],
    loc="lower right", fontsize=9,
)

# --- bottom: the largest feasible reduction moves smoothly with each parameter ---
for key, (name, vals, _, _, color, title), (_, _, cut) in zip(
    ("util", "mfu", "bracket"), AXES, results
):
    ax[key].fill_between(vals, cut.min(1) / kW, cut.max(1) / kW, color=color, alpha=0.2)
    ax[key].plot(vals, cut.mean(1) / kW, color=color, lw=2.4)
    ax[key].axhline(request, color="0.4", ls=":", lw=1.3, label="an example grid request")
    ax[key].set(xlabel=name, title=title, ylim=(0, cut.max() / kW * 1.12))
    ax[key].legend(loc="lower left", fontsize=8)
ax["util"].set_ylabel("largest guaranteed power reduction (kW)")

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/sensitivity_sweeps.{ext}", dpi=150)

print(f"center largest guaranteed reduction ≈ {center_cut:.0f} kW "
      f"(example grid request drawn at {request:.0f} kW)")
for (name, _, _, _, _, _), (ap, ad, cut) in zip(AXES, results):
    m = cut.mean(1) / kW
    span = (m.max() - m.min()) / m.max() * 100
    direction = "falls" if m[-1] < m[0] else "rises"
    print(f"{name:38s}  ordering agreement ≥ {min(ap.min(), ad.min()):.4f}  |  "
          f"reduction {direction} {span:.0f}% from one end of the range to the other")
print("deadline omitted: non-binding in the load regime "
      "(short-context replay ships only a few KB per job)")
