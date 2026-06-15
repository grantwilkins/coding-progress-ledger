"""Power reduction vs deadline (companion to T7's deadline finding).

How much power the dispatch can free as the deadline grows from 1 s to 60 s, for
two pools. Short context (small contexts, cheap to move): the deadline never
binds — moves finish almost instantly, so the limit is destination capacity, a
steady-state cap with no deadline in it, and the curve is flat. Long context
(big KV, expensive to move): both move primitives throttle on transfer time, so
a tighter deadline directly caps how much can be shed, and the curve ramps.

Below ~5 s the deadline is shorter than the migration startup latencies
(connection ramp, batch formation, pipeline fill), so no move can complete and
the achievable reduction is zero — the curve sits at the floor until it clears.
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
from power import PoolPower

SHORT_POOL = replace(PoolPower(), mean_context_tokens=3378)
SHORT_WL = replace(Workload(), t_mix=((1.0, 8.0, 0.5),))
LONG_POOL = replace(PoolPower(), mean_context_tokens=130000)
LONG_WL = replace(Workload(), t_mix=((0.5, 10.5, 1.0), (0.5, 12.0, 0.9)))
N_NODES, SEEDS, kW = 4, range(8), 1e3
MOVE = Movement()
# Dense early (startup floor + first rise), out to 300 s so the long-context line plateaus too.
DEADLINES = np.unique(np.concatenate([np.linspace(1, 30, 25), np.linspace(30, 300, 28)]))
STARTUP = max(Event().tau_src, Event().tau_pre, Event().tau_in)  # no move completes below this


def reductions(pool, wl, seed):
    """Largest power reduction (kW) at each deadline; zero below the startup floor."""
    pop = generate(pool, wl, n_nodes=N_NODES, seed=seed)
    imp = compute(pop, pool)
    target = 2 * bind_dp(imp).sum()  # unreachable → solve returns the max-shed ceiling
    return np.array([
        0.0 if d <= STARTUP else solve(pop, pool, imp, target, Event(D=d), MOVE).shed_guaranteed
        for d in DEADLINES
    ]) / kW


def band(pool, wl):
    m = np.array([reductions(pool, wl, s) for s in SEEDS])
    return m.mean(0), m.min(0), m.max(0)


short, long = band(SHORT_POOL, SHORT_WL), band(LONG_POOL, LONG_WL)

fig, ax = plt.subplots(figsize=(8.2, 5))
for (mean, lo, hi), color, label in [
    (long, "tab:red", "long context — limited by transfer time (deadline binds)"),
    (short, "tab:blue", "short context — limited by destination capacity (deadline does not bind)"),
]:
    ax.fill_between(DEADLINES, lo, hi, color=color, alpha=0.15)
    ax.plot(DEADLINES, mean, color=color, lw=2.4, label=label)
ax.axvline(STARTUP, color="0.5", ls=":", lw=1.2)
ax.text(STARTUP + 0.6, ax.get_ylim()[1] * 0.04, "below this, no move\nfinishes in time",
        fontsize=8, color="0.4")
ax.set(xlabel="deadline (seconds)", ylabel="largest power reduction achievable (kW)",
       title="A tighter deadline caps the power reduction — only when the data is large to move")
ax.legend(loc="center right", fontsize=9)

fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/deadline_sweep.{ext}", dpi=150)

for tag, (mean, _, _) in (("short", short), ("long", long)):
    knee = DEADLINES[mean >= 0.99 * mean.max()][0]  # deadline at which it reaches its plateau
    print(f"{tag:5s} context: levels off at {mean.max():5.2f} kW, "
          f"plateauing by a ~{knee:.0f} s deadline")
