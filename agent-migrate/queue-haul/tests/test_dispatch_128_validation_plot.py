"""Claim:
The 128-node validation plot compares greedy, MILP, and LP on a common feasible
target sweep, so reported method gaps are optimization gaps rather than ceiling
mismatches.

Plausible wrong implementations:
- Sweep to the LP ceiling and make the integer policies infeasible near the end.
- Accidentally solve MILP as another LP or omit the integer optimum.
- Compare raw cost instead of disruption intensity per requested kW.
- Let greedy look better than the MILP optimum for the same target.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_dispatch_128_validation import fixed_event, norm_cost, population, run_case, scaled_event


def test_128_validation_uses_common_feasible_integer_ceiling():
    pool, pop, imp = population()
    for event in (fixed_event(), scaled_event()):
        S, plans, ceilings = run_case(pool, pop, imp, event, fracs=np.array([0.2, 0.7]))
        assert S[-1] < min(ceilings.values())
        assert set(plans) == {"integer greedy", "MILP", "LP"}
        assert all(p.feasible for xs in plans.values() for p in xs)

        costs = norm_cost(plans, S)
        assert np.all(costs["LP"] <= costs["MILP"] + 1e-6)
        assert np.all(costs["MILP"] <= costs["integer greedy"] + 1e-6)
