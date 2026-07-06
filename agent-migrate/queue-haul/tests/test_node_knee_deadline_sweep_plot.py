"""Claim:
The node-knee deadline sweep compares old additive dispatch against node-expected
methods on the same population, placement, deadline, and target.

Plausible wrong implementations:
- Plot active-floor watts as if they were node-expected watts.
- Count disruption cost even when the node-expected target is missed.
- Regenerate placement or population per method/deadline.
- Keep stale exploration methods in the narrowed dispatch graph.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_deadline_sweep import run_sweep

METHODS = {"additive LP", "active-knee LP relaxation", "active-knee MILP", "live greedy", "random jobs"}


def test_node_knee_deadline_sweep_uses_node_expected_target():
    # Target basis is full node-expected removable power. The additive LP can satisfy its
    # certified target while missing the source-node curve; active_kw here is only the
    # conservative plateau floor, so it should not be used as the target proof.
    _, target_kw, rows = run_sweep(deadlines=np.array([300.0]))
    by_method = {r["method"]: r for r in rows}
    assert set(by_method) == METHODS
    additive = by_method["additive LP"]
    active = by_method["active-knee LP relaxation"]

    for r in rows:
        assert r["target_basis"] == "full_node_expected"
        assert np.isclose(r["target_kw"], target_kw)
        assert np.isclose(r["target_kw"], r["target_frac"] * r["full_node_kw"])

    assert additive["node_kw"] < target_kw
    assert additive["active_kw"] < target_kw
    assert not additive["hit"] and np.isnan(additive["cost_s"])

    assert active["node_kw"] >= target_kw
    assert active["hit"] and np.isfinite(active["cost_s"])
