"""Claim:
The node-knee scale/workload sweep compares every scalable method on the same
population, source placement, scaled destination event, deadline, and modeled
node-expected target.

Plausible wrong implementations:
- Use a different target or population per method within one scenario.
- Forget one of the narrowed scalable baselines.
- Scale the source nodes but leave destination capacity fixed by accident.
- Report disruption cost for misses as if they hit the modeled target.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_scale_workload_sweep import COLORS, run_sweep


def test_scale_workload_sweep_uses_common_scenarios_for_all_methods():
    rows = run_sweep(workloads=("ordinary_chat",), nodes=(1,), deadlines=(10.0, 30.0), target_fracs=(0.25, 0.45))
    assert len(rows) == 1 * 1 * 2 * 2 * len(COLORS)
    assert {r["method"] for r in rows} == set(COLORS)

    scenarios = {}
    for r in rows:
        key = (r["session_class"], r["source_nodes"], r["deadline_s"], r["target_frac"])
        scenarios.setdefault(key, []).append(r)

    for rs in scenarios.values():
        assert {r["method"] for r in rs} == set(COLORS)
        assert len({r["jobs"] for r in rs}) == 1
        assert len({r["target_kw"] for r in rs}) == 1
        assert len({r["full_node_kw"] for r in rs}) == 1
        assert all(r["dest_nodes"] == 12 * r["source_nodes"] for r in rs)
        assert all(r["workers"] == 4 * r["source_nodes"] for r in rs)
        assert all((r["hit"] and np.isfinite(r["cost_s"])) or ((not r["hit"]) and np.isnan(r["cost_s"])) for r in rs)


def test_scale_workload_sweep_targets_are_fractional_node_expected_power():
    rows = run_sweep(workloads=("agentic_tool_loop",), nodes=(1,), deadlines=(30.0,), target_fracs=(0.25, 0.65))
    for r in rows:
        assert np.isclose(r["target_kw"], r["target_frac"] * r["full_node_kw"])
