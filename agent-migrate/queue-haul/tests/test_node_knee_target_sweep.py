"""Claim:
The fixed-deadline node-knee target sweep replays the original 4-node dispatch
setup while varying only the requested modeled node-expected power.

Plausible wrong implementations:
- Sweep the old active-floor certificate instead of modeled node-expected power.
- Change source size, deadline, or population across methods in one scenario.
- Keep stale additive/node-knee exploration methods in the narrowed comparison.
- Count disruption cost for methods that miss the modeled node target.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_target_sweep import COLORS, EVENT, N_NODES, plot_rows, run_sweep


def test_target_sweep_fixes_setup_and_varies_modeled_power_request():
    rows = run_sweep(workloads=("agentic_tool_loop",), target_fracs=np.array([0.10, 0.40]))
    assert len(rows) == 2 * len(COLORS)
    assert {r["method"] for r in rows} == set(COLORS)
    assert "additive LP" not in {r["method"] for r in rows}

    by_target = {}
    for r in rows:
        by_target.setdefault(r["target_frac"], []).append(r)
        assert r["source_nodes"] == N_NODES
        assert r["deadline_s"] == EVENT.D
        assert np.isclose(r["target_kw"], r["target_frac"] * r["full_node_kw"])
        assert (r["hit"] and np.isfinite(r["cost_s"])) or ((not r["hit"]) and np.isnan(r["cost_s"]))
        if r["hit"]:
            assert np.isclose(r["requested_intensity_s_per_kw"], r["cost_s"] / r["target_kw"])

    for rs in by_target.values():
        assert {r["method"] for r in rs} == set(COLORS)
        assert len({r["jobs"] for r in rs}) == 1
        assert len({r["target_kw"] for r in rs}) == 1
        assert len({r["full_node_kw"] for r in rs}) == 1


def test_plot_rows_removes_additive_and_renames_methods():
    rows = [
        {"method": "additive LP", "hit": "False"},
        {"method": "active-knee LP relaxation", "hit": "True"},
        {"method": "active-knee MILP", "hit": "True"},
        {"method": "live greedy", "hit": "True"},
        {"method": "random jobs", "hit": "True"},
    ]
    for r in rows:
        r.update({k: "1" for k in (
            "source_nodes", "jobs", "deadline_s", "target_frac", "target_kw", "full_node_kw",
            "node_kw", "achieved_over_target", "active_kw", "cost_s",
            "intensity_s_per_kw", "requested_intensity_s_per_kw",
        )})

    out = plot_rows(rows)

    assert [r["method"] for r in out] == ["LP relaxation", "MILP", "greedy", "random"]
