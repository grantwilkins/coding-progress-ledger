"""Claim:
The fixed-deadline node-knee target sweep replays the original 4-node dispatch
setup while varying only the requested modeled node-expected power.

Plausible wrong implementations:
- Sweep the old active-floor certificate instead of modeled node-expected power.
- Change source size, deadline, or population across methods in one scenario.
- Keep stale additive/node-knee exploration methods in the narrowed comparison.
- Count disruption cost for methods that miss the modeled node target.
- Use a submaximal target when measuring max-request achieved shed by deadline.
- Treat random jobs as a single draw instead of reporting repeated draws with bounds.
- Use linear deadline spacing or a linear deadline axis for the deadline plot.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_target_sweep import (
    COLORS,
    DEADLINES,
    EVENT,
    N_NODES,
    RANDOM_SEEDS,
    _series,
    plot_rows,
    run_deadline_sweep,
    run_sweep,
)

RUNS_PER_CONFIG = len(COLORS) + len(RANDOM_SEEDS) - 1


def test_target_sweep_fixes_setup_and_varies_modeled_power_request():
    rows = run_sweep(workloads=("agentic_tool_loop",), target_fracs=np.array([0.10, 0.40]))
    assert len(rows) == 2 * RUNS_PER_CONFIG
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
        assert sorted(r["replicate"] for r in rs if r["method"] == "random") == list(RANDOM_SEEDS)
        assert all(r["replicate"] == 0 for r in rs if r["method"] != "random")


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
    assert [r["replicate"] for r in out] == [0, 0, 0, 0]


def test_deadline_sweep_uses_requested_deadlines_and_full_power_target():
    deadlines = np.array([1.0, 300.0])
    rows = run_deadline_sweep(workloads=("agentic_tool_loop",), deadlines=deadlines)

    assert len(rows) == len(deadlines) * RUNS_PER_CONFIG
    assert {r["method"] for r in rows} == set(COLORS)
    assert {r["deadline_s"] for r in rows} == set(deadlines)
    for r in rows:
        assert r["source_nodes"] == N_NODES
        assert r["target_frac"] == 1.0
        assert np.isclose(r["target_kw"], r["full_node_kw"])
        assert 0.0 <= r["node_kw"] <= r["full_node_kw"]
        assert (r["hit"] and np.isfinite(r["cost_s"])) or ((not r["hit"]) and np.isnan(r["cost_s"]))


def test_deadline_grid_is_log_tick_spaced_to_300s():
    assert np.array_equal(DEADLINES, np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0]))


def test_series_aggregates_random_replicates_for_error_bounds():
    rows = [
        {"target_kw": 1.0, "node_kw": 2.0, "cost_s": 5.0},
        {"target_kw": 1.0, "node_kw": 4.0, "cost_s": 9.0},
        {"target_kw": 2.0, "node_kw": 8.0, "cost_s": 11.0},
    ]
    x, y, lo, hi, n = _series(rows, "node_kw", "cost_s", "target_kw")

    assert np.allclose(x, [3.0, 8.0])
    assert np.allclose(y, [7.0, 11.0])
    assert np.allclose(lo, [5.0, 11.0])
    assert np.allclose(hi, [9.0, 11.0])
    assert n == [2, 1]
