"""Claim:
The agentic DES target sweep varies requested node-expected shed on the x-axis
and reports disruption per requested kW for solver-selected, egress-realized,
and rebuild-realized outcomes.

Plausible wrong implementations:
- Use active-floor watts instead of full node-expected watts for the target.
- Report cost per achieved kW as the plotted requested-kW intensity.
- Count a DES miss as a finite requested-kW disruption point.
- Regenerate a non-agentic workload or change the fixed deadline across methods.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_agentic_des_sweep import COLORS, EVENT, ORDERING, SESSION_CLASS, STAGES, run_sweep
from plot_node_knee_target_sweep import N_NODES


def test_agentic_des_sweep_reports_requested_shed_intensity():
    rows = run_sweep(target_fracs=np.array([0.10, 0.40]))
    assert len(rows) == 2 * len(COLORS)
    assert {r["method"] for r in rows} == set(COLORS)
    assert {r["session_class"] for r in rows} == {SESSION_CLASS}
    assert {r["source_nodes"] for r in rows} == {N_NODES}
    assert {r["deadline_s"] for r in rows} == {EVENT.D}
    assert {r["ordering"] for r in rows} == {ORDERING}

    for r in rows:
        assert np.isclose(r["target_kw"], r["target_frac"] * r["full_node_kw"])
        for stage in STAGES:
            hit = r[f"{stage}_hit"]
            requested = r[f"{stage}_requested_s_per_kw"]
            if hit:
                assert np.isclose(requested, r["cost_s"] / r["target_kw"])
            else:
                assert np.isnan(requested)
            assert r[f"{stage}_delivered_s_per_kw"] >= 0 or np.isnan(r[f"{stage}_delivered_s_per_kw"])

        assert r["egress_realized_node_kw"] <= r["selected_node_kw"] + 1e-9
        assert r["rebuild_realized_node_kw"] <= r["egress_realized_node_kw"] + 1e-9
