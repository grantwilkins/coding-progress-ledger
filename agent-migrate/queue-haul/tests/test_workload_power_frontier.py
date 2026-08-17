"""
Claim:
The workload frontier pairs every Queue-Haul request within each sampled
workload/calibration draw, uses a certified integer phase-load maximum for its
capacity curve, normalizes by that draw's removable power, and weights each
draw-by-factor case once.

Plausible wrong implementations:
- Resample the workload, timing, or power model independently by request.
- Divide attained power by requested power or a global maximum.
- Use the rounded target-following LP plateau as the capacity endpoint.
- Overweight a case because it contributes extra rows or policies.
- Use a different Queue-Haul draw or solver at the shared two-thirds point.
- Plot target-following policies against their own target instead of the
  distribution of maximum safely attainable power.
- Collapse distinct bandwidth states into a shared curve.
- Plot an intermediate joint state or omit one of the five declared display states.
- Filter the raw factorial sweep down to only the five displayed states.
- Maximize the obsolete one-dimensional power load instead of sampled af+bg.
- Let a released constraint have lower capacity than its paired constrained case.
"""

import numpy as np
import pytest

import workload_adaptation_campaign as adaptation
from profiles import ModelProfile
from workload_power_frontier import (
    CAPACITY_SOLVER, DISPLAY_STATES, SOLVERS, capacity_release_audit,
    capacity_summary, planning_request_w, power_summary, request_grid, sweep,
    write_capacity_plot,
)


def test_display_states_match_the_five_declared_action_cases():
    assert DISPLAY_STATES == adaptation.DISPLAY_CASES
    assert len(DISPLAY_STATES) == len(set(DISPLAY_STATES)) == 5


def test_capacity_figure_has_half_column_canvas(tmp_path):
    rows = [{
        "constraint_state": state, "requested_fraction": request,
        "attainment_rate": 1 - request,
    } for state in DISPLAY_STATES for request in (0, 1)]

    write_capacity_plot(rows, tmp_path / "frontier")
    image = adaptation.plt.imread(tmp_path / "frontier.png")

    assert image.shape[:2] == tuple(int(value * adaptation.plot_style.SAVE_DPI)
                                    for value in (2.2, 3.35))


def test_raw_lp_endpoint_explicitly_requests_its_maximum_fallback():
    assert planning_request_w(2 / 3, 42) == 28
    assert planning_request_w(1, 42) == 42 + adaptation.POWER_TOLERANCE_W


def test_capacity_summary_uses_one_maximum_per_paired_draw():
    rows = [{
        "replicate": replicate, "factor_case_id": state,
        "policy": "queue_haul_lp", "requested_fraction": 1,
        "capacity_mip_shed_w": capacity,
        "maximum_attainable_fraction": capacity,
    } for state, _, _ in adaptation.factorial_cases()
            for replicate, capacity in enumerate((.25, .75))]

    summary = capacity_summary(rows, grid=(0, .5, 1))
    none = [row for row in summary if row["constraint_state"] == "none"]

    assert [row["attainment_rate"] for row in none] == [1, .5, 0]
    assert all(row["cases"] == 2 for row in summary)
    conflicting = {**rows[0], "maximum_attainable_fraction": .5}
    with pytest.raises(RuntimeError, match="conflicting"):
        capacity_summary([*rows, conflicting], grid=(0, .5, 1))


def test_capacity_release_closure_retains_a_known_tighter_plan():
    rows = [{
        "replicate": 0, "factor_case_id": state,
        "maximum_removable_w": 100,
        "capacity_mip_shed_w": 60 if state == "bandwidth" else 59,
    } for state, _, _ in adaptation.factorial_cases()]

    audit = capacity_release_audit(rows, close=True)

    none = next(row for row in rows if row["factor_case_id"] == "none")
    assert none["maximum_attainable_shed_w"] == 60
    assert audit["raw_solver_inversions"] == 1

    next(row for row in rows if row["factor_case_id"] == "bandwidth")[
        "capacity_mip_shed_w"
    ] = 61
    with pytest.raises(RuntimeError, match="release tolerance"):
        capacity_release_audit(rows, close=True)


def test_power_summary_weights_cases_once_and_keeps_watts():
    rows = [{
        "case_id": case, "policy": "queue_haul_lp", "requested_fraction": .5,
        "maximum_removable_w": maximum, "safely_attained_shed_w": attained,
    } for case, maximum, attained in (
        ("a", 10, 5), ("b", 20, 10), ("c", 30, 15), ("d", 40, 20),
    )]

    summary = power_summary(rows)[0]
    assert summary["cases"] == 4
    assert summary["maximum_removable_w_median"] == 25
    assert summary["safely_attained_w_median"] == 12.5

    with pytest.raises(RuntimeError, match="weight each case once"):
        power_summary([*rows, rows[0]])


def test_sampled_frontier_is_paired_normalized_and_monotone():
    rows, _ = sweep(samples=1, points=2, seed=3)
    action_rows, _ = adaptation.simulate(samples=1, seed=3)
    fractions = request_grid(2)

    assert len(rows) == 8 * len(SOLVERS) * len(fractions)
    assert len({row["power_bootstrap_index"] for row in rows}) == 1
    assert len({row["timing_fit_sha256"] for row in rows}) == 1
    assert len({row["maximum_removable_w"] for row in rows}) == 1
    assert {row["capacity_solver"] for row in rows} == {CAPACITY_SOLVER}
    assert capacity_release_audit(rows)["violations"] == 0
    assert all(np.isclose(
        row["requested_shed_w"],
        row["requested_fraction"] * row["maximum_removable_w"],
    ) for row in rows)
    profile = ModelProfile.load(adaptation.PROFILE)
    bootstrap = profile.case().phase_power.bootstrap
    assert all(np.allclose(
        (row["phase_p0_w"], row["phase_delta_w"],
         row["phase_a_s_per_prefill_token"],
         row["phase_b_s_per_decode_token"]),
        bootstrap[row["power_bootstrap_index"]],
    ) for row in rows)
    assert all(0 <= row["maximum_attainable_fraction"] <= 1 + 1e-8
               for row in rows)
    assert all(np.isclose(
        row["safely_attained_fraction"],
        row["safely_attained_shed_w"] / row["maximum_removable_w"],
    ) for row in rows)
    for case in {row["case_id"] for row in rows}:
        for policy in SOLVERS:
            attained = [row["safely_attained_shed_w"] for row in rows
                        if row["case_id"] == case and row["policy"] == policy]
            assert all(right >= left for left, right in zip(attained, attained[1:]))
    frontier = {row["factor_case_id"]: row for row in rows
                if row["policy"] == "queue_haul_lp"
                and np.isclose(row["requested_fraction"], 2 / 3)}
    action = {row["case_id"]: row for row in action_rows}
    assert frontier.keys() == action.keys()
    assert all(np.isclose(frontier[case]["requested_shed_w"], row["target_w"])
               and np.isclose(frontier[case]["raw_safe_shed_w"],
                              row["planned_shed_w"])
               and frontier[case]["target_met_by_30s"] == row["target_met"]
               for case, row in action.items())
