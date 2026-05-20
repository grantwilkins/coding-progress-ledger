"""
Claim:
The retained-state frontier reports the largest rounded queue-safe
retained-prefill fraction for each policy and deadline scale, using the requested
miss-rate and release-relative delay-over-deadline safety definition, and
carries state-TB plus replay/state-transfer retained-prefill shares at the 30m
drain frontier.

Plausible wrong implementations:
- Treat raw p95 delay as delay divided by deadline.
- Ignore rounded retained-prefill shortfall when marking a row safe.
- Report the first safe retained-prefill fraction instead of the largest safe fraction.
- Mix zero-window burst rows into the 30m drain frontier.
- Classify rounded retained-prefill shortfall as a resource bottleneck instead of rounding.
- Drop the action-mix diagnostics from the frontier row.
"""

from __future__ import annotations

from experiments.run_retained_state_frontier import (
    DRAIN_WINDOWS_S,
    FRONTIER_POLICIES,
    MAIN_POLICY,
    ORACLE_POLICY,
    PLOT_DRAIN_WINDOW_S,
    _failure_mode,
    _frontier_rows,
    _is_safe,
)
from problem import WORKLOAD_DEADLINE_S, make_problem
from catalog import get_model


def test_safe_definition_uses_target_miss_and_normalized_p95_boundaries():
    metrics = {
        "retained_prefill_moved_s": 10.0,
        "retained_prefill_target_s": 10.0,
        "deadline_miss_rate": 0.01,
        "p95_reconstruction_delay_ratio": 1.0,
    }

    assert _is_safe(metrics)

    assert not _is_safe({**metrics, "retained_prefill_moved_s": 9.99})
    assert not _is_safe({**metrics, "deadline_miss_rate": 0.011})
    assert not _is_safe({**metrics, "p95_reconstruction_delay_ratio": 1.001})


def test_frontier_uses_largest_safe_retained_prefill_fraction_and_marks_none_safe():
    rows = [
        _row("policy-a", 1.0, 0.2, True),
        _row("policy-a", 1.0, 0.3, False),
        _row("policy-a", 1.0, 0.4, True),
        _row("policy-a", 1.0, 0.7, True, drain_window_s=0.0),
        _row("policy-b", 1.0, 0.2, False),
    ]

    frontier = _frontier_rows(rows, policies=("policy-a", "policy-b"), deadline_scales=(1.0,))

    assert frontier[0]["max_safe_retained_prefill_fraction"] == 0.4
    assert frontier[0]["p95_delay_at_frontier"] == 4.0
    assert frontier[0]["drain_window_s"] == 1800.0
    assert frontier[0]["average_equivalent_state_target_tb_at_frontier"] == 3.0
    assert frontier[0]["actual_evacuated_state_tb_at_frontier"] == 4.0
    assert frontier[0]["actual_evacuated_nvl72_hbm_fraction_at_frontier"] == 4.0 / 13.4
    assert frontier[0]["replay_retained_prefill_fraction_at_frontier"] == 0.25
    assert frontier[0]["state_transfer_retained_prefill_fraction_at_frontier"] == 0.75
    assert frontier[1]["max_safe_retained_prefill_fraction"] == "UNSAFE"


def test_failure_mode_separates_rounding_deadline_and_resource_bottlenecks():
    row = {
        "retained_prefill_moved_s": 9.0,
        "retained_prefill_target_s": 10.0,
        "deadline_miss_rate": 0.0,
        "p95_delay_over_deadline": 0.5,
        "network_capacity_pressure": 0.1,
        "prefill_capacity_pressure": 0.1,
    }
    assert _failure_mode(row) == "rounding artifact"

    assert _failure_mode({**row, "retained_prefill_moved_s": 10.0, "deadline_miss_rate": 0.02}) == "deadline misses"
    assert (
        _failure_mode(
            {
                **row,
                "retained_prefill_moved_s": 10.0,
                "p95_delay_over_deadline": 1.1,
                "network_capacity_pressure": 1.2,
            }
        )
        == "network bottleneck"
    )


def test_make_problem_scales_deadline_without_changing_workload():
    problem = make_problem(
        get_model("GLM-5"),
        "transition-coupled",
        deadline_scale=0.5,
        workload_source="fixed",
    )

    assert (problem.deadline_s == 0.5 * WORKLOAD_DEADLINE_S).all()
    assert (
        problem.T
        == make_problem(get_model("GLM-5"), "transition-coupled", workload_source="fixed").T
    ).all()


def test_frontier_uses_realistic_grid_drain_windows_with_burst_reference():
    assert DRAIN_WINDOWS_S == (0.0, 900.0, 1800.0, 3600.0)
    assert PLOT_DRAIN_WINDOW_S == 1800.0


def test_frontier_leads_with_deadline_penalty_and_keeps_cvxpy_as_oracle():
    assert MAIN_POLICY == "deadline-penalty-rounded"
    assert ORACLE_POLICY == "CVXPY-rounded"
    assert FRONTIER_POLICIES[:2] == (MAIN_POLICY, ORACLE_POLICY)


def _row(policy, deadline_scale, retained_prefill_fraction, safe, drain_window_s=1800.0):
    return {
        "policy": policy,
        "deadline_scale": deadline_scale,
        "drain_window_s": drain_window_s,
        "retained_prefill_fraction": retained_prefill_fraction,
        "safe": safe,
        "p95_delay_s": retained_prefill_fraction * 10.0,
        "p95_delay_over_deadline": 0.9,
        "deadline_miss_rate": 0.0,
        "network_capacity_pressure": 0.2,
        "prefill_capacity_pressure": 0.3,
        "average_equivalent_state_target_tb": 3.0,
        "actual_evacuated_state_tb": 4.0,
        "actual_evacuated_nvl72_hbm_fraction": 4.0 / 13.4,
        "replay_retained_prefill_fraction": 0.25,
        "state_transfer_retained_prefill_fraction": 0.75,
        "drain_completion_s": 61.0,
    }
