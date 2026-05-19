"""
Claim:
The source-load frontier reports the largest rounded queue-safe source-load
fraction for each policy and deadline scale, using the requested miss-rate and
delay-over-deadline safety definition, and carries replay/state-transfer load
shares at the frontier.

Plausible wrong implementations:
- Treat raw p95 delay as delay divided by deadline.
- Ignore rounded source-load shortfall when marking a row safe.
- Report the first safe source-load fraction instead of the largest safe fraction.
- Classify rounded source-load shortfall as a resource bottleneck instead of rounding.
- Drop the action-mix diagnostics from the frontier row.
"""

from __future__ import annotations

from experiments.run_source_load_frontier import _failure_mode, _frontier_rows, _is_safe
from problem import WORKLOAD_SLACK, make_problem
from catalog import get_model


def test_safe_definition_uses_target_miss_and_normalized_p95_boundaries():
    metrics = {
        "source_load_moved_s": 10.0,
        "source_load_target_s": 10.0,
        "deadline_miss_rate": 0.01,
        "p95_reconstruction_delay_ratio": 1.0,
    }

    assert _is_safe(metrics)

    assert not _is_safe({**metrics, "source_load_moved_s": 9.99})
    assert not _is_safe({**metrics, "deadline_miss_rate": 0.011})
    assert not _is_safe({**metrics, "p95_reconstruction_delay_ratio": 1.001})


def test_frontier_uses_largest_safe_source_load_fraction_and_marks_none_safe():
    rows = [
        _row("policy-a", 1.0, 0.2, True),
        _row("policy-a", 1.0, 0.3, False),
        _row("policy-a", 1.0, 0.4, True),
        _row("policy-b", 1.0, 0.2, False),
    ]

    frontier = _frontier_rows(rows, policies=("policy-a", "policy-b"), deadline_scales=(1.0,))

    assert frontier[0]["max_safe_source_load_fraction"] == 0.4
    assert frontier[0]["p95_delay_at_frontier"] == 4.0
    assert frontier[0]["replay_load_fraction_at_frontier"] == 0.25
    assert frontier[0]["state_transfer_load_fraction_at_frontier"] == 0.75
    assert frontier[1]["max_safe_source_load_fraction"] == "UNSAFE"


def test_failure_mode_separates_rounding_deadline_and_resource_bottlenecks():
    row = {
        "source_load_moved_s": 9.0,
        "source_load_target_s": 10.0,
        "deadline_miss_rate": 0.0,
        "p95_delay_over_deadline": 0.5,
        "max_network_busy_fraction": 0.1,
        "max_prefill_busy_fraction": 0.1,
    }
    assert _failure_mode(row) == "rounding artifact"

    assert _failure_mode({**row, "source_load_moved_s": 10.0, "deadline_miss_rate": 0.02}) == "deadline misses"
    assert (
        _failure_mode(
            {
                **row,
                "source_load_moved_s": 10.0,
                "p95_delay_over_deadline": 1.1,
                "max_network_busy_fraction": 1.2,
            }
        )
        == "network bottleneck"
    )


def test_make_problem_scales_deadline_without_changing_workload():
    problem = make_problem(get_model("GLM-5"), "transition-coupled", deadline_scale=0.5)

    assert (problem.deadline_s == 0.5 * WORKLOAD_SLACK).all()
    assert (problem.T == make_problem(get_model("GLM-5"), "transition-coupled").T).all()


def _row(policy, deadline_scale, source_load_fraction, safe):
    return {
        "policy": policy,
        "deadline_scale": deadline_scale,
        "source_load_fraction": source_load_fraction,
        "safe": safe,
        "p95_delay_s": source_load_fraction * 10.0,
        "p95_delay_over_deadline": 0.9,
        "deadline_miss_rate": 0.0,
        "max_network_busy_fraction": 0.2,
        "max_prefill_busy_fraction": 0.3,
        "replay_load_fraction": 0.25,
        "state_transfer_load_fraction": 0.75,
    }
