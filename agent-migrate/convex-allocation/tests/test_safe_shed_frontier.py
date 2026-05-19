"""
Claim:
The safe-shed frontier reports the largest rounded queue-safe shed fraction for
each policy and slack multiplier, using the requested miss and normalized-delay
safety definition, and carries the replay/state shed share at the frontier.

Plausible wrong implementations:
- Treat raw p95 delay as the deadline-normalized p95 delay.
- Ignore rounded under-shed when marking a row safe.
- Report the first safe shed fraction instead of the largest safe fraction.
- Classify rounded under-shed as a resource bottleneck instead of a rounding artifact.
- Drop the action-mix diagnostics from the frontier row.
"""

from __future__ import annotations

from experiments.run_safe_shed_frontier import _failure_mode, _frontier_rows, _is_safe
from problem import WORKLOAD_SLACK, make_problem
from catalog import get_model


def test_safe_definition_uses_target_miss_and_normalized_p95_boundaries():
    metrics = {
        "rounded_shed_achieved": 10.0,
        "rounded_shed_target": 10.0,
        "deadline_miss_rate": 0.01,
        "p95_normalized_reconstruction_delay": 1.0,
    }

    assert _is_safe(metrics)

    assert not _is_safe({**metrics, "rounded_shed_achieved": 9.99})
    assert not _is_safe({**metrics, "deadline_miss_rate": 0.011})
    assert not _is_safe({**metrics, "p95_normalized_reconstruction_delay": 1.001})


def test_frontier_uses_largest_safe_shed_fraction_and_marks_none_safe():
    rows = [
        _row("policy-a", 1.0, 0.2, True),
        _row("policy-a", 1.0, 0.3, False),
        _row("policy-a", 1.0, 0.4, True),
        _row("policy-b", 1.0, 0.2, False),
    ]

    frontier = _frontier_rows(rows, policies=("policy-a", "policy-b"), slack_multipliers=(1.0,))

    assert frontier[0]["max_safe_shed_fraction"] == 0.4
    assert frontier[0]["p95_delay_at_frontier"] == 4.0
    assert frontier[0]["replay_shed_frac_at_frontier"] == 0.25
    assert frontier[0]["state_shed_frac_at_frontier"] == 0.75
    assert frontier[1]["max_safe_shed_fraction"] == "UNSAFE"


def test_failure_mode_separates_rounding_slack_and_resource_bottlenecks():
    row = {
        "rounded_shed_achieved": 9.0,
        "rounded_shed_target": 10.0,
        "miss_rate": 0.0,
        "p95_normalized_delay": 0.5,
        "max_net_busy": 0.1,
        "max_prefill_busy": 0.1,
    }
    assert _failure_mode(row) == "rounding artifact"

    assert _failure_mode({**row, "rounded_shed_achieved": 10.0, "miss_rate": 0.02}) == "deadline misses"
    assert (
        _failure_mode(
            {
                **row,
                "rounded_shed_achieved": 10.0,
                "p95_normalized_delay": 1.1,
                "max_net_busy": 1.2,
            }
        )
        == "network bottleneck"
    )


def test_make_problem_scales_slack_without_changing_workload():
    problem = make_problem(get_model("GLM-5"), "transition-coupled", slack_multiplier=0.5)

    assert (problem.deadline_s == 0.5 * WORKLOAD_SLACK).all()
    assert (problem.T == make_problem(get_model("GLM-5"), "transition-coupled").T).all()


def _row(policy, slack_multiplier, shed_fraction, safe):
    return {
        "policy": policy,
        "slack_multiplier": slack_multiplier,
        "shed_fraction": shed_fraction,
        "safe": safe,
        "p95_delay": shed_fraction * 10.0,
        "p95_normalized_delay": 0.9,
        "miss_rate": 0.0,
        "max_net_busy": 0.2,
        "max_prefill_busy": 0.3,
        "replay_shed_frac": 0.25,
        "state_shed_frac": 0.75,
    }
