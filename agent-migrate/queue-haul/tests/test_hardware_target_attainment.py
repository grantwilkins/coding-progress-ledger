"""
Claim:
Hardware target attainment divides realized shed by requested shed within each
episode before pooling repeats at each policy and requested fraction.

Plausible wrong implementations:
- Divide median realized shed by median requested shed after pooling.
- Normalize by removable power instead of the requested target.
- Mix conditions or policies while computing quantiles.
"""

import pytest

from plot_hardware_target_attainment import summarize


def test_target_attainment_normalizes_each_episode_before_pooling():
    scenarios = [
        {"condition_index": 0, "requested_shed_fraction": .6},
        {"condition_index": 1, "requested_shed_fraction": .8},
    ]
    rows = [{
        "condition_index": str(condition), "policy": policy,
        "realized_shed_w": str(realized), "requested_shed_w": str(requested),
        "status": "complete", "request_failures": "0", "deadline_met": "True",
    } for policy, condition, requested, realized in (
        ("queue_haul", 0, 10, 12), ("queue_haul", 0, 20, 20),
        ("queue_haul", 1, 20, 10), ("queue_haul", 1, 40, 40),
        ("greedy", 0, 10, 8), ("greedy", 0, 20, 20),
        ("greedy", 1, 20, 10), ("greedy", 1, 40, 20),
    )]
    summary = {(row["policy"], row["requested_fraction"]): row
               for row in summarize(rows, scenarios)}

    assert summary["queue_haul", .6]["median"] == pytest.approx(1.1)
    assert summary["queue_haul", .8]["median"] == pytest.approx(.75)
    assert summary["greedy", .6]["median"] == pytest.approx(.9)
