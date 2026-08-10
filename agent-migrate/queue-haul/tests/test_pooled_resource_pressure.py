"""
Claim:
The pooled resource plot weights every case once at one requested-shed point
and reports a mean and 95% case-bootstrap interval for each normalized budget.

Plausible wrong implementations:
- Mix cases from different requested-shed coordinates.
- Duplicate one case and omit another while retaining the same row count.
- Average destination percentages instead of summing physical use and capacity.
- Combine resources with incompatible physical units.
- Treat an out-of-budget utilization as a valid percentage.
"""

import pytest

from plot_pooled_resource_pressure import mean_ci, summarize, total_utilization


def test_resource_summary_preserves_case_policy_and_request_coordinates():
    rows = [
        {"case_id": case, "policy": policy, "requested_fraction": fraction,
         "used|resource": value, "capacity|resource": 1,
         "unit|resource": "work"}
        for fraction, values in ((.5, (.25, .75)), (.75, (.1, .2)))
        for policy in ("queue_haul_lp", "queue_haul_greedy")
        for case, value in zip(("a", "b"), values)
    ]
    selected, summary = summarize(
        rows, .5, policies=("queue_haul_lp", "queue_haul_greedy"),
        budgets={"budget": ("resource",)},
    )
    assert len(selected) == 4
    assert {(row["mean"], row["lower_95"], row["upper_95"])
            for row in summary} == {(.5, .25, .75)}

    duplicated = [*rows[:1], {**rows[1], "case_id": "a"}, *rows[2:]]
    with pytest.raises(RuntimeError, match="weight each case once"):
        summarize(duplicated, .5, policies=("queue_haul_lp",),
                  budgets={"budget": ("resource",)})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mean_ci((0, 1.01))


def test_total_budget_sums_physical_use_and_capacity():
    row = {
        "used|small": 9, "capacity|small": 10, "unit|small": "work",
        "used|large": 10, "capacity|large": 100, "unit|large": "work",
    }
    assert total_utilization(row, ("small", "large")) == pytest.approx(19 / 110)
    with pytest.raises(ValueError, match="matching units"):
        total_utilization({**row, "unit|large": "bytes"}, ("small", "large"))
