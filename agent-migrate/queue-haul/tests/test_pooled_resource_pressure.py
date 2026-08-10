"""
Claim:
The pooled resource plot weights every case once at one requested-shed point
and reports a mean and 95% case-bootstrap interval for each normalized budget.

Plausible wrong implementations:
- Mix cases from different requested-shed coordinates.
- Duplicate one case and omit another while retaining the same row count.
- Pool different resources or policies before computing the interval.
- Treat an out-of-budget utilization as a valid percentage.
"""

import pytest

from plot_pooled_resource_pressure import mean_ci, summarize


def test_resource_summary_preserves_case_policy_and_request_coordinates():
    rows = [
        {"case_id": case, "policy": policy, "requested_fraction": fraction,
         "resource": value}
        for fraction, values in ((.5, (.25, .75)), (.75, (.1, .2)))
        for policy in ("queue_haul_lp", "queue_haul_greedy")
        for case, value in zip(("a", "b"), values)
    ]
    selected, summary = summarize(
        rows, .5, policies=("queue_haul_lp", "queue_haul_greedy"),
        resources=("resource",),
    )
    assert len(selected) == 4
    assert {(row["mean"], row["lower_95"], row["upper_95"])
            for row in summary} == {(.5, .25, .75)}

    duplicated = [*rows[:1], {**rows[1], "case_id": "a"}, *rows[2:]]
    with pytest.raises(RuntimeError, match="weight each case once"):
        summarize(duplicated, .5, policies=("queue_haul_lp",),
                  resources=("resource",))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mean_ci((0, 1.01))
