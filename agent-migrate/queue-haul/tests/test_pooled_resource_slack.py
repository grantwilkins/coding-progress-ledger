"""
Claim:
Resource slack starts at one, decreases in completion order, and reaches zero
when any component constraint within a resource class is exhausted.

Plausible wrong implementations:
- Accumulate sessions in identifier order instead of completion order.
- Plot utilization rather than residual slack.
- Average component constraints and hide the first exhausted component.
- Charge work from a session that has not completed by the deadline.
"""

import pytest

from plot_pooled_resource_slack import completion_slack


def test_completion_slack_tracks_the_first_component_to_bind():
    rows = completion_slack(
        {"late": 2, "early": 1},
        {
            "early": {"a": .4, "b": .1},
            "late": {"a": .3, "b": .8},
            "unfinished": {"a": .3, "b": .1},
        },
        {"budget": ("a", "b")},
        3,
    )
    assert rows == [
        (0, {"budget": 1}),
        (1, {"budget": pytest.approx(.6)}),
        (2, {"budget": pytest.approx(.1)}),
        (3, {"budget": pytest.approx(.1)}),
    ]
    with pytest.raises(ValueError, match="completion-ordered"):
        completion_slack({"missing": 1}, {}, {"budget": ("a",)}, 3)
