"""
Claim:
The pooled frontier reports attained/requested within each equally weighted
case, then reports the median and interquartile range at each policy/request.

Plausible wrong implementations:
- Pool raw watts so high-power cases dominate the summary.
- Divide attained shed by removable rather than requested power.
- Treat the zero request as zero rather than fully attained.
- Weight a case more because it has more sessions or policies.
- Mix requested-shed coordinates while computing uncertainty bands.
- Compute quartiles across policies rather than across cases.
"""

import pytest

from plot_pooled_shed_frontier import pooled_summary


def test_pooled_summary_normalizes_cases_and_keeps_policy_coordinates():
    rows = [
        {"case_id": case, "policy": policy, "requested_fraction": request,
         "safely_attained_fraction": value}
        for policy, request, values in (
            ("queue_haul_lp", 0, (0, 0, 0, 0)),
            ("queue_haul_lp", .5, (.1, .2, .3, .4)),
            ("queue_haul_greedy", .5, (.05, .1, .15, .2)),
        )
        for case, value in zip("abcd", values)
    ]

    summary = {(row["policy"], row["requested_fraction"]): row
               for row in pooled_summary(rows)}
    assert summary["queue_haul_lp", .5] == {
        "policy": "queue_haul_lp", "requested_fraction": .5,
        "lower_quartile": pytest.approx(.35), "median": pytest.approx(.5),
        "upper_quartile": pytest.approx(.65), "cases": 4,
    }
    assert summary["queue_haul_greedy", .5]["median"] == pytest.approx(.25)
    assert summary["queue_haul_lp", 0]["median"] == 1

    duplicated = [*rows, next(row for row in rows
                              if row["policy"] == "queue_haul_lp"
                              and row["requested_fraction"] == .5)]
    with pytest.raises(RuntimeError, match="weight each case once"):
        pooled_summary(duplicated)
