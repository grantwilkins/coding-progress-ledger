"""
Claim:
The policy CDF pools every paired workload-constraint case once and reports the
first exact nonlinear shed attainment time, including the power-window delay.

Plausible wrong implementations:
- Divide by successful cases and hide missing deadline-attainment mass.
- Accumulate sessions out of completion order or linearize their shed gains.
- Omit the trailing power-window delay.
- Include the same draw-constraint case more than once for a policy.
"""

import pytest
from types import SimpleNamespace

from plot_workload_policy_attainment import (
    attainment_curve, attainment_time, execution_commits,
)


def test_attainment_time_uses_completion_order_full_set_and_power_window():
    seen = []

    def shed(moved):
        seen.append(tuple(moved))
        return {("early",): 3, ("early", "late"): 7}[tuple(moved)]

    assert attainment_time([(8, "late"), (2, "early")], shed, 6, 5) == 13
    assert seen == [("early",), ("early", "late")]


def test_attainment_curve_retains_misses_and_rejects_duplicate_cases():
    rows = [
        {"replicate": 0, "case_id": "hbm", "policy": "queue_haul",
         "attainment_time_s": 10},
        {"replicate": 0, "case_id": "none", "policy": "queue_haul",
         "attainment_time_s": ""},
        {"replicate": 1, "case_id": "hbm", "policy": "queue_haul",
         "attainment_time_s": 20},
    ]
    x, y = attainment_curve(rows, "queue_haul")
    assert x.tolist() == [0, 10, 20]
    assert y.tolist() == pytest.approx([0, 1 / 3, 2 / 3])
    with pytest.raises(RuntimeError, match="one row"):
        attainment_curve(rows + [rows[0]], "queue_haul")


def test_tail_cannot_change_deadline_attainment():
    row = lambda session, time: SimpleNamespace(
        session_id=session, committed_s=time,
    )
    assert execution_commits((row("primary", 20),), (row("tail", 40),)) \
        == [(20, "primary"), (40, "tail")]
    with pytest.raises(RuntimeError, match="before the deadline"):
        execution_commits((), (row("tail", 30),))
