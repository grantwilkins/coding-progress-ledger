"""
Claim:
Resource slack starts at one, decreases in completion order, and reaches zero
when any component constraint within a resource class is exhausted.

Plausible wrong implementations:
- Accumulate sessions in identifier order instead of completion order.
- Plot utilization rather than residual slack.
- Average component constraints and hide the first exhausted component.
- Charge work from a session that has not completed by the deadline.
- Credit deadline-blind with work completed after the evaluation deadline.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import plot_pooled_resource_slack as slack
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


def test_deadline_blind_plans_long_but_is_evaluated_at_30s(monkeypatch):
    @dataclass(frozen=True)
    class Problem:
        deadline_s: float = 30
        end_s: float = 30
        power_limit_w: float = 0
        final_state: str = "awake"
        assumed_shutdown_s: float | None = None

    seen = {}
    move = SimpleNamespace(session_id="s", method="kv_transfer",
                           destination_pool="p")
    result = SimpleNamespace(moves=(move,))

    def solve(problem, *_args, **_kwargs):
        seen["planning_deadline_s"] = problem.deadline_s
        return result

    def table(problem, *_args, **_kwargs):
        capacities = np.array((problem.deadline_s, problem.deadline_s, 1.0))
        return SimpleNamespace(
            sessions=(SimpleNamespace(session_id="s"),),
            candidates=(SimpleNamespace(session=0, method="kv_transfer", pool=0),),
            resources=csr_matrix(np.array((.5, .5, .5)).reshape(3, 1)),
            resource_names=("kv:p", "route:p", "service:p"),
            resource_capacities=capacities,
        )

    def predict(problem, *_args, **_kwargs):
        seen["evaluation_deadline_s"] = problem.deadline_s
        return SimpleNamespace(
            sessions=(SimpleNamespace(session_id="s", committed_s=31),),
            modeled_source_power_at_deadline_w=90,
        )

    monkeypatch.setattr(slack.campaign, "source_power", lambda *_args: 100)
    monkeypatch.setattr(slack.campaign, "solve", solve)
    monkeypatch.setattr(slack, "candidate_table", table)
    monkeypatch.setattr(slack, "ExpectedPower", lambda *_args: None)
    monkeypatch.setattr(slack, "_expected_scenario", lambda problem, _moves: problem)
    monkeypatch.setattr(slack, "predict", predict)

    rows, met = slack._plan_timeline(
        Problem(), SimpleNamespace(pools=(SimpleNamespace(pool_id="p"),)),
        {}, None, "deadline_blind", "lp_work_first", 0, 20,
    )

    assert seen == {"planning_deadline_s": 90, "evaluation_deadline_s": 30}
    assert rows == [(0, {"VRAM": 1, "Network bandwidth": 1,
                         "Prefill capacity": 1}),
                    (30, {"VRAM": 1, "Network bandwidth": 1,
                          "Prefill capacity": 1})]
    assert not met
