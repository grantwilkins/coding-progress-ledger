"""
Claim:
The stress-frontier plot scales every policy by their shared maximum, and
Queue-Haul plans with the LP rather than an exact MILP. Every state labeled as
a power bootstrap samples a measured curve or a joint analytic parameter draw.
The Queue-Haul row retains the better fully simulated LP or greedy deadline power.

Plausible wrong implementations:
- Normalize each policy independently.
- Resolve the Queue-Haul policy to the exact max-shed MILP.
- Silently reuse one central power curve for every bootstrap state.
- Let power-bootstrap cardinality reshuffle the timing Latin hypercube.
- Compare fractional credits instead of the modeled power attained by deadline.
- Let replanning make the attained frontier decrease at a looser deadline.
"""

from types import SimpleNamespace
import csv

import pytest

import stress_frontier_campaign as campaign


def test_suite_has_40_equal_weight_stratified_states():
    phase = SimpleNamespace(bootstrap=((1, 2, 3, 4),))
    sources = {name: SimpleNamespace(relative_error=error) for name, error in (
        ("service", .1), ("replay", .2), ("kv_transfer", .3))}
    profile = SimpleNamespace(case=lambda: SimpleNamespace(phase_power=phase), sources=sources)
    states = campaign.stress_states(profile, 3)
    assert len(states) == 40
    assert sum(row["weight"] for row in states) == pytest.approx(1)
    assert {row["regime"] for row in states} == {row[0] for row in campaign.REGIMES}
    assert states == campaign.stress_states(profile, 3)
    varied = SimpleNamespace(bootstrap=((1, 2, 3, 4), (2, 3, 4, 5)))
    varied_profile = SimpleNamespace(
        case=lambda: SimpleNamespace(phase_power=varied), sources=sources)
    varied_states = campaign.stress_states(varied_profile, 3)
    assert [{k: v for k, v in row.items() if k != "power_bootstrap_index"}
            for row in states] == [
                {k: v for k, v in row.items() if k != "power_bootstrap_index"}
                for row in varied_states]


def test_suite_rejects_a_profile_without_power_bootstrap_draws():
    phase = SimpleNamespace(bootstrap=(), measured_power_bootstrap=())
    profile = SimpleNamespace(case=lambda: SimpleNamespace(phase_power=phase))
    with pytest.raises(ValueError, match="power bootstrap draws"):
        campaign.stress_states(profile)


def test_coverage_is_fifth_smallest_of_40():
    assert campaign.fifth_smallest(range(40)) == 4
    with pytest.raises(ValueError, match="exactly 40"):
        campaign.fifth_smallest(range(39))


def test_reduction_labels_unpromoted_frontier_as_modeled(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    rows = [{"deadline_s": deadline, "policy": policy, "state_id": str(state),
             "shed_by_deadline_w": state}
            for deadline in campaign.DEADLINES
            for policy in campaign.POLICIES
            for state in range(40)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(campaign, "_plot", lambda *_args: None)
    result = campaign.reduce([path], tmp_path / "frontier.json", {"passed": False})
    assert not result["empirical"]
    assert {row["coverage_90_shed_w"] for row in result["frontier"]} == {4}
    assert {row["claim"] for row in result["frontier"]} == {
        "modeled stress-suite sensitivity"}


def test_reduction_carries_tighter_deadline_plans_forward(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    rows = [{"deadline_s": deadline, "policy": policy, "state_id": str(state),
             "shed_by_deadline_w": 10 if policy == "queue_haul" and deadline == 10 else 0}
            for deadline in campaign.DEADLINES
            for policy in campaign.POLICIES
            for state in range(40)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(campaign, "_plot", lambda *_args: None)

    result = campaign.reduce([path], tmp_path / "frontier.json")

    assert [row["coverage_90_shed_w"] for row in result["frontier"]
            if row["policy"] == "queue_haul"] == [10] * len(campaign.DEADLINES)


def test_plot_normalizes_to_the_shared_maximum(tmp_path, monkeypatch):
    values = range(2, 2 * len(campaign.POLICIES) + 1, 2)
    rows = [{"deadline_s": 10, "policy": policy,
             "coverage_90_shed_w": value}
            for policy, value in zip(campaign.POLICIES, values)]
    lines = []
    monkeypatch.setattr(campaign.plot_style, "policy_style", lambda policy: {})
    import matplotlib.axes
    monkeypatch.setattr(matplotlib.axes.Axes, "plot",
                        lambda self, x, y, **kwargs: lines.append(y))
    campaign._plot(rows, tmp_path / "frontier", False)
    assert lines == [[value / max(values)] for value in values]


def test_stress_frontier_plans_queue_haul_with_the_lp():
    assert campaign.network.joint_solver("queue_haul") == "lp_work_first"
    assert campaign.LP_SOLVER == "lp_work_first_best_effort"
    assert campaign.POWER_BLIND_SOLVER == "lp_power_blind_best_effort"
    assert campaign.GREEDY_SOLVER == "greedy_best_effort"
    assert "greedy" in campaign.POLICIES
    assert not {"greedy_lagrangian", "exact_modeled_milp_optimum"} & set(campaign.POLICIES)


def test_queue_haul_guard_keeps_the_lower_deadline_power():
    lp = SimpleNamespace(modeled_source_power_at_deadline_w=140)
    greedy = SimpleNamespace(modeled_source_power_at_deadline_w=120)
    assert campaign._best_outcome((("lp", lp), ("greedy", greedy))) == ("greedy", greedy)
