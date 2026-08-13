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


def test_coverage_is_fifth_smallest_of_40():
    assert campaign.fifth_smallest(range(40)) == 4
    with pytest.raises(ValueError, match="exactly 40"):
        campaign.fifth_smallest(range(39))


def test_reduction_labels_unpromoted_frontier_as_modeled(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    rows = [{"deadline_s": deadline, "policy": policy,
             "shed_by_deadline_w": state}
            for deadline in campaign.DEADLINES
            for policy in (*campaign.POLICIES, campaign.REFERENCE)
            for state in range(40)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(campaign, "_plot", lambda *_args: None)
    result = campaign.reduce([path], tmp_path / "frontier.json", {"passed": False})
    assert not result["empirical"]
    assert {row["coverage_90_shed_w"] for row in result["frontier"]} == {4}
    assert {row["claim"] for row in result["frontier"]} == {
        "modeled stress-suite sensitivity"}
