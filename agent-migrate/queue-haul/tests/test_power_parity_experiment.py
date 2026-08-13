"""Focused power parity plans stay randomized, matched, and complete."""

from pathlib import Path

import power_parity_experiment as experiment


SOURCE = Path(__file__).parents[1] / "outputs/policy-hardware-width8-packing-plan/plan.json"


def test_random_counts_cover_the_complete_curve():
    counts = experiment.migration_counts(50, 7)

    assert len(counts) == 50
    assert set(counts) == set(range(1, 9))
    assert counts == experiment.migration_counts(50, 7)


def test_plan_matches_every_policy_on_each_random_sample():
    plan = experiment.make_plan(SOURCE, episodes=8, seed=7)

    assert len(plan["scenarios"]) == 8 * len(experiment.POLICIES)
    for repeat in range(8):
        rows = [row for row in plan["scenarios"] if row["repeat"] == repeat]
        assert {row["policy"] for row in rows} == set(experiment.POLICIES)
        assert len({(row["match_id"], len(row["moves"])) for row in rows}) == 1
        assert all(row["power_validation"]["window_s"] == 5 for row in rows)
