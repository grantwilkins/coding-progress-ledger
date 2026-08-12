"""Claim: independent bandwidth and prefill drops change feasible simulated plans."""

import csv
import json

from repair_plan_shift_campaign import run


def test_snapshot_replans_change_action_mix_and_simulate(tmp_path):
    report = run(tmp_path)
    plans = json.loads((tmp_path / "plans.json").read_text())["plans"]
    mixes = list(csv.DictReader((tmp_path / "action_mix.csv").open()))
    diffs = list(csv.DictReader((tmp_path / "plan_diffs.csv").open()))

    assert report["passed"] and len(plans) == len(mixes) == len(diffs) == 3
    assert all(row["planner_feasible"] and row["simulated_deadline_met"]
               for row in plans)
    assert all(row["original_plan_violations"] for row in plans[1:])
    assert all(int(row["changed_sessions"]) > 0 for row in diffs[1:])
    assert len({tuple(row[key] for key in (
        "east_replay", "east_kv_transfer",
        "germany_replay", "germany_kv_transfer",
    )) for row in mixes}) == 3
    assert (tmp_path / "action_mix.png").stat().st_size > 0
