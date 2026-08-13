"""
Claim:
Germany-only bandwidth and East-only prefill drops independently change the
feasible simulated plan.

Plausible wrong implementations:
- Both destination routes are throttled.
- Germany rather than East loses prefill headroom.
- One degradation leaks into the other independent branch.
- Action changes are compared against a degraded rather than original plan.
"""

import csv
import json

from repair_plan_shift_campaign import GERMANY_BANDWIDTH_MBPS, run


def test_snapshot_replans_change_action_mix_and_simulate(tmp_path):
    report = run(tmp_path)
    plans = json.loads((tmp_path / "plans.json").read_text())["plans"]
    mixes = list(csv.DictReader((tmp_path / "action_mix.csv").open()))
    diffs = list(csv.DictReader((tmp_path / "plan_diffs.csv").open()))

    assert report["passed"] and len(plans) == len(mixes) == len(diffs) == 3
    assert all(row["planner_feasible"] and row["simulated_deadline_met"]
               for row in plans)
    assert all(len(row["moves"]) == 14 for row in plans)
    assert all(row["original_plan_violations"] for row in plans[1:])
    assert all(int(row["changed_sessions"]) > 0 for row in diffs[1:])
    assert len({tuple(row[key] for key in (
        "east_replay", "east_kv_transfer",
        "germany_replay", "germany_kv_transfer",
    )) for row in mixes}) == 3
    original, germany_bandwidth, east_prefill = plans
    assert germany_bandwidth["bandwidth_mbps"] == {
        "east": original["bandwidth_mbps"]["east"],
        "germany": GERMANY_BANDWIDTH_MBPS}
    assert (germany_bandwidth["east_prefill_rho"],
            germany_bandwidth["germany_prefill_rho"]) == (.25, .25)
    assert east_prefill["bandwidth_mbps"] == original["bandwidth_mbps"]
    assert (east_prefill["east_prefill_rho"],
            east_prefill["germany_prefill_rho"]) == (.976, .25)
    assert (tmp_path / "action_mix.png").stat().st_size > 0
    assert (tmp_path / "action_mix.pdf").stat().st_size > 0
