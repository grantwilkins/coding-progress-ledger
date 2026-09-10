import csv
import json
from collections import Counter

import pytest

import constrained_state_campaign as campaign
import marginal_state_campaign as marginal


def test_offline_boundaries_freeze_balanced_live_comparison(tmp_path):
    inputs = campaign.default_inputs()
    states = []
    for axis, work, tokens in ((None, [0, 0], 963152),
                               ("prefill", [.36, .012], 963152),
                               ("serving", [.037, 1.35], 963152),
                               ("hbm", [0, 0], 875760)):
        states.append({"wan_mbps": 10000,
                       **{f"n_{name}": int(name == axis) for name in campaign.AXES},
                       "capacity_inputs": {"baseline_work": work,
                                           "kv_capacity_tokens": tokens}})
    source = {"frozen": campaign.freeze_inputs(inputs, 7), "states": states}
    plan = marginal.prepare(source, tmp_path)

    assert json.loads((tmp_path / "plan.json").read_text()) == plan
    assert campaign._verify_plan(plan) == plan["sha256"]
    assert len(plan["schedule"]) == 90
    assert Counter((row["family"], row["policy"]) for row in plan["schedule"]) == {
        (state["family"], policy): 3
        for state in plan["states"] for policy in campaign.POLICIES}
    assert {row["repeat"] for row in plan["schedule"]} == {0, 1, 2}
    assert {row["pack_id"] for row in plan["schedule"]} == {"large-r0"}
    assert marginal.prepare(source, tmp_path / "repeat") == plan

    demands = [row for row in plan["frozen"]["inputs"]["action_demands"]
               if row["pack_id"] == "large-r0"]
    by_family = {row["family"]: row for row in plan["states"]}
    for family, resource, action in (("wan", "wan", "kv_transfer"),
                                     ("prefill", "prefill", "replay"),
                                     ("serving", "service", "replay"),
                                     ("hbm", "hbm", "replay")):
        demand = sum(row[f"d_{resource}"] for row in demands if row["action"] == action)
        assert by_family[family]["capacity_inputs"]["resources"][resource] == pytest.approx(
            .95 * demand, abs=1)
    for policy in ("queue_haul", "greedy"):
        rows = [row for row in plan["offline_decisions"] if row["policy"] == policy]
        assert len({(row["kv_transfer"], row["replay"]) for row in rows}) > 1
        coupled = next(row for row in rows if row["family"] == "wan_prefill")
        assert coupled["admitted"] == 8
        assert coupled["kv_transfer"] > 0 and coupled["replay"] > 0
        assert next(row for row in rows if row["family"] == "serving")["admitted"] == 7
    annotation = next(row for row in plan["annotations"]
                      if row["state_id"] == by_family["wan_prefill"]["state_id"])
    assert annotation["mixed_full"]
    assert not annotation["kv_full"] and not annotation["replay_full"]

    decisions = {(row["state_id"], row["policy"]): row["moves"]
                 for row in plan["offline_decisions"]}
    raw = [{"episode_id": job["episode_id"],
            "capacity_inputs": by_family[job["family"]]["capacity_inputs"],
            "decisions": [{**move, "completion_s": None}
                          for move in decisions[job["state_id"], job["policy"]]]}
           for job in plan["schedule"]]
    path = tmp_path / "raw_episodes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in raw[:5]) + "\n")
    assert not marginal.summarize(plan, tmp_path)["complete"]
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    validation = marginal.summarize(plan, tmp_path)
    assert validation["complete"] and validation["three_repeats_per_case_policy"]
    assert all(row["action_shift"] and row["mixed_observed"]
               for row in validation["action_checks"])
    # Dispatch choices remain observable even when every attempt misses the target.
    with (tmp_path / "summary.csv").open() as handle:
        assert {row["attained"] for row in csv.DictReader(handle)} == {"0"}
    for episode in raw:
        for move in episode["decisions"]:
            move["action"] = "kv_transfer"
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(ValueError, match="action"):
        marginal.summarize(plan, tmp_path)
