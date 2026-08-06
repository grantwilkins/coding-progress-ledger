"""
Claim:
Isolated bars select the faster method within each matched site/bandwidth/
context/repeat cell, while joint bars count one recorded site-method action per
session from completed attempts and normalize within each bar.

Plausible wrong implementations:
- Compare isolated actions across different bandwidths, contexts, or repeats.
- Count a failed attempt in addition to its completed retry.
- Merge East/West or replay/KV into fewer than four joint actions.
- Normalize a policy by all campaign actions instead of its own decisions.
"""

import csv
import json

from plot_network_action_breakdown import ACTIONS, isolated, joint


def _isolated(root, site, rows):
    root.mkdir()
    fields = ("status", "destination", "policy", "bandwidth",
              "condition_index", "repeat", "migration_s")
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for cell, replay, kv in rows:
            for method, value in (("replay", replay), ("kv_transfer", kv)):
                writer.writerow({"status": "complete", "destination": site,
                                 "policy": method, "bandwidth": cell[0],
                                 "condition_index": cell[1], "repeat": cell[2],
                                 "migration_s": value})


def _attempt(root, scenario_id, attempt, status, policy, moves=()):
    path = root / "scenarios" / scenario_id / attempt
    path.mkdir(parents=True)
    (path / "result.json").write_text(json.dumps({"status": status}))
    (path / "scenario.json").write_text(json.dumps({
        "policy": policy,
        "sessions": [{"session_id": move[0]} for move in moves],
    }))
    (path / "decision.json").write_text(json.dumps({"moves": [
        {"session_id": session, "destination_instance": site, "method": method}
        for session, site, method in moves
    ]}))


def test_isolated_winners_are_paired_within_scenario_cells(tmp_path):
    root = tmp_path / "east"
    _isolated(root, "east", [
        (("slow", "small", "0"), 1, 3),
        (("fast", "large", "0"), 5, 2),
    ])

    row = isolated(root, "east")

    assert row["total"] == 2
    assert row["counts"] == {
        ("east", "replay"): 1, ("east", "kv_transfer"): 1,
        ("west", "replay"): 0, ("west", "kv_transfer"): 0,
    }


def test_joint_counts_only_completed_attempt_actions_per_policy(tmp_path):
    _attempt(tmp_path, "a", "attempt-0001", "failed", "queue_haul", (
        ("ignored", "east", "kv_transfer"),))
    _attempt(tmp_path, "a", "attempt-0002", "complete", "queue_haul", (
        ("a0", "east", "replay"), ("a1", "west", "kv_transfer")))
    _attempt(tmp_path, "b", "attempt-0001", "complete", "random", (
        ("b0", "east", "kv_transfer"), ("b1", "west", "replay")))

    rows = {row["label"]: row for row in joint(tmp_path)}

    assert rows["Queue-Haul LP"]["total"] == 2
    assert rows["Queue-Haul LP"]["counts"] == dict.fromkeys(ACTIONS, 0) | {
        ("east", "replay"): 1, ("west", "kv_transfer"): 1}
    assert rows["Random"]["counts"] == dict.fromkeys(ACTIONS, 0) | {
        ("east", "kv_transfer"): 1, ("west", "replay"): 1}
