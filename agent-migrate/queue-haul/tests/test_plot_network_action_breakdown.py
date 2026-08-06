"""
Claim:
The four displayed joint-policy bars count one recorded site-method action per
session from completed attempts and normalize within each policy.

Plausible wrong implementations:
- Count a failed attempt in addition to its completed retry.
- Merge East/West or replay/KV into fewer than four joint actions.
- Normalize a policy by all campaign actions instead of its own decisions.
- Retain Lagrangian or random policies outside the requested comparison.
"""

import json

from plot_network_action_breakdown import ACTIONS, joint


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


def test_joint_counts_completed_attempts_and_excludes_other_policies(tmp_path):
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
    assert set(rows) == {"Queue-Haul LP"}
