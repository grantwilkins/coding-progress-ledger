"""
Claim:
Every policy consumes the same frozen hardware episode, reaction latency starts
at one policy epoch, and failed or incomplete episodes remain in curve denominators.

Plausible wrong implementations:
- Resample sessions or contexts independently for each policy.
- Measure from each migration's own start and hide scheduler wait.
- Let random omit sessions that Queue-Haul must migrate.
- Condition completion curves only on successful migrations.
- Pair continuation TTFT with a control from another episode.
"""

import json

import pytest

from policy_hardware_campaign import (
    completion_curve,
    make_plan,
    reduce_run,
)


def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "queue-haul-migration-manifest-v2",
        "workload": "coding",
        "sessions": [{
            "id": f"s{i}", "job_class": "coding", "rank": i,
            "state_code": f"C{i}", "turns": [{
                "time_s": 0, "input_tokens": 4096,
                "append_tokens": 32, "output_tokens": 1, "reset": False,
            }],
        } for i in range(4)],
    }))
    return path


def test_plan_pairs_every_policy_on_the_same_complete_episode(tmp_path):
    plan = make_plan(manifest(tmp_path), episodes=2, sessions=4, seed=7)

    for episode in range(2):
        rows = [row for row in plan["scenarios"]
                if row["episode"] == episode]
        signatures = {
            tuple(sorted((item["session_id"], item["initial_tokens"])
                         for item in row["sessions"]))
            for row in rows
        }
        assert len(signatures) == 1
        expected = {item[0] for item in signatures.pop()}
        assert all(
            {move["session_id"] for move in row["moves"]} == expected
            for row in rows if row["kind"] == "migration"
        )


def test_reduction_uses_common_epoch_and_keeps_failed_denominator(tmp_path):
    control = {
        "scenario_id": "control", "match_id": "same", "episode": 0,
        "policy": "control", "kind": "control", "deadline_s": 10,
        "sessions": [{"session_id": name, "initial_tokens": 4096}
                     for name in ("a", "b")],
    }
    base = {
        **control, "kind": "migration",
        "sessions": [{"session_id": name, "initial_tokens": 4096}
                     for name in ("a", "b")],
        "moves": [{"session_id": name, "method": "replay", "order": order}
                  for order, name in enumerate(("a", "b"))],
    }
    queue = {**base, "scenario_id": "queue", "policy": "queue_haul"}
    failed = {**base, "scenario_id": "failed", "policy": "random"}
    plan = {
        "episodes": 1, "policies": ["queue_haul", "random"],
        "scenarios": [control, queue, failed],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))

    def write(scenario, result):
        path = tmp_path / "scenarios" / scenario / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))

    write("control", {
        "status": "complete",
        "continuations": [
            {"session_id": name, "start_ns": 0, "first_byte_ns": 100_000_000}
            for name in ("a", "b")
        ],
    })
    write("queue", {
        "status": "complete",
        "migrations": [{
            "queued_ns": 1_000_000_000,
            "initial_start_ns": start,
            "switch_end_ns": first + 500_000_000,
            "move": {"session_id": name, "method": "replay", "order": order},
            "initial": {"first_byte_ns": first},
        } for order, (name, start, first) in enumerate((
            ("a", 2_000_000_000, 3_000_000_000),
            ("b", 4_000_000_000, 5_000_000_000),
        ))],
        "continuations": [
            {"session_id": name, "start_ns": 6_000_000_000,
             "first_byte_ns": 6_200_000_000}
            for name in ("a", "b")
        ],
    })
    write("failed", {"status": "failed"})

    migrations, summaries = reduce_run(tmp_path)
    queue_rows = [row for row in migrations
                  if row["policy"] == "queue_haul"]
    assert [row["reaction_readiness_s"] for row in queue_rows] == [2, 4]
    assert [row["scheduler_wait_s"] for row in queue_rows] == [1, 3]
    assert [row["migration_ttft_s"] for row in queue_rows] == [1, 1]
    assert [row["continuation_ttft_delta_s"] for row in queue_rows] \
        == pytest.approx([.1, .1])
    random = next(row for row in summaries if row["policy"] == "random")
    assert random["planned_migrations"] == 2
    assert random["completed_migrations"] == 0
    x, y = completion_curve(migrations, summaries, "random",
                            "reaction_readiness_s")
    assert not len(x) and not len(y)
