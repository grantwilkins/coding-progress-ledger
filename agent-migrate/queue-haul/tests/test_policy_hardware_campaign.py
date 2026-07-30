"""
Claim:
Every policy consumes the same frozen hardware episode, reaction latency starts
at one policy epoch, failed episodes remain in curve denominators, and the
isolated-fastest baseline independently minimizes each session's migration time.

Plausible wrong implementations:
- Resample sessions or contexts independently for each policy.
- Measure from each migration's own start and hide scheduler wait.
- Let a policy omit sessions or execute migrations in parallel.
- Pick one globally fastest method instead of choosing independently per session.
- Order isolated-fastest sessions by an unselected method or reverse duration.
- Condition completion curves only on successful migrations.
- Pair continuation TTFT with a control from another episode.
- Stretch every timing metric to the campaign deadline instead of its data.
- Compare unpaired policies or aggregate source power once per migration.
"""

import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import policy_hardware_campaign as campaign
from policy_hardware_campaign import (
    EXECUTION_CONTRACT,
    completion_curve,
    make_plan,
    prepare,
    reduce_run,
    validate_policy_plan,
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
    manifest_path = manifest(tmp_path)
    plan = make_plan(manifest_path, episodes=3, sessions=4, seed=7)
    assert plan == make_plan(
        manifest_path, episodes=3, sessions=4, seed=7
    )
    assert plan["execution_contract"] == EXECUTION_CONTRACT
    assert plan["model_profile"]["sha256"]
    assert not Path(plan["model_profile"]["path"]).is_absolute()

    episode_order = [row["episode"] for row in plan["scenarios"]]
    assert sum(
        i == 0 or episode != episode_order[i - 1]
        for i, episode in enumerate(episode_order)
    ) == 3
    for episode in range(3):
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
    queue_moves = [
        move for row in plan["scenarios"] if row["policy"] == "queue_haul"
        for move in row["moves"] if move["method"] == "kv_transfer"
    ]
    assert queue_moves
    assert all(move["planned_rate_limit_bytes_per_s"] > 0
               and move["planned_quiesce_s"] > 0 for move in queue_moves)
    invalid = deepcopy(plan)
    next(row for row in invalid["scenarios"]
         if row["kind"] == "migration")["move_concurrency"] = 2
    with pytest.raises(ValueError, match="sequential"):
        validate_policy_plan(invalid)


def test_isolated_fastest_chooses_per_session_then_orders_by_chosen_duration(
        monkeypatch):
    durations = {
        ("a", "replay"): 5, ("a", "kv_transfer"): 2,
        ("b", "replay"): 1, ("b", "kv_transfer"): 4,
        ("c", "replay"): 3, ("c", "kv_transfer"): 6,
    }
    monkeypatch.setattr(
        campaign, "_duration",
        lambda session, method, *_: durations[session.session_id, method],
    )
    scenario = SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=name)
                       for name in ("a", "b", "c")),
        links=(SimpleNamespace(link_id="link", bytes_per_s=1),),
    )
    profile = SimpleNamespace(case=lambda: object())

    moves = campaign._moves(
        "isolated_fastest", scenario,
        {("source", "destination"): ("link",)}, profile, seed=0,
    )

    assert [(row["session_id"], row["method"], row["order"])
            for row in moves] == [
        ("b", "replay", 0),
        ("a", "kv_transfer", 1),
        ("c", "replay", 2),
    ]


def test_prepared_job_is_self_locating_and_keeps_failures_visible(tmp_path):
    out = tmp_path / "queue-haul/outputs/policy"
    prepare(manifest(tmp_path), out, episodes=1, sessions=4)

    job = (out / "run.sh").read_text()
    assert 'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"' in job
    assert "--stack-scenarios 30" in job
    assert "--fail-fast" not in job
    assert '[[ -f "$QH_POLICY_RUN_ROOT/plan.json" ]]' in job
    assert (out / "run.sbatch").exists()


def test_reduction_uses_common_epoch_and_keeps_failed_denominator(tmp_path):
    control = {
        "scenario_id": "control", "match_id": "same", "episode": 0,
        "policy": "control", "kind": "control", "deadline_s": 10,
        "sessions": [{"session_id": name, "initial_tokens": 4096}
                     for name in ("a", "b")],
    }
    base = {
        **control, "kind": "migration", "move_concurrency": 1,
        "sessions": [{"session_id": name, "initial_tokens": 4096}
                     for name in ("a", "b")],
        "moves": [
            {"session_id": name, "method": method, "order": order}
            for order, (name, method) in enumerate(
                (("a", "replay"), ("b", "kv_transfer"))
            )
        ],
    }
    queue = {**base, "scenario_id": "queue", "policy": "queue_haul"}
    failed = {**base, "scenario_id": "failed", "policy": "random"}
    plan = {
        "episodes": 1, "policies": ["queue_haul", "random"],
        "execution_contract": EXECUTION_CONTRACT,
        "scenarios": [control, queue, failed],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))

    def write(scenario, result):
        path = tmp_path / "scenarios" / scenario / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))

    write("control", {
        "status": "complete", "allocation_id": "job-a",
        "continuations": [
            {"session_id": name, "start_ns": 0, "first_byte_ns": 100_000_000}
            for name in ("a", "b")
        ],
    })
    write("queue", {
        "status": "complete", "allocation_id": "job-a",
        "migrations": [{
            "queued_ns": 1_000_000_000,
            "initial_start_ns": start,
            "initial_end_ns": first + 100_000_000,
            "pause_start_ns": first + 200_000_000,
            "catch_up_start_ns": None, "catch_up_end_ns": None,
            "switch_end_ns": first + 500_000_000,
            "move": {
                "session_id": name,
                "method": "replay" if order == 0 else "kv_transfer",
                "order": order,
            },
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
    assert [row["first_token_s"] for row in queue_rows] == [5.2, 5.2]
    assert [row["continuation_ttft_delta_s"] for row in queue_rows] \
        == pytest.approx([.1, .1])
    assert (tmp_path / "policy_gantt.csv").exists()
    assert (tmp_path / "policy_hardware_gantt.pdf").exists()
    random = next(row for row in summaries if row["policy"] == "random")
    assert random["planned_migrations"] == 2
    assert random["completed_migrations"] == 0
    x, y = completion_curve(migrations, summaries, "random",
                            "reaction_readiness_s")
    assert not len(x) and not len(y)

    queue_result = json.loads(
        (tmp_path / "scenarios/queue/result.json").read_text()
    )
    write("queue", {**queue_result, "allocation_id": "job-b"})
    split_rows, split_summaries = reduce_run(tmp_path)
    assert all(
        math.isnan(row["continuation_ttft_delta_s"])
        for row in split_rows if row["policy"] == "queue_haul"
    )
    assert not next(
        row for row in split_summaries if row["policy"] == "queue_haul"
    )["matched_control_complete"]


def test_plot_pairs_qh_with_greedy_at_metric_and_episode_levels(
        tmp_path, monkeypatch):
    rows = [
        {"policy": policy, "reaction_readiness_s": ready,
         "migration_ttft_s": ttft, "reaction_commit_s": commit}
        for policy, ready, ttft, commit in (
            ("queue_haul", 10, 1, 11), ("greedy", 9, 2, 12),
            ("random", 100, 100, 100),
        )
    ]
    summaries = [
        {"policy": policy, "planned_migrations": 1, "deadline_s": 180,
         "realized_source_power_drop_w": power}
        for policy, power in (
            ("queue_haul", 300), ("greedy", 200), ("random", 400),
        )
    ]
    monkeypatch.setattr(campaign.plt, "close", lambda _: None)

    campaign.plot(rows, summaries, tmp_path)

    figure = campaign.plt.gcf()
    assert [text.get_text() for text in figure.legends[0].texts] == [
        campaign.LABELS["queue_haul"], campaign.LABELS["greedy"],
    ]
    assert len(figure.axes[1].lines) == 2
    assert figure.axes[1].get_xlim()[1] < 5
    assert {
        tuple(line.get_xdata()) for line in figure.axes[3].lines
    } == {(300,), (200,)}
