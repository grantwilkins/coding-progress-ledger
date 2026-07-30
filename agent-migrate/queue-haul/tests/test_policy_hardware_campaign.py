"""
Claim:
Every policy consumes the same frozen hardware episode, reaction latency starts
at one policy epoch, all sessions launch concurrently, and failed episodes
remain in deadline-attainment denominators.

Plausible wrong implementations:
- Resample sessions or contexts independently for each policy.
- Measure from each migration's own start and hide scheduler wait.
- Let a policy omit sessions or use a width below the episode size.
- Condition completion curves only on successful migrations.
- Pair continuation TTFT with a control from another episode.
- Stretch every timing metric to the campaign deadline instead of its data.
- Omit the fixed-method controls or aggregate source power once per migration.
- Count commits just after the deadline or linearize the nonlinear power curve.
- Plot destination-only prefill instead of migration-to-first-token latency.
- Average committed-session fractions instead of nonlinear episode power.
- Compute Pareto dominance across unmatched episodes or requested targets.
"""

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import policy_hardware_campaign as campaign
from policy_hardware_campaign import (
    EXECUTION_CONTRACT,
    completion_curve,
    deadline_attainment,
    make_plan,
    pareto_points,
    power_shed_quantiles,
    prepare,
    reduce_run,
    validate_policy_plan,
)


def manifest(tmp_path, sessions=4):
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
        } for i in range(sessions)],
    }))
    return path


def test_plan_pairs_every_policy_on_the_same_complete_episode(
        tmp_path, monkeypatch):
    manifest_path = manifest(tmp_path)
    bandwidths = []
    problem = campaign._problem
    monkeypatch.setattr(
        campaign, "_problem",
        lambda profile, sessions, bandwidth, deadline: (
            bandwidths.append(bandwidth),
            problem(profile, sessions, bandwidth, deadline),
        )[1],
    )
    plan = make_plan(
        manifest_path, episodes=2, sessions=4, seed=7,
        bandwidths_mbps=(5_000, 10_000),
        required_deadlines_s=(30, 45),
    )
    assert plan == make_plan(
        manifest_path, episodes=2, sessions=4, seed=7,
        bandwidths_mbps=(5_000, 10_000),
        required_deadlines_s=(30, 45),
    )
    assert plan["execution_contract"] == EXECUTION_CONTRACT
    assert plan["model_profile"]["sha256"]
    assert not Path(plan["model_profile"]["path"]).is_absolute()

    episode_order = [row["episode"] for row in plan["scenarios"]]
    assert sum(
        i == 0 or episode != episode_order[i - 1]
        for i, episode in enumerate(episode_order)
    ) == 8
    for episode in range(8):
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
        assert all(
            row["move_concurrency"] == len(row["sessions"])
            for row in rows if row["kind"] == "migration"
        )
    samples = {}
    for row in plan["scenarios"]:
        if row["policy"] == "control":
            samples.setdefault(row["sample_id"], set()).add(tuple(
                (session["session_id"], session["initial_tokens"])
                for session in row["sessions"]
            ))
    assert len(samples) == 2
    assert all(len(signatures) == 1 for signatures in samples.values())
    assert {row["bandwidth_mbps"] for row in plan["scenarios"]} \
        == {5_000, 10_000}
    assert set(bandwidths) == {5_000, 10_000}
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
    with pytest.raises(ValueError, match="complete episode"):
        validate_policy_plan(invalid)


def test_fixed_context_pack_preserves_width_and_pairing(tmp_path):
    plan = make_plan(
        manifest(tmp_path, 8), episodes=1, sessions=8, seed=7,
        bandwidths_mbps=(10_000,), required_deadlines_s=(30,),
        context_packs=("small",),
    )

    assert plan["context_packs"] == {"small": [4096] * 8}
    assert plan["workload_profiles"] == []
    assert plan["token_distributions"] == ["fixed"]
    assert len(plan["scenarios"]) == 5
    assert all(row["move_concurrency"] == 8 for row in plan["scenarios"])
    assert all(
        [session["initial_tokens"] for session in row["sessions"]]
        == [4096] * 8 for row in plan["scenarios"]
    )
    with pytest.raises(ValueError, match="context packs"):
        make_plan(
            manifest(tmp_path, 8), episodes=1, sessions=4,
            context_packs=("small",),
        )


def test_policy_appends_unadmitted_sessions_as_fastest_tail(monkeypatch):
    monkeypatch.setattr(campaign, "plan", lambda *_args, **_kwargs:
                        SimpleNamespace(moves=(SimpleNamespace(
                            session_id="a", method="replay", order=0,
                            rate_limit_bytes_per_s=None, quiesce_s=0,
                        ),)))
    monkeypatch.setattr(
        campaign, "_duration",
        lambda session, method, *_: {
            ("b", "replay"): 4, ("b", "kv_transfer"): 2,
            ("c", "replay"): 1, ("c", "kv_transfer"): 3,
        }[session.session_id, method],
    )
    scenario = SimpleNamespace(
        sessions=tuple(SimpleNamespace(session_id=name)
                       for name in ("a", "b", "c")),
        links=(SimpleNamespace(link_id="link", bytes_per_s=1),),
    )
    moves = campaign._moves(
        "queue_haul", scenario, {}, SimpleNamespace(case=lambda: object()), 0,
    )
    assert [(row["session_id"], row["method"], row["deadline_admitted"])
            for row in moves] == [
        ("a", "replay", True),
        ("c", "replay", False),
        ("b", "kv_transfer", False),
    ]


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


def test_prepare_cli_accepts_frozen_model_profile(tmp_path):
    args = campaign.parse_args([
        "prepare", "--out", str(tmp_path),
        "--model-profile", str(campaign.DEFAULT_MODEL),
    ])
    assert args.model_profile == campaign.DEFAULT_MODEL


def test_prepared_job_is_self_locating_and_keeps_failures_visible(tmp_path):
    out = tmp_path / "queue-haul/outputs/policy"
    prepare(manifest(tmp_path), out, episodes=1, sessions=4)

    job = (out / "run.sh").read_text()
    assert 'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"' in job
    assert "--stack-scenarios 30" in job
    assert "run=(" in job and '"${run[@]}"' in job
    assert "--fail-fast" not in job
    assert '[[ -f "$QH_POLICY_RUN_ROOT/plan.json" ]]' in job
    assert (out / "run.sbatch").exists()
    sbatch = (out / "run.sbatch").read_text()
    assert "module load gcc/14.2.0 openblas/0.3.28 uv/0.8.4" in sbatch
    assert "50e98f65de09ebfe196f270c8b5c595636853646eb5536dca92f27bd45c084ab" in sbatch
    assert "QH_PORT_OFFSET" in sbatch
    assert "scontrol show job" in sbatch


def test_reduction_uses_common_epoch_and_keeps_failed_denominator(tmp_path):
    control = {
        "scenario_id": "control", "match_id": "same", "episode": 0,
        "policy": "control", "kind": "control", "deadline_s": 10,
        "condition": "coding-uniform_support-10s",
        "context_profile": "coding",
        "token_distribution": "uniform_support", "required_deadline_s": 10,
        "power_target_fraction": 1,
        "sessions": [{"session_id": name, "initial_tokens": 4096}
                     for name in ("a", "b")],
    }
    base = {
        **control, "kind": "migration", "move_concurrency": 2,
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
        "power_target_fraction": 1,
        "model_profile": {
            "path": "queue-haul/profiles/gpt_oss_20b_a100_tp1.json",
            "sha256": campaign.profiler.file_hash(campaign.DEFAULT_MODEL),
        },
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
    assert (tmp_path / "policy_hardware_destination_ttft_cdf.pdf").exists()
    assert (tmp_path / "policy_hardware_power_shed_over_time.pdf").exists()
    assert (tmp_path / "policy_hardware_measured_pareto.pdf").exists()
    random = next(row for row in summaries if row["policy"] == "random")
    assert random["planned_migrations"] == 2
    assert random["completed_migrations"] == 0
    x, y = completion_curve(migrations, summaries, "random",
                            "reaction_readiness_s")
    assert not len(x) and not len(y)
    attainment = list(csv.DictReader(
        (tmp_path / "policy_attainment.csv").open()
    ))
    assert {
        row["policy"]: float(row["power_attainment_fraction"])
        for row in attainment
    } == {"queue_haul": 1, "random": 0}

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
            ("kv_only", 11, 3, 14), ("replay_only", 12, 4, 16),
            ("random", 100, 100, 100),
        )
    ]
    summaries = [
        {"policy": policy, "planned_migrations": 1, "deadline_s": 180,
         "realized_source_power_drop_w": power}
        for policy, power in (
            ("queue_haul", 300), ("greedy", 200),
            ("kv_only", 250), ("replay_only", 150), ("random", 400),
        )
    ]
    monkeypatch.setattr(campaign.plt, "close", lambda _: None)

    campaign.plot(rows, summaries, tmp_path)

    figure = campaign.plt.gcf()
    assert [text.get_text() for text in figure.legends[0].texts] == [
        campaign.LABELS["queue_haul"], campaign.LABELS["greedy"],
        campaign.LABELS["kv_only"], campaign.LABELS["replay_only"],
    ]
    assert len(figure.axes[1].lines) == 4
    assert figure.axes[1].get_xlim()[1] < 5


def test_deadline_attainment_uses_episode_target_and_inclusive_deadline():
    class QuadraticPower:
        @staticmethod
        def power(load):
            return 100 + 100 * load ** 2

    rows = deadline_attainment(
        [5, 10], 4, [6, 9, 10.5, 11], QuadraticPower(),
        power_window_s=1,
    )

    assert [row["committed_by_deadline"] for row in rows] == [1, 1, 2, 2]
    assert [row["committed_before_power_window"] for row in rows] \
        == [1, 1, 1, 2]
    assert [row["power_attainment_fraction"] for row in rows] \
        == pytest.approx([.4375, .4375, .59375, .75])


def test_destination_ttft_cdf_includes_migration_time(tmp_path, monkeypatch):
    rows = [
        {"scenario_id": "a", "policy": "queue_haul",
         "migration_ttft_s": value, "continuation_ttft_s": .1}
        for value in (3, 5)
    ]
    summaries = [{
        "scenario_id": "a", "policy": "queue_haul",
        "planned_migrations": 2,
    }]
    monkeypatch.setattr(campaign.plt, "close", lambda _: None)

    campaign.plot_destination_ttft(rows, summaries, tmp_path)

    assert campaign.plt.gcf().axes[0].lines[0].get_xdata().tolist() \
        == [0, 3, 5]


def test_power_shed_curve_uses_nonlinear_episode_power_and_keeps_failures():
    class QuadraticPower:
        @staticmethod
        def power(load):
            return 100 + 100 * load ** 2

    rows = [
        {"scenario_id": "complete", "policy": "queue_haul",
         "reaction_commit_s": value}
        for value in (1, 3)
    ]
    summaries = [
        {"scenario_id": scenario, "policy": "queue_haul",
         "planned_migrations": 4}
        for scenario in ("complete", "failed")
    ]

    low, median, high = power_shed_quantiles(
        rows, summaries, "queue_haul", QuadraticPower(),
        np.asarray([0, 1, 2, 3]),
    )

    assert low == pytest.approx([0, 10.9375, 10.9375, 18.75])
    assert median == pytest.approx([0, 21.875, 21.875, 37.5])
    assert high == pytest.approx([0, 32.8125, 32.8125, 56.25])


def test_pareto_points_use_achieved_shed_and_only_matched_peers():
    attainment = [
        {"scenario_id": scenario, "power_attainment_fraction": shed}
        for scenario, shed in (("q", .8), ("g", .5), ("k", .6), ("r", .1))
    ]
    summaries = [
        {"scenario_id": scenario, "match_id": match, "policy": policy,
         "commit_100_s": commit, "required_deadline_s": 10,
         "deadline_s": 100}
        for scenario, match, policy, commit in (
            ("q", "a", "queue_haul", 8),
            ("g", "a", "greedy", 9),
            ("k", "a", "kv_only", 7),
            ("r", "b", "replay_only", 5),
        )
    ]

    points = {
        row["policy"]: row for row in pareto_points(attainment, summaries)
    }

    assert points["queue_haul"]["shed_percent"] == 80
    assert points["queue_haul"]["deadline_fraction"] == .8
    assert points["queue_haul"]["pareto"]
    assert not points["greedy"]["pareto"]
    assert points["kv_only"]["pareto"]
    assert points["replay_only"]["pareto"]
