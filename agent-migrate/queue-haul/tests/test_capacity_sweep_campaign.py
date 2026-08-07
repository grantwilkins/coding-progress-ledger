"""Claims for the fixed-deadline destination-load and goodput sweeps."""

import json
import pytest

from capacity_sweep_campaign import (
    COMMIT_DEADLINE_S,
    CONTEXTS,
    GOODPUT_CAPS_MBPS,
    LOAD_BASE_FRACTIONS,
    adaptive_load_fractions,
    credited_sessions,
    knee_indices,
    make_campaign,
    make_live_plan,
    shapley_watts,
    source_session_rates,
    write_campaign,
)


def test_deadline_credit_requires_continuation_and_route_commit_by_30_seconds():
    rows = [
        {"session_id": "at", "committed_s": 30, "first_token_s": 29},
        {"session_id": "late", "committed_s": 30.001, "first_token_s": 29},
        {"session_id": "no-token", "committed_s": 20, "first_token_s": None},
        {"session_id": "token-late", "committed_s": 20, "first_token_s": 31},
    ]
    assert COMMIT_DEADLINE_S == 30
    assert credited_sessions(rows) == {"at"}


def test_two_group_shapley_exactly_attributes_nonlinear_power_shed():
    values = {frozenset(): 0, frozenset({"replay"}): 7,
              frozenset({"kv"}): 5,
              frozenset({"replay", "kv"}): 15}
    replay, kv = shapley_watts(lambda groups: values[frozenset(groups)])
    assert replay == pytest.approx(8.5)
    assert kv == pytest.approx(6.5)
    assert replay + kv == 15


def test_load_grid_adds_one_lp_crossing_bracket_midpoint():
    assert LOAD_BASE_FRACTIONS == (0, .25, .5, .7, .85, .95)
    assert adaptive_load_fractions([150, 140, 110, 90, 60, 20], 100) == (
        0, .25, .5, .6, .7, .85, .95,
    )
    assert adaptive_load_fractions([150] * 6, 100)[-1] == 1
    assert .125 in adaptive_load_fractions([90] * 6, 100)


def test_knees_bracket_capacity_in_each_sweep_direction():
    assert knee_indices([120, 110, 80, 50], 100, feasible_first=True) == (1, 2)
    assert knee_indices([50, 80, 110, 120], 100, feasible_first=False) == (1, 2)
    assert knee_indices([120] * 4, 100, feasible_first=True) == (2, 3)
    assert knee_indices([50] * 4, 100, feasible_first=False) == (2, 3)


def test_source_is_eight_equal_streams_at_four_rps_and_64_to_1_tokens():
    expected_f, expected_g = source_session_rates(8)
    assert expected_f == 64
    assert expected_g == 1
    assert 8 * expected_f == 4 * 128
    assert 8 * expected_g == 4 * 2


def test_sweep_points_are_fixed_and_inside_profiled_goodput_range():
    assert GOODPUT_CAPS_MBPS == (1000, 1600, 2500, 4000, 5000, 7000, 10000)


def test_live_plan_is_two_knees_four_policies_three_repeats(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    template = {
        "manifest": {"path": str(manifest), "sha256": "hash"},
        "scenarios": [{"sessions": [
            {"session_id": f"m{i}", "job_class": "coding", "turn_index": 0,
             "initial_tokens": context, "order": i}
            for i, context in enumerate(CONTEXTS)
        ]}],
    }
    rows = [{
        "policy": policy, "load_fraction": cell / 4,
        "configured_goodput_mbps": 10_000, "measured_goodput_mbps": 10_000,
        "moves": [{"session_id": "s0", "method": "replay", "order": 0,
                   "destination_instance": "destination", "path": ["link"],
                   "rate_limit_bytes_per_s": None, "quiesce_s": None,
                   "destination_pool": "dedicated-sink"}],
    } for cell in range(4) for policy in
        ("lp", "greedy", "replay_only", "kv_only")]
    campaign = {
        "schema": "queue-haul-capacity-sweep-v1", "campaign": "load",
        "rows": rows, "profile": {"path": "profile", "sha256": "hash"},
        "live_validation": {"lp_knee_indices": [1, 2], "repeats": 3,
                            "policies": ["lp", "greedy", "replay_only", "kv_only"]},
    }
    plan = make_live_plan(campaign, template)
    assert len(plan["scenarios"]) == 24
    assert {row["required_deadline_s"] for row in plan["scenarios"]} == {30}
    assert {row["deadline_s"] for row in plan["scenarios"]} == {180}
    assert all(len(row["sessions"]) == len(row["moves"]) == 1
               for row in plan["scenarios"])
    assert {row["load_fraction"] for row in plan["scenarios"]} == {.25, .5}


def test_generated_campaign_has_positive_target_and_is_json_serializable():
    campaign = make_campaign("goodput")
    assert campaign["rows"][0]["requested_shed_w"] > 0
    assert campaign["rows"][0]["achieved_shed_w"] > 0
    json.dumps(campaign)


def test_goodput_campaign_writes_both_plot_families(tmp_path):
    write_campaign("goodput", tmp_path)
    assert (tmp_path / "goodput_capacity.png").exists()
    assert (tmp_path / "goodput_capacity_stack.png").exists()
