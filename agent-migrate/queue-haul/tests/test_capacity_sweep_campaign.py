"""Claims for the fixed-deadline destination-load and goodput sweeps."""

import json
import pytest
import capacity_sweep_campaign as capacity

from capacity_sweep_campaign import (
    COMMIT_DEADLINE_S,
    CONTEXTS,
    FULL_DRAIN_BANDWIDTHS_MBPS,
    FULL_DRAIN_LOADS,
    GOODPUT_CAPS_MBPS,
    LIVE_REPEATS,
    LOAD_BASE_FRACTIONS,
    adaptive_load_fractions,
    arrival_trace,
    credited_sessions,
    full_drain_times,
    knee_indices,
    make_campaign,
    make_full_drain_campaign,
    make_full_drain_plan,
    make_live_plan,
    median_ci,
    phase2a_schedule,
    phase2b_schedule,
    shapley_watts,
    source_session_rates,
    write_campaign,
    validate_live_rows,
    validate_full_drain_rows,
    write_live_results,
)


POLICIES = ("lp", "greedy", "replay_only", "kv_only")


def live_rows(values, repeats=range(10)):
    rows = []
    for load, policies in values.items():
        selected = repeats[load] if isinstance(repeats, dict) else repeats
        for repeat in selected:
            trace = f"{load}-{repeat}"
            for policy in POLICIES:
                value = policies[policy]
                rows.append({
                    "scenario_id": f"{load}-{repeat}-{policy}",
                    "load_fraction": load, "repeat": repeat,
                    "policy": policy, "trace_id": trace,
                    "offered_rho": load, "offered_rho_prefill": load * .75,
                    "offered_rho_decode": load * .25,
                    "requested_shed_w": 100,
                    "achieved_shed_w": value[repeat % len(value)],
                    "replay_w": 40, "kv_w": 30,
                    "unmet_w": 30, "credited_sessions": 4,
                    "right_censored": False,
                })
    return rows


def test_deadline_credit_requires_continuation_and_route_commit_by_30_seconds():
    rows = [
        {"session_id": "at", "committed_s": 30, "first_token_s": 29},
        {"session_id": "late", "committed_s": 30.001, "first_token_s": 29},
        {"session_id": "no-token", "committed_s": 20, "first_token_s": None},
        {"session_id": "token-late", "committed_s": 20, "first_token_s": 31},
    ]
    assert COMMIT_DEADLINE_S == 30
    assert credited_sessions(rows) == {"at"}


def test_full_drain_time_is_the_later_of_last_commit_and_last_token():
    assert full_drain_times([
        {"committed_s": 32, "first_token_s": 31},
        {"committed_s": 29, "first_token_s": 35},
    ]) == (32, 35, 35)


def test_two_group_shapley_exactly_attributes_nonlinear_power_shed():
    values = {frozenset(): 0, frozenset({"replay"}): 7,
              frozenset({"kv"}): 5,
              frozenset({"replay", "kv"}): 15}
    replay, kv = shapley_watts(lambda groups: values[frozenset(groups)])
    assert replay == pytest.approx(8.5)
    assert kv == pytest.approx(6.5)
    assert replay + kv == 15


def test_load_grid_is_dense_below_saturation_and_refines_rapid_changes():
    assert LOAD_BASE_FRACTIONS == (
        0, .25, .5, .65, .75, .8, .85, .875, .9, .925, .95, .975)
    watts = [150] * len(LOAD_BASE_FRACTIONS); watts[6:] = [100] * 6
    assert .825 in adaptive_load_fractions(watts, 147.2)
    assert adaptive_load_fractions([150] * 12, 147.2) == LOAD_BASE_FRACTIONS


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


def test_live_plan_is_dense_common_trace_matrix_with_complete_source_set(tmp_path):
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
        "calibration": {"service_calibration": {
            "prefill_s": .3, "decode_s": .1, "total_s": .4,
        }},
        "live_validation": {"lp_knee_indices": [1, 2], "repeats": LIVE_REPEATS,
                            "policies": ["lp", "greedy", "replay_only", "kv_only"]},
    }
    plan = make_live_plan(campaign, template)
    assert len(plan["scenarios"]) == 4 * 4 * LIVE_REPEATS
    assert {row["required_deadline_s"] for row in plan["scenarios"]} == {30}
    assert {row["deadline_s"] for row in plan["scenarios"]} == {180}
    assert all(len(row["sessions"]) == 8 and len(row["moves"]) == 1
               and row["allow_partial_moves"] for row in plan["scenarios"])
    assert {row["load_fraction"] for row in plan["scenarios"]} == {0, .25, .5, .75}
    for load in (0, .25, .5, .75):
        for repeat in range(LIVE_REPEATS):
            assert len({row["arrival_trace"]["trace_id"] for row in plan["scenarios"]
                        if row["load_fraction"] == load
                        and row["repeat"] == repeat}) == 1


def test_trace_rho_is_scheduled_service_work_and_median_ci_is_exact_for_constants():
    calibration = {"service_calibration": {
        "prefill_s": .3, "decode_s": .1, "total_s": .4}}
    trace = arrival_trace(.8, 0, calibration)
    assert trace["rho"] == pytest.approx(trace["rho_prefill"] + trace["rho_decode"])
    assert trace["rho"] == pytest.approx(.8, abs=.4 / 30)
    assert median_ci([7] * 10) == (7, 7, 7)


def test_full_drain_plan_forces_all_eight_with_long_paired_traces(tmp_path):
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
    campaign = make_full_drain_campaign(2500)
    plan = make_full_drain_plan(campaign, template)
    assert FULL_DRAIN_BANDWIDTHS_MBPS == (1000, 2500, 5000, 10000)
    assert FULL_DRAIN_LOADS == (
        .85, .875, .8875, .9, .9125, .925, .9375, .95, .9625, .975)
    assert len(plan["scenarios"]) == len(FULL_DRAIN_LOADS) * 2 * LIVE_REPEATS
    assert {row["bandwidth_mbps"] for row in plan["scenarios"]} == {2500}
    assert all(len(row["sessions"]) == len(row["moves"]) == 8
               and not row["allow_partial_moves"] for row in plan["scenarios"])
    assert {row["policy"] for row in plan["scenarios"]} == {
        "replay_only", "kv_only"}
    for load in FULL_DRAIN_LOADS:
        for repeat in range(LIVE_REPEATS):
            cell = [row for row in plan["scenarios"]
                    if row["load_fraction"] == load and row["repeat"] == repeat]
            assert len({row["arrival_trace"]["trace_id"] for row in cell}) == 1
            assert max(cell[0]["arrival_trace"]["offsets_s"]) > 180


def test_full_drain_validation_requires_paired_complete_cells():
    rows = [{
        "scenario_id": policy, "configured_goodput_mbps": 1000,
        "load_fraction": .9, "repeat": 0, "policy": policy,
        "trace_id": "shared", "planned_sessions": 8, "full_drain_s": 31,
    } for policy in ("replay_only", "kv_only")]
    validate_full_drain_rows(rows)
    rows[0]["planned_sessions"] = 7
    with pytest.raises(RuntimeError, match="eight"):
        validate_full_drain_rows(rows)


def test_generated_campaign_has_positive_target_and_is_json_serializable():
    campaign = make_campaign("goodput")
    assert campaign["rows"][0]["requested_shed_w"] > 0
    assert campaign["rows"][0]["achieved_shed_w"] > 0
    json.dumps(campaign)


def test_goodput_campaign_writes_both_plot_families(tmp_path):
    write_campaign("goodput", tmp_path)
    assert (tmp_path / "goodput_capacity.png").exists()
    assert (tmp_path / "goodput_capacity_stack.png").exists()

def test_incremental_trace_phases_preserve_base_and_support_30_repeats():
    calibration = {"service_calibration": {
        "prefill_s": .3, "decode_s": .1, "total_s": .4}}
    traces = [arrival_trace(.8, repeat, calibration) for repeat in range(30)]
    assert traces[0]["offsets_s"][0] == pytest.approx(.025)
    assert traces[10]["offsets_s"][0] == pytest.approx(1 / 24)
    assert traces[20]["offsets_s"][0] == pytest.approx(7 / 120)
    assert len({row["trace_id"] for row in traces}) == 30
    assert len({row["offsets_s"][0] for row in traces}) == 30
    with pytest.raises(ValueError, match="repeat"):
        arrival_trace(.8, 30, calibration)


def base_values(lp):
    return {
        load: {
            "lp": [lp(load)], "greedy": [100 if load <= .9 else 90],
            "replay_only": [70], "kv_only": (
                [70] * 6 + [100] * 4 if load == .5 else [70]),
        }
        for load in LOAD_BASE_FRACTIONS
    }


def test_phase2a_unions_knee_wide_cells_and_policy_midpoints():
    rows = live_rows(base_values(lambda load: 100 if load <= .8 else 90))
    schedule, selection = phase2a_schedule(rows)
    assert selection["knee_loads"] == [.8, .85]
    assert selection["wide_loads"] == [.5]
    assert selection["midpoint_loads"] == [.825, .9125000000000001]
    assert schedule[.5] == schedule[.8] == schedule[.85] == tuple(range(10, 20))
    assert schedule[.825] == tuple(range(10))
    assert schedule[.9125000000000001] == tuple(range(10))


@pytest.mark.parametrize("lp", [
    lambda load: 90,
    lambda load: 100,
])
def test_phase2a_hard_fails_when_base_does_not_bracket_full_shed(lp):
    with pytest.raises(ValueError, match="bracket"):
        phase2a_schedule(live_rows(base_values(lp)))


def test_phase2b_only_extends_combined_wide_cells_with_20_29():
    repeats = {.5: range(20), .8: range(20), .825: range(10)}
    values = {
        .5: {policy: [90] for policy in POLICIES},
        .8: {policy: ([80] * 10 + [100] * 10
                      if policy == "lp" else [90]) for policy in POLICIES},
        .825: {policy: ([80] * 5 + [100] * 5
                        if policy == "greedy" else [90]) for policy in POLICIES},
    }
    assert phase2b_schedule(live_rows(values, repeats)) == {
        .8: tuple(range(20, 30)), .825: tuple(range(20, 30))}


def adaptive_fixture(tmp_path):
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
        "policy": policy, "load_fraction": load,
        "configured_goodput_mbps": 10_000, "measured_goodput_mbps": 10_000,
        "moves": [{"session_id": "s0", "method": "replay", "order": 0,
                   "destination_instance": "destination", "path": ["link"],
                   "rate_limit_bytes_per_s": None, "quiesce_s": None,
                   "destination_pool": "dedicated-sink"}],
    } for load in (.8, .825) for policy in POLICIES]
    campaign = {
        "schema": "queue-haul-capacity-sweep-v1", "campaign": "load",
        "rows": rows, "profile": {"path": "profile", "sha256": "hash"},
        "calibration": {"service_calibration": {
            "prefill_s": .3, "decode_s": .1, "total_s": .4}},
        "live_validation": {"lp_knee_indices": [0, 1], "repeats": 10,
                            "policies": list(POLICIES)},
    }
    return campaign, template


def test_adaptive_plan_uses_exact_variable_schedule_and_complete_source_set(tmp_path):
    campaign, template = adaptive_fixture(tmp_path)
    schedule = {.8: tuple(range(10, 20)), .825: tuple(range(10))}
    plan = make_live_plan(campaign, template, schedule, "phase2a", ["prior"])
    assert len(plan["scenarios"]) == 80
    assert plan["adaptive_phase"] == "phase2a"
    assert plan["prior_plan_sha256"] == ["prior"]
    assert plan["load_repeat_schedule"] == [
        {"load_fraction": .8, "repeats": list(range(10, 20))},
        {"load_fraction": .825, "repeats": list(range(10))},
    ]
    assert all(len(row["sessions"]) == 8 for row in plan["scenarios"])
    assert {(row["load_fraction"], row["repeat"], row["policy"])
            for row in plan["scenarios"]} == {
        (load, repeat, policy)
        for load, repeats in schedule.items()
        for repeat in repeats for policy in POLICIES}
    for load, repeats in schedule.items():
        for repeat in repeats:
            assert len({row["arrival_trace"]["trace_id"] for row in plan["scenarios"]
                        if row["load_fraction"] == load
                        and row["repeat"] == repeat}) == 1


def test_live_row_validation_accepts_variable_repeats():
    repeats = {.8: range(20), .825: (*range(10), *range(20, 30))}
    values = {load: {policy: [90] for policy in POLICIES} for load in repeats}
    validate_live_rows(live_rows(values, repeats))


@pytest.mark.parametrize("fault", ["duplicate", "missing", "trace"])
def test_live_row_validation_rejects_invalid_common_trace_matrix(fault):
    rows = live_rows({.8: {policy: [90] for policy in POLICIES}}, range(1))
    if fault == "duplicate":
        rows.append(dict(rows[0]))
    elif fault == "missing":
        rows.pop()
    else:
        rows[0]["trace_id"] = "wrong"
    with pytest.raises(RuntimeError):
        validate_live_rows(rows)


def test_merge_rejects_overlapping_phase_results():
    rows = live_rows({.8: {policy: [90] for policy in POLICIES}}, range(1))
    with pytest.raises(RuntimeError, match="duplicate"):
        capacity.merge_live_rows([rows, [dict(rows[0])]])


def test_final_summary_reports_variable_repeats_and_writes_figures(tmp_path):
    repeats = {.8: range(20), .825: (*range(10), *range(20, 30))}
    values = {load: {policy: [90] for policy in POLICIES} for load in repeats}
    write_live_results(live_rows(values, repeats), tmp_path, "load")
    summary = json.loads((tmp_path / "live_summary.json").read_text())
    assert summary["repeats_by_load"] == {"0.8": 20, "0.825": 20}
    assert summary["min_repeats_per_cell"] == summary["max_repeats_per_cell"] == 20
    for name in ("live_capacity.csv", "load_capacity_live.png",
                 "load_capacity_live.pdf", "load_capacity_components_live.png",
                 "load_capacity_components_live.pdf"):
        assert (tmp_path / name).exists()
