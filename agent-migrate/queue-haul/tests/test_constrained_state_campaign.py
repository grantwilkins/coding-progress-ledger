"""The constrained-state campaign keeps discovery, census, and execution frozen."""

import itertools
import json

import pytest

import constrained_state_campaign as campaign


def axis_row(axis, count, **values):
    return {"axis": axis, "count": count, "warmup_s": 30,
            "measurement_s": 30, "telemetry_complete": True,
            **telemetry(), **values}

def contract():
    return {
        "residual_tolerances": dict.fromkeys(campaign.RESOURCES, .1),
        "min_telemetry_coverage": .95,
        "max_scheduled_load_error": .05,
        "min_samples": 2,
        "queue_growth_tolerance": .1,
        "prefill_throughput_tolerance": 1,
        "min_power_samples": 100,
        "max_power_gap_s": .5,
    }


def telemetry():
    return {
        "measurement_id": "measurement",
        "telemetry_sha256": "0" * 64,
        "measurement_started_ns": 1_000_000_000,
        "measurement_ended_ns": 31_000_000_000,
        "sample_count": 30,
        "telemetry_coverage": 1,
        "scheduled_load_error": 0,
    }



def frozen_session(session_id):
    return {
        "session_id": session_id,
        "request_sha256": campaign.digest(["request", session_id]),
        "continuation_sha256": campaign.digest(["continuation", session_id]),
    }


def completed(session_id, action, completion_s):
    return {
        **frozen_session(session_id), "action": action,
        "completion_s": completion_s, "reconstruction_verified": True,
        "continuation_verified": True,
        "reconstruction_evidence_sha256": "2" * 64,
    }


def test_discovery_retains_first_invalid_point_and_returns_last_valid_counts():
    rows = [
        axis_row("hbm", 1, rejected=False, evicted=False,
                 persistently_resident=True),
        axis_row("hbm", 2, rejected=True, evicted=False,
                 persistently_resident=False),
        axis_row("serving", 1, slo_met=True, queue_growth=0),
        axis_row("serving", 2, slo_met=False, queue_growth=1),
        axis_row("prefill", 1, queue_growth=0, completed_throughput=10,
                 scheduled_throughput=10),
        axis_row("prefill", 2, queue_growth=0, completed_throughput=10,
                 scheduled_throughput=11),
        axis_row("prefill", 3, queue_growth=0, completed_throughput=10,
                 scheduled_throughput=13),
    ]

    limits, retained = campaign.discovery_limits(rows, contract())

    assert limits == {"hbm": 1, "serving": 1, "prefill": 2}
    assert [row["valid"] for row in retained] == [True, False, True, False,
                                                   True, True, False]
    with pytest.raises(ValueError, match="first invalid"):
        campaign.discovery_limits(rows[:-1], contract())

    rows[0]["sample_count"] = 1
    with pytest.raises(ValueError, match="discovery telemetry"):
        campaign.discovery_limits(rows, contract())


def test_candidate_population_contains_axes_and_seeded_integer_sobol_sample():
    limits = {"hbm": 3, "serving": 2, "prefill": 4}

    first = campaign.candidate_counts(limits, seed=17)
    second = campaign.candidate_counts(limits, seed=17)

    assert first == second
    assert len(first) == len(set(first))
    assert set(itertools.chain(
        ((value, 0, 0) for value in range(4)),
        ((0, value, 0) for value in range(3)),
        ((0, 0, value) for value in range(5)),
    )) <= set(first)
    assert all(0 <= hbm <= 3 and 0 <= serving <= 2 and 0 <= prefill <= 4
               for hbm, serving, prefill in first)


def small_problem():
    pack = {"pack_id": "pack", "sessions": [
        frozen_session("a"), frozen_session("b"),
    ], "power_gains": [0.0, 0.4, 0.4, 2.0]}
    demands = [{
        "pack_id": "pack", "session_id": session, "action": action,
        "duration_s": 2 if action == "kv_transfer" else 1,
        "d_wan": 1 if action == "kv_transfer" else 0,
        "d_service": 0 if action == "kv_transfer" else 1,
        "d_prefill": 0 if action == "kv_transfer" else 1,
        "d_hbm": 1,
    } for session in ("a", "b") for action in campaign.ACTIONS]
    return pack, demands


def timing(pack, moves, capacity):
    """Tiny deterministic stand-in for the production simulator adapter."""
    durations = [
        (2 / capacity["wan"] if move["action"] == "kv_transfer"
         else 1 / capacity["prefill"])
        for move in moves
    ]
    return {"makespan_s": sum(durations),
            "completion_s": dict(zip(
                (move["session_id"] for move in moves),
                itertools.accumulate(durations)))}
timing.identity = {"name": "test-simulator", "version": 1,
                   "source_sha256": campaign.callable_sha256(timing)}



@pytest.mark.parametrize(("capacity", "expected"), (
    ((2, 2, 2, 2), "Slack"),
    ((2, 0, 0, 2), "KV-only feasible"),
    ((0, 2, 2, 2), "Replay-only feasible"),
    ((1, 1, 1, 2), "Mixed actions required"),
    ((1, 1, 1, 1), "Full drain infeasible"),
))
def test_oracle_assigns_exact_action_classes(capacity, expected):
    pack, demands = small_problem()

    row = campaign.oracle_case(pack, demands, dict(zip(campaign.RESOURCES,
                                                       capacity)), 2.0,
                               dict.fromkeys(campaign.RESOURCES, 2.0), timing)

    assert row["oracle_class"] == expected
    assert row["j"] == (expected != "Full drain infeasible")
    assert row["k"] == (expected in {"Slack", "KV-only feasible"})
    assert row["r"] == (expected in {"Slack", "Replay-only feasible"})
    if expected == "Full drain infeasible":
        assert row["delta_hbm"] == pytest.approx(1.6)


def test_per_session_greedy_uses_state_but_ignores_aggregate_contention():
    pack, demands = small_problem()

    slow_wan = campaign.per_session_greedy(
        pack, demands, dict.fromkeys(campaign.RESOURCES, 2), timing)
    fast_wan = campaign.per_session_greedy(
        pack, demands, {"wan": 10, "prefill": .5, "service": 2, "hbm": 2},
        timing)

    assert slow_wan == [
        {"session_id": "a", "action": "replay", "dispatch": True},
        {"session_id": "b", "action": "replay", "dispatch": True},
    ]
    assert {move["action"] for move in fast_wan} == {"kv_transfer"}


def test_oracle_uses_scheduled_feasibility_and_reports_static_discordance():
    pack, demands = small_problem()

    row = campaign.oracle_case(
        pack, demands, dict.fromkeys(campaign.RESOURCES, 2), 2,
        dict.fromkeys(campaign.RESOURCES, 2),
        lambda pack, moves, capacity: {"makespan_s": 30 if moves else 0,
                                      "completion_s": {}},
    )

    assert row["static_p_star"] == 2
    assert row["p_star"] == 0
    assert row["static_scheduled_discordant"] is True


def test_oracle_records_all_counterfactuals_and_minimal_joint_improvement():
    pack, demands = small_problem()
    capacity = {"wan": .5, "service": 2, "prefill": .5, "hbm": 2}

    row = campaign.oracle_case(
        pack, demands, capacity, 2, dict.fromkeys(campaign.RESOURCES, 2), timing)
    values = json.loads(row["relaxation_values"])

    assert len(values) == 16
    assert json.loads(row["minimal_improving_subsets"])
    assert values["none"] == row["p_star"]


def test_selection_is_per_class_uniform_and_execution_crosses_all_packs():
    census = [{"state_id": f"{name}-{index}", "state_hash": f"h-{name}-{index}",
               "oracle_class": name}
              for name in campaign.ORACLE_CLASSES for index in range(7)]

    selected = campaign.select_states(census, seed=9)
    packs = [{"pack_id": f"pack-{index}", "sha256": f"p-{index}"}
             for index in range(10)]
    schedule = campaign.execution_schedule(selected, packs, seed=9,
                                            inputs_hash="inputs")

    assert all(len(selected["draws"][name]) == 6
               for name in campaign.ORACLE_CLASSES)
    assert not selected["shortages"] and len(selected["states"]) == 30
    assert all(0 < row["inclusion_probability"] <= 1
               and row["analysis_weight"] == pytest.approx(
                   1 / row["inclusion_probability"])
               for row in selected["states"])
    assert set(selected["strata"]) == set(campaign.ORACLE_CLASSES)
    assert len(schedule) == 30 * 10 * 5 * 2
    for block_id in {row["block_id"] for row in schedule}:
        block = [row for row in schedule if row["block_id"] == block_id]
        assert {row["policy"] for row in block} == set(campaign.POLICIES)
        assert sorted(row["policy_order"] for row in block) == list(range(5))


def test_selection_reports_a_class_shortage_without_fabricating_states():
    census = [{"state_id": f"state-{index}", "state_hash": f"hash-{index}",
               "oracle_class": "Slack"} for index in range(2)]

    selected = campaign.select_states(census, seed=1)

    assert selected["shortages"] == {
        "Slack": 4,
        "KV-only feasible": 6,
        "Replay-only feasible": 6,
        "Mixed actions required": 6,
        "Full drain infeasible": 6,
    }
    assert [row["state_id"] for row in selected["states"]] == [
        "state-0", "state-1"]


def test_trailing_window_attainment_uses_completion_times():
    pack, _ = small_problem()
    completions = [(20.0, "a"), (25.0, "b")]

    assert campaign.modeled_window_relief(pack, completions, 30) == pytest.approx(2)
    assert campaign.modeled_target_time(pack, completions, 2) == pytest.approx(30)
    assert campaign.modeled_window_relief(
        pack, [(20, "a"), (29, "b")], 30) == pytest.approx(.72)


def test_background_validation_hard_fails_telemetry_and_rejects_ingest_limit():
    state = {"state_id": "state", "n_hbm": 0, "n_serving": 0,
             "n_prefill": 0, "wan_setting": "natural"}
    row = {**state, "warmup_s": 30, "measurement_s": 30,
           "telemetry_complete": True, "background_valid": True,
           "kv_ingest_limiting": False,
           **{f"b_{name}": 2 for name in campaign.RESOURCES},
           **telemetry()}

    valid = campaign.validate_background_states([row], [state], contract())

    assert valid[0]["valid"] and len(valid[0]["state_hash"]) == 64
    with pytest.raises(ValueError, match="telemetry"):
        campaign.validate_background_states(
            [{key: value for key, value in row.items() if key != "b_hbm"}],
            [state], contract())
    loaded = {**state, "state_id": "loaded", "n_hbm": 1}
    rejected = campaign.validate_background_states(
        [row, {**row, **loaded, "kv_ingest_limiting": True}],
        [state, loaded], contract())
    assert next(value for value in rejected if value["state_id"] == "loaded")["valid"] is False

    with pytest.raises(ValueError, match="exceeds empty reference"):
        campaign.validate_background_states(
            [row, {**row, **loaded, "b_hbm": 2.2}], [state, loaded], contract())


def test_census_crosses_valid_states_and_packs_only():
    pack, demands = small_problem()
    background = [{"state_id": "valid", "state_hash": "v",
                   "wan_setting": "natural", "valid": True,
                   **{f"b_{name}": 2 for name in campaign.RESOURCES}},
                  {"state_id": "invalid", "state_hash": "i",
                   "wan_setting": "natural", "valid": False,
                   **{f"b_{name}": 0 for name in campaign.RESOURCES}}]

    rows = campaign.oracle_census(background, [pack], demands, 2, timing)

    assert len(rows) == 1
    assert rows[0]["state_id"] == "valid"
    assert rows[0]["pack_id"] == "pack"
    assert rows[0]["oracle_class"] == "Slack"


def test_census_requires_an_empty_capacity_measurement_per_wan_setting():
    pack, demands = small_problem()
    background = [{"state_id": "loaded", "state_hash": "hash",
                   "wan_setting": "natural", "valid": True,
                   "n_hbm": 1, "n_serving": 0, "n_prefill": 0,
                   **{f"b_{name}": 1 for name in campaign.RESOURCES}}]

    with pytest.raises(ValueError, match="empty-destination"):
        campaign.oracle_census(background, [pack], demands, 2, timing)


def full_inputs():
    packs = [{"pack_id": f"pack-{pack}",
              "sessions": [frozen_session(f"{pack}-{index}")
                           for index in range(8)],
              "power_gains": [float(mask) for mask in range(256)]}
             for pack in range(10)]
    demands = [{"pack_id": pack["pack_id"],
                "session_id": session["session_id"], "action": action,
                **{f"d_{name}": 1 for name in campaign.RESOURCES}}
               for pack in packs for session in pack["sessions"]
               for action in campaign.ACTIONS]
    return {"packs": packs,
            "profile": {"model": "gpt-oss", "gpu": "A100"},
            "action_demands": demands, "target": 255,
            "wan_settings": {name: {} for name in campaign.WAN_SETTINGS},
            "background_manifest": {"seed": 3, "sessions": []},
            "timing_model": dict(timing.identity),
            "measurement_contract": contract()}


def test_freeze_inputs_hashes_and_validates_the_complete_contract():
    inputs = full_inputs()
    packs = inputs["packs"]

    first = campaign.freeze_inputs(inputs, {"sobol": 7, "selection": 9})
    second = campaign.freeze_inputs(inputs, {"sobol": 7, "selection": 9})

    assert first == second
    assert set(first["hashes"]) == set(inputs) | {"policies"}
    assert len(first["inputs_hash"]) == 64
    selected = {"states": [{"state_id": "state", "sha256": "state-hash",
                            "inclusion_probability": 1,
                            "analysis_weight": 1}]}
    schedule = campaign.execution_schedule(
        selected, packs, seed=9, inputs_hash=first["inputs_hash"])
    assert schedule[0]["pack_hash"] == campaign.digest(packs[0])
    assert first["constants"] == {"D_s": 30, "W_s": 5, "T_mig_s": 25}
    with pytest.raises(ValueError, match="ten eight-session packs"):
        campaign.freeze_inputs({**inputs, "packs": packs[:-1]},
                               {"sobol": 7, "selection": 9})

def test_timing_identity_binds_the_callable_source():
    frozen = campaign.freeze_inputs(
        full_inputs(), {"sobol": 7, "selection": 9})

    def imposter(pack, moves, capacity):
        return timing(pack, moves, capacity)
    imposter.identity = timing.identity

    with pytest.raises(ValueError, match="source hash"):
        campaign.verify_timing(imposter, frozen)


def test_freeze_rejects_nonmonotone_power_gain_lattice():
    inputs = full_inputs()
    inputs["packs"][0]["power_gains"][3] = 0

    with pytest.raises(ValueError, match="monotone"):
        campaign.freeze_inputs(inputs, {"sobol": 7, "selection": 9})



def test_compile_campaign_writes_a_frozen_pre_execution_bundle(tmp_path):
    frozen = campaign.freeze_inputs(
        full_inputs(), {"sobol": 7, "selection": 9})
    discovery = [
        axis_row("hbm", 1, rejected=True, evicted=False,
                 persistently_resident=False),
        axis_row("serving", 1, slo_met=False, queue_growth=1),
        axis_row("prefill", 1, queue_growth=0, completed_throughput=0,
                 scheduled_throughput=2),
    ]
    candidates = campaign.candidate_states(
        {"hbm": 0, "serving": 0, "prefill": 0}, 7)
    measured = [{
        **state, "warmup_s": 30, "measurement_s": 30,
        "telemetry_complete": True, "background_valid": True,
        "kv_ingest_limiting": False,
        **{f"b_{name}": 8 for name in campaign.RESOURCES},
        **telemetry(),
    } for state in candidates]

    out = tmp_path / "compiled"
    compiled = campaign.compile_campaign(
        frozen, discovery, measured, out, timing=timing)

    assert compiled["limits"] == {"hbm": 0, "serving": 0, "prefill": 0}
    assert len(compiled["schedule"]) == 3 * 10 * 5 * 2
    for name in ("frozen_inputs.json", "background_discovery.csv",
                 "background_states.csv", "oracle_census.csv",
                 "selected_states.json", "execution_schedule.csv",
                 "class_histogram.csv"):
        assert (out / name).is_file()


def episode_fixture(policy, decisions):
    pack, demands = small_problem()
    background = [{
        "state_id": "state", "state_hash": "state-hash", "valid": True,
        **{f"b_{name}": 2 for name in campaign.RESOURCES},
        **{f"b0_{name}": 2 for name in campaign.RESOURCES},
    }]
    schedule = [{
        "episode_id": "episode", "state_id": "state",
        "state_hash": "state-hash", "pack_id": "pack",
        "pack_hash": campaign.digest(pack), "inputs_hash": "inputs",
        "policy": policy, "repeat": 0, "block_id": "block",
        "policy_order": 0, "deadline_s": 30, "power_window_s": 5,
        "migration_window_s": 25, "analysis_weight": 1,
        "inclusion_probability": 1,
    }]
    schedule[0]["policy_input_sha256"] = campaign.policy_input_hash(schedule[0])
    power = [{"t_s": index / 4, "source_power_w": 99.6,
              "control_power_w": 100} for index in range(121)]
    raw = [{
        **schedule[0], "planner": policy,
        "background_recreated": True, "background_valid": True,
        "background_warmup_s": 30,
        "residual_measurement_id": "residual",
        "residual_telemetry_sha256": "1" * 64,
        "residual_started_ns": 1, "residual_ended_ns": 2,
        "residual_sample_count": 30, "residual_telemetry_coverage": 1,
        "scheduled_load_error": 0,
        **{f"observed_b_{name}": 2 for name in campaign.RESOURCES},
        "decisions": decisions,
        "policy_output_sha256": campaign.digest(decisions),
        "power_samples": power,
        "power_samples_sha256": campaign.digest(power),
        "power_pair_id": "source-control-pair",
        "source_power_sha256": "3" * 64,
        "control_power_sha256": "4" * 64,
    }]
    return pack, demands, background, schedule, raw


def reduce_fixture(fixture):
    pack, demands, background, schedule, raw = fixture
    return campaign.normalize_episodes(
        raw, schedule, [pack], demands, 2, background, contract(), timing)


def test_episode_reduction_keeps_policy_nonattainment_valid():
    decisions = [
        completed("a", "replay", 20),
        {"session_id": "b", "action": "not_moved", "completion_s": None},
    ]

    row = reduce_fixture(episode_fixture("queue_haul", decisions))[0]

    assert row["evidence_status"] == "valid"
    assert (row["replay_count"], row["kv_count"], row["not_moved_count"]) \
        == (1, 0, 1)
    assert json.loads(row["not_moved_sessions"]) == ["b"]
    assert row["modeled_relief_at_deadline"] == pytest.approx(.4)
    assert row["measured_relief_at_deadline"] == pytest.approx(.4)
    assert row["modeled_target_time_s"] is None
    assert row["measured_target_time_s"] is None
    assert row["selection_quality"] == 1
    assert row["residual_started_ns"] == 1
    assert row["residual_ended_ns"] == 2
    assert row["power_pair_id"] == "source-control-pair"


def test_episode_reduction_derives_instrumentation_and_reconstruction_failures():
    decisions = [
        completed("a", "replay", 20),
        {"session_id": "b", "action": "not_moved", "completion_s": None},
    ]
    fixture = episode_fixture("queue_haul", decisions)
    fixture[-1][0]["observed_b_hbm"] = 1.5

    row = reduce_fixture(fixture)[0]

    assert row["evidence_status"] == "retryable_instrumentation_failure"
    assert row["evidence_valid"] is False

    fixture = episode_fixture("queue_haul", decisions)
    fixture[-1][0]["observed_b_hbm"] = 2.2
    with pytest.raises(ValueError, match="exceeds empty reference"):
        reduce_fixture(fixture)

    fixture = episode_fixture("queue_haul", decisions)
    fixture[-1][0]["decisions"][0]["continuation_verified"] = False
    fixture[-1][0]["policy_output_sha256"] = campaign.digest(
        fixture[-1][0]["decisions"])
    assert reduce_fixture(fixture)[0]["evidence_status"] == "reconstruction_failure"


def test_episode_reduction_checks_hashes_identity_and_restricted_policy():
    decisions = [completed(session, "kv_transfer", 10 + index)
                 for index, session in enumerate(("a", "b"))]
    fixture = episode_fixture("kv_only", decisions)
    fixture[-1][0]["inputs_hash"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        reduce_fixture(fixture)

    fixture = episode_fixture("kv_only", decisions)
    fixture[-1][0]["decisions"][1]["action"] = "replay"
    with pytest.raises(ValueError, match="output hash"):
        reduce_fixture(fixture)
    fixture[-1][0]["policy_output_sha256"] = campaign.digest(
        fixture[-1][0]["decisions"])
    with pytest.raises(ValueError, match="kv_only"):
        reduce_fixture(fixture)



def test_episode_reduction_rejects_unfinished_moves_and_sparse_power():
    decisions = [completed("a", "replay", 20),
                 {**completed("b", "replay", 21), "completion_s": None}]
    with pytest.raises(ValueError, match="invalid decisions"):
        reduce_fixture(episode_fixture("queue_haul", decisions))

    fixture = episode_fixture("queue_haul", [
        completed("a", "replay", 20),
        {"session_id": "b", "action": "not_moved", "completion_s": None},
    ])
    fixture[-1][0]["power_samples"] = fixture[-1][0]["power_samples"][::120]
    fixture[-1][0]["power_samples_sha256"] = campaign.digest(
        fixture[-1][0]["power_samples"])
    with pytest.raises(ValueError, match="power samples"):
        reduce_fixture(fixture)

def test_pre_execution_outputs_include_the_class_histogram(tmp_path):
    background = [{"state_id": "state"}]
    census = [{"state_id": "state", "pack_id": "pack", "oracle_class": "Slack"}]
    selected = {"states": [{"state_id": "state"}], "shortages": {}}

    campaign.write_outputs(tmp_path, background, census, selected)

    assert "candidate_population,Slack,1" in (
        tmp_path / "class_histogram.csv").read_text()


def test_plots_write_all_required_figure_families(tmp_path):
    episodes = [{"state_id": "state", "pack_id": "pack", "repeat": 0,
                 "policy": policy, "evidence_valid": True, "analysis_weight": 1,
                 "modeled_relief_at_deadline": 1.0,
                 "measured_relief_at_deadline": 1.0,
                 "modeled_target_time_s": 20.0,
                 "measured_target_time_s": 20.0, "replay_count": 1,
                 "kv_count": 0, "not_moved_count": 1}
                for policy in campaign.POLICIES]
    census = [{"state_id": "state", "pack_id": "pack", "p_star": 1,
               "j": True, "oracle_class": "Slack",
               "resource_labels": '["wan"]'}]

    campaign.plot_results(episodes, census, tmp_path)

    for name in ("relative_relief", "target_attainment",
                 "action_composition", "class_histograms"):
        assert (tmp_path / f"{name}.png").is_file()

def test_wan_relaxation_uses_the_natural_empty_reference():
    pack, demands = small_problem()
    common = {"valid": True, "n_hbm": 0, "n_serving": 0, "n_prefill": 0}
    background = [
        {"state_id": "natural", "state_hash": "n", "wan_setting": "natural",
         **common, **{f"b_{name}": 2 for name in campaign.RESOURCES}},
        {"state_id": "controlled", "state_hash": "c",
         "wan_setting": "controlled_40", **common, "b_wan": 1,
         "b_service": 2, "b_prefill": 2, "b_hbm": 2},
        {"state_id": "loaded", "state_hash": "l",
         "wan_setting": "controlled_40", **common, "n_serving": 1,
         "b_wan": 1, "b_service": 0, "b_prefill": 0, "b_hbm": 2},
    ]

    row = next(value for value in campaign.oracle_census(
        background, [pack], demands, 2, timing) if value["state_id"] == "loaded")

    assert row["p_star"] == pytest.approx(.4)
    assert row["delta_wan"] == pytest.approx(1.6)


def test_non_wan_relaxation_uses_the_same_setting_empty_reference():
    pack, demands = small_problem()
    common = {"valid": True, "n_hbm": 0, "n_serving": 0, "n_prefill": 0}
    background = [
        {"state_id": "natural", "state_hash": "n", "wan_setting": "natural",
         **common, **{f"b_{name}": 2 for name in campaign.RESOURCES}},
        {"state_id": "controlled", "state_hash": "c",
         "wan_setting": "controlled_40", **common,
         "b_wan": 1, "b_service": 1, "b_prefill": 2, "b_hbm": 2},
        {"state_id": "loaded", "state_hash": "l",
         "wan_setting": "controlled_40", **common, "n_serving": 1,
         "b_wan": .5, "b_service": 0, "b_prefill": 2, "b_hbm": 2},
    ]

    row = next(value for value in campaign.oracle_census(
        background, [pack], demands, 2, timing) if value["state_id"] == "loaded")

    assert row["p_star"] == 0
    assert row["delta_service"] == pytest.approx(.4)


def test_overlapping_state_strata_have_exact_union_inclusion_probability():
    census = [
        {"state_id": "overlap", "state_hash": "o", "oracle_class": name}
        for name in campaign.ORACLE_CLASSES[:2]
    ] + [
        {"state_id": "slack", "state_hash": "s",
         "oracle_class": campaign.ORACLE_CLASSES[0]},
        {"state_id": "kv", "state_hash": "k",
         "oracle_class": campaign.ORACLE_CLASSES[1]},
    ]

    selected = campaign.select_states(census, seed=4, quota=1)

    for row in selected["states"]:
        expected = .75 if row["state_id"] == "overlap" else .5
        assert row["inclusion_probability"] == pytest.approx(expected)


def test_summary_collapses_repeats_before_inverse_probability_weighting():
    census = [
        {"state_id": state, "pack_id": "pack", "p_star": 1, "j": True,
         "oracle_class": "Slack"}
        for state in ("light", "heavy")
    ]
    episodes = [
        {"state_id": "light", "pack_id": "pack", "repeat": repeat,
         "policy": "queue_haul", "evidence_valid": True,
         "analysis_weight": 1, "modeled_relief_at_deadline": relief,
         "measured_relief_at_deadline": relief, "selection_quality": 1,
         "modeled_target_attained": True, "measured_target_attained": True}
        for repeat, relief in enumerate((0, 2))
    ] + [{
        "state_id": "heavy", "pack_id": "pack", "repeat": 0,
        "policy": "queue_haul", "evidence_valid": True,
        "analysis_weight": 9, "modeled_relief_at_deadline": 10,
        "measured_relief_at_deadline": 10, "selection_quality": 1,
        "modeled_target_attained": True, "measured_target_attained": True,
    }]

    rows = campaign.summarize_results(episodes, census)
    row = next(value for value in rows
               if value["policy"] == "queue_haul"
               and value["metric"] == "modeled_relative_relief"
               and value["aggregation"] == "class_balanced")

    assert row["value"] == pytest.approx(9.1)
    assert row["cases"] == 2 and row["episodes"] == 3


def test_figure_estimands_collapse_repeats_and_apply_design_weights():
    census = [{"state_id": state, "pack_id": "pack", "p_star": 1,
               "oracle_class": "Slack"} for state in ("light", "heavy")]
    episodes = [
        {"state_id": "light", "pack_id": "pack", "policy": "queue_haul",
         "repeat": repeat, "evidence_valid": True, "analysis_weight": 1,
         "modeled_relief_at_deadline": relief}
        for repeat, relief in enumerate((0, 2))
    ] + [{
        "state_id": "heavy", "pack_id": "pack", "policy": "queue_haul",
        "repeat": 0, "evidence_valid": True, "analysis_weight": 9,
        "modeled_relief_at_deadline": 10,
    }]

    cases = campaign._repeat_cases(episodes, census)
    weighted = campaign._analysis_weights(
        cases, census, lambda value: value["p_star"] > 0, "class_balanced")

    assert len(cases) == 2
    light = next(row for row in cases if row["state_id"] == "light")
    assert sum(row["modeled_relief_at_deadline"] for row in light["repeats"]) / 2 == 1
    assert {row["state_id"]: weight for row, weight in weighted} \
        == pytest.approx({"light": .1, "heavy": .9})
