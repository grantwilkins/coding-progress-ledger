"""The constrained-state campaign keeps discovery, census, and execution frozen."""

import itertools
import json

import pytest

import constrained_state_campaign as campaign


def axis_row(axis, count, **values):
    return {"axis": axis, "count": count, "warmup_s": 30,
            "measurement_s": 30, **values}


def test_discovery_retains_first_invalid_point_and_returns_last_valid_counts():
    rows = [
        axis_row("hbm", 1, rejected=False, evicted=False,
                 persistently_resident=True),
        axis_row("hbm", 2, rejected=True, evicted=False,
                 persistently_resident=False),
        axis_row("serving", 1, slo_met=True, queue_growth=0),
        axis_row("serving", 2, slo_met=False, queue_growth=1),
        axis_row("prefill", 1, queue_growth=0, completed_throughput=10),
        axis_row("prefill", 2, queue_growth=0, completed_throughput=12),
        axis_row("prefill", 3, queue_growth=0, completed_throughput=12),
    ]

    limits, retained = campaign.discovery_limits(rows)

    assert limits == {"hbm": 1, "serving": 1, "prefill": 2}
    assert [row["valid"] for row in retained] == [True, False, True, False,
                                                   True, True, False]
    with pytest.raises(ValueError, match="first invalid"):
        campaign.discovery_limits(rows[:-1])


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
        {"session_id": "a"}, {"session_id": "b"},
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
                               dict.fromkeys(campaign.RESOURCES, 2.0))

    assert row["oracle_class"] == expected
    assert row["j"] == (expected != "Full drain infeasible")
    assert row["k"] == (expected in {"Slack", "KV-only feasible"})
    assert row["r"] == (expected in {"Slack", "Replay-only feasible"})
    if expected == "Full drain infeasible":
        assert row["delta_hbm"] == pytest.approx(1.6)


def test_per_session_greedy_ignores_capacity_and_dispatches_every_fastest_action():
    pack, demands = small_problem()

    moves = campaign.per_session_greedy(pack, demands)

    assert moves == [
        {"session_id": "a", "action": "replay", "dispatch": True},
        {"session_id": "b", "action": "replay", "dispatch": True},
    ]


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

    assert campaign.window_relief(pack, completions, 30) == pytest.approx(2)
    assert campaign.target_time(pack, completions, 2) == pytest.approx(30)
    assert campaign.window_relief(
        pack, [(20, "a"), (29, "b")], 30) == pytest.approx(.72)


def test_background_validation_hard_fails_telemetry_and_rejects_ingest_limit():
    state = {"state_id": "state", "n_hbm": 0, "n_serving": 0,
             "n_prefill": 0, "wan_setting": "natural"}
    row = {**state, "warmup_s": 30, "measurement_s": 30,
           "telemetry_complete": True, "background_valid": True,
           "kv_ingest_limiting": False,
           **{f"b_{name}": 2 for name in campaign.RESOURCES}}

    valid = campaign.validate_background_states([row], [state])

    assert valid[0]["valid"] and len(valid[0]["state_hash"]) == 64
    with pytest.raises(ValueError, match="telemetry"):
        campaign.validate_background_states(
            [{key: value for key, value in row.items() if key != "b_hbm"}],
            [state])
    rejected = campaign.validate_background_states(
        [{**row, "kv_ingest_limiting": True}], [state])
    assert rejected[0]["valid"] is False


def test_census_crosses_valid_states_and_packs_only():
    pack, demands = small_problem()
    background = [{"state_id": "valid", "state_hash": "v",
                   "wan_setting": "natural", "valid": True,
                   **{f"b_{name}": 2 for name in campaign.RESOURCES}},
                  {"state_id": "invalid", "state_hash": "i",
                   "wan_setting": "natural", "valid": False,
                   **{f"b_{name}": 0 for name in campaign.RESOURCES}}]

    rows = campaign.oracle_census(background, [pack], demands, 2)

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
        campaign.oracle_census(background, [pack], demands, 2)


def full_inputs():
    packs = [{"pack_id": f"pack-{pack}",
              "sessions": [{"session_id": f"{pack}-{index}"}
                           for index in range(8)],
              "power_gains": [float(mask) for mask in range(256)]}
             for pack in range(10)]
    demands = [{"pack_id": pack["pack_id"],
                "session_id": session["session_id"], "action": action,
                "duration_s": 1,
                **{f"d_{name}": 1 for name in campaign.RESOURCES}}
               for pack in packs for session in pack["sessions"]
               for action in campaign.ACTIONS]
    return {"packs": packs,
            "profile": {"model": "gpt-oss", "gpu": "A100"},
            "action_demands": demands, "target": 255,
            "wan_settings": {name: {} for name in campaign.WAN_SETTINGS},
            "background_manifest": {"seed": 3, "sessions": []}}


def test_freeze_inputs_hashes_and_validates_the_complete_contract():
    inputs = full_inputs()
    packs = inputs["packs"]

    first = campaign.freeze_inputs(inputs, {"sobol": 7, "selection": 9})
    second = campaign.freeze_inputs(inputs, {"sobol": 7, "selection": 9})

    assert first == second
    assert set(first["hashes"]) == set(inputs) | {"policies"}
    assert len(first["inputs_hash"]) == 64
    selected = {"states": [{"state_id": "state", "sha256": "state-hash"}]}
    schedule = campaign.execution_schedule(
        selected, packs, seed=9, inputs_hash=first["inputs_hash"])
    assert schedule[0]["pack_hash"] == campaign.digest(packs[0])
    assert first["constants"] == {"D_s": 30, "W_s": 5, "T_mig_s": 25}
    with pytest.raises(ValueError, match="ten eight-session packs"):
        campaign.freeze_inputs({**inputs, "packs": packs[:-1]},
                               {"sobol": 7, "selection": 9})


def test_compile_campaign_writes_a_frozen_pre_execution_bundle(tmp_path):
    frozen = campaign.freeze_inputs(
        full_inputs(), {"sobol": 7, "selection": 9})
    discovery = [
        axis_row("hbm", 1, rejected=True, evicted=False,
                 persistently_resident=False),
        axis_row("serving", 1, slo_met=False, queue_growth=1),
        axis_row("prefill", 1, queue_growth=0, completed_throughput=0),
    ]
    candidates = campaign.candidate_states(
        {"hbm": 0, "serving": 0, "prefill": 0}, 7)
    measured = [{
        **state, "warmup_s": 30, "measurement_s": 30,
        "telemetry_complete": True, "background_valid": True,
        "kv_ingest_limiting": False,
        **{f"b_{name}": 8 for name in campaign.RESOURCES},
    } for state in candidates]

    out = tmp_path / "compiled"
    compiled = campaign.compile_campaign(frozen, discovery, measured, out)

    assert compiled["limits"] == {"hbm": 0, "serving": 0, "prefill": 0}
    assert len(compiled["schedule"]) == 3 * 10 * 5 * 2
    for name in ("frozen_inputs.json", "background_discovery.csv",
                 "background_states.csv", "oracle_census.csv",
                 "selected_states.json", "execution_schedule.csv",
                 "class_histogram.csv"):
        assert (out / name).is_file()


def test_episode_reduction_keeps_nonattainment_valid_and_counts_not_moved():
    pack, demands = small_problem()
    schedule = [{"episode_id": "episode", "state_id": "state",
                 "pack_id": "pack", "policy": "queue_haul", "repeat": 0}]
    raw = [{**schedule[0], "background_recreated": True,
            "background_valid": True, "residual_verified": True,
            "decisions": [
                {"session_id": "a", "action": "replay", "completion_s": 20},
                {"session_id": "b", "action": "not_moved",
                 "completion_s": None},
            ]}]

    row = campaign.normalize_episodes(raw, schedule, [pack], demands, 2)[0]

    assert row["evidence_valid"] is True
    assert (row["replay_count"], row["kv_count"], row["not_moved_count"]) \
        == (1, 0, 1)
    assert json.loads(row["completion_times"]) == {"a": 20.0}
    assert json.loads(row["not_moved_sessions"]) == ["b"]
    assert row["achieved_relief"] == pytest.approx(.4)
    assert row["target_time_s"] is None and row["target_attained"] is False


def test_episode_reduction_checks_frozen_identity_and_restricted_policy():
    pack, demands = small_problem()
    schedule = [{"episode_id": "episode", "state_id": "state",
                 "state_hash": "state-hash", "pack_id": "pack",
                 "pack_hash": "pack-hash", "inputs_hash": "inputs",
                 "policy": "kv_only", "repeat": 0}]
    raw = [{**schedule[0], "inputs_hash": "changed",
            "background_recreated": True, "background_valid": True,
            "residual_verified": True,
            "decisions": [
                {"session_id": "a", "action": "kv_transfer",
                 "completion_s": 10},
                {"session_id": "b", "action": "kv_transfer",
                 "completion_s": 11},
            ]}]

    with pytest.raises(ValueError, match="identity"):
        campaign.normalize_episodes(raw, schedule, [pack], demands, 2)
    raw[0]["inputs_hash"] = "inputs"
    raw[0]["decisions"][1]["action"] = "replay"
    with pytest.raises(ValueError, match="kv_only"):
        campaign.normalize_episodes(raw, schedule, [pack], demands, 2)


def test_pre_execution_outputs_include_the_class_histogram(tmp_path):
    background = [{"state_id": "state"}]
    census = [{"state_id": "state", "oracle_class": "Slack"}]
    selected = {"states": [{"state_id": "state"}], "shortages": {}}

    campaign.write_outputs(tmp_path, background, census, selected)

    assert "candidate_population,Slack,1" in (
        tmp_path / "class_histogram.csv").read_text()


def test_plots_write_all_required_figure_families(tmp_path):
    episodes = [{"state_id": "state", "pack_id": "pack", "repeat": 0,
                 "policy": policy, "evidence_valid": True,
                 "achieved_relief": 1.0, "target_time_s": 20.0,
                 "target_attained": True, "replay_count": 1,
                 "kv_count": 0, "not_moved_count": 1}
                for policy in campaign.POLICIES]
    census = [{"state_id": "state", "pack_id": "pack", "p_star": 1,
               "j": True, "oracle_class": "Slack",
               "resource_labels": '["wan"]'}]

    campaign.plot_results(episodes, census, tmp_path)

    for name in ("relative_relief", "target_attainment",
                 "action_composition", "class_histograms"):
        assert (tmp_path / f"{name}.png").is_file()
