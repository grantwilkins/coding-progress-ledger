"""Prospective selection and frozen-pair checks for the stress experiment."""

import json

import repair_hardware_campaign as hardware
import network_campaign as network
import repair_stress_campaign as campaign


def test_complete_preflight_selects_two_robust_cells_per_axis():
    sweep = json.loads((campaign.DEFAULT_OUT / "preflight.json").read_text())

    selected = campaign._selected_cells(sweep)

    assert len(sweep["cells"]) == 3840
    assert {axis: [row["context_seed"] for row in rows]
            for axis, rows in selected.items()} == {
        "bandwidth": [172, 142],
        "prefill": [124, 20],
        "joint": [124, 20],
    }
    for rows in selected.values():
        for row in rows:
            assert row["repair_target_s"] <= campaign.REPAIR_LATEST_S
            assert row["control_shortfall_fraction"] >= 0.05
            assert row["control_shortfall_w"] >= 1.5
            assert row["target_time_gap_s"] >= 10
            assert row["causal_redirected_sessions"] >= 2
    assert all(row["causal_method_switches"] >= 3
               for row in selected["bandwidth"])
    assert all(row["causal_destination_switches"] >= 4
               for row in selected["prefill"])


def test_hardware_plan_freezes_three_pairs_for_two_workloads_per_axis():
    plan = campaign.make_hardware_plan(
        campaign.DEFAULT_BASE_PLAN, campaign.DEFAULT_OUT / "preflight.json")

    campaign.validate_hardware_plan(plan)
    assert len(plan["episodes"]) == 36
    assert set(plan["selection"]) == set(campaign.FAULT_AXES)
    assert all(row["fault_at_s"] == 1 for row in plan["episodes"])
    assert all(row["detection_at_s"] == 2 for row in plan["episodes"])
    assert all(row["migration_cutoff_s"] == 25 for row in plan["episodes"])
    for pair in {row["pair"] for row in plan["episodes"]}:
        rows = [row for row in plan["episodes"] if row["pair"] == pair]
        assert {row["policy"] for row in rows} == {
            hardware.APPLY_POLICY, hardware.CONTROL_POLICY}
        assert len({row["expected_initial_moves_sha256"] for row in rows}) == 1
    assert {row["fault_axis"] for row in plan["episodes"]} \
        == set(campaign.FAULT_AXES)


def test_episode_scenario_uses_stress_deadline_target_and_headroom():
    base = json.loads(campaign.DEFAULT_BASE_PLAN.read_text())
    parent = json.loads(campaign._resolve(base["parent"]["path"]).read_text())
    template = hardware._template(parent)
    episode = {
        "episode_id": "stress",
        "bandwidth_state": "germany",
        "prefill_state": "germany",
        "target_shed_fraction": 0.6,
        "power_deadline_s": 30,
        "observation_horizon_s": 120,
        "healthy_east_load": 0.25,
    }

    scenario = hardware._scenario(template, base, episode)

    assert scenario["requested_shed_fraction"] == 0.6
    assert scenario["deadline_s"] == 30
    assert scenario["planning_deadline_s"] == 30
    assert scenario["full_horizon_s"] == 120
    assert scenario["background"]["east"][0] == 0.25


def test_remote_host_checks_have_a_finite_connect_timeout():
    cluster = network.Cluster.load(
        campaign.ROOT / "azure_network_cluster_east_germany.json")

    command = network.ssh_command(
        cluster.destinations[0], campaign.ROOT / "key", ["true"])

    assert "ConnectTimeout=10" in command


def test_multiaxis_reducer_requires_every_repair_control_pair(tmp_path):
    plan = campaign.make_hardware_plan(
        campaign.DEFAULT_BASE_PLAN, campaign.DEFAULT_OUT / "preflight.json")
    requested = 30.928316227187338
    for episode in plan["episodes"]:
        repair = episode["policy"] == hardware.APPLY_POLICY
        root = tmp_path / "episodes" / episode["episode_id"]
        root.mkdir(parents=True)
        (root / "result.json").write_text(json.dumps({
            "repair_outcome": "applied" if repair else "disabled",
            "event_s": 1.0, "decision_s": 2.0,
            "proposal_s": 2.1 if repair else None,
            "apply_s": 2.2 if repair else None,
            "solver_timings": ([{"duration_s": .1}] if repair else []),
            "redirected_sessions": 3 if repair else 0,
            "causal_redirected_sessions": 3 if repair else 0,
            "causal_method_switches": 2 if repair else 0,
            "causal_destination_switches": 1 if repair else 0,
            "requested_shed_w": requested,
            "cutoff_shed_w": requested if repair else requested - 3,
            "target_met_by_cutoff": repair,
            "time_to_target_s": 18 if repair else 40,
            "eventual_target_met": True,
            "predecision_shed_w": 0,
            "initial_moves_sha256": episode[
                "expected_initial_moves_sha256"],
            "ttft_recorded": True,
            "requests": [{
                "session_id": "common", "method": "replay",
                "destination_instance": "east", "ttft_s": 1.0,
                "request": {"status_code": 200, "start_ns": 1,
                            "first_byte_ns": 1_000_000_001,
                            "end_ns": 1_100_000_001},
            }],
        }))

    summary = campaign.reduce_hardware(plan, tmp_path)

    assert summary["passed"] is True
    assert summary["completed_episodes"] == 36
    assert summary["repair_target_by_cutoff"] == 18
    assert summary["control_target_by_cutoff"] == 0
    assert all(row["pairs"] == row["passed_pairs"] == 6
               for row in summary["axes"].values())
