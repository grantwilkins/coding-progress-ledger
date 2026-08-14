"""Prospective selection and frozen-pair checks for the stress experiment."""

import json

import repair_hardware_campaign as hardware
import repair_stress_campaign as campaign
import network_campaign as network


def test_complete_preflight_selects_only_declared_qualifier():
    sweep = json.loads((campaign.DEFAULT_OUT / "preflight.json").read_text())

    selected = campaign._selected_cell(sweep)

    assert len(sweep["cells"]) == 380
    assert sweep["qualified_cells"] == [{
        "target_fraction": 0.5,
        "healthy_east_load": 0.5,
        "move_concurrency": 4,
        "context_seed": 20,
    }]
    assert selected["repair_target_s"] < campaign.REPAIR_LATEST_S
    assert selected["control_shortfall_fraction"] >= 0.05
    assert selected["predecision_fraction"] == 0
    assert selected["diff"]["redirected_sessions"] == 5


def test_hardware_plan_freezes_five_randomized_pairs():
    plan = campaign.make_hardware_plan(
        campaign.DEFAULT_BASE_PLAN, campaign.DEFAULT_OUT / "preflight.json")

    campaign.validate_hardware_plan(plan)
    assert len(plan["episodes"]) == 10
    assert plan["selection"]["context_seed"] == 20
    assert all(row["fault_at_s"] == 1 for row in plan["episodes"])
    assert all(row["detection_at_s"] == 2 for row in plan["episodes"])
    assert all(row["migration_cutoff_s"] == 25 for row in plan["episodes"])
    for pair in range(5):
        rows = [row for row in plan["episodes"] if row["pair"] == pair]
        assert {row["policy"] for row in rows} == {
            hardware.APPLY_POLICY, hardware.CONTROL_POLICY}
        assert len({row["expected_initial_moves_sha256"] for row in rows}) == 1


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
        "healthy_east_load": 0.25,
    }

    scenario = hardware._scenario(template, base, episode)

    assert scenario["requested_shed_fraction"] == 0.6
    assert scenario["deadline_s"] == 30
    assert scenario["planning_deadline_s"] == 30
    assert scenario["background"]["east"][0] == 0.25


def test_remote_host_checks_have_a_finite_connect_timeout():
    cluster = network.Cluster.load(
        campaign.ROOT / "azure_network_cluster_east_germany.json")

    command = network.ssh_command(
        cluster.destinations[0], campaign.ROOT / "key", ["true"])

    assert "ConnectTimeout=10" in command
