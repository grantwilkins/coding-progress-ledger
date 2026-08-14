import csv
import json
from pathlib import Path

import testbed_calibration_campaign as campaign


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def test_reducer_enforces_timing_and_operational_gates(tmp_path):
    timing = [{"predicted_s": 10, "observed_s": 10.5,
               "correctness_failures": 0} for _ in range(9)]
    service, migration, operational = (tmp_path / name for name in (
        "service.csv", "migration.csv", "operational.csv"))
    write(service, timing); write(migration, timing)
    write(operational, [{"state": state, "repeat": repeat,
                         "predicted_feasible": True, "observed_feasible": True,
                         "correctness_failures": 0}
                        for state in campaign.STATES for repeat in range(3)])
    result = campaign.reduce(service, migration, operational)
    assert result["service_gate_passed"]
    assert result["migration_gate_passed"]
    assert result["operational_gate_passed"]


def test_inventory_recovers_timing_concurrency_and_route_rate(tmp_path):
    attempt = tmp_path / "scenarios" / "case" / "attempt-0001"
    stack = tmp_path / "stacks" / "run"
    attempt.mkdir(parents=True); stack.mkdir(parents=True)
    scenario = {"scenario_id": "case", "deadline_s": 10, "bandwidth": "natural",
                "background": {"east": [.6, 0]},
                "sessions": [{"session_id": "a", "initial_tokens": 100}]}
    request = {"start_ns": 2_000_000_000, "end_ns": 5_000_000_000,
               "first_byte_ns": 4_000_000_000, "prompt_tokens": 100,
               "cached_tokens": 80, "stream_chunks": [{"monotonic_ns": 3_000_000_000}]}
    move = {"session_id": "a", "destination_instance": "east",
            "method": "kv_transfer", "deadline_admitted": True, "request": request}
    result = {"started_ns": 1_000_000_000, "ended_ns": 5_000_000_000,
              "deadline_met": True, "requests": [move], "source_sleep_ns": [5_100_000_000],
              "wire_bytes": {"kv/east/target_to_client": 1_000},
              "connections": [], "resp_transfers": [{"connection_id": "c", "command": "GET",
                  "start_ns": "2000000000", "end_ns": "3000000000", "payload_bytes": "900"}]}
    (attempt / "scenario.json").write_text(json.dumps(scenario))
    (attempt / "result.json").write_text(json.dumps(result))
    write(stack / "proxy_connections.csv", [{"connection_id": "c", "route": "kv/east"}])
    write(stack / "power.csv", [
        {"monotonic_ns": value, "power_w": watts, "valid": 1, "gpu": 0}
        for value, watts in ((0, 100), (2_000_000_000, 200),
                             (4_000_000_000, 300), (6_000_000_000, 100))])
    rows, summary = campaign.inventory([tmp_path])
    assert rows[0]["kv_bytes"] == 900
    assert rows[0]["route_goodput_bytes_per_s"] == 1000
    assert rows[0]["destination_ready_ns"] == 3_000_000_000
    assert rows[0]["concurrent_kv"] == 1
    assert rows[0]["source_power_mean_w"] == 250
    assert rows[0]["source_power_p95_w"] == 300
    assert rows[0]["source_power_samples"] == 2
    assert summary["requires_new_per_migration_transfer_ids"] is False
    assert summary["source_power_covered_migrations"] == 1


def test_compact_campaign_is_balanced_and_fit_ready():
    plan = campaign.compact_campaign()
    cells = plan["migration_screening_cells"]
    assert len(cells) == 12
    assert {value: sum(row["destination"] == value for row in cells)
            for value in ("east", "germany")} == {"east": 6, "germany": 6}
    assert {value: sum(row["action_mix"] == value for row in cells)
            for value in campaign.ACTION_MIXES} == {
                "replay_only": 4, "kv_only": 4, "mixed": 4}
    assert {value: sum(row["bandwidth"] == value for row in cells)
            for value in campaign.BANDWIDTHS} == {"natural": 6, "controlled_40": 6}
    mixtures = [row["mixture"] for row in plan["power_anchors"] if "mixture" in row]
    assert len(mixtures) == 13
    assert set(mixtures) == set(campaign.MIXTURES)


def test_select_power_anchors(tmp_path):
    measurements, plan = tmp_path / "measurements.csv", tmp_path / "plan.json"
    rows = [{"mixture": mixture, "target_service_load": load}
            for mixture in ("prefill75", "mixed", "decode")
            for load in (.1, .45, .65)]
    write(measurements, rows)
    plan.write_text(json.dumps({"power_anchors": [
        {"mixture": row["mixture"], "load": row["target_service_load"]}
        for row in rows]}))
    selected = campaign.select_power_anchors(measurements, plan)
    assert [(row["mixture"], float(row["target_service_load"])) for row in selected] == [
        (row["mixture"], row["target_service_load"]) for row in rows]
    assert {row["validation_group"] for row in selected} == {
        "prefill_heavy", "mixed", "decode"}


def test_fit_timing_calibrates_route_endpoint_and_residual(tmp_path):
    profile = Path(campaign.__file__).with_name("profiles") / \
        "gpt_oss_20b_a100_tp1_azure_300w.json"
    parent, telemetry = tmp_path / "parent.json", tmp_path / "telemetry.csv"
    parent.write_text(json.dumps({"network_contract": {"paths": {
        node: {"natural_mbps": 1000} for node in ("east", "germany")}}}))
    base = {"bandwidth": "natural", "migration_start_ns": 0,
            "commit_ns": 10_000_000_000, "concurrent_replay": 0,
            "replay_tokens": 0}
    write(telemetry, [
        {**base, "destination": "east", "method": "kv_transfer",
         "kv_bytes": 400_000_000, "route_goodput_bytes_per_s": 50_000_000,
         "effective_kv_ingest_bytes_per_s": 45_000_000},
        {**base, "destination": "germany", "method": "kv_transfer",
         "kv_bytes": 2_000_000_000, "route_goodput_bytes_per_s": 500_000_000,
         "effective_kv_ingest_bytes_per_s": 250_000_000},
        {**base, "destination": "east", "method": "replay", "kv_bytes": "",
         "route_goodput_bytes_per_s": "", "effective_kv_ingest_bytes_per_s": "",
         "concurrent_replay": 1, "replay_tokens": 30_000},
    ])
    out_profile, out_parent = tmp_path / "profile.json", tmp_path / "calibrated.json"
    result = campaign.fit_timing(profile, parent, telemetry, out_profile, out_parent)
    calibrated = json.loads(out_parent.read_text())
    assert calibrated["network_contract"]["paths"]["east"]["natural_mbps"] == 400
    assert result["destination_ingest_bytes_per_s"] == 250_000_000
    assert result["timing_gate_passed"]
    assert not result["migration_gate_passed"]


def test_regional_timing_holds_out_largest_context_and_checks_kv_bytes(tmp_path):
    profile = Path(campaign.__file__).with_name("profiles") / \
        "gpt_oss_20b_a100_tp1_azure_300w.json"
    model = campaign.ModelProfile.load(profile).case()
    parent, telemetry = tmp_path / "parent.json", tmp_path / "telemetry.csv"
    paths = {node: {"natural_mbps": 1000,
                    "controlled_mbps": {"40": 400, "80": 800}}
             for node in ("east", "germany")}
    parent.write_text(json.dumps({
        "network_contract": {"paths": paths, "aggregate": {
            "natural_mbps": 2000, "controlled_mbps": {"40": 800, "80": 1600}}},
        "scenarios": [{"bandwidth": "natural",
                       "bandwidth_mbps": {"east": 1000, "germany": 1000}}],
    }))
    rows = []
    contexts = (4096, 8192, 16384)
    rates = {"east": {"controlled_40": 80e6, "controlled_80": 120e6,
                      "natural": 150e6},
             "germany": {"controlled_40": 200e6, "controlled_80": 250e6,
                          "natural": 300e6}}
    replay_factors = {"east": .8, "germany": .6}
    for destination in rates:
        for bandwidth, rate in rates[destination].items():
            for context in contexts:
                for repeat in range(3):
                    kv_bytes = model.kv_transfer.sealed_bytes(context)
                    kv_s = kv_bytes / rate + (2 if destination == "east" else 1)
                    x, values = model.replay.by_concurrency[1]
                    replay_rate = model.replay.rate(context, 1) \
                        if x[0] <= context <= x[-1] else min(values)
                    replay_s = replay_factors[destination] * (
                        context / replay_rate + model.replay_completion_s)
                    base = {"scenario_id": f"{destination}-{bandwidth}-{context}-{repeat}",
                            "destination": destination, "bandwidth": bandwidth,
                            "context_tokens": context, "migration_start_ns": 0,
                            "repeat": repeat}
                    rows.extend((
                        {**base, "method": "kv_transfer", "kv_bytes": kv_bytes,
                         "replay_tokens": 0, "commit_ns": round(kv_s * 1e9)},
                        {**base, "method": "replay", "kv_bytes": "",
                         "replay_tokens": context,
                         "commit_ns": round(replay_s * 1e9)},
                    ))
    write(telemetry, rows)
    calibrated, predictions = tmp_path / "calibrated.json", tmp_path / "predictions.csv"

    result = campaign.fit_regional_timing(
        profile, parent, telemetry, calibrated, predictions)

    assert result["holdout_context"] == 16384
    assert result["kv_byte_mismatches"] == 0
    assert result["migration_gate_passed"]
    assert all(value["coverage"] >= .9 for value in result["held_out"].values())
    contract = json.loads(calibrated.read_text())["network_contract"]
    assert all("migration_components" in value for value in contract["paths"].values())


def test_bundle_summary_preserves_modeled_label_and_gates(tmp_path):
    values = [
        {"schema": "queue-haul-phase-power-fit-v1", "gate_passed": False},
        {"schema": "queue-haul-compact-timing-fit-v1", "timing_gate_passed": True,
         "migration_gate_passed": False},
        {"schema": "queue-haul-stress-frontier-plan-v1", "states": [{}] * 40,
         "deadlines_s": [10, 15]},
        {"schema": "queue-haul-stress-frontier-v1", "reference_label":
         "exact modeled MILP optimum", "frontier": [{}] * 16},
    ]
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f"{index}.json"; path.write_text(json.dumps(value)); paths.append(path)
    result = campaign.bundle_summary(paths)
    assert result["claim"] == "modeled stress-suite sensitivity"
    assert not result["promotion_ready"]
    assert result["pareto"]["states"] == 40


def test_plan_spans_requested_migration_design(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"scenarios": [{"condition_id": "joint-shaped",
        "repeat": 0, "sessions": [{"initial_tokens": 10}, {"initial_tokens": 20},
                                     {"initial_tokens": 30}]}]}))
    plan = campaign.make_plan(parent)
    assert len(plan["migration_cells"]) == 2 * 3 * 4 * 3 * 3 * 2 * 3
    assert len(plan["migration_screening_cells"]) == 2 * 3 * 4 * 3


def test_network_plan_materializes_fixed_screening_moves(tmp_path):
    parent = tmp_path / "parent.json"; campaign_path = tmp_path / "campaign.json"
    sessions = [{"session_id": str(i), "template_id": str(i),
                 "initial_tokens": value} for i, value in enumerate((10, 20, 30, 40))]
    parent.write_text(json.dumps({"scenarios": [{"condition_id": "joint-shaped",
        "repeat": 0, "sessions": sessions}], "network_contract": {"paths": {
            node: {"natural_mbps": 100, "controlled_mbps": {"40": 40}}
            for node in ("east", "germany")}}}))
    campaign_path.write_text(json.dumps({"pack": "p", "migration_screening_cells": [{
        "destination": "east", "destination_prefill_load": .6, "concurrency": 2,
        "action_mix": "mixed", "context_tokens": 20, "bandwidth": "controlled_40",
        "repeat": 0}]}))
    plan = campaign.network_plan(parent, campaign_path)
    scenario = plan["scenarios"][0]
    assert [row["method"] for row in scenario["moves"]] == ["replay", "kv_transfer"]
    assert scenario["workload"] == "migration_calibration"
    assert scenario["background"] == {"east": [.6, 0], "germany": [0, 0]}
    assert scenario["bandwidth_mbps"] == {"east": 40, "germany": 40}


def test_network_plan_groups_bandwidth_to_avoid_stack_reloads(tmp_path):
    parent = tmp_path / "parent.json"; campaign_path = tmp_path / "campaign.json"
    session = {"session_id": "a", "template_id": "a", "initial_tokens": 10}
    parent.write_text(json.dumps({"scenarios": [{"condition_id": "joint-shaped",
        "repeat": 0, "sessions": [session]}], "network_contract": {"paths": {
            node: {"natural_mbps": 100, "controlled_mbps": {"40": 40}}
            for node in ("east", "germany")}}}))
    base = {"destination": "east", "destination_prefill_load": .25,
            "concurrency": 1, "action_mix": "replay_only", "context_tokens": 10,
            "repeat": 0}
    campaign_path.write_text(json.dumps({"pack": "p", "migration_screening_cells": [
        {**base, "bandwidth": value} for value in ("natural", "controlled_40", "natural")]}))
    plan = campaign.network_plan(parent, campaign_path)
    assert [row["bandwidth"] for row in plan["scenarios"]] == [
        "controlled_40", "natural", "natural"]


def test_network_plan_skips_completed_scenarios(tmp_path):
    parent = tmp_path / "parent.json"; campaign_path = tmp_path / "campaign.json"
    session = {"session_id": "a", "template_id": "a", "initial_tokens": 10}
    parent.write_text(json.dumps({"scenarios": [{"condition_id": "joint-shaped",
        "repeat": 0, "sessions": [session]}], "network_contract": {"paths": {
            node: {"natural_mbps": 100, "controlled_mbps": {"40": 40}}
            for node in ("east", "germany")}}}))
    cell = {"destination": "east", "destination_prefill_load": .25,
            "concurrency": 1, "action_mix": "replay_only", "context_tokens": 10,
            "repeat": 0, "bandwidth": "natural"}
    campaign_path.write_text(json.dumps({"pack": "p", "migration_screening_cells": [cell]}))
    full = campaign.network_plan(parent, campaign_path)
    result = tmp_path / "old" / "scenarios" / full["scenarios"][0]["scenario_id"] \
        / "attempt-0001" / "result.json"
    result.parent.mkdir(parents=True); result.write_text("{}")
    assert campaign.network_plan(parent, campaign_path, [tmp_path / "old"])["scenarios"] == []
