import csv
import json

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
    rows, summary = campaign.inventory([tmp_path])
    assert rows[0]["kv_bytes"] == 900
    assert rows[0]["route_goodput_bytes_per_s"] == 1000
    assert rows[0]["destination_ready_ns"] == 3_000_000_000
    assert rows[0]["concurrent_kv"] == 1
    assert summary["requires_new_per_migration_transfer_ids"] is False


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
    assert scenario["background"] == {"east": [.6, 0], "germany": [0, 0]}
    assert scenario["bandwidth_mbps"] == {"east": 40, "germany": 40}
