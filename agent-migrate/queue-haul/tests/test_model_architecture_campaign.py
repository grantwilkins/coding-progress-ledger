"""
Claim:
Pinned model/hardware arms use one exact-token measurement matrix, measured
heterogeneous KV geometry, paired workload rates, and guarded live follow-up.

Plausible wrong implementations:
- Change token shapes or profiling cells between models or hardware.
- Collapse hybrid cache groups or infer transferred bytes from model config.
- Normalize every model to the same load in the fixed-arrival comparison.
- Accept timing false positives, an out-of-grid crossover, or a tiny action shift.
- Pick model-specific live cells instead of 36 paired executions.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import model_architecture_campaign as campaign
from profiles import KVGeometry, WorkloadProfile, WorkloadRecord, Source
from test_execution_simulator import model


def test_collection_plan_is_identical_exact_token_evidence(tmp_path):
    plan = campaign.make_collection_plan(
        campaign.ROOT / "outputs/coding-manifest.json",
        "Qwen/Qwen3.8-27B", "A100", seed=7,
    )

    assert len(plan["scenarios"]) == 375
    assert plan["revision"] == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert plan["hardware"] == "A100"
    smoke = plan["scenarios"][0]
    assert (smoke["context_size"], smoke["move_concurrency"],
            smoke["method"], smoke["bandwidth_mbps"], smoke["smoke"]) == (
                32256, 8, "kv_transfer", 1000, True)
    assert all(
        row["initial_tokens"] == scenario["context_size"]
        for scenario in plan["scenarios"] for row in scenario["sessions"]
    )
    cells = {
        (row["context_size"], row["move_concurrency"], row["method"],
         row["bandwidth_mbps"], row["repeat"])
        for row in plan["scenarios"] if row["kind"] == "migration"
    }
    assert cells == {
        (context, width, method, bandwidth, repeat)
        for context in campaign.CONTEXTS for width in campaign.CONCURRENCIES
        for method in campaign.METHODS for bandwidth in campaign.BANDWIDTH_MBPS
        for repeat in range(campaign.REPEATS)
    }


def test_geometry_requires_complete_group_accounting_and_matches_wire_bytes():
    geometry = [
        {"context_tokens": context, "repeat": repeat, "group": group,
         "resident_bytes": resident, "capacity_bytes": capacity,
         "transfer_bytes": transfer}
        for context, values in ((10, ((20, 100, 90), (10, 80, 60))),
                                (20, ((40, 100, 180), (20, 80, 100))))
        for repeat in range(2)
        for group, (resident, capacity, transfer) in zip(
            ("full", "sliding"), values)
    ]
    migrations = [
        {"method": "kv_transfer", "activity": "none", "concurrency": 1,
         "success": True, "repeat": repeat, "measured_prompt_tokens": context,
         "measured_kv_bytes": total}
        for context, total in ((10, 150), (20, 280))
        for repeat in range(2)
    ]

    raw, curve = campaign.geometry_evidence(
        geometry, migrations, contexts=(10, 20), repeats=2,
        heterogeneous=True,
    )

    assert raw == {
        "groups": ["full", "sliding"],
        "capacity_bytes": [100, 80],
        "resident_bytes": [[10, 20, 10], [20, 40, 20]],
    }
    assert curve == [[10, 150], [20, 280]]
    with pytest.raises(ValueError, match="complete"):
        campaign.geometry_evidence(
            geometry[:-1], migrations, contexts=(10, 20), repeats=2,
            heterogeneous=True,
        )
    bad = [*migrations]
    bad[-1] = {**bad[-1], "measured_kv_bytes": 281}
    with pytest.raises(ValueError, match="transfer bytes"):
        campaign.geometry_evidence(
            geometry, bad, contexts=(10, 20), repeats=2,
            heterogeneous=True,
        )


def test_geometry_collector_uses_live_allocations_and_heterogeneous_gets(tmp_path):
    model_name = "google/gemma-4-26B-A4B-it"
    registration = {
        "schema": campaign.KV_GEOMETRY_SCHEMA, "chunk_tokens": 10,
        "groups": [
            {"group": "full", "kernel_group": 0, "engine_group": 0,
             "object_group": 0, "layer_indices": [0], "tokens_per_block": 1,
             "slots_per_block": 1, "num_blocks": 100, "block_bytes": 10,
             "capacity_bytes": 1000, "chunk_bytes": 30},
            {"group": "sliding", "kernel_group": 1, "engine_group": 1,
             "object_group": 1, "layer_indices": [1], "tokens_per_block": 1,
             "slots_per_block": 1, "num_blocks": 80, "block_bytes": 5,
             "capacity_bytes": 400, "chunk_bytes": 20},
        ],
        "object_groups": [
            {"object_group": 0, "kernel_groups": [0],
             "sw_size_chunks": -1, "chunk_bytes": 30},
            {"object_group": 1, "kernel_groups": [1],
             "sw_size_chunks": 2, "chunk_bytes": 20},
        ],
    }
    scenarios = []
    allocations = []
    for bandwidth, suffix in ((1000, "a"), (5000, "b")):
        scenario_id, request_id = f"scenario-{suffix}", f"request-{suffix}"
        scenarios.append({
            "scenario_id": scenario_id, "kind": "migration",
            "method": "kv_transfer", "activity": "none", "concurrency": 1,
            "move_concurrency": 1, "context_size": 10, "repeat": 0,
            "bandwidth_mbps": bandwidth,
        })
        allocations.append({
            "schema": campaign.KV_ALLOCATION_SCHEMA,
            "monotonic_ns": 11, "request_id": request_id,
            "prompt_tokens": 10, "tokens": 10, "output_tokens": 0,
            "external_tokens": 10, "blocks": {"0": 2, "1": 3},
        })
        root = tmp_path / "scenarios" / scenario_id
        root.mkdir(parents=True)
        (root / "result.json").write_text(json.dumps({
            "status": "complete", "session_cache_keys": {
                "session": [f"full-{suffix}", f"sliding-{suffix}"],
            },
            "migrations": [{
                "move": {"session_id": "session", "method": "kv_transfer"},
                "error": "", "initial_start_ns": 10, "initial_end_ns": 20,
                "initial": {"request_id": request_id, "prompt_tokens": 10,
                            "logical_kv_bytes": 50},
            }],
        }))
        (root / "resp_transfers.csv").write_text(
            "command,key_hashes,start_ns,end_ns,payload_bytes\n"
            f"GET,old-{suffix},1,2,999\n"
            f"GET,full-{suffix},12,13,30\n"
            f"GET,sliding-{suffix},14,15,20\n"
        )
    (tmp_path / "plan.json").write_text(json.dumps({
        "campaign_schema": campaign.SCHEMA, "model": model_name,
        "hardware": "A100", "scenarios": scenarios,
    }))
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "dirty": False, "lmcache_mode": "mp", "config": {
            "model": model_name, "architecture_campaign": True,
        },
    }))
    debug = tmp_path / "debug" / "testbed_1"
    debug.mkdir(parents=True)
    log = "\x1b[32mQH_KV_GEOMETRY " + json.dumps(registration) + "\x1b[0m\n"
    log += "".join("QH_KV_ALLOCATION " + json.dumps(row) + "\n"
                   for row in allocations)
    (debug / "sink.log").write_text(log)

    rows = campaign.collect_geometry_rows(
        tmp_path, contexts=(10,), repeats=1, bandwidths=(1000, 5000),
    )

    assert rows == [
        {"context_tokens": 10, "repeat": 0, "group": "full",
         "resident_bytes": 20, "capacity_bytes": 1000,
         "transfer_bytes": 30},
        {"context_tokens": 10, "repeat": 0, "group": "sliding",
         "resident_bytes": 15, "capacity_bytes": 400,
         "transfer_bytes": 20},
    ]
    assert campaign._json_markers(
        debug / "sink.log", "QH_KV_GEOMETRY ") == [registration]
    args = campaign.parse_args([
        "collect-geometry", "--run-root", str(tmp_path),
        "--out", str(tmp_path / "geometry.csv"),
    ])
    assert args.command == "collect-geometry" and isinstance(args.out, Path)


def workload() -> WorkloadProfile:
    return WorkloadProfile(
        "w", Source("measured", "trace", (10, 20), 0),
        (WorkloadRecord("coding", "active", 10, 20, 10, 10, 0, 100,
                        "source_dc"),
         WorkloadRecord("coding", "active", 20, 20, 10, 10, 0, 200,
                        "source_dc")),
    )


def profiled(tmp_path, name: str, F: float):
    base = model(tmp_path, tp=1, kv_capacity=1000)
    case = replace(base.case(), F=F)
    geometry = KVGeometry.parse({
        "groups": ["attention"], "capacity_bytes": [1000],
        "resident_bytes": [[10, 10], [20, 20]],
    })
    return replace(base, profile_id=name, model=name, hardware="A100",
                   cases={"central": case}, kv_geometry=geometry)


def test_fixed_arrivals_stay_paired_while_matched_load_is_model_specific(tmp_path):
    shapes = campaign.session_shapes(workload(), repeat=2)
    fast, slow = profiled(tmp_path, "fast", 200), profiled(tmp_path, "slow", 50)
    fixed_scale = campaign.arrival_scale(fast, shapes)
    fast_fixed = campaign.sim_sessions(shapes, fixed_scale)
    slow_fixed = campaign.sim_sessions(shapes, fixed_scale)
    slow_matched = campaign.sim_sessions(
        shapes, campaign.arrival_scale(slow, shapes))

    assert [(row.expected_f, row.expected_g) for row in fast_fixed] == [
        (row.expected_f, row.expected_g) for row in slow_fixed]
    assert sum(slow.case().service_load(row.expected_f, row.expected_g)
               for row in slow_matched) == pytest.approx(.4)
    assert sum(slow.case().service_load(row.expected_f, row.expected_g)
               for row in slow_fixed) > .4


def test_timing_gate_rejects_false_deadline_feasibility(tmp_path):
    profile = profiled(tmp_path, "timing", 100)
    transfer = replace(
        profile.case().kv_transfer,
        bytes_by_context=((10, 100), (20, 200)),
        destination_bytes_per_s=100, initial_completion_s=0,
    )
    profile = replace(
        profile, cases={"central": replace(profile.case(),
                                            kv_transfer=transfer)})
    rows = [
        {"method": method, "activity": "none", "concurrency": 1,
         "success": True, "repeat": 2, "bandwidth_mbps": 10000,
         "measured_prompt_tokens": context,
         "measured_processed_tokens": context,
         "measured_kv_bytes": context * 10,
         "initial_time_to_first_response_s": context / 100
             if method == "replay" else context / 10}
        for method in campaign.METHODS for context in (10, 20)
    ]

    assert campaign.timing_gate(profile, rows)["passed"]
    rows[0]["initial_time_to_first_response_s"] = 20
    failed = campaign.timing_gate(profile, rows)
    assert not failed["passed"]
    assert failed["false_feasible_deadlines"] == 1


def test_profile_freeze_uses_measured_curve_and_parallel_action_power(tmp_path):
    base = json.loads((campaign.ROOT / "profiles/gpt_oss_20b_h100_tp1.json").read_text())
    migrations = [{
        "method": method, "activity": "none", "concurrency": 1,
        "success": True, "repeat": repeat, "bandwidth_mbps": bandwidth,
        "measured_prompt_tokens": context,
        "measured_processed_tokens": context,
        "measured_kv_bytes": context * 1000,
        "initial_time_to_first_response_s": (
            context / 1000 if method == "replay" else
            max(context * 1000 / (bandwidth * 1e6 / 8),
                context * 1000 / 1e8) + .1),
    } for method in campaign.METHODS for context in campaign.CONTEXTS
        for repeat in range(3) for bandwidth in campaign.BANDWIDTH_MBPS]
    scenarios = [{
        "kind": "migration", "method": method, "activity": "none",
        "repeat": repeat, "concurrency": width,
        "source_added_power_w": width, "destination_added_power_w": width * 2,
    } for method in campaign.METHODS for repeat in (0, 1)
        for width in campaign.CONCURRENCIES]
    geometry = {
        "groups": ["attention"], "capacity_bytes": [1_000_000_000],
        "resident_bytes": [[context, context * 1000]
                           for context in campaign.CONTEXTS],
    }
    raw = campaign.fit_profile_raw(
        base, migrations, scenarios, geometry,
        [[context, context * 1000] for context in campaign.CONTEXTS])
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    frozen = campaign.ModelProfile.load(path)

    assert frozen.case().kv_transfer.sealed_bytes(12000) == 12_000_000
    assert frozen.max_destination_kv_streams == 8
    assert frozen.case().action_power_w["kv_transfer"].power(8, False) == 16


def screened_rows():
    rows = []
    models = tuple(campaign.MODELS)
    shares = {
        models[0]: (.1, .1), models[1]: (.4, .8), models[2]: (.6, .9),
    }
    for hardware in campaign.HARDWARE:
        for cell, bandwidth in enumerate((1000, 10000)):
            for repeat in range(3):
                for model_name in models:
                    share = shares[model_name][cell]
                    rows.append({
                        "hardware": hardware, "model": model_name,
                        "variant": "both", "workload": "coding",
                        "load_mode": "fixed", "repeat": repeat,
                        "bandwidth_mbps": bandwidth, "deadline_s": 19,
                        "shed_fraction": 2 / 3, "initial_serviceable": True,
                        "feasible": not (cell == 1 and model_name == models[2]),
                        "kv_share": share, "replay_share": 1 - share,
                        "predicted_makespan_s": 18 + cell / 2,
                        "sessions": [{"session_id": f"s{i}",
                                      "context_tokens": 10 + i}
                                     for i in range(8)],
                        "moves": [{"session_id": f"s{i}",
                                   "method": "kv_transfer" if i / 8 < share
                                   else "replay"} for i in range(8)],
                    })
    return rows


def test_interest_gate_and_live_selection_require_paired_effects(tmp_path):
    rows = screened_rows()
    gate = campaign.campaign_gate(rows, bootstrap_samples=400)
    live = campaign.select_live_rows(rows)

    assert gate["passed"]
    assert gate["crossover_cells"]
    assert any(row["material_action_shift"] for row in gate["comparisons"])
    assert len(live) == 36
    assert {(row["hardware"], row["model"], row["repeat"])
            for row in live} == {
                (hardware, model_name, repeat)
                for hardware in campaign.HARDWARE
                for model_name in campaign.MODELS for repeat in range(3)
            }
    plans = campaign.make_live_plans(
        live, campaign.ROOT / "outputs/coding-manifest.json", tmp_path)
    assert len(plans) == 6
    assert {len(json.loads(path.read_text())["scenarios"])
            for path in plans} == {6}
    for path in plans:
        campaign.validate_arm_plan(json.loads(path.read_text()))

    unchanged = [{**row, "kv_share": .5, "replay_share": .5,
                  "feasible": True} for row in rows]
    assert not campaign.campaign_gate(
        unchanged, bootstrap_samples=100)["passed"]
