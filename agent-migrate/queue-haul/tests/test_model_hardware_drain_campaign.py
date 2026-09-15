import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

import model_hardware_drain_campaign as campaign


def test_natural_state_gate_requires_entire_output_and_cold_replay_control():
    from copy import deepcopy
    from network_campaign import state_equivalence_passed
    request = {"status": 200, "done": True, "exact_token_timestamps": True,
               "finish_reason": "stop", "output_tokens": 3, "recorded_output_tokens": 3,
               "token_ids": [7, 8, 200002], "prompt_tokens": 32239, "cached_tokens": 0}
    row = {"context_tokens": 32239, "expected_wire_bytes": 1000, "kv_wire_bytes": 1010,
           **{key: deepcopy(request) for key in ("kv", "replay", "replay_control")}}
    row["kv"]["cached_tokens"] = 32000
    assert state_equivalence_passed(row, 256)
    row["replay_control"]["token_ids"][0] = 9
    assert not state_equivalence_passed(row, 256)
    row["replay_control"]["token_ids"][0] = 7
    row["kv"]["finish_reason"] = "length"
    assert not state_equivalence_passed(row, 256)


@pytest.mark.parametrize("nodes", [("east", "germany"), ("germany",), ("east",)])
def test_full_restore_reference_preserves_cold_replay_difference(tmp_path, nodes):
    from copy import deepcopy
    from network_campaign import state_equivalence_passed, full_state_reference
    request = {"status": 200, "done": True, "exact_token_timestamps": True,
               "finish_reason": "stop", "output_tokens": 3, "recorded_output_tokens": 3,
               "token_ids": [7, 8, 9], "prompt_tokens": 512, "cached_tokens": 0}
    rows = []
    for node in nodes:
        for tokens in (512, 513):
            row = {"destination": node, "context_tokens": tokens, "warm": {"context_hash": str(tokens)},
                   "expected_wire_bytes": 2000, "kv_wire_bytes": 2010,
                   **{key: deepcopy(request) for key in ("kv", "replay", "replay_control")}}
            for key in ("kv", "replay", "replay_control"):
                row[key]["prompt_tokens"] = tokens
            row["kv"]["cached_tokens"] = 512
            row["kv"]["token_ids"][0] = 10
            rows.append(row)
    geometry = {"chunk_tokens": 256, "object_groups": [{"chunk_bytes": 1000, "sw_size_chunks": -1}]}
    baseline = {"model": "qwen", "geometry": geometry, "rows": rows,
                "diagnostic_only": True, "ignore_eos": False, "forced_token": None}
    path = tmp_path / "full.json"
    path.write_text(json.dumps(baseline))
    (tmp_path / "source.log").write_text("QH_KV_GEOMETRY " + json.dumps(geometry) + "\n")
    assert full_state_reference(path, "qwen", 256) == baseline
    with pytest.raises(ValueError, match="checkpoint/runtime mismatch"):
        full_state_reference(path, "qwen", 256, tmp_path / "source.log")
    compact = deepcopy(rows[0])
    compact.update(expected_wire_bytes=1000, kv_wire_bytes=1010)
    assert not state_equivalence_passed(compact, 256)
    assert state_equivalence_passed(compact, 256, rows[0])
    for key, value in (("kv_wire_bytes", 999), ("destination", "different"),
                       ("warm", {"context_hash": "different"})):
        changed = {**compact, key: value}
        assert not state_equivalence_passed(changed, 256, rows[0])
    compact["kv"]["token_ids"][-1] = 20
    assert not state_equivalence_passed(compact, 256, rows[0])
    for invalid_rows in (rows[:-1], [rows[0], *rows],
                         [{**row, "destination": "unknown"} for row in rows]):
        path.write_text(json.dumps({**baseline, "rows": invalid_rows}))
        with pytest.raises(ValueError, match="invalid full-history"):
            full_state_reference(path, "qwen", 256)
    rows[0]["kv_wire_bytes"] = 1
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="invalid full-history"):
        full_state_reference(path, "qwen", 256)


def test_deadline_plot_reports_completed_action_changes_and_failures(tmp_path):
    rows = [{"model": model, "deadline_s": str(deadline), "status": "complete",
             "target_met": "True", "east_region": "southeastasia",
             "germany_region": "southcentralus", "east_replay": str(replay),
             "germany_replay": "0", "east_kv_transfer": str(8 - replay),
             "germany_kv_transfer": "0"}
            for model, deadline, replay in [
                ("openai/gpt-oss-20b", 15, 2), ("openai/gpt-oss-20b", 30, 4),
                ("Qwen/Qwen3.8-27B", 15, 6), ("Qwen/Qwen3.8-27B", 30, 6)]]
    rows.append({**rows[0], "status": "failed", "east_replay": "8"})
    result = campaign.deadline_action_mix(rows, tmp_path)
    assert result["within_model_changes"] == {"openai/gpt-oss-20b": True, "Qwen/Qwen3.8-27B": False}
    assert all(result["between_model_differences"].values())
    assert not result["all_episodes_completed"]
    cell = next(row for row in result["cells"] if row["model"] == "openai/gpt-oss-20b"
                and row["deadline_s"] == 15)
    assert (cell["failed"], cell["east_replay"]) == (1, 2)
    assert (tmp_path / "deadline_action_mix.png").exists()


def test_network_profile_uses_live_rates_and_rejects_incomplete_smoke(tmp_path):
    import network_campaign as network
    import model_architecture_campaign as architecture

    def request(context, method, index=0):
        return {"status": 200, "done": True, "finish_reason": "length",
                "output_tokens": 128, "recorded_output_tokens": 128,
                "exact_token_timestamps": True, "first_ns": 10**9,
                "last_token_ns": 3 * 10**9, "start_ns": index * 10**6,
                "prompt_tokens": context,
                "cached_tokens": context // 256 * 256 if method == "kv_transfer" else 0}

    contexts, rows, smoke = [4096, 16384, 32256], [], []
    (tmp_path / "requests").mkdir()
    registration = {"schema": architecture.KV_GEOMETRY_SCHEMA, "chunk_tokens": 256,
        "groups": [{"group": "full", "kernel_group": 0, "engine_group": 0,
                    "object_group": 0, "tokens_per_block": 16, "slots_per_block": 1,
                    "num_blocks": 100, "block_bytes": 1024,
                    "capacity_bytes": 102400, "chunk_bytes": 16384}],
        "object_groups": [{"object_group": 0, "kernel_groups": [0], "chunk_bytes": 16384,
                           "sw_size_chunks": -1},
                          {"object_group": 1, "kernel_groups": [1], "chunk_bytes": 512,
                           "sw_size_chunks": 1}]}
    registration["groups"].append({**registration["groups"][0], "group": "window",
        "kernel_group": 1, "engine_group": 1, "object_group": 1,
        "block_bytes": 32, "capacity_bytes": 3200, "chunk_bytes": 512})
    for node in ("east", "germany"):
        directory = tmp_path / "nodes" / node
        directory.mkdir(parents=True)
        (directory / "sink.log").write_text("QH_KV_GEOMETRY " + json.dumps(registration))
        for context in contexts:
            for repeat in range(3):
                for method in campaign.ACTIONS:
                    row = {"destination": node, "context_tokens": context, "repeat": repeat,
                           "method": method, "passed": True, "chunk_tokens": 256,
                           "destination_ready_s": context / 10000,
                           "kv_wire_bytes": context * 64 + 512 if method == "kv_transfer" else 0,
                           "mean_tpot_s": .01}
                    rows.append(row)
                    (tmp_path / "requests" / f"{node}-{context}-{repeat}-{method}.json").write_text(
                        json.dumps({"measurement": row, "request": request(context, method)}))
        smoke.extend({"destination": node, "session_id": f"s{index}",
                      "method": method, "request": request(32256, method, index)}
                     for index, method in enumerate(["replay", "kv_transfer"] * 4))
    report = {"schema": network.MIGRATION_TIMING_SCHEMA, "status": "complete",
              "model": "openai/gpt-oss-20b", "rows": rows, "contexts": contexts,
              "all_passed": True, "literal_token_timing": True,
              "source_sleep_wake_passed": True, "bandwidth": "controlled_40",
              "repeats": 3, "completed": len(rows),
              "concurrent_smoke": {"passed": True, "sessions": 8, "context_tokens": 32256,
                  "requests": smoke, "wire_bytes": {f"kv/{node}/target_to_client": 100
                                                     for node in ("east", "germany")}},
              "node_reports": {node: {"kv_capacity_tokens": 400000} for node in ("east", "germany")},
              "runtime": {"vllm": "0.24.0", "lmcache": "0.5.1"},
              "calibration_sha256": "c",
              "network_contract": {"paths": {node: {"controlled_mbps": {"40": 400}}
                                               for node in ("east", "germany")}}}
    path = tmp_path / "report.json"
    state = {"forced_token": None, "ignore_eos": False, "passed": True, "geometry": registration,
             "rows": [{"destination": node, "context_tokens": context,
                       "expected_wire_bytes": context // 256 * 16384 + 512,
                       "kv_wire_bytes": context // 256 * 16384 + 512,
                       **{key: {**request(context, method), "token_ids": list(range(32)),
                                "output_tokens": 32, "recorded_output_tokens": 32}
                          for key, method in (("kv", "kv_transfer"), ("replay", "replay"),
                                              ("replay_control", "replay"))}}
                      for node in ("east", "germany") for context in (32239, 32256)]}
    report["state_equivalence"] = state
    (tmp_path / "state_equivalence.json").write_text(json.dumps(state))
    (tmp_path / "source.log").write_text("QH_KV_GEOMETRY " + json.dumps(registration))
    path.write_text(json.dumps(report))
    out = tmp_path / "profile.json"
    gate = campaign.freeze_network_profile(tmp_path, out)
    profile = campaign._gated_profile(out, "H100")
    assert gate["schema"] == campaign.NETWORK_GATE
    assert profile.case().decode.rate(16384, 1) == pytest.approx(100)
    assert profile.kv_capacity_tokens == 400000
    state["rows"][0]["kv"]["token_ids"][0] = -1
    (tmp_path / "state_equivalence.json").write_text(json.dumps(state))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        campaign.freeze_network_profile(tmp_path, out)
    state["rows"][0]["kv"]["token_ids"][0] = 0
    (tmp_path / "state_equivalence.json").write_text(json.dumps(state))
    report["concurrent_smoke"]["requests"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete two-route"):
        campaign.freeze_network_profile(tmp_path, out)


def test_a100_command_runs_all_three_arms_end_to_end(monkeypatch, tmp_path):
    (tmp_path / "cluster.json").write_text('{"destinations": [{"id": "east"}, {"id": "germany"}]}')
    (tmp_path / "calibration.json").write_text('{"paths": {"east": {}, "germany": {}}}')
    models = iter(sorted(campaign.MODELS["A100"]))
    monkeypatch.setattr(campaign, "_profile", lambda path, hardware: (
        SimpleNamespace(model=next(models)), path))
    monkeypatch.setattr(campaign, "_snapshot", lambda *_args: None)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(campaign, "reduce",
                        lambda roots, out, expected, repeats=5: {
                            "roots": roots, "out": out, "expected": expected})

    result = campaign.run(
        "A100", [tmp_path / f"p{i}.json" for i in range(3)],
        tmp_path / "cluster.json", tmp_path / "calibration.json",
        tmp_path / "manifest.json", tmp_path / "run", tmp_path / "key")

    assert len(calls) == 21
    assert all(call[1]["check"] and call[1]["env"]["QH_RUNTIME"] == "native"
               for call in calls)
    assert all(call[1]["env"]["QH_NATIVE_RUNTIME_VERSIONS"] == "0.24.0,0.5.1"
               for call in calls)
    assert all("drain" in call[0] for call in calls[:3])
    assert all("--stack-block" in call[0] for call in calls[3:18])
    assert [call[1]["env"]["QH_MODEL_PROFILE"] for call in calls[3:6]] \
        != [call[1]["env"]["QH_MODEL_PROFILE"] for call in calls[6:9]]
    assert all("reduce" in call[0] for call in calls[18:])
    assert result["roots"] == [(tmp_path / "run").resolve()]
    assert result["expected"] == {
        (model, "A100") for model in campaign.MODELS["A100"]}


@pytest.mark.parametrize("repeats", [1, 5])
def test_h100_is_a_separate_complete_command(monkeypatch, tmp_path, repeats):
    (tmp_path / "cluster").write_text('{"destinations": [{"id": "east"}, {"id": "germany"}]}')
    (tmp_path / "calibration").write_text('{"paths": {"east": {}, "germany": {}}}')
    monkeypatch.setattr(campaign, "_profile", lambda path, hardware: (
        SimpleNamespace(model="openai/gpt-oss-20b"), path))
    monkeypatch.setattr(campaign, "_snapshot", lambda *_args: None)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(campaign, "reduce", lambda *_args, **_kwargs: {})

    campaign.run("H100", [tmp_path / "profile.json"],
                 *(tmp_path / name for name in
                   ("cluster", "calibration", "manifest", "run", "key")), repeats=repeats)

    assert len(calls) == 2 + repeats
    assert [call[call.index("--stack-block") + 1] for call in calls
            if "--stack-block" in call] == [str(i) for i in range(repeats)]


def test_profile_requires_the_adjacent_passing_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    path = tmp_path / "profile.json"
    path.write_text("{}")
    profile = SimpleNamespace(
        model="openai/gpt-oss-20b", hardware="H100", precision="BF16",
        tensor_parallel=1, kv_geometry=object())
    monkeypatch.setattr(campaign.ModelProfile, "load", lambda _path: profile)
    gate = {"schema": "queue-haul-model-architecture-gate-v1",
            "model": profile.model, "hardware": "H100",
            "passed": False, "launch": {"passed": True},
            "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".gate.json").write_text(json.dumps(gate))

    with pytest.raises(ValueError, match="gated BF16"):
        campaign._profile(path, "H100")
    gate["passed"] = True
    path.with_suffix(".gate.json").write_text(json.dumps(gate))
    assert campaign._profile(path, "H100") == (profile, path.relative_to(tmp_path))


def test_reduce_writes_arm_table_and_both_canonical_figures(monkeypatch, tmp_path):
    arm = tmp_path / "run/arms/gpt"
    arm.mkdir(parents=True)
    profile = campaign.ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
    (arm / "profile.json").write_bytes(profile.read_bytes())
    monkeypatch.setattr(campaign, "_gated_profile",
                        lambda path, _hardware: campaign.ModelProfile.load(path))
    plan = {
        "design": "drain", "model_profile": {"path": str(profile),
            "sha256": hashlib.sha256(profile.read_bytes()).hexdigest()},
        "manifest": {"sha256": "m"},
        "cluster": {"destinations": [
            {"id": "east", "region": "eastus2"},
            {"id": "germany", "region": "germanywestcentral"}]},
        "scenarios": [{"condition_index": index % 10, "repeat": index // 10,
                       "sessions": [{"initial_tokens": 100}]}
                      for index in range(50)],
    }
    plan_path = arm / "plan.json"
    plan_path.write_text(json.dumps(plan))
    (arm / "run_metadata.json").write_text(json.dumps({
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "runtime_environment": {"QH_RUNTIME": "native",
                                "QH_LMCACHE_MODE": "mp"}}))
    (arm / "summary.json").write_text(json.dumps({
        "expected": 50, "completed": 49, "failed": 1, "missing": 0,
        "valid": False}))
    rows = [{
        "status": "failed" if index == 0 else "complete",
        "attempt": "1", "excluded_attempts": "0",
        "modeled_power_attainment_s": str(10 + index / 10),
        "time_to_target_s": "" if index == 0 else str(5 + index / 10),
        "target_met": str(index > 0),
        "modeled_power_deadline_met": str(index > 0), "east_replay": "1",
        "east_kv_transfer": "2", "germany_replay": "2",
        "germany_kv_transfer": "3",
    } for index in range(50)]
    with (arm / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = campaign.reduce(
        [tmp_path / "run"], tmp_path / "out",
        {("openai/gpt-oss-20b", "A100")})

    assert summary["openai/gpt-oss-20b / A100"] == {
        "episodes": 50, "completed_episodes": 49, "failed_episodes": 1,
        "action_mix_episodes": 50,
        "retried_episodes": 0, "excluded_attempts": 0,
        "drain_deadline_attainment": .98,
        "modeled_power_deadline_attainment": .98}
    assert len(list(csv.DictReader(
        (tmp_path / "out/drain_episodes.csv").open()))) == 50
    for name in ("drain_attainment_ecdf", "drain_action_mix"):
        assert (tmp_path / "out" / f"{name}.png").is_file()
        assert (tmp_path / "out" / f"{name}.pdf").is_file()
    with pytest.raises(ValueError, match="incomplete arm set"):
        campaign.reduce([tmp_path / "run"], tmp_path / "partial")
    assert not (tmp_path / "partial").exists()

    for index, row in enumerate(plan["scenarios"]):
        row["scenario_id"] = str(index)
    plan_path.write_text(json.dumps(plan))
    metadata_path = arm / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    for index, row in enumerate(rows):
        row.update(repeat=str(index // 10), scenario_id=str(index))
        if index >= 10:
            row.update(status="missing", attempt="0")
    (arm / "summary.json").write_text(json.dumps({
        "expected": 50, "completed": 9, "failed": 1, "missing": 40}))
    def write_rows():
        with (arm / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0])
            writer.writeheader()
            writer.writerows(rows)
    write_rows()
    selected = campaign.reduce([tmp_path / "run"], tmp_path / "single",
                               {("openai/gpt-oss-20b", "A100")}, repeats=1)
    assert selected["openai/gpt-oss-20b / A100"]["episodes"] == 10
    assert selected["openai/gpt-oss-20b / A100"]["failed_episodes"] == 1
    assert json.loads((tmp_path / "single/campaign_selection.json").read_text()) == {
        "repeats": 1, "expected_episodes": 10}
    with pytest.raises(ValueError, match="invalid drain arm"):
        campaign._rows([tmp_path / "run"])
    rows[1]["scenario_id"] = "unmatched"
    write_rows()
    with pytest.raises(ValueError, match="unmatched selected"):
        campaign._rows([tmp_path / "run"], repeats=1)



def test_timing_reuse_requires_matching_metadata_and_complete_raw_evidence(tmp_path):
    import network_campaign as network
    metadata = {"model": "qwen", "revision": "revision123", "bandwidth": "controlled_40",
                "contexts": [4096], "repeats": 1, "destinations": ["east"],
                "cluster": {"destinations": ({"id": "east"},)},
                "runtime": {"vllm": "0.24.0", "lmcache": "0.5.1"}}
    (tmp_path / "timing_metadata.json").write_text(json.dumps(metadata))
    log = "revision123 Initializing a V1 LLM engine (v0.24.0) LMCache v0.5.1"
    (tmp_path / "source.log").write_text(log)
    (tmp_path / "nodes/east").mkdir(parents=True)
    (tmp_path / "nodes/east/sink.log").write_text(log)
    (tmp_path / "requests").mkdir()
    rows = []
    for method in ("kv_transfer", "replay"):
        row = {"destination": "east", "context_tokens": 4096, "repeat": 0,
               "method": method, "passed": True, "chunk_tokens": 256, **{key: metadata[key] for key in ("model", "revision", "bandwidth")}}
        rows.append(row)
        request = {"status": 200, "done": True, "finish_reason": "length", "output_tokens": 128,
                   "recorded_output_tokens": 128, "exact_token_timestamps": True,
                   "first_ns": 10**9, "last_token_ns": 3*10**9, "prompt_tokens": 4096,
                   "cached_tokens": 4096 if method == "kv_transfer" else 0}
        (tmp_path / "requests" / f"east-4096-0-{method}.json").write_text(json.dumps({"measurement": row, "request": request}))
    progress = {"schema": network.MIGRATION_TIMING_SCHEMA, "literal_token_timing": True,
                "completed": 2, "expected": 2, "rows": rows}
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    assert network.timing_reference_rows(tmp_path, metadata) == rows
    with pytest.raises(ValueError, match="configuration changed"):
        network.timing_reference_rows(tmp_path, {**metadata, "bandwidth": "different"})
    (tmp_path / "source.log").write_text("different revision/runtime")
    with pytest.raises(ValueError, match="checkpoint/runtime changed"):
        network.timing_reference_rows(tmp_path, metadata)
    (tmp_path / "source.log").write_text(log)
    rows.pop()
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="incomplete timing"):
        network.timing_reference_rows(tmp_path, metadata)


def test_timing_reuse_rejects_full_history_geometry(tmp_path):
    from network_campaign import validate_timing_geometry
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir(); fresh.mkdir()
    geometry = {"chunk_tokens": 256, "object_groups": [{"chunk_bytes": 1000, "sw_size_chunks": 1}]}
    for root in (old, fresh):
        (root / "source.log").write_text("QH_KV_GEOMETRY " + json.dumps(geometry))
    validate_timing_geometry(old, fresh)
    geometry["object_groups"][0]["sw_size_chunks"] = -1
    (old / "source.log").write_text("QH_KV_GEOMETRY " + json.dumps(geometry))
    with pytest.raises(ValueError, match="compact geometry changed"):
        validate_timing_geometry(old, fresh)


def test_route_subset_retains_gated_parent_and_measured_envelope():
    import network_campaign as network
    raw = {'paths': {'east': {'simultaneous_mbps': [100, 120]},
                     'germany': {'simultaneous_mbps': [200, 240]}},
           'hosts': {key: {'id': key} for key in ('source', 'east', 'germany')},
           'clock_uncertainty_ms': {'source': 1, 'east': 1, 'germany': 1},
           'aggregate_simultaneous_mbps': [300, 360]}
    cluster = {'source': {'id': 'source'}, 'destinations': [{'id': 'germany'}]}
    selected = campaign._route_calibration(raw, cluster)
    assert selected['paths'] == {'germany': raw['paths']['germany']}
    assert selected['aggregate_simultaneous_mbps'] == [200, 240]
    assert set(selected['hosts']) == set(selected['clock_uncertainty_ms']) == {'source', 'germany'}
    assert selected['provenance']['parent_calibration_sha256'] == network.profiler.object_hash(raw)
    assert set(raw['paths']) == {'east', 'germany'}
    with pytest.raises(ValueError, match='absent from gated calibration'):
        campaign._route_calibration(raw, {**cluster, 'destinations': [{'id': 'unknown'}]})


def test_deadline_plot_labels_only_measured_route(tmp_path):
    row = {'model': 'openai/gpt-oss-20b', 'deadline_s': '30', 'status': 'complete',
           'target_met': 'True', 'east_region': '', 'germany_region': 'southcentralus',
           'east_replay': '0', 'east_kv_transfer': '0',
           'germany_replay': '3', 'germany_kv_transfer': '5'}
    result = campaign.deadline_action_mix([row], tmp_path)
    assert result['cells'][0]['germany_replay'] == 3
    with pytest.raises(ValueError, match='absent route'):
        campaign.deadline_action_mix([{**row, 'east_replay': '1'}], tmp_path)


def test_sample_distribution_counts_holds_and_excludes_failures(tmp_path):
    rows = [{"model": "openai/gpt-oss-20b", "condition_index": str(i),
             "status": "complete", "deadline_s": "30", "east_replay": "0",
             "germany_replay": "2", "east_kv_transfer": "0",
             "germany_kv_transfer": "3", "hold": "3"} for i in range(2)]
    rows.append({**rows[0], "status": "failed", "hold": ""})
    result = campaign.sample_action_mix(rows, tmp_path)["openai/gpt-oss-20b"]
    assert result == {"cases": 3, "completed": 2, "failed": 1,
                      "action_mix_distribution": {"replay=2,kv=3,hold=3": 2},
                      "route_action_mix_distribution": {"east_replay=0,germany_replay=2,east_kv_transfer=0,germany_kv_transfer=3,hold=3": 2}}
