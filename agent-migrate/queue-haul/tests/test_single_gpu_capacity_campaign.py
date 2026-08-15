"""Semantic tests for non-gating single-A100 capacity discovery."""

import json

import pytest

import migration_testbed as testbed
import single_gpu_capacity_campaign as capacity


def test_plan_is_single_a100_phase_and_non_gating():
    plan = capacity.make_plan(seed=7)

    assert len(plan["cells"]) == 15
    assert plan["runtime"]["gpu_count"] == 1
    assert plan["runtime"]["tensor_parallel_size"] == 1
    assert plan["runtime"]["dtype"] == "bfloat16"
    assert plan["semantics"]["campaign_gate"] is False
    assert all(value % 784 == 0 for value in
               plan["contexts"]["Qwen/Qwen3.8-27B"])


@pytest.mark.parametrize("model", capacity.MODELS)
def test_capacity_commands_preserve_model_geometry(monkeypatch, model):
    monkeypatch.setenv("QH_RUNTIME", "native")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    cfg = capacity.model_config(model)
    vllm_command = testbed.vllm_cmd(cfg, "sink", [], gpu_index=0)
    vllm = testbed.shell(vllm_command)
    cache = testbed.shell(testbed.mp_server_cmd(
        cfg, "sink", l2_port=cfg.lmc_port))

    assert cfg.capacity_discovery
    assert not cfg.architecture_campaign
    assert cfg.max_model_len == 32768
    assert cfg.max_num_seqs == 256
    assert "--tensor-parallel-size 1" in vllm
    assert "--dtype bfloat16" in vllm
    assert "--gpu-memory-utilization 0.9" in vllm
    assert "--disable-hybrid-kv-cache-manager" not in vllm
    assert f'"lmcache.mp.port":{cfg.sink_lmc_port}' in vllm_command[-1]
    assert f"--port {cfg.sink_lmc_port}" in cache
    if model == "Qwen/Qwen3.8-27B":
        assert "--max-num-batched-tokens 1567" in vllm
        assert "--chunk-size 784" in cache
        assert "--separate-object-groups" in cache
    if model == "google/gemma-4-26B-A4B-it":
        assert "--limit-mm-per-prompt" in vllm
        assert '"image":0,"audio":0' in vllm
        assert "image=0,audio=0" not in vllm


def test_summary_keeps_launch_failure_as_discovery_outcome():
    result = {
        "model": "Qwen/Qwen3.8-27B", "revision": "rev",
        "context_tokens": 32144, "launchable": False,
        "outcome_error": {"phase": "launch", "kind": "oom"},
        "runtime_geometry": None, "probes": [],
    }

    row = capacity.cell_summary(result)

    assert row["launchable"] is False
    assert row["outcome_error_phase"] == "launch"
    assert row["outcome_error_kind"] == "oom"
    assert row["max_completed_burst_width"] == 0


def test_runtime_contract_failure_is_not_a_capacity_kind():
    error = capacity.RuntimeContractError("BF16 proof failed")

    assert not capacity.recordable_outcome(error, False, "context_rejected")
    assert not capacity.recordable_outcome(error, True, "service_error")
    assert capacity.recordable_outcome(RuntimeError("OOM"), False, "oom")
    assert capacity.recordable_outcome(
        RuntimeError("request timed out"), True, "service_error")


def test_summary_distinguishes_completed_burst_and_true_concurrency():
    result = {
        "model": "openai/gpt-oss-20b", "revision": "rev",
        "context_tokens": 16384, "launchable": True,
        "outcome_error": None, "runtime_geometry": {
            "kv_capacity_tokens": 100000, "available_kv_cache_gib": 20,
            "kv_cache_dtype_proof": True,
        },
        "probes": [
            {"width": 8, "all_completed": True, "saturated": False,
             "engine_exited": False, "request_error": None,
             "peak_running_requests": 8, "peak_waiting_requests": 0},
            {"width": 16, "all_completed": True, "saturated": True,
             "engine_exited": False, "request_error": None,
             "peak_running_requests": 9, "peak_waiting_requests": 7},
        ],
    }

    row = capacity.cell_summary(result)

    assert row["max_completed_burst_width"] == 16
    assert row["max_peak_running_requests"] == 9
    assert row["first_saturated_width"] == 16


def test_reduce_has_no_performance_gate(tmp_path):
    plan = capacity.make_plan(seed=3)
    for cell in plan["cells"]:
        directory = tmp_path / "cells" / cell["cell_id"]
        directory.mkdir(parents=True)
        identity = {"cell": cell["cell_id"]}
        identity["sha256"] = capacity.digest(identity)
        result = {
            "schema": capacity.SCHEMA,
            "plan_sha256": capacity.digest(plan), **cell,
            "status": "complete", "discovery_only": True,
            "runtime_identity": identity,
            "runtime_identity_sha256": identity["sha256"],
            "launchable": False,
            "outcome_error": {"phase": "launch", "kind": "oom"},
            "runtime_geometry": None, "probes": [],
            "not_run_widths": list(capacity.WIDTHS),
        }
        (directory / "result.json").write_text(json.dumps(result))

    summary = capacity.reduce(plan, tmp_path)

    assert summary["campaign_gate"] is False
    assert all(row["outcome_error_kind"] == "oom"
               for row in summary["rows"])
