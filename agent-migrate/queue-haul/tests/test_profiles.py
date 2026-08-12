"""
Claim:
Profiles preserve measured ranges, sealed-block KV bytes, destination ingestion
capacity, and measured action-power curves.

Plausible wrong implementations:
- Accept a convex or decreasing power curve that violates the controller model.
- Extrapolate a rate or power curve outside its measured range.
- Accept measured values without a source and error range.
- Accept a nonpositive resident KV capacity.
- Reuse KV capacity from a different engine and memory configuration.
- Sample workload columns independently and create records absent from the trace.
- Treat a legacy idle record as active or retain it as a third internal state.
- Transfer a partial block proportionally or round it up.
- Accept malformed action-power concurrency curves.
"""

import json
from pathlib import Path

import pytest

from profiles import (
    PROFILE_SCHEMA, WORKLOAD_SCHEMA, ModelProfile, WorkloadProfile,
)


def source(reference="run"):
    return {"kind": "measured", "reference": reference, "valid_range": [1, 2], "relative_error": 0.1}


def profile():
    rate = {"1": [[1, 100], [1000, 50]], "2": [[1, 80], [1000, 40]]}
    case = {
        "F": 100, "G": 80, "power_curve": [[0, 10], [0.5, 30], [1, 40]],
        "prefill_tps": rate, "decode_tps": rate, "replay_tps": rate,
        "replay_completion_s": 0.2,
        "kv_transfer": {"block_tokens": 4, "block_bytes": 100, "setup_s": 1,
                        "destination_bytes_per_s": 50, "initial_completion_s": 0.25,
                        "catch_up_fixed_s": 0.4, "tail_replay_tps": 20},
        "switch_s": 0.1, "sleep_power_delta_w": -8, "sleep_s": 3,
        "shutdown_s": 4,
        "action_power_w": {"replay": {"1": [1, 2], "2": [1.5, 3]},
                           "kv_transfer": {"1": [2, 3]},
                           "replay_on_request": {"1": [1, 2]},
                           "catch_up": {"1": [1, 2]},
                           "sleep": {"1": [1, 0]}, "off": {"1": [1, 0]}},
    }
    return {
        "schema": "queue-haul-model-profile-v4", "profile_id": "p", "status": "fitted",
        "model": "m", "hardware": "h", "precision": "bf16", "tensor_parallel": 1,
        "gpus_per_node": 8, "power_scope": "gpu", "power_window_s": 5,
        "max_ell": 1, "kv_capacity_tokens": 1000,
        "max_destination_replays": 1, "max_destination_kv_streams": 1,
        "sources": {k: source(k) for k in (
            "power", "service", "capacity", "replay", "kv_transfer", "transitions"
        )},
        "cases": {"central": case},
    }


def write(tmp_path, value, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_power_curve_is_concave_and_never_extrapolates(tmp_path):
    p = ModelProfile.load(write(tmp_path, profile()))
    assert p.case().power_curve.power(0.75) == pytest.approx(35)
    assert p.case().power_curve.power(-1e-14) == pytest.approx(10)
    with pytest.raises(ValueError, match="outside"):
        p.case().power_curve.power(1.01)

    raw = profile()
    raw["cases"]["central"]["power_curve"] = [[0, 10], [0.5, 20], [1, 40]]
    with pytest.raises(ValueError, match="concave"):
        ModelProfile.load(write(tmp_path, raw, "convex.json"))


def test_rate_range_sealed_kv_bytes_and_action_power_are_explicit(tmp_path):
    case = ModelProfile.load(write(tmp_path, profile())).case()
    assert case.prefill.rate(500.5, 1) == pytest.approx(75)
    assert [
        (case.kv_transfer.sealed_blocks(tokens),
         case.kv_transfer.sealed_bytes(tokens),
         case.kv_transfer.tail_tokens(tokens))
        for tokens in (3, 4, 11)
    ] == [(0, 0, 3), (1, 100, 0), (2, 200, 3)]
    assert case.action_power_w["replay"].power(1, False) == 2
    assert case.action_power_w["replay"].power(2, False) == 3
    with pytest.raises(ValueError, match="unsupported concurrency"):
        case.action_power_w["replay"].power(3, False)
    with pytest.raises(ValueError, match="unsupported concurrency"):
        case.prefill.rate(10, 3)
    with pytest.raises(ValueError, match="outside"):
        case.prefill.rate(0, 1)
    assert case.kv_transfer.initial_completion_s == .25
    assert case.kv_transfer.catch_up_fixed_s == .4
    assert case.kv_transfer.tail_replay_tps == 20


def test_default_profile_uses_measured_h100_capacity_and_rates():
    path = Path(__file__).parents[1] / "profiles/gpt_oss_20b_h100_tp1.json"
    profile = ModelProfile.load(path)

    assert profile.hardware == "NVIDIA H100 NVL 94GB"
    assert profile.kv_capacity_tokens == 1205376
    assert (profile.case().F, profile.case().G) == (11415.78, 451.32)


def test_version_two_profiles_do_not_inherit_zero_cost_catch_up(tmp_path):
    raw = profile()
    raw["schema"] = "queue-haul-model-profile-v2"

    with pytest.raises(ValueError, match="queue-haul-model-profile-v4"):
        ModelProfile.load(write(tmp_path, raw))


def test_missing_source_and_estimated_bounds_hard_fail(tmp_path):
    raw = profile()
    del raw["sources"]["power"]
    with pytest.raises(ValueError, match="missing sources"):
        ModelProfile.load(write(tmp_path, raw, "missing.json"))

    raw = profile()
    raw["status"] = "estimated"
    with pytest.raises(ValueError, match="profile cases"):
        ModelProfile.load(write(tmp_path, raw, "estimated.json"))

    raw = profile()
    del raw["cases"]["central"]["action_power_w"]["replay_on_request"]
    with pytest.raises(ValueError, match="action_power_w fields"):
        ModelProfile.load(write(tmp_path, raw, "action.json"))

    raw = profile()
    raw["kv_capacity_tokens"] = 0
    with pytest.raises(ValueError, match="invalid profile"):
        ModelProfile.load(write(tmp_path, raw, "capacity.json"))


def test_workload_sampling_preserves_complete_records(tmp_path):
    raw = {
        "schema": "queue-haul-workload-profile-v2", "profile_id": "w", "source": source("trace"),
        "records": [
            {"job_type": "human", "state": "cold", "context_tokens": 10,
             "prompt_tokens": 2, "output_tokens": 1,
             "request_gap_s": 100, "tool_delay_s": 0, "log_bytes": 40,
             "log_location": "source_dc"},
            {"job_type": "agent", "state": "active", "context_tokens": 100,
             "prompt_tokens": 20, "output_tokens": 10,
             "request_gap_s": 1, "tool_delay_s": 2, "log_bytes": 400,
             "log_location": "source_dc"},
        ],
    }
    w = WorkloadProfile.load(write(tmp_path, raw, "workload.json"))
    observed = {(r.context_tokens, r.request_gap_s, r.log_location) for r in w.records}
    a, b = w.sample(50, 7), w.sample(50, 7)
    assert a == b
    assert {(r.context_tokens, r.request_gap_s, r.log_location) for r in a} <= observed

    raw["records"][0]["state"] = "idle"
    assert WorkloadProfile.load(write(tmp_path, raw, "idle.json")).records[0] == w.records[0]


def test_checked_in_profiles_load_with_uncertainty_and_provenance():
    root = Path(__file__).parents[1] / "profiles"
    paths = list(root.glob("*.json"))
    models = [ModelProfile.load(path) for path in paths
              if json.loads(path.read_text()).get("schema") == PROFILE_SCHEMA]
    workloads = [WorkloadProfile.load(path) for path in paths
                 if json.loads(path.read_text()).get("schema") == WORKLOAD_SCHEMA]
    assert len(models) >= 2
    assert all(model.status == "estimated" for model in models)
    assert all(set(model.cases) == {"central", "faster", "slower"}
               for model in models)
    model = models[0]
    assert model.sources["transitions"].kind == "assumed"
    assert {w.records[0].job_type for w in workloads} == {
        "interactive_coding", "coding", "agentic_tool_loop"
    }
