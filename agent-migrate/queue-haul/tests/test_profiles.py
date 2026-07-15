"""
Claim:
Profiles preserve one measured load coordinate, reject unsupported extrapolation,
and resample complete workload records rather than independent fields.

Plausible wrong implementations:
- Accept a convex or decreasing power curve that violates the controller model.
- Extrapolate a rate or power curve outside its measured range.
- Accept measured values without a source and error range.
- Sample workload columns independently and create records absent from the trace.
"""

import json

import pytest

from profiles import ModelProfile, WorkloadProfile


def source(reference="run"):
    return {"kind": "measured", "reference": reference, "valid_range": [1, 2], "relative_error": 0.1}


def profile():
    rate = {"1": [[1, 100], [1000, 50]], "2": [[1, 80], [1000, 40]]}
    case = {
        "F": 100, "G": 80, "power_curve": [[0, 10], [0.5, 30], [1, 40]],
        "prefill_tps": rate, "decode_tps": rate, "replay_tps": rate,
        "kv_transfer": {"block_tokens": 4, "block_bytes": 100, "setup_s": 1,
                        "block_processing_s": 0.5, "sync_s": 0.25},
        "switch_s": 0.1, "sleep_power_w": 2, "sleep_s": 3, "shutdown_s": 4,
        "action_power_w": {"replay": [1, 2], "kv_transfer": [2, 3],
                           "replay_on_request": [1, 2], "catch_up": [1, 2],
                           "sleep": [1, 0], "off": [1, 0]},
    }
    return {
        "schema": "queue-haul-model-profile-v1", "profile_id": "p", "status": "fitted",
        "model": "m", "hardware": "h", "precision": "bf16", "tensor_parallel": 1,
        "gpus_per_node": 8, "power_scope": "gpu", "power_window_s": 5,
        "max_ell": 1, "max_parallel_moves": 2,
        "sources": {k: source(k) for k in ("power", "service", "replay", "kv_transfer", "transitions")},
        "cases": {"central": case},
    }


def write(tmp_path, value, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_power_curve_is_concave_and_never_extrapolates(tmp_path):
    p = ModelProfile.load(write(tmp_path, profile()))
    assert p.case().power_curve.power(0.75) == pytest.approx(35)
    with pytest.raises(ValueError, match="outside"):
        p.case().power_curve.power(1.01)

    raw = profile()
    raw["cases"]["central"]["power_curve"] = [[0, 10], [0.5, 20], [1, 40]]
    with pytest.raises(ValueError, match="concave"):
        ModelProfile.load(write(tmp_path, raw, "convex.json"))


def test_rate_range_concurrency_and_kv_rounding_are_explicit(tmp_path):
    case = ModelProfile.load(write(tmp_path, profile())).case()
    assert case.prefill.rate(500.5, 1) == pytest.approx(75)
    assert case.kv_transfer.blocks(11) == 3
    assert case.kv_transfer.bytes(11) == 300
    assert case.action_power_w["replay"] == (1, 2)
    with pytest.raises(ValueError, match="unsupported concurrency"):
        case.prefill.rate(10, 3)
    with pytest.raises(ValueError, match="outside"):
        case.prefill.rate(0, 1)


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


def test_workload_sampling_preserves_complete_records(tmp_path):
    raw = {
        "schema": "queue-haul-workload-profile-v1", "profile_id": "w", "source": source("trace"),
        "records": [
            {"job_type": "human", "state": "idle", "context_tokens": 10,
             "prompt_tokens": 2, "output_tokens": 1,
             "request_gap_s": 100, "tool_delay_s": 0, "log_bytes": 40, "log_external": True},
            {"job_type": "agent", "state": "active", "context_tokens": 100,
             "prompt_tokens": 20, "output_tokens": 10,
             "request_gap_s": 1, "tool_delay_s": 2, "log_bytes": 400, "log_external": False},
        ],
    }
    w = WorkloadProfile.load(write(tmp_path, raw, "workload.json"))
    observed = {(r.context_tokens, r.request_gap_s, r.log_external) for r in w.records}
    a, b = w.sample(50, 7), w.sample(50, 7)
    assert a == b
    assert {(r.context_tokens, r.request_gap_s, r.log_external) for r in a} <= observed
