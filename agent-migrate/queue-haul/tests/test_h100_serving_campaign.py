"""The serving campaign pins complete, optimized-H100 evidence."""

import json
from types import SimpleNamespace

import pytest

import h100_serving_campaign as campaign
import migration_testbed as testbed
import power_model_campaign as power


def test_plan_pins_target_models_three_prefill_repeats_and_h100_runtime():
    plan = campaign.make_plan(7)

    assert plan["models"] == list(campaign.MODELS)
    assert "openai/gpt-oss-20b" in campaign.PREFILL_MODELS
    assert not plan["runtime"]["enforce_eager"]
    assert plan["rps_plan"]["hardware"] == "h100"
    assert len(plan["prefill"]["cells"]) == 27
    assert len(plan["power_cells"]) == 111
    assert plan["power_max_ell"] == 16
    assert all(sum(cell["context_tokens"] == context
                   and cell["concurrency"] == 1
                   for cell in plan["prefill"]["cells"]) == 3
               for context in campaign.CONTEXTS)


def test_plan_rejects_eager_or_revision_drift():
    plan = campaign.make_plan()
    plan["runtime"]["enforce_eager"] = True
    with pytest.raises(ValueError, match="invalid"):
        campaign.validate_plan(plan)


def test_optimized_runtime_requires_compilation_and_cuda_graphs():
    testbed.validate_h100_optimized_runtime(
        "vllm serve model", "torch.compile finished; CUDA graphs captured")
    with pytest.raises(RuntimeError, match="force eager"):
        testbed.validate_h100_optimized_runtime(
            "vllm serve model --enforce-eager", "compile CUDA graphs")
    with pytest.raises(RuntimeError, match="did not prove"):
        testbed.validate_h100_optimized_runtime("vllm serve model", "ready")


def test_power_command_is_model_specific_and_optimized():
    args = SimpleNamespace(model="Qwen/Qwen3.8-27B", vllm="vllm",
                           host="127.0.0.1", port=8100)
    command = power.server_command(args)
    text = " ".join(command)

    assert testbed.model_spec(args.model).revision in command
    assert "--max-num-batched-tokens 1567" in text
    assert "--language-model-only" in command
    assert "--enforce-eager" not in command
    assert "--gpu-memory-utilization .9" in text

def test_prefill_summary_rejects_cached_or_incomplete_work():
    plan = campaign.make_plan()
    cell = plan["prefill"]["cells"][0]
    row = {"status": 200, "error": "", "done": True,
           "finish_reason": "length", "output_tokens": 1,
           "recorded_output_tokens": 1, "planned_output_tokens": 1,
           "prompt_tokens": cell["context_tokens"], "cached_tokens": 1,
           "exact_token_timestamps": True, "ttft_s": .1,
           "start_ns": 1, "end_ns": 2}
    with pytest.raises(RuntimeError, match="uncached"):
        campaign.summarize(plan, campaign.MODELS[0], cell, [row], "identity")
