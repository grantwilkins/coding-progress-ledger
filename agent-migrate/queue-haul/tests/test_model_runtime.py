"""
Claim:
Each architecture-campaign checkpoint uses its pinned snapshot and validated
hybrid-cache geometry on the same BF16, TP1, 32K, eight-session runtime.

Plausible wrong implementations:
- Reuse the GPT-OSS revision for Qwen or Gemma.
- Disable the hybrid KV manager inherited from the GPT-OSS campaign.
- Give Qwen LMCache chunks that do not match its unified attention block.
- Admit a launch whose log reports the wrong KV dtype or block geometry.
- Leave Gemma multimodal caches enabled during the text-only comparison.
"""

from dataclasses import replace

import pytest

import migration_testbed as testbed


MODELS = {
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
}


@pytest.mark.parametrize(("model", "revision"), MODELS.items())
def test_model_snapshots_use_their_own_pinned_revision(tmp_path, model, revision):
    cfg = testbed.Config(model=model, hf_home=tmp_path)

    assert testbed.model_path(cfg) == (
        testbed.model_snapshot_dir(tmp_path, model) / revision
    )


def test_campaign_launches_share_controls_but_keep_model_cache_geometry(
        monkeypatch):
    monkeypatch.setenv("QH_RUNTIME", "native")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    configs = {model: testbed.model_campaign_config(model) for model in MODELS}

    assert {(cfg.max_model_len, cfg.max_num_seqs, cfg.architecture_campaign)
            for cfg in configs.values()} == {(32768, 8, True)}

    qwen = configs["Qwen/Qwen3.8-27B"]
    qwen_vllm = testbed.shell(testbed.vllm_cmd(qwen, "source"))
    qwen_cache = testbed.shell(testbed.mp_server_cmd(qwen, "source"))
    assert "--language-model-only" in qwen_vllm
    assert "--mamba-cache-mode align" in qwen_vllm
    assert "--max-num-batched-tokens 1567" in qwen_vllm
    assert "--chunk-size 784" in qwen_cache
    assert "--separate-object-groups" in qwen_cache

    gemma_vllm = testbed.shell(testbed.vllm_cmd(
        configs["google/gemma-4-26B-A4B-it"], "source"))
    assert "--limit-mm-per-prompt image=0,audio=0" in gemma_vllm

    for cfg in configs.values():
        command = testbed.shell(testbed.vllm_cmd(cfg, "source"))
        assert "--tensor-parallel-size 1" in command
        assert "--dtype bfloat16" in command
        assert "--kv-cache-dtype auto" in command
        assert "--gpu-memory-utilization 0.9" in command
        assert "--disable-hybrid-kv-cache-manager" not in command
        assert "speculative" not in command


def test_campaign_rejects_drift_before_launch():
    qwen = testbed.model_campaign_config("Qwen/Qwen3.8-27B")

    with pytest.raises(ValueError, match="runtime geometry"):
        testbed.validate_model_runtime(
            replace(qwen, max_num_batched_tokens=8192))


def test_campaign_log_must_prove_bf16_and_qwen_unified_block():
    qwen = testbed.model_campaign_config("Qwen/Qwen3.8-27B")
    good = """
    Using bfloat16 data type to store kv cache
    Setting attention block size to 784 tokens
    """
    testbed.validate_model_runtime_log(qwen, good)

    with pytest.raises(RuntimeError, match="784-token"):
        testbed.validate_model_runtime_log(
            qwen, good.replace("784 tokens", "768 tokens"))
    with pytest.raises(RuntimeError, match="BF16"):
        testbed.validate_model_runtime_log(
            qwen, good.replace("bfloat16", "float8_e4m3fn"))

    direct = """
    QH_KV_CACHE_DTYPE_VERIFIED dtype=torch.bfloat16 entries=36 tensors=48
    Setting attention block size to 784 tokens
    """
    testbed.validate_model_runtime_log(qwen, direct)
    with pytest.raises(RuntimeError, match="BF16"):
        testbed.validate_model_runtime_log(
            qwen, direct.replace("torch.bfloat16", "torch.float16"))
    with pytest.raises(RuntimeError, match="BF16"):
        testbed.validate_model_runtime_log(
            qwen, direct.replace("entries=36", "entries=0"))


def test_unknown_models_fail_instead_of_inheriting_a_known_revision(tmp_path):
    with pytest.raises(ValueError, match="unsupported model"):
        testbed.model_path(testbed.Config(model="example/unknown", hf_home=tmp_path))
