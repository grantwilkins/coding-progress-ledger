"""
Claim:
The service-surface runbook preserves the MVP experiment design: isolated
prefill rho(T), context-dependent decode G(T), and one mixed interaction grid.

Plausible wrong implementations:
- Enable chunked prefill for the isolated prefill probe.
- Collapse decode context sensitivity to one prompt length.
- Let prompt/output lengths exceed the served context window.
- Let passthrough vLLM flags override typed runbook knobs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import stage1_service_surface as s


def args(**kwargs):
    base = dict(
        model="org/model",
        served_model_name=None,
        hardware="A100",
        tp=1,
        gpus_per_node=1,
        max_model_len=32768,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        kv_cache_dtype="auto",
        hold_s=45.0,
        prefill_lens=None,
        decode_prompt_lens=None,
        decode_output_len=512,
        mixed_output_len=512,
        mixed_prefill_min=256,
        mixed_prefill_max=16384,
        mixed_points=16,
        mixed_seed=0,
        powertrace_root=Path("/pt"),
        run_root=Path("/runs"),
        run_id="unit",
        python="python3",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_lengths_prune_defaults_but_reject_explicit_overflow():
    assert s.pruned_lens(5000, None) == [256, 1024, 4096]
    assert s.decode_prompts(5000, 512, None) == [256, 4096]

    with pytest.raises(ValueError, match="exceed max_model_len"):
        s.pruned_lens(5000, [1024, 8192])
    with pytest.raises(ValueError, match="exceed served window"):
        s.decode_prompts(5000, 512, [1024, 8192])
    with pytest.raises(ValueError, match="mixed prefill range"):
        s.mixed_prefill_range(5000, 512, 256, 8192)


def test_prefill_server_disables_chunking_and_passthrough_cannot_override():
    a = args()

    assert "--enable-chunked-prefill" not in s.serve_cmd(a, False, [])
    assert "--enable-chunked-prefill" in s.serve_cmd(a, True, ["--async-scheduling"])
    with pytest.raises(ValueError, match="duplicates typed flag"):
        s.serve_cmd(a, True, ["--max-model-len=4096"])


def test_runbook_contains_decode_context_sweep_and_mixed_grid():
    a = args(decode_prompt_lens=[256, 4096], prefill_lens=[256, 1024, 4096])

    text = s.runbook(a, [])

    assert text.count("prefill_staircase.py") == 1
    assert "--input-lens 256 1024 4096" in text
    assert text.count("decode_staircase.py") == 2
    assert "--prompt-len 256 --output-len 512" in text
    assert "--prompt-len 4096 --output-len 512" in text
    assert "mixed_grid.py" in text
    assert "--n-points 16 --seed 0 --prefill-min 256 --prefill-max 16384 --mixed-output-len 512" in text
