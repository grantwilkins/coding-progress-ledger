"""
Claim:
The service-curve wrapper only builds the minimal powertrace-sim runbook needed for
Queue-Haul curves, preserving context limits and probe-specific vLLM flags.

Plausible wrong implementations:
- Let passthrough flags override typed launch knobs.
- Run prefill with chunked-prefill enabled.
- Send prompts longer than the served context window.
- Accidentally include extra probes beyond the requested curves.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import service_curve_runner as s1


def args(**kwargs):
    base = dict(
        model="org/model",
        served_model_name=None,
        hardware="H100",
        tp=8,
        gpus_per_node=8,
        max_model_len=5000,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        kv_cache_dtype="auto",
        hold_s=45.0,
        prefill_lens=None,
        decode_output_len=2048,
        mixed_output_len=512,
        mixed_points=16,
        probes=["decode_staircase", "prefill_staircase", "mixed_grid"],
        powertrace_root=Path("/pt"),
        run_root=Path("/runs"),
        run_id="unit",
        python="python3",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_prefill_lengths_prune_defaults_but_reject_explicit_overflow():
    assert s1.pruned_prefill_lens(5000, None) == [256, 1024, 4096]

    with pytest.raises(ValueError, match="exceed max_model_len"):
        s1.pruned_prefill_lens(5000, [1024, 8192])


def test_serve_command_keeps_prefill_chunking_off_and_rejects_duplicates():
    a = args(served_model_name="served")

    decode = s1.serve_cmd(a, "decode_staircase", ["--trust-remote-code"])
    prefill = s1.serve_cmd(a, "prefill_staircase", [])

    assert "vllm serve org/model" in decode
    assert "--served-model-name served" in decode
    assert "--enable-chunked-prefill" in decode
    assert "--enable-chunked-prefill" not in prefill
    assert "--trust-remote-code" in decode
    with pytest.raises(ValueError, match="duplicates typed flag"):
        s1.serve_cmd(a, "decode_staircase", ["--max-model-len=4096"])


def test_runbook_contains_only_requested_probes_and_pruned_lengths():
    a = args(probes=["prefill_staircase", "mixed_grid"])

    text = s1.runbook(a, [])

    assert "decode_staircase.py" not in text
    assert "prefill_staircase.py" in text
    assert "mixed_grid.py" in text
    assert "--input-lens 256 1024 4096" in text
    assert "--prefill-max 4096" in text
    assert "65536" not in text
