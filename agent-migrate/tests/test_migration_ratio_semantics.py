"""
Claim:
GLM-5 context sweeps report TTFT divided by KV transfer time across fixed
bandwidth lines, with transfer time using Gbps and KV bytes at the token level.

Plausible wrong implementations:
- Treat 1 Gbps as 1 GB/s or otherwise miss the bits-per-byte conversion.
- Drop the dense linear bandwidth sweep or collapse it into one unlabeled series.
- Use transfer/TTFT instead of TTFT/transfer.
- Break the expected bandwidth or context scaling of the ratio.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "kv-transfer-early-experiment" / "migration_ratio.py"
SPEC = importlib.util.spec_from_file_location("migration_ratio", MODULE)
migration_ratio = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = migration_ratio
SPEC.loader.exec_module(migration_ratio)


def test_glm5_one_gbps_ratio_matches_hand_derived_1k_context():
    ratio = migration_ratio.context_ratio_frame("GLM-5", 1.0, [1_000])["ratio"].iloc[0]
    replay_flops = 2 * 40e9 * 1_000 + 78 * 64 * (256 + 256) * 1_000**2
    replay_s = replay_flops / (8 * (1_979 / 2) * 1e12 * 0.35)
    kv_bytes = 1_000 * 2 * 78 * (512 + 64)
    transfer_s = kv_bytes * 8 / 1e9
    assert ratio == pytest.approx(replay_s / transfer_s)


def test_glm5_context_ratio_increases_and_scales_with_bandwidth():
    df = migration_ratio.context_ratio_frame("GLM-5", 1.0, [1_000, 1_000_000])
    assert df["ratio"].iloc[1] > df["ratio"].iloc[0]

    one_gbps = migration_ratio.context_ratio_frame("GLM-5", 1.0, [100_000])["ratio"].iloc[0]
    two_gbps = migration_ratio.context_ratio_frame("GLM-5", 2.0, [100_000])["ratio"].iloc[0]
    assert two_gbps == pytest.approx(2 * one_gbps)


def test_glm5_context_grid_keeps_one_line_per_bandwidth():
    bandwidths = [0.5, 1.0, 2.0]
    df = migration_ratio.context_ratio_grid("GLM-5", bandwidths, [1_000, 10_000])
    assert set(df["bandwidth_gbps"]) == set(bandwidths)
    assert df.groupby("bandwidth_gbps").size().to_dict() == {0.5: 2, 1.0: 2, 2.0: 2}


def test_glm5_bandwidth_surface_uses_many_linear_bandwidth_lines():
    bandwidths = migration_ratio.GLM5_CONTEXT_BANDWIDTHS_GBPS
    assert len(bandwidths) == 500
    assert bandwidths[0] == pytest.approx(0.1)
    assert bandwidths[-1] == pytest.approx(25)
    diffs = migration_ratio.np.diff(bandwidths)
    assert diffs.min() == pytest.approx(diffs.max())


def test_glm5_surface_context_and_ratio_window_match_plot_contract():
    contexts = migration_ratio.GLM5_CONTEXT_TOKENS
    assert len(contexts) == 500
    assert contexts[0] == pytest.approx(1_000)
    assert contexts[-1] == pytest.approx(10_000_000)
    assert migration_ratio.GLM5_CONTEXT_RATIO_YLIM == (1e-3, 1e3)
