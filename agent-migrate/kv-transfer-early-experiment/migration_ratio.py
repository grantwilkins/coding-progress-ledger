"""Modeled prompt replay time divided by runnable-state transfer time.

The compute estimate is not end-to-end TTFT: it excludes queueing, tokenization,
runtime overhead, and the first decode step. Ratios above one favor state transfer.
Architecture constants come from the public model configs available 2026-07-22.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize

B200_SYSTEM_FP8_DENSE_FLOPS = 36e15
MFU = 0.35
EFFECTIVE_FLOPS = B200_SYSTEM_FP8_DENSE_FLOPS * MFU
CONTEXTS = np.geomspace(1_000, 1_000_000, 500)
BANDWIDTHS_GBPS = np.linspace(0.1, 25, 500)

BytesFn = Callable[[int], float]
FlopsFn = Callable[[int], float]


@dataclass(frozen=True)
class Model:
    label: str
    active_b: float | None
    max_context: int
    state_bytes: BytesFn | None
    attention_flops: FlopsFn | None
    architecture: str
    cache_precision: str
    color: str


def causal_pairs(tokens: int, window: int | None = None) -> int:
    """Causal query-key pairs, optionally capped to a sliding/top-k window."""
    width = min(tokens, window or tokens)
    return width * (width + 1) // 2 + (tokens - width) * width


def compressed_pairs(tokens: int, ratio: int, topk: int | None = None) -> int:
    """Pairs after grouping consecutive keys by ``ratio`` and optional top-k."""
    covered = min(tokens, ratio * topk) if topk else tokens
    groups, tail = divmod(covered, ratio)
    pairs = ratio * groups * (groups + 1) // 2 + tail * (groups + 1)
    return pairs + (tokens - covered) * topk if topk else pairs


def gqa_state(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: float = 1) -> BytesFn:
    return lambda tokens: tokens * 2 * layers * kv_heads * head_dim * bytes_per_elem


def mla_state(layers: int, bytes_per_elem: float = 1) -> BytesFn:
    return lambda tokens: tokens * layers * (512 + 64) * bytes_per_elem


def inkling_state(tokens: int) -> float:
    global_kv = gqa_state(11, 8, 128, 2)(tokens)
    local_kv = gqa_state(55, 16, 128, 2)(min(tokens, 512))
    return global_kv + local_kv


def deepseek_v4_state(tokens: int) -> float:
    c4_entries, c128_entries, window = math.ceil(tokens / 4), math.ceil(tokens / 128), 128
    c4 = (window + c4_entries) * 512 + c4_entries * 128 / 2
    c128 = (window + c128_entries) * 512
    return 30 * c4 + 31 * c128


def nemotron_state(tokens: int) -> float:
    kv = gqa_state(12, 2, 128)(tokens)
    ssm = 48 * 256 * 64 * 128 * 2
    conv = 48 * 256 * 64 * 4 * 2
    return kv + ssm + conv


def attention_flops(layers: int, heads: int, qk_v_dim: int, window: int | None = None) -> FlopsFn:
    return lambda tokens: 2 * layers * heads * qk_v_dim * causal_pairs(tokens, window)


def inkling_attention(tokens: int) -> float:
    return attention_flops(11, 64, 256)(tokens) + attention_flops(55, 64, 256, 512)(tokens)


def glm52_attention(tokens: int) -> float:
    sparse = 2 * 78 * 64 * 512 * causal_pairs(tokens, 2048)
    indexer = 2 * 21 * 32 * 128 * causal_pairs(tokens)
    return sparse + indexer


def deepseek_v4_attention(tokens: int) -> float:
    local = 61 * causal_pairs(tokens, 128)
    compressed = 30 * compressed_pairs(tokens, 4, 1024) + 31 * compressed_pairs(tokens, 128)
    return 2 * 128 * 1024 * (local + compressed)


MODELS = (
    Model("Inkling NVFP4", 41, 1_048_576, inkling_state, inkling_attention, "11 global + 55 SWA layers", "BF16 KV", "#1f77b4"),
    Model("GLM-5.2", 40, 1_048_576, mla_state(78), glm52_attention, "MLA + DSA IndexShare", "FP8 KV", "#e377c2"),
    Model("DeepSeek-V4-Pro", 49, 1_048_576, deepseek_v4_state, deepseek_v4_attention, "30 CSA + 31 HCA layers", "FP8 KV + FP4 index", "#d62728"),
    Model("Kimi K3", None, 1_048_576, None, None, "KDA + MLA; config pending", "undisclosed", "#ff7f0e"),
    Model("Qwen3.7-Max", None, 1_048_576, None, None, "closed model", "undisclosed", "#9467bd"),
    Model("Nemotron 3 Ultra", 55, 1_048_576, nemotron_state, attention_flops(12, 64, 256), "48 Mamba + 12 attention; 262K native", "FP8 KV + FP16 Mamba", "#2ca02c"),
)


def model(label: str) -> Model:
    return next(item for item in MODELS if item.label == label)


def modeled_models() -> tuple[Model, ...]:
    return tuple(item for item in MODELS if item.state_bytes and item.attention_flops)


def replay_time(model: Model, tokens: int) -> float:
    assert model.active_b is not None and model.attention_flops is not None
    return (2 * model.active_b * 1e9 * tokens + model.attention_flops(tokens)) / EFFECTIVE_FLOPS


def transfer_time(model: Model, tokens: int, bandwidth_gbps: float) -> float:
    assert model.state_bytes is not None
    return model.state_bytes(tokens) * 8 / (bandwidth_gbps * 1e9)


def ratio(model: Model, tokens: int, bandwidth_gbps: float) -> float:
    return replay_time(model, tokens) / transfer_time(model, tokens, bandwidth_gbps)


def style_axes(ax: plt.Axes) -> None:
    ax.axhline(1, color="k", lw=1.2, ls=":", alpha=0.6)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.15)
    ax.set_ylabel("Modeled replay time / state-transfer time")


def plot_context_ratio() -> None:
    target = model("GLM-5.2")
    norm, cmap = Normalize(BANDWIDTHS_GBPS.min(), BANDWIDTHS_GBPS.max()), plt.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bandwidth in BANDWIDTHS_GBPS:
        ax.plot(CONTEXTS, [ratio(target, int(t), bandwidth) for t in CONTEXTS], color=cmap(norm(bandwidth)), lw=0.9, alpha=0.85)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlim(CONTEXTS.min(), CONTEXTS.max())
    ax.set_ylim(1e-3, 1e3)
    ax.fill_between(CONTEXTS, 1, 1e3, alpha=0.06, color="#B1040E")
    ax.fill_between(CONTEXTS, 1e-3, 1, alpha=0.06, color="#008566")
    ax.text(2e3, 105, "Transfer runnable state", color="#B1040E", style="italic")
    ax.text(8e5, 0.005, "Replay context", color="#008566", ha="right", style="italic")
    ax.set_xlabel("Context size (tokens)")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Bandwidth (Gbps)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"glm5_context_ratio_bandwidths.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tokens, bandwidths = 100_000, np.geomspace(0.1, 100, 500)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for item in modeled_models():
        ax.plot(bandwidths, [ratio(item, tokens, bw) for bw in bandwidths], label=item.label, color=item.color, lw=2.2)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xlabel("Inter-site bandwidth (Gbps)")
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"migration_ratio.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    plot_context_ratio()

    print(f"{'Model':24} {'state@100k':>12} {'replay':>9} {'xover':>9}")
    for item in MODELS:
        if item.state_bytes is None:
            print(f"{item.label:24} {'undisclosed':>32}")
            continue
        state, replay = item.state_bytes(tokens), replay_time(item, tokens)
        print(f"{item.label:24} {state / 1e9:10.2f}GB {replay:8.2f}s {state * 8 / replay / 1e9:7.2f}G")


if __name__ == "__main__":
    main()
