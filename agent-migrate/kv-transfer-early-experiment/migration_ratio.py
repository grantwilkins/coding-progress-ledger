"""
Migration decision ratio: t_replay / t_kv_transfer vs. inter-site bandwidth.

When the ratio > 1, replay is slower than shipping KV → ship KV.
When the ratio < 1, replay is faster → just replay the prompt.
The crossover at ratio = 1 is the phase boundary.

Hardware reference: 8× H100 SXM, dense bf16, MFU = 0.35.
KV sizes: bf16, from released architecture configs.
Prefill: 2·A·T (dense FFN) + 2·H_q·(d_qk + d_v)·pairs(T), where pairs(T) counts
         causal (query, key) pairs under each model's attention layout —
         sequence compression (CSA/HCA), top-k sparsity (DSA), sliding windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

# ── Hardware ──────────────────────────────────────────────────────────────────
H100_BF16_DENSE_TFLOPS = 1_979 / 2  # dense = half of sparsity peak
N_GPUS = 8
MFU = 0.35
NODE_EFF_FLOPS = N_GPUS * H100_BF16_DENSE_TFLOPS * 1e12 * MFU  # ~2.77 PFLOP/s
NODE_HBM_GB = N_GPUS * 80.0  # 8x H100 SXM 80 GB
WEIGHT_BYTES_PER_PARAM = 1.0  # FP8 serving weights
RUNTIME_HEADROOM_GB = 100.0  # KV cache + activations

BPE = 2  # bf16 bytes per element
CONTEXT_MODEL = "DeepSeek V4 Pro"  # HCA layers stay quadratic, so the ratio
CONTEXT_STEM = "deepseekv4_context_ratio_bandwidths"  # sweeps a wide range
CONTEXT_BANDWIDTHS_GBPS = np.linspace(0.1, 25, 500)
CONTEXT_TOKENS = np.geomspace(1_000, 10_000_000, 500)
CONTEXT_RATIO_YLIM = (1e-3, 1e2)
RATIO_YLIM = (1e-3, 1e2)
KV_REGION_COLOR = "#B1040E"
CONTEXT_REGION_COLOR = "#008566"
REGION_ALPHA = 0.06
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# ── Model specs ───────────────────────────────────────────────────────────────
KVFn = Callable[[int], float]


@dataclass(frozen=True)
class Attn:
    """A group of layers sharing one attention layout.

    compress: KV sequence compressed to T/compress entries (CSA/HCA).
    topk:     compressed entries each query attends to (DSA); 0 = dense.
    window:   uncompressed sliding-window entries, always attended.
    qk_dim/v_dim: per-group overrides for hybrid head widths; 0 = model default.
    """

    layers: int
    compress: int = 1
    topk: int = 0
    window: int = 0
    qk_dim: int = 0
    v_dim: int = 0

    def pairs(self, T: int) -> float:
        """Causal (query, key) pairs summed over this group's layers.

        Dense costs T²/(2m). Top-k caps each query at k entries once the
        compressed pool outgrows k (T > k·m), making the group linear in T.
        """
        if self.topk and T > self.topk * self.compress:
            core = self.topk * T - self.topk**2 * self.compress / 2
        else:
            core = T**2 / (2 * self.compress)
        return self.layers * (core + self.window * T)


@dataclass(frozen=True)
class Model:
    label: str  # display name
    active_b: float  # active params (billions)
    total_b: float  # total params (billions), sets the instance size
    attn: tuple[Attn, ...]  # attention layout by layer group
    query_heads: int
    qk_dim: int
    v_dim: int
    kv_bytes: KVFn  # total bf16 KV bytes as f(tokens)
    color: str = "k"
    ls: str | tuple = "-"  # distinct texture per model


# ── KV size functions ─────────────────────────────────────────────────────────
def gqa_kv(layers: int, kv_heads: int, head_dim: int) -> KVFn:
    """GQA/MHA: 2 (K+V) × layers × kv_heads × head_dim × bf16."""
    per_tok = 2 * BPE * layers * kv_heads * head_dim
    return lambda T: T * per_tok


def mla_kv(layers: int, kv_lora_rank: int = 512, rope_dim: int = 64) -> KVFn:
    """MLA: stores compressed latent + RoPE key per layer."""
    per_tok = BPE * layers * (kv_lora_rank + rope_dim)
    return lambda T: T * per_tok


def dsv4_kv(T: int) -> float:
    """DeepSeek-V4 Pro CSA/HCA: 30 c4a layers + 31 c128a layers.
    Each layer stores 512-dim shared KV entry (bf16) per compressed entry,
    c4a layers additionally store 128-dim indexer cache per entry.
    128-entry sliding window per layer."""
    c4a, c128a, sw = 30, 31, 128
    entry_bytes = 512 * BPE
    idx_bytes = 128 * BPE
    c4a_n = math.ceil(T / 4)
    c128a_n = math.ceil(T / 128)
    return c4a * ((sw + c4a_n) * entry_bytes + c4a_n * idx_bytes) + c128a * (
        (sw + c128a_n) * entry_bytes
    )


def gemma4_kv(T: int) -> float:
    """Five global shared-KV layers plus 25 local GQA layers."""
    return BPE * (5 * 2 * 512 * T + 2 * 25 * 8 * 256 * min(T, 1024))


def gpt_oss_kv(T: int) -> float:
    """Twelve global and 12 128-token local GQA layers."""
    return 2 * BPE * 8 * 64 * (12 * T + 12 * min(T, 128))


# ── Model catalogue ───────────────────────────────────────────────────────────
# Released model configurations, sorted by KV size at 100k tokens.
MODELS = [
    Model(
        "DeepSeek V4 Pro",
        active_b=49,
        total_b=1600,
        # 2 HCA + 59 interleaved CSA/HCA = 30 CSA + 31 HCA; m=4/128, top-k=1024,
        # 128-token sliding window on every layer.
        attn=(
            Attn(30, compress=4, topk=1024, window=128),
            Attn(31, compress=128, window=128),
        ),
        query_heads=128,
        qk_dim=512,
        v_dim=512,
        kv_bytes=dsv4_kv,
        color=OKABE_ITO["blue"],
        ls="-",
    ),
    Model(
        "Gemma 4 26B-A4B",
        active_b=3.8,
        total_b=25.2,
        attn=(Attn(25, topk=1024), Attn(5, qk_dim=512, v_dim=512)),
        query_heads=16,
        qk_dim=256,
        v_dim=256,
        kv_bytes=gemma4_kv,
        color=OKABE_ITO["orange"],
        ls=(0, (7, 2, 1, 2)),
    ),
    Model(
        "gpt-oss-20b",
        active_b=3.61,
        total_b=20.91,
        attn=(Attn(12), Attn(12, topk=128)),
        query_heads=64,
        qk_dim=64,
        v_dim=64,
        kv_bytes=gpt_oss_kv,
        color=OKABE_ITO["vermillion"],
        ls=(0, (2, 1, 2, 3)),
    ),
    Model(
        "Qwen3.8 27B",
        active_b=27,
        total_b=27,
        attn=(Attn(16),),
        query_heads=24,
        qk_dim=256,
        v_dim=256,
        kv_bytes=gqa_kv(16, 4, 256),
        color=OKABE_ITO["reddish_purple"],
        ls=(0, (8, 2)),
    ),
    Model(
        "Kimi K2.6",
        active_b=32,
        total_b=1000,
        attn=(Attn(61),),
        query_heads=64,
        qk_dim=192,
        v_dim=128,
        kv_bytes=mla_kv(61, 512, 64),
        color=OKABE_ITO["bluish_green"],
        ls=(0, (3, 1, 1, 1, 1, 1)),
    ),
    Model(
        "GLM 5",
        active_b=40,
        total_b=744,
        attn=(Attn(78, topk=2048),),  # DSA, index_topk=2048
        query_heads=64,
        qk_dim=256,
        v_dim=256,
        kv_bytes=mla_kv(78, 512, 64),
        color=OKABE_ITO["sky_blue"],
        ls=(0, (9, 3)),
    ),
]


# ── Cost model ────────────────────────────────────────────────────────────────
def nodes(m: Model) -> int:
    """Instances needed to hold FP8 weights plus runtime headroom. A model too
    large for one node is served across several, and prefill sees all of them."""
    weights = m.total_b * WEIGHT_BYTES_PER_PARAM
    return math.ceil((weights + RUNTIME_HEADROOM_GB) / NODE_HBM_GB)


def eff_flops(m: Model) -> float:
    return nodes(m) * NODE_EFF_FLOPS


def prefill_flops(m: Model, T: int) -> float:
    ffn = 2.0 * m.active_b * 1e9 * T
    attention = sum(
        2.0
        * m.query_heads
        * ((g.qk_dim or m.qk_dim) + (g.v_dim or m.v_dim))
        * g.pairs(T)
        for g in m.attn
    )
    return ffn + attention


def t_replay(m: Model, T: int) -> float:
    return prefill_flops(m, T) / eff_flops(m)


def t_transfer(m: Model, T: int, bw_gbps: float) -> float:
    return m.kv_bytes(T) * 8.0 / (bw_gbps * 1e9)


def model(label: str) -> Model:
    return next(m for m in MODELS if m.label == label)


def context_ratio_frame(label: str, bw_gbps: float, contexts) -> pd.DataFrame:
    m = model(label)
    return pd.DataFrame(
        {
            "context_tokens": T,
            "ratio": t_replay(m, int(T)) / t_transfer(m, int(T), bw_gbps),
        }
        for T in contexts
    )


def context_ratio_grid(label: str, bandwidths_gbps, contexts) -> pd.DataFrame:
    return pd.concat(
        [
            context_ratio_frame(label, float(bw), contexts).assign(
                bandwidth_gbps=float(bw)
            )
            for bw in bandwidths_gbps
        ],
        ignore_index=True,
    )


def shade_regions(ax, x, kv_at, ctx_at):
    """Shade above/below ratio = 1 and label the two decisions.

    kv_at and ctx_at are (x in axes fraction, y in data) anchors, picked per
    figure to land in the whitespace the curves leave open.
    """
    lo, hi = ax.get_ylim()
    ax.fill_between(x, 1.0, hi, alpha=REGION_ALPHA, color=KV_REGION_COLOR, zorder=0)
    ax.fill_between(
        x, lo, 1.0, alpha=REGION_ALPHA, color=CONTEXT_REGION_COLOR, zorder=0
    )
    ax.set_ylim(lo, hi)
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data
    ax.text(
        *kv_at, "Transfer KV cache", color=KV_REGION_COLOR, style="italic", transform=tr
    )
    ax.text(
        *ctx_at,
        "Transfer context",
        color=CONTEXT_REGION_COLOR,
        style="italic",
        transform=tr,
    )


def plot_context_ratio(label: str = CONTEXT_MODEL, stem: str = CONTEXT_STEM):
    df = context_ratio_grid(label, CONTEXT_BANDWIDTHS_GBPS, CONTEXT_TOKENS)
    norm = Normalize(CONTEXT_BANDWIDTHS_GBPS.min(), CONTEXT_BANDWIDTHS_GBPS.max())
    cmap = plt.colormaps["viridis"]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bw, group in df.groupby("bandwidth_gbps", sort=True):
        ax.plot(
            group["context_tokens"],
            group["ratio"],
            color=cmap(norm(bw)),
            lw=0.9,
            alpha=0.85,
        )
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(CONTEXT_TOKENS.min(), CONTEXT_TOKENS.max())
    ax.set_ylim(*CONTEXT_RATIO_YLIM)
    shade_regions(ax, CONTEXT_TOKENS, (0.04, 25), (0.56, 3e-3))
    ax.set_xlabel("Context size (tokens)")
    ax.set_ylabel(r"$t^{R}/t^{KV}$")
    ax.grid(True, which="both", alpha=0.15)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Bandwidth (Gbps)")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")


# ── Plot ──────────────────────────────────────────────────────────────────────
def main():
    T = 100_000  # context length for the plot
    bw = np.geomspace(0.1, 100, 500)  # Gbps

    df = pd.DataFrame(
        [
            {
                "Model": m.label,
                "bandwidth_gbps": b,
                "ratio": t_replay(m, T) / t_transfer(m, T, b),
            }
            for m in MODELS
            for b in bw
        ]
    )
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    lines = {}
    for m in MODELS:
        g = df[df["Model"] == m.label]
        (lines[m.label],) = ax.plot(
            g["bandwidth_gbps"],
            g["ratio"],
            color=m.color,
            ls=m.ls,
            lw=2.2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.6)
    ax.set_xlim(1e-1, 1e2)
    ax.set_ylim(*RATIO_YLIM)
    shade_regions(ax, bw, (0.04, 20), (0.63, 4e-3))

    ax.set_xlabel("Inter-site bandwidth (Gbps)")
    ax.set_ylabel(r"$t^{R}/t^{KV}$")
    ax.grid(True, which="both", alpha=0.15)
    # Legend follows the curves top to bottom. All lines share a slope, so this
    # is also the order they cross ratio = 1 going left to right.
    top_first = sorted(MODELS, key=lambda m: -t_replay(m, T) / t_transfer(m, T, bw[0]))
    ax.legend(
        [lines[m.label] for m in top_first],
        [m.label for m in top_first],
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig("migration_ratio.png", dpi=220, bbox_inches="tight")
    fig.savefig("migration_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote migration_ratio.png / .pdf")
    plot_context_ratio()

    # ── summary table ──
    print(
        f"\n{'Model':34s} {'KV KiB/tok':>11s} {'replay (s)':>10s} "
        f"{'xover Gbps':>11s}"
    )
    print("-" * 70)
    for m in MODELS:
        rp = t_replay(m, T)
        kv = m.kv_bytes(T)
        xo = kv * 8 / rp / 1e9
        print(f"{m.label:34s} {kv/T/1024:9.1f}   {rp:9.2f}   {xo:9.2f}")


if __name__ == "__main__":
    main()
