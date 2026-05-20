"""
Migration decision ratio: t_replay / t_kv_transfer vs. inter-site bandwidth.

When the ratio > 1, replay is slower than shipping KV → ship KV.
When the ratio < 1, replay is faster → just replay the prompt.
The crossover at ratio = 1 is the phase boundary.

Hardware reference: 8× H100 SXM, dense bf16, MFU = 0.35.
KV sizes: bf16, from released architecture configs.
Prefill: 2·A·T (dense FFN) + L·H_q·(d_qk + d_v)·T² (causal attention),
         with architecture-specific attention scaling for compressed/hybrid models.
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

# ── Hardware ──────────────────────────────────────────────────────────────────
H100_BF16_DENSE_TFLOPS = 1_979 / 2          # dense = half of sparsity peak
N_GPUS    = 8
MFU       = 0.35
EFF_FLOPS = N_GPUS * H100_BF16_DENSE_TFLOPS * 1e12 * MFU   # ~2.77 PFLOP/s

BPE = 2   # bf16 bytes per element

# ── Model specs ───────────────────────────────────────────────────────────────
KVFn = Callable[[int], float]

@dataclass(frozen=True)
class Model:
    label: str               # display name
    active_b: float          # active params (billions)
    softmax_layers: int      # layers that produce per-token KV
    query_heads: int
    qk_dim: int
    v_dim: int
    kv_bytes: KVFn           # total bf16 KV bytes as f(tokens)
    attn_scale: float = 1.0  # effective sequence compression for attention FLOPs
    color: str = "k"
    ls: str = "-"

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
    idx_bytes   = 128 * BPE
    c4a_n   = math.ceil(T / 4)
    c128a_n = math.ceil(T / 128)
    return (c4a  * ((sw + c4a_n)   * entry_bytes + c4a_n * idx_bytes)
          + c128a * ((sw + c128a_n) * entry_bytes))

# ── Model catalogue ───────────────────────────────────────────────────────────
MODELS = [
    Model("DeepSeek-V3",
          active_b=37, softmax_layers=61, query_heads=128,
          qk_dim=192, v_dim=128,
          kv_bytes=mla_kv(61, 512, 64),
          color="#1f77b4"),

    Model("DeepSeek-V4 Pro",
          active_b=49, softmax_layers=61, query_heads=128,
          qk_dim=512, v_dim=512,
          kv_bytes=dsv4_kv,
          attn_scale=(30/4 + 31/128) / 61,
          color="#d62728",),

    Model("Qwen3 (235B)",
          active_b=22, softmax_layers=94, query_heads=64,
          qk_dim=128, v_dim=128,
          kv_bytes=gqa_kv(94, 4, 128),
          color="#2ca02c"),

    Model("Qwen3.5 (397B)",
          active_b=17, softmax_layers=15, query_heads=32,
          qk_dim=256, v_dim=256,
          kv_bytes=gqa_kv(15, 2, 256),
          color="#9467bd"),

    Model("Llama-3.1 (405B)",
          active_b=405, softmax_layers=126, query_heads=128,
          qk_dim=128, v_dim=128,
          kv_bytes=gqa_kv(126, 8, 128),
          color="#8c564b"),

    Model("GLM-5",
          active_b=40, softmax_layers=78, query_heads=64,
          qk_dim=256, v_dim=256,
          kv_bytes=mla_kv(78, 512, 64),
          color="#e377c2"),
]

# ── Cost model ────────────────────────────────────────────────────────────────
def prefill_flops(m: Model, T: int) -> float:
    ffn  = 2.0 * m.active_b * 1e9 * T
    attn = (m.softmax_layers * m.query_heads
            * (m.qk_dim + m.v_dim) * T**2 * m.attn_scale)
    return ffn + attn

def t_replay(m: Model, T: int) -> float:
    return prefill_flops(m, T) / EFF_FLOPS

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

def plot_glm5_context_ratio():
    bw_gbps = 1.0
    contexts = np.geomspace(1_000, 1_000_000, 500)
    df = context_ratio_frame("GLM-5", bw_gbps, contexts)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=df, x="context_tokens", y="ratio", color=model("GLM-5").color, lw=2.4, ax=ax)
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1_000, 1_000_000)
    ax.set_xlabel("Context size (tokens)")
    ax.set_ylabel(r"TTFT / Time to Transfer KV")
    ax.set_title("GLM-5 at 1 Gbps")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig("glm5_context_ratio_1gbps.png", dpi=220, bbox_inches="tight")
    fig.savefig("glm5_context_ratio_1gbps.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote glm5_context_ratio_1gbps.png / .pdf")

# ── Plot ──────────────────────────────────────────────────────────────────────
def main():
    T = 100_000                      # context length for the plot
    bw = np.geomspace(0.1, 100, 500) # Gbps

    df = pd.DataFrame(
        [
            {"Model": m.label, "bandwidth_gbps": b,
             "ratio": t_replay(m, T) / t_transfer(m, T, b)}
            for m in MODELS for b in bw
        ]
    )
    palette = {m.label: m.color for m in MODELS}

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(
        data=df, x="bandwidth_gbps", y="ratio", hue="Model",
        palette=palette, linewidth=2.2, ax=ax,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.6)
    yl = ax.get_ylim()
    ax.fill_between(bw, 1.0, yl[1], alpha=0.06, color="#B1040E", zorder=0)
    ax.fill_between(bw, yl[0], 1.0, alpha=0.06, color="#008566", zorder=0)
    ax.set_ylim(yl)

    ax.text(0.2, 40.5, "Transfer KV cache",
             color="#B1040E", ha="left", style="italic")
    ax.text(8.0, 0.05, "Transfer context",
     color="#008566", ha="left", style="italic")

    ax.set_xlabel("Inter-site bandwidth (Gbps)")
    ax.set_ylabel(r"TTFT / Time to Transfer KV")
    ax.grid(True, which="both", alpha=0.15)
    ax.set_xlim(1e-1,1e2)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", frameon=False)

    fig.tight_layout()
    fig.savefig("migration_ratio.png", dpi=220, bbox_inches="tight")
    fig.savefig("migration_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote migration_ratio.png / .pdf")
    plot_glm5_context_ratio()

    # ── summary table ──
    print(f"\n{'Model':34s} {'KV KiB/tok':>11s} {'replay (s)':>10s} "
          f"{'xover Gbps':>11s}")
    print("-" * 70)
    for m in MODELS:
        rp = t_replay(m, T)
        kv = m.kv_bytes(T)
        xo = kv * 8 / rp / 1e9
        print(f"{m.label:34s} {kv/T/1024:9.1f}   {rp:9.2f}   {xo:9.2f}")

if __name__ == "__main__":
    main()
