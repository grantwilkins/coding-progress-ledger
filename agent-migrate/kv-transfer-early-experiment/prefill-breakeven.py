"""
KV_transfer vs prompt_replay timing for reconstituting an in_flight LLM
context at another site.

This is a deliberately small, auditable model for paper figures.  It is not a
serving benchmark.  The goal is to make the tradeoff legible:

    ship KV state over the network     vs.     ship/reuse text and replay prefill

Main corrections relative to the earlier draft:
  * Treat the H100 number as dense/no_sparsity bf16 Tensor Core peak. NVIDIA's
    public table reports 1,979 TFLOP/s bf16 Tensor Core with sparsity; dense is
    half of that.
  * Compute replay time from total prefill FLOPs at each context length, rather
    than assuming one constant token/s rate for every length.
  * Use released context windows and mark hypothetical points that exceed them.
  * Model KV bytes from the architecture actually described in the released
    configs/cards. In particular, GLM_5 uses MLA/DSA compressed KV, not full MHA;
    DeepSeek_V4_Pro uses CSA/HCA sequence_compressed KV, not DeepSeek_V3_style MLA.

Assumptions to keep explicit in the paper:
  * KV state is bf16 unless otherwise stated; quantized KV and allocator padding
    are not modeled.
  * MFU is an operator_tunable assumption, not a fact about vLLM or any model.
  * Prefill FLOPs are approximate: dense/MoE cost is 2 * active_params per token;
    causal softmax attention uses triangular average cost; linear_attention and
    recurrent state are treated as part of active_parameter cost, not token_linear
    KV. DeepSeek_V4 compressed_attention compute is approximated by its 4x/128x
    compression mix.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Reference hardware and stack assumptions
# -----------------------------------------------------------------------------

# NVIDIA H100 SXM public table: 1,979 TFLOP/s BF16 Tensor Core *with sparsity*.
# Dense/no_sparsity peak is half of that.  Use dense for conservative prefill.
H100_SXM_BF16_SPARSE_TFLOPS = 1_979.0
H100_SXM_BF16_DENSE_TFLOPS = H100_SXM_BF16_SPARSE_TFLOPS / 2.0
DEFAULT_N_GPUS = 8
DEFAULT_MFU = 0.35

BYTES_PER_KV_ELEM = 2  # bf16
DEFAULT_PLOT_CONTEXT = 100_000
DEFAULT_CONTEXTS = (1_000, 10_000, 100_000, 1_000_000)


# -----------------------------------------------------------------------------
# Model metadata
# -----------------------------------------------------------------------------

KVFn = Callable[[int], float]


@dataclass(frozen=True)
class Model:
    name: str
    active_b: float
    max_context: int
    native_context: int
    softmax_layers: int
    query_heads: int
    qk_dim: int
    v_dim: int
    kv_bytes_total: KVFn
    arch: str
    note: str = ""
    attention_scale: float = 1.0


def gqa_kv_bytes(layers: int, kv_heads: int, head_dim: int) -> KVFn:
    """Return total bf16 KV bytes for GQA/MHA softmax layers."""

    bytes_per_token = 2 * BYTES_PER_KV_ELEM * layers * kv_heads * head_dim
    return lambda tokens: tokens * bytes_per_token


def mla_kv_bytes(layers: int, kv_lora_rank: int = 512, qk_rope_dim: int = 64) -> KVFn:
    """Return total bf16 KV bytes for MLA_style compressed latent KV.

    MLA stores a compressed KV latent plus a RoPE key.  It does not store
    separate full K and V tensors for every KV head.
    """

    bytes_per_token = BYTES_PER_KV_ELEM * layers * (kv_lora_rank + qk_rope_dim)
    return lambda tokens: tokens * bytes_per_token


def deepseek_v4_pro_kv_bytes(tokens: int) -> float:
    """Approximate bf16 KV bytes for DeepSeek_V4_Pro at `tokens` context.

    vLLM's public arithmetic for 1M context models 30 c4a layers and 31 c128a
    layers.  Each cached shared_KV entry is 512 bf16 values; c4a layers also keep
    a 128_dim bf16 indexer cache per compressed entry.  Each layer keeps a small
    uncompressed sliding_window component of 128 entries.
    """

    c4a_layers = 30
    c128a_layers = 31
    sliding_window_entries = 128
    shared_kv_entry_bytes = 512 * BYTES_PER_KV_ELEM
    c4a_indexer_entry_bytes = 128 * BYTES_PER_KV_ELEM

    c4a_entries = math.ceil(tokens / 4)
    c128a_entries = math.ceil(tokens / 128)

    c4a_layer = (
        sliding_window_entries + c4a_entries
    ) * shared_kv_entry_bytes + c4a_entries * c4a_indexer_entry_bytes
    c128a_layer = (sliding_window_entries + c128a_entries) * shared_kv_entry_bytes
    return c4a_layers * c4a_layer + c128a_layers * c128a_layer


def deepseek_v4_attention_scale() -> float:
    """Approximate softmax_attention sequence compression for DeepSeek_V4_Pro.

    This is intentionally simple: 30 layers see roughly T/4 compressed entries;
    31 layers see roughly T/128 entries.  It ignores top_k sparsification and the
    small local window, so it is a readable approximation rather than a kernel
    model.
    """

    return (30 / 4 + 31 / 128) / 61


MODELS: tuple[Model, ...] = (
    Model(
        name="DeepSeek_V4_Pro",
        active_b=49.0,
        native_context=1_048_576,
        max_context=1_048_576,
        softmax_layers=61,
        query_heads=128,
        qk_dim=512,
        v_dim=512,
        kv_bytes_total=deepseek_v4_pro_kv_bytes,
        attention_scale=deepseek_v4_attention_scale(),
        arch="CSA/HCA compressed attention",
        note="49B active; 61 layers; c4a/c128a KV model",
    ),
    Model(
        name="Kimi_K2.6",
        active_b=32.0,
        native_context=262_144,
        max_context=262_144,
        softmax_layers=61,
        query_heads=64,
        qk_dim=192,  # qk_nope_head_dim + qk_rope_head_dim = 128 + 64
        v_dim=128,
        kv_bytes_total=mla_kv_bytes(layers=61, kv_lora_rank=512, qk_rope_dim=64),
        arch="MLA",
        note="1T total / 32B active; 256K context",
    ),
    Model(
        name="GLM_5",
        active_b=40.0,
        native_context=202_752,
        max_context=202_752,
        softmax_layers=78,
        query_heads=64,
        qk_dim=256,
        v_dim=256,
        kv_bytes_total=mla_kv_bytes(layers=78, kv_lora_rank=512, qk_rope_dim=64),
        arch="MLA + DSA",
        note="744B total / 40B active; not full_MHA KV",
    ),
    Model(
        name="Qwen3_235B_A22B",
        active_b=22.0,
        native_context=32_768,
        max_context=131_072,
        softmax_layers=94,
        query_heads=64,
        qk_dim=128,
        v_dim=128,
        kv_bytes_total=gqa_kv_bytes(layers=94, kv_heads=4, head_dim=128),
        arch="GQA",
        note="235B total / 22B active; 32K native, 131K with YaRN",
    ),
    Model(
        name="Qwen3.5_397B_A17B",
        active_b=17.0,
        native_context=262_144,
        max_context=1_010_000,
        softmax_layers=15,
        query_heads=32,
        qk_dim=256,
        v_dim=256,
        kv_bytes_total=gqa_kv_bytes(layers=15, kv_heads=2, head_dim=256),
        arch="hybrid DeltaNet + 15 softmax layers",
        note="397B total / 17B active; 262K native, extendable to ~1.01M",
    ),
    Model(
        name="Qwen3_Next_80B_A3B",
        active_b=3.0,
        native_context=262_144,
        max_context=1_010_000,
        softmax_layers=12,
        query_heads=16,
        qk_dim=256,
        v_dim=256,
        kv_bytes_total=gqa_kv_bytes(layers=12, kv_heads=2, head_dim=256),
        arch="hybrid DeltaNet + 12 softmax layers",
        note="80B total / 3B active in HF card",
    ),
)


# -----------------------------------------------------------------------------
# Cost model
# -----------------------------------------------------------------------------


def effective_flops(n_gpus: int, mfu: float) -> float:
    return n_gpus * H100_SXM_BF16_DENSE_TFLOPS * 1e12 * mfu


def dense_prefill_flops(model: Model, tokens: int) -> float:
    return 2.0 * model.active_b * 1e9 * tokens


def attention_prefill_flops(model: Model, tokens: int) -> float:
    """Approximate causal prefill attention FLOPs.

    For a causal prompt of length T, the average attended prefix length is about
    T/2.  Counting multiply_add as two FLOPs, QK and AV together give:

        L * Hq * (qk_dim + v_dim) * T^2

    The model_specific attention_scale is used only for sequence_compressed
    attention such as DeepSeek_V4's c4a/c128a mix.
    """

    return (
        model.softmax_layers
        * model.query_heads
        * (model.qk_dim + model.v_dim)
        * (tokens**2)
        * model.attention_scale
    )


def prefill_flops(model: Model, tokens: int) -> float:
    return dense_prefill_flops(model, tokens) + attention_prefill_flops(model, tokens)


def replay_time_s(model: Model, tokens: int, eff_flops: float) -> float:
    return prefill_flops(model, tokens) / eff_flops


def transmit_time_s(model: Model, tokens: int, bandwidth_gbps: float) -> float:
    bits = model.kv_bytes_total(tokens) * 8.0
    return bits / (bandwidth_gbps * 1e9)


def crossover_bw_gbps(model: Model, tokens: int, eff_flops: float) -> float:
    rp = replay_time_s(model, tokens, eff_flops)
    if rp <= 0:
        return float("inf")
    return model.kv_bytes_total(tokens) * 8.0 / rp / 1e9


def prefill_tok_s(model: Model, tokens: int, eff_flops: float) -> float:
    return tokens / replay_time_s(model, tokens, eff_flops)


# -----------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------


def make_sweep(
    models: Iterable[Model],
    contexts: Iterable[int],
    bandwidths_gbps: np.ndarray,
    eff_flops: float,
) -> pd.DataFrame:
    rows = []
    for m in models:
        for tokens in contexts:
            rp = replay_time_s(m, tokens, eff_flops)
            kv_gb = m.kv_bytes_total(tokens) / 1e9
            cb = crossover_bw_gbps(m, tokens, eff_flops)
            for bw in bandwidths_gbps:
                tx = transmit_time_s(m, tokens, float(bw))
                rows.append(
                    {
                        "model": m.name,
                        "tokens": tokens,
                        "within_context_window": tokens <= m.max_context,
                        "native_context": m.native_context,
                        "max_context": m.max_context,
                        "bandwidth_gbps": float(bw),
                        "kv_GB_decimal": kv_gb,
                        "t_transmit_s": tx,
                        "t_replay_s": rp,
                        "winner": "ship_kv" if tx < rp else "replay_prompt",
                        "prefill_tok_s_at_T": prefill_tok_s(m, tokens, eff_flops),
                        "crossover_bw_gbps_at_T": cb,
                        "arch": m.arch,
                        "note": m.note,
                    }
                )
    return pd.DataFrame(rows)


def print_summary(
    models: Iterable[Model],
    plot_context: int,
    eff_flops: float,
    n_gpus: int,
    mfu: float,
) -> None:
    print(
        f"=== Reference: {n_gpus}x H100 SXM, dense bf16 peak "
        f"{n_gpus * H100_SXM_BF16_DENSE_TFLOPS / 1000:.2f} PFLOP/s, "
        f"MFU={mfu:.0%}, effective={eff_flops / 1e15:.2f} PFLOP/s ==="
    )
    print(f"=== Plot/reference context: {plot_context:,} tokens ===\n")

    header = (
        f"{'Model':26s} {'active':>8s} {'ctx':>10s} {'KV@ctx':>10s} "
        f"{'replay':>10s} {'prefill':>11s} {'xover':>10s}  arch"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        kv_gb = m.kv_bytes_total(plot_context) / 1e9
        rp = replay_time_s(m, plot_context, eff_flops)
        tok_s = prefill_tok_s(m, plot_context, eff_flops)
        cb = crossover_bw_gbps(m, plot_context, eff_flops)
        ctx_mark = "ok" if plot_context <= m.max_context else "hyp"
        print(
            f"{m.name:26s} {m.active_b:6.1f}B {ctx_mark:>10s} "
            f"{kv_gb:8.2f}GB {rp:8.2f}s {tok_s:9.0f}t/s "
            f"{cb:8.2f}G  {m.arch}"
        )

    print(
        "\nContext marks: ok = within the released/advertised context window; hyp = hypothetical extrapolation."
    )


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, save_pdf: bool) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_migration_times(
    models: tuple[Model, ...],
    bandwidths_gbps: np.ndarray,
    plot_context: int,
    eff_flops: float,
    out_dir: Path,
    save_pdf: bool,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=False)

    for ax, m in zip(axes.flat, models):
        tx = np.array([transmit_time_s(m, plot_context, bw) for bw in bandwidths_gbps])
        rp = replay_time_s(m, plot_context, eff_flops)
        cb = crossover_bw_gbps(m, plot_context, eff_flops)

        ax.plot(bandwidths_gbps, tx, lw=2.0, label="ship KV")
        ax.axhline(rp, lw=1.6, ls="--", label="replay prompt")
        ax.axvline(cb, lw=0.9, ls=":")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ctx_note = "" if plot_context <= m.max_context else " (hyp.)"
        ax.set_title(f"{m.name}{ctx_note}\n{cb:.2g} Gbps crossover", fontsize=9.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Inter_site bandwidth (Gbps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time to runnable context (s)")

    axes[0, 0].legend(fontsize=8.5, loc="best", frameon=True)
    fig.suptitle(
        f"KV transfer vs prompt replay at {plot_context:,} tokens "
        f"({DEFAULT_N_GPUS}x H100 dense bf16, MFU={DEFAULT_MFU:.0%} unless overridden)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, out_dir, "migration_times", save_pdf)
    plt.close(fig)


def plot_crossover_bars(
    models: tuple[Model, ...],
    plot_context: int,
    eff_flops: float,
    out_dir: Path,
    save_pdf: bool,
) -> None:
    ordered = sorted(
        models, key=lambda m: crossover_bw_gbps(m, plot_context, eff_flops)
    )
    names = [m.name for m in ordered]
    xs = [crossover_bw_gbps(m, plot_context, eff_flops) for m in ordered]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(ordered))
    ax.barh(y, xs)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("Crossover bandwidth (Gbps); above this, shipping KV is faster")
    ax.set_title(
        f"Architecture_dependent KV_transfer crossover at {plot_context:,} tokens"
    )

    for ref_bw in (1, 10, 100, 400, 1000):
        ax.axvline(ref_bw, lw=0.6, ls=":")
        ax.text(ref_bw, len(ordered) - 0.35, f"{ref_bw:g}G", ha="center", fontsize=8)

    for i, m in enumerate(ordered):
        kv_per_token = m.kv_bytes_total(plot_context) / plot_context
        label = f"{xs[i]:.2g}G  |  {kv_per_token / 1024:.1f} KiB/token  |  {m.arch}"
        ax.text(xs[i] * 1.08, i, label, va="center", fontsize=8)

    ax.set_xlim(max(0.05, min(xs) / 2), max(1000, max(xs) * 8))
    fig.tight_layout()
    save_figure(fig, out_dir, "crossover_bandwidth", save_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir", type=Path, default=Path("."), help="output directory"
    )
    parser.add_argument(
        "--mfu",
        type=float,
        default=DEFAULT_MFU,
        help="model FLOP utilization assumption",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=DEFAULT_N_GPUS, help="number of H100 SXM GPUs"
    )
    parser.add_argument(
        "--plot_context_tokens",
        type=int,
        default=DEFAULT_PLOT_CONTEXT,
        help="context length used for the two explanatory plots",
    )
    parser.add_argument(
        "--bw_min", type=float, default=1.0, help="minimum bandwidth in Gbps"
    )
    parser.add_argument(
        "--bw_max", type=float, default=1000.0, help="maximum bandwidth in Gbps"
    )
    parser.add_argument(
        "--bw_points",
        type=int,
        default=240,
        help="number of log_spaced bandwidth points",
    )
    parser.add_argument(
        "--save_pdf", action="store_true", help="also save PDF versions of the figures"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eff = effective_flops(args.n_gpus, args.mfu)
    bandwidths = np.geomspace(args.bw_min, args.bw_max, args.bw_points)

    print_summary(MODELS, args.plot_context_tokens, eff, args.n_gpus, args.mfu)

    df = make_sweep(MODELS, DEFAULT_CONTEXTS, bandwidths, eff)
    csv_path = args.out_dir / "migration_sweep_corrected.csv"
    df.to_csv(csv_path, index=False)

    configure_plot_style()
    plot_migration_times(
        MODELS, bandwidths, args.plot_context_tokens, eff, args.out_dir, args.save_pdf
    )
    plot_crossover_bars(
        MODELS, args.plot_context_tokens, eff, args.out_dir, args.save_pdf
    )

    print("\nWrote:")
    print(f"  {csv_path}")
    print(f"  {args.out_dir / 'migration_times.png'}")
    print(f"  {args.out_dir / 'crossover_bandwidth.png'}")
    if args.save_pdf:
        print(f"  {args.out_dir / 'migration_times.pdf'}")
        print(f"  {args.out_dir / 'crossover_bandwidth.pdf'}")


if __name__ == "__main__":
    main()
