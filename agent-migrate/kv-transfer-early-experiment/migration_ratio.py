"""Public H200 TTFT divided by architecture-derived state-transfer time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize

CONTEXT_BANDWIDTHS_GBPS = np.linspace(0.1, 25, 150)
BytesFn = Callable[[int], int]


@dataclass(frozen=True)
class Benchmark:
    label: str
    released: str
    hardware: str
    state_bytes: BytesFn
    ttft_seconds: tuple[tuple[int, float], ...]
    cache_precision: str
    color: str


def qwen35_state(tokens: int) -> int:
    """10-layer BF16 GQA KV plus 30 Gated-DeltaNet recurrent/conv states."""
    kv = tokens * 10 * 2 * 2 * 256 * 2
    recurrent = 30 * 32 * 128 * 128 * 4
    conv = 30 * (2 * 16 * 128 + 32 * 128) * 4 * 2
    return kv + recurrent + conv


def kimi_k25_state(tokens: int) -> int:
    """BF16 MLA latent and RoPE key for all 61 layers."""
    return tokens * 61 * (512 + 64) * 2


BENCHMARKS = (
    Benchmark(
        "Qwen3.5-35B-A3B",
        "2026-02",
        "1x H200 SXM",
        qwen35_state,
        ((1024, 0.077), (8192, 0.2), (32768, 0.6), (65536, 1.6), (98304, 2.7), (131072, 4.2), (262144, 12.4)),
        "BF16 KV + FP32 recurrent state",
        "#7b2cbf",
    ),
    Benchmark("Kimi-K2.5", "2026-01", "8x H200", kimi_k25_state, ((1024, 0.112),), "BF16 MLA", "#008566"),
)

UNMODELED = (
    ("Inkling NVFP4", "2026-07", "No public no-cache TTFT"),
    ("GLM-5.2", "2026-07", "Published TTFT has 90% KV hits"),
    ("DeepSeek-V4-Pro", "2026-04", "No comparable public TTFT"),
    ("Kimi K3", "2026-07", "Weights/config pending"),
    ("Qwen3.7-Max", "2026-07", "Closed dimensions and no comparable TTFT"),
    ("Nemotron 3 Ultra", "2026-04", "No comparable public TTFT"),
)


def transfer_time(state_bytes: int, bandwidth_gbps: float) -> float:
    return state_bytes * 8 / (bandwidth_gbps * 1e9)


def ratio(ttft_seconds: float, state_bytes: int, bandwidth_gbps: float) -> float:
    return ttft_seconds / transfer_time(state_bytes, bandwidth_gbps)


def crossover_gbps(ttft_seconds: float, state_bytes: int) -> float:
    return state_bytes * 8 / ttft_seconds / 1e9


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="both", alpha=0.15)


def plot_context_ratio() -> None:
    benchmark = BENCHMARKS[0]
    contexts = np.array([point[0] for point in benchmark.ttft_seconds])
    ttfts = np.array([point[1] for point in benchmark.ttft_seconds])
    norm, cmap = Normalize(CONTEXT_BANDWIDTHS_GBPS.min(), CONTEXT_BANDWIDTHS_GBPS.max()), plt.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bandwidth in CONTEXT_BANDWIDTHS_GBPS:
        values = [ratio(ttft, benchmark.state_bytes(int(tokens)), bandwidth) for tokens, ttft in zip(contexts, ttfts)]
        ax.plot(contexts, values, marker=".", ms=2, color=cmap(norm(bandwidth)), lw=0.9, alpha=0.85)
    ax.axhline(1, color="k", lw=1.2, ls=":", alpha=0.6)
    ax.set(xscale="log", yscale="log", xlabel="Measured input context (tokens)", ylabel="Public TTFT / state-transfer time")
    style_axes(ax)
    ax.text(1.3e3, 3, "Transfer state", color="#B1040E", style="italic")
    ax.text(2.2e5, 0.03, "Replay context", color="#008566", ha="right", style="italic")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Bandwidth (Gbps)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"benchmark_context_ratio_bandwidths.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for benchmark in BENCHMARKS:
        contexts = [point[0] for point in benchmark.ttft_seconds]
        crossovers = [crossover_gbps(ttft, benchmark.state_bytes(tokens)) for tokens, ttft in benchmark.ttft_seconds]
        ax.plot(contexts, crossovers, "o-", label=f"{benchmark.label} ({benchmark.hardware})", color=benchmark.color, lw=2.2)
    ax.set(xscale="log", yscale="log", xlabel="Measured input context (tokens)", ylabel="Crossover bandwidth (Gbps)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"migration_ratio.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    plot_context_ratio()

    print(f"{'Model':22} {'context':>9} {'state':>10} {'TTFT':>8} {'xover':>9}")
    for benchmark in BENCHMARKS:
        for tokens, ttft in benchmark.ttft_seconds:
            state = benchmark.state_bytes(tokens)
            print(f"{benchmark.label:22} {tokens:9,d} {state / 1e9:8.2f}GB {ttft:7.3f}s {crossover_gbps(ttft, state):7.2f}G")
    for label, _, reason in UNMODELED:
        print(f"{label:22} {reason}")


if __name__ == "__main__":
    main()
