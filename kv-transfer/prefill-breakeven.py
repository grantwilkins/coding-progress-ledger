"""
KV-transfer vs context-replay timing for migrating an in-flight LLM request
between two sites at a fixed inter-site bandwidth.

Single, rigorous reference setup — no mixed benchmark sources:
  Reference HW:        8× NVIDIA H100 SXM5, bf16   (peak 7.91 PFLOPS)
  Reference engine:    vLLM with chunked prefill   (assumed MFU = 0.35)
  Reference context:   100,000 tokens (the prefill rate is evaluated here)

Per-model prefill rate is derived analytically from architecture:

  prefill_FLOPs/token(T)  =  2·active_params              (dense forward)
                          +  4·L_softmax·n_q·head_dim·T   (attention; MAC×2)
  prefill_tok_s           =  effective_FLOPS / prefill_FLOPs/token(T_ref)

Strategy A — ship the KV cache:    t_A = T · kv_bpt · 8 / bw_bps
Strategy B — ship prompt + replay: t_B = T / prefill_tok_s
Crossover (T cancels):             bw* = 8 · kv_bpt · prefill_tok_s

Above bw*, ship the KV. Below, replay.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Reference hardware + serving stack — single source of truth
# ============================================================================

H100_PEAK_FLOPS_BF16 = 989e12     # NVIDIA H100 SXM5 spec
N_GPUS = 8
MFU = 0.35                         # vLLM chunked prefill, production-grade
EFFECTIVE_FLOPS = N_GPUS * H100_PEAK_FLOPS_BF16 * MFU   # 2.77 PFLOPS

T_REF_FOR_PREFILL = 100_000        # context at which prefill rate is evaluated


# ============================================================================
# Model catalog
# Specs from HuggingFace config.json (April 2026). All architecture numbers
# are facts, not benchmarks — the only judgement call is MFU above.
# ============================================================================

@dataclass
class Model:
    name: str
    active_B: float    # active params, billions
    kv_bpt: int        # KV bytes per token (bf16), accounting for MLA / hybrid
    L_softmax: int     # number of softmax-attention layers
    n_q: int           # query heads per softmax layer
    head_dim: int      # attention head dim
    arch: str          # short architecture note


MODELS = [
    Model("DeepSeek-V4-Pro", active_B=49.0, kv_bpt=70_272,
          L_softmax=61, n_q=128, head_dim=512,
          arch="MLA, latent=512+RoPE64"),
    Model("Kimi-K2.6", active_B=32.0, kv_bpt=70_272,
          L_softmax=61, n_q=64, head_dim=192,
          arch="MLA, hd=v128+rope64"),
    Model("GLM-5", active_B=40.0, kv_bpt=1_277_952,
          L_softmax=78, n_q=64, head_dim=64,
          arch="full MHA n_kv=64"),
    Model("Qwen3-235B-A22B", active_B=22.0, kv_bpt=192_512,
          L_softmax=94, n_q=64, head_dim=128,
          arch="GQA n_kv=4"),
    Model("Qwen3.5-397B-A17B", active_B=17.0, kv_bpt=30_720,
          L_softmax=15, n_q=32, head_dim=256,
          arch="hybrid 15/60 softmax"),
    Model("Qwen3-Next-80B-A3B", active_B=3.9, kv_bpt=24_576,
          L_softmax=12, n_q=16, head_dim=256,
          arch="hybrid 12/48 softmax"),
]


# ============================================================================
# Math
# ============================================================================

def prefill_flops_per_token(m, T):
    dense = 2 * m.active_B * 1e9
    attn = 4 * m.L_softmax * m.n_q * m.head_dim * T
    return dense + attn


def prefill_tok_s(m, T=T_REF_FOR_PREFILL):
    return EFFECTIVE_FLOPS / prefill_flops_per_token(m, T)


def t_transmit(bw_gbps, T, m):
    return T * m.kv_bpt * 8 / (bw_gbps * 1e9)


def t_replay(T, m):
    return T / prefill_tok_s(m)


def crossover_bw_gbps(m):
    return 8 * m.kv_bpt * prefill_tok_s(m) / 1e9


# ============================================================================
# Sweep
# ============================================================================

BANDWIDTHS_GBPS = np.arange(1, 1001, 5)
CONTEXT_LENGTHS = [1_000, 10_000, 100_000, 1_000_000]

rows = []
for m in MODELS:
    rate = prefill_tok_s(m)
    cb = crossover_bw_gbps(m)
    for T in CONTEXT_LENGTHS:
        for bw in BANDWIDTHS_GBPS:
            tA = t_transmit(bw, T, m)
            tB = t_replay(T, m)
            rows.append({
                "model": m.name,
                "tokens": T,
                "bandwidth_gbps": bw,
                "kv_GB": T * m.kv_bpt / 1e9,
                "t_transmit_s": tA,
                "t_replay_s": tB,
                "winner": "transmit" if tA < tB else "replay",
                "prefill_tok_s": rate,
                "crossover_bw_gbps": cb,
            })

df = pd.DataFrame(rows)
df.to_csv("migration.csv", index=False)


# ============================================================================
# Stdout: derivation table
# ============================================================================

print(f"=== Reference: 8×H100 bf16  ·  peak {N_GPUS*H100_PEAK_FLOPS_BF16/1e15:.2f} PFLOPS"
      f"  ·  MFU {MFU:.0%}  ·  effective {EFFECTIVE_FLOPS/1e15:.2f} PFLOPS ===")
print(f"=== Prefill rate evaluated at T_ref = {T_REF_FOR_PREFILL:,} tokens ===\n")
print(f"{'Model':22s} {'active':>8s} {'kv/tok':>9s}  "
      f"{'dense':>10s} {'attn':>10s}  {'prefill':>10s} {'crossover':>10s}  "
      f"arch")
print("-" * 115)
for m in MODELS:
    dense = 2 * m.active_B * 1e9
    attn = 4 * m.L_softmax * m.n_q * m.head_dim * T_REF_FOR_PREFILL
    rate = prefill_tok_s(m)
    cb = crossover_bw_gbps(m)
    print(f"{m.name:22s} {m.active_B:>6.1f}B  {m.kv_bpt/1024:>6.1f} KB  "
          f"{dense/1e12:>7.2f} TF  {attn/1e12:>7.2f} TF  "
          f"{rate:>7.0f} t/s  {cb:>7.1f} Gbps  {m.arch}")

print("\n=== Migration time at 100 Gbps ===")
print(f"{'Model':22s} " + " ".join(f"{T//1000:>5d}k tok" for T in CONTEXT_LENGTHS)
      + "   |  replay 1M")
for m in MODELS:
    tx = " ".join(f"{t_transmit(100, T, m):>8.2f}s" for T in CONTEXT_LENGTHS)
    print(f"{m.name:22s} {tx}   |  {t_replay(1_000_000, m):>5.0f}s")


# ============================================================================
# Plot styling
# ============================================================================

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
COLORS = plt.cm.tab10(np.arange(len(MODELS)))


# ============================================================================
# Plot 1: small multiples — one panel per model, fixed at 1M-token context
# ============================================================================

T_REF = 1_000_000

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
for ax, m in zip(axes.flat, MODELS):
    t_tx = T_REF * m.kv_bpt * 8 / (BANDWIDTHS_GBPS * 1e9)
    t_rp = t_replay(T_REF, m)
    cb = crossover_bw_gbps(m)

    ax.fill_between(BANDWIDTHS_GBPS, np.minimum(t_tx, t_rp), np.maximum(t_tx, t_rp),
                    where=(t_tx < t_rp), color="tab:green", alpha=0.18)
    ax.fill_between(BANDWIDTHS_GBPS, np.minimum(t_tx, t_rp), np.maximum(t_tx, t_rp),
                    where=(t_tx >= t_rp), color="tab:orange", alpha=0.18)

    ax.plot(BANDWIDTHS_GBPS, t_tx, color="tab:blue", lw=2.2, label="ship KV")
    ax.axhline(t_rp, color="tab:red", lw=1.8, ls="--", label="re-prefill")
    ax.axvline(cb, color="black", lw=0.7, ls=":", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{m.name}\ncrossover {cb:.1f} Gbps · replay {t_rp:.0f}s",
                 fontsize=10)

for ax in axes[-1, :]:
    ax.set_xlabel("Bandwidth (Gbps)")
for ax in axes[:, 0]:
    ax.set_ylabel("Migration time (s)")

axes[0, 0].legend(fontsize=9, loc="upper right", frameon=True)
fig.suptitle(
    f"Migration time at {T_REF:,}-token context  (8×H100, MFU=35%).  "
    "Green = ship KV.  Orange = re-prefill.",
    fontsize=12, y=1.00,
)
fig.tight_layout()
fig.savefig("transmission_time.png", dpi=200, bbox_inches="tight")
fig.savefig("transmission_time.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================================
# Plot 2: crossover bandwidth bar chart
# ============================================================================

ordered = sorted(MODELS, key=crossover_bw_gbps)
xs = [crossover_bw_gbps(m) for m in ordered]
names = [m.name for m in ordered]
colors_ord = [COLORS[MODELS.index(m)] for m in ordered]

fig, ax = plt.subplots(figsize=(11, 5.5))
ys = np.arange(len(ordered))
ax.barh(ys, xs, color=colors_ord, alpha=0.85)
ax.set_yticks(ys)
ax.set_yticklabels(names)
ax.set_xscale("log")
ax.set_xlabel("Crossover bandwidth (Gbps) — above this, KV-transfer is faster than replay")
ax.set_title(
    f"When does KV-transfer overtake context replay?  "
    f"(8×H100, MFU={MFU:.0%}, T_ref={T_REF_FOR_PREFILL:,})"
)

for ref_bw, lbl in [(10, "10 G"), (100, "100 G"), (400, "400 G"), (1000, "1 T")]:
    ax.axvline(ref_bw, color="gray", lw=0.5, ls=":", alpha=0.6)
    ax.text(ref_bw, len(ordered) - 0.3, lbl, fontsize=8, color="gray", ha="center")

for i, m in enumerate(ordered):
    ax.text(xs[i] * 1.08, i,
            f"  {m.kv_bpt/1024:>6.1f} KB/tok  ·  {prefill_tok_s(m):>6.0f} tok/s  ·  {m.arch}",
            va="center", fontsize=8.5, color="black", family="monospace")

ax.set_xlim(0.3, 1e3)
fig.tight_layout()
fig.savefig("crossover_bandwidth.png", dpi=200, bbox_inches="tight")
fig.savefig("crossover_bandwidth.pdf", bbox_inches="tight")
plt.close(fig)


print("\nWrote:")
print("  migration.csv")
print("  transmission_time.{png,pdf}")
print("  crossover_bandwidth.{png,pdf}")
