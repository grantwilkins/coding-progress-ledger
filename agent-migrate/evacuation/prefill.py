"""First-principles prefill-rate model for the Qwen3 suite on 8x H100 (TP=8).

Prefill is compute-bound under serving load (a long sequence is itself a large
token batch), so per-instance throughput follows the FLOP roofline

    rho(T) = EFF / (2 N_act  +  C T),     C = 2 L_attn H_q d_head,          (tok/s)

where the FFN/projection cost is 2 N_act FLOP per token and causal attention adds
C FLOP per token, growing linearly in context length T. EFF = G * peak * MFU is
the sustained 8-GPU compute. We use BF16 (H100 SXM peak 989.5 TFLOP/s) and
MFU = 0.35, which reproduces the published Qwen3-235B and Qwen3-Next 8x H100
prefill rates exactly and is consistent (slightly conservative on a per-GPU basis,
as TP=8 pays comms) with the measured single-GPU A3B prefill peaks of 45-57k tok/s.
See assumptions.md for the calibration sources.

Two regimes set the *prefill bound* per model:
  - short context (T < T* = 2 N_act / C): FFN-bound, flat ceiling EFF / (2 N_act);
  - long context (T > T*): attention-bound, rho ~ EFF / (C T), decaying as 1/T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H100_BF16_FLOPS = 989.5e12   # H100 SXM, BF16 dense peak (sparsity off)
N_GPU = 8                    # TP=8, one warm instance
MFU = 0.35                   # sustained utilization (calibrated, see module docstring)
EFF = N_GPU * H100_BF16_FLOPS * MFU  # ~2.77 PFLOP/s sustained per instance

ANCHOR_T = np.array([1_000.0, 10_000.0, 100_000.0, 1_000_000.0])


@dataclass(frozen=True)
class Arch:
    n_active: float   # active params (= total for dense models)
    attn_layers: int  # full-attention layers (Qwen3-Next: 12 of 48; rest: all)
    q_heads: int      # query heads (the T^2 attention term scales with these, not KV heads)
    head_dim: int


# Full Qwen3 lineup, configs from each model's published HF config.json.
ARCH: dict[str, Arch] = {
    "Qwen3-0.6B":          Arch(0.6e9, 28, 16, 128),
    "Qwen3-1.7B":          Arch(1.7e9, 28, 16, 128),
    "Qwen3-4B":            Arch(4.0e9, 36, 32, 128),
    "Qwen3-8B":            Arch(8.0e9, 36, 32, 128),
    "Qwen3-14B":           Arch(14.0e9, 40, 40, 128),
    "Qwen3-32B":           Arch(32.0e9, 64, 64, 128),
    "Qwen3-30B-A3B":       Arch(3.0e9, 48, 32, 128),
    "Qwen3-235B-A22B":     Arch(22.0e9, 94, 64, 128),
    "Qwen3-Next-80B-A3B":  Arch(3.0e9, 12, 16, 256),  # 36 linear layers add no T^2 term
}


def attn_coef(a: Arch) -> float:
    """C: causal-attention FLOP per token-of-context (the 1/T term's numerator)."""
    return 2.0 * a.attn_layers * a.q_heads * a.head_dim


def rho(name: str, T) -> np.ndarray:
    """Prefill rate (tok/s) at context length T for one 8x H100 instance."""
    a = ARCH[name]
    return EFF / (2.0 * a.n_active + attn_coef(a) * np.asarray(T, float))


def anchors(name: str) -> np.ndarray:
    """Prefill rate at the four canonical context anchors (1k..1M)."""
    return rho(name, ANCHOR_T)


def ffn_ceiling(name: str) -> float:
    """Flat short-context prefill ceiling EFF / (2 N_act) (tok/s)."""
    return EFF / (2.0 * ARCH[name].n_active)


def attn_crossover(name: str) -> float:
    """Context T* = 2 N_act / C where attention cost overtakes FFN cost (tokens)."""
    return 2.0 * ARCH[name].n_active / attn_coef(ARCH[name])


def main() -> None:
    print(f"EFF = 8x H100 BF16 x MFU={MFU} = {EFF:.3e} FLOP/s (per TP=8 instance)\n")
    hdr = f"{'model':20s}" + "".join(f"{f'@{t:g}':>9s}" for t in ANCHOR_T)
    print(hdr + f"  {'ceiling':>9s} {'T*(tok)':>9s}")
    for name in ARCH:
        rrow = "".join(f"{r:9.0f}" for r in anchors(name))
        print(f"{name:20s}{rrow}  {ffn_ceiling(name):9.0f} {attn_crossover(name):9.0f}")
    print("\nceiling = max prefill rate (short context); T* = context where prefill "
          "starts decaying as 1/T.")


if __name__ == "__main__":
    main()
