"""Single-session replay/KV-transfer boundary (report Figure 1).

ratio = replay_time / transfer_time as a function of context length T:
    replay   = beta*T/lambda + T/rho(T)   (ship context bytes, then prefill)
    transfer = eta*T/lambda               (ship KV cache)
With rho(T) = EFF/(2 N_act + C T) the crossover (ratio=1) sits at
    T* = (EFF (eta - beta)/lambda - 2 N_act) / C,
~318k tokens at the scenario's 8 Gbps WAN link: the body of the snapshot
distribution replays, only the long tail ships KV. Constants come from
instance.py / prefill.py (single source of truth).

Usage:
    cd evacuation && uv run python plot_migration_ratio.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import prefill
from instance import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, LAMBDA_BPS, MODEL

OUT = Path(__file__).resolve().parent / "outputs"
T = np.geomspace(1e3, 1e6, 600)
LAMBDAS_GBPS = (0.5, 1.0, 2.0, 4.0)  # bytes/s * 1e9; scenario link is 1.0
COLORS = ("#9bbcd4", "#3a7ca5", "#e8943a", "#c44536")


def ratio(lam: float) -> np.ndarray:
    replay = BETA_BYTES_PER_TOK * T / lam + T / prefill.rho(MODEL, T)
    transfer = ETA_BYTES_PER_TOK * T / lam
    return replay / transfer


def crossover_tokens(lam: float) -> float:
    a = prefill.ARCH[MODEL]
    return (prefill.EFF * (ETA_BYTES_PER_TOK - BETA_BYTES_PER_TOK) / lam
            - 2.0 * a.n_active) / prefill.attn_coef(a)


def main():
    print(f"{'link (GB/s)':>12s} {'crossover T* (tokens)':>22s}")
    for g in LAMBDAS_GBPS:
        print(f"{g:12.1f} {crossover_tokens(g * 1e9):22.0f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for g, c in zip(LAMBDAS_GBPS, COLORS):
        lw = 3.0 if g * 1e9 == LAMBDA_BPS else 1.8
        ax.plot(T, ratio(g * 1e9), color=c, lw=lw, label=f"{g:g} GB/s")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T[0], T[-1])
    yl = (3e-2, 3e1)
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.7)
    ax.fill_between(T, 1.0, yl[1], color="#c44536", alpha=0.06, zorder=0)
    ax.fill_between(T, yl[0], 1.0, color="#4a9b54", alpha=0.06, zorder=0)
    ax.set_ylim(yl)

    ax.text(9e5, yl[1] / 1.3, "Transfer KV cache", color="#8a3122",
            style="italic", fontsize=14.5, ha="right", va="top")
    ax.text(9e5, yl[0] * 1.35, "Replay context", color="#2f6b38",
            style="italic", fontsize=14.5, ha="right", va="bottom")

    ax.set_xlabel("Context length $T$ (tokens)", fontsize=19)
    ax.set_ylabel("Replay time / KV-transfer time", fontsize=19)
    ax.tick_params(labelsize=16)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=14, loc="upper left", framealpha=0.92, title="link rate")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "migration_ratio.pdf", bbox_inches="tight")
    fig.savefig(OUT / "migration_ratio.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'migration_ratio.png'}")


if __name__ == "__main__":
    main()
