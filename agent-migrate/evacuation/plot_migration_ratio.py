"""Single-session replay/KV-transfer boundary (report Figure 1).

ratio = replay_time / transfer_time at fixed context T=100k tokens:
    replay   = beta*T/lambda + T/rho   (ship context bytes, then prefill)
    transfer = eta*T/lambda            (ship KV cache)
Crossover (ratio=1) at lambda* = rho*(eta - beta), matching Table 2.

MODELS, eta, beta, rho all come from instance.py (single source of truth), so
this figure can never drift from the paper's Qwen3 suite again.

Usage:
    cd evacuation && uv run python plot_migration_ratio.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import MODELS

OUT = Path(__file__).resolve().parent / "outputs"
T = 100_000.0
RHO_100K = MODELS[0].prefill_anchor_T[2]  # 100_000.0 anchor
assert RHO_100K == T
COLORS = ["#3a7ca5", "#c44536", "#4a9b54", "#e8943a", "#6a4c93", "#8a3122"]
BW_GBPS = np.geomspace(0.1, 100.0, 600)
LAMBDA = BW_GBPS * 1e9 / 8.0  # bytes/s


def rho_at_100k(m) -> float:
    return float(m.prefill_anchor_rho[2])


def ratio(m, lam):
    replay = m.beta_bytes_per_tok * T / lam + T / rho_at_100k(m)
    transfer = m.eta_bytes_per_tok * T / lam
    return replay / transfer


def crossover_gbps(m) -> float:
    return rho_at_100k(m) * (m.eta_bytes_per_tok - m.beta_bytes_per_tok) * 8.0 / 1e9


def main():
    print(f"{'Model':18s} {'crossover (Gbps)':>16s}")
    for m in MODELS:
        print(f"{m.name:18s} {crossover_gbps(m):16.2f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for m, c in zip(MODELS, COLORS):
        ax.plot(BW_GBPS, ratio(m, LAMBDA), color=c, lw=2.2, label=m.name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(BW_GBPS[0], BW_GBPS[-1])
    yl = (3e-2, 3e1)
    ax.axhline(1.0, color="k", lw=1.2, ls=":", alpha=0.7)
    ax.fill_between(BW_GBPS, 1.0, yl[1], color="#c44536", alpha=0.06, zorder=0)
    ax.fill_between(BW_GBPS, yl[0], 1.0, color="#4a9b54", alpha=0.06, zorder=0)
    ax.set_ylim(yl)

    ax.text(95, yl[1] / 1.3, "Transfer KV cache", color="#8a3122",
            style="italic", fontsize=14.5, ha="right", va="top")
    ax.text(95, yl[0] * 1.35, "Transfer context", color="#2f6b38",
            style="italic", fontsize=14.5, ha="right", va="bottom")

    ax.set_xlabel("Inter-site bandwidth (Gbps)", fontsize=19)
    ax.set_ylabel("Replay time / KV-transfer time", fontsize=19)
    ax.tick_params(labelsize=16)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=15, loc="upper left", framealpha=0.92, ncol=1)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "migration_ratio.pdf", bbox_inches="tight")
    fig.savefig(OUT / "migration_ratio.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'migration_ratio.png'}")


if __name__ == "__main__":
    main()
