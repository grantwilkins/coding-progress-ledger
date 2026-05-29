"""Action regime diagram: replay fraction vs bandwidth, at two prefill speeds.

For each (lambda, rho) the staged pipeline picks an action mix; we plot the
replay fraction R / (R + S) of moved jobs. More bandwidth makes state transfer
(KV-cache, network-heavy) cheaper and pushes the mix toward state; faster
prefill makes replay cheaper and pushes it back up. The 50/50 crossover is the
endogenous boundary (Section 15): prefill speed slides it along the bandwidth
axis without changing the curve's shape.

The default-rho curve and a pessimistic rho=0.5 curve are computed here over an
extended lambda grid (the saved sweep tops out at lambda=8, before the
crossover) using the (model, log-T) class aggregation for speed.

Usage:
    cd evacuation && uv run python plot_action_regime.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from pipeline import run_pipeline

OUT = Path(__file__).resolve().parent / "outputs"
LAM = np.logspace(np.log10(0.25), np.log10(64.0), 18)


def _replay_frac(rho_scale: float) -> np.ndarray:
    out = []
    for lam in LAM:
        inst = build_instance(total_jobs=10_000, n_bins=160,
                              lambda_scale=float(lam), rho_scale=rho_scale)
        s3 = run_pipeline(inst).s3
        r, s = float(s3.x_R.sum()), float(s3.x_S.sum())
        out.append(r / (r + s) if r + s > 0 else np.nan)
    return np.array(out)


def _crossover(frac, target=0.5):
    """log-interpolated lambda where frac first crosses target."""
    f = frac - target
    for i in range(len(f) - 1):
        if f[i] * f[i + 1] < 0:
            a = f[i] / (f[i] - f[i + 1])
            return float(LAM[i] * (LAM[i + 1] / LAM[i]) ** a)
    return None


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for rho_scale, color, label in [(1.0, "#3a7ca5", "$\\rho$ default"),
                                    (0.5, "#c44536", "$\\rho \\times 0.5$ (slow prefill)")]:
        frac = _replay_frac(rho_scale)
        ax.plot(LAM, 100 * frac, marker="o", ms=4, lw=1.9, color=color, label=label)
        xc = _crossover(frac)
        if xc:
            ax.axvline(xc, color=color, ls=":", lw=1.2)
            ax.annotate(f"50/50 @ $\\Lambda$={xc:.1f}", (xc, 50), color=color, fontsize=8,
                        textcoords="offset points", xytext=(4, 6), rotation=90, va="bottom")
    ax.axhline(50, color="0.6", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("bandwidth scale $\\Lambda$")
    ax.set_ylabel("replay share of moved jobs  $R/(R+S)$  (%)")
    ax.set_title("Action regime: bandwidth sets the mix, prefill speed shifts the boundary")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "action_regime.pdf")
    fig.savefig(OUT / "action_regime.png", dpi=150)
    print(f"wrote {OUT / 'action_regime.pdf'}")


if __name__ == "__main__":
    main()
