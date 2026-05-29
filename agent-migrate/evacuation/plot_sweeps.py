"""Plot the parameter sweeps in `outputs/sweeps.json` (written by sweeps.py).

One figure per sweep, saved as `outputs/sweep_<name>.{pdf,png}`:
    seed         spread of Z*, phi*, H* across random instances
    rho_scale    feasibility / pressure / action mix vs prefill-speed scale
    W_rebalance  evacuation gap from uniform->demand-proportional warm pools
    sigma_scale  worst-class cost, per-model misses, heavy-tail fraction
    lambda_scale feasibility / pressure / action mix vs network-bandwidth scale
    total_jobs   feasibility / pressure / cost vs offered load

Usage:
    cd evacuation && uv run python plot_sweeps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"sweep_{name}.pdf")
    fig.savefig(OUT / f"sweep_{name}.png", dpi=150)
    print(f"wrote {OUT / f'sweep_{name}.pdf'}")


def _x(runs):
    return [r["sweep_value"] for r in runs]


def plot_seed(runs, diag):
    s = diag["seed"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, lab in zip(axes, ("Z_star", "phi_star", "H_star"),
                            ("$Z^\\star$ (unmoved jobs)", "$\\phi^\\star$ (peak pressure)",
                             "$H^\\star$ (worst-class cost, s)")):
        vals = [r[key] for r in runs]
        mean = float(np.mean(vals))
        std = s[key]["std"]
        ax.scatter(_x(runs), vals, s=22, color="#3a7ca5", zorder=3)
        ax.axhline(mean, color="black", lw=1.0)
        ax.axhspan(mean - std, mean + std, color="0.85", zorder=0)
        ax.set_xlabel("seed")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab.split(' ')[0]}  cv={s[key]['cv']:.3g}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Seed sweep: instance-to-instance variability", y=1.02)
    _save(fig, "seed")


def _feasibility_panel(ax, runs):
    ax.plot(_x(runs), [r["Z_star"] for r in runs], marker="o", color="#c44536",
            label="$Z^\\star$")
    ax.set_ylabel("$Z^\\star$ (unmoved jobs)", color="#c44536")
    ax.tick_params(axis="y", labelcolor="#c44536")
    ax2 = ax.twinx()
    ax2.plot(_x(runs), [100 * r["evac_fraction"] for r in runs], marker="s",
             color="#3a7ca5", label="evac %")
    ax2.set_ylabel("evacuated (%)", color="#3a7ca5")
    ax2.tick_params(axis="y", labelcolor="#3a7ca5")


def _pressure_panel(ax, runs):
    ax.plot(_x(runs), [r["phi_star"] for r in runs], marker="o", color="black",
            label="$\\phi^\\star$")
    ax.plot(_x(runs), [max(r["dest_net_pressure"]) for r in runs], marker="^",
            lw=1.2, label="max net pressure")
    ax.plot(_x(runs), [r["max_ing_pressure"] for r in runs], marker="v",
            lw=1.2, label="max ingest pressure")
    ax.set_ylabel("normalized pressure")
    ax.legend(fontsize=8)


def _action_panel(ax, runs):
    ax.plot(_x(runs), [r["total_replay"] for r in runs], marker="o",
            label="replay")
    ax.plot(_x(runs), [r["total_state"] for r in runs], marker="s",
            label="state transfer")
    ax.set_ylabel("jobs moved by action")
    ax.legend(fontsize=8)


def _scale_figure(runs, diag, name, xlabel, logx, flips=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _feasibility_panel(axes[0], runs)
    _pressure_panel(axes[1], runs)
    _action_panel(axes[2], runs)
    for ax in axes:
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    if flips:
        for ax in axes:
            for label, xv in flips.items():
                if xv is not None:
                    ax.axvline(xv, color="0.5", ls="--", lw=1.0)
    fig.suptitle(f"{name} sweep", y=1.02)
    _save(fig, name)


def plot_rho(runs, diag):
    d = diag["rho_scale"]
    _scale_figure(runs, diag, "rho_scale", "prefill-speed scale $\\rho$", logx=True,
                  flips={"flip": d["action_flip_point"],
                         "feas": d["feasibility_threshold"]})


def plot_lambda(runs, diag):
    d = diag["lambda_scale"]
    _scale_figure(runs, diag, "lambda_scale", "bandwidth scale $\\lambda$", logx=True,
                  flips={"flip": d["action_flip_point"],
                         "feas": d["feasibility_threshold"]})


def plot_total_jobs(runs, diag):
    _scale_figure(runs, diag, "total_jobs", "offered load (jobs)", logx=True)


def plot_w(runs, diag):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].plot(_x(runs), [r["Z_star"] for r in runs], marker="o", color="#c44536")
    axes[0].set_ylabel("$Z^\\star$ (unmoved jobs)")
    axes[0].set_title(f"evac gap = {diag['W_rebalance']['gap']:.2f} jobs")
    axes[1].plot(_x(runs), [r["phi_star"] for r in runs], marker="o", color="black")
    axes[1].set_ylabel("$\\phi^\\star$ (peak pressure)")
    for ax in axes:
        ax.set_xlabel("rebalance $\\alpha$  (uniform $\\to$ proportional $W$)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("W_rebalance sweep: warm-pool allocation", y=1.02)
    _save(fig, "W_rebalance")


def plot_sigma(runs, diag):
    d = diag["sigma_scale"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(_x(runs), d["H_star_curve"], marker="o", color="black")
    axes[0].set_ylabel("$H^\\star$ (worst-class cost, s)")
    for name, curve in d["model_z_curve"].items():
        axes[1].plot(_x(runs), curve, marker=".", label=name)
    axes[1].set_ylabel("unmoved jobs per model")
    axes[1].legend(fontsize=6)
    axes[2].plot(_x(runs), [100 * f for f in d["bucket_256k_1M_fraction"]],
                 marker="s", color="#3a7ca5")
    axes[2].set_ylabel("jobs in 256k–1M tokens (%)")
    for ax in axes:
        ax.set_xlabel("token-length spread $\\sigma$ scale")
        ax.grid(True, alpha=0.3)
    fig.suptitle("sigma_scale sweep: heavy-tail token lengths", y=1.02)
    _save(fig, "sigma_scale")


PLOTTERS = {"seed": plot_seed, "rho_scale": plot_rho, "W_rebalance": plot_w,
            "sigma_scale": plot_sigma, "lambda_scale": plot_lambda,
            "total_jobs": plot_total_jobs}


def main() -> None:
    payload = json.loads((OUT / "sweeps.json").read_text())
    runs, diag = payload["runs"], payload["diagnostics"]
    for name, plot in PLOTTERS.items():
        if name in runs:
            plot(runs[name], diag)


if __name__ == "__main__":
    main()
