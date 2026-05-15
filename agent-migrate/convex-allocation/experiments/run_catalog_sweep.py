from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines import solve_replay_only, solve_state_only
from catalog import catalog_models
from coefficients import ACTIONS, compute_coefficients
from cvxpy_solver import solve_cvxpy
from metrics import action_mix, shed_action_mix, shed_achieved, utilization
from mirror_descent import solve_mirror_descent
from problem import make_problem, saturation_diagnostics

REGIMES = ("bandwidth-spread", "prefill-spread", "background-load-spread")
MODEL_LABELS = {
    "DeepSeek-V4-Pro": "DeepSeek",
    "GLM-5": "GLM-5",
    "Qwen3-Next-80B-A3B": "Qwen3-Next",
}
MODEL_COLORS = {
    "DeepSeek-V4-Pro": "#4c78a8",
    "GLM-5": "#f58518",
    "Qwen3-Next-80B-A3B": "#54a24b",
}
REGIME_LABELS = {
    "bandwidth-spread": "Link bandwidth varies",
    "prefill-spread": "Prefill load varies",
    "background-load-spread": "Background load + cache",
}
DEST_LABELS = {
    "bandwidth-spread": ("1 Gbps", "10 Gbps", "100 Gbps"),
    "prefill-spread": ("20% prefill load", "50% prefill load", "80% prefill load"),
    "background-load-spread": ("low network load", "balanced load", "cached context"),
}
ACTION_COLORS = {"replay_frac": "#4c78a8", "state_frac": "#f58518", "stay_frac": "#c9c2bd"}
REGIME_XLABELS = ("bandwidth", "prefill load", "background load")


def _policy_row(model, regime, policy, result, problem, coeffs, diagnostics):
    net_util, prefill_util = utilization(problem, coeffs, result.allocation if hasattr(result, "allocation") else result.y)
    y = result.allocation if hasattr(result, "allocation") else result.y
    mix = action_mix(problem, y)
    shed_mix = shed_action_mix(problem, y)
    feasible = getattr(result, "feasible", True)
    return {
        "model": model,
        "regime": regime,
        "policy": policy,
        "objective": f"{result.objective:.10g}" if feasible else "INFEASIBLE",
        "shed_achieved": f"{shed_achieved(problem, y):.10g}",
        "max_net_util": f"{float(np.max(net_util)):.10g}",
        "max_prefill_util": f"{float(np.max(prefill_util)):.10g}",
        **{key: f"{value:.10g}" for key, value in mix.items()},
        **{key: f"{value:.10g}" for key, value in shed_mix.items()},
        "replay_demand_over_capacity": f"{diagnostics[0]:.10g}",
        "state_bytes_over_capacity": f"{diagnostics[1]:.10g}",
        "feasible": str(bool(feasible)),
    }


def run_sweep():
    out = ROOT / "outputs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = {}

    for model in catalog_models():
        for regime in REGIMES:
            problem = make_problem(model, regime)
            coeffs = compute_coefficients(problem)
            diagnostics = saturation_diagnostics(problem)
            print(f"{model.name} / {regime}: replay={diagnostics[0]:.3f}, state={diagnostics[1]:.3f}")
            cvx = solve_cvxpy(problem)
            md = solve_mirror_descent(problem, iterations=5000, eta_x0=2.0, eta_l0=0.2, max_backtracks=20)
            replay = solve_replay_only(problem)
            state = solve_state_only(problem)
            results = {
                "CVXPY": cvx,
                "mirror-descent-best": md,
                "replay-only": replay,
                "state-only": state,
            }
            for policy, result in results.items():
                rows.append(_policy_row(model.name, regime, policy, result, problem, coeffs, diagnostics))
            cells[(model.name, regime)] = (problem, coeffs, results)

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plot_headline(cells, out)
    plot_heatmaps(cells, out)
    plot_utilization(cells, out)
    plot_policy_objectives(cells, out)
    plot_convergence(cells, out)
    plot_convergence_grid(cells, out)
    plot_crossover(out)


def plot_headline(cells, out):
    fig, ax = plt.subplots(figsize=(8.7, 5.2), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for model in catalog_models():
        replay_share = []
        for regime in REGIMES:
            problem, _, results = cells[(model.name, regime)]
            replay_share.append(shed_action_mix(problem, results["CVXPY"].y)["replay_shed_frac"])
        ax.plot(
            x,
            replay_share,
            marker="o",
            linewidth=2.6,
            color=MODEL_COLORS[model.name],
            label=MODEL_LABELS[model.name],
        )
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.fill_between([-0.15, len(REGIMES) - 0.85], 0, 0.5, color=ACTION_COLORS["state_frac"], alpha=0.08)
    ax.fill_between([-0.15, len(REGIMES) - 0.85], 0.5, 1.0, color=ACTION_COLORS["replay_frac"], alpha=0.08)
    ax.text(2.05, 0.82, "replay-heavy", ha="right", color=ACTION_COLORS["replay_frac"], fontsize=12)
    ax.text(2.05, 0.18, "state-heavy", ha="right", color="#b25a00", fontsize=12)
    ax.set_xticks(x, REGIME_XLABELS)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("replay share of shed work")
    ax.set_title("Replay share of shed work by regime")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out / "headline_action_mix.png", dpi=200)
    plt.close(fig)


def plot_heatmaps(cells, out):
    chosen = max(
        cells,
        key=lambda key: min(
            shed_action_mix(cells[key][0], cells[key][2]["CVXPY"].y)["replay_shed_frac"],
            shed_action_mix(cells[key][0], cells[key][2]["CVXPY"].y)["state_shed_frac"],
        ),
    )
    model, regime = chosen
    problem, _, results = cells[chosen]
    data = results["CVXPY"].y / problem.d[:, None]
    cols = [f"{dest}\n{action}" for dest in DEST_LABELS[regime] for action in ACTIONS] + ["Stay"]
    rows = [f"{int(t):,} tokens\n{int(d)} requests" for t, d in zip(problem.T, problem.d)]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    sns.heatmap(
        data,
        vmin=0,
        vmax=1,
        cmap="magma",
        ax=ax,
        xticklabels=cols,
        yticklabels=rows,
        cbar_kws={"label": "fraction of class"},
    )
    ax.set_title(f"Class allocation for the most mixed cell: {MODEL_LABELS[model]}, {REGIME_LABELS[regime]}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_ylabel("workload class")
    ax.tick_params(length=0)
    fig.savefig(out / "allocation_heatmap_per_scenario.png", dpi=200)
    plt.close(fig)


def plot_utilization(cells, out):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for model in catalog_models():
        net = []
        prefill = []
        for regime in REGIMES:
            problem, coeffs, results = cells[(model.name, regime)]
            u_net, u_prefill = utilization(problem, coeffs, results["CVXPY"].y)
            net.append(float(np.max(u_net)))
            prefill.append(float(np.max(u_prefill)))
        color = MODEL_COLORS[model.name]
        ax.plot(x, net, marker="o", color=color, linewidth=2.4, label=f"{MODEL_LABELS[model.name]} network")
        ax.plot(x, prefill, marker="s", color=color, linewidth=2.4, linestyle="--", label=f"{MODEL_LABELS[model.name]} prefill")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, REGIME_XLABELS)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("max destination utilization")
    ax.set_title("Resource ceilings used by the convex optimum")
    ax.legend(frameon=False, fontsize=10, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out / "utilization_vs_policy.png", dpi=200)
    plt.close(fig)


def plot_policy_objectives(cells, out):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for model in catalog_models():
        replay_ratio = []
        state_ratio = []
        for regime in REGIMES:
            _, _, results = cells[(model.name, regime)]
            cvx_obj = results["CVXPY"].objective
            replay_ratio.append(results["replay-only"].objective / cvx_obj if results["replay-only"].feasible else np.nan)
            state_ratio.append(results["state-only"].objective / cvx_obj if results["state-only"].feasible else np.nan)
        color = MODEL_COLORS[model.name]
        ax.plot(x, replay_ratio, marker="o", color=color, linewidth=2.4, label=f"{MODEL_LABELS[model.name]} replay greedy")
        ax.plot(x, state_ratio, marker="s", color=color, linewidth=2.4, linestyle="--", label=f"{MODEL_LABELS[model.name]} state greedy")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, REGIME_XLABELS)
    ax.set_ylabel("objective / CVXPY optimum")
    ax.set_title("Greedy objective relative to CVXPY")
    ax.set_ylim(0.9, 5.4)
    ax.legend(frameon=False, fontsize=10, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out / "objective_vs_policy.png", dpi=200)
    plt.close(fig)


def plot_convergence(cells, out):
    key = ("Qwen3-Next-80B-A3B", "prefill-spread")
    problem, _, results = cells[key]
    hist = results["mirror-descent-best"].history
    t = np.arange(1, hist["shed"].size + 1)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(t, hist["shed"] / problem.B_shed, color=MODEL_COLORS[key[0]], linewidth=2.6)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("shed achieved / target")
    ax.set_title("Mirror descent reaches the shed constraint quickly")
    fig.savefig(out / "convergence_one_scenario.png", dpi=200)
    plt.close(fig)


def plot_convergence_grid(cells, out):
    regime = "prefill-spread"
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for model in catalog_models():
        _, _, results = cells[(model.name, regime)]
        cvx_obj = results["CVXPY"].objective
        hist = results["mirror-descent-best"].history
        gap = np.maximum((hist["best_objective"] - cvx_obj) / max(1.0, abs(cvx_obj)), 1e-12)
        gap[~np.isfinite(gap)] = np.nan
        ax.plot(np.arange(1, gap.size + 1), gap, color=MODEL_COLORS[model.name], linewidth=2.4, label=MODEL_LABELS[model.name])
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best feasible relative objective gap")
    ax.set_title("Mirror descent objective convergence in the prefill-load regime")
    ax.legend(frameon=False)
    fig.savefig(out / "convergence_grid.png", dpi=200)
    plt.close(fig)


def plot_crossover(out):
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    y = np.arange(len(catalog_models()))
    for row, model in zip(y, catalog_models()):
        computed = model.crossover_gbps
        published = model.published_crossover_gbps
        ax.hlines(row, 0.1, computed, color="#4c78a8", linewidth=6, alpha=0.75, label="replay faster" if row == 0 else None)
        ax.hlines(row, computed, 220, color="#f58518", linewidth=6, alpha=0.75, label="state faster" if row == 0 else None)
        ax.plot(computed, row, marker="o", color="black", markersize=5, label="computed crossover" if row == 0 else None)
        ax.plot(published, row, marker="|", color="black", markersize=14, label="FINDINGS.md crossover" if row == 0 else None)
        ax.text(computed * 1.08, row + 0.13, f"{computed:.1f} Gbps", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 220)
    ax.set_yticks(y, [MODEL_LABELS[m.name] for m in catalog_models()])
    ax.set_xlabel("bandwidth (Gbps)")
    ax.set_title("Single-request replay/state crossover")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(out / "crossover_recovery.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_sweep()
