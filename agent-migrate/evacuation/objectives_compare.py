"""Compare Stage 1 evacuation objectives on the same convex resource model.

Four variants over one base feasible set: throughput, max-min, and proportional
fairness under both utility weightings (w_q = n_q population, w_q = 1 class).
Each is solved Stage 1 -> Stage 2 at a fixed deadline; prints a comparison table
(job / token / KV evacuation + fairness floors), writes
outputs/objective_metrics.{json,csv}, and renders the comparison plots. With
--sweep it also renders the deadline-sweep plot.

Usage:
    cd evacuation && uv run python objectives_compare.py [--D 120] [--sweep]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from objective_metrics import (BUCKET_LABELS, by_bucket, evac_fraction,
                               evac_summary, metrics, model_token_grid)
from stage1 import solve_stage1
from stage2 import solve_stage2


@dataclass(frozen=True)
class Variant:
    key: str
    objective: str
    weights: str
    label: str
    color: str


VARIANTS = (
    Variant("throughput", "throughput", "population", "throughput", "#3a7ca5"),
    Variant("max_min", "max_min", "population", "max-min", "#c44536"),
    Variant("prop_fair_pop", "prop_fair", "population", "prop-fair $w_q{=}n_q$", "#4c9a52"),
    Variant("prop_fair_cls", "prop_fair", "class", "prop-fair $w_q{=}1$", "#8052a0"),
)
N_BINS = 5
SWEEP_D = (30, 60, 120, 300, 600, 1200)
OUT = Path(__file__).resolve().parent / "outputs"


def solve_all(D):
    inst = build_instance(D=D, n_bins=N_BINS)
    runs = {}
    for v in VARIANTS:
        s1 = solve_stage1(inst, v.objective, v.weights)
        runs[v.key] = (s1, solve_stage2(inst, s1))
    return inst, runs


def print_table(inst, runs):
    print(f"\n{'variant':>14s}  {'job_evac':>8s}  {'tok_evac':>8s}  {'kv_evac':>8s}  "
          f"{'min_u':>7s}  {'p10_u':>7s}  {'starved':>7s}  {'alpha*':>7s}  {'phi*':>7s}")
    for v in VARIANTS:
        s1, s2 = runs[v.key]
        e = evac_summary(inst, s1.z)
        a = "-" if s1.alpha_star is None else f"{s1.alpha_star:.4f}"
        print(f"{v.key:>14s}  {e['evacuated_fraction_total']:8.4f}  "
              f"{e['token_weighted_evacuation']:8.4f}  {e['kv_weighted_evacuation']:8.4f}  "
              f"{e['min_class_evacuated_fraction']:7.4f}  {e['p10_class_evacuated_fraction']:7.4f}  "
              f"{e['num_starved_classes']:7d}  {a:>7s}  {s2.phi_star:7.4f}")


def write_artifacts(inst, runs):
    OUT.mkdir(exist_ok=True)
    allm = {v.key: metrics(inst, *runs[v.key]) for v in VARIANTS}
    (OUT / "objective_metrics.json").write_text(json.dumps(allm, indent=2, default=str))

    rows = []
    for v in VARIANTS:
        m = allm[v.key]
        row = {"variant": v.key, "objective": v.objective, "weights": v.weights,
               "Z_star": m["Z_star"], "alpha_star": m["alpha_star"], "U_star": m["U_star"],
               **{f"evac_{k}": val for k, val in m["evacuation_stage1"].items()},
               "phi_star": m["pressure"]["phi_star"]}
        for b, bv in m["by_bucket"].items():
            row[f"bucket_{b}"] = bv["evacuated_fraction"]
        for mm, mv in m["by_model"].items():
            row[f"model_{mm}"] = mv["evacuated_fraction"]
        rows.append(row)
    keys = list({k for r in rows for k in r})
    with (OUT / "objective_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT/'objective_metrics.json'} and .csv")
    return allm


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    print(f"wrote {OUT/name}.pdf")


def plot_cdf(inst, runs, D):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in VARIANTS:
        u = evac_fraction(inst, runs[v.key][0].z)
        order = np.argsort(u)
        y = np.cumsum(inst.n[order]) / inst.n.sum()  # population-weighted
        ax.step(np.concatenate([[0], u[order]]), np.concatenate([[0], y]),
                where="post", color=v.color, lw=2, label=v.label)
    ax.set_xlabel("class evacuated fraction  $u_q$")
    ax.set_ylabel("fraction of jobs in classes with $u \\leq u_q$")
    ax.set_title(f"Class evacuation CDF (population-weighted), $D={D}$s")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_cdf")


def plot_token_cdf(inst, runs, D):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = np.argsort(inst.T)
    T = inst.T[order]
    for v in VARIANTS:
        evac_tok = ((inst.n - runs[v.key][0].z) * inst.T)[order]
        total = evac_tok.sum()
        ax.step(T, np.cumsum(evac_tok) / total, where="post", color=v.color, lw=2,
                label=f"{v.label}  ({total/1e9:.2f}B tok)")
    ax.set_xscale("log")
    ax.set_xlabel("token length  $T_q$")
    ax.set_ylabel("fraction of evacuated tokens in classes with $T \\leq T_q$")
    ax.set_title(f"Evacuated-token CDF over context length, $D={D}$s")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_token_cdf")


def _grouped(ax, labels, series, title, ylabel):
    x = np.arange(len(labels)); n = len(VARIANTS); w = 0.8 / n
    for i, v in enumerate(VARIANTS):
        ax.bar(x + (i - (n - 1) / 2) * w, series[v.key], w, color=v.color, label=v.label)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False)


def plot_buckets(inst, runs, D):
    series = {v.key: [by_bucket(inst, runs[v.key][0].z).get(b, {}).get("evacuated_fraction", 0.0)
                      for b in BUCKET_LABELS] for v in VARIANTS}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped(ax, BUCKET_LABELS, series, f"Evacuated fraction by token bucket, $D={D}$s",
             "evacuated fraction")
    _save(fig, "obj_buckets")


def plot_evac_measures(inst, runs, D):
    """Job count understates state left behind: compare job / token / KV evacuation."""
    measures = [("evacuated_fraction_total", "jobs"),
                ("token_weighted_evacuation", "tokens"),
                ("kv_weighted_evacuation", "KV bytes")]
    summ = {v.key: evac_summary(inst, runs[v.key][0].z) for v in VARIANTS}
    x = np.arange(len(VARIANTS)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (key, lab) in enumerate(measures):
        ax.bar(x + (i - 1) * w, [summ[v.key][key] for v in VARIANTS], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([v.label for v in VARIANTS], rotation=15, ha="right")
    ax.set_ylabel("evacuated fraction"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Evacuation weighted by jobs vs tokens vs KV bytes, $D={D}$s")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False, title="weighted by")
    _save(fig, "obj_evac_measures")


def plot_model_token_heatmap(inst, runs, D):
    """Per-(model, token-bucket) evacuated fraction -- finer than model-only bars."""
    M, B = inst.M_names, BUCKET_LABELS
    n = len(VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.8))
    for idx, (ax, v) in enumerate(zip(np.atleast_1d(axes), VARIANTS)):
        g = model_token_grid(inst, runs[v.key][0].z)
        im = ax.imshow(g, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(B))); ax.set_xticklabels(B, rotation=45, ha="right", fontsize=7)
        if idx == 0:
            ax.set_yticks(range(len(M))); ax.set_yticklabels(M, fontsize=7)
        else:
            ax.set_yticks([])
        ax.set_title(v.label, fontsize=9)
        for i in range(len(M)):
            for j in range(len(B)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.01,
                 label="evacuated fraction")
    fig.suptitle(f"Evacuated fraction by model x token length, $D={D}$s")
    fig.savefig(OUT / "obj_model_token_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "obj_model_token_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT/'obj_model_token_heatmap'}.pdf")


def plot_tradeoff(inst, runs, D):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for v in VARIANTS:
        e = evac_summary(inst, runs[v.key][0].z)
        ax.scatter(e["evacuated_fraction_total"], e["min_class_evacuated_fraction"],
                   s=120, color=v.color, zorder=3)
        ax.annotate(v.label, (e["evacuated_fraction_total"], e["min_class_evacuated_fraction"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.set_xlabel("total evacuated fraction (jobs)")
    ax.set_ylabel("worst-class evacuated fraction  $\\min_q u_q$")
    ax.set_title(f"Throughput vs worst-class protection, $D={D}$s")
    ax.grid(True, alpha=0.3)
    _save(fig, "obj_tradeoff")


def plot_pressure(inst, runs, D):
    n = len(VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.8))
    for ax, v in zip(axes, VARIANTS):
        s2 = runs[v.key][1]
        top = sorted(s2.pressures.items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
        ax.barh(range(len(top)), [val for _, val in top], color=v.color)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([k for k, _ in top], fontsize=7)
        ax.set_xlabel("normalized pressure")
        ax.set_title(f"{v.label}  ($\\phi^*={s2.phi_star:.3f}$)", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"Top-10 Stage 2 pressure indices, $D={D}$s")
    _save(fig, "obj_pressure")


def plot_action_mix(inst, runs, D):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(VARIANTS))
    R = np.array([runs[v.key][0].x_R.sum() for v in VARIANTS])
    S = np.array([runs[v.key][0].x_S.sum() for v in VARIANTS])
    ax.bar(x, R, 0.55, color="#3a7ca5", label="replay")
    ax.bar(x, S, 0.55, bottom=R, color="#edaaa0", label="state transfer")
    ax.set_xticks(x); ax.set_xticklabels([v.label for v in VARIANTS], rotation=15, ha="right")
    ax.set_ylabel("evacuated jobs")
    ax.set_title(f"Replay/state mix by objective, $D={D}$s")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_action_mix")


def plot_sweep():
    keys = ("evac", "tok", "kv", "minu", "phi")
    data = {v.key: {k: [] for k in keys} for v in VARIANTS}
    for D in SWEEP_D:
        inst, runs = solve_all(D)
        for v in VARIANTS:
            s1, s2 = runs[v.key]
            e = evac_summary(inst, s1.z)
            data[v.key]["evac"].append(e["evacuated_fraction_total"])
            data[v.key]["tok"].append(e["token_weighted_evacuation"])
            data[v.key]["kv"].append(e["kv_weighted_evacuation"])
            data[v.key]["minu"].append(e["min_class_evacuated_fraction"])
            data[v.key]["phi"].append(s2.phi_star)
    panels = [("evac", "job evacuated fraction"), ("tok", "token evacuated fraction"),
              ("kv", "KV evacuated fraction"), ("minu", "min class $u_q$"),
              ("phi", "peak pressure $\\phi^*$")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, ylabel) in zip(axes.ravel(), panels):
        for v in VARIANTS:
            ax.plot(SWEEP_D, data[v.key][key], "o-", color=v.color, label=v.label)
        ax.set_xscale("log"); ax.set_xlabel("deadline $D$ (s)"); ax.set_ylabel(ylabel)
        ax.set_xticks(SWEEP_D); ax.set_xticklabels(SWEEP_D)
        ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=7)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Deadline sweep")
    _save(fig, "obj_sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, default=120.0,
                    help="fixed deadline for the comparison plots (D>=300 is degenerate)")
    ap.add_argument("--sweep", action="store_true", help="also render the deadline sweep")
    args = ap.parse_args()

    inst, runs = solve_all(args.D)
    print_table(inst, runs)
    write_artifacts(inst, runs)
    plot_cdf(inst, runs, args.D)
    plot_token_cdf(inst, runs, args.D)
    plot_buckets(inst, runs, args.D)
    plot_evac_measures(inst, runs, args.D)
    plot_model_token_heatmap(inst, runs, args.D)
    plot_tradeoff(inst, runs, args.D)
    plot_pressure(inst, runs, args.D)
    plot_action_mix(inst, runs, args.D)
    if args.sweep:
        plot_sweep()


if __name__ == "__main__":
    main()
