"""Compare three Stage 1 evacuation objectives on the same convex resource model.

Solves throughput / max_min / prop_fair once at a fixed deadline (Stage 1 -> Stage 2),
prints a comparison table, writes outputs/objective_metrics.{json,csv}, and renders
Plots 1-6. With --sweep it also renders the deadline-sweep Plot 7.

Usage:
    cd evacuation && uv run python objectives_compare.py [--D 120] [--weights population|class] [--sweep]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instance import build_instance
from objective_metrics import BUCKET_LABELS, by_bucket, evac_fraction, evac_summary, metrics
from stage1 import solve_stage1
from stage2 import solve_stage2

OBJECTIVES = ("throughput", "max_min", "prop_fair")
COLORS = {"throughput": "#3a7ca5", "max_min": "#c44536", "prop_fair": "#4c9a52"}
N_BINS = 5
SWEEP_D = (30, 60, 120, 300, 600, 1200)
OUT = Path(__file__).resolve().parent / "outputs"


def solve_all(D, weights):
    inst = build_instance(D=D, n_bins=N_BINS)
    runs = {}
    for obj in OBJECTIVES:
        s1 = solve_stage1(inst, obj, weights)
        s2 = solve_stage2(inst, s1)
        runs[obj] = (s1, s2)
    return inst, runs


def print_table(inst, runs):
    print(f"\n{'objective':>11s}  {'evacuated':>10s}  {'evac_frac':>9s}  "
          f"{'min_u':>7s}  {'p10_u':>7s}  {'p50_u':>7s}  {'starved':>7s}  "
          f"{'alpha*':>7s}  {'phi*':>7s}")
    for obj in OBJECTIVES:
        s1, s2 = runs[obj]
        e = evac_summary(inst, s1.z)
        a = "-" if s1.alpha_star is None else f"{s1.alpha_star:.4f}"
        print(f"{obj:>11s}  {e['total_evacuated']:10.1f}  "
              f"{e['evacuated_fraction_total']:9.4f}  "
              f"{e['min_class_evacuated_fraction']:7.4f}  "
              f"{e['p10_class_evacuated_fraction']:7.4f}  "
              f"{e['p50_class_evacuated_fraction']:7.4f}  "
              f"{e['num_starved_classes']:7d}  {a:>7s}  {s2.phi_star:7.4f}")


def write_artifacts(inst, runs):
    OUT.mkdir(exist_ok=True)
    allm = {obj: metrics(inst, *runs[obj]) for obj in OBJECTIVES}
    (OUT / "objective_metrics.json").write_text(json.dumps(allm, indent=2, default=str))

    rows = []
    for obj in OBJECTIVES:
        m = allm[obj]
        row = {"objective": obj, "Z_star": m["Z_star"], "alpha_star": m["alpha_star"],
               "U_star": m["U_star"], **{f"evac_{k}": v for k, v in m["evacuation_stage1"].items()},
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
    for obj in OBJECTIVES:
        s1, _ = runs[obj]
        u = evac_fraction(inst, s1.z)
        order = np.argsort(u)
        x = u[order]
        y = np.cumsum(inst.n[order]) / inst.n.sum()  # population-weighted
        ax.step(np.concatenate([[0], x]), np.concatenate([[0], y]),
                where="post", color=COLORS[obj], lw=2, label=obj)
    ax.set_xlabel("class evacuated fraction  $u_q$")
    ax.set_ylabel("fraction of jobs in classes with $u \\leq u_q$")
    ax.set_title(f"Plot 1 - Class evacuation CDF (population-weighted), $D={D}$s")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_cdf")


def plot_token_cdf(inst, runs, D):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = np.argsort(inst.T)
    T = inst.T[order]
    for obj in OBJECTIVES:
        s1, _ = runs[obj]
        evac_tok = ((inst.n - s1.z) * inst.T)[order]
        total = evac_tok.sum()
        y = np.cumsum(evac_tok) / total
        ax.step(T, y, where="post", color=COLORS[obj], lw=2,
                label=f"{obj}  ({total/1e9:.2f}B tok)")
    ax.set_xscale("log")
    ax.set_xlabel("token length  $T_q$")
    ax.set_ylabel("fraction of evacuated tokens in classes with $T \\leq T_q$")
    ax.set_title(f"Evacuated-token CDF over context length, $D={D}$s")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_token_cdf")


def _grouped(ax, labels, series, title, ylabel):
    x = np.arange(len(labels)); w = 0.26
    for i, obj in enumerate(OBJECTIVES):
        ax.bar(x + (i - 1) * w, series[obj], w, color=COLORS[obj], label=obj)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False)


def plot_buckets(inst, runs, D):
    series = {obj: [by_bucket(inst, runs[obj][0].z).get(b, {}).get("evacuated_fraction", 0.0)
                    for b in BUCKET_LABELS] for obj in OBJECTIVES}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped(ax, BUCKET_LABELS, series, f"Plot 2 - Evacuated fraction by token bucket, $D={D}$s",
             "evacuated fraction")
    _save(fig, "obj_buckets")


def plot_models(inst, runs, D):
    M = inst.M_names
    series = {obj: [(inst.n[inst.model_idx == m] - runs[obj][0].z[inst.model_idx == m]).sum()
                    / inst.n[inst.model_idx == m].sum() for m in range(len(M))]
              for obj in OBJECTIVES}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped(ax, M, series, f"Plot 3 - Evacuated fraction by model, $D={D}$s", "evacuated fraction")
    _save(fig, "obj_models")


def plot_tradeoff(inst, runs, D):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for obj in OBJECTIVES:
        s1, _ = runs[obj]
        ev = float((inst.n - s1.z).sum())
        minu = float(evac_fraction(inst, s1.z).min())
        ax.scatter(ev, minu, s=120, color=COLORS[obj], zorder=3)
        ax.annotate(obj, (ev, minu), textcoords="offset points", xytext=(8, 6))
    ax.set_xlabel("total evacuated jobs")
    ax.set_ylabel("worst-class evacuated fraction  $\\min_q u_q$")
    ax.set_title(f"Plot 4 - Throughput vs worst-class protection, $D={D}$s")
    ax.grid(True, alpha=0.3)
    _save(fig, "obj_tradeoff")


def plot_pressure(inst, runs, D):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=False)
    for ax, obj in zip(axes, OBJECTIVES):
        _, s2 = runs[obj]
        top = sorted(s2.pressures.items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
        names = [k for k, _ in top]
        ax.barh(range(len(top)), [v for _, v in top], color=COLORS[obj])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("normalized pressure"); ax.set_title(f"{obj}  ($\\phi^*={s2.phi_star:.3f}$)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"Plot 5 - Top-10 Stage 2 pressure indices, $D={D}$s")
    _save(fig, "obj_pressure")


def plot_action_mix(inst, runs, D):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(OBJECTIVES))
    R = np.array([runs[o][0].x_R.sum() for o in OBJECTIVES])
    S = np.array([runs[o][0].x_S.sum() for o in OBJECTIVES])
    ax.bar(x, R, 0.55, color="#3a7ca5", label="replay")
    ax.bar(x, S, 0.55, bottom=R, color="#edaaa0", label="state transfer")
    ax.set_xticks(x); ax.set_xticklabels(OBJECTIVES)
    ax.set_ylabel("evacuated jobs")
    ax.set_title(f"Plot 6 - Replay/state mix by objective, $D={D}$s")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(frameon=False)
    _save(fig, "obj_action_mix")


def plot_sweep(weights):
    data = {o: {"evac": [], "minu": [], "p10": [], "phi": []} for o in OBJECTIVES}
    for D in SWEEP_D:
        inst, runs = solve_all(D, weights)
        for o in OBJECTIVES:
            s1, s2 = runs[o]
            e = evac_summary(inst, s1.z)
            data[o]["evac"].append(e["evacuated_fraction_total"])
            data[o]["minu"].append(e["min_class_evacuated_fraction"])
            data[o]["p10"].append(e["p10_class_evacuated_fraction"])
            data[o]["phi"].append(s2.phi_star)
    panels = [("evac", "total evacuated fraction"), ("minu", "min class $u_q$"),
              ("p10", "p10 class $u_q$"), ("phi", "peak pressure $\\phi^*$")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, ylabel) in zip(axes.ravel(), panels):
        for o in OBJECTIVES:
            ax.plot(SWEEP_D, data[o][key], "o-", color=COLORS[o], label=o)
        ax.set_xscale("log"); ax.set_xlabel("deadline $D$ (s)"); ax.set_ylabel(ylabel)
        ax.set_xticks(SWEEP_D); ax.set_xticklabels(SWEEP_D)
        ax.grid(True, alpha=0.3); ax.legend(frameon=False)
    fig.suptitle("Plot 7 - Deadline sweep")
    _save(fig, "obj_sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, default=120.0,
                    help="fixed deadline for Plots 1-6 (D=300 is degenerate: all evacuate)")
    ap.add_argument("--weights", choices=("population", "class"), default="population")
    ap.add_argument("--sweep", action="store_true", help="also render Plot 7 (deadline sweep)")
    args = ap.parse_args()

    inst, runs = solve_all(args.D, args.weights)
    print_table(inst, runs)
    write_artifacts(inst, runs)
    plot_cdf(inst, runs, args.D)
    plot_token_cdf(inst, runs, args.D)
    plot_buckets(inst, runs, args.D)
    plot_models(inst, runs, args.D)
    plot_tradeoff(inst, runs, args.D)
    plot_pressure(inst, runs, args.D)
    plot_action_mix(inst, runs, args.D)
    if args.sweep:
        plot_sweep(args.weights)


if __name__ == "__main__":
    main()
