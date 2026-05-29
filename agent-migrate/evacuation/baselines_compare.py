"""Baselines vs optimizer across the deadline sweep.

The optimizer is represented by its matched comparator per metric (throughput ->
token-weighted evac, max-min -> min-class, pipeline -> phi); every baseline is
evaluated cap-hard (evacuation, fairness) and cap-soft (phi). The cap-hard greedy
fills smallest-demand-first (the rational "evacuate easy jobs first" order).
Random is averaged over N_SEEDS; deterministic baselines run once (std ~ 0).

Job-count evacuated fraction is reported in the table but is order-sensitive
(a greedy can pump its count by cherry-picking small jobs), so the plots lead
with the order-robust token-weighted evac, min-class, and phi.

Usage:
    cd evacuation && uv run python baselines_compare.py

Writes `outputs/baselines.{json,csv}` and prints the D=120 s snapshot table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from baselines import BASELINES, allocate, pressures
from instance import build_instance
from objective_metrics import evac_summary
from plot_evacuation_vs_deadline import D_SWEEP_S
from stage1 import solve_stage1
from stage2 import solve_stage2

N_BINS = 5
N_SEEDS = 50
SNAPSHOT_D = 120
KEYS = ("evac_frac", "token_wtd", "kv_wtd", "min_class", "phi", "num_starved")


def _one(inst, name, seed):
    hard = allocate(inst, name, hard=True, seed=seed)
    soft = allocate(inst, name, hard=False, seed=seed)
    ev = evac_summary(inst, hard.z)
    return {
        "evac_frac": ev["evacuated_fraction_total"],
        "token_wtd": ev["token_weighted_evacuation"],
        "kv_wtd": ev["kv_weighted_evacuation"],
        "min_class": ev["min_class_evacuated_fraction"],
        "num_starved": ev["num_starved_classes"],
        "phi": pressures(inst, soft.x_R, soft.x_S).phi_star,
    }


def _agg(inst, name):
    runs = [_one(inst, name, s) for s in (range(N_SEEDS) if name == "random" else (0,))]
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in runs], float)
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def run():
    Ds = [float(D) for D in D_SWEEP_S]
    opt = {k: [] for k in ("throughput_evac", "throughput_tokenwtd", "throughput_kvwtd",
                           "maxmin_minclass", "pipeline_phi")}
    bl = {n: {} for n in BASELINES}
    for D in Ds:
        inst = build_instance(D=D, n_bins=N_BINS)
        thr = solve_stage1(inst, "throughput")
        ev = evac_summary(inst, thr.z)
        opt["throughput_evac"].append(ev["evacuated_fraction_total"])
        opt["throughput_tokenwtd"].append(ev["token_weighted_evacuation"])
        opt["throughput_kvwtd"].append(ev["kv_weighted_evacuation"])
        opt["maxmin_minclass"].append(
            evac_summary(inst, solve_stage1(inst, "max_min").z)["min_class_evacuated_fraction"])
        opt["pipeline_phi"].append(solve_stage2(inst, thr).phi_star)
        for n in BASELINES:
            for k, v in _agg(inst, n).items():
                bl[n].setdefault(k, []).append(v)
        print(f"D={D:6.0f}s  thr_tok={opt['throughput_tokenwtd'][-1]:.3f}  "
              f"random_tok={bl['random']['token_wtd_mean'][-1]:.3f}  "
              f"random_phi={bl['random']['phi_mean'][-1]:.2f}")
    return {"D": Ds, "optimizer": opt, "baselines": bl}


def write_csv(path, data):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["D", "method", "metric", "mean", "std"])
        for i, D in enumerate(data["D"]):
            for metric, key in (("evac_frac", "throughput_evac"),
                                ("token_wtd", "throughput_tokenwtd"),
                                ("kv_wtd", "throughput_kvwtd"),
                                ("min_class", "maxmin_minclass"),
                                ("phi", "pipeline_phi")):
                w.writerow([D, "optimizer", metric, data["optimizer"][key][i], 0.0])
            for n, m in data["baselines"].items():
                for k in KEYS:
                    w.writerow([D, n, k, m[f"{k}_mean"][i], m[f"{k}_std"][i]])


def _pm(mean, std):
    return f"{mean:.3f}±{std:.3f}" if std > 1e-9 else f"{mean:.3f}"


def print_snapshot(data):
    i = data["D"].index(float(SNAPSHOT_D))
    o = data["optimizer"]
    head = f"{'method':<14}{'evac*':>14}{'tok':>9}{'kv':>9}{'min_cls':>14}{'starved':>9}{'phi':>14}"
    print(f"\nSnapshot at D={SNAPSHOT_D}s  (*evac is job-count, order-sensitive)\n{head}\n{'-'*len(head)}")
    print(f"{'opt:throughput':<14}{o['throughput_evac'][i]:>14.3f}"
          f"{o['throughput_tokenwtd'][i]:>9.3f}{o['throughput_kvwtd'][i]:>9.3f}"
          f"{'-':>14}{'-':>9}{'-':>14}")
    print(f"{'opt:max_min':<14}{'-':>14}{'-':>9}{'-':>9}{o['maxmin_minclass'][i]:>14.3f}{'-':>9}{'-':>14}")
    print(f"{'opt:pipeline':<14}{'-':>14}{'-':>9}{'-':>9}{'-':>14}{'-':>9}{o['pipeline_phi'][i]:>14.3f}")
    for n, m in data["baselines"].items():
        print(f"{n:<14}{_pm(m['evac_frac_mean'][i], m['evac_frac_std'][i]):>14}"
              f"{m['token_wtd_mean'][i]:>9.3f}{m['kv_wtd_mean'][i]:>9.3f}"
              f"{_pm(m['min_class_mean'][i], m['min_class_std'][i]):>14}"
              f"{m['num_starved_mean'][i]:>9.1f}{_pm(m['phi_mean'][i], m['phi_std'][i]):>14}")


def main():
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    data = run()
    with (out / "baselines.json").open("w") as fh:
        json.dump(data, fh, indent=2)
    write_csv(out / "baselines.csv", data)
    print_snapshot(data)
    print(f"\nwrote {out / 'baselines.json'} and {out / 'baselines.csv'}")


if __name__ == "__main__":
    main()
