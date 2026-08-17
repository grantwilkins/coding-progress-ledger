"""Plot policy attainment pooled over workload-constraint draws."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style
import workload_adaptation_campaign as campaign
from planner import plan


POLICIES = {
    "queue_haul": "lp_highs", "isolated_fastest": "isolated_fastest",
    "queue_haul_power_blind": "lp_power_blind",
}
plot_style.apply()


def attainment_rows(samples=1000, seed=campaign.DEFAULT_SEED, sessions=28,
                    target_fraction=2 / 3):
    if samples < 1 or sessions < 1 or not 0 < target_fraction <= 1:
        raise ValueError("invalid policy-attainment controls")
    profile = campaign.ModelProfile.load(campaign.PROFILE)
    templates, _ = campaign.load_templates(campaign.MANIFEST, profile)
    timing_rows = campaign.read_csv(campaign.TIMING)
    parent = json.loads(campaign.TIMING_PARENT.read_text())
    campaign.central_timing_fits()
    rng, rows = np.random.default_rng(seed), []
    for replicate in range(samples):
        draw = campaign.sample_draw(
            profile, templates, timing_rows, parent, rng, replicate, seed,
            sessions,
        )
        sampled_profile, pack, fits, power_index, timing_hash, _ = draw
        for case_id, label, constraints in campaign.factorial_cases():
            problem, architecture, routes, target = campaign.build_problem(
                sampled_profile, pack, constraints, target_fraction, fits,
            )
            for policy, solver in POLICIES.items():
                result = plan(
                    problem, sampled_profile, routes, solver, seed=replicate,
                    destination=architecture, admission_mode="normal",
                )
                shed = result.initial_source_power_w \
                    - result.expected_source_power_at_deadline_w
                rows.append({
                    "replicate": replicate, "case_id": case_id,
                    "bound_constraint": label, "policy": policy,
                    "power_bootstrap_index": power_index,
                    "timing_fit_sha256": timing_hash, "target_w": target,
                    "attainment_fraction": max(0, min(1, shed / target)),
                    "target_met": result.feasible
                    and result.power_shortfall_w <= campaign.POWER_TOLERANCE_W,
                })
    return rows


def attainment_curve(rows, policy):
    selected = [row for row in rows if row["policy"] == policy]
    cases = {(int(row["replicate"]), row["case_id"]) for row in selected}
    if not selected or len(selected) != len(cases):
        raise RuntimeError("attainment CDF requires one row per paired case")
    values = np.sort([100 * float(row["attainment_fraction"])
                      for row in selected])
    return values, np.arange(1, len(values) + 1) / len(values)


def write_plot(rows, path):
    fig, axis = plt.subplots(figsize=plot_style.WIDE_FIGSIZE)
    for policy in POLICIES:
        x, y = attainment_curve(rows, policy)
        axis.step(x, y, where="post", **plot_style.policy_style(policy))
    axis.set(xlim=(0, 101), ylim=(0, 1.02),
             xlabel="Power-target attainment by deadline (%)",
             ylabel="Cumulative distribution")
    axis.grid(alpha=.25)
    axis.legend(loc="upper left", frameon=False,
                fontsize=plot_style.LEGEND_FONT_SIZE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=campaign.DEFAULT_SEED)
    parser.add_argument("--sessions", type=int, default=28)
    parser.add_argument("--target", type=float, default=2 / 3)
    parser.add_argument("--out", type=Path, default=campaign.OUT / "policy_attainment")
    args = parser.parse_args()
    rows = attainment_rows(args.samples, args.seed, args.sessions, args.target)
    write_csv(rows, args.out.with_suffix(".csv"))
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
