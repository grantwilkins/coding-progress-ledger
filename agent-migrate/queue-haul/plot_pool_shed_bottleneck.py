"""Reproduce a serving-pool bottleneck example with the existing measured model."""

import hashlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style
import pool_shed_campaign as c

OUT = c.ROOT / "outputs/a100-pool-shed-bottleneck"
DEADLINES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6)
BASE = (64, .5, 40, 3)


def main():
    started = time.perf_counter()
    plan = c.load_plan(c.OUT)
    cases = list(dict.fromkeys([(64, .5, 40, d) for d in DEADLINES] +
                              [(n, u, w, 3) for n, u, w in ((32, .5, 40), (128, .5, 40),
                               (64, .25, 40), (64, .4, 40), (64, .6, 40), (64, .75, 40),
                               (64, .5, 20), (64, .5, 30), (64, .5, 50), (64, .5, 80),
                               (c.GPUS, .5, 40), (c.GPUS, .5, 333360))]))
    sources = {**plan["sources"], Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = {"parent_identity": plan["identity"], "sources": sources, "cases": cases,
              "status": "experimental_regional_timing_gate_failed",
              "scope": "64-GPU pool example, not a simultaneous 20 MW shed; full-fleet controls retained",
              "interval_scope": "paired calibration/network sensitivity, not fidelity confidence bounds",
              "assumptions": plan["assumptions"]}
    report["identity"] = c.digest(report)
    rows, central = [], None
    expanded_difference = 0.
    for n, load, wan, deadline in cases:
        local = {**plan, "identity": report["identity"],
                 "config": {**plan["config"], "gpus": n, "installed_gpu_w": n * 300}}
        for draw in range(plan["config"]["draws"] + 1):
            cell = (("measured_pack", 0), load, draw, wan, deadline)
            result = c.run_cell(local, cell)
            if (n, load, wan, deadline) == BASE and draw == 0:
                central = result
                expanded = c.run_cell(local, cell, expanded=True)
                expanded_difference = max(abs(expanded["results"][p]["shed_fraction"] - result["results"][p]["shed_fraction"]) for p in c.POLICIES)
                assert expanded_difference < .01, "example is sensitive to library expansion"
                replay = result["results"]["replay_only"]["resource_utilization"]
                assert np.allclose(replay[8:10], 1) and replay[-1] < .02, "claimed replay bottleneck changed"
            for policy, value in result["results"].items():
                rows.append(dict(gpus=n, load=load, wan_gbps=wan, deadline_s=deadline, draw=draw, policy=policy,
                                 shed_fraction=value["shed_fraction"], action_fractions=value["action_fractions"],
                                 resource_utilization=value["resource_utilization"],
                                 qh_gap=result["results"]["queue_haul"]["shed_fraction"] - value["shed_fraction"]))
    assert central is not None
    OUT.mkdir(parents=True, exist_ok=True)
    fleet = c.sample_fleet("measured_pack", gpus=64, gpus_per_node=8)
    endpoint, budgets = np.array(central["endpoint_gbps"]) / 8e-9, np.array(central["budgets_gbps"]) / 8e-9
    timing = plan["calibration"]["timing"][0]
    r, k = c.include_isolated(*c.library(fleet), c.isolated_methods(fleet, .5, endpoint, budgets, timing))
    table = c.schedule_table(fleet, r, k, .5, 3, endpoint, budgets, timing)
    # Optimistic serial reuse bound within this library; not an executable GPU schedule.
    table.matrix[8:10] *= table.duration / table.deadline
    relaxed_replay = c.execute(table, c.select(table, "replay_only"))["shed_fraction"]
    assert central["results"]["queue_haul"]["shed_fraction"] > relaxed_replay + .05
    table.matrix[8:10] = 0
    unlimited = {p: c.execute(table, c.select(table, p))["shed_fraction"] for p in ("queue_haul", "replay_only")}
    assert np.allclose(list(unlimited.values()), 1), "migration capacity is not the only blocking resource"
    c.write_csv(OUT / "sensitivity.csv", rows)
    c.write_json(OUT / "example.json", {**report, "central": central,
                 "expanded_library_max_difference": expanded_difference, "evaluations": len(rows) + len(c.POLICIES) + 3,
                 "replay_gpu_time_relaxation_fraction": relaxed_replay, "without_migration_limit": unlimited,
                 "simulation_s": time.perf_counter() - started})
    plot(rows, central)


def plot(rows, central):
    plot_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for policy in c.POLICIES:
        values = [[100 * r["shed_fraction"] for r in rows if
                   (r["gpus"], r["load"], r["wan_gbps"], r["deadline_s"], r["policy"]) == (64, .5, 40, d, policy)
                   and r["draw"] > 0] for d in DEADLINES]
        low, median, high = np.quantile(values, [.05, .5, .95], axis=1)
        axes[0].plot(DEADLINES, median, **plot_style.policy_style(policy))
        axes[0].fill_between(DEADLINES, low, high, color=plot_style.POLICY_COLORS[policy], alpha=.15)
    axes[0].set(title="Deadline sensitivity", xlabel="Deadline (s)", ylabel="Removed source workload (%)", ylim=(0, 105))
    values = np.array([central["results"][p]["action_fractions"] for p in c.POLICIES]) * 100
    replay, kv = values[:, [0, 2]].sum(1), values[:, [1, 3]].sum(1)
    for action, widths, left in (("replay", replay, np.zeros(5)), ("kv_transfer", kv, replay)):
        axes[1].barh(np.arange(5), widths, left=left, color=plot_style.ACTION_COLORS[action], label=plot_style.ACTION_NAMES[action])
    for i, total in enumerate(replay + kv):
        axes[1].text(total + 1, i, f"{total:.1f}%", va="center", fontsize=9)
    axes[1].set(yticks=np.arange(5), yticklabels=[plot_style.POLICY_NAMES[p] for p in c.POLICIES],
                title="3-second action mix", xlabel="Removed source workload (%)", xlim=(0, 108))
    axes[1].tick_params(axis="y", labelsize=9)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=9, loc="lower right")
    for i, policy in enumerate(("replay_only", "greedy", "queue_haul")):
        usage = np.array(central["results"][policy]["resource_utilization"])[[8, 9, -1]] * 100
        axes[2].bar(np.arange(3) + (i - 1) * .25, usage, width=.25,
                    color=plot_style.POLICY_COLORS[policy], label=plot_style.POLICY_NAMES[policy])
    axes[2].set(xticks=np.arange(3), xticklabels=["Replay slots\nEast", "Replay slots\nGermany", "Shared\nWAN"],
                title="3-second resource use", ylabel="Reserved capacity (%)", ylim=(0, 105))
    axes[2].tick_params(axis="x", labelsize=10)
    fig.suptitle("GPT-OSS / A100: 64-GPU source pool, two 64-GPU destinations, 50% resident load\n8 GPUs/node; 512 source sessions; assumed 40 Gbit/s shared WAN")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .82), ncol=5, fontsize=10)
    fig.text(.5, .015, "Experimental: regional timing validation fails. Replay slots are a batch-library constraint, not measured FLOP saturation. Bands: calibration/network p05–p95.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .87))
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"bottleneck.{extension}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
