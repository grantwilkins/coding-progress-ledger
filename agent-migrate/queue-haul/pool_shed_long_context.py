"""Full-fleet sensitivity to reweighting existing long-context coding cohorts."""

from dataclasses import replace
import hashlib
from itertools import product
from pathlib import Path
import time

import numpy as np

import pool_shed_campaign as c

OUT = c.ROOT / "outputs/a100-full-fleet-long-context-wan-sweep"
CASES = (("original", 8, 0), ("long8", 8, 24000), ("long16", 16, 24000))
DEADLINES = (10, 20, 25, 30, 35, 40, 45, 60)
WANS = (40, 100, 400, 1000)


def main():
    started = time.perf_counter()
    plan = c.load_plan(c.OUT)
    report = {"parent_identity": plan["identity"], "sources": {**plan["sources"],
              Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "cases": CASES, "deadlines_s": DEADLINES, "wan_gbps": WANS,
              "source_gpus": c.GPUS, "destination_gpus_each": c.GPUS, "resident_load": .5,
              "status": "experimental_regional_timing_gate_failed", "assumptions": [
                  "Case density applies to source and residents: original/long8 use eight sessions/GPU; long16 uses sixteen; no invented migration backlog"
                  if a.startswith("eight resident sessions/GPU") else a for a in plan["assumptions"]],
              "scope": "Reweight existing sampled coding cohorts; original retains eight sessions/GPU; long cases use stated density at source and residents, preserving reference load via cadence; long-context batch serialization remains a transfer assumption; sampled ranges are sensitivity, not fidelity bounds",
              "replay_bound_scope": "Optimistic serial GPU-time reuse within the eligible library, not an executable schedule"}
    report["identity"] = c.digest(report)
    rows, fleets = [], {}
    for (case, density, minimum), snapshot in product(CASES, range(plan["config"]["snapshots"])):
        base = c.sample_fleet("coding", snapshot)
        fleet = base
        if minimum:
            eligible = np.flatnonzero(base.context >= minimum)
            assert len(eligible), "snapshot has no long-context cohorts"
            ids = np.resize(eligible, len(base.count))
            counts = np.full(len(ids), c.GPUS * density // len(ids))
            counts[-1] += c.GPUS * density - counts.sum()
            fields = {key: getattr(base, key)[ids] for key in ("context", "prompt", "output", "t1", "kv", "log", "demand")}
            fields["demand"] *= c.SOURCE_LOAD * c.GPUS / (counts @ fields["demand"])
            fleet = replace(base, **fields, count=counts)
        assert fleet.baseline_kv < fleet.kv_capacity
        assert fleet.count.sum() == c.GPUS * density and np.isclose(fleet.count @ fleet.demand, c.SOURCE_LOAD * c.GPUS)
        fleets[f"{case}/{snapshot}"] = {"counts": fleet.count.tolist(), "context_tokens": fleet.context.tolist(),
            "reference_demand": fleet.demand.tolist(), "kv_bytes": float(fleet.count @ fleet.kv),
            "resident_kv_fraction": fleet.baseline_kv / fleet.kv_capacity}
        replay, kv = c.library(fleet)
        for draw, wan, deadline in product(range(plan["config"]["draws"] + 1), WANS, DEADLINES):
            samples = c.network_samples()
            endpoint = samples[plan["network_indices"][draw]].copy() if draw else np.r_[np.median(samples[:, :2], axis=0), 0.]
            endpoint[2] = endpoint[:2].sum()
            budgets, timing = c.bandwidth(endpoint, fleet.nodes, wan), plan["calibration"]["timing"][draw]
            r, k = c.include_isolated(replay, kv, c.isolated_methods(fleet, .5, endpoint, budgets, timing))
            table = c.schedule_table(fleet, r, k, .5, deadline, endpoint, budgets, timing)
            results = c.compare(table)
            table.matrix[len(fleet.count):len(fleet.count) + 2] *= table.duration / deadline
            bound = c.execute(table, c.select(table, "replay_only"))["shed_fraction"]
            assert bound + 1e-8 >= results["replay_only"]["shed_fraction"]
            for policy, value in results.items():
                rows.append(dict(case=case, snapshot=snapshot, draw=draw, wan_gbps=wan, deadline_s=deadline,
                    policy=policy, shed_fraction=value["shed_fraction"], replay_gpu_time_bound=bound,
                    kv_fraction=sum(value["action_fractions"][1::2]),
                    qh_gap=results["queue_haul"]["shed_fraction"] - value["shed_fraction"]))
    OUT.mkdir(parents=True, exist_ok=True)
    c.write_csv(OUT / "sensitivity.csv", rows)
    c.write_json(OUT / "experiment.json", {**report, "fleets": fleets, "evaluations": len(rows) * 6 // 5,
                                          "wall_s": time.perf_counter() - started})
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    model = plot_style.MODEL_NAMES["openai/gpt-oss-20b"] + " / " + plot_style.AGENTIC_HARDWARE_NAMES["a100"]
    for wan in WANS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        for ax, (case, density, minimum) in zip(axes, CASES):
            for policy in c.POLICIES:
                values = [[100 * r["shed_fraction"] for r in rows if
                           (r["case"], r["policy"], r["wan_gbps"], r["deadline_s"]) == (case, policy, wan, d)] for d in DEADLINES]
                low, median, high = np.quantile(values, [0, .5, 1], axis=1)
                ax.plot(DEADLINES, median, **plot_style.policy_style(policy))
                ax.fill_between(DEADLINES, low, high, color=plot_style.POLICY_COLORS[policy], alpha=.12)
            ax.set(title=f"{'24K–32K contexts' if minimum else 'Original coding contexts'}; {density} sessions/GPU",
                   xlabel="Deadline (s)", ylim=(0, 105))
        axes[0].set_ylabel("Removed source workload (%)")
        fig.suptitle(f"Full 20 MW installed {model} fleet; two equal destinations at 50% load\nAssumed shared WAN budget {wan:g} Gbit/s; node endpoint ceilings also apply")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=9)
        fig.text(.5, .09, "Experimental: regional timing validation fails; long-context batches use serialization fallback. Bands span four snapshots × nine calibration/network cases.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .9))
        for extension in ("png", "pdf"):
            fig.savefig(OUT / f"shed-{wan}.{extension}", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
