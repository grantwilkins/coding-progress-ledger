"""Supplemental assumed replay-prefix/private-KV sensitivity; no new measurements."""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import highspy
import numpy as np

import pool_shed_campaign as q
from pool_shed_calibration import replay_seconds
from pool_shed_execution import kv_transfer_bytes

OUT = q.ROOT / "outputs/a100-cache-sensitivity"
VARIANTS = {"baseline": (0., 1., 1.), "reported800MB": (0., 1., .8e9 / (32768 * 49152)),
            "replay25": (.25, 1., .8e9 / (32768 * 49152)), "private50": (0., .5, .8e9 / (32768 * 49152)),
            "combined50": (.25, .5, .8e9 / (32768 * 49152)), "combined10": (.25, .1, .8e9 / (32768 * 49152))}
CASES = list(product((66666, 6666), VARIANTS, (30, 120, 300)))
SCOPE = ("Central coding snapshot 0, resident load 0.5, WAN 1000 Gbps; assumed cache fractions, not measurements. "
         "Nonbaseline wire scale uses user-reported 0.80 GB decimal per 32768 tokens as an effective-wire anchor. "
         "Private fractions are an additional sensitivity, not established properties of that anchor. "
         "Fixed shared prefixes persist after resets, capped by current context; new tokens beyond the fixed prefix remain private. "
         "Replay hit fractions assume additional work savings relative to measured timings; native cache hits in those timings are unknown. "
         "Reusable prefixes are assumed already present at the destination; their provisioning is outside the deadline. "
         "All variants receive the same union of candidate batches, including each variant's isolated-fastest projections. "
         "All methods and variants use the same fixed 1s geometric base clock, subdivided at resolution0.5, plus observed recovery events. "
         "Context, logs, standing demand, resident KV memory and source power are unchanged. "
         "Power is a linear allocation of modeled active-minus-awake-idle watts, not installed nameplate power.")


def sources(calibration):
    return {**q.provenance(calibration), Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def fleet_for(gpus, variant, calibration):
    hit, private, scale = VARIANTS[variant]
    fleet = q.sample_fleet("coding", 0, gpus, 8)
    cached = np.floor(fleet.context * hit / 256) * 256
    metadata = {**fleet.metadata, "planning_reference_s": 4., "replay_cached_tokens": cached.tolist(), "kv_wire_scale": scale,
                "kv_shared_tokens": (np.floor(fleet.context * (1 - private) / 256) * 256).tolist()}
    fleet = replace(fleet, metadata=metadata, t1=replay_seconds(fleet.context, calibration, cached))
    return replace(fleet, kv=kv_transfer_bytes(fleet, fleet.context, calibration))


def build(index, calibration):
    gpus, variant, deadline = CASES[index]
    fleet = fleet_for(gpus, variant, calibration)
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    budgets, timing = q.bandwidth(endpoint, fleet.nodes, 1000), calibration["timing"][0]
    fleets = [fleet_for(gpus, name, calibration) for name in VARIANTS]
    pairs = [q.library(f) for f in fleets]
    replay, kv = q.action_closure(np.vstack([p[0] for p in pairs]), np.vstack([p[1] for p in pairs]))
    pairs = [q.include_isolated(replay, kv, q.isolated_methods(f, .5, endpoint, budgets, timing)) for f in fleets]
    replay, kv = q.action_closure(np.vstack([p[0] for p in pairs]), np.vstack([p[1] for p in pairs]))
    return q.schedule_table(fleet, replay, kv, .5, deadline, endpoint, budgets, timing)


def check(value):
    gpus, _, deadline = CASES[value["index"]]
    if value["case"] != list(CASES[value["index"]]) or set(value["results"]) != set(q.POLICIES):
        raise ValueError("case or policy coverage mismatch")
    for policy, result in value["results"].items():
        numeric = [v for k, v in result.items() if k.endswith(("_requests", "_work_s"))] + [result[k] for k in ("batch_replica_seconds", "transferred_bytes", "peak_destination_load", "protected_serving_load", "action_fractions")]
        if not np.isfinite(np.concatenate([np.atleast_1d(v) for v in numeric])).all():
            raise ValueError("nonfinite queue accounting")
        shed, admitted = result["shed_fraction"], result["admitted_shed_fraction"]
        if not np.isfinite([shed, admitted, result["max_relative_residual"]]).all() or not -1e-8 <= shed <= admitted + 1e-8 <= 1 + 2e-8:
            raise ValueError("invalid handoff fractions")
        if result["max_relative_residual"] > 1e-8 or result["last_completion_s"] > deadline + 1e-8:
            raise ValueError("resource or deadline violation")
        if not np.allclose(np.asarray(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"], result["pending_resident_debt_work_s"], rtol=1e-8, atol=1e-6):
            raise ValueError("resident debt conservation failed")
        if np.max(result["peak_destination_load"] + result["protected_serving_load"]) > 1 + 1e-8 or np.max(result["batch_replica_seconds"]) > gpus * deadline * (1 + 1e-8):
            raise ValueError("destination compute capacity exceeded")
        if np.any(np.array(result["transferred_bytes"]) > np.array(value["budgets_bytes_per_s"]) * deadline * (1 + 1e-8)):
            raise ValueError("network capacity exceeded")
        for total, a, b in (("buffered_requests", "source_buffered_requests", "transferred_buffered_requests"), ("pending_buffered_requests", "source_buffered_requests", "pending_destination_buffered_requests"), ("buffered_requests", "pending_buffered_requests", "completed_buffered_requests"), ("pending_buffered_work_s", "pending_source_buffer_work_s", "pending_backlog_reference_work_s")):
            if not np.isclose(result[total], result[a] + result[b], rtol=1e-9, atol=1e-6):
                raise ValueError("buffer conservation failed")
        fractions = np.array(result["action_fractions"])
        if not np.isclose(fractions.sum(), shed, rtol=1e-9, atol=1e-8) or (policy == "replay_only" and np.any(fractions[1::2] > 1e-8)) or (policy == "kv_only" and np.any(fractions[::2] > 1e-8)):
            raise ValueError("action accounting or pure-policy mask violated")


def run(index, out):
    started, timestamp = time.monotonic(), time.time()
    calibration = q.calibration(0)
    pinned = sources(calibration)
    table = build(index, calibration)
    certificates = {policy: q.certify(table, q.select(table, policy)) for policy in q.POLICIES}
    if max(v["shed_fraction"] for v in certificates.values()) > certificates["queue_haul"]["shed_fraction"] + 1e-8:
        raise ValueError("initial static LP baseline containment failed")
    value = {"initial_static_certificates": certificates, "index": index, "case": list(CASES[index]), "scope": SCOPE, "sources": pinned,
             "started_epoch_s": timestamp, "python": platform.python_version(), "solver_version": highspy.Highs().version(),
             "metadata": table.fleet.metadata, "budgets_bytes_per_s": table.budgets.tolist(),
             "session_counts": table.fleet.count.tolist(), "context_tokens": table.fleet.context.tolist(), "initial_replay_s": table.fleet.t1.tolist(),
             "initial_kv_bytes": table.fleet.kv.tolist(), "columns": len(table.gains),
             "results": {policy: q.execute_feedback(table, table, policy, calibration["timing"][0], calibration) for policy in q.POLICIES}}
    value["metrics"] = {policy: {"handoff_fraction": r["shed_fraction"], "kv_source_fraction": sum(r["action_fractions"][1::2]),
        "power_mw": r["shed_fraction"] * table.fleet.gpus * table.fleet.metadata["source_power"]["delta_w"] / 1e6}
        for policy, r in value["results"].items()}
    value["wall_s"] = time.monotonic() - started
    check(value)
    if sources(calibration) != pinned:
        raise ValueError("scientific sources changed during execution")
    path = out / f"case-{index:02d}.json"
    if path.exists():
        raise FileExistsError(path)
    q.write_json(path, value)


def reduce(out):
    expected = {out / f"case-{i:02d}.json" for i in range(len(CASES))}
    if set(out.glob("case-*.json")) != expected:
        raise ValueError("reduction requires exactly 36 complete cases")
    pinned, rows, cases = sources(q.calibration(0)), [], []
    for path in sorted(expected):
        value = json.loads(path.read_text())
        if path.name != f"case-{value['index']:02d}.json" or value["sources"] != pinned:
            raise ValueError("case filename or source provenance mismatch")
        check(value)
        cases.append(value)
        gpus, variant, deadline = value["case"]
        for policy, result in value["results"].items():
            kv = sum(result["action_fractions"][1::2])
            rows.append({"gpus": gpus, "variant": variant, "deadline_s": deadline, "policy": policy,
                         "handoff_fraction": result["shed_fraction"], "admitted_fraction": result["admitted_shed_fraction"],
                         "power_mw": result["shed_fraction"] * gpus * value["metadata"]["source_power"]["delta_w"] / 1e6,
                         "kv_source_fraction": kv, "kv_handoff_share": kv / result["shed_fraction"] if result["shed_fraction"] else 0.})
    for gpus, deadline, hit in product((66666, 6666), (30, 120, 300), (0., .25)):
        controls = [v for v in cases if v["case"][0] == gpus and v["case"][2] == deadline and VARIANTS[v["case"][1]][0] == hit]
        for value in controls[1:]:
            for field in ("shed_fraction", "action_fractions", "transferred_bytes", "batch_replica_seconds"):
                if not np.allclose(value["results"]["replay_only"][field], controls[0]["results"]["replay_only"][field], rtol=1e-10, atol=1e-8):
                    raise ValueError("KV-only assumptions changed the replay-only control")
    q.write_csv(out / "summary.csv", rows)
    q.write_json(out / "summary.json", {"scope": SCOPE, "sources": pinned, "cases": 36, "evaluations": 180, "rows": rows,
        "case_wall_s_sum": sum(v["wall_s"] for v in cases), "observed_campaign_span_s": max(v["started_epoch_s"] + v["wall_s"] for v in cases) - min(v["started_epoch_s"] for v in cases),
        "runtime_scope": "Sum of per-case monotonic durations; campaign span uses epoch starts plus case durations and includes scheduling gaps."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    for metric, label, factor in (("handoff_fraction", "Source handed off (%)", 100), ("power_mw", "Shed power proxy (MW)", 1), ("kv_source_fraction", "Source handed off through KV (%)", 100)):
        fig, axes = plt.subplots(2, 6, figsize=(18, 7), sharex=True, sharey="row")
        for ax, (gpus, variant) in zip(axes.flat, product((66666, 6666), VARIANTS)):
            for policy in q.POLICIES:
                selected = sorted((r for r in rows if (r["gpus"], r["variant"], r["policy"]) == (gpus, variant, policy)), key=lambda r: r["deadline_s"])
                ax.plot([r["deadline_s"] for r in selected], [factor * r[metric] for r in selected], label=plot_style.POLICY_NAMES[policy], color=plot_style.POLICY_COLORS[policy], linestyle=plot_style.POLICY_LINESTYLES[policy])
            hit, private, _ = VARIANTS[variant]
            title = "Original wire baseline" if variant == "baseline" else f"800 MB; replay hit {hit:.0%}\nKV private {private:.0%}"
            ax.set(title=f"{title}\n{gpus:,} GPUs/site", xscale="log", xticks=[30, 120, 300], xticklabels=["30", "120", "300"])
            ax.set_ylim(bottom=0)
        fig.supxlabel("Handoff deadline (s)")
        fig.supylabel(label)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5)
        fig.tight_layout(rect=(.02, .02, 1, .91))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{metric}.{suffix}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", type=int, choices=range(len(CASES)))
    mode.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    reduce(args.out) if args.plot else run(args.case, args.out)
