"""Bench orchestrator: trace -> manifest -> per_policy plans -> CSV + plot."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from ledger_progress import from_jsonl

from .manifest import build_manifest
from .manifest_io import write_json as write_manifest_json
from .metrics import cost_weighted_duplication_factor, repeated_prefix_fraction, state_layer_breakdown
from .plans_io import write_plan
from .policies import Plan, run_policy
from .profiles import load_bundle


AUDIT_COLUMNS = (
    "policy", "state_id", "content_hash", "state_layer", "site", "mode",
    "tokens", "bytes", "cost_s", "materialization_count", "ideal_materialization_count",
    "total_cost_s", "num_consumers", "consumer_node_ids", "reason",
)


def run_bench(
    trace_path: str | Path,
    out_dir: str | Path,
    policies: list[str],
    model_path: str | Path,
    sites_path: str | Path,
    model_name: str,
    tau: int = 1,
) -> dict:
    trace_path = Path(trace_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = from_jsonl(str(trace_path))
    manifest = build_manifest(ledger)
    write_manifest_json(manifest, out_dir / "manifest.json")
    bundle = load_bundle(model_path, sites_path, model_name)

    plans: dict[str, Plan] = {}
    for name in policies:
        plan = run_policy(name, manifest, bundle, tau=tau)
        plans[name] = plan
        plan_dir = out_dir / name
        write_plan(plan, plan_dir)

    _write_audit_csv(plans, manifest, out_dir / "state_materialization_breakdown.csv")
    summary = _summary(plans, manifest, tau=tau)
    _write_summary_csv(summary, out_dir / "results.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_audit_csv(plans: dict[str, Plan], manifest, out_path: Path) -> None:
    rows = []
    for policy_name, plan in plans.items():
        for m in plan.materializations:
            state = manifest.state_objects[m.state_id]
            rows.append({
                "policy": policy_name,
                "state_id": m.state_id,
                "content_hash": m.content_hash,
                "state_layer": state.layer,
                "site": m.site,
                "mode": m.mode,
                "tokens": state.tokens,
                "bytes": state.bytes if state.bytes is not None else "",
                "cost_s": f"{m.cost_s:.9g}",
                "materialization_count": m.materialization_count,
                "ideal_materialization_count": 1,
                "total_cost_s": f"{m.total_cost_s:.9g}",
                "num_consumers": len(m.consumers),
                "consumer_node_ids": ",".join(m.consumers),
                "reason": m.reason,
            })
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _summary(plans: dict[str, Plan], manifest, tau: int) -> dict:
    return {
        "manifest": {
            "workflow_id": manifest.workflow_id,
            "node_count": len(manifest.nodes),
            "state_count": len(manifest.state_objects),
            "repeated_prefix_fraction": repeated_prefix_fraction(manifest),
        },
        "tau": tau,
        "policies": {
            name: {
                "total_cost_s": plan.total_cost_s(),
                "cost_weighted_duplication_factor": cost_weighted_duplication_factor(plan),
                "state_layer_breakdown": state_layer_breakdown(plan, manifest),
                "n_placements": len(plan.placements),
                "n_materialization_rows": len(plan.materializations),
            }
            for name, plan in plans.items()
        },
    }


def _write_summary_csv(summary: dict, out_path: Path) -> None:
    columns = ["policy", "tau", "total_cost_s", "cost_weighted_duplication_factor",
               "n_materialization_rows"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, info in summary["policies"].items():
            writer.writerow({
                "policy": name,
                "tau": summary["tau"],
                "total_cost_s": f"{info['total_cost_s']:.9g}",
                "cost_weighted_duplication_factor":
                    f"{info['cost_weighted_duplication_factor']:.9g}",
                "n_materialization_rows": info["n_materialization_rows"],
            })
