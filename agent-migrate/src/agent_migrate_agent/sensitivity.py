"""Sensitivity sweep over the load_bearing cost_model constants.

Why this exists: the headline "shared_state_aware beats request_level" claim
sits on three constants — `kv_bytes_per_token`, `dst_prefill_tok_s`, and
`link_bps` — whose realistic ranges span >1 order of magnitude in 2025_2026.
A point_estimate result is not defensible. This sweep runs the bench across
a grid of (kv_bytes, link_bps) and reports whether the policy gap survives.

Usage:

    agent_migrate_sensitivity \\
        --trace examples/traces/toy_subagent_trace.jsonl \\
        --out runs/sensitivity_demo \\
        --kv_bytes 10000,70656,327680 \\
        --link_bps 5e9,25e9,100e9

The output is a CSV with one row per (kv_bytes, link_bps, policy) and a
boolean `gap_robust` column indicating whether `shared_state_aware` strictly
beats `request_level_with_site_cache` at that grid point.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ledger_progress import from_jsonl

from .manifest import build_manifest
from .metrics import cost_weighted_duplication_factor
from .policies import run_policy
from .profiles import LinkProfile, ProfileBundle, load_bundle


@dataclass(frozen=True)
class GridPoint:
    kv_bytes_per_token: int
    link_bps: float


def _override_bundle(bundle: ProfileBundle, kv_bytes: int, link_bps: float) -> ProfileBundle:
    """Return a new bundle with kv_bytes_per_token and all link bandwidths overridden."""
    new_model = type(bundle.model)(
        name=bundle.model.name,
        active_params_b=bundle.model.active_params_b,
        kv_bytes_per_token=kv_bytes,
        notes=bundle.model.notes,
    )
    new_links = {
        key: LinkProfile(site_a=link.site_a, site_b=link.site_b, effective_bps=link_bps)
        for key, link in bundle.links.items()
    }
    return ProfileBundle(
        model=new_model,
        sites=bundle.sites,
        links=new_links,
        home_site=bundle.home_site,
    )


GAP_REL_TOL = 1e-9  # ties within this relative gap classify as "no gap"


def run_sweep(
    trace_path: str | Path,
    out_dir: str | Path,
    model_path: str | Path,
    sites_path: str | Path,
    model_name: str,
    kv_bytes_grid: list[int],
    link_bps_grid: list[float],
    policies: list[str],
    tau: int = 1,
    reference_policy: str = "request_level_with_site_cache",
    challenger_policy: str = "shared_state_aware",
    gap_rel_tol: float = GAP_REL_TOL,
) -> list[dict]:
    """Run the bench across a grid of (kv_bytes, link_bps) points.

    Returns a list of result rows. Also writes `sensitivity.csv` to `out_dir`.
    `gap_robust` is True iff `challenger_policy` strictly beats
    `reference_policy` on cost_weighted duplication factor at that grid point.
    """
    trace_path = Path(trace_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = from_jsonl(str(trace_path))
    manifest = build_manifest(ledger)
    base_bundle = load_bundle(model_path, sites_path, model_name)

    rows: list[dict] = []
    for kv_bytes in kv_bytes_grid:
        for link_bps in link_bps_grid:
            bundle = _override_bundle(base_bundle, kv_bytes, link_bps)
            point_metrics: dict[str, dict] = {}
            for policy in policies:
                plan = run_policy(policy, manifest, bundle, tau=tau)
                point_metrics[policy] = {
                    "total_cost_s": plan.total_cost_s(),
                    "cost_weighted_duplication_factor": cost_weighted_duplication_factor(plan),
                }
            ref = point_metrics.get(reference_policy)
            chal = point_metrics.get(challenger_policy)
            gap_robust: bool | str = ""
            gap_abs_s: float | str = ""
            if ref is not None and chal is not None:
                gap_abs_s = ref["total_cost_s"] - chal["total_cost_s"]
                scale = max(abs(ref["total_cost_s"]), abs(chal["total_cost_s"]), 1e-30)
                gap_robust = gap_abs_s / scale > gap_rel_tol

            crossover_bps = 8.0 * kv_bytes * max(s.prefill_tok_s for s in bundle.sites.values())
            link_above_crossover = link_bps > crossover_bps
            for policy, metrics in point_metrics.items():
                rows.append({
                    "kv_bytes_per_token": kv_bytes,
                    "link_bps": link_bps,
                    "crossover_bps_seattle": crossover_bps,
                    "link_above_crossover": link_above_crossover,
                    "policy": policy,
                    "total_cost_s": metrics["total_cost_s"],
                    "cost_weighted_duplication_factor": metrics["cost_weighted_duplication_factor"],
                    "gap_robust": gap_robust,
                    "gap_abs_s": gap_abs_s,
                })

    _write_sweep_csv(rows, out_dir / "sensitivity.csv")
    return rows


def _write_sweep_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "kv_bytes_per_token", "link_bps", "crossover_bps_seattle",
        "link_above_crossover", "policy", "total_cost_s",
        "cost_weighted_duplication_factor", "gap_robust", "gap_abs_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "link_bps": f"{row['link_bps']:.6g}",
                "crossover_bps_seattle": f"{row['crossover_bps_seattle']:.6g}",
                "total_cost_s": f"{row['total_cost_s']:.9g}",
                "cost_weighted_duplication_factor": f"{row['cost_weighted_duplication_factor']:.9g}",
                "gap_abs_s": f"{row['gap_abs_s']:.9g}" if isinstance(row["gap_abs_s"], float) else "",
            })


def gap_survival_rate(rows: list[dict]) -> float:
    """Fraction of grid points where the challenger strictly beats the reference."""
    seen: set[tuple] = set()
    survived = 0
    total = 0
    for row in rows:
        key = (row["kv_bytes_per_token"], row["link_bps"])
        if key in seen:
            continue
        seen.add(key)
        if row["gap_robust"] == "":
            continue
        total += 1
        if row["gap_robust"]:
            survived += 1
    return survived / total if total else 0.0
