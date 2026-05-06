"""R3 — model_architecture axis for the K8 regime sweep.

Architecture changes two things at once (per
`kv_transfer_early_experiment/FINDINGS.md`):

  * `kv_bytes_per_token`           — KV_transfer wallclock at the link.
  * `single_stream_prefill_tok_s`  — replay wallclock at the destination.

R3 sweeps both. For each model profile we re_run the K8 cells with a
*model_aware* budget: the K8 site prefill capacity is interpreted as
"aggregate tokens/s for compact_kv (the K8 baseline), at this concurrency
level" and rescaled by the model's relative per_stream prefill rate. This
preserves K8's `loose / moderate / tight` axis as a *site concurrency*
knob while letting model architecture shift the absolute token budget.
The ResourceBudget link bandwidth is unchanged across models — link is
infrastructure, not architecture.

The headline artifact is a "regime_flip" table listing cells where the
best_policy or dominant_bottleneck label disagrees across architectures
— that is the only signal that justifies re_running the W_anchor regimes
(Workstream W) under more profiles.

R3 deliberately does not invent a new simulator, episode shape, or
policy; it just reuses K8's `run_k8_cell` / `estimate_k8_cell` code path
with a model_aware budget plug_in.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .episode import MobilityEpisode
from .fluid_sim import simulate_fluid
from .k8_regime import (
    LINK_BW_GBPS,
    N_WORKFLOWS,
    PREFILL_CAPS_TOK_S,
    STATE_SCALES_BYTES,
    K8_POLICIES,
    PolicyMetric,
    RegimeCell,
    _aggregate_plan_estimate,
    _dominant_bottleneck,
    _policy_metric,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
    summarize_cells,
)
from .manifest import ServingGroupManifest
from .profiles import ProfileBundle
from .resources import ResourceBudget
from .warmness import WarmnessMap


# Default architecture spread for R3. Order matters only for stable column
# layout in CSV artifacts. `compact_kv` is the K8 baseline so it stays first
# for diff readability.
R3_DEFAULT_PROFILES: tuple[str, ...] = (
    "compact_kv",         # MLA_like (Kimi_K2.6 class) — the K8 baseline
    "vanilla_gqa_fp16",   # vanilla GQA (Llama_3-70B class) — heaviest KV
    "frontier_v4_fp8",    # CSA/HCA compressed (DeepSeek_V4_Pro class) — lightest KV
    "glm_5_mla",          # MLA + DSA (GLM_5 class) — mid_range
    "qwen3_next_hybrid",  # hybrid (Qwen3_Next_80B_A3B class) — fastest replay
)

# The K8 baseline profile that defines the prefill_capacity axis: K8's
# (loose / moderate / tight) prefill_tok_s values are calibrated to
# compact_kv's per_stream rate. Other models scale relative to this.
_K8_BASELINE_PROFILE = "compact_kv"


@dataclass(frozen=True)
class R3CellRegime:
    cell_id: str
    n_workflows: int
    state_scale: str
    prefill_capacity: str
    link_gbps: int
    best_policy_by_model: dict[str, str]
    bottleneck_by_model: dict[str, str]
    mixed_p50_s_by_model: dict[str, float]

    @property
    def best_policy_flips(self) -> bool:
        return len(set(self.best_policy_by_model.values())) > 1

    @property
    def bottleneck_flips(self) -> bool:
        return len(set(self.bottleneck_by_model.values())) > 1


def make_r3_budget(
    cell: RegimeCell,
    bundle: ProfileBundle,
    *,
    baseline_prefill_tok_s: float | None = None,
) -> ResourceBudget:
    """K8 budget rescaled by the model's per_stream prefill rate.

    K8's prefill capacity is calibrated to `compact_kv` (the K8 baseline).
    For any other model, the same site's aggregate token rate scales with
    `single_stream_prefill_tok_s` since site concurrency (the loose /
    moderate / tight knob) is held fixed. Link bandwidth is unchanged
    across models.
    """
    if baseline_prefill_tok_s is None:
        baseline_prefill_tok_s = _baseline_prefill_tok_s(bundle)
    base = make_k8_budget(cell)
    rate_ratio = bundle.model.single_stream_prefill_tok_s / baseline_prefill_tok_s
    rescaled_prefill = {
        site: cap * rate_ratio
        for site, cap in base.prefill_tok_s_per_site.items()
    }
    return ResourceBudget(
        network_bps_per_link=base.network_bps_per_link,
        prefill_tok_s_per_site=rescaled_prefill,
        workspace_hydrate_bps_per_site=base.workspace_hydrate_bps_per_site,
        kv_memory_bytes_per_site=base.kv_memory_bytes_per_site,
    )


def run_r3_cell(
    cell: RegimeCell,
    bundle: ProfileBundle,
    *,
    baseline_prefill_tok_s: float | None = None,
) -> list[PolicyMetric]:
    """Exact K4 evaluation of one cell under one model profile."""
    if baseline_prefill_tok_s is None:
        baseline_prefill_tok_s = _baseline_prefill_tok_s(bundle)
    episode, manifests = make_k8_episode(cell)
    budget = make_r3_budget(cell, bundle, baseline_prefill_tok_s=baseline_prefill_tok_s)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    metrics: list[PolicyMetric] = []
    for policy_name, policy_fn in K8_POLICIES.items():
        if policy_name == "random_diversification":
            plan = policy_fn(episode, manifests, bundle, warmness, budget, seed=cell.seed)
        else:
            plan = policy_fn(episode, manifests, bundle, warmness, budget)
        result = simulate_fluid(episode, manifests, plan, bundle, warmness, budget)
        metrics.append(_policy_metric(cell, policy_name, result))
    return metrics


def estimate_r3_cell(
    cell: RegimeCell,
    bundle: ProfileBundle,
    *,
    baseline_prefill_tok_s: float | None = None,
) -> list[PolicyMetric]:
    """Aggregate_estimator path under one model profile."""
    if baseline_prefill_tok_s is None:
        baseline_prefill_tok_s = _baseline_prefill_tok_s(bundle)
    episode, manifests = make_k8_episode(cell)
    budget = make_r3_budget(cell, bundle, baseline_prefill_tok_s=baseline_prefill_tok_s)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    metrics: list[PolicyMetric] = []
    for policy_name, policy_fn in K8_POLICIES.items():
        if policy_name == "random_diversification":
            plan = policy_fn(episode, manifests, bundle, warmness, budget, seed=cell.seed)
        else:
            plan = policy_fn(episode, manifests, bundle, warmness, budget)
        p50, makespan, bottleneck = _aggregate_plan_estimate(
            episode, manifests, plan, bundle, warmness, budget,
        )
        metrics.append(PolicyMetric(
            cell=cell,
            policy=policy_name,
            p50_resume_s=p50,
            p90_resume_s=0.9 * makespan,
            p95_resume_s=0.95 * makespan,
            makespan_s=makespan,
            dominant_bottleneck=bottleneck,
        ))
    return metrics


def run_r3_sweep(
    repo_root: str | Path,
    *,
    model_names: tuple[str, ...] = R3_DEFAULT_PROFILES,
    n_values: tuple[int, ...] = N_WORKFLOWS,
    state_scales: tuple[str, ...] = tuple(STATE_SCALES_BYTES),
    prefill_caps: tuple[str, ...] = tuple(PREFILL_CAPS_TOK_S),
    link_gbps_values: tuple[int, ...] = LINK_BW_GBPS,
    exact: bool = False,
) -> dict[str, list[PolicyMetric]]:
    """Run the K8 cells once per model profile with a model_aware budget."""
    repo_root = Path(repo_root)
    baseline_bundle = default_bundle(repo_root, _K8_BASELINE_PROFILE)
    baseline_rate = baseline_bundle.model.single_stream_prefill_tok_s
    out: dict[str, list[PolicyMetric]] = {}
    runner = run_r3_cell if exact else estimate_r3_cell
    for name in model_names:
        bundle = default_bundle(repo_root, name)
        rows: list[PolicyMetric] = []
        for n in n_values:
            for state_scale in state_scales:
                for prefill_capacity in prefill_caps:
                    for link_gbps in link_gbps_values:
                        cell = RegimeCell(n, state_scale, prefill_capacity, link_gbps)
                        rows.extend(runner(
                            cell, bundle, baseline_prefill_tok_s=baseline_rate,
                        ))
        out[name] = rows
    return out


def summarize_r3(
    metrics_by_model: dict[str, list[PolicyMetric]],
) -> list[R3CellRegime]:
    """Collapse per_model metrics into per_cell regime rows."""
    summaries_by_model = {
        name: {s["cell_id"]: s for s in summarize_cells(rows)}
        for name, rows in metrics_by_model.items()
    }
    cell_ids: set[str] = set()
    for name in metrics_by_model:
        cell_ids.update(summaries_by_model[name])
    rows: list[R3CellRegime] = []
    for cell_id in sorted(cell_ids):
        first = next(
            summaries_by_model[name][cell_id]
            for name in metrics_by_model
            if cell_id in summaries_by_model[name]
        )
        rows.append(R3CellRegime(
            cell_id=cell_id,
            n_workflows=int(first["n_workflows"]),
            state_scale=str(first["state_scale"]),
            prefill_capacity=str(first["prefill_capacity"]),
            link_gbps=int(first["link_gbps"]),
            best_policy_by_model={
                name: str(summaries_by_model[name][cell_id]["best_policy"])
                for name in metrics_by_model
                if cell_id in summaries_by_model[name]
            },
            bottleneck_by_model={
                name: str(summaries_by_model[name][cell_id]["dominant_bottleneck"])
                for name in metrics_by_model
                if cell_id in summaries_by_model[name]
            },
            mixed_p50_s_by_model={
                name: float(summaries_by_model[name][cell_id]["mixed_p50_resume_s"])
                for name in metrics_by_model
                if cell_id in summaries_by_model[name]
            },
        ))
    return rows


def write_r3_artifacts(
    metrics_by_model: dict[str, list[PolicyMetric]],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = summarize_r3(metrics_by_model)
    model_names = list(metrics_by_model)

    with (out / "r3_regime_by_model.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cell_id", "n_workflows", "state_scale", "prefill_capacity",
            "link_gbps", "best_policy_flips", "bottleneck_flips",
            *(f"best_policy_{name}" for name in model_names),
            *(f"bottleneck_{name}" for name in model_names),
            *(f"mixed_p50_s_{name}" for name in model_names),
        ])
        writer.writeheader()
        for row in rows:
            record: dict[str, object] = {
                "cell_id": row.cell_id,
                "n_workflows": row.n_workflows,
                "state_scale": row.state_scale,
                "prefill_capacity": row.prefill_capacity,
                "link_gbps": row.link_gbps,
                "best_policy_flips": row.best_policy_flips,
                "bottleneck_flips": row.bottleneck_flips,
            }
            for name in model_names:
                record[f"best_policy_{name}"] = row.best_policy_by_model.get(name, "")
                record[f"bottleneck_{name}"] = row.bottleneck_by_model.get(name, "")
                p50 = row.mixed_p50_s_by_model.get(name)
                record[f"mixed_p50_s_{name}"] = "" if p50 is None else f"{p50:.9g}"
            writer.writerow(record)

    flips = [r for r in rows if r.best_policy_flips or r.bottleneck_flips]
    flip_summary = {
        "total_cells": len(rows),
        "best_policy_flips": sum(1 for r in rows if r.best_policy_flips),
        "bottleneck_flips": sum(1 for r in rows if r.bottleneck_flips),
        "models": model_names,
        "flip_cells": [
            {
                "cell_id": r.cell_id,
                "best_policy_by_model": dict(sorted(r.best_policy_by_model.items())),
                "bottleneck_by_model": dict(sorted(r.bottleneck_by_model.items())),
            }
            for r in flips
        ],
    }
    (out / "r3_flip_summary.json").write_text(
        json.dumps(flip_summary, indent=2, sort_keys=False) + "\n"
    )

    flip_count_by_axis = _flip_counts_by_axis(rows)
    (out / "r3_flip_counts_by_axis.json").write_text(
        json.dumps(flip_count_by_axis, indent=2, sort_keys=True) + "\n"
    )

    (out / "README.md").write_text(
        "# R3 — model_architecture axis for the K8 regime sweep\n\n"
        "`r3_regime_by_model.csv` holds the per_cell best_policy and "
        "dominant_bottleneck label under each model profile. The "
        "`best_policy_flips` and `bottleneck_flips` columns mark cells "
        "where architecture flips the regime label, and `r3_flip_summary.json` "
        "gives the headline counts.\n\n"
        "Each model profile is run with a model_aware budget: K8's prefill "
        "capacity (loose / moderate / tight) is rescaled by the model's "
        "per_stream prefill rate relative to compact_kv (the K8 baseline). "
        "Link bandwidth is unchanged.\n\n"
        "Aggregate K8 estimator caveat: the same calibration that applies to "
        "`runs/k8_regime_map/` applies here. Cells where the architecture "
        "flips the label are candidates for exact K4 + V1 re_validation; "
        "do not quote timing claims off this artifact alone.\n"
    )


def _baseline_prefill_tok_s(bundle: ProfileBundle) -> float:
    """Per_stream prefill rate that K8's `loose / moderate / tight`
    capacity values are calibrated to. We resolve this by reading
    `compact_kv` from the model file every time (rather than caching) so
    a profile_yaml change immediately re_anchors the rate."""
    if bundle.model.name == _K8_BASELINE_PROFILE:
        return bundle.model.single_stream_prefill_tok_s
    raise RuntimeError(
        "baseline_prefill_tok_s must be supplied when bundle is not the K8 baseline; "
        "callers should pass `default_bundle(repo, 'compact_kv').model.single_stream_prefill_tok_s`"
    )


def _flip_counts_by_axis(rows: list[R3CellRegime]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        "state_scale": defaultdict(lambda: {"best_policy_flips": 0, "bottleneck_flips": 0, "total": 0}),
        "prefill_capacity": defaultdict(lambda: {"best_policy_flips": 0, "bottleneck_flips": 0, "total": 0}),
        "link_gbps": defaultdict(lambda: {"best_policy_flips": 0, "bottleneck_flips": 0, "total": 0}),
        "n_workflows": defaultdict(lambda: {"best_policy_flips": 0, "bottleneck_flips": 0, "total": 0}),
    }
    for row in rows:
        for axis_name, axis_value in (
            ("state_scale", row.state_scale),
            ("prefill_capacity", row.prefill_capacity),
            ("link_gbps", str(row.link_gbps)),
            ("n_workflows", str(row.n_workflows)),
        ):
            bucket = counts[axis_name][axis_value]
            bucket["total"] += 1
            if row.best_policy_flips:
                bucket["best_policy_flips"] += 1
            if row.bottleneck_flips:
                bucket["bottleneck_flips"] += 1
    return {axis: dict(buckets) for axis, buckets in counts.items()}
