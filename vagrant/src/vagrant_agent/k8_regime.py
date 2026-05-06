"""K8 regime-map sweep.

K8 asks a different question than K7: not "does one policy win?", but
"which abstraction is sufficient in this cell of the mobility design
space?"  The sweep is intentionally synthetic and deterministic; it is a
map-building harness, not a production trace adapter.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .adapters.herd import HerdSpec, build_herd_episode
from .episode import MobilityEpisode
from .fluid_sim import ALL_RESOURCES, SimulationResult, simulate_fluid
from .manifest import ServingGroupManifest
from .profiles import ProfileBundle, load_bundle
from .reconstitution import (
    cache_reuse,
    kv_all,
    mixed_min_pressure,
    random_mode,
    replay_all,
    workspace_sticky,
)
from .resources import ResourceBudget
from .resources import reconstitution_cost
from .warmness import WarmnessMap


STATE_SCALES_BYTES: dict[str, int] = {
    "tiny": 30_000_000,
    "swe_bench": 33_000_000,
    "medium": 500_000_000,
    "monorepo": 5_000_000_000,
    "large_artifact": 20_000_000_000,
}

PREFILL_CAPS_TOK_S: dict[str, float] = {
    "loose": 300_000.0,
    "moderate": 100_000.0,
    "tight": 30_000.0,
}

LINK_BW_GBPS: tuple[int, ...] = (1, 5, 25, 100)
N_WORKFLOWS: tuple[int, ...] = (10, 100, 1_000, 10_000)

K8_POLICIES = {
    "strong_reuse": cache_reuse,
    "replay_all": replay_all,
    "kv_all": kv_all,
    "workspace_sticky": workspace_sticky,
    "random_diversification": random_mode,
    "mixed_min_pressure": mixed_min_pressure,
}


@dataclass(frozen=True)
class RegimeCell:
    n_workflows: int
    state_scale: str
    prefill_capacity: str
    link_gbps: int
    seed: int = 8000

    @property
    def cell_id(self) -> str:
        return (
            f"n{self.n_workflows}_{self.state_scale}_"
            f"{self.prefill_capacity}_{self.link_gbps}g"
        )


@dataclass(frozen=True)
class PolicyMetric:
    cell: RegimeCell
    policy: str
    p50_resume_s: float
    p90_resume_s: float
    p95_resume_s: float
    makespan_s: float
    dominant_bottleneck: str


@dataclass(frozen=True)
class EstimatorCalibrationRow:
    cell: RegimeCell
    policy: str
    exact_p50_resume_s: float
    aggregate_p50_resume_s: float
    exact_dominant_bottleneck: str
    aggregate_dominant_bottleneck: str

    @property
    def relative_p50_error(self) -> float:
        if self.exact_p50_resume_s == 0.0:
            return 0.0 if self.aggregate_p50_resume_s == 0.0 else math.inf
        return abs(self.aggregate_p50_resume_s - self.exact_p50_resume_s) / self.exact_p50_resume_s

    @property
    def bottleneck_agrees(self) -> bool:
        return self.exact_dominant_bottleneck == self.aggregate_dominant_bottleneck


def default_bundle(repo_root: str | Path, model_name: str = "compact_kv") -> ProfileBundle:
    root = Path(repo_root)
    return load_bundle(root / "configs" / "model_profiles.yaml",
                       root / "configs" / "sites_3site.yaml",
                       model_name)


def make_k8_budget(cell: RegimeCell) -> ResourceBudget:
    sites = ("phoenix", "seattle", "austin")
    link_bps = float(cell.link_gbps) * 1e9
    prefill = PREFILL_CAPS_TOK_S[cell.prefill_capacity]
    return ResourceBudget(
        network_bps_per_link={
            tuple(sorted([a, b])): link_bps
            for i, a in enumerate(sites) for b in sites[i + 1:]
        },
        prefill_tok_s_per_site={site: prefill for site in sites},
        workspace_hydrate_bps_per_site={site: 1e9 for site in sites},
        kv_memory_bytes_per_site={site: math.inf for site in sites},
    )


def make_k8_episode(cell: RegimeCell) -> tuple[MobilityEpisode, dict[str, ServingGroupManifest]]:
    """Build a deterministic single-source evacuation cell.

    The herd adapter's distributions are useful for K7, but K8 needs named
    state-scale axes.  We build a tiny-distribution episode for stable shape
    and then deterministically replace each workspace payload with the exact
    scale requested by the cell.
    """
    spec = HerdSpec(
        n_workflows=cell.n_workflows,
        workspace_bytes_distribution="tiny",
        prompt_tokens_distribution="medium",
        warm_cache_fraction=0.0,
        home_asymmetry="all_same",
        seed=cell.seed + cell.n_workflows,
    )
    episode, manifests = build_herd_episode(
        spec,
        source_sites=("phoenix",),
        destination_sites=("seattle", "austin"),
        episode_id=f"k8_{cell.cell_id}",
    )
    target_bytes = STATE_SCALES_BYTES[cell.state_scale]
    for manifest in manifests.values():
        for state in manifest.state_objects.values():
            if state.layer == "workspace":
                state.bytes = target_bytes
                state.content_hash = f"{state.content_hash}:{cell.state_scale}:{target_bytes}"
    return replace(episode, notes=f"K8 regime cell {cell.cell_id}"), manifests


def run_k8_cell(cell: RegimeCell, bundle: ProfileBundle | None = None) -> list[PolicyMetric]:
    if bundle is None:
        raise ValueError("bundle is required; call default_bundle(repo_root) at the boundary")
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
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


def estimate_k8_cell(cell: RegimeCell, bundle: ProfileBundle | None = None) -> list[PolicyMetric]:
    """Fast deterministic aggregate evaluator for full K8 sweeps.

    Exact K4 is retained for focused cells and tests.  Full K8 includes
    10K-workflow cells; the event-level simulator is intentionally simple
    and too slow there, so the map uses an aggregate service-time estimate:
    sum each policy's unique cold materialization demand per resource and
    divide by the configured capacity.  The estimator preserves the K3/K5
    resource semantics and bottleneck ordering, but not per-action event
    timing.
    """
    if bundle is None:
        raise ValueError("bundle is required; call default_bundle(repo_root) at the boundary")
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
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


def run_k8_sweep(
    bundle: ProfileBundle,
    *,
    n_values: tuple[int, ...] = N_WORKFLOWS,
    state_scales: tuple[str, ...] = tuple(STATE_SCALES_BYTES),
    prefill_caps: tuple[str, ...] = tuple(PREFILL_CAPS_TOK_S),
    link_gbps_values: tuple[int, ...] = LINK_BW_GBPS,
    exact: bool = False,
) -> list[PolicyMetric]:
    rows: list[PolicyMetric] = []
    for n in n_values:
        for state_scale in state_scales:
            for prefill_capacity in prefill_caps:
                for link_gbps in link_gbps_values:
                    runner = run_k8_cell if exact else estimate_k8_cell
                    rows.extend(runner(
                        RegimeCell(n, state_scale, prefill_capacity, link_gbps),
                        bundle,
                    ))
    return rows


def calibrate_k8_estimator(
    bundle: ProfileBundle,
    *,
    n_values: tuple[int, ...] = (10, 100),
    state_scales: tuple[str, ...] = ("tiny", "medium", "monorepo"),
    prefill_caps: tuple[str, ...] = ("loose", "tight"),
    link_gbps_values: tuple[int, ...] = (1, 25, 100),
) -> list[EstimatorCalibrationRow]:
    rows: list[EstimatorCalibrationRow] = []
    for n in n_values:
        for state_scale in state_scales:
            for prefill_capacity in prefill_caps:
                for link_gbps in link_gbps_values:
                    cell = RegimeCell(n, state_scale, prefill_capacity, link_gbps)
                    exact = {row.policy: row for row in run_k8_cell(cell, bundle)}
                    aggregate = {row.policy: row for row in estimate_k8_cell(cell, bundle)}
                    for policy in sorted(exact):
                        rows.append(EstimatorCalibrationRow(
                            cell=cell,
                            policy=policy,
                            exact_p50_resume_s=exact[policy].p50_resume_s,
                            aggregate_p50_resume_s=aggregate[policy].p50_resume_s,
                            exact_dominant_bottleneck=exact[policy].dominant_bottleneck,
                            aggregate_dominant_bottleneck=aggregate[policy].dominant_bottleneck,
                        ))
    return rows


def summarize_cells(rows: list[PolicyMetric]) -> list[dict[str, object]]:
    by_cell: dict[RegimeCell, list[PolicyMetric]] = {}
    for row in rows:
        by_cell.setdefault(row.cell, []).append(row)
    summaries: list[dict[str, object]] = []
    for cell, metrics in sorted(by_cell.items(), key=lambda kv: kv[0].cell_id):
        best = min(metrics, key=lambda m: (m.p50_resume_s, m.policy))
        strong = next(m for m in metrics if m.policy == "strong_reuse")
        mixed = next(m for m in metrics if m.policy == "mixed_min_pressure")
        bottleneck = _dominant_bottleneck([m.dominant_bottleneck for m in metrics])
        gap = (
            (strong.p50_resume_s - mixed.p50_resume_s) / strong.p50_resume_s
            if strong.p50_resume_s > 0 else 0.0
        )
        summaries.append({
            "cell_id": cell.cell_id,
            "n_workflows": cell.n_workflows,
            "state_scale": cell.state_scale,
            "prefill_capacity": cell.prefill_capacity,
            "link_gbps": cell.link_gbps,
            "best_policy": best.policy,
            "best_p50_resume_s": best.p50_resume_s,
            "dominant_bottleneck": bottleneck,
            "strong_reuse_p50_resume_s": strong.p50_resume_s,
            "mixed_p50_resume_s": mixed.p50_resume_s,
            "mixed_vs_strong_reuse_gap_frac": gap,
        })
    return summaries


def write_k8_artifacts(rows: list[PolicyMetric], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "regime_policy_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cell_id", "n_workflows", "state_scale", "prefill_capacity",
            "link_gbps", "policy", "p50_resume_s", "p90_resume_s",
            "p95_resume_s",
            "makespan_s", "dominant_bottleneck",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "cell_id": row.cell.cell_id,
                "n_workflows": row.cell.n_workflows,
                "state_scale": row.cell.state_scale,
                "prefill_capacity": row.cell.prefill_capacity,
                "link_gbps": row.cell.link_gbps,
                "policy": row.policy,
                "p50_resume_s": f"{row.p50_resume_s:.9g}",
                "p90_resume_s": f"{row.p90_resume_s:.9g}",
                "p95_resume_s": f"{row.p95_resume_s:.9g}",
                "makespan_s": f"{row.makespan_s:.9g}",
                "dominant_bottleneck": row.dominant_bottleneck,
            })
    summaries = summarize_cells(rows)
    with (out / "regime_cell_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)
    (out / "regime_cell_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot_k8_heatmaps(summaries, out)
    (out / "README.md").write_text(
        "# K8 regime-map artifacts\n\n"
        "`regime_policy_metrics.csv` contains per-policy metrics. "
        "`regime_cell_summary.csv` / `.json` contain the best-policy and "
        "dominant-bottleneck map. The emitted full sweep uses K8's aggregate "
        "service-time estimator so 1K/10K workflow cells are tractable; "
        "`run_k8_cell(... )` remains the exact K4 simulator path for focused "
        "validation cells. `exact_vs_aggregate.csv`, when present, compares "
        "sampled exact K4 cells to the aggregate estimator and is the source "
        "of truth for how much confidence to put in aggregate heatmap labels.\n"
    )


def write_k8_calibration_artifacts(
    rows: list[EstimatorCalibrationRow],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_by_cell: dict[RegimeCell, tuple[str, str]] = {}
    by_cell: dict[RegimeCell, list[EstimatorCalibrationRow]] = {}
    for row in rows:
        by_cell.setdefault(row.cell, []).append(row)
    for cell, cell_rows in by_cell.items():
        exact_best = min(
            cell_rows,
            key=lambda r: (r.exact_p50_resume_s, r.policy),
        ).policy
        aggregate_best = min(
            cell_rows,
            key=lambda r: (r.aggregate_p50_resume_s, r.policy),
        ).policy
        best_by_cell[cell] = (exact_best, aggregate_best)
    with (out / "exact_vs_aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cell_id", "n_workflows", "state_scale", "prefill_capacity",
            "link_gbps", "policy", "exact_p50_resume_s",
            "aggregate_p50_resume_s", "relative_p50_error",
            "exact_dominant_bottleneck", "aggregate_dominant_bottleneck",
            "bottleneck_agrees", "exact_best_policy", "aggregate_best_policy",
            "best_policy_agrees",
        ])
        writer.writeheader()
        for row in rows:
            exact_best, aggregate_best = best_by_cell[row.cell]
            writer.writerow({
                "cell_id": row.cell.cell_id,
                "n_workflows": row.cell.n_workflows,
                "state_scale": row.cell.state_scale,
                "prefill_capacity": row.cell.prefill_capacity,
                "link_gbps": row.cell.link_gbps,
                "policy": row.policy,
                "exact_p50_resume_s": f"{row.exact_p50_resume_s:.9g}",
                "aggregate_p50_resume_s": f"{row.aggregate_p50_resume_s:.9g}",
                "relative_p50_error": f"{row.relative_p50_error:.9g}",
                "exact_dominant_bottleneck": row.exact_dominant_bottleneck,
                "aggregate_dominant_bottleneck": row.aggregate_dominant_bottleneck,
                "bottleneck_agrees": row.bottleneck_agrees,
                "exact_best_policy": exact_best,
                "aggregate_best_policy": aggregate_best,
                "best_policy_agrees": exact_best == aggregate_best,
            })


def plot_k8_heatmaps(summaries: list[dict[str, object]], out_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    state_scales = list(STATE_SCALES_BYTES)
    n_values = list(N_WORKFLOWS)
    policies = sorted({str(s["best_policy"]) for s in summaries})
    policy_to_i = {p: i for i, p in enumerate(policies)}
    bottlenecks = list(ALL_RESOURCES)
    bottleneck_to_i = {b: i for i, b in enumerate(bottlenecks)}

    for prefill_capacity in PREFILL_CAPS_TOK_S:
        for link_gbps in LINK_BW_GBPS:
            panel = [
                s for s in summaries
                if s["prefill_capacity"] == prefill_capacity and s["link_gbps"] == link_gbps
            ]
            if not panel:
                continue
            _plot_label_heatmap(
                panel, n_values, state_scales,
                value_key="best_policy",
                value_to_i=policy_to_i,
                labels=policies,
                title=f"K8 best policy: {prefill_capacity} prefill, {link_gbps} Gbps link",
                out_path=out / f"best_policy_{prefill_capacity}_{link_gbps}g.png",
            )
            _plot_label_heatmap(
                panel, n_values, state_scales,
                value_key="dominant_bottleneck",
                value_to_i=bottleneck_to_i,
                labels=bottlenecks,
                title=f"K8 dominant bottleneck: {prefill_capacity} prefill, {link_gbps} Gbps link",
                out_path=out / f"dominant_bottleneck_{prefill_capacity}_{link_gbps}g.png",
            )


def _policy_metric(cell: RegimeCell, policy_name: str, result: SimulationResult) -> PolicyMetric:
    return PolicyMetric(
        cell=cell,
        policy=policy_name,
        p50_resume_s=result.p50_resume_s(),
        p90_resume_s=result.p90_resume_s(),
        p95_resume_s=result.p95_resume_s(),
        makespan_s=result.makespan_s,
        dominant_bottleneck=_dominant_bottleneck([a.bottleneck for a in result.actions]),
    )


def _aggregate_plan_estimate(
    episode: MobilityEpisode,
    manifests: dict[str, ServingGroupManifest],
    plan,
    bundle: ProfileBundle,
    warmness: WarmnessMap,
    budget: ResourceBudget,
) -> tuple[float, float, str]:
    network_bits: dict[tuple[str, str], float] = {}
    prefill_tokens: dict[str, float] = {}
    workspace_bytes: dict[str, float] = {}
    materialized: set[tuple[str, str]] = set()

    for wf in episode.workflows:
        src = wf.src_site or episode.source_sites[0]
        for action in plan[wf.workflow_id]:
            key = (action.state_id, action.dst_site)
            if key in materialized or warmness.is_warm(action.state_id, action.dst_site):
                continue
            state = manifests[wf.workflow_id].state_objects[action.state_id]
            cost = reconstitution_cost(
                state, action.mode, action.src_site or src, action.dst_site, bundle, warmness,
            )
            materialized.add(key)
            if cost.network_bytes:
                link = tuple(sorted([action.src_site or src, action.dst_site]))
                network_bits[link] = network_bits.get(link, 0.0) + 8.0 * cost.network_bytes
            if cost.prefill_tokens:
                prefill_tokens[action.dst_site] = (
                    prefill_tokens.get(action.dst_site, 0.0) + cost.prefill_tokens
                )
            if cost.workspace_bytes:
                workspace_bytes[action.dst_site] = (
                    workspace_bytes.get(action.dst_site, 0.0) + cost.workspace_bytes
                )

    service_times: dict[str, float] = {}
    for link, bits in network_bits.items():
        cap = budget.network_bps_per_link.get(link, math.inf)
        service_times[f"network:{link[0]}-{link[1]}"] = 0.0 if cap == math.inf else bits / cap
    for site, tokens in prefill_tokens.items():
        cap = budget.prefill_tok_s_per_site.get(site, math.inf)
        service_times[f"prefill:{site}"] = 0.0 if cap == math.inf else tokens / cap
    for site, bytes_ in workspace_bytes.items():
        cap = budget.workspace_hydrate_bps_per_site.get(site, math.inf)
        service_times[f"workspace:{site}"] = 0.0 if cap == math.inf else bytes_ / cap

    if not service_times:
        return 0.0, 0.0, "none"
    resource, makespan = max(service_times.items(), key=lambda kv: (kv[1], kv[0]))
    bottleneck = resource.split(":", 1)[0]
    return 0.5 * makespan, makespan, bottleneck


def _dominant_bottleneck(values: list[str]) -> str:
    values = [v for v in values if v != "none"]
    if not values:
        return "none"
    counts = {v: values.count(v) for v in set(values)}
    max_count = max(counts.values())
    tied = sorted(v for v, count in counts.items() if count == max_count)
    if len(tied) == 1:
        return tied[0]
    # Stable tie-break: choose the median bottleneck among tied resources by
    # first appearance, rather than alphabetic order.
    first_positions = {v: values.index(v) for v in tied}
    return min(tied, key=lambda v: first_positions[v])


def _plot_label_heatmap(
    panel: list[dict[str, object]],
    n_values: list[int],
    state_scales: list[str],
    *,
    value_key: str,
    value_to_i: dict[str, int],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    grid = [[math.nan for _ in state_scales] for _ in n_values]
    lookup = {
        (int(s["n_workflows"]), str(s["state_scale"])): str(s[value_key])
        for s in panel
    }
    for y, n in enumerate(n_values):
        for x, state_scale in enumerate(state_scales):
            value = lookup.get((n, state_scale))
            if value is not None:
                grid[y][x] = value_to_i[value]

    cmap = ListedColormap(plt.get_cmap("tab10").colors[:max(1, len(labels))])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(0, len(labels) - 1), aspect="auto")
    ax.set_xticks(range(len(state_scales)), state_scales, rotation=25, ha="right")
    ax.set_yticks(range(len(n_values)), [str(n) for n in n_values])
    ax.set_xlabel("workspace/artifact scale")
    ax.set_ylabel("N workflows")
    ax.set_title(title)
    for y, n in enumerate(n_values):
        for x, state_scale in enumerate(state_scales):
            label = lookup.get((n, state_scale), "")
            ax.text(x, y, _short_label(label), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, ticks=range(len(labels)))
    cbar.ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _short_label(label: str) -> str:
    parts = label.split("_")
    if label == "random_diversification":
        return "random"
    if label == "mixed_min_pressure":
        return "mixed"
    if label == "workspace_sticky":
        return "sticky"
    if label == "strong_reuse":
        return "reuse"
    return parts[0] if parts else label
