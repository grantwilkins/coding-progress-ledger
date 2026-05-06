"""Gate 1 exact claim-cell table."""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .fluid_sim import SimulationResult, simulate_fluid
from .k8_regime import (
    K8_POLICIES,
    RegimeCell,
    default_bundle,
    estimate_k8_cell,
    make_k8_budget,
    make_k8_episode,
)
from .k8_validation import ValidationTarget, default_validation_targets
from .profiles import ProfileBundle
from .warmness import WarmnessMap


@dataclass(frozen=True)
class ClaimCellRow:
    target: str
    cell_id: str
    n_workflows: int
    state_scale: str
    prefill_capacity: str
    link_gbps: int
    exact_best_policy: str
    aggregate_best_policy: str
    aggregate_best_policy_agrees: bool
    exact_best_policy_bottleneck: str
    aggregate_best_policy_bottleneck: str
    aggregate_exact_best_policy_bottleneck: str
    aggregate_exact_best_policy_bottleneck_agrees: bool
    resume_metric_kind: str
    exact_p50_k4_resume_proxy_s: float
    exact_p90_k4_resume_proxy_s: float
    exact_p99_k4_resume_proxy_s: float
    exact_makespan_s: float
    strong_reuse_p50_s: float
    mixed_min_pressure_p50_s: float
    random_diversification_p50_s: float
    mixed_beats_strong_reuse: bool
    mixed_beats_random_diversification: bool
    exact_claim_status: str
    aggregate_trust_status: str
    rationale: str


def run_claim_cells(
    bundle: ProfileBundle,
    targets: tuple[ValidationTarget, ...] | None = None,
) -> list[ClaimCellRow]:
    rows: list[ClaimCellRow] = []
    for target in targets or default_validation_targets():
        exact = _run_exact_results(target.cell, bundle)
        aggregate = {m.policy: m for m in estimate_k8_cell(target.cell, bundle)}
        exact_best = min(exact, key=lambda p: (exact[p].p50_resume_s(), p))
        aggregate_best = min(aggregate, key=lambda p: (aggregate[p].p50_resume_s, p))
        exact_best_bottleneck = _dominant_bottleneck(exact[exact_best])
        aggregate_best_bottleneck = aggregate[aggregate_best].dominant_bottleneck
        aggregate_exact_best_bottleneck = aggregate[exact_best].dominant_bottleneck
        mixed = exact["mixed_min_pressure"].p50_resume_s()
        strong = exact["strong_reuse"].p50_resume_s()
        random = exact["random_diversification"].p50_resume_s()
        row = ClaimCellRow(
            target=target.label,
            cell_id=target.cell.cell_id,
            n_workflows=target.cell.n_workflows,
            state_scale=target.cell.state_scale,
            prefill_capacity=target.cell.prefill_capacity,
            link_gbps=target.cell.link_gbps,
            exact_best_policy=exact_best,
            aggregate_best_policy=aggregate_best,
            aggregate_best_policy_agrees=exact_best == aggregate_best,
            exact_best_policy_bottleneck=exact_best_bottleneck,
            aggregate_best_policy_bottleneck=aggregate_best_bottleneck,
            aggregate_exact_best_policy_bottleneck=aggregate_exact_best_bottleneck,
            aggregate_exact_best_policy_bottleneck_agrees=(
                exact_best_bottleneck == aggregate_exact_best_bottleneck
            ),
            resume_metric_kind="k4_reconstitution_proxy_not_c5_task_resume",
            exact_p50_k4_resume_proxy_s=exact[exact_best].p50_resume_s(),
            exact_p90_k4_resume_proxy_s=exact[exact_best].p90_resume_s(),
            exact_p99_k4_resume_proxy_s=_percentile(exact[exact_best], 0.99),
            exact_makespan_s=exact[exact_best].makespan_s,
            strong_reuse_p50_s=strong,
            mixed_min_pressure_p50_s=mixed,
            random_diversification_p50_s=random,
            mixed_beats_strong_reuse=mixed < strong,
            mixed_beats_random_diversification=mixed < random,
            exact_claim_status=_exact_claim_status(mixed < strong, mixed < random),
            aggregate_trust_status=_aggregate_trust_status(
                exact_best == aggregate_best,
                exact_best_bottleneck == aggregate_exact_best_bottleneck,
            ),
            rationale=target.rationale,
        )
        rows.append(row)
    return rows


def write_claim_cell_table(rows: list[ClaimCellRow], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ClaimCellRow.__dataclass_fields__.keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(repo_root: str | Path) -> None:
    repo = Path(repo_root)
    bundle = default_bundle(repo)
    rows = run_claim_cells(bundle)
    write_claim_cell_table(rows, repo / "runs" / "claim_cells" / "exact_claim_cell_table.csv")


def _run_exact_results(cell: RegimeCell, bundle: ProfileBundle) -> dict[str, SimulationResult]:
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    results: dict[str, SimulationResult] = {}
    for policy_name, policy_fn in K8_POLICIES.items():
        if policy_name == "random_diversification":
            plan = policy_fn(episode, manifests, bundle, warmness, budget, seed=cell.seed)
        else:
            plan = policy_fn(episode, manifests, bundle, warmness, budget)
        results[policy_name] = simulate_fluid(episode, manifests, plan, bundle, warmness, budget)
    return results


def _percentile(result: SimulationResult, q: float) -> float:
    if not result.per_workflow_finish_s:
        return 0.0
    finishes = sorted(result.per_workflow_finish_s.values())
    idx = max(0, min(len(finishes) - 1, math.ceil(q * len(finishes)) - 1))
    return finishes[idx]


def _dominant_bottleneck(result: SimulationResult) -> str:
    weights: dict[str, float] = {}
    for action in result.actions:
        if action.bottleneck == "none":
            continue
        elapsed = max(action.finished_s - action.started_s, 0.0)
        weights[action.bottleneck] = weights.get(action.bottleneck, 0.0) + elapsed
    if not weights:
        return "none"
    return max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _exact_claim_status(mixed_beats_strong: bool, mixed_beats_random: bool) -> str:
    if mixed_beats_strong and not mixed_beats_random:
        return "planner_claim_ambiguous_random_competitive"
    if mixed_beats_strong and mixed_beats_random:
        return "richer_planning_supported"
    if not mixed_beats_strong and mixed_beats_random:
        return "strong_reuse_sufficient"
    return "random_diversification_competitive"


def _aggregate_trust_status(best_agrees: bool, bottleneck_agrees: bool) -> str:
    if best_agrees and bottleneck_agrees:
        return "aggregate_labels_match_exact"
    if best_agrees:
        return "aggregate_policy_only_matches_exact"
    if bottleneck_agrees:
        return "aggregate_bottleneck_only_matches_exact"
    return "aggregate_disagrees_use_exact_only"


__all__ = ["ClaimCellRow", "main", "run_claim_cells", "write_claim_cell_table"]
