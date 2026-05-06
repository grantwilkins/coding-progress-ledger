"""V1 exact validation for K8 aggregate regime_map cells.

K8's full heatmaps are intentionally aggregate_estimated.  This module
selects representative claim cells, reruns them through exact K4, and
reports where aggregate labels are usable versus where exact simulation is
required before making timing or bottleneck claims.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .k8_regime import (
    K8_POLICIES,
    PolicyMetric,
    RegimeCell,
    default_bundle,
    estimate_k8_cell,
    run_k8_cell,
    summarize_cells,
)
from .profiles import ProfileBundle


@dataclass(frozen=True)
class ValidationTarget:
    label: str
    cell: RegimeCell
    rationale: str


@dataclass(frozen=True)
class ValidationPolicyRow:
    target: ValidationTarget
    policy: str
    exact_p50_resume_s: float
    aggregate_p50_resume_s: float
    exact_p95_resume_s: float
    aggregate_p95_resume_s: float
    exact_makespan_s: float
    aggregate_makespan_s: float
    exact_dominant_bottleneck: str
    aggregate_dominant_bottleneck: str
    exact_best_policy: str
    aggregate_best_policy: str
    exact_best_policy_bottleneck: str
    aggregate_best_policy_bottleneck: str
    exact_cell_bottleneck: str
    aggregate_cell_bottleneck: str
    exact_winner_margin_frac: float

    @property
    def p50_relative_error(self) -> float:
        return _relative_error(self.exact_p50_resume_s, self.aggregate_p50_resume_s)

    @property
    def p95_relative_error(self) -> float:
        return _relative_error(self.exact_p95_resume_s, self.aggregate_p95_resume_s)

    @property
    def policy_bottleneck_agrees(self) -> bool:
        return self.exact_dominant_bottleneck == self.aggregate_dominant_bottleneck

    @property
    def best_policy_agrees(self) -> bool:
        return self.exact_best_policy == self.aggregate_best_policy

    @property
    def cell_bottleneck_agrees(self) -> bool:
        return self.exact_cell_bottleneck == self.aggregate_cell_bottleneck

    @property
    def best_policy_bottleneck_agrees(self) -> bool:
        return self.exact_best_policy_bottleneck == self.aggregate_best_policy_bottleneck


def default_validation_targets() -> tuple[ValidationTarget, ...]:
    """Representative cells for claim validation.

    All defaults stay at N<=100 so V1 remains refreshable in normal
    development while covering the main pressure classes used in K8/K9
    writing: reuse_scale SWE state, prefill pressure, slow links,
    multi_resource medium state, monorepo workspace pressure, and
    large_artifact network/workspace pressure.
    """
    return (
        ValidationTarget(
            "swe_bench_reuse_scale",
            RegimeCell(100, "swe_bench", "moderate", 25, seed=8101),
            "SWE_bench_sized state under moderate capacity; checks reuse_regime claims.",
        ),
        ValidationTarget(
            "tiny_prefill_pressure",
            RegimeCell(100, "tiny", "tight", 100, seed=8102),
            "Small state with tight prefill and fast link; candidate landing_pressure cell.",
        ),
        ValidationTarget(
            "tiny_slow_link",
            RegimeCell(100, "tiny", "loose", 1, seed=8103),
            "Small state with slow link; checks network_label behavior when bytes are small.",
        ),
        ValidationTarget(
            "medium_multi_resource",
            RegimeCell(100, "medium", "tight", 5, seed=8104),
            "Medium state, tight prefill, and 5 Gbps link; K9_style multi_resource cell.",
        ),
        ValidationTarget(
            "monorepo_workspace_pressure",
            RegimeCell(100, "monorepo", "loose", 100, seed=8105),
            "Large workspace with fast network; candidate workspace_locality cell.",
        ),
        ValidationTarget(
            "large_artifact_slow_link",
            RegimeCell(10, "large_artifact", "loose", 1, seed=8106),
            "Large artifact over slow link; checks extreme network/workspace boundary.",
        ),
        ValidationTarget(
            "large_artifact_fast_link",
            RegimeCell(10, "large_artifact", "loose", 100, seed=8107),
            "Large artifact over fast link; checks workspace_pressure label.",
        ),
    )


def run_k8_validation(
    bundle: ProfileBundle,
    targets: tuple[ValidationTarget, ...] | None = None,
) -> list[ValidationPolicyRow]:
    rows: list[ValidationPolicyRow] = []
    for target in targets or default_validation_targets():
        exact = {row.policy: row for row in run_k8_cell(target.cell, bundle)}
        aggregate = {row.policy: row for row in estimate_k8_cell(target.cell, bundle)}
        rows.extend(compare_policy_metrics(target, exact, aggregate))
    return rows


def compare_policy_metrics(
    target: ValidationTarget,
    exact: dict[str, PolicyMetric],
    aggregate: dict[str, PolicyMetric],
) -> list[ValidationPolicyRow]:
    expected = set(K8_POLICIES)
    if set(exact) != expected or set(aggregate) != expected:
        raise ValueError("exact and aggregate metrics must cover the fixed K8 policy set")

    exact_best = _best_policy(exact)
    aggregate_best = _best_policy(aggregate)
    exact_best_bottleneck = exact[exact_best].dominant_bottleneck
    aggregate_best_bottleneck = aggregate[aggregate_best].dominant_bottleneck
    exact_cell_bottleneck = _cell_bottleneck(list(exact.values()))
    aggregate_cell_bottleneck = _cell_bottleneck(list(aggregate.values()))
    margin = _winner_margin_frac(list(exact.values()))
    rows: list[ValidationPolicyRow] = []
    for policy in sorted(expected):
        e = exact[policy]
        a = aggregate[policy]
        rows.append(ValidationPolicyRow(
            target=target,
            policy=policy,
            exact_p50_resume_s=e.p50_resume_s,
            aggregate_p50_resume_s=a.p50_resume_s,
            exact_p95_resume_s=e.p95_resume_s,
            aggregate_p95_resume_s=a.p95_resume_s,
            exact_makespan_s=e.makespan_s,
            aggregate_makespan_s=a.makespan_s,
            exact_dominant_bottleneck=e.dominant_bottleneck,
            aggregate_dominant_bottleneck=a.dominant_bottleneck,
            exact_best_policy=exact_best,
            aggregate_best_policy=aggregate_best,
            exact_best_policy_bottleneck=exact_best_bottleneck,
            aggregate_best_policy_bottleneck=aggregate_best_bottleneck,
            exact_cell_bottleneck=exact_cell_bottleneck,
            aggregate_cell_bottleneck=aggregate_cell_bottleneck,
            exact_winner_margin_frac=margin,
        ))
    return rows


def summarize_validation(rows: list[ValidationPolicyRow]) -> list[dict[str, object]]:
    by_target: dict[str, list[ValidationPolicyRow]] = {}
    for row in rows:
        by_target.setdefault(row.target.label, []).append(row)

    summaries: list[dict[str, object]] = []
    for label, target_rows in sorted(by_target.items()):
        first = target_rows[0]
        p50_errors = [r.p50_relative_error for r in target_rows]
        p95_errors = [r.p95_relative_error for r in target_rows]
        bottleneck_agreements = sum(1 for r in target_rows if r.policy_bottleneck_agrees)
        summaries.append({
            "target": label,
            "cell_id": first.target.cell.cell_id,
            "n_workflows": first.target.cell.n_workflows,
            "state_scale": first.target.cell.state_scale,
            "prefill_capacity": first.target.cell.prefill_capacity,
            "link_gbps": first.target.cell.link_gbps,
            "rationale": first.target.rationale,
            "exact_best_policy": first.exact_best_policy,
            "aggregate_best_policy": first.aggregate_best_policy,
            "best_policy_agrees": first.best_policy_agrees,
            "exact_best_policy_bottleneck": first.exact_best_policy_bottleneck,
            "aggregate_best_policy_bottleneck": first.aggregate_best_policy_bottleneck,
            "best_policy_bottleneck_agrees": first.best_policy_bottleneck_agrees,
            "exact_cell_bottleneck": first.exact_cell_bottleneck,
            "aggregate_cell_bottleneck": first.aggregate_cell_bottleneck,
            "cell_bottleneck_agrees": first.cell_bottleneck_agrees,
            "policy_bottleneck_agreement": f"{bottleneck_agreements}/{len(target_rows)}",
            "median_p50_relative_error": _median(p50_errors),
            "max_p50_relative_error": max(p50_errors),
            "median_p95_relative_error": _median(p95_errors),
            "max_p95_relative_error": max(p95_errors),
            "exact_winner_margin_frac": first.exact_winner_margin_frac,
            "trust_label": _trust_label(first, p50_errors, p95_errors),
        })
    return summaries


def write_k8_validation_artifacts(rows: list[ValidationPolicyRow], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = summarize_validation(rows)
    with (out / "claim_cell_policy_validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "target", "cell_id", "n_workflows", "state_scale", "prefill_capacity",
            "link_gbps", "policy", "exact_p50_resume_s", "aggregate_p50_resume_s",
            "p50_relative_error", "exact_p95_resume_s", "aggregate_p95_resume_s",
            "p95_relative_error", "exact_makespan_s", "aggregate_makespan_s",
            "exact_dominant_bottleneck", "aggregate_dominant_bottleneck",
            "policy_bottleneck_agrees", "exact_best_policy", "aggregate_best_policy",
            "best_policy_agrees", "exact_best_policy_bottleneck",
            "aggregate_best_policy_bottleneck", "best_policy_bottleneck_agrees",
            "exact_cell_bottleneck", "aggregate_cell_bottleneck",
            "cell_bottleneck_agrees", "exact_winner_margin_frac",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(_policy_row_dict(row))

    with (out / "claim_cell_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)
    (out / "claim_cell_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / "README.md").write_text(_validation_readme(summaries))


def write_k8_validation_doc(
    rows: list[ValidationPolicyRow],
    out_path: str | Path,
) -> None:
    summaries = summarize_validation(rows)
    Path(out_path).write_text(_validation_doc(summaries) + "\n")


def _best_policy(metrics: dict[str, PolicyMetric]) -> str:
    return min(metrics.values(), key=lambda m: (m.p50_resume_s, m.policy)).policy


def _cell_bottleneck(metrics: list[PolicyMetric]) -> str:
    return str(summarize_cells(metrics)[0]["dominant_bottleneck"])


def _winner_margin_frac(metrics: list[PolicyMetric]) -> float:
    ordered = sorted(metrics, key=lambda m: (m.p50_resume_s, m.policy))
    if len(ordered) < 2:
        return 0.0
    best, second = ordered[0], ordered[1]
    if second.p50_resume_s <= 0.0:
        return 0.0
    return (second.p50_resume_s - best.p50_resume_s) / second.p50_resume_s


def _relative_error(exact: float, aggregate: float) -> float:
    if exact == 0.0:
        return 0.0 if aggregate == 0.0 else math.inf
    return abs(aggregate - exact) / exact


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _trust_label(
    row: ValidationPolicyRow,
    p50_errors: list[float],
    p95_errors: list[float],
) -> str:
    if row.best_policy_agrees and row.cell_bottleneck_agrees:
        if _median(p50_errors) <= 0.25 and _median(p95_errors) <= 0.25:
            return "timing_reliable"
        return "label_reliable"
    if not row.best_policy_agrees and row.exact_winner_margin_frac <= 0.05:
        return "policy_boundary"
    return "needs_exact_k4"


def _policy_row_dict(row: ValidationPolicyRow) -> dict[str, object]:
    cell = row.target.cell
    return {
        "target": row.target.label,
        "cell_id": cell.cell_id,
        "n_workflows": cell.n_workflows,
        "state_scale": cell.state_scale,
        "prefill_capacity": cell.prefill_capacity,
        "link_gbps": cell.link_gbps,
        "policy": row.policy,
        "exact_p50_resume_s": f"{row.exact_p50_resume_s:.9g}",
        "aggregate_p50_resume_s": f"{row.aggregate_p50_resume_s:.9g}",
        "p50_relative_error": f"{row.p50_relative_error:.9g}",
        "exact_p95_resume_s": f"{row.exact_p95_resume_s:.9g}",
        "aggregate_p95_resume_s": f"{row.aggregate_p95_resume_s:.9g}",
        "p95_relative_error": f"{row.p95_relative_error:.9g}",
        "exact_makespan_s": f"{row.exact_makespan_s:.9g}",
        "aggregate_makespan_s": f"{row.aggregate_makespan_s:.9g}",
        "exact_dominant_bottleneck": row.exact_dominant_bottleneck,
        "aggregate_dominant_bottleneck": row.aggregate_dominant_bottleneck,
        "policy_bottleneck_agrees": row.policy_bottleneck_agrees,
        "exact_best_policy": row.exact_best_policy,
        "aggregate_best_policy": row.aggregate_best_policy,
        "best_policy_agrees": row.best_policy_agrees,
        "exact_best_policy_bottleneck": row.exact_best_policy_bottleneck,
        "aggregate_best_policy_bottleneck": row.aggregate_best_policy_bottleneck,
        "best_policy_bottleneck_agrees": row.best_policy_bottleneck_agrees,
        "exact_cell_bottleneck": row.exact_cell_bottleneck,
        "aggregate_cell_bottleneck": row.aggregate_cell_bottleneck,
        "cell_bottleneck_agrees": row.cell_bottleneck_agrees,
        "exact_winner_margin_frac": f"{row.exact_winner_margin_frac:.9g}",
    }


def _validation_readme(summaries: list[dict[str, object]]) -> str:
    return (
        "# K8 exact_validation artifacts\n\n"
        "`claim_cell_policy_validation.csv` compares exact K4 and aggregate "
        "K8 estimates for each selected claim cell and policy. "
        "`claim_cell_summary.csv` / `.json` collapse that into best_policy "
        "agreement, bottleneck agreement, p50/p95 timing error, and a trust "
        "label.\n\n"
        "Trust labels:\n"
        "- `timing_reliable`: aggregate best policy and bottleneck agree, "
        "with median p50 and p95 relative error <= 25%.\n"
        "- `label_reliable`: labels agree, but timing should not be quoted.\n"
        "- `policy_boundary`: aggregate/exact best differs, but the exact "
        "winner margin is <= 5%.\n"
        "- `needs_exact_k4`: use exact K4 before making claims.\n\n"
        f"Validated cells: {len(summaries)}.\n"
    )


def _validation_doc(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# V1 — exact validation of K8 regime cells",
        "",
        "**Artifacts:** `runs/k8_validation/`  ",
        "**Runner:** `uv run python scripts/run_k8_validation.py`  ",
        "**Date:** 2026_05_06",
        "",
        "## Purpose",
        "",
        "K8 heatmaps are aggregate_estimated and should be read as candidate "
        "regime discovery. V1 reruns selected claim cells through exact K4 "
        "and records best_policy agreement, dominant_bottleneck agreement, "
        "and p50/p95 timing error.",
        "",
        "## Summary",
        "",
        "| Target | Cell | Exact best | Aggregate best | Exact bottleneck | "
        "Aggregate bottleneck | Exact best bottleneck | Median p50 err | Median p95 err | Trust |",
        "| ------ | ---- | ---------- | -------------- | ---------------- | "
        "-------------------- | --------------------- | --------------: | --------------: | ----- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target']} | `{row['cell_id']}` | "
            f"{row['exact_best_policy']} | {row['aggregate_best_policy']} | "
            f"{row['exact_cell_bottleneck']} | {row['aggregate_cell_bottleneck']} | "
            f"{row['exact_best_policy_bottleneck']} | "
            f"{float(row['median_p50_relative_error']):.1%} | "
            f"{float(row['median_p95_relative_error']):.1%} | "
            f"{row['trust_label']} |"
        )
    lines.extend([
        "",
        "## Reading",
        "",
        "Only cells labeled `timing_reliable` should be used for aggregate "
        "timing claims. `label_reliable` cells can support qualitative regime "
        "labels but need exact K4 for wall_clock numbers. `policy_boundary` "
        "means the heatmap winner is unstable near a small exact margin. "
        "`needs_exact_k4` means aggregate K8 is useful only as a search hint.",
        "",
        "Two caveats matter. First, aggregate p50/p95 are service_time "
        "approximations (`0.50 * makespan` and `0.95 * makespan`), not "
        "completion_CDF estimates. Second, K8's cell bottleneck heatmap uses "
        "the existing cell_level summary convention, while exact K4 also "
        "reports the bottleneck of the exact winning policy. Those definitions "
        "should not be mixed in paper claims.",
    ])
    return "\n".join(lines)


def main(repo_root: str | Path) -> None:
    repo = Path(repo_root)
    bundle = default_bundle(repo)
    rows = run_k8_validation(bundle)
    write_k8_validation_artifacts(rows, repo / "runs" / "k8_validation")
    write_k8_validation_doc(rows, repo / "docs" / "K8_exact_validation.md")


__all__ = [
    "ValidationPolicyRow",
    "ValidationTarget",
    "compare_policy_metrics",
    "default_validation_targets",
    "main",
    "run_k8_validation",
    "summarize_validation",
    "write_k8_validation_artifacts",
    "write_k8_validation_doc",
]
