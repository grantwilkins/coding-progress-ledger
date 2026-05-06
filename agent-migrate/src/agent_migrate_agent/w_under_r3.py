"""Retest W1/W2/W3 workload anchors under the R3 model_profile axis.

Triggered by the R3 finding that architecture flips the dominant
bottleneck label on ≥25% of K8 cells (workspace vs prefill vs network).
Workload anchors classify their regime hypothesis from the dominant
bottleneck under strong reuse — so an architecture that flips the
bottleneck can flip the regime label, falsifying the anchor's hypothesis.

This module does NOT introduce a new simulator, episode shape, or policy.
It is a thin matrix runner over `(anchor, model_profile, capacity_cell)`,
using the existing `classify_regime`.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .k8_regime import default_bundle, RegimeCell
from .profiles import ProfileBundle
from .r3_model_sweep import R3_DEFAULT_PROFILES, make_r3_budget
from .resources import ResourceBudget
from .workloads import (
    ANCHORS,
    RegimeClassification,
    WorkloadAnchor,
    classify_regime,
)


@dataclass(frozen=True)
class AnchorR3Cell:
    state_scale: str
    prefill_capacity: str
    link_gbps: int


@dataclass(frozen=True)
class AnchorR3Result:
    anchor_name: str
    model_profile: str
    cell: AnchorR3Cell
    n_workflows: int
    classification: RegimeClassification

    @property
    def matches_hypothesis(self) -> bool:
        return self.classification.matches_hypothesis


# A small, focused set of capacity cells. Each anchor's regime hypothesis
# was originally pinned under one of these in workloads.py tests; we
# cross_reference all three so a hypothesis miss can't hide behind a
# single capacity setting.
W_R3_CELLS: tuple[AnchorR3Cell, ...] = (
    AnchorR3Cell(state_scale="medium", prefill_capacity="loose", link_gbps=1),
    AnchorR3Cell(state_scale="medium", prefill_capacity="tight", link_gbps=5),
    AnchorR3Cell(state_scale="medium", prefill_capacity="loose", link_gbps=100),
)


def run_w_r3_matrix(
    repo_root: str | Path,
    *,
    anchors: tuple[WorkloadAnchor, ...] | None = None,
    model_names: tuple[str, ...] = R3_DEFAULT_PROFILES,
    cells: tuple[AnchorR3Cell, ...] = W_R3_CELLS,
    n_workflows: int = 8,
) -> list[AnchorR3Result]:
    """Cross_product run: every (anchor, model_profile, cell) → classify."""
    if anchors is None:
        anchors = tuple(ANCHORS.values())
    repo_root = Path(repo_root)
    baseline_bundle = default_bundle(repo_root, "compact_kv")
    baseline_rate = baseline_bundle.model.single_stream_prefill_tok_s
    rows: list[AnchorR3Result] = []
    for anchor in anchors:
        for model_name in model_names:
            bundle = default_bundle(repo_root, model_name)
            for cell in cells:
                # Use the model_aware R3 budget so prefill capacity scales
                # with the architecture's per_stream prefill rate. Without
                # this, classify_regime is architecture_blind: the
                # workspace_byte path (ARTIFACT_COPY at the link) dominates
                # for anchors like W2 and the model profile contributes
                # nothing.
                budget = make_r3_budget(
                    RegimeCell(
                        n_workflows=n_workflows,
                        state_scale=cell.state_scale,
                        prefill_capacity=cell.prefill_capacity,
                        link_gbps=cell.link_gbps,
                    ),
                    bundle,
                    baseline_prefill_tok_s=baseline_rate,
                )
                classification = classify_regime(
                    anchor, bundle, budget, n_workflows=n_workflows,
                )
                rows.append(AnchorR3Result(
                    anchor_name=anchor.name,
                    model_profile=model_name,
                    cell=cell,
                    n_workflows=n_workflows,
                    classification=classification,
                ))
    return rows


def write_w_r3_artifacts(rows: list[AnchorR3Result], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "w_under_r3.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "anchor", "model_profile", "state_scale", "prefill_capacity",
            "link_gbps", "n_workflows", "hypothesis", "observed_regime",
            "matches_hypothesis", "dominant_bottleneck", "strong_p50_s",
            "mixed_p50_s", "mixed_vs_strong_gap_frac",
        ])
        writer.writeheader()
        for row in rows:
            anchor = ANCHORS[row.anchor_name]
            classification = row.classification
            writer.writerow({
                "anchor": row.anchor_name,
                "model_profile": row.model_profile,
                "state_scale": row.cell.state_scale,
                "prefill_capacity": row.cell.prefill_capacity,
                "link_gbps": row.cell.link_gbps,
                "n_workflows": row.n_workflows,
                "hypothesis": anchor.regime_hypothesis,
                "observed_regime": classification.observed_regime,
                "matches_hypothesis": classification.matches_hypothesis,
                "dominant_bottleneck": classification.dominant_bottleneck,
                "strong_p50_s": f"{classification.strong_reuse_p50_resume_s:.9g}",
                "mixed_p50_s": f"{classification.mixed_p50_resume_s:.9g}",
                "mixed_vs_strong_gap_frac": f"{classification.mixed_vs_strong_gap_frac:.9g}",
            })

    flips_by_anchor: dict[str, list[dict]] = {a: [] for a in {r.anchor_name for r in rows}}
    by_anchor_baseline: dict[str, str] = {}
    for row in rows:
        if row.model_profile == "compact_kv":
            key = (row.anchor_name, row.cell)
            by_anchor_baseline[key] = row.classification.observed_regime
    for row in rows:
        if row.model_profile == "compact_kv":
            continue
        baseline = by_anchor_baseline.get((row.anchor_name, row.cell))
        if baseline is None:
            continue
        if baseline != row.classification.observed_regime:
            flips_by_anchor[row.anchor_name].append({
                "model_profile": row.model_profile,
                "cell": {
                    "state_scale": row.cell.state_scale,
                    "prefill_capacity": row.cell.prefill_capacity,
                    "link_gbps": row.cell.link_gbps,
                },
                "baseline_regime": baseline,
                "observed_regime": row.classification.observed_regime,
                "dominant_bottleneck": row.classification.dominant_bottleneck,
            })

    summary = {
        "total_rows": len(rows),
        "anchors": sorted({r.anchor_name for r in rows}),
        "model_profiles": list(R3_DEFAULT_PROFILES),
        "cells": [c.__dict__ for c in W_R3_CELLS],
        "regime_flips_vs_compact_kv_baseline": flips_by_anchor,
    }
    (out / "w_under_r3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(
        "# W under R3 — anchor regime classification across model profiles\n\n"
        "`w_under_r3.csv` reports per-(anchor, model_profile, cell) regime\n"
        "classification. `w_under_r3_summary.json` lists the cells where the\n"
        "observed regime differs from the `compact_kv` baseline. A flip means\n"
        "model architecture is enough to move the anchor across the regime\n"
        "map — the W_anchors must then carry per_profile hypothesis labels,\n"
        "not just one.\n"
    )
