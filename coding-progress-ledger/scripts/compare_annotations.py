#!/usr/bin/env python3
"""Compare two annotation spec sets (Workstream H, inter-annotator).

Given two directories of pilot annotation JSON specs (v1 and v2) and a
runs directory, this script computes per-pilot agreement metrics and
emits a markdown report plus a JSON summary. The metrics are
deliberately simple — kappa-style scores don't apply because each
annotator defines their own leaf set, not labels over a fixed item.
What we report is what an honest reader of two annotations would
look at:

- final coding-progress delta
- leaf count delta
- per-category leaf count delta
- REOPEN / BLOCKED count delta
- terminal status distribution agreement
- root_task description match (sanity)

Usage:
    python scripts/compare_annotations.py \
        --v1 annotations/swe_agent_pilot \
        --v2 annotations/swe_agent_pilot_v2 \
        --runs-dir runs/swe_agent_pilot \
        --report-md datasets/h_inter_annotator_report.md \
        --report-json datasets/h_inter_annotator_summary.json \
        [--only swe_agent_pilot_s_01 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress.queries import CODING_CATEGORIES  # noqa: E402

from scripts.annotate_pilots_from_spec import build_session  # noqa: E402


@dataclass(frozen=True)
class PilotMetrics:
    pilot_id: str
    annotator_label: str
    n_leaves: int
    coding_progress: float
    overall_progress: float
    category_counts: Dict[str, int]
    status_counts: Dict[str, int]
    n_reopens: int
    n_blocks: int
    n_splits: int
    n_invalidates: int
    root_task: str
    annotation_minutes: int


def _spec_metrics(spec: dict, label: str) -> PilotMetrics:
    s = build_session(spec)
    overall = s.score()
    coding = s.score(categories=CODING_CATEGORIES)

    cat_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for sub in s.ledger.subtasks.values():
        cat_counter[sub.category.name] += 1
        status_counter[sub.status.name] += 1

    n_reopens = sum(1 for ev in spec["events"] if ev.get("op") == "reopen")
    n_blocks = sum(1 for ev in spec["events"] if ev.get("op") == "block")
    n_splits = sum(1 for ev in spec["events"] if ev.get("op") == "split")
    n_invalidates = sum(1 for ev in spec["events"] if ev.get("op") == "invalidate")

    quality = spec.get("quality", {}) or {}
    return PilotMetrics(
        pilot_id=spec["pilot_id"],
        annotator_label=label,
        n_leaves=len(s.ledger.subtasks),
        coding_progress=coding.progress,
        overall_progress=overall.progress,
        category_counts=dict(cat_counter),
        status_counts=dict(status_counter),
        n_reopens=n_reopens,
        n_blocks=n_blocks,
        n_splits=n_splits,
        n_invalidates=n_invalidates,
        root_task=spec.get("root_task", ""),
        annotation_minutes=int(quality.get("annotation_time_minutes", 0) or 0),
    )


def _category_l1_distance(a: Dict[str, int], b: Dict[str, int]) -> int:
    """Manhattan distance between two category-count vectors over a
    union-of-keys axis. 0 means perfect agreement on category counts."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def _verdict(progress_delta: float, leaf_delta: int, cat_l1: int) -> str:
    """Coarse verdict label. 'high' if progress agrees within 0.10 and
    leaves within 1 and category vector within 1; 'moderate' if within
    0.20 / 2 / 3; 'low' otherwise. Tunable -- the exact thresholds are
    not load-bearing, the report shows the raw numbers anyway."""
    if abs(progress_delta) <= 0.10 and abs(leaf_delta) <= 1 and cat_l1 <= 1:
        return "high"
    if abs(progress_delta) <= 0.20 and abs(leaf_delta) <= 2 and cat_l1 <= 3:
        return "moderate"
    return "low"


def compare_pair(spec_v1: dict, spec_v2: dict) -> Dict[str, Any]:
    if spec_v1["pilot_id"] != spec_v2["pilot_id"]:
        raise ValueError(f"pilot_id mismatch: {spec_v1['pilot_id']!r} vs {spec_v2['pilot_id']!r}")
    m1 = _spec_metrics(spec_v1, "v1")
    m2 = _spec_metrics(spec_v2, "v2")
    progress_delta = m2.coding_progress - m1.coding_progress
    leaf_delta = m2.n_leaves - m1.n_leaves
    cat_l1 = _category_l1_distance(m1.category_counts, m2.category_counts)
    verdict = _verdict(progress_delta, leaf_delta, cat_l1)
    return {
        "pilot_id": m1.pilot_id,
        "v1": _metrics_dict(m1),
        "v2": _metrics_dict(m2),
        "progress_delta": progress_delta,
        "leaf_delta": leaf_delta,
        "category_l1_distance": cat_l1,
        "reopen_delta": m2.n_reopens - m1.n_reopens,
        "block_delta": m2.n_blocks - m1.n_blocks,
        "split_delta": m2.n_splits - m1.n_splits,
        "verdict": verdict,
    }


def _metrics_dict(m: PilotMetrics) -> Dict[str, Any]:
    return {
        "n_leaves": m.n_leaves,
        "coding_progress": m.coding_progress,
        "overall_progress": m.overall_progress,
        "category_counts": m.category_counts,
        "status_counts": m.status_counts,
        "n_reopens": m.n_reopens,
        "n_blocks": m.n_blocks,
        "n_splits": m.n_splits,
        "n_invalidates": m.n_invalidates,
        "annotation_minutes": m.annotation_minutes,
        "root_task": m.root_task,
    }


def render_report(comparisons: List[Dict[str, Any]]) -> str:
    if not comparisons:
        return "# Inter-annotator comparison\n\nNo pairs to compare.\n"
    n = len(comparisons)
    avg_progress_delta = sum(c["progress_delta"] for c in comparisons) / n
    avg_abs_progress_delta = sum(abs(c["progress_delta"]) for c in comparisons) / n
    avg_leaf_delta = sum(c["leaf_delta"] for c in comparisons) / n
    avg_cat_l1 = sum(c["category_l1_distance"] for c in comparisons) / n
    verdict_counts = Counter(c["verdict"] for c in comparisons)

    lines = []
    lines.append("# Inter-annotator comparison")
    lines.append("")
    lines.append("Comparison of two independent annotation passes over the same SWE-agent pilots. Per-pilot metrics: final coding-progress delta, leaf count delta, category-vector L1 distance, REOPEN/BLOCK count deltas. \"v1\" is the original annotator; \"v2\" is the second.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Pairs compared: **{n}**")
    lines.append(f"- Mean coding-progress delta (v2 − v1): **{avg_progress_delta:+.3f}**")
    lines.append(f"- Mean *absolute* coding-progress delta: **{avg_abs_progress_delta:.3f}**")
    lines.append(f"- Mean leaf count delta (v2 − v1): **{avg_leaf_delta:+.2f}**")
    lines.append(f"- Mean category-vector L1 distance: **{avg_cat_l1:.2f}**")
    lines.append(f"- Verdict distribution: {dict(verdict_counts)}")
    lines.append("")
    lines.append("## Per pilot")
    lines.append("")
    lines.append("| pilot | v1 progress | v2 progress | Δ | v1 leaves | v2 leaves | cat L1 | Δ reopens | Δ blocks | verdict |")
    lines.append("|-------|------------:|------------:|--:|----------:|----------:|-------:|----------:|---------:|--------:|")
    for c in comparisons:
        lines.append(
            f"| `{c['pilot_id']}` "
            f"| {c['v1']['coding_progress']:.2f} "
            f"| {c['v2']['coding_progress']:.2f} "
            f"| {c['progress_delta']:+.2f} "
            f"| {c['v1']['n_leaves']} "
            f"| {c['v2']['n_leaves']} "
            f"| {c['category_l1_distance']} "
            f"| {c['reopen_delta']:+d} "
            f"| {c['block_delta']:+d} "
            f"| {c['verdict']} |"
        )
    lines.append("")
    lines.append("## Per-pilot detail")
    lines.append("")
    for c in comparisons:
        lines.append(f"### `{c['pilot_id']}`")
        lines.append("")
        lines.append(f"- v1 root_task: \"{c['v1']['root_task']}\"")
        lines.append(f"- v2 root_task: \"{c['v2']['root_task']}\"")
        lines.append("")
        lines.append("| field | v1 | v2 |")
        lines.append("|-------|----|----|")
        lines.append(f"| n_leaves | {c['v1']['n_leaves']} | {c['v2']['n_leaves']} |")
        lines.append(f"| coding_progress | {c['v1']['coding_progress']:.3f} | {c['v2']['coding_progress']:.3f} |")
        lines.append(f"| overall_progress | {c['v1']['overall_progress']:.3f} | {c['v2']['overall_progress']:.3f} |")
        lines.append(f"| category_counts | {c['v1']['category_counts']} | {c['v2']['category_counts']} |")
        lines.append(f"| status_counts | {c['v1']['status_counts']} | {c['v2']['status_counts']} |")
        lines.append(f"| n_reopens | {c['v1']['n_reopens']} | {c['v2']['n_reopens']} |")
        lines.append(f"| n_blocks | {c['v1']['n_blocks']} | {c['v2']['n_blocks']} |")
        lines.append(f"| annotation_minutes | {c['v1']['annotation_minutes']} | {c['v2']['annotation_minutes']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two annotation spec sets.")
    parser.add_argument("--v1", required=True, type=Path, help="Directory containing v1 specs.")
    parser.add_argument("--v2", required=True, type=Path, help="Directory containing v2 specs.")
    parser.add_argument("--report-md", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Optional: limit to one or more pilot_ids.",
    )
    args = parser.parse_args(argv)

    v1_specs = sorted(args.v1.glob("*.json"))
    v2_specs = sorted(args.v2.glob("*.json"))
    v1_by_id = {p.stem: p for p in v1_specs}
    v2_by_id = {p.stem: p for p in v2_specs}

    pairs = sorted(set(v1_by_id) & set(v2_by_id))
    only = set(args.only) if args.only else None
    if only is not None:
        pairs = [p for p in pairs if p in only]

    comparisons: List[Dict[str, Any]] = []
    for pilot_id in pairs:
        spec1 = json.loads(v1_by_id[pilot_id].read_text(encoding="utf-8"))
        spec2 = json.loads(v2_by_id[pilot_id].read_text(encoding="utf-8"))
        comparisons.append(compare_pair(spec1, spec2))

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(comparisons), encoding="utf-8")
    args.report_json.write_text(
        json.dumps({"pairs": comparisons, "n_pairs": len(comparisons)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[compare_annotations] {len(comparisons)} pair(s) -> {args.report_md}",
        file=sys.stderr,
    )
    only_in_v1 = sorted(set(v1_by_id) - set(v2_by_id))
    only_in_v2 = sorted(set(v2_by_id) - set(v1_by_id))
    if only_in_v1:
        print(f"[compare_annotations] only in v1: {only_in_v1}", file=sys.stderr)
    if only_in_v2:
        print(f"[compare_annotations] only in v2: {only_in_v2}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
