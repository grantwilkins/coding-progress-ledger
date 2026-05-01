#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress.serialization import from_jsonl, load_events_jsonl


DEFAULT_LIVE_ROOT = Path("runs/swe_agent_live")
EVENT_MATRIX = "EVENT_OBSERVABILITY_MATRIX.md"
PARITY_REPORT = "PARITY_REPORT.md"
FRONTIER_POLICY = (
    "raw-step live instrumentation does not invent discovered-but-unattempted validation obligations; "
    "submit-without-validation is represented as complete_visible_frontier+no_validation_frontier unless "
    "the agent emits explicit ledger_ops for validation work"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pairs = load_pairs(args.live_root)
    comparisons = [compare_pair(live, retro) for live, retro in pairs]
    args.live_root.mkdir(parents=True, exist_ok=True)
    (args.live_root / EVENT_MATRIX).write_text(render_event_matrix(comparisons), encoding="utf-8")
    (args.live_root / PARITY_REPORT).write_text(render_report(comparisons), encoding="utf-8")
    print(f"wrote {args.live_root / PARITY_REPORT}")
    return 0


def load_pairs(live_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for metadata_path in sorted(live_root.glob("*/live_instrumentation.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        retro = Path(metadata["source_run_dir"])
        if not retro.exists():
            raise FileNotFoundError(f"missing retrospective source for {metadata_path}: {retro}")
        pairs.append((metadata_path.parent, retro))
    if not pairs:
        raise FileNotFoundError(f"no live_instrumentation.json files under {live_root}")
    return pairs


def compare_pair(live_dir: Path, retro_dir: Path) -> dict[str, Any]:
    live_summary = _summary(live_dir)
    retro_summary = _summary(retro_dir)
    live_ledger = from_jsonl(str(live_dir / "ledger.jsonl"))
    retro_ledger = from_jsonl(str(retro_dir / "ledger.jsonl"))
    live_events = load_events_jsonl(str(live_dir / "ledger.jsonl"))
    retro_events = load_events_jsonl(str(retro_dir / "ledger.jsonl"))
    metadata = json.loads((live_dir / "live_instrumentation.json").read_text(encoding="utf-8"))
    return {
        "instance_id": metadata["instance_id"],
        "final_success": metadata["final_success"],
        "live_dir": live_dir,
        "retro_dir": retro_dir,
        "live_events": len(live_events),
        "retro_events": len(retro_events),
        "wire_events": metadata["wire_event_count"],
        "live_event_types": _event_type_counts(live_events),
        "retro_event_types": _event_type_counts(retro_events),
        "live_statuses": _status_counts(live_events),
        "retro_statuses": _status_counts(retro_events),
        "live_categories": _category_counts(live_ledger),
        "retro_categories": _category_counts(retro_ledger),
        "live_timestamps": sum(1 for event in live_events if event.timestamp),
        "retro_timestamps": sum(1 for event in retro_events if event.timestamp),
        "live_coding_progress": live_summary["final_coding_progress"],
        "retro_coding_progress": retro_summary["final_coding_progress"],
        "live_shape": shape_class(live_summary),
        "retro_shape": shape_class(retro_summary),
    }


def shape_class(summary: dict[str, Any]) -> str:
    progress = summary["final_coding_progress"]
    validation = summary.get("category_active_weight_final", {}).get("validation", 0)
    validation_done = summary.get("category_completed_weight_final", {}).get("validation", 0)
    if validation == 0 and validation_done == 0:
        validation_shape = "no_validation_frontier"
    elif validation_done < validation:
        validation_shape = "validation_gap"
    else:
        validation_shape = "validation_complete"
    if progress >= 0.95:
        progress_shape = "complete_visible_frontier"
    elif progress >= 0.5:
        progress_shape = "partial_visible_frontier"
    else:
        progress_shape = "low_visible_frontier"
    return f"{progress_shape}+{validation_shape}"


def policy_adjusted_parity(item: dict[str, Any]) -> bool:
    if item["live_shape"] == item["retro_shape"]:
        return abs(item["live_coding_progress"] - item["retro_coding_progress"]) <= 0.05
    return (
        item["retro_shape"] == "partial_visible_frontier+validation_gap"
        and item["live_shape"] == "complete_visible_frontier+no_validation_frontier"
        and item["live_categories"].get("validation", 0) == 0
        and item["retro_categories"].get("validation", 0) > 0
    )


def render_event_matrix(comparisons: list[dict[str, Any]]) -> str:
    rows = [
        ("INIT", "mechanical", "Sidecar creates one root event from the first wire timestamp."),
        ("ADD_SUBTASK investigation/product/artifact from emitted tool action", "mechanical", "N3 live runs produce these from `tool_name`/`command`."),
        ("ADD_SUBTASK validation from emitted validation command", "mechanical", "Only when the agent actually runs pytest/tox/python repro."),
        ("ADD_SUBTASK validation obligation without emitted validation command", "annotation_only", "Requires semantic judgment that validation was discovered but not attempted."),
        ("UPDATE_STATUS complete from tool observation", "mechanical", "N3 marks observed tool actions complete when a following tool observation exists."),
        ("UPDATE_STATUS start from command without observation", "mechanical", "Sidecar can emit in-progress work for issued commands with no observation."),
        ("UPDATE_STATUS blocked", "annotation_only", "Needs a semantic stuck/block judgment; not present in the N3 live adapter."),
        ("REOPEN_SUBTASK", "annotation_only", "Needs evidence that prior completion was invalidated by later work."),
        ("INVALIDATE_SUBTASK", "annotation_only", "Needs semantic replacement/deletion judgment."),
        ("SPLIT_SUBTASK", "weakly_inferable", "Explicit `ledger_ops` can produce it; raw step adapter cannot reliably infer grouping."),
    ]
    lines = ["# Event observability matrix", ""]
    lines.append("| Event / transition | Level | N4 note |")
    lines.append("|---|---|---|")
    for event, level, note in rows:
        lines.append(f"| {event} | {level} | {note} |")
    lines.extend(["", "## Event types seen in N3 pairs", ""])
    lines.append("| Instance | Retrospective event types | Live event types |")
    lines.append("|---|---|---|")
    for item in comparisons:
        lines.append(
            f"| `{item['instance_id']}` | `{_counts(item['retro_event_types'])}` | `{_counts(item['live_event_types'])}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(comparisons: list[dict[str, Any]]) -> str:
    lines = ["# N4 — Live-vs-retrospective parity report", ""]
    lines.append("This compares the two N3 live sidecar ledgers against the retrospective SWE-agent pilot ledgers for the same instances. Shape classes are primary; scalar progress is secondary.")
    lines.extend(["", "## Frontier Policy", ""])
    lines.append(FRONTIER_POLICY + ".")
    lines.extend(["", "## Verdict", ""])
    if all(policy_adjusted_parity(item) for item in comparisons):
        verdict = "N4 policy-adjusted parity gate passes; N5 may proceed under the no-validation-frontier policy."
    else:
        verdict = "N4 parity gate does not pass yet; do not proceed to N5 live N=20."
    lines.append(verdict)
    lines.append("")
    lines.append("| Instance | Success | Retrospective shape | Live shape | Retrospective coding | Live coding | Delta | Scalar within 0.05 | Policy parity |")
    lines.append("|---|---:|---|---|---:|---:|---:|---|---|")
    for item in comparisons:
        delta = item["live_coding_progress"] - item["retro_coding_progress"]
        lines.append(
            f"| `{item['instance_id']}` | {item['final_success']} | `{item['retro_shape']}` | `{item['live_shape']}` | "
            f"{item['retro_coding_progress']:.3f} | {item['live_coding_progress']:.3f} | {delta:+.3f} | "
            f"{'yes' if abs(delta) <= 0.05 else 'no'} | {'yes' if policy_adjusted_parity(item) else 'no'} |"
        )
    lines.extend(["", "## Schema And Shape Parity", ""])
    lines.append("| Instance | Retrospective events | Live events | Retrospective categories | Live categories | Retrospective statuses | Live statuses |")
    lines.append("|---|---:|---:|---|---|---|---|")
    for item in comparisons:
        lines.append(
            f"| `{item['instance_id']}` | {item['retro_events']} | {item['live_events']} | `{_counts(item['retro_categories'])}` | "
            f"`{_counts(item['live_categories'])}` | `{_counts(item['retro_statuses'])}` | `{_counts(item['live_statuses'])}` |"
        )
    lines.extend(["", "## Divergences", ""])
    lines.extend(_divergence_lines(comparisons))
    lines.extend(["", "## K2 Evidence-Gap Check", ""])
    lines.append("| K2 gap | N4 result | Classification |")
    lines.append("|---|---|---|")
    lines.append("| Structured edit/submit action evidence | Closed for emitted live actions: `wire_events.jsonl` carries `tool_name`, `command`, observation, and terminal `exit_status`. | closed for N3 emitted actions |")
    lines.append("| Agent-vs-harness submit provenance | Partially closed: N3 records terminal `exit_status` on the final emitted assistant action, but the selected pair contains explicit submit-style traces rather than the six harness-forced ambiguous pilots. | partial |")
    lines.append("| Baseline failing test output before edits | Not closed: N3 replays normalized traces and does not run pre-fix tests. | open |")
    lines.append("| Full command stdout/stderr beyond source truncation | Not closed: N3 uses the same normalized observations available retrospectively. | open |")
    lines.append("| Per-edit before/after file state | Not closed: N3 records commands and observations, not file snapshots around every edit. | open |")
    lines.append("| Hidden-work/repro validity gap | Not closed by this adapter: the live trace preserves visible commands but does not decide whether a repro exercised the issue. | open |")
    lines.extend(["", "## Observability Matrix", ""])
    lines.append(f"See `{EVENT_MATRIX}`. Summary: mechanical events are available for emitted tool actions; validation obligations, blocked states, reopens, invalidations, and semantic splits remain annotation-only or weakly inferable unless the agent emits explicit `ledger_ops`. The accepted live policy therefore does not add validation obligations in raw-step mode.")
    lines.extend(["", "## Timestamp Realism", ""])
    lines.append("Every N3 live ledger event has a non-null timestamp, while the retrospective pilot ledgers have none. The intervals are replay-time timestamps from normalized traces, not real SWE-agent wall-clock durations; they are sufficient to exercise timestamp plumbing but not to calibrate deadline models.")
    lines.append("")
    return "\n".join(lines)


def _divergence_lines(comparisons: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Divergence | Instances | Assignment | Consequence |",
        "|---|---|---|---|",
        "| Live sidecar emits one leaf per visible assistant command; retrospective ledgers collapse many commands into semantic work leaves. | both | true semantic ambiguity for raw-step adapter | Event counts differ even when final shape matches. Explicit `ledger_ops` or a smarter adapter is needed for semantic grouping. |",
        "| Retrospective `WIPACrepo__iceprod-339` includes an unstarted validation leaf; live sidecar has no validation leaf because the agent emitted no validation command. | `WIPACrepo__iceprod-339` | accepted frontier-policy divergence | Live coding progress is 1.000 while retrospective coding progress is 0.667; estimator features must use `no_validation_frontier` / `submit_without_validation`, not scalar parity, for this case. |",
        "| Live ledgers have timestamps; retrospective ledgers do not. | both | expected instrumentation difference | Timestamp-aware features can run on N3 live ledgers but cannot be compared to retrospective wall-clock intervals. |",
    ]
    return lines


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build N4 live-vs-retrospective parity report.")
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    return parser.parse_args(argv)


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary_by_category.json").read_text(encoding="utf-8"))


def _event_type_counts(events) -> dict[str, int]:
    return dict(Counter(event.event_type.value for event in events))


def _status_counts(events) -> dict[str, int]:
    return dict(Counter(str(event.payload.get("status")) for event in events if event.event_type.value == "update_status"))


def _category_counts(ledger) -> dict[str, int]:
    return dict(Counter(subtask.category.value for subtask in ledger.subtasks.values()))


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)) or "none"


if __name__ == "__main__":
    raise SystemExit(main())
