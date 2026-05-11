from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl
from .readers import rows_to_turns
from .runner import annotate_turns


TAIL_AUDIT_FIELDS = [
    "trace_key",
    "source",
    "total_steps",
    "total_rows",
    "final_opened_unit_count",
    "first_step_where_opened_units_reach_final_count",
    "first_row_index_where_opened_units_reach_final_count",
    "steps_remaining_after_that_point",
    "rows_remaining_after_that_point",
    "fraction_of_trace_remaining_after_that_point",
    "category_of_final_unit",
    "age_of_final_unit_at_end",
    "tail_steps_after_opened_100pct_category_change_count",
    "tail_steps_after_opened_100pct_same_category_fraction",
    "exit_status",
]

CURVE_FIELDS = [
    "trace_key",
    "source",
    "row_index",
    "step",
    "point_type",
    "opened_unit_progress",
    "closed_unit_progress",
    "step_progress",
    "total",
    "done",
    "final_total",
    "final_done",
]

SNIPPET_FIELDS = [
    "trace_key",
    "source",
    "window",
    "row_index",
    "step",
    "kind",
    "current_category",
    "current_unit_age",
    "tool",
    "command",
    "observation_snippet",
]


@dataclass
class _TraceMeta:
    trace_key: str
    source: str
    raw_row_index: int | None
    final_total: int
    final_done: int
    total_steps: int
    exit_status: str


@dataclass
class _TraceState:
    meta: _TraceMeta
    total_rows: int = 0
    first_step: int | None = None
    first_row_index: int | None = None
    first_category: str = ""
    last_category: str = ""
    age_at_end: int = 0
    tail_rows: int = 0
    tail_same_category_rows: int = 0
    tail_category_changes: int = 0

    def feed(self, row: dict[str, str]) -> None:
        self.total_rows += 1
        step = int(row["step"])
        total = int(row["total"])
        category = row.get("current_category", "")
        if self.first_step is None and total >= self.meta.final_total:
            self.first_step = step
            self.first_row_index = self.total_rows
            self.first_category = category
        elif self.first_step is not None:
            self.tail_rows += 1
            self.tail_same_category_rows += int(category == self.first_category)
            self.tail_category_changes += int(category != self.last_category)
        self.last_category = category
        self.age_at_end = int(row.get("current_unit_age") or 0)

    def audit_row(self) -> dict[str, Any]:
        first_step = self.first_step or self.meta.total_steps
        first_row = self.first_row_index or self.total_rows
        steps_remaining = self.meta.total_steps - first_step
        rows_remaining = self.total_rows - first_row
        return {
            "trace_key": self.meta.trace_key,
            "source": self.meta.source,
            "total_steps": self.meta.total_steps,
            "total_rows": self.total_rows,
            "final_opened_unit_count": self.meta.final_total,
            "first_step_where_opened_units_reach_final_count": first_step,
            "first_row_index_where_opened_units_reach_final_count": first_row,
            "steps_remaining_after_that_point": steps_remaining,
            "rows_remaining_after_that_point": rows_remaining,
            "fraction_of_trace_remaining_after_that_point": steps_remaining / max(1, self.meta.total_steps),
            "category_of_final_unit": self.first_category,
            "age_of_final_unit_at_end": self.age_at_end,
            "tail_steps_after_opened_100pct_category_change_count": self.tail_category_changes,
            "tail_steps_after_opened_100pct_same_category_fraction": self.tail_same_category_rows / self.tail_rows
            if self.tail_rows
            else 0.0,
            "exit_status": self.meta.exit_status,
        }


def evaluate_progress_label_audit(
    turns_csv: Path,
    traces_csv: Path,
    raw_dir: Path,
    report_dir: Path,
    *,
    top_n: int = 50,
    category_n: int = 10,
    plot_limit: int = 12,
) -> dict[str, int]:
    report_dir.mkdir(parents=True, exist_ok=True)
    metas = _read_trace_meta(traces_csv)
    states = _audit_states(turns_csv, metas)
    rows = sorted(
        (state.audit_row() for state in states.values()),
        key=lambda row: (-float(row["fraction_of_trace_remaining_after_that_point"]), row["trace_key"]),
    )
    non_artifact = [row for row in rows if row["category_of_final_unit"] != "ARTIFACT"]
    category_samples = _category_samples(rows, category_n)
    selected_keys = _selected_trace_keys(rows, category_samples, top_n)

    audit_by_key = {row["trace_key"]: row for row in rows}
    curve_rows = _curve_rows(turns_csv, metas, selected_keys, audit_by_key)
    snippet_rows = _inspection_snippets(raw_dir, metas, selected_keys, audit_by_key)

    _write_csv(report_dir / "tail_progress_audit.csv", TAIL_AUDIT_FIELDS, rows)
    _write_csv(report_dir / "tail_progress_audit_non_artifact.csv", TAIL_AUDIT_FIELDS, non_artifact)
    _write_csv(report_dir / "classifier_final_unit_samples.csv", TAIL_AUDIT_FIELDS, category_samples)
    _write_csv(report_dir / "progress_curve_traces.csv", CURVE_FIELDS, curve_rows)
    _write_csv(report_dir / "inspection_snippets.csv", SNIPPET_FIELDS, snippet_rows)
    _plot_curves(report_dir / "progress_curve_traces.png", curve_rows, [str(row["trace_key"]) for row in rows[:plot_limit]])
    _write_readme(report_dir / "README.md", len(rows), len(non_artifact), len(selected_keys))
    return {"traces": len(rows), "non_artifact_traces": len(non_artifact), "selected_traces": len(selected_keys)}


def _read_trace_meta(path: Path) -> dict[str, _TraceMeta]:
    metas = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("parse_error"):
                continue
            final_total = int(row.get("final_total") or 0)
            total_steps = int(row.get("total_turns") or 0)
            if final_total <= 0 or total_steps <= 0:
                continue
            raw_row_index = int(row["raw_row_index"]) if row.get("raw_row_index") else _raw_index_from_trace_key(row["trace_key"])
            metas[row["trace_key"]] = _TraceMeta(
                trace_key=row["trace_key"],
                source=row["source"],
                raw_row_index=raw_row_index,
                final_total=final_total,
                final_done=int(row.get("final_done") or final_total),
                total_steps=total_steps,
                exit_status=row.get("exit_status", ""),
            )
    return metas


def _audit_states(turns_csv: Path, metas: dict[str, _TraceMeta]) -> dict[str, _TraceState]:
    states = {key: _TraceState(meta) for key, meta in metas.items()}
    with turns_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            state = states.get(row["trace_key"])
            if state:
                state.feed(row)
    return {key: state for key, state in states.items() if state.total_rows}


def _category_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category_of_final_unit"])].append(row)
    samples = []
    for category in sorted(by_category):
        samples.extend(
            sorted(
                by_category[category],
                key=lambda row: (-int(row["age_of_final_unit_at_end"]), -float(row["fraction_of_trace_remaining_after_that_point"]), row["trace_key"]),
            )[:limit]
        )
    return samples


def _selected_trace_keys(rows: list[dict[str, Any]], category_samples: list[dict[str, Any]], top_n: int) -> set[str]:
    return {str(row["trace_key"]) for row in rows[:top_n]} | {str(row["trace_key"]) for row in category_samples}


def _curve_rows(
    turns_csv: Path,
    metas: dict[str, _TraceMeta],
    selected_keys: set[str],
    audit_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    row_indexes: dict[str, int] = defaultdict(int)
    seen = set()
    with turns_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["trace_key"]
            if key not in selected_keys:
                continue
            meta = metas[key]
            row_indexes[key] += 1
            seen.add(key)
            rows.append(_curve_row(row, meta, int(audit_rows[key]["total_rows"]), row_indexes[key], "observed"))
    for key in sorted(seen):
        meta = metas[key]
        rows.append(
            {
                "trace_key": key,
                "source": meta.source,
                "row_index": row_indexes[key],
                "step": meta.total_steps,
                "point_type": "closed_terminal",
                "opened_unit_progress": 1.0,
                "closed_unit_progress": 1.0,
                "step_progress": 1.0,
                "total": meta.final_total,
                "done": meta.final_done,
                "final_total": meta.final_total,
                "final_done": meta.final_done,
            }
        )
    return rows


def _curve_row(row: dict[str, str], meta: _TraceMeta, total_rows: int, row_index: int, point_type: str) -> dict[str, Any]:
    return {
        "trace_key": row["trace_key"],
        "source": meta.source,
        "row_index": row_index,
        "step": int(row["step"]),
        "point_type": point_type,
        "opened_unit_progress": int(row["total"]) / max(1, meta.final_total),
        "closed_unit_progress": int(row["done"]) / max(1, meta.final_done),
        "step_progress": row_index / max(1, total_rows),
        "total": int(row["total"]),
        "done": int(row["done"]),
        "final_total": meta.final_total,
        "final_done": meta.final_done,
    }


def _inspection_snippets(
    raw_dir: Path,
    metas: dict[str, _TraceMeta],
    selected_keys: set[str],
    audit_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_source: dict[str, dict[int, str]] = defaultdict(dict)
    for key in selected_keys:
        meta = metas[key]
        if meta.raw_row_index is not None:
            selected_by_source[meta.source][meta.raw_row_index] = key

    snippets = []
    for source, index_to_key in sorted(selected_by_source.items()):
        raw_path = raw_dir / source / "train.jsonl"
        found = 0
        for raw_index, raw in enumerate(read_jsonl(raw_path)):
            key = index_to_key.get(raw_index)
            if key is None:
                continue
            snippets.extend(_trace_snippets(key, source, raw, audit_rows[key]))
            found += 1
            if found == len(index_to_key):
                break
    return snippets


def _trace_snippets(trace_key: str, source: str, raw: dict[str, Any], audit_row: dict[str, Any]) -> list[dict[str, Any]]:
    [(_, turns)] = list(rows_to_turns([raw], source=source))
    rows, _ = annotate_turns(turns)
    first = int(audit_row["first_row_index_where_opened_units_reach_final_count"])
    windows = {
        "opened_100pct": range(max(1, first - 2), min(len(rows), first + 5) + 1),
        "tail_end": range(max(1, len(rows) - 7), len(rows) + 1),
    }
    snippets = []
    for window, indexes in windows.items():
        for row_index in indexes:
            turn = turns[row_index - 1]
            row = rows[row_index - 1]
            snippets.append(
                {
                    "trace_key": trace_key,
                    "source": source,
                    "window": window,
                    "row_index": row_index,
                    "step": turn.step,
                    "kind": turn.kind,
                    "current_category": row.current_category,
                    "current_unit_age": row.current_unit_age,
                    "tool": turn.tool or "",
                    "command": _shorten(turn.command or ""),
                    "observation_snippet": _shorten(turn.response or ""),
                }
            )
    return snippets


def _plot_curves(path: Path, rows: list[dict[str, Any]], selected: list[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trace[str(row["trace_key"])].append(row)
    selected = [key for key in selected if key in by_trace]
    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 2.2 * len(selected)), squeeze=False)
    for axis, key in zip(axes.flat, selected):
        trace_rows = sorted(by_trace[key], key=lambda row: (int(row["row_index"]), row["point_type"]))
        xs = [int(row["row_index"]) for row in trace_rows]
        axis.plot(xs, [float(row["opened_unit_progress"]) for row in trace_rows], label="opened", linewidth=1)
        axis.plot(xs, [float(row["closed_unit_progress"]) for row in trace_rows], label="closed", linewidth=1)
        axis.plot(xs, [float(row["step_progress"]) for row in trace_rows], label="step", linewidth=1, linestyle="--")
        axis.set_title(key, fontsize=8)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("progress")
    axes.flat[0].legend(fontsize=7, ncol=3)
    axes.flat[-1].set_xlabel("row index")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_readme(path: Path, trace_count: int, non_artifact_count: int, selected_count: int) -> None:
    path.write_text(
        "\n".join(
            [
                "# Progress Label Audit",
                "",
                "Primary question: Does opened-unit progress reach 100% while a large fraction of the trace still remains?",
                "",
                "This is a read-only label audit. It does not change the estimator, filter, tracker model, or classification rules.",
                "",
                "## Artifacts",
                "",
                "- `tail_progress_audit.csv`: all valid traces ranked by remaining fraction after opened-unit progress first reaches 100%.",
                "- `tail_progress_audit_non_artifact.csv`: the same ranking excluding final `ARTIFACT` units.",
                "- `progress_curve_traces.csv` and `progress_curve_traces.png`: opened-unit, closed-unit, and step progress for selected worst traces.",
                "- `inspection_snippets.csv`: compact command/tool and observation evidence around the opened-100% point and trace tail.",
                "- `classifier_final_unit_samples.csv`: longest final units by final category.",
                "",
                "## Decision Rubric",
                "",
                "If opened-unit progress is bad but closed-unit progress looks sane, switch the belief target from opened units to closed units.",
                "",
                "If both opened and closed units are bad, keep units as features and change the prediction target to remaining steps/actions/tool calls.",
                "",
                "If category mistakes explain the long tails, fix segmentation/classification before rerunning the tracker.",
                "",
                "## Run Summary",
                "",
                f"- audited traces: {trace_count}",
                f"- non-artifact traces: {non_artifact_count}",
                f"- selected traces with curve/snippet evidence: {selected_count}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _raw_index_from_trace_key(trace_key: str) -> int | None:
    parts = trace_key.split(":", 2)
    return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None


def _shorten(text: str, limit: int = 240) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."
