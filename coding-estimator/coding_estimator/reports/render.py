"""H6 — render the canonical eval report from EvalCell / SliceCell rows.

The jinja template at `reports/templates/eval_report.md.j2` is the single
source of truth for layout; this module just shapes pandas/dataclass
output into the dicts the template expects.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from coding_estimator.eval.harness import EvalCell
from coding_estimator.eval.slices import MIN_PER_SLICE, SliceCell

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "reports" / "templates"
TEMPLATE_NAME = "eval_report.md.j2"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _eval_row(c: EvalCell) -> dict:
    if not c.feasible:
        return {
            "scheme": c.scheme, "source_slice": c.source_slice,
            "target": c.target, "model": c.model,
            "n_runs_train": "n/a", "n_runs_test": "n/a", "n_checkpoints_test": "n/a",
            "positive_rate_data": "n/a (insufficient data)",
            "auroc": "n/a", "brier": "n/a",
            "brier_ci": "n/a", "log_loss": "n/a", "ece": "n/a",
        }
    return {
        "scheme": c.scheme, "source_slice": c.source_slice,
        "target": c.target, "model": c.model,
        "n_runs_train": _fmt(c.n_runs_train),
        "n_runs_test": _fmt(c.n_runs_test),
        "n_checkpoints_test": _fmt(c.n_checkpoints_test),
        "positive_rate_data": _fmt(c.positive_rate_data),
        "auroc": _fmt(c.auroc),
        "brier": _fmt(c.brier),
        "brier_ci": f"[{_fmt(c.brier_ci_low)}, {_fmt(c.brier_ci_high)}]",
        "log_loss": _fmt(c.log_loss),
        "ece": _fmt(c.ece),
    }


def _slice_row(c: SliceCell) -> dict:
    return {
        "scheme": c.scheme, "source_slice": c.source_slice,
        "target": c.target, "model": c.model,
        "slice_value": c.slice_value,
        "n_runs": _fmt(c.n_runs),
        "n_checkpoints": _fmt(c.n_checkpoints),
        "positives": _fmt(c.positives),
        "negatives": _fmt(c.negatives),
        "auroc": "n/a (insufficient data)" if not c.feasible else _fmt(c.auroc),
        "brier": "n/a" if not c.feasible else _fmt(c.brier),
        "ece": "n/a" if not c.feasible else _fmt(c.ece),
    }


def render_eval_report(
    *,
    title: str,
    cells: list[EvalCell],
    slices: list[SliceCell] | None = None,
    summary: str | None = None,
    headline_filter=None,
) -> str:
    """`headline_filter(cell) -> bool` selects rows for the headline
    table. Defaults to feasible cross-source / loso cells when present,
    else the full set."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    tpl = env.get_template(TEMPLATE_NAME)

    if headline_filter is None:
        loso = [c for c in cells if c.scheme == "loso"]
        headline_cells = loso if loso else cells
    else:
        headline_cells = [c for c in cells if headline_filter(c)]
    headline_cells = sorted(
        headline_cells, key=lambda c: (c.scheme, c.source_slice, c.target, c.model)
    )

    by_scheme: dict[str, list[EvalCell]] = {}
    for c in cells:
        by_scheme.setdefault(c.scheme, []).append(c)
    scheme_groups = [
        {
            "scheme": k,
            "rows": [_eval_row(c) for c in sorted(
                v, key=lambda x: (x.source_slice, x.target, x.model)
            )],
        }
        for k, v in sorted(by_scheme.items())
    ]

    slice_groups: list[dict] = []
    if slices:
        by_kind: dict[str, list[SliceCell]] = {}
        for sc in slices:
            by_kind.setdefault(sc.slice_kind, []).append(sc)
        for kind in sorted(by_kind):
            rows = sorted(
                by_kind[kind],
                key=lambda x: (x.scheme, x.source_slice, x.target, x.model, x.slice_value),
            )
            slice_groups.append({"kind": kind, "rows": [_slice_row(c) for c in rows]})

    return tpl.render(
        title=title,
        summary=summary,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        headline_cells=[_eval_row(c) for c in headline_cells],
        scheme_groups=scheme_groups,
        slice_groups=slice_groups,
        min_per_slice=MIN_PER_SLICE,
    )


def write_eval_report(path: Path, **kwargs) -> Path:
    md = render_eval_report(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "EvalCell", "SliceCell",
    "render_eval_report", "write_eval_report",
    "asdict",
]
