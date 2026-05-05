"""H6 — eval report renders headline + per-scheme + slice tables.

Claim:
    `render_eval_report` produces a markdown document with:
      - a "Headline metrics" table
      - one section per scheme present in `cells`
      - one slice section per slice_kind in `slices` (when non-empty)
    Infeasible cells render `n/a (insufficient data)` (not 0.0) and
    feasible cells include numeric AUROC/Brier/ECE.

Plausible wrong implementations:
    - infeasible cell rendered with raw None (-> "None" in output)
    - missing scheme section if a scheme has only one row
    - slice tables emitted even when slices=[] (template would render
      empty header rows)
"""

from __future__ import annotations

from coding_estimator.eval.harness import EvalCell
from coding_estimator.eval.slices import SliceCell
from coding_estimator.reports.render import render_eval_report


def _ok_cell(target="y_x", model="g4", scheme="loro", source_slice="tb_live") -> EvalCell:
    return EvalCell(
        target=target, model=model, scheme=scheme, source_slice=source_slice,
        feasible=True,
        n_runs_train=10, n_runs_test=2, n_checkpoints_test=40,
        positive_rate_data=0.25, predicted_positive_rate=0.30,
        auroc=0.65, brier=0.18, log_loss=0.55, ece=0.07,
        brier_ci_low=0.10, brier_ci_high=0.25,
    )


def _na_cell(target="y_x", model="g4", scheme="ltfo", source_slice="src_a") -> EvalCell:
    return EvalCell(
        target=target, model=model, scheme=scheme, source_slice=source_slice,
        feasible=False,
        n_runs_train=None, n_runs_test=None, n_checkpoints_test=None,
        positive_rate_data=None, predicted_positive_rate=None,
        auroc=None, brier=None, log_loss=None, ece=None,
        brier_ci_low=None, brier_ci_high=None,
        note="insufficient data",
    )


def test_headline_includes_loso_when_present() -> None:
    cells = [_ok_cell(scheme="loro"), _ok_cell(scheme="loso")]
    md = render_eval_report(title="t", cells=cells)
    # Headline section appears before the scheme sections, and contains
    # the loso row but not necessarily the loro row.
    assert "Headline metrics" in md
    head, _ = md.split("## Scheme: `loro`", 1)
    assert "loso" in head


def test_infeasible_cells_render_insufficient_data() -> None:
    md = render_eval_report(title="t", cells=[_na_cell(scheme="loro")])
    assert "n/a (insufficient data)" in md
    assert "None" not in md.replace("None>", "")  # nothing should render the literal None


def test_slice_section_only_when_slices_present() -> None:
    cells = [_ok_cell()]
    md_no_slices = render_eval_report(title="t", cells=cells, slices=[])
    assert "Slice-specific metrics" not in md_no_slices

    sl = [SliceCell(
        target="y_x", model="g4", scheme="loro", source_slice="tb_live",
        slice_kind="phase", slice_value="early",
        feasible=True, n_runs=2, n_checkpoints=10, positives=5, negatives=5,
        auroc=0.6, brier=0.2, ece=0.05,
    )]
    md = render_eval_report(title="t", cells=cells, slices=sl)
    assert "Slice-specific metrics" in md
    assert "Slice kind: `phase`" in md


def test_per_scheme_sections_emitted() -> None:
    cells = [_ok_cell(scheme="loro"), _ok_cell(scheme="ltfo"), _ok_cell(scheme="loso")]
    md = render_eval_report(title="t", cells=cells)
    for s in ("loro", "ltfo", "loso"):
        assert f"## Scheme: `{s}`" in md
