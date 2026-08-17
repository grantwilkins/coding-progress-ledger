"""
Claim:
Repair-attainment summaries preserve paired populations and response plots use
the requested separate-file labels.

Plausible wrong implementations:
- Drop censored interventions from the CDF denominator.
- Select applied interventions based on their attainment time.
- Aggregate action transitions without conserving pending actions.
- Keep abbreviated resources or stale action-axis and legend labels.
- Emit panels too wide or with typography too small for side-by-side use.
- Give the two exported panels different physical dimensions.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import plot_repair_attainment as plotter


def _bundle():
    cells = []
    for seed in (0, 1):
        for axis in ("bandwidth", "prefill", "joint"):
            cells.append({
                "sweep_phase": plotter.COMMON_PHASE,
                "target_fraction": .5,
                "context_seed": seed,
                "fault_axis": axis,
                "fault_at_s": 1.0,
                "detection_at_s": 2.0,
                "migration_cutoff_s": 25.0,
                "observation_horizon_s": 120.0,
                "healthy_east_load": .5,
                "move_concurrency": 4,
                "requested_shed_w": 30.0,
                "outcome": "applied",
                "diff": {"changed_sessions": 2},
                "repair_target_s": 20.0 if seed == 0 else None,
                "control_target_s": 30.0 if seed == 0 else None,
            })
    return {
        "schema": plotter.SOURCE_SCHEMA,
        "fault_axes": {axis: [] for axis in ("bandwidth", "prefill", "joint")},
        "context_seeds": [0, 1],
        "cells": cells,
    }


def test_paired_rows_require_full_cross_product():
    bundle = _bundle()
    assert len(plotter.paired_rows(bundle, .5)) == 6
    bundle["cells"].pop()
    try:
        plotter.paired_rows(bundle, .5)
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete paired sweep was accepted")


def test_cdf_retains_nonattainment_in_denominator():
    rows = plotter.paired_rows(_bundle(), .5)
    curve = plotter.attainment_curve(rows, 120.0, 25.0)
    at_25 = next(row for row in curve if row["time_s"] == 25.0)
    at_120 = next(row for row in curve if row["time_s"] == 120.0)
    assert at_25["replan_fraction"] == .5
    assert at_25["no_replan_fraction"] == 0
    assert at_120["replan_fraction"] == .5
    assert at_120["no_replan_fraction"] == .5


def test_applied_population_is_selected_without_using_attainment():
    bundle = _bundle()
    bundle["cells"][0]["outcome"] = "unchanged"
    rows = plotter.paired_rows(bundle, .5, "applied")
    assert len(rows) == 5
    assert all(row["repair_outcome"] == "applied" for row in rows)


def test_transition_summary_conserves_pending_actions():
    bundle = _bundle()
    for row in bundle["cells"]:
        row["transition_counts"] = {
            "pending": 10, "retained": 4, "method": 3,
            "destination": 2, "removed": 1,
        }
    summary = plotter.transition_summary(bundle, .5)
    assert len(summary) == 3
    assert all(row["pending_actions"] == 20 for row in summary)
    assert all(row["retained_fraction"] == .4 for row in summary)


def test_response_plots_are_separate_and_use_requested_labels(monkeypatch):
    plt.close("all")
    bundle = _bundle()
    for row in bundle["cells"]:
        row["transition_counts"] = {
            "pending": 10, "retained": 4, "method": 3,
            "destination": 2, "removed": 1,
        }
    rows = plotter.paired_rows(bundle, .5)
    curve = plotter.attainment_curve(rows, 120, 25)
    saved = []
    monkeypatch.setattr(plt.Figure, "savefig",
                        lambda figure, path, **kwargs: saved.append(Path(path)))
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plotter.plot_response(curve, plotter.transition_summary(bundle, .5),
                          25, 120, Path("repair_response"))

    figures = [plt.figure(number) for number in plt.get_fignums()[-2:]]
    assert all(tuple(figure.get_size_inches()) == plotter.PANEL_FIGSIZE
               for figure in figures)
    assert saved == [Path("repair_response.png"), Path("repair_response.pdf"),
                     Path("repair_actions.png"), Path("repair_actions.pdf")]
    assert "25 s goal" in [text.get_text() for text in
                           figures[0].axes[0].get_legend().get_texts()]
    assert [tick.get_text() for tick in figures[1].axes[0].get_yticklabels()] == [
        "Bandwidth", "Prefill", "Both"]
    assert figures[1].axes[0].get_xlabel() == "Actions (%)"
    assert "Diff. Action" in [text.get_text() for text in
                              figures[1].axes[0].get_legend().get_texts()]
    assert all(text.get_fontsize() >= plotter.PANEL_LEGEND_FONT_SIZE
               for figure in figures for text in
               figure.axes[0].get_legend().get_texts())
    plt.close("all")


def test_attainment_export_matches_action_panel_size(monkeypatch):
    rows = plotter.paired_rows(_bundle(), .5)
    curve = plotter.attainment_curve(rows, 120, 25)
    monkeypatch.setattr(plt.Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plotter.plot(curve, 25, 120, Path("attainment_cdf"))

    assert tuple(plt.gcf().get_size_inches()) == plotter.PANEL_FIGSIZE
    assert plt.gca().get_xlabel() == "Time (s)"
    assert "25 s goal" in [text.get_text() for text in
                           plt.gca().get_legend().get_texts()]
    plt.close("all")
