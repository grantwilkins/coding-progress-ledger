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
