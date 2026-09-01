import json
import subprocess

import matplotlib.pyplot as plt
import plot_style
import pytest

import planner_scaling_campaign as scaling


def test_small_cells_are_paired_and_reach_the_target():
    greedy = scaling.measure_cell(28, "greedy")
    lp = scaling.measure_cell(28, "lp")

    assert greedy["target_w"] == lp["target_w"]
    assert greedy["source_replicas"] == lp["source_replicas"]
    assert greedy["selected_credit_w"] >= greedy["target_w"]
    assert lp["selected_credit_w"] >= lp["target_w"]
    assert greedy["milp_recovery_s"] == lp["milp_recovery_s"] == 0
    assert greedy["candidate_universe_slots"] > 0
    assert lp["materialized_candidates"] > 0
    assert "candidate_slots" not in greedy | lp


def test_plot_writes_planning_time_with_dnf_outcomes(tmp_path, monkeypatch):
    applied = []
    monkeypatch.setattr(plot_style, "apply", lambda: applied.append(True))
    rows = [
        {"solver": "greedy", "sessions": 1_000, "status": "ok",
         "selection_s": .1},
        {"solver": "greedy", "sessions": 10_000, "status": "ok",
         "selection_s": 1},
        {"solver": "lp", "sessions": 1_000, "status": "ok", "selection_s": 1},
        {"solver": "lp", "sessions": 10_000, "status": "timeout",
         "time_limit_s": 10},
    ]

    scaling.plot(rows, tmp_path / "scaling")

    assert applied == [True]
    assert (tmp_path / "scaling.pdf").is_file()
    assert (tmp_path / "scaling.png").is_file()
    plt.close("all")


def test_identity_round_trips_and_rows_ignore_stage_files(tmp_path):
    args = scaling.parser().parse_args(["run", "--out", str(tmp_path)])
    identity = scaling._identity(args)
    assert json.loads(json.dumps(identity)) == identity
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "1000-greedy-0.json").write_text(
        '{"status":"ok","solver":"greedy"}')
    (cells / "1000-greedy-0.stage.json").write_text('{"state":"solve"}')

    assert scaling._rows(tmp_path) == [
        {"status": "ok", "solver": "greedy"}]


def test_rss_monitor_tolerates_a_vanished_process(monkeypatch):
    monkeypatch.setattr(scaling.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 1, "", ""))
    assert scaling._process_rss_mib(123) is None


def test_rss_monitor_hard_fails_on_denied_process_access(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("not permitted")

    monkeypatch.setattr(scaling.subprocess, "run", denied)
    with pytest.raises(RuntimeError, match="resident-memory monitor failed"):
        scaling._process_rss_mib(123)
