"""
Claim:
Migration timing parity compares each pre-run modeled episode makespan with the
same condition, repeat, and policy's measured hardware makespan.

Plausible wrong implementations:
- Join only by policy and mix conditions or repeats.
- Compare modeled makespan with destination TTFT or target-attainment time.
- Drop unmatched episodes silently.
- Report signed bias as absolute error.
"""

import csv

import pytest

from plot_migration_timing_parity import metrics, paired_points


def _write(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_pairing_uses_condition_repeat_and_policy(tmp_path):
    predictions, measurements = tmp_path / "pred.csv", tmp_path / "real.csv"
    _write(predictions,
           ("condition_index", "repeat", "policy", "predicted_makespan_s"),
           [{"condition_index": condition, "repeat": 0, "policy": policy,
             "predicted_makespan_s": value}
            for condition, policy, value in
            ((0, "queue_haul", 10), (1, "queue_haul", 20),
             (0, "greedy", 30), (1, "greedy", 40))])
    _write(measurements,
           ("condition_index", "repeat", "policy", "migration_s"),
           [{"condition_index": condition, "repeat": 0, "policy": policy,
             "migration_s": value}
            for condition, policy, value in
            ((1, "greedy", 41), (0, "queue_haul", 11),
             (1, "queue_haul", 21), (0, "greedy", 31))])
    rows = paired_points(predictions, measurements)
    assert [(row["predicted_makespan_s"], row["measured_makespan_s"])
            for row in rows] == [(40, 41), (10, 11), (20, 21), (30, 31)]


def test_metrics_distinguish_bias_mae_and_rmse():
    rows = [{"policy": policy, "predicted_makespan_s": 10,
             "measured_makespan_s": measured}
            for policy in ("queue_haul", "greedy") for measured in (8, 14)]
    row = metrics(rows)[0]
    assert row["bias_s"] == 1
    assert row["mae_s"] == 3
    assert row["rmse_s"] == pytest.approx(10 ** .5)
