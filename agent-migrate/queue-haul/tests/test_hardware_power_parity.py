"""Direct source-power parity must use raw, settled trace windows."""

import csv

import matplotlib.pyplot as plt
import pytest

from plot_hardware_power_parity import (
    METHODS, normalize, predicted_shed, settled_pre_available,
    source_power_shed, write_plot,
)


def test_source_power_shed_excludes_migration_power(tmp_path):
    path = tmp_path / "power.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("monotonic_ns", "wall_ns", "gpu", "power_w",
                         "utilization_pct", "memory_mib", "valid"))
        for second in range(10):
            for gpu in range(2):
                power = 80 if gpu else 200 if second < 3 else 500 if second < 6 else 80
                writer.writerow((second * 10**9, 0, gpu, power, 0, 0, 1))
    result = {"migrations": [{"initial_start_ns": 3 * 10**9,
                               "switch_end_ns": 6 * 10**9}]}

    before, after, shed = source_power_shed(path, result)

    assert (before, after, shed) == (200, 80, 120)


def test_source_power_shed_requires_two_gpus_and_covered_windows(tmp_path):
    path = tmp_path / "power.csv"
    path.write_text(
        "monotonic_ns,wall_ns,gpu,power_w,utilization_pct,memory_mib,valid\n"
        "0,0,0,100,0,0,1\n")
    result = {"migrations": [{"initial_start_ns": 3 * 10**9,
                               "switch_end_ns": 6 * 10**9}]}
    with pytest.raises(RuntimeError, match="two measured GPUs"):
        source_power_shed(path, result)


def test_settled_pre_window_requires_window_and_guard():
    assert settled_pre_available(0, 2 * 10**9)
    assert not settled_pre_available(1, 2 * 10**9)


def test_normalization_uses_complete_cohort_maximum_request():
    rows = [{"method": method, "requested_shed_w": 20,
             "predicted_shed_w": 10, "measured_shed_w": -2}
            for method in METHODS]
    rows.append({"method": METHODS[0], "requested_shed_w": 40,
                 "predicted_shed_w": 10, "measured_shed_w": -2})

    normalized, scale = normalize(rows)

    assert scale == 40
    assert {(row["predicted_percent"], row["measured_percent"])
            for row in normalized} == {(25, -5)}


def test_prediction_counts_only_planner_admitted_moves():
    class Curve:
        @staticmethod
        def power(load):
            return 100 * load

    scenario = {"sessions": [{}] * 4, "moves": [
        {"deadline_admitted": True}, {"deadline_admitted": True},
        {"deadline_admitted": False}, {"deadline_admitted": False},
    ]}

    assert predicted_shed(scenario, Curve()) == 20


def test_plot_preserves_all_methods_repeats_and_parity(tmp_path, monkeypatch):
    rows, scale = normalize([
        {"method": method, "requested_shed_w": 10,
         "predicted_shed_w": 10, "measured_shed_w": repeat}
        for method in METHODS for repeat in (8, 12)
    ])
    monkeypatch.setattr(plt, "close", lambda _: None)

    write_plot(rows, scale, tmp_path / "parity")

    axis = plt.gcf().axes[0]
    assert axis.lines[0].get_xdata().tolist() == axis.lines[0].get_ydata().tolist()
    assert axis.get_xlim() == axis.get_ylim()
    assert len(axis.collections) == len(METHODS)
    assert all(len(collection.get_offsets()) == 2 for collection in axis.collections)
    assert {text.get_text() for text in axis.texts} >= {"Overshed", "Undershed"}
