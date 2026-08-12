"""
Claim:
Each point compares predicted and measured source-power shed using the maximum
requested shed across the complete hardware cohort as one shared denominator.

Plausible wrong implementations:
- Normalize each method or row independently.
- Recompute the maximum after filtering methods.
- Use requested rather than predicted shed on the x-axis.
- Plot sink-power rise rather than measured source-power shed.
- Include methods other than Queue-Haul LP and Greedy.
"""

import csv

import matplotlib.pyplot as plt
import pytest

from plot_hardware_power_parity import METHODS, load_points, write_plot


def test_power_validation_uses_one_maximum_request_denominator(tmp_path,
                                                                monkeypatch):
    source = tmp_path / "summary.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "workload", "policy", "seed", "target_w",
            "planned_source_drop_w", "measured_source_drop_w",
        ))
        writer.writeheader()
        for index, method in enumerate((*METHODS, "random")):
            writer.writerow({
                "workload": "w", "policy": method, "seed": index,
                "target_w": 40 if method == "random" else 20,
                "planned_source_drop_w": 10, "measured_source_drop_w": -2,
            })

    rows, scale = load_points(source)

    assert scale == 40
    assert {row["method"] for row in rows} == set(METHODS)
    assert {(row["predicted_percent"], row["measured_percent"])
            for row in rows} == {(25, -5)}
    monkeypatch.setattr(plt, "close", lambda _: None)
    write_plot(rows, scale, tmp_path / "parity")
    axis = plt.gcf().axes[0]
    assert axis.lines[0].get_xdata().tolist() == axis.lines[0].get_ydata().tolist()
    assert axis.get_xlim() == axis.get_ylim()
    assert len(axis.collections) == len(METHODS)


def test_power_validation_requires_both_displayed_methods(tmp_path):
    source = tmp_path / "summary.csv"
    source.write_text(
        "workload,policy,seed,target_w,planned_source_drop_w,measured_source_drop_w\n"
        "w,lp,0,10,10,10\n")
    with pytest.raises(ValueError, match="both methods"):
        load_points(source)
