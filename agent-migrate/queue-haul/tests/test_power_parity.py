"""
Claim:
The parity plot compares planned source-power reduction with measured source
power reduction in watts and uses a shared-axis y=x reference.

Plausible wrong implementations:
- Swap expected and measured axes.
- Plot measured sink power instead of measured source reduction.
- Use requested target power instead of the planned reduction.
- Include policies other than Queue-Haul and greedy.
- Fail to mark measured reductions below expected reductions as undershoots.
- Clip negative measurements or draw y=x across unequal axis limits.
"""

import csv

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

from plot_migration_results import load_power_parity, write_power_parity


def test_power_parity_uses_planned_and_measured_source_reduction(tmp_path,
                                                                  monkeypatch):
    source = tmp_path / "summary.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "policy", "target_w", "planned_source_drop_w",
            "measured_source_drop_w", "measured_sink_rise_w",
        ))
        writer.writeheader()
        writer.writerow({"policy": "lp", "target_w": 99,
                         "planned_source_drop_w": 10,
                         "measured_source_drop_w": -2,
                         "measured_sink_rise_w": 50})
        writer.writerow({"policy": "greedy", "target_w": 99,
                         "planned_source_drop_w": 20,
                         "measured_source_drop_w": 18,
                         "measured_sink_rise_w": 50})
        writer.writerow({"policy": "lp", "target_w": 99,
                         "planned_source_drop_w": 10,
                         "measured_source_drop_w": 12,
                         "measured_sink_rise_w": 50})
        writer.writerow({"policy": "random", "target_w": 99,
                         "planned_source_drop_w": 30,
                         "measured_source_drop_w": 50,
                         "measured_sink_rise_w": 50})
    monkeypatch.setattr(plt, "close", lambda _: None)

    write_power_parity(load_power_parity(source), tmp_path)

    ax = plt.gcf().axes[0]
    assert [points.get_offsets().tolist() for points in ax.collections[:3]] \
        == [[[10, 12]], [[10, -2]], [[20, 18]]]
    assert ax.collections[0].get_facecolor()[0].tolist() \
        == list(to_rgba("tab:blue", .75))
    marker = MarkerStyle("x")
    assert ax.collections[1].get_paths()[0].vertices.tolist() \
        == marker.get_path().transformed(marker.get_transform()).vertices.tolist()
    assert ax.lines[0].get_xdata().tolist() == ax.lines[0].get_ydata().tolist()
    assert ax.get_xlim() == ax.get_ylim()
