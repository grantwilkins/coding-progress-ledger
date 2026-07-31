"""
Claim:
The parity plot compares planned source-power reduction with measured source
power reduction in watts and uses a shared-axis y=x reference.

Plausible wrong implementations:
- Swap expected and measured axes.
- Plot measured sink power instead of measured source reduction.
- Use requested target power instead of the planned reduction.
- Clip negative measurements or draw y=x across unequal axis limits.
"""

import csv

import matplotlib.pyplot as plt

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
    monkeypatch.setattr(plt, "close", lambda _: None)

    write_power_parity(load_power_parity(source), tmp_path)

    ax = plt.gcf().axes[0]
    assert ax.collections[0].get_offsets().tolist() == [[10, -2]]
    assert ax.lines[0].get_xdata().tolist() == ax.lines[0].get_ydata().tolist()
    assert ax.get_xlim() == ax.get_ylim()
