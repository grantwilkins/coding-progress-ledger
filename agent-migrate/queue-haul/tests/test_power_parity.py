"""
Claim:
The parity plot compares planned source-power reduction with measured source
power reduction in watts and uses a shared-axis y=x reference. The disruption
CDF divides summed measured session downtime by measured watts saved.

Plausible wrong implementations:
- Swap expected and measured axes.
- Plot measured sink power instead of measured source reduction.
- Use requested target power instead of the planned reduction.
- Include policies other than Queue-Haul and greedy.
- Fail to mark measured reductions below expected reductions as undershoots.
- Clip negative measurements or draw y=x across unequal axis limits.
- Divide by planned watts or average rather than sum session downtime.
- Treat zero or negative measured savings as a valid ratio.
"""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

from plot_migration_results import load_disruption, load_power_parity, write_power_parity


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


def test_disruption_sums_session_downtime_over_positive_measured_savings(tmp_path):
    source = tmp_path / "scenario_summary.csv"
    fields = ("policy", "workload", "run_root", "planned_source_drop_w",
              "measured_source_drop_w")
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"policy": "lp", "workload": "w", "run_root": "root/run",
                         "planned_source_drop_w": 100, "measured_source_drop_w": 5})
        writer.writerow({"policy": "random", "workload": "w", "run_root": "root/bad",
                         "planned_source_drop_w": 100, "measured_source_drop_w": 0})
    run = tmp_path / "w" / "run"
    run.mkdir(parents=True)
    (run / "controller_manifest.json").write_text(json.dumps(
        {"sessions": [{"downtime_s": 2}, {"downtime_s": 3}]}))

    assert load_disruption(source) == [{"policy": "lp", "session_s_per_w": 1}]
