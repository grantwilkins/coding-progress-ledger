"""
Claim:
The parity plot compares source-power reduction recalculated with the stated
two-price model against measured source-power reduction on a shared-axis y=x
reference.

Plausible wrong implementations:
- Swap expected and measured axes.
- Plot measured sink power instead of measured source reduction.
- Use the old load calibration denominators in the new power equation.
- Remove the wrong sessions or sum power per session instead of aggregate load.
- Include policies other than Queue-Haul and greedy.
- Fail to mark measured reductions below expected reductions as undershoots.
- Clip negative measurements or draw y=x across unequal axis limits.
"""

import csv
import json

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

from plot_migration_results import load_power_parity, two_price_power, write_power_parity


def test_two_price_power_matches_idle_and_unit_prefill_load():
    assert two_price_power(0, 0) == 84.9875
    assert two_price_power(4290.8614, 0) == pytest.approx(222.51715494375958)


def test_power_parity_recalculates_selected_session_reduction(tmp_path):
    source = tmp_path / "summary.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "policy", "workload", "run_root", "measured_source_drop_w",
        ))
        writer.writeheader()
        writer.writerow({"policy": "lp", "workload": "test", "run_root": "run",
                         "measured_source_drop_w": 12})
    run = tmp_path / "test" / "run"
    run.mkdir(parents=True)
    sessions = [
        {"id": "moved", "ell_pre": 1, "ell_dec": 0},
        {"id": "kept", "ell_pre": 0, "ell_dec": 1},
    ]
    manifest = {
        "input_manifest": {"load_calibration": {"F_prefill_tps": 4290.8614,
                                                   "G_decode_tps": 148.4846},
                           "sessions": sessions},
        "sessions": [{"id": "moved"}],
    }
    (run / "controller_manifest.json").write_text(json.dumps(manifest))

    rows = load_power_parity(source)
    assert rows[0]["policy"] == "lp"
    assert rows[0]["expected_w"] == pytest.approx(69.06644061805889)
    assert rows[0]["measured_w"] == 12


def test_power_parity_plot_marks_measured_undershoots(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "close", lambda _: None)
    rows = [
        {"policy": "lp", "expected_w": 10, "measured_w": 12},
        {"policy": "lp", "expected_w": 10, "measured_w": -2},
        {"policy": "greedy", "expected_w": 20, "measured_w": 18},
    ]
    write_power_parity(rows, tmp_path)

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
