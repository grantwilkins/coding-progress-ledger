"""
Claim:
The pooled CDF reports the fraction of designed cases whose modeled shed reaches
the common target, including the power-window delay and retaining misses as
missing CDF mass.

Plausible wrong implementations:
- Omit the power-window delay.
- Divide by successful cases instead of all designed cases.
- Accumulate sessions in identifier rather than completion order.
- Add independent marginal gains despite the concave source-power curve.
"""

import pytest
import matplotlib.pyplot as plt

from plot_pooled_attainment_cdf import attainment_curve, attainment_time, write_plot


def test_attainment_time_uses_completion_order_full_set_and_power_window():
    seen = []

    def shed(moved):
        seen.append(tuple(moved))
        return {("early",): 3, ("early", "late"): 7}[tuple(moved)]

    assert attainment_time(
        [(8, "late"), (2, "early")], shed, 6, 5,
    ) == 13
    assert seen == [("early",), ("early", "late")]


def test_attainment_curve_retains_misses_in_denominator():
    rows = [
        {"case_id": "a", "policy": "queue_haul_lp", "attainment_time_s": 10},
        {"case_id": "b", "policy": "queue_haul_lp", "attainment_time_s": ""},
        {"case_id": "c", "policy": "queue_haul_lp", "attainment_time_s": 20},
    ]
    x, y = attainment_curve(rows, "queue_haul_lp")
    assert x.tolist() == [0, 10, 20]
    assert y.tolist() == pytest.approx([0, 1 / 3, 2 / 3])


def test_plot_carries_missing_mass_to_horizon(tmp_path, monkeypatch):
    rows = [
        {"case_id": case, "policy": policy, "attainment_time_s": time,
         "horizon_s": 90, "requested_fraction": 2 / 3}
        for policy in (
            "queue_haul_lp", "queue_haul_greedy", "independent_fastest",
            "replay_only", "kv_only", "power_blind", "deadline_blind",
        ) for case, time in (("a", 10), ("b", ""))
    ]
    monkeypatch.setattr(plt, "close", lambda _: None)
    write_plot(rows, tmp_path / "cdf")
    assert all(line.get_xdata()[-1] == 90 for line in plt.gca().lines[:-1])
    assert all(line.get_ydata()[-1] == pytest.approx(.5)
               for line in plt.gca().lines[:-1])
