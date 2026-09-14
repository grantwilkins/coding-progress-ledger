import json

import numpy as np
import pytest

from plot_wan_prefill_results import pooled, replay_interval


def test_episode_ecdf_uses_last_completion_and_retains_incomplete_episodes():
    rows = []
    for times in ([10] * 8, [10] * 7 + [35], [10] * 7 + [None], [10] * 7):
        rows.append(dict(campaign="wan", policy="greedy", not_moved_count=8 - len(times),
                         decisions=json.dumps([
                             dict(session_id=str(i), action="replay", completion_s=t,
                                  error="failed" if t is None else None)
                             for i, t in enumerate(times)])))
    x, y, counts, total = pooled(rows, "wan", "greedy")
    np.testing.assert_array_equal(x, [0, 10, 35])
    np.testing.assert_array_equal(y, [0, 1 / 4, 2 / 4])
    assert total == 32 and counts == dict(replay=31, not_selected=1)
    with pytest.raises(ValueError, match="missing episodes"):
        pooled(rows, "prefill", "greedy")


def test_action_interval_resamples_episodes_and_preserves_constant_mix():
    def row(replays):
        return dict(campaign="wan", policy="greedy", decisions=json.dumps([
            dict(action="replay" if i < replays else "kv_transfer") for i in range(8)]))
    np.testing.assert_array_equal(replay_interval([row(4)] * 13, "wan", "greedy"), [.5, .5])
    lo, hi = replay_interval([row(0)] * 13 + [row(8)] * 13, "wan", "greedy")
    assert 0 <= lo < .5 < hi <= 1
    assert hi - lo > .25


def test_action_attainment_keeps_late_targets_and_partial_plan_endpoints():
    from plot_wan_prefill_tradeoff import episode_point

    pack = dict(sessions=[dict(session_id=str(i)) for i in range(8)],
                power_gains=[0] * 255 + [156])
    decisions = [dict(session_id=str(i), action="kv_transfer", completion_s=25, error=None)
                 for i in range(8)]
    row = dict(decisions=json.dumps(decisions), target_time_s="30", deadline_s="30",
               target_attained="True")
    assert episode_point(row, pack, 156, 5) == (100, 30)
    decisions[-1]["completion_s"] = 31
    row.update(decisions=json.dumps(decisions), target_time_s="", target_attained="False")
    assert episode_point(row, pack, 156, 5) == (100, 36)
    row["decisions"] = json.dumps(decisions[:4])
    assert episode_point(row, pack, 156, 5) == (100, None)
    row["target_attained"] = "True"
    with pytest.raises(ValueError, match="deadline attainment"):
        episode_point(row, pack, 156, 5)


def test_tradeoff_uses_observed_full_plan_endpoints(tmp_path, monkeypatch):
    import csv
    from pathlib import Path
    from matplotlib.axes import Axes
    from plot_wan_prefill_tradeoff import plot

    counts = []
    scatter = Axes.scatter
    def record(self, x, y, **kwargs):
        counts.append(len(x))
        return scatter(self, x, y, **kwargs)
    monkeypatch.setattr(Axes, "scatter", record)
    root = Path(__file__).resolve().parents[1] / "outputs/robustness-a100-20260907"
    plot(root, tmp_path)
    assert counts == [52] * 3 + [39] * 3
    with (tmp_path / "action_attainment.csv").open() as stream:
        points = list(csv.DictReader(stream))
    assert len({p["episode_id"] for p in points}) == 273
    assert {float(p["kv_share_percent"]) for p in points} == {0, 37.5, 50, 62.5, 100}
    assert all(float(p["wan_mbps"]) != 10000 if p["campaign"] == "wan"
               else float(p["prefill_rps"]) > 0 for p in points)

    assert {p["policy"] for p in points} == {"queue_haul", "greedy", "per_session_greedy"}
    assert all(p["attainment_time_s"] != "" for p in points)
    endpoints = [p for p in points if p["policy"] == "per_session_greedy"]
    assert len(endpoints) == 91
    assert {float(p["kv_share_percent"]) for p in endpoints} == {0, 100}

    with (tmp_path / "wan_kv_estimates.csv").open() as stream:
        estimates = list(csv.DictReader(stream))
    assert [float(e["wan_mbps"]) for e in estimates] == [2500, 3000, 3250, 3500]
    for e in estimates:
        size, overhead = float(e["total_kv_bytes"]), float(e["overhead_s"])
        assert 8 * size / 1e10 + overhead == pytest.approx(float(e["control_mean_completion_s"]))
        assert float(e["attainment_time_s"]) == pytest.approx(
            8 * size / (float(e["wan_mbps"]) * 1e6) + overhead + 5)
