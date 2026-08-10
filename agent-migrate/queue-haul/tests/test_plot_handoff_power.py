"""
Claim:
Handoff plots show regional per-GPU 500 ms power means from state preparation
through GPU sleep, normalized by the 300 W per-GPU TDP.

Plausible wrong implementations:
- use the wrong event as the shared time origin
- place a boundary sample in the wrong 500 ms bin
- assign a destination's samples to the wrong region
- aggregate power across regions instead of within each region and time bin
- hide the sub-second traffic cutover by rendering it as a zero-width span
- normalize by a measured baseline or 400 W instead of the 300 W TDP
- crop at traffic cutover instead of including drain and GPU sleep
- retain bespoke region colors or use lines too thin for a paper figure
"""

import csv
import json

import plot_handoff_power as p


def test_reduce_aligns_power_regions_and_queue_depth(tmp_path, monkeypatch):
    phases = {name: {"wall_ns": value, "monotonic_ns": value} for name, value in {
        "pre_start": 1_000_000_000, "pre_end": 2_000_000_000,
        "handoff_start": 2_000_000_000, "handoff_end": 2_500_000_000,
        "switch_start": 2_500_000_000, "traffic_switched": 2_500_100_000,
        "source_drained": 2_750_000_000, "sleep_start": 2_750_000_000,
        "sleep_ready": 3_000_000_000,
        "post_start": 3_000_000_000, "post_end": 4_000_000_000,
    }.items()}
    (tmp_path / "result.json").write_text(json.dumps({
        "schema": "queue-haul-three-node-handoff-v2", "phases": phases,
        "scenario": {"policy": "kv_only",
                     "background": {"east": [.5, 0], "germany": [.5, 0]}},
    }))
    powers = {"sweden": (220, 230, 120, 80),
              "east": (100, 160, 170, 180),
              "germany": (110, 150, 160, 170)}
    moments = (1_500_000_000, 2_250_000_000,
               2_750_000_000, 3_500_000_000)
    for node, values in powers.items():
        path = tmp_path / "power.csv" if node == "sweden" else \
            tmp_path / f"nodes/{node}/power.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "wall_ns", "gpu", "power_w",
                             "utilization_pct", "memory_mib", "valid"))
            for moment, watts in zip(moments, values):
                writer.writerow((0, moment, 0, watts, 0, 0, 1))
        with (tmp_path / f"metrics_{node}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("wall_ns", "vllm:num_requests_running",
                             "vllm:num_requests_waiting"))
            for index, moment in enumerate(moments):
                writer.writerow((moment, index + 1, index))

    monkeypatch.setattr(p.plt, "close", lambda _: None)
    rows = p.reduce(tmp_path)

    assert len(rows) == 15
    assert {(row["node"], row["phase"], row["mean_power_w"])
            for row in rows if row["node"] == "sweden"} == {
                ("sweden", "pre", 220),
                ("sweden", "migration", 230),
                ("sweden", "barrier", 120),
                ("sweden", "sleep", 120), ("sweden", "post", 80)}
    with (tmp_path / "queue_summary.csv").open() as handle:
        queue = list(csv.DictReader(handle))
    assert len(queue) == 15
    assert (tmp_path / "power_handoff.png").is_file()
    axis = p.plt.gcf().axes[0]
    assert axis.get_xlim() == (0, 1)
    assert axis.get_ylabel() == "Normalized Power (%)"
    source = next(line for line in axis.lines
                  if line.get_label() == "sweden-central")
    region_lines = axis.lines[:3]
    assert [line.get_color() for line in region_lines] \
        == list(p.plt.get_cmap("tab10").colors[:3])
    assert [line.get_linewidth() for line in region_lines] == [2, 2, 2]
    assert axis.get_legend().get_texts()[0].get_fontsize() == 14
    assert list(source.get_xdata()) == [.25, .75]
    assert list(source.get_ydata()) == [100 * 230 / 300, 100 * 120 / 300]
    cutover = next(line for line in axis.lines
                   if line.get_label() == "Switch")
    assert list(cutover.get_xdata()) == [.5001, .5001]
    spans = {patch.get_label(): (patch.get_x(), patch.get_width())
             for patch in axis.patches}
    assert spans == {
        "Migration": (0, .5),
        "Barrier": (.5001, .2499),
        "Sleep": (.75, .25),
    }


def test_bin_mean_uses_fixed_half_second_windows():
    assert p.bin_mean([(.49, 10), (.5, 20), (.99, 40), (1, 80)]) == (
        [.25, .75, 1.25], [10, 30, 80])
