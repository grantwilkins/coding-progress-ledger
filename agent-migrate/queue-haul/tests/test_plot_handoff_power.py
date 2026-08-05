import csv
import json

import plot_handoff_power as p


def test_reduce_aligns_power_regions_and_queue_depth(tmp_path):
    phases = {name: {"wall_ns": value} for name, value in {
        "pre_start": 1_000_000_000, "pre_end": 2_000_000_000,
        "handoff_start": 2_000_000_000, "handoff_end": 2_500_000_000,
        "switch_start": 2_500_000_000, "traffic_switched": 2_500_100_000,
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
    with (tmp_path / "proxy_bytes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("monotonic_ns", "wall_ns", "interval_ns", "connection_id",
                         "route", "direction", "bytes", "billed"))
        writer.writerow((0, 1_500_000_000, 250_000_000, "a", "kv/east",
                         "client_to_target", 1000, 1))
        writer.writerow((0, 2_500_000_000, 250_000_000, "a", "kv/east",
                         "target_to_client", 2000, 1))
        writer.writerow((0, 2_500_000_000, 250_000_000, "b", "kv/germany",
                         "target_to_client", 3000, 1))

    rows = p.reduce(tmp_path)

    assert len(rows) == 12
    assert {(row["node"], row["phase"], row["mean_power_w"])
            for row in rows if row["node"] == "sweden"} == {
                ("sweden", "pre", 220), ("sweden", "migration", 230),
                ("sweden", "source_fall", 120), ("sweden", "post", 80)}
    with (tmp_path / "queue_summary.csv").open() as handle:
        queue = list(csv.DictReader(handle))
    assert len(queue) == 12
    assert (tmp_path / "power_handoff.png").is_file()
