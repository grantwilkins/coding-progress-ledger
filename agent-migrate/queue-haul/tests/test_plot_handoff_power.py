import csv
import json

import plot_handoff_power as p


def test_reduce_aligns_three_nodes_to_phase_wall_clock(tmp_path):
    phases = {name: {"wall_ns": value} for name, value in {
        "pre_start": 1_000_000_000, "pre_end": 2_000_000_000,
        "handoff_start": 2_000_000_000, "sleep_ready": 3_000_000_000,
        "post_start": 3_000_000_000, "post_end": 4_000_000_000,
    }.items()}
    (tmp_path / "result.json").write_text(json.dumps({"phases": phases}))
    for node, values in {"sweden": (200, 50), "east": (100, 150),
                         "west": (110, 160)}.items():
        path = tmp_path / "power.csv" if node == "sweden" else \
            tmp_path / f"nodes/{node}/power.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "wall_ns", "gpu", "power_w",
                             "utilization_pct", "memory_mib", "valid"))
            writer.writerow((0, 1_500_000_000, 0, values[0], 0, 0, 1))
            writer.writerow((0, 3_500_000_000, 0, values[1], 0, 0, 1))

    rows = p.reduce(tmp_path)

    assert {(row["node"], row["phase"], row["mean_power_w"])
            for row in rows} == {
                ("sweden", "pre", 200), ("sweden", "post", 50),
                ("east", "pre", 100), ("east", "post", 150),
                ("west", "pre", 110), ("west", "post", 160)}
    assert (tmp_path / "power_handoff.png").is_file()
