import csv
import json

import power_rate_sweep as sweep


def test_reduce_emits_concave_envelope(tmp_path):
    rows = [
        {"rate_rps": 1, "window_s": 1, "prompt_tokens": 100,
         "output_tokens": 0, "start_ns": 0, "end_ns": 10,
         "power_mean_w": 20},
        {"rate_rps": 2, "window_s": 1, "prompt_tokens": 200,
         "output_tokens": 0, "start_ns": 0, "end_ns": 10,
         "power_mean_w": 50},
    ]
    (tmp_path / "levels.json").write_text(json.dumps(rows))
    for rate, watts in ((1, 20), (2, 50)):
        with (tmp_path / f"power-r{rate}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "wall_ns", "power_w",
                             "utilization_pct", "memory_mib"))
            writer.writerow((1, 1, watts, 1, 1))

    sweep.reduce(tmp_path, 100, 100, 10, 2)

    with (tmp_path / "power_curve.csv").open() as handle:
        curve = list(csv.DictReader(handle))
    assert [(float(row["ell"]), float(row["power_w"])) for row in curve] == [
        (0, 10), (2, 50)]
    assert [row["ell"] for row in json.loads((tmp_path / "reduced.json").read_text())] == [1, 2]
