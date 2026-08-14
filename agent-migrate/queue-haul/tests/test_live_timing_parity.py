"""The timing parity plot pools only unseen live replay and KV predictions."""

import csv
from itertools import product
import json

import matplotlib.pyplot as plt
from types import SimpleNamespace

from plot_live_timing_parity import (
    ACTIONS, REGIONS, load, load_history, queue_rows, write, write_queue,
)


def test_live_timing_parity_pools_four_holdout_paths(tmp_path, monkeypatch):
    source = tmp_path / "predictions.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "scenario_id", "split", "method", "destination", "context_tokens",
            "predicted_s", "initial_time_to_first_response_s"))
        writer.writeheader()
        for index, (action, region) in enumerate(product(ACTIONS, REGIONS)):
            writer.writerow({"scenario_id": str(index), "split": "holdout", "method": action,
                             "destination": region, "context_tokens": 8192,
                             "predicted_s": index + 1,
                             "initial_time_to_first_response_s": index + 1.1})
    rows = load(source)
    monkeypatch.setattr(plt, "close", lambda _: None)

    write(rows, tmp_path / "parity")

    axis = plt.gcf().axes[0]
    assert len(rows) == len(ACTIONS) * len(REGIONS)
    assert axis.lines[0].get_xdata().tolist() == axis.lines[0].get_ydata().tolist()
    assert axis.get_xlim() == axis.get_ylim()
    assert len(axis.collections) == len(ACTIONS) * len(REGIONS)
    for suffix in ("csv", "png", "pdf"):
        assert (tmp_path / f"parity.{suffix}").stat().st_size


def test_history_uses_frozen_replay_and_regional_kv_model(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "valid_context_tokens": [8192, 31488],
        "replay_tps": [[8192, 2048], [31488, 2048]],
        "kv_initial_completion_s": 1,
        "kv_destination_bytes_per_s": 100,
        "kv_effective_path_bytes_per_s": {
            "australiaeast": 50, "southcentralus": 200},
    }))
    monkeypatch.setattr("plot_live_timing_parity.collect", lambda _: [
        {"scenario_id": "mixed", "method": "replay",
         "destination": "australiaeast",
         "context_tokens": 8192, "measured_kv_bytes": 0,
         "initial_time_to_first_response_s": 5},
        {"scenario_id": "mixed", "method": "kv_transfer",
         "destination": "australiaeast",
         "context_tokens": 8192, "measured_kv_bytes": 100,
         "initial_time_to_first_response_s": 4},
        {"scenario_id": "mixed", "method": "kv_transfer",
         "destination": "southcentralus",
         "context_tokens": 8192, "measured_kv_bytes": 100,
         "initial_time_to_first_response_s": 3},
    ])
    kv = SimpleNamespace(sealed_bytes=lambda _tokens: 100)
    profile = SimpleNamespace(case=lambda: SimpleNamespace(kv_transfer=kv))
    monkeypatch.setattr("plot_live_timing_parity.ModelProfile.load", lambda _: profile)

    rows = load_history(tmp_path, model, tmp_path / "profile.json")

    assert [row["predicted_s"] for row in rows] == [4, 3, 2]
    assert {row["cohort"] for row in rows} == {"historical"}
    queues = queue_rows(rows)
    assert queues == [{"scenario_id": "mixed", "action": "mixed",
                       "predicted_s": 4, "measured_s": 5}]
    write_queue(queues, tmp_path / "queue")
    assert (tmp_path / "queue.png").stat().st_size
