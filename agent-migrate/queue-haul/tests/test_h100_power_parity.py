"""The H100 power parity view uses the frozen fit and all measured cells."""

import json

import matplotlib.pyplot as plt

from plot_h100_power_parity import STAGE_COUNTS, load, write


def test_h100_power_parity_has_exact_reference_and_complete_grid(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    fit = {"power_idle_w": 90, "power_amplitude_w": 100,
           "alpha_s_per_prefill_token": .001,
           "beta_s_per_decode_token": .01}
    result = {"schema": "queue-haul-rational-power-fit-v1", "fit": fit,
              "validation": {"holdout_mae_w": 2.5, "holdout_r2": .94}}
    (root / "fit.json").write_text(json.dumps(result))
    (root / "metadata.json").write_text(json.dumps({
        "gpu": {"name": "NVIDIA H100 NVL", "uuid": "GPU-new",
                "power_limit_w": 400}}))
    cells = []
    sequence = 0
    for stage, count in STAGE_COUNTS.items():
        for index in range(count):
            family = "idle" if stage == "idle" else (
                "prefill", "decode", "agentic", "campaign")[index % 4]
            cells.append({"sequence": sequence, "stage": stage, "family": family,
                          "prompt_tokens": 1024, "output_tokens": 16,
                          "concurrency": 1, "realized_prefill_tps": index + 1,
                          "realized_decode_tps": index / 10,
                          "power_mean_w": 100 + index / 10,
                          "cached_prompt_tokens": 0, "completed_requests": 1,
                          "power_samples": 100, "window_s": 12})
            sequence += 1
    (root / "cells.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in cells))
    history = tmp_path / "history"
    history.mkdir()
    (history / "metadata.json").write_text(json.dumps({
        "gpu": {"name": "NVIDIA H100 NVL", "uuid": "GPU-old",
                "power_limit_w": 400}}))
    prior = [{**cells[index], "sequence": index} for index in range(2)]
    (history / "cells.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prior))
    rows, loaded = load(root, [history])
    monkeypatch.setattr(plt, "close", lambda _: None)

    write(rows, loaded, tmp_path / "parity")

    axis = plt.gcf().axes[0]
    assert len(rows) == 113
    assert {row["cohort"] for row in rows} == {"fit_campaign", "retrospective"}
    assert axis.lines[0].get_xdata().tolist() == axis.lines[0].get_ydata().tolist()
    assert axis.get_xlim() == axis.get_ylim()
    expected = 90 + 100 * .001 / 1.001
    assert abs(rows[0]["predicted_power_w"] - expected) < 1e-12
    for suffix in ("csv", "png", "pdf"):
        assert (tmp_path / f"parity.{suffix}").stat().st_size
