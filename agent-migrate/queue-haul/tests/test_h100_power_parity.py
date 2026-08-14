"""
Claim:
The H100 power parity view plots every measured cell once by workload family and
reports MAE and R² over that complete pooled set.

Plausible wrong implementations:
- Drop retrospective or confirmation cells while removing their legend labels.
- Retain cohort overlays that plot confirmation cells twice.
- Keep classifying dots by campaign cohort instead of workload family.
- Report stored holdout statistics instead of recomputing pooled metrics.
"""

import json

import matplotlib.pyplot as plt

from plot_h100_power_parity import STAGE_COUNTS, load, write


def test_h100_power_parity_has_exact_reference_and_complete_grid(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    fit = {"power_idle_w": 90, "power_amplitude_w": 100,
           "alpha_s_per_prefill_token": .001,
           "beta_s_per_decode_token": .01}
    result = {"schema": "queue-haul-rational-power-fit-v1", "fit": fit}
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
    rows = load(root, [history])
    monkeypatch.setattr(plt, "close", lambda _: None)

    write(rows, tmp_path / "parity")

    axis = plt.gcf().axes[0]
    assert len(rows) == 113
    assert {row["cohort"] for row in rows} == {"fit_campaign", "retrospective"}
    assert axis.lines[0].get_xdata().tolist() == axis.lines[0].get_ydata().tolist()
    assert axis.get_xlim() == axis.get_ylim()
    expected = 90 + 100 * .001 / 1.001
    assert abs(rows[0]["predicted_power_w"] - expected) < 1e-12
    for suffix in ("csv", "png", "pdf"):
        assert (tmp_path / f"parity.{suffix}").stat().st_size


def test_h100_power_parity_pools_metrics_and_classifies_only_by_family(
        tmp_path, monkeypatch):
    rows = [
        {"family": "prefill", "cohort": "fit_campaign", "stage": "confirmation",
         "predicted_power_w": 0., "measured_power_w": 0.},
        {"family": "prefill", "cohort": "retrospective", "stage": "discovery",
         "predicted_power_w": 1., "measured_power_w": 2.},
        {"family": "decode", "cohort": "fit_campaign", "stage": "discovery",
         "predicted_power_w": 2., "measured_power_w": 2.},
        {"family": "decode", "cohort": "retrospective", "stage": "confirmation",
         "predicted_power_w": 3., "measured_power_w": 4.},
    ]
    monkeypatch.setattr(plt, "close", lambda _: None)

    write(rows, tmp_path / "parity")

    axis = plt.gcf().axes[0]
    assert not axis.get_title()
    assert [item.get_label() for item in axis.collections] == ["Prefill", "Decode"]
    assert sum(len(item.get_offsets()) for item in axis.collections) == len(rows)
    assert axis.texts[0].get_text() == "MAE 0.50 W\n$R^2$ 0.750"
