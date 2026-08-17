"""The phase-plane plot preserves measured coordinates and SLO sensitivity."""

import csv
import json

import matplotlib.pyplot as plt

from plot_service_headroom_phase_plane import aggregate, parse_slo, write


def test_phase_plane_aggregates_replicates_and_changes_with_slo(tmp_path, monkeypatch):
    samples = [
        {"stage": stage, "direction": direction, "target_rho": rho, "block": block,
         "rho_p": rho_p, "rho_d": rho_d, "p90_ttft_s": ttft,
         "p90_mean_tpot_s": tpot, "stable": stable}
        for stage, block in (("discovery", 0), ("confirmation", 3))
        for direction, rho, rho_p, rho_d, ttft, tpot, stable in (
            ("baseline", .25, .02, .23, .12, .03, True),
            ("prefill_heavy", .7, .34, .36, .55, .06, True),
            ("balanced", .7, .24, .46, .57, .05, False),
        )
    ]
    rows = aggregate(samples)
    monkeypatch.setattr(plt, "close", lambda _: None)
    slos = [parse_slo("tight:.2:.04"), parse_slo("paper:1:.1")]

    write(rows, slos, tmp_path / "phase-plane")

    output = list(csv.DictReader((tmp_path / "phase-plane.csv").open()))
    prefill = [row for row in output if row["direction"] == "prefill_heavy"]
    assert [row["feasible"] for row in prefill] == ["False", "True"]
    assert rows[0]["replicates"] == 2
    assert len(plt.gcf().axes) == 2
    assert "no 2D contour" in plt.gcf().texts[-1].get_text()
    assert json.loads((tmp_path / "phase-plane.json").read_text())["schema"]
    for suffix in ("csv", "json", "png", "pdf"):
        assert (tmp_path / f"phase-plane.{suffix}").stat().st_size
