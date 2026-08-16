"""The fixed-shape plot uses canonical models, SLOs, and whiskers."""

import json

import fixed_shape_slo_campaign as campaign
import plot_fixed_shape_slo_curve as plot
import plot_style


def test_plot_writes_combined_curve_with_boundary_whiskers(tmp_path):
    roots = []
    for index, model in enumerate(plot_style.MODELS):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        curve = [{"model": model, "offered_rps": rate,
                  "replicates": 3 if rate in (1, 2) else 1,
                  "exact_completion_rate_min": 1,
                  "p90_ttft_s": rate / 2, "p90_ttft_s_min": rate / 2 - .01,
                  "p90_ttft_s_max": rate / 2 + .01,
                  "p90_mean_tpot_s": rate / 20,
                  "p90_mean_tpot_s_min": rate / 20 - .001,
                  "p90_mean_tpot_s_max": rate / 20 + .001}
                 for rate in campaign.RATES]
        (root / "summary.json").write_text(json.dumps({
            "schema": campaign.SCHEMA, "model": model, "hardware": "h100",
            "input_tokens": 3920, "output_tokens": 1024,
            "requests_per_point": 32, "rates_rps": list(campaign.RATES),
            "ttft_slo_s": 1, "tpot_slo_s": .1,
            "boundary": {"predecessor_rps": 1, "first_violating_rps": 2},
            "curve": curve,
        }))
        roots.append(root)

    rows, contract = plot.load(roots)
    plot.write(rows, contract, tmp_path / "curve")

    assert len(rows) == 3 * len(campaign.RATES)
    assert sum(row["boundary"] for row in rows) == 6
    assert contract["ttft_slo_s"] == 1
    for suffix in ("csv", "json", "png", "pdf"):
        assert (tmp_path / f"curve.{suffix}").stat().st_size
