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
            "boundary": ({"predecessor_rps": 1, "first_violating_rps": 2}
                         if index else None),
            "whisker_rates_rps": [1, 2],
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


def test_plot_accepts_upper_curve_without_slo_or_whiskers(tmp_path):
    roots = []
    for index, model in enumerate(plot_style.MODELS):
        root = tmp_path / f"upper-{index}"
        root.mkdir()
        (root / "summary.json").write_text(json.dumps({
            "schema": campaign.SCHEMA, "model": model, "hardware": "h100",
            "input_tokens": 3920, "output_tokens": 1024,
            "requests_per_point": 32, "rates_rps": list(campaign.RATES),
            "ttft_slo_s": None, "tpot_slo_s": None, "boundary": None,
            "whisker_rates_rps": [], "curve": [{
                "model": model, "offered_rps": rate, "replicates": 1,
                "exact_completion_rate_min": 1, "p90_ttft_s": rate,
                "p90_ttft_s_min": rate, "p90_ttft_s_max": rate,
                "p90_mean_tpot_s": rate / 10,
                "p90_mean_tpot_s_min": rate / 10,
                "p90_mean_tpot_s_max": rate / 10} for rate in campaign.RATES]}))
        roots.append(root)

    rows, contract = plot.load(roots)
    plot.write(rows, contract, tmp_path / "upper")

    assert not any(row["boundary"] for row in rows)
    assert contract["ttft_slo_s"] is contract["tpot_slo_s"] is None


def test_dense_knees_merge_with_base_contract(tmp_path):
    roots = []
    for index, model in enumerate(plot_style.MODELS):
        root = tmp_path / f"knee-{index}"
        root.mkdir()
        (root / "summary.json").write_text(json.dumps({
            "schema": campaign.SCHEMA, "model": model, "hardware": "h100",
            "input_tokens": 3920, "output_tokens": 1024,
            "requests_per_point": 32, "curve": [{"model": model,
                "offered_rps": campaign.KNEE_RATES[model][0]}]}))
        roots.append(root)

    rows = plot.load_knees(roots, {"hardware": "h100", "input_tokens": 3920,
                                   "output_tokens": 1024,
                                   "requests_per_point": 32})

    assert len(rows) == 3 and not any(row["boundary"] for row in rows)


def test_dense_plot_keeps_base_rate_ticks(tmp_path):
    contract = {"hardware": "h100", "input_tokens": 3920,
                "output_tokens": 1024, "requests_per_point": 32,
                "rates_rps": list(campaign.RATES), "ttft_slo_s": None,
                "tpot_slo_s": None}
    rows = [{"model": model, "offered_rps": rate, "boundary": False,
             "p90_ttft_s": 1, "p90_ttft_s_min": 1, "p90_ttft_s_max": 1,
             "p90_mean_tpot_s": .1, "p90_mean_tpot_s_min": .1,
             "p90_mean_tpot_s_max": .1}
            for model in plot_style.MODELS for rate in (*campaign.RATES,
                                                         *campaign.KNEE_RATES[model])]

    plot.write(rows, contract, tmp_path / "dense")

    assert (tmp_path / "dense.png").stat().st_size
