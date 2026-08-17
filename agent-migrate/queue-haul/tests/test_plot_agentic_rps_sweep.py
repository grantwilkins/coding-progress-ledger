import agentic_rps_sweep_campaign as campaign
import plot_agentic_rps_sweep as plotter


def result(model):
    curve = []
    for index, rate in enumerate(campaign.RATES_RPS):
        ttft = 1 + index
        tpot = .04 + index * .03
        curve.append({
            "offered_rps": rate,
            "repeats": 3 if rate in (.25, .5) else 1,
            "p90_ttft_s_median": ttft,
            "p90_ttft_s_minimum": ttft - .1,
            "p90_ttft_s_maximum": ttft + .1,
            "p90_mean_tpot_s_median": tpot,
            "p90_mean_tpot_s_minimum": tpot - .005,
            "p90_mean_tpot_s_maximum": tpot + .005,
        })
    return {
        "model": model,
        "slo": {"p90_ttft_s": 2, "p90_mean_tpot_s": .1},
        "first_confirmed_violation_rps": .5,
        "curve": curve,
    }


def test_plot_writes_one_two_panel_figure_with_every_model(tmp_path):
    summary = {
        "schema": campaign.SCHEMA,
        "stage": "reduced",
        "models": {model: result(model) for model in campaign.MODELS},
    }

    output = plotter.plot(summary, tmp_path)

    assert output.name == "agentic-rps-sweep"
    assert output.with_suffix(".pdf").is_file()
    assert output.with_suffix(".png").is_file()
