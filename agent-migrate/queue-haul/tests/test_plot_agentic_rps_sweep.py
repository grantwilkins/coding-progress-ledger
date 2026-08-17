"""
Claim:
The figure plots only GPT-OSS-20B in increasing RPS order against fixed SLOs.

Plausible wrong implementations:
- Plot all models from the summary.
- Preserve arbitrary input order instead of increasing RPS.
- Use observed measurements or model-colored lines as SLOs.
"""

import matplotlib.pyplot as plt

import plot_agentic_rps_sweep as plotter


def test_plot_selects_sorted_gpt_oss_curve_and_fixed_slos(tmp_path, monkeypatch):
    def result(offset):
        return {
            "slo": {"p90_ttft_s": 2, "p90_tpot_s": .1},
            "curve": [
                {"offered_rps": rate, "p90_ttft_s_median": offset + rate,
                 "p90_tpot_s_median": offset + rate / 100}
                for rate in (2, 1)
            ],
        }

    summary = {
        "schema": plotter.SCHEMA,
        "stage": "reduced",
        "models": {
            plotter.MODEL: result(0),
            "plausible/wrong-model": result(10),
        },
    }
    monkeypatch.setattr(plt, "close", lambda figure: None)

    output = plotter.plot(summary, tmp_path)
    axes = plt.gcf().axes

    assert output.with_suffix(".pdf").is_file()
    assert len(axes) == 2
    assert list(axes[0].lines[0].get_xdata()) == [1, 2]
    assert list(axes[0].lines[0].get_ydata()) == [1, 2]
    assert axes[0].lines[0].get_label() == "OpenHands Agentic"
    assert set(axes[0].lines[1].get_ydata()) == {2}
    assert axes[0].lines[1].get_color() == "black"
    assert axes[0].lines[1].get_linestyle() == ":"
    plt.close("all")
