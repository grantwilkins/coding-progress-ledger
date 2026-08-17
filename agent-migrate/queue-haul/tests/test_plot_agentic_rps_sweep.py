"""
Claim:
The figure plots only GPT-OSS-20B in increasing RPS order against fixed SLOs.

Plausible wrong implementations:
- Plot all models from the summary.
- Preserve arbitrary input order instead of increasing RPS.
- Use observed measurements or model-colored lines as SLOs.
"""

import matplotlib.pyplot as plt
import pytest

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


def test_plot_rejects_legacy_mean_tpot_summary(tmp_path):
    with pytest.raises(ValueError, match="not reduced evidence"):
        plotter.plot({"schema": "queue-haul-agentic-rps-sweep-v2",
                      "stage": "reduced"}, tmp_path)


def test_plot_compares_matching_hardware_with_one_shared_slo(tmp_path,
                                                             monkeypatch):
    summary = {
        "schema": plotter.SCHEMA, "stage": "reduced", "hardware": "a100",
        "request_shape": {"prompt_tokens": 3920, "output_tokens": 1024},
        "models": {plotter.MODEL: {
            "slo": {"p90_ttft_s": 2, "p90_tpot_s": .1},
            "curve": [{"offered_rps": 1, "p90_ttft_s_median": 3,
                       "p90_tpot_s_median": .03}],
        }},
    }
    h100 = {
        "schema": plotter.SCHEMA, "stage": "reduced", "hardware": "h100",
        "request_shape": summary["request_shape"],
        "models": {plotter.MODEL: {
            "slo": summary["models"][plotter.MODEL]["slo"],
            "curve": [{"offered_rps": 1, "p90_ttft_s_median": 1,
                       "p90_tpot_s_median": .05}],
        }},
    }
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plotter.plot(summary, tmp_path, h100)
    lines = plt.gcf().axes[0].lines

    assert [line.get_label() for line in lines] == [
        "Agent - A100", "Agent - H100", "SLO"]
    assert [list(line.get_ydata()) for line in lines] == [[3], [1], [2, 2]]
    plt.close("all")


def test_plot_rejects_mismatched_h100_request_shape(tmp_path):
    summary = {"schema": plotter.SCHEMA, "stage": "reduced", "hardware": "a100",
               "request_shape": {"prompt_tokens": 3920, "output_tokens": 1024},
               "models": {plotter.MODEL: {"curve": [], "slo": {}}}}
    h100 = {"schema": plotter.SCHEMA, "stage": "reduced", "hardware": "h100",
            "request_shape": {"prompt_tokens": 3920, "output_tokens": 512}}

    with pytest.raises(ValueError, match="request shape"):
        plotter.plot(summary, tmp_path, h100)
