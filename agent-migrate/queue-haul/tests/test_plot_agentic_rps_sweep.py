"""
Claim:
The figure plots only GPT-OSS-20B in increasing RPS order against fixed SLOs.

Plausible wrong implementations:
- Plot all models from the summary.
- Preserve arbitrary input order instead of increasing RPS.
- Use observed measurements or model-colored lines as SLOs.
"""

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import pytest

import plot_agentic_rps_sweep as plotter
import plot_style


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
    assert plt.gcf().get_figwidth() <= 3.5
    assert axes[0].get_position().x0 < axes[1].get_position().x0
    assert axes[0].get_xlim() == axes[1].get_xlim() == (0, 8.25)
    assert list(axes[0].get_xticks()) == [0, 2, 4, 6, 8]
    assert axes[1].get_ylim() == (0, 52)
    assert axes[1].get_yticks()[-1] == 50
    assert list(axes[0].lines[0].get_xdata()) == [1, 2]
    assert list(axes[0].lines[0].get_ydata()) == [1, 2]
    assert axes[0].lines[0].get_label() == "OpenHands Agentic"
    assert set(axes[0].lines[1].get_ydata()) == {1}
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
    figure = plt.gcf()
    lines = figure.axes[0].lines

    assert figure.get_figwidth() == 5.2
    assert [line.get_label() for line in lines] == [
        "Agent - A100", "Agent - H100", "SLO"]
    assert [list(line.get_ydata()) for line in lines] == [[3], [1], [1, 1]]
    assert [list(line.get_ydata()) for line in figure.axes[1].lines] \
        == [[30], [50], [50, 50]]
    assert lines[-1].get_linewidth() > lines[0].get_linewidth()
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "Agent - A100", "Agent - H100", "SLO"]
    plt.close("all")


def test_plot_rejects_mismatched_h100_request_shape(tmp_path):
    summary = {"schema": plotter.SCHEMA, "stage": "reduced", "hardware": "a100",
               "request_shape": {"prompt_tokens": 3920, "output_tokens": 1024},
               "models": {plotter.MODEL: {"curve": [], "slo": {}}}}
    h100 = {"schema": plotter.SCHEMA, "stage": "reduced", "hardware": "h100",
            "request_shape": {"prompt_tokens": 3920, "output_tokens": 512}}

    with pytest.raises(ValueError, match="request shape"):
        plotter.plot(summary, tmp_path, h100)


def test_v4_plot_shows_raw_points_exact_intervals_and_summary_slo(
        tmp_path, monkeypatch):
    curve = [{
        "offered_rps": rate, "realized_rps_median": rate + .1,
        "p90_ttft_s_median": value, "p90_ttft_s_ci_low": value - .1,
        "p90_ttft_s_ci_high": value + .1,
        "p90_tpot_s_median": .02, "p90_tpot_s_ci_low": .015,
        "p90_tpot_s_ci_high": .025,
        "points": [
            {"block": block, "realized_rps": rate + block / 100,
             "status": "numeric", "p90_ttft_s": value + block / 100,
             "p90_tpot_s": .02 + block / 10000}
            for block in range(2)
        ],
    } for rate, value in ((10, 1.2), (1, .4))]
    curve[0]["points"].append({
        "block": 2, "realized_rps": 10.2, "status": "service_failure",
        "p90_ttft_s": None, "p90_tpot_s": None,
    })
    summary = {
        "schema": plotter.SLO_SCHEMA, "stage": "reduced",
        "hardware": "h100", "comparison_sha256": "shared",
        "shared_runtime_sha256": "runtime", "launch_git_sha": "git",
        "models": {plotter.MODEL: {
            "slo": {"p90_ttft_s": .9, "p90_tpot_s": .05,
                    "source": "fixed-paper-reference"},
            "curve": curve,
        }},
    }
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plotter.plot(summary, tmp_path)
    axis = plt.gcf().axes[0]
    errorbar = next(container for container in axis.containers
                    if isinstance(container, ErrorbarContainer))
    slo = next(line for line in axis.lines if line.get_label() == "SLO")

    assert list(errorbar.lines[0].get_xdata()) == [1, 10]
    assert list(errorbar.lines[0].get_ydata()) == [.4, 1.2]
    assert len(axis.collections[0].get_offsets()) == 4
    failure = next(collection for collection in axis.collections
                   if collection.get_label() ==
                   "Censored service failure")
    assert list(failure.get_offsets()[0]) == [10.2, 1.02]
    assert set(slo.get_ydata()) == {.9}
    assert axis.get_xlabel() == "Rate (req/s)"
    assert axis.get_xscale() == "log"
    assert axis.get_xlim()[0] < 1
    assert axis.get_xlim()[1] > 10.2
    plt.close("all")


def test_quick_plot_marks_metric_violations(tmp_path, monkeypatch):
    curve = [{
        "offered_rps": rate, "realized_rps_median": rate,
        "p90_ttft_s_median": ttft, "p90_ttft_s_ci_low": ttft - .1,
        "p90_ttft_s_ci_high": ttft + .1,
        "p90_tpot_s_median": tpot, "p90_tpot_s_ci_low": tpot - .005,
        "p90_tpot_s_ci_high": tpot + .005,
        "points": [{"status": "numeric", "realized_rps": rate,
                    "p90_ttft_s": ttft, "p90_tpot_s": tpot}],
    } for rate, ttft, tpot in ((.125, .8, .04), (.25, .8, .06))]
    summary = {
        "schema": plotter.QUICK_SCHEMA, "stage": "reduced",
        "hardware": "h100", "comparison_sha256": "shared",
        "shared_runtime_sha256": "runtime", "launch_git_sha": "git",
        "models": {plotter.MODEL: {
            "slo": {"p90_ttft_s": 1., "p90_tpot_s": .05},
            "curve": curve,
        }},
    }
    monkeypatch.setattr(plt, "close", lambda figure: None)

    plotter.plot(summary, tmp_path, ttft_slo_s=.7)
    figure = plt.gcf()
    ttft_axis = figure.axes[0]
    axis = figure.axes[1]
    violations = next(collection for collection in axis.collections
                      if collection.get_label() == plot_style.SLO_VIOLATION_NAME)
    errorbar = next(container for container in axis.containers
                    if isinstance(container, ErrorbarContainer))

    assert set(next(line for line in ttft_axis.lines
                    if line.get_label() == "SLO").get_ydata()) == {.7}
    assert len(next(collection for collection in ttft_axis.collections
                    if collection.get_label() ==
                    plot_style.SLO_VIOLATION_NAME).get_offsets()) == 2
    assert list(violations.get_offsets()[0]) == [.25, 60]
    assert errorbar.lines[0].get_marker() == "o"
    assert errorbar.lines[0].get_markevery() == [0]
    assert violations.get_edgecolors()[0].tolist() == pytest.approx(
        plt.matplotlib.colors.to_rgba(plot_style.AGENTIC_HARDWARE_COLORS["h100"])
    )
    assert plot_style.SLO_VIOLATION_NAME in [
        text.get_text() for text in figure.legends[0].get_texts()
    ]
    assert axis.get_xscale() == "log"
    plt.close("all")
