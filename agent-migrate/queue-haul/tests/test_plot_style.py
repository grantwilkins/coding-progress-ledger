"""
Claim:
Output plots share one stable policy identity and typography registry.

Plausible wrong implementations:
- Retain a Tab10 color or shift an Okabe-Ito color to another policy.
- Reuse a line style or retain the old per-session-fastest name.
- Silently style an unknown policy or fail to apply the documented font sizes.
"""

import matplotlib
import pytest

import plot_style


def test_policy_styles_are_stable_and_unique():
    assert plot_style.POLICY_NAMES["isolated_fastest"] == "True Greedy"
    assert [matplotlib.colors.to_hex(plot_style.POLICY_COLORS[policy])
            for policy in plot_style.POLICIES] == [
        "#0072b2", "#e69f00", "#f0e442", "#d55e00",
        "#56b4e9", "#cc79a7", "#009e73", "#000000",
    ]
    assert len({str(value) for value in plot_style.POLICY_LINESTYLES.values()}) \
        == len(plot_style.POLICIES)
    with pytest.raises(KeyError):
        plot_style.policy_style("unknown")


def test_apply_sets_documented_defaults():
    plot_style.apply()
    assert tuple(matplotlib.rcParams["figure.figsize"]) == plot_style.FIGSIZE
    assert matplotlib.rcParams["font.size"] == plot_style.FONT_SIZE
    assert matplotlib.rcParams["legend.fontsize"] == plot_style.LEGEND_FONT_SIZE
    assert matplotlib.rcParams["lines.linewidth"] == plot_style.LINE_WIDTH
