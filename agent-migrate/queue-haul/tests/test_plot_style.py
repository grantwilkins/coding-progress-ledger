"""
Claim:
Output plots share one stable policy identity and typography registry.

Plausible wrong implementations:
- Shift a policy to another Tab10 color.
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
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
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
