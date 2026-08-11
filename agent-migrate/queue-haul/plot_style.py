"""Canonical style for figures written under ``outputs/``."""

import matplotlib


FIGSIZE = (8, 5)
COMPACT_FIGSIZE = (5, 4)
FONT_SIZE = 15
LEGEND_FONT_SIZE = ANNOTATION_FONT_SIZE = 11
LINE_WIDTH = 3
SAVE_DPI = 220
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "isolated_fastest",
    "kv_only", "replay_only", "queue_haul_power_blind",
    "queue_haul_deadline_blind",
)
POLICY_NAMES = dict(zip(POLICIES, (
    "Queue-Haul LP", "Queue-Haul Greedy", "Queue-Haul Lagrangian Greedy",
    "True Greedy", "KV Migrate Only", "Replay Context Only",
    "Queue-Haul Power Blind", "Queue-Haul Deadline Blind",
)))
POLICY_COLORS = dict(zip(POLICIES, matplotlib.colormaps["tab10"].colors))
POLICY_LINESTYLES = dict(zip(POLICIES, (
    "-", "--", (0, (3, 1, 1, 1)), (0, (5, 1)), "-.", ":",
    (0, (3, 1)), (0, (1, 1)),
)))


def apply():
    matplotlib.rcParams.update({
        "figure.figsize": FIGSIZE, "savefig.dpi": SAVE_DPI,
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "legend.fontsize": LEGEND_FONT_SIZE,
        "lines.linewidth": LINE_WIDTH,
    })


def policy_style(policy):
    return {
        "color": POLICY_COLORS[policy],
        "linestyle": POLICY_LINESTYLES[policy],
        "linewidth": LINE_WIDTH,
        "label": POLICY_NAMES[policy],
    }
