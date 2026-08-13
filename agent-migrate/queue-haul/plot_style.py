"""Canonical style for figures written under ``outputs/``."""

import matplotlib


FIGSIZE = (8, 5)
WIDE_FIGSIZE = (8, 4)
COMPACT_FIGSIZE = (5, 4)
FONT_SIZE = 15
LARGE_FONT_SIZE = 17
LEGEND_FONT_SIZE = ANNOTATION_FONT_SIZE = 11
LARGE_LEGEND_FONT_SIZE = 12
LARGE_ANNOTATION_FONT_SIZE = 13
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
POLICY_COLORS = dict(zip(POLICIES, (
    "#0072B2", "#E69F00", "#F0E442", "#D55E00",
    "#56B4E9", "#CC79A7", "#009E73", "#000000",
)))
POLICY_LINESTYLES = dict(zip(POLICIES, (
    "-", "--", (0, (3, 1, 1, 1)), (0, (5, 1)), "-.", ":",
    (0, (3, 1)), (0, (1, 1)),
)))
ACTION_NAMES = {
    "replay": "Replay", "kv_transfer": "KV transfer",
    "east_replay": "Replay → East", "east_kv_transfer": "KV transfer → East",
    "germany_replay": "Replay → Germany",
    "germany_kv_transfer": "KV transfer → Germany",
    "not_moved": "Remains at source",
}
ACTION_COLORS = {
    "replay": "#E98300", "kv_transfer": "#006CB8",
    "east_replay": "#F6B65B", "germany_replay": "#D55E00",
    "east_kv_transfer": "#56B4E9", "germany_kv_transfer": "#0072B2",
    "not_moved": "#999999",
}
ACTION_HATCHES = {
    "east_replay": "..", "east_kv_transfer": "xx",
    "germany_replay": "//", "germany_kv_transfer": "\\\\",
    "not_moved": "",
}
POWER_VALIDATION_METHODS = ("lp", "greedy", "milp", "power-unaware", "random")
POWER_VALIDATION_NAMES = dict(zip(POWER_VALIDATION_METHODS, (
    "Queue-Haul LP", "Queue-Haul Greedy", "MILP", "Power unaware", "Random",
)))
POWER_VALIDATION_COLORS = dict(zip(POWER_VALIDATION_METHODS, (
    POLICY_COLORS["queue_haul"], POLICY_COLORS["greedy"],
    "#009E73", "#CC79A7", "#000000",
)))
POWER_VALIDATION_MARKERS = dict(zip(
    POWER_VALIDATION_METHODS, ("o", "s", "^", "D", "x")))


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
