"""Canonical style for figures written under ``outputs/``."""

import matplotlib


FIGSIZE = (8, 5)
WIDE_FIGSIZE = (8, 4)
COMPACT_FIGSIZE = (5, 4)
COLUMN_FIGSIZE = (4, 3)
FONT_SIZE = 15
COLUMN_FONT_SIZE = 11
COLUMN_LEGEND_FONT_SIZE = 9
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
SHORT_POLICY_NAMES = {**POLICY_NAMES, "queue_haul": "Queue-Haul"}
POLICY_COLORS = dict(zip(POLICIES, (
    "#0072B2", "#E69F00", "#F0E442", "#D55E00",
    "#56B4E9", "#CC79A7", "#009E73", "#000000",
)))
POLICY_LINESTYLES = dict(zip(POLICIES, (
    "-", "--", (0, (3, 1, 1, 1)), (0, (5, 1)), "-.", ":",
    (0, (3, 1)), (0, (1, 1)),
)))
REFERENCE = "exact_modeled_milp_optimum"
POLICY_NAMES[REFERENCE] = "Exact modeled MILP optimum"
POLICY_COLORS[REFERENCE] = "#000000"
POLICY_LINESTYLES[REFERENCE] = "-"
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
TIMING_ACTION_NAMES = {"replay": ACTION_NAMES["replay"],
                       "kv_transfer": ACTION_NAMES["kv_transfer"],
                       "mixed": "Mixed"}
TIMING_ACTION_COLORS = {"replay": ACTION_COLORS["replay"],
                        "kv_transfer": ACTION_COLORS["kv_transfer"],
                        "mixed": "#009E73"}
POWER_FAMILY_NAMES = {
    "idle": "Idle", "prefill": "Prefill", "decode": "Decode",
    "agentic": "Agentic", "campaign": "Campaign mix",
}
POWER_FAMILY_COLORS = {
    "idle": "#777777", "prefill": "#E98300", "decode": "#006CB8",
    "agentic": "#009E73", "campaign": "#CC79A7",
}
POWER_FAMILY_MARKERS = dict(zip(
    POWER_FAMILY_NAMES, ("o", "^", "s", "D", "P")))
ACTION_HATCHES = {
    "replay": "..", "kv_transfer": "xx",
    "east_replay": "..", "east_kv_transfer": "xx",
    "germany_replay": "//", "germany_kv_transfer": "\\\\",
    "not_moved": "",
}
REPAIR_NAMES = {
    "unchanged": "No repair needed",
    "applied": "Repair applied",
    "revised_maximum": "Target unattainable",
}
REPAIR_COLORS = {
    "unchanged": "#999999",
    "applied": POLICY_COLORS["queue_haul"],
    "revised_maximum": "#D55E00",
}
SCHEDULE_COMPARISON_NAMES = {
    "replan": "QH replan",
    "no_replan": "No replan",
}
SCHEDULE_COMPARISON_COLORS = {
    "replan": POLICY_COLORS["queue_haul"],
    "no_replan": "#555555",
}
SCHEDULE_COMPARISON_LINESTYLES = {
    "replan": "-",
    "no_replan": "--",
}
EVENT_NAMES = {
    "resource_shift": "10× resource drop",
    "repair_decision": "Replan decision",
    "shed_target": "Shed target",
}
EVENT_COLORS = {
    "resource_shift": "#D55E00",
    "repair_decision": POLICY_COLORS["queue_haul"],
    "shed_target": "#000000",
}
EVENT_LINESTYLES = {
    "resource_shift": (0, (3, 1)),
    "repair_decision": (0, (1, 1)),
    "shed_target": "--",
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


def policy_style(policy, names=POLICY_NAMES):
    return {
        "color": POLICY_COLORS[policy],
        "linestyle": POLICY_LINESTYLES[policy],
        "linewidth": LINE_WIDTH,
        "label": names[policy],
    }
