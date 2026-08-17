"""Canonical style for figures written under ``outputs/``."""

import matplotlib


FIGSIZE = (8, 5)
WIDE_FIGSIZE = (8, 4)
COMPACT_FIGSIZE = (5, 4)
COLUMN_FIGSIZE = (4, 3)
HALF_COLUMN_FIGSIZE = (1.65, 1.75)
FONT_SIZE = 15
COLUMN_FONT_SIZE = 11
COLUMN_LEGEND_FONT_SIZE = 9
HALF_COLUMN_FONT_SIZE = 7.5
HALF_COLUMN_LEGEND_FONT_SIZE = 6.5
HALF_COLUMN_ANNOTATION_FONT_SIZE = 7
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
COMPACT_POLICY_NAMES = {
    **POLICY_NAMES, "kv_only": "KV Migrate", "replay_only": "Replay Context",
}
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
    "not_moved": "Not moved",
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
POWER_FAMILY_NAMES = {"idle": "Idle", "sessions": "Sessions"}
POWER_FAMILY_COLORS = {"idle": "#777777", "sessions": "#006CB8"}
POWER_FAMILY_MARKERS = {"idle": "o", "sessions": "s"}
SERVICE_LOADS = ("prefill_heavy", "decode_heavy")
SERVICE_LOAD_NAMES = {
    "prefill_heavy": "Prefill-heavy added load",
    "decode_heavy": "Decode-heavy added load",
}
SERVICE_LOAD_COLORS = {
    "prefill_heavy": "#D55E00",
    "decode_heavy": "#0072B2",
}
SERVICE_LOAD_LINESTYLES = {
    "prefill_heavy": "--",
    "decode_heavy": "-",
}
SERVICE_LOAD_MARKERS = {"prefill_heavy": "o", "decode_heavy": "s"}
SERVICE_MIXES = ("prefill_heavy", "balanced", "decode_heavy")
SERVICE_MIX_NAMES = {
    **SERVICE_LOAD_NAMES,
    "balanced": "Balanced added load",
}
SERVICE_MIX_COLORS = {
    **SERVICE_LOAD_COLORS,
    "balanced": "#009E73",
}
SERVICE_MIX_LINESTYLES = {
    **SERVICE_LOAD_LINESTYLES,
    "balanced": ":",
}
SERVICE_MIX_MARKERS = {
    **SERVICE_LOAD_MARKERS,
    "balanced": "^",
}
SERVICE_EVIDENCE_STAGE_NAMES = {
    "discovery": "Discovery",
    "held_out": "Held-out confirmation",
}
ACTION_HATCHES = {
    "replay": "..", "kv_transfer": "xx",
    "east_replay": "..", "east_kv_transfer": "xx",
    "germany_replay": "//", "germany_kv_transfer": "\\\\",
    "not_moved": "--",
}
RESOURCE_STATE_NAMES = {
    "none": "None bound", "bandwidth": "Bandwidth",
    "hbm": "HBM", "bandwidth-hbm": "HBM + bandwidth",
    "dest_compute": "Dest. compute",
    "bandwidth-dest_compute": "Bandwidth + compute",
    "dest_compute-hbm": "HBM + compute",
    "bandwidth-dest_compute-hbm": "All bound",
}
RESOURCE_STATE_COLORS = {
    "none": POLICY_COLORS["queue_haul"],
    "bandwidth": "#CC79A7",
    "hbm": POLICY_COLORS["greedy"],
    "bandwidth-hbm": POLICY_COLORS["greedy"],
    "dest_compute": POLICY_COLORS["queue_haul_power_blind"],
    "bandwidth-dest_compute": POLICY_COLORS["queue_haul_power_blind"],
    "dest_compute-hbm": "#000000", "bandwidth-dest_compute-hbm": "#000000",
}
RESOURCE_STATE_LINESTYLES = {
    "none": "-", "bandwidth": "--", "hbm": "-", "bandwidth-hbm": "--",
    "dest_compute": "-", "bandwidth-dest_compute": "--",
    "dest_compute-hbm": "-", "bandwidth-dest_compute-hbm": "--",
}
MODELS = ("openai/gpt-oss-20b", "Qwen/Qwen3.8-27B",
          "google/gemma-4-26B-A4B-it")
MODEL_NAMES = dict(zip(MODELS, ("GPT-OSS-20B", "Qwen3.8-27B",
                                "Gemma-4-26B-A4B")))
MODEL_COLORS = dict(zip(MODELS, ("#0072B2", "#D55E00", "#009E73")))
MODEL_LINESTYLES = dict(zip(MODELS, ("-", "--", ":")))
MODEL_MARKERS = dict(zip(MODELS, ("o", "s", "^")))
SERVICE_DIRECTIONS = ("baseline", "prefill_heavy", "balanced", "decode_heavy")
SERVICE_DIRECTION_NAMES = dict(zip(SERVICE_DIRECTIONS, (
    "Baseline", "Prefill-heavy", "Balanced", "Decode-heavy")))
SERVICE_DIRECTION_COLORS = dict(zip(SERVICE_DIRECTIONS, (
    "#777777", "#D55E00", "#009E73", "#0072B2")))
SERVICE_DIRECTION_LINESTYLES = dict(zip(SERVICE_DIRECTIONS, (
    ":", "-", "-.", "--")))
SERVICE_DIRECTION_MARKERS = dict(zip(SERVICE_DIRECTIONS, ("D", "^", "s", "o")))
AGENTIC_WORKLOAD_NAME = "OpenHands Agentic"
AGENTIC_WORKLOAD_COLOR = MODEL_COLORS["openai/gpt-oss-20b"]
AGENTIC_WORKLOAD_MARKER = "o"
AGENTIC_HARDWARE = ("a100", "h100")
AGENTIC_HARDWARE_NAMES = {"a100": "A100", "h100": "H100"}
AGENTIC_HARDWARE_COLORS = {"a100": "#0072B2", "h100": "#D55E00"}
AGENTIC_HARDWARE_MARKERS = {"a100": "o", "h100": "s"}
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
    "replan": "Queue-Haul replan",
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
REPAIR_ACTION_NAMES = {
    "retained": "Retained",
    "method": "Changed method",
    "destination": "Changed destination",
    "removed": "Removed from plan",
}
REPAIR_ACTION_SHORT_NAMES = {
    "retained": "Retained",
    "method": "Diff. Action",
    "destination": "Destination",
    "removed": "Removed",
}
REPAIR_ACTION_COLORS = {
    "retained": "#B3B3B3",
    "method": POLICY_COLORS["queue_haul"],
    "destination": "#E69F00",
    "removed": "#555555",
}
REPAIR_ACTION_HATCHES = {
    "retained": "",
    "method": "xx",
    "destination": "//",
    "removed": "..",
}
RESOURCE_FAULT_SHORT_NAMES = {
    "bandwidth": "BW",
    "prefill": "PF",
    "joint": "Both",
}
RESOURCE_FAULT_NAMES = {
    "bandwidth": "Bandwidth",
    "prefill": "Prefill",
    "joint": "Both",
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


def half_column(axis):
    axis.tick_params(which="major", labelsize=HALF_COLUMN_FONT_SIZE,
                     length=2, pad=1)
    axis.tick_params(which="minor", length=1)
    axis.xaxis.label.set_size(HALF_COLUMN_FONT_SIZE)
    axis.yaxis.label.set_size(HALF_COLUMN_FONT_SIZE)


def policy_style(policy, names=POLICY_NAMES):
    return {
        "color": POLICY_COLORS[policy],
        "linestyle": POLICY_LINESTYLES[policy],
        "linewidth": LINE_WIDTH,
        "label": names[policy],
    }
