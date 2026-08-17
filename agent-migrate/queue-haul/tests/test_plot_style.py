"""
Claim:
Output plots share one stable policy identity and typography registry.

Plausible wrong implementations:
- Retain a Tab10 color or shift an Okabe-Ito color to another policy.
- Reuse a line style or retain the old per-session-fastest name.
- Silently style an unknown policy or fail to apply the documented font sizes.
- Reuse an action hatch so grayscale action segments become ambiguous.
- Change a model's identity between architecture-campaign panels.
- Leave Bandwidth visually conflated with None bound in the pooled frontier.
"""

import matplotlib
import pytest

import plot_style


def test_policy_styles_are_stable_and_unique():
    assert plot_style.POLICY_NAMES["isolated_fastest"] == "True Greedy"
    assert plot_style.COMPACT_POLICY_NAMES["kv_only"] == "KV Migrate"
    assert plot_style.COMPACT_POLICY_NAMES["replay_only"] == "Replay Context"
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


def test_selected_actions_have_unique_hatches():
    assert plot_style.TIMING_ACTION_NAMES["mixed"] == "Mixed"
    assert plot_style.TIMING_ACTION_COLORS["mixed"] == "#009E73"
    actions = ("east_replay", "east_kv_transfer", "germany_replay",
               "germany_kv_transfer")
    assert len({plot_style.ACTION_HATCHES[action] for action in actions}) == 4


def test_repair_comparison_has_canonical_policy_and_event_styles():
    assert plot_style.SCHEDULE_COMPARISON_NAMES == {
        "replan": "Queue-Haul replan", "no_replan": "No replan"}
    assert plot_style.SCHEDULE_COMPARISON_COLORS["replan"] \
        == plot_style.POLICY_COLORS["queue_haul"]
    assert set(plot_style.EVENT_NAMES) == {
        "resource_shift", "repair_decision", "shed_target"}


def test_power_validation_methods_have_unique_markers():
    assert len(set(plot_style.POWER_VALIDATION_MARKERS.values())) == 5


def test_power_families_have_shared_names_colors_and_markers():
    assert set(plot_style.POWER_FAMILY_NAMES) == {"idle", "sessions"}
    assert set(plot_style.POWER_FAMILY_NAMES) == set(plot_style.POWER_FAMILY_COLORS) \
        == set(plot_style.POWER_FAMILY_MARKERS)
    assert len(set(plot_style.POWER_FAMILY_MARKERS.values())) == 2


def test_service_load_directions_have_shared_visual_identities():
    assert set(plot_style.SERVICE_LOADS) == set(plot_style.SERVICE_LOAD_NAMES) \
        == set(plot_style.SERVICE_LOAD_COLORS) \
        == set(plot_style.SERVICE_LOAD_LINESTYLES) \
        == set(plot_style.SERVICE_LOAD_MARKERS)
    assert len(set(plot_style.SERVICE_LOAD_COLORS.values())) == 2
    assert len(set(plot_style.SERVICE_LOAD_MARKERS.values())) == 2


def test_model_architectures_have_one_canonical_visual_identity():
    assert set(plot_style.MODELS) == set(plot_style.MODEL_NAMES) \
        == set(plot_style.MODEL_COLORS) == set(plot_style.MODEL_LINESTYLES) \
        == set(plot_style.MODEL_MARKERS)
    assert len(set(plot_style.MODEL_COLORS.values())) == len(plot_style.MODELS)


def test_service_directions_have_one_canonical_visual_identity():
    assert set(plot_style.SERVICE_DIRECTIONS) == set(
        plot_style.SERVICE_DIRECTION_NAMES) == set(
        plot_style.SERVICE_DIRECTION_COLORS) == set(
        plot_style.SERVICE_DIRECTION_LINESTYLES) == set(
        plot_style.SERVICE_DIRECTION_MARKERS)
    assert len(set(plot_style.SERVICE_DIRECTION_MARKERS.values())) \
        == len(plot_style.SERVICE_DIRECTIONS)


def test_agentic_hardware_has_one_canonical_visual_identity():
    assert set(plot_style.AGENTIC_HARDWARE) == set(
        plot_style.AGENTIC_HARDWARE_NAMES) == set(
        plot_style.AGENTIC_HARDWARE_COLORS) == set(
        plot_style.AGENTIC_HARDWARE_MARKERS)


def test_displayed_resource_states_have_distinct_canonical_colors():
    states = ("hbm", "bandwidth", "dest_compute",
              "bandwidth-dest_compute-hbm", "none")
    colors = [matplotlib.colors.to_hex(plot_style.RESOURCE_STATE_COLORS[state])
              for state in states]

    assert plot_style.RESOURCE_STATE_COLORS["bandwidth"] == "#CC79A7"
    assert len(set(colors)) == len(states)
