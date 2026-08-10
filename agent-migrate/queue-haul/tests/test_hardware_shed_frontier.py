"""
Claim:
The requested-shed frontier caps harmless overshed at the request, never
decreases as the request rises, and retains the last safely attained value when
a later plan is unsafe or insufficient.

Plausible wrong implementations:
- Plot raw bundle overshoot above the requested shed.
- Drop or zero an unsafe point instead of drawing a plateau.
- Allow a later lower-performing plan to make the frontier decrease.
- Confuse requested watts with a normalized attainment fraction.
- Credit a deadline-blind plan with its long-horizon rather than 30-second shed.
"""

from types import SimpleNamespace

import pytest

import plot_hardware_shed_frontier as frontier
from plot_hardware_shed_frontier import plateau_attainment


def test_plateau_attainment_caps_overshed_and_preserves_safe_envelope():
    assert plateau_attainment(
        [0, 10, 20, 30], [0, 12, 8, 25], [True, True, False, True],
    ) == [0, 10, 10, 25]
    with pytest.raises(ValueError):
        plateau_attainment([0, 2, 1], [0, 2, 1], [True] * 3)


def test_deadline_blind_frontier_uses_evaluation_horizon(monkeypatch):
    actual, planned = object(), object()
    result = SimpleNamespace(
        expected_source_power_at_deadline_w=20, moves=("late",))
    seen = {}
    monkeypatch.setattr(frontier, "_expected_scenario",
                        lambda problem, moves: seen.update(
                            problem=problem, moves=moves) or "evaluation")
    monkeypatch.setattr(frontier, "predict", lambda scenario, *_args, **_kwargs:
                        seen.update(scenario=scenario) or SimpleNamespace(
                            modeled_source_power_at_deadline_w=70))

    assert frontier.evaluated_source_power(
        actual, planned, result, None, None) == 70
    assert seen == {"problem": actual, "moves": ("late",),
                    "scenario": "evaluation"}
    assert frontier.evaluated_source_power(
        actual, actual, result, None, None) == 20
