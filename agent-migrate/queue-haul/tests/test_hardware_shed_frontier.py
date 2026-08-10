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
"""

import pytest

from plot_hardware_shed_frontier import plateau_attainment


def test_plateau_attainment_caps_overshed_and_preserves_safe_envelope():
    assert plateau_attainment(
        [0, 10, 20, 30], [0, 12, 8, 25], [True, True, False, True],
    ) == [0, 10, 10, 25]
    with pytest.raises(ValueError):
        plateau_attainment([0, 2, 1], [0, 2, 1], [True] * 3)
