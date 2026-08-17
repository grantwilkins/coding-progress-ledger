"""
Claim:
The policy CDF pools every paired workload-constraint draw once and reports its
nonlinear source-power target-attainment fraction.

Plausible wrong implementations:
- Average constraint summaries instead of equally weighted cases.
- Drop low-attainment cases or normalize by successful cases.
- Include the same draw-constraint case more than once for a policy.
- Plot a 0-1 fraction as though it were already a percentage.
"""

import pytest

from plot_workload_policy_attainment import attainment_curve


def test_attainment_curve_weights_every_paired_case_once_and_scales_percent():
    rows = [
        {"replicate": 0, "case_id": "hbm", "policy": "queue_haul",
         "attainment_fraction": 1},
        {"replicate": 0, "case_id": "none", "policy": "queue_haul",
         "attainment_fraction": .25},
        {"replicate": 1, "case_id": "hbm", "policy": "queue_haul",
         "attainment_fraction": 0},
    ]
    x, y = attainment_curve(rows, "queue_haul")
    assert x.tolist() == [0, 25, 100]
    assert y.tolist() == pytest.approx([1 / 3, 2 / 3, 1])
    with pytest.raises(RuntimeError, match="one row"):
        attainment_curve(rows + [rows[0]], "queue_haul")
