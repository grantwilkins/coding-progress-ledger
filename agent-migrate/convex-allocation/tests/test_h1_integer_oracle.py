"""
Claim:
The H1 integer oracle is a tiny exhaustive discrete feasibility check that uses
the same movement, resource-pressure, and absolute-deadline pass criteria as the
fixed-target H1 stress table.

Plausible wrong implementations:
- Optimize or report a fractional allocation instead of enumerating integer ones.
- Use release-relative deadline metrics instead of absolute event-start metrics.
- Let a resource or deadline violation report as a pass.
- Grow the oracle into a large sweep that is no longer a bounded sanity check.
"""

from __future__ import annotations

import numpy as np

from experiments.run_h1_integer_oracle import (
    DRAIN_WINDOW_S,
    RELEASE_POLICY,
    RETAINED_PREFILL_FRACTION,
    _verdict,
    h1_integer_oracle_row,
    make_h1_integer_problem,
)


def test_h1_integer_oracle_is_small_discrete_and_passes_h1_criteria():
    problem = make_h1_integer_problem()
    row = h1_integer_oracle_row(problem)

    assert problem.G == 2
    assert int(np.sum(problem.d)) == 8
    assert row["enumerated_allocations"] <= 5000
    assert row["drain_window_s"] == DRAIN_WINDOW_S
    assert row["retained_prefill_fraction"] == RETAINED_PREFILL_FRACTION
    assert row["release_policy"] == RELEASE_POLICY
    assert row["target_moved_fraction"] >= 1.0
    assert row["network_capacity_pressure"] <= 1.0
    assert row["prefill_capacity_pressure"] <= 1.0
    assert row["absolute_p95_delay_over_deadline"] <= 1.0
    assert row["absolute_deadline_miss_rate"] <= 0.01
    assert row["verdict"] == "Pass"


def test_h1_integer_oracle_verdict_uses_absolute_deadline_and_pressure_bounds():
    metrics = {
        "network_capacity_pressure": 0.9,
        "prefill_capacity_pressure": 0.9,
        "absolute_p95_delay_over_deadline": 0.9,
        "absolute_deadline_miss_rate": 0.0,
    }

    assert _verdict(metrics, 0.99) == "Target shortfall"
    assert _verdict({**metrics, "network_capacity_pressure": 1.01}, 1.0) == "Network overload"
    assert _verdict({**metrics, "prefill_capacity_pressure": 1.01}, 1.0) == "Prefill overload"
    assert _verdict({**metrics, "absolute_p95_delay_over_deadline": 1.01}, 1.0) == "P95 deadline miss"
    assert _verdict({**metrics, "absolute_deadline_miss_rate": 0.02}, 1.0) == "Deadline miss rate"
