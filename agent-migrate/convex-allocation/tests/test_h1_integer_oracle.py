"""
Claim:
The H1 integer oracle is a tiny exhaustive discrete feasibility check that uses
the same movement, resource-pressure, and absolute-deadline pass criteria as the
fixed-target H1 stress table, alongside the same H1 allocation methods and per
method runtimes.

Plausible wrong implementations:
- Optimize or report a fractional allocation instead of enumerating integer ones.
- Use release-relative deadline metrics instead of absolute event-start metrics.
- Let a resource or deadline violation report as a pass.
- Drop one of the H1 methods or report only the oracle row.
- Report runtime for only the oracle or omit solve time from method rows.
- Grow the oracle into a large sweep that is no longer a bounded sanity check.
"""

from __future__ import annotations

import numpy as np

from experiments.run_h1_integer_oracle import (
    DRAIN_WINDOW_S,
    RELEASE_POLICY,
    RETAINED_PREFILL_FRACTION,
    _verdict,
    h1_integer_rows,
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
    assert row["runtime_s"] >= 0.0


def test_h1_integer_table_includes_oracle_and_h1_methods_with_runtime():
    rows = h1_integer_rows(make_h1_integer_problem())
    by_policy = {row["policy"]: row for row in rows}

    assert list(by_policy) == [
        "Integer feasibility oracle",
        "Deadline-aware",
        "Online queue",
        "Least loaded",
        "Replay only",
        "State only",
    ]
    assert by_policy["Integer feasibility oracle"]["enumerated_allocations"] > 0
    for policy, row in by_policy.items():
        assert row["target_moved_fraction"] >= 1.0
        assert row["runtime_s"] >= 0.0
        if policy != "Integer feasibility oracle":
            assert row["enumerated_allocations"] == ""
    assert any(row["verdict"] != "Pass" for row in rows)


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
