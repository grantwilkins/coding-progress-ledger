"""
Claim:
The timeline accumulates physical per-session work in measured completion order
and normalizes each resource exactly once by the deadline-aware capacity.

Plausible wrong implementations:
- Sort by session or request start instead of completion time.
- Normalize each increment twice or use one resource's capacity for another.
- Drop an action/destination contribution while accumulating a session.
- Mix nanoseconds and seconds or silently accept unmatched evidence.
"""

import pytest

from plot_hardware_constraint_timeline import cumulative_resource_timeline


def test_cumulative_resource_timeline_orders_and_conserves_physical_work():
    rows = cumulative_resource_timeline(
        {"late": 2, "early": 1},
        {"late": {"kv": 1, "service": 1},
         "early": {"kv": 1, "service": 3}},
        {"kv": 2, "service": 4},
    )

    assert rows == [
        (0, {"kv": 0, "service": 0}),
        (1, {"kv": .5, "service": .75}),
        (2, {"kv": 1, "service": 1}),
    ]
    with pytest.raises(ValueError):
        cumulative_resource_timeline({"a": 1}, {}, {"kv": 1})
