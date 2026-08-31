from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from planner_quality import evaluate_case, exact_selection
from pool_planner import Candidate, CandidateTable


def test_exact_aggregate_oracle_certifies_target_and_minimum_duration():
    candidates = tuple(
        Candidate(i, "replay", 0, gain, 1, duration, (), 0, (0, 0), 0)
        for i, (gain, duration) in enumerate(((6, 1), (5, 2), (5, 2)))
    )
    table = CandidateTable(
        tuple(SimpleNamespace(session_id=str(i)) for i in range(3)), candidates,
        csr_matrix(np.eye(3)), csr_matrix(np.array(((.6, .5, .5),))),
        ("route",), (1,), ("fraction",), 10,
    )

    maximum, duration = exact_selection(table, 10)

    assert maximum == pytest.approx(10)
    assert duration == pytest.approx(4)
    assert exact_selection(table, 11, maximum) == (pytest.approx(10), None)


def test_miss_only_balancing_closes_demonstrated_pool_choice_gap():
    row = evaluate_case(
        "agentic_tool_loop", 128, 1004, 300, .38, 1, "removable",
    )

    assert row["lp_power_hit"] and row["greedy_power_hit"]
    assert row["greedy_exact_max_ratio"] == pytest.approx(1)
