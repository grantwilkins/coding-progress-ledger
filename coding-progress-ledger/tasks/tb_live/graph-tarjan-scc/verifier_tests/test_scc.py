import pytest
from tarjan_scc import scc


def test_empty_graph():
    assert scc({}) == []


def test_single_node_no_edges():
    assert scc({0: []}) == [[0]]


def test_self_loop():
    assert scc({0: [0]}) == [[0]]


def test_two_node_cycle():
    assert scc({0: [1], 1: [0]}) == [[0, 1]]


def test_linear_chain_dag():
    assert scc({0: [1], 1: [2], 2: []}) == [[0], [1], [2]]


def test_two_separate_sccs():
    assert scc({0: [1], 1: [0], 2: [3], 3: [2]}) == [[0, 1], [2, 3]]


def test_mixed_scc_and_dag():
    adj = {0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [3]}
    assert scc(adj) == [[0, 1, 2], [3, 4]]


def test_disconnected_with_isolated():
    assert scc({0: [1], 1: [0], 5: []}) == [[0, 1], [5]]
