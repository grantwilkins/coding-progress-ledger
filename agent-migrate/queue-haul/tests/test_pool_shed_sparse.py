from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, issparse

import pool_shed_campaign as q


def dense_library(fleet, expanded):
    n, found = len(fleet.count), []
    def add(selected, replayed):
        r, k = np.zeros(n), np.zeros(n)
        np.add.at(r, replayed, 1)
        np.add.at(k, selected, 1)
        found.extend((np.r_[r, np.zeros(n)], np.r_[np.zeros(n), k - r]))
    for i in range(n):
        add([i], [])
        add([i], [i])
    keys = [fleet.context, -fleet.context, -fleet.gain / fleet.t1, -fleet.gain / fleet.kv]
    if expanded:
        keys += [fleet.gain / fleet.t1, fleet.gain / fleet.kv, fleet.demand, -fleet.demand]
    ratio = (fleet.kv - fleet.log) / fleet.t1
    for template in fleet.templates:
        for key in keys:
            order = sorted(template, key=lambda i: (key[i], i))
            for width in range(1, len(order) + 1):
                selected = order[:width]
                ranked = sorted(selected, key=lambda i: (-ratio[i], i))
                for sequence in (ranked, ranked[::-1]) if expanded else (ranked,):
                    for cut in range(width + 1):
                        add(selected, sequence[:cut])
    values = np.unique(found, axis=0)
    return np.split(values[values.sum(1) > 0], 2, axis=1)


@pytest.mark.parametrize("expanded", [False, True])
def test_sparse_candidates_preserve_dense_order_and_repeated_histories(expanded, monkeypatch):
    fleet = replace(q.compact_fleet(q.sample_fleet("coding"), [2, 4, 7, 9]), templates=[[0, 1, 1, 2, 3, 3]])
    expected = dense_library(fleet, expanded)
    for actual in (q.library(fleet, expanded), q.library(fleet, expanded, sparse=True)):
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.toarray() if issparse(a) else a, b)
    monkeypatch.setattr(csr_matrix, "toarray", lambda *a, **k: pytest.fail("sparse candidate construction densified"))
    r, k = q.library(fleet, expanded, sparse=True)
    r, k = q.include_isolated(r, k, np.array([False, True, False, True]))
    assert issparse(r) and issparse(k) and r.nnz + k.nnz <= 6 * r.shape[0]


def test_sparse_closure_handles_duplicates_zeros_and_fractional_counts():
    r = coo_matrix(([1., 2., 0., .5], ([0, 0, 1, 2], [7, 7, 4, 3])), shape=(4, 9))
    k = csr_matrix(([2., 1.], ([1, 2], [6, 2])), shape=r.shape)
    for a, b in zip(q.action_closure(r, k), q.action_closure(r.toarray(), k.toarray())):
        np.testing.assert_array_equal(a.toarray(), b)
    for a, b in zip(q.include_isolated(r.tocsr(), k, np.arange(9) % 2 == 0),
                    q.include_isolated(r.toarray(), k.toarray(), np.arange(9) % 2 == 0)):
        np.testing.assert_array_equal(a.toarray(), b)
    assert q.action_closure(csr_matrix((0, 9)), csr_matrix((0, 9)))[0].shape == (0, 9)
    with pytest.raises(ValueError, match="nonnegative"):
        q.action_closure(coo_matrix(([2., -1.], ([0, 0], [0, 0])), shape=(1, 2)), csr_matrix((1, 2)))


def test_compact_metadata_preserves_global_resources_and_source_identity():
    base = q.replica_fleet(q.sample_fleet("coding", gpus=8))
    fleet = replace(base, metadata={**base.metadata, "fixed_host_shares": True, "host_migration_gbps": 80.,
        "kv_shared_tokens": np.arange(24) * 16, "replay_cached_tokens": 0})
    ids = np.array([7, 2, 7])
    local = q.compact_fleet(fleet, ids)
    assert (local.gpus, local.kv_capacity, local.gpus_per_node) == (fleet.gpus, fleet.kv_capacity, 8)
    assert local.metadata["source_session_rps"] == fleet.metadata["source_session_rps"]
    assert local.metadata["turn_sequences"] == [fleet.metadata["turn_sequences"][i] for i in ids]
    np.testing.assert_equal(local.metadata["kv_shared_tokens"], [112, 32, 112])
    assert local.metadata["replay_cached_tokens"] == 0 and not local.templates
    np.testing.assert_array_equal(local.gain, fleet.gain[ids])
    local.context[0] += 1
    local.metadata["turn_offset"][0] += 1
    assert fleet.context[7] != local.context[0]
    assert fleet.metadata["turn_offset"][7] != local.metadata["turn_offset"][0]
    simple = SimpleNamespace(count=np.ones(3), gain=np.array([.1, .2, .3]), memory_tokens=np.array([16, 32, 48]), metadata={}, gpus=4)
    view = q.compact_fleet(simple, [2, 0])
    np.testing.assert_array_equal(view.gain, [.3, .1])
    np.testing.assert_array_equal(view.memory_tokens, [48, 16])
    assert view.gpus == 4
    for invalid in ([-1], [24], [1.5], [[1]]):
        with pytest.raises(ValueError, match="indices"):
            q.compact_fleet(fleet, invalid)


@pytest.mark.parametrize("causal", [False, True])
def test_local_phase_primitives_preserve_full_context_reset_and_host_caps(causal):
    fleet = q.sample_fleet("coding_long", gpus=8)
    if causal:
        fleet = q.replica_fleet(fleet)
        fleet = replace(fleet, metadata={**fleet.metadata, "fixed_host_shares": True, "host_migration_gbps": 80.})
    measured = {**q.calibration(0), "forecast_load": .5}
    timing = measured["timing"][0]
    for ids in ([2], [0, 1, 3, 4, 9, 12, 14, 21]):
        counts = np.zeros(len(fleet.count))
        counts[ids] = 1
        local = q.compact_fleet(fleet, ids)
        for action in (0, 1):
            for route in (0, 1):
                a = q.nominal_action(fleet, counts, action, route, 1e8, timing, measured)
                b = q.nominal_action(local, counts[ids], action, route, 1e8, timing, measured)
                assert a[0] == b[0]
                np.testing.assert_array_max_ulp(np.asarray(a[1:]), np.asarray(b[1:]), maxulp=4)


@pytest.mark.parametrize("causal", [False, True])
def test_sparse_table_preserves_candidate_physics_and_original_constraints(causal):
    fleet = q.sample_fleet("coding", gpus=8)
    if causal:
        fleet = q.replica_fleet(fleet)
        fleet = replace(fleet, metadata={**fleet.metadata, "fixed_host_shares": True, "host_migration_gbps": 80.})
    r, k = q.library(fleet)
    ids = np.unique(np.linspace(0, len(r) - 1, 16).astype(int))
    r, k = r[ids], k[ids]
    endpoint = q.network_samples()[0]
    args = (.5, 120., endpoint, q.bandwidth(endpoint, fleet.nodes, 1000), q.calibration(0)["timing"][0])
    dense, sparse = [q.schedule_table(fleet, a, b, *args) for a, b in ((r, k), (csr_matrix(r), csr_matrix(k)))]
    assert issparse(sparse.matrix) and issparse(sparse.replay) and issparse(sparse.kv)
    for key in ("replay", "kv", "route", "duration", "release", "kv_release", "log_bytes", "kv_bytes", "rate",
                "eligible", "fastest", "matrix", "capacities", "gains", "debt", "nominal_commit", "service_time"):
        a, b = getattr(dense, key), getattr(sparse, key)
        np.testing.assert_allclose(a, b.toarray() if issparse(b) else b, rtol=2e-15, atol=1e-12, err_msg=key)
    for policy in q.POLICIES:
        np.testing.assert_array_equal(q.policy_mask(dense, policy), q.policy_mask(sparse, policy))


def test_sparse_batch_rows_keep_unselected_long_contexts_out_of_packing():
    fleet = q.Fleet(np.ones(3), np.array([10., 20., 30.]), np.ones(3), np.ones(3), np.array([1., 3., 5.]),
                    np.ones(3), np.ones(3), np.full(3, .1), [], 8, 1e6,
                    {"packing_context_tokens": [10., 20.], "batch_context_limit": 25.})
    r = np.array([[1, 2, 0], [1, 0, 1], [0, 0, 0.]])
    k = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0.]])
    args = (.5, 30., np.full(3, 100.), np.full(3, 100.),
            {"kv_completion_s": 0., "kv_batch_completion_s": 0., "beta": 0., "packing_kappa": [.2, .8]})
    dense, sparse = [q.schedule_table(fleet, a, b, *args) for a, b in ((r, k), (csr_matrix(r), csr_matrix(k)))]
    np.testing.assert_array_equal(dense.duration, sparse.duration)
    np.testing.assert_allclose(sparse.duration, [5.8, 6., 0., 5.8, 6., 0.], rtol=2e-16)
    for policy in q.POLICIES:
        a, b = q.select(dense, policy), q.select(sparse, policy)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert q.certify(sparse, b)["patterns"] == q.certify(dense, b)["patterns"]
    with pytest.raises(ValueError, match="batch"):
        q.schedule_table(fleet, csr_matrix(r / 2), csr_matrix(k), *args)
    fleet.metadata["timing_load_factor"] = np.nan
    with pytest.raises(ValueError, match="batch"):
        q.schedule_table(fleet, csr_matrix(r), csr_matrix(k), *args)
