from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_network import history_bytes, phase_rate_cap, transport_workers


def fleet(**metadata):
    return SimpleNamespace(gpus=8, gpus_per_node=8, count=np.array([32., 32.]),
                           metadata={"fixed_host_shares": True, "resident_affinity": True, **metadata})


def test_worker_scaling_is_explicit_and_preserves_legacy_partial_hosts():
    f = fleet(destination_gpus=8)
    f.gpus, f.count = 10, np.array([40., 40.])
    assert transport_workers(f).tolist() == [8, 8, 10]
    assert 16 * phase_rate_cap(f, [1, 0], [1, 1]) == 20e9 / 8
    f.metadata.pop("fixed_host_shares")
    assert transport_workers(f).tolist() == [1, 1, 2]
    assert phase_rate_cap(f, [1, 1], [9, 1]) == np.inf
    f.metadata["destination_gpus"] = 8.5
    with pytest.raises(ValueError):
        transport_workers(f)


def test_largest_individual_history_limits_a_mixed_phase_without_idle_borrowing():
    f = fleet()
    cap = phase_rate_cap(f, [1, 1], [9, 1])
    assert cap == pytest.approx(80e9 / 8 / 64 * 10 / 9)
    assert cap * 9 / 10 == pytest.approx(80e9 / 8 / 64)
    assert phase_rate_cap(f, [8, 0], [9, 1000]) == 80e9 / 8 / 8
    assert phase_rate_cap(f, [0, 0], [0, 0]) == 80e9 / 8 / 8
    assert phase_rate_cap(fleet(host_migration_gbps=40), [1, 1], [9, 1]) == cap / 2


def test_source_shares_bound_both_routes_and_destination_shares_bound_eight_gpus():
    f = fleet(destination_gpus=64)
    f.count = np.ones(64)
    rates = np.array([phase_rate_cap(f, c, np.arange(1, 65)) for c in np.eye(64)])
    assert rates[:32].sum() + rates[32:].sum() == 80e9 / 8
    f.gpus, f.metadata["destination_gpus"] = 64, 8
    packs = np.eye(8).repeat(8, axis=1)
    assert sum(phase_rate_cap(f, c, np.ones(64)) for c in packs) == 80e9 / 8


@pytest.mark.parametrize("change", [
    {"metadata": {"fixed_host_shares": "yes"}},
    {"metadata": {"resident_affinity": False}},
    {"metadata": {"host_migration_gbps": 0}},
    {"metadata": {"host_migration_gbps": np.inf}},
    {"metadata": {"destination_gpus": 0}},
    {"gpus_per_node": 1}, {"gpus": 8.5}, {"count": np.array([65.])},
])
def test_fixed_share_contract_rejects_unsupported_configurations(change):
    f = fleet()
    for key, value in change.items():
        f.metadata.update(value) if key == "metadata" else setattr(f, key, value)
    with pytest.raises(ValueError):
        transport_workers(f)


@pytest.mark.parametrize("counts,volume", [([1], [1]), ([1, 0], [1, np.nan]),
                                          ([1, 0], [1, -1]), ([.5, 0], [1, 1])])
def test_phase_cap_rejects_invalid_history_vectors(counts, volume):
    with pytest.raises(ValueError):
        phase_rate_cap(fleet(), counts, volume)


def scenario():
    from pool_shed_campaign import Fleet

    f = Fleet(np.array([8.]), np.array([16.]), np.array([0.]), np.array([0.]), np.array([10.]),
              np.array([16000.]), np.array([32.]), np.array([.01]), [[0]], 8, 1e6,
              {"fixed_host_shares": True, "resident_affinity": True, "host_migration_gbps": 6400 * 8e-9,
               "turn_sequences": [[]], "source_session_rps": 0.}, 8)
    timing = {"beta": 0., "kappa": 1., "kv_completion_s": 1., "kv_batch_completion_s": 1.,
              "resident_replay_loss": 1., "regional_kv_bytes_per_s": np.array([1e4, 1e4])}
    measured = {"kv_block_tokens": 16, "kv_block_bytes": 16000., "kv_tail_replay_tps": 100.,
                "switch_s": 0., "F": 100., "G": 10., "forecast_load": .5,
                "replay_context_tokens": [1., 1000.], "replay_tps": [100., 100.], "replay_completion_s": 0.}
    return f, timing, measured


def test_fixed_shares_reprice_singleton_fastest(monkeypatch):
    import pool_shed_campaign as campaign

    f, timing, measured = scenario()
    monkeypatch.setattr(campaign, "calibration", lambda _: measured)
    endpoint = np.array([1e4, 1e4, 2e4])
    assert campaign.isolated_methods(f, .5, endpoint, endpoint * 8, timing).tolist() == [True]
    assert campaign.nominal_action(f, np.ones(1), 1, 0, 1e4, timing, measured)[2] == pytest.approx(161.)
    f.metadata.pop("fixed_host_shares")
    assert campaign.isolated_methods(f, .5, endpoint, endpoint * 8, timing).tolist() == [False]


def test_nominal_and_temporal_plans_reprice_reset_delta_separately(monkeypatch):
    from dataclasses import replace
    import pool_shed_campaign as campaign
    from pool_shed_execution import PooledExecution
    from pool_shed_planner import phase_profile

    f, timing, measured = scenario()
    measured["kv_block_bytes"] = 16.
    f = replace(f, count=np.array([4., 4.]), context=np.array([160., 32.]), kv=np.array([160., 32.]),
                log=np.array([320., 64.]), t1=np.ones(2), demand=np.array([.01, .01]),
                prompt=np.zeros(2), output=np.zeros(2), templates=[[0, 1]],
                metadata={**f.metadata, "source_session_rps": 10., "turn_work_s": [[0.], [.16]],
                          "turn_sequences": [[{"context": 16, "prompt": 0, "output": 0, "reset": True}],
                                             [{"context": 32, "prompt": 16, "output": 0}]]})
    monkeypatch.setattr(campaign, "calibration", lambda _: measured)
    endpoint = np.array([1e4, 1e4, 2e4])
    table = campaign.schedule_table(f, np.zeros((1, 2)), np.ones((1, 2)), .5, 10., endpoint, endpoint * 8, timing)
    profile = phase_profile(table, np.ones(2), 1, 0, 0., np.arange(11.), np.full((2, 10), .5), timing, measured)
    # Initial 192 B / 120 B/s, reset delta 32 B / 200 B/s, then the measured 1 s endpoint phase.
    assert profile["finish"] == pytest.approx(2.76)
    assert profile["network"].sum() == pytest.approx(224.)
    assert table.nominal_commit == pytest.approx([2.76, 2.76])
    execution = PooledExecution(table, timing, measured)
    execution.admit([1., 0.])
    execution.advance(10.)
    assert execution.result()["last_completion_s"] == pytest.approx(profile["finish"])
    assert execution.result()["transferred_bytes"] == pytest.approx([224., 0., 224.])
    assert history_bytes(f, [16, 48], 1, measured, f.context, [True, False]).tolist() == [16., 16.]
    assert history_bytes(f, [16, 32], 0, measured, f.context, [True, False]).tolist() == [32., 0.]


@pytest.mark.parametrize("context,origin,reset", [([-1], None, None), ([np.nan], None, None),
    ([16], [-1], [False]), ([16], [16], [1]), ([16], [16], None)])
def test_history_bytes_rejects_invalid_context_or_reset(context, origin, reset):
    f, _, measured = scenario()
    with pytest.raises(ValueError):
        history_bytes(f, context, 1, measured, origin, reset)


@pytest.mark.parametrize("action,value", [(0, -1.), (1, np.nan)])
def test_history_bytes_rejects_invalid_frozen_payload(action, value):
    f, _, measured = scenario()
    (f.kv if action else f.log)[:] = value
    with pytest.raises(ValueError):
        history_bytes(f, f.context, action, measured)
