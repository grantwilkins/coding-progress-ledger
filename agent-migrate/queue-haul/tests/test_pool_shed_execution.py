from types import SimpleNamespace

import numpy as np
import pytest

from pool_shed_execution import _buffered, _quiesce, execute_pooled, flow_rates


def case(context=(100., 100.), replay=((1., 0.), (0., 0.)), kv=((0., 0.), (0., 1.)),
         work=(8., 1.), deadline=10., gpus=1, load=0., demand=(.4, .4), route=(0, 1)):
    context = np.array(context)
    fleet = SimpleNamespace(count=np.ones(len(context)), context=context, demand=np.array(demand),
                            memory_tokens=context, baseline_kv=0., kv_capacity=1e8, gpus=gpus,
                            gain=np.ones(len(context)) / len(context), t1=np.array(work),
                            log=np.array([2., 2.]), kv=np.array([100., 98.]),
                            metadata={"turn_sequences": [[] for _ in context], "source_session_rps": 0.})
    table = SimpleNamespace(fleet=fleet, replay=np.array(replay), kv=np.array(kv), route=np.array(route),
                            load=load, deadline=deadline, endpoint=np.array([10., 10.]), budgets=np.array([10., 10., 10.]))
    calibration = {"kv_block_tokens": 1, "kv_block_bytes": 1., "kv_tail_replay_tps": 100., "switch_s": 0.}
    timing = {"kappa": 1., "beta": 0., "kv_completion_s": 0., "kv_batch_completion_s": 0.}
    return table, timing, calibration


def run(table, timing, calibration, mass=None):
    return execute_pooled(table, np.ones(len(table.route)) if mass is None else mass, timing, calibration)


def test_released_network_capacity_finishes_feasible_mixed_schedule():
    table, timing, calibration = case()
    result = run(table, timing, calibration)
    assert result["shed_fraction"] == 1
    assert result["last_completion_s"] == pytest.approx(10)
    assert result["transferred_bytes"] == pytest.approx([2, 98, 100])


def test_independent_execution_does_not_read_planning_certificate():
    table, timing, calibration = case()
    table.matrix, table.duration, table.rate, table.eligible = None, None, None, None
    assert run(table, timing, calibration)["shed_fraction"] == 1


def test_flow_sharing_preserves_route_endpoint_and_shared_limits():
    rates = flow_rates(np.array([2., 1., 3.]), np.array([0, 0, 1]), np.array([4., 9.]), np.array([7., 20., 16.]))
    assert rates == pytest.approx([7 / 3, 7 / 3, 3])


def test_incoming_serving_slows_unfinished_replay():
    table, timing, calibration = case(deadline=30., route=(0, 0), demand=(.4, .4))
    table.fleet.kv[1] = 1.
    timing["beta"] = 1.
    busy = run(table, timing, calibration)
    table.fleet.demand[:] = 0
    idle = run(table, timing, calibration)
    assert busy["last_completion_s"] > idle["last_completion_s"] + 2
    assert busy["peak_destination_load"] == pytest.approx([.8, 0])


def test_batch_mass_reuses_compute_instead_of_reserving_replica_for_deadline():
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      work=(1., 3.), route=(0, 0), deadline=5.)
    result = run(table, timing, calibration)
    assert result["shed_fraction"] == 1
    assert result["last_completion_s"] == pytest.approx(4.4)
    assert result["batch_replica_seconds"] == pytest.approx([4, 0])


def test_partial_block_requires_tail_compute_without_ingestion():
    table, timing, calibration = case(context=(103., 100.), replay=((0., 0.),), kv=((1., 0.),), route=(0,))
    calibration["kv_block_tokens"] = 10
    calibration["kv_block_bytes"] = 10
    table.deadline = 11
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(10.03)
    assert result["batch_replica_seconds"][0] == pytest.approx(.03)


def test_source_quiesce_uses_complete_recorded_turns_and_reset():
    table, timing, calibration = case()
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 100, "prompt": 10, "output": 5},
        {"context": 0, "prompt": 8, "output": 2, "reset": True}], []])
    time, context, reset, exhausted = _quiesce(table.fleet, np.array([1, 0]), 1.2)
    assert time == 2
    assert context[0] == 10 and reset[0] and exhausted
    assert _quiesce(table.fleet, np.array([1, 0]), 100)[1][0] == 10


def test_cyclic_trace_wrap_invalidates_snapshot_and_does_not_repeat_sampled_turn():
    table, _, _ = case()
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 0, "prompt": 10, "output": 5},
        {"context": 15, "prompt": 8, "output": 2}], []], sequence_cycle=True, turn_offset=[1, 0])
    end, context, reset, exhausted = _quiesce(table.fleet, np.array([1, 0]), 2.)
    assert end == 2 and context[0] == 15 and reset[0] and not exhausted


def test_same_plan_completions_are_monotone_with_longer_horizon():
    table, timing, calibration = case()
    values = []
    for deadline in (1., 5., 9., 10., 20.):
        table.deadline = deadline
        values.append(run(table, timing, calibration)["shed_fraction"])
    assert values == sorted(values)


def test_missing_source_contract_fails():
    table, timing, calibration = case()
    del table.fleet.metadata["turn_sequences"]
    with pytest.raises(ValueError, match="source turn sequences"):
        run(table, timing, calibration)


def test_regional_draw_changes_execution_instead_of_using_central_values():
    table, timing, calibration = case(deadline=30.)
    central = run(table, timing, calibration)
    timing["regional_kv_bytes_per_s"] = [5., 5.]
    assert run(table, timing, calibration)["last_completion_s"] > central["last_completion_s"] + 9


def test_pending_migration_memory_is_reserved_before_another_handoff():
    table, timing, calibration = case(context=(40., 40.), replay=((0., 0.), (0., 0.)),
                                      kv=((1., 0.), (0., 1.)), route=(0, 0), deadline=30.)
    table.fleet.kv[:] = [1, 40]
    table.fleet.kv_capacity = 100
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 40, "prompt": 30, "output": 0}], []])
    result = run(table, timing, calibration)
    assert result["memory_blocked_batch_mass"] == 1
    assert result["shed_fraction"] == .5
    assert max(result["peak_reserved_kv_tokens"]) <= 100


def test_buffered_requests_conserve_mass_and_compete_with_other_migrations():
    table, timing, calibration = case(replay=((0., 0.), (0., 1.)), kv=((1., 0.), (0., 0.)),
                                      work=(1., 8.), route=(0, 0), deadline=30.)
    table.fleet.kv[0] = 1
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 100 + i, "prompt": 1, "output": 0} for i in range(20)], []])
    timing["kv_completion_s"] = 2
    timing["beta"] = 1
    calibration.update(F=1., G=1.)
    busy = run(table, timing, calibration)
    calibration["F"] = 1e6
    idle = run(table, timing, calibration)
    assert busy["buffered_requests"] > 0
    assert busy["buffered_requests"] == pytest.approx(busy["completed_buffered_requests"] + busy["pending_buffered_requests"])
    assert busy["last_completion_s"] > idle["last_completion_s"] + 1


def test_full_reset_replay_uses_rebuild_curve_not_partial_block_rate():
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,), deadline=10.)
    table.fleet.t1[0] = .1
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 0, "prompt": 1000, "output": 0, "reset": True}], []])
    calibration.update(replay_context_tokens=[100., 1000.], replay_tps=[1000., 1000.],
                       replay_completion_s=0., kv_tail_replay_tps=1., F=1000., G=1000.)
    table.endpoint[:], table.budgets[:] = 1e6, 1e6
    result = run(table, timing, calibration)
    assert result["shed_fraction"] == .5
    assert result["last_completion_s"] == pytest.approx(2.002)


def test_buffered_arrivals_are_counted_at_turn_start_not_completion():
    table, _, calibration = case()
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 0, "prompt": 1, "output": 0},
        {"context": 1, "prompt": 7, "output": 0},
        {"context": 8, "prompt": 9, "output": 0}], []])
    calibration.update(F=1., G=1.)
    assert _buffered(table.fleet, np.array([1, 0]), 1., 1.9, calibration) == (1, 7)
    assert _buffered(table.fleet, np.array([1, 0]), 1., 2., calibration) == (1, 7)
    assert _buffered(table.fleet, np.array([1, 0]), 1., 2.1, calibration) == (2, 16)


def test_buffered_serving_cannot_drain_without_reference_headroom():
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,), load=.5, demand=(.5, 0.), deadline=30.)
    table.fleet.kv[0] = 1
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 100 + i, "prompt": 1, "output": 0} for i in range(20)], []])
    timing["kv_completion_s"] = 2
    calibration.update(F=1., G=1.)
    result = run(table, timing, calibration)
    assert result["peak_destination_load"][0] == 1
    assert result["buffered_requests"] > 0
    assert result["completed_buffered_requests"] == 0
    assert result["pending_buffered_requests"] == result["buffered_requests"]
