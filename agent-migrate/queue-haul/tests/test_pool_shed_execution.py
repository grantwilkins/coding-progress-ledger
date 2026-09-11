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


def test_cache_sensitivity_preserves_context_rate_and_charges_new_private_tokens():
    from pool_shed_calibration import replay_seconds
    from pool_shed_execution import catchup, initial_work, kv_transfer_bytes

    table, timing, calibration = case()
    calibration.update(replay_context_tokens=[100, 200], replay_tps=[10, 5], replay_completion_s=2.,
                       kv_block_tokens=10, kv_block_bytes=10)
    fleet, counts = table.fleet, np.array([1., 0.])
    fleet.metadata.update(replay_cached_tokens=[50, 50], kv_shared_tokens=[50, 50], kv_wire_scale=.5)
    fleet.t1 = replay_seconds(fleet.context, calibration, fleet.metadata["replay_cached_tokens"])
    fleet.kv = kv_transfer_bytes(fleet, fleet.context, calibration)
    assert initial_work(fleet, counts, 0, 0, None, timing, calibration) == (2., 7.)
    assert initial_work(fleet, counts, 0, 0, np.array([200., 100.]), timing, calibration) == (400., 32.)
    assert initial_work(fleet, counts, 1, 0, None, timing, calibration) == (25., 0.)
    assert initial_work(fleet, counts, 1, 0, np.array([203., 100.]), timing, calibration) == (75., 0.)
    assert catchup(fleet, counts, 1, 0, np.array([203., 100.]), np.array([False, False]), timing, calibration) == pytest.approx((50., .03))
    assert catchup(fleet, counts, 0, 0, np.array([200., 100.]), np.array([False, False]), timing, calibration) == (400., 32.)
    assert catchup(fleet, counts, 0, 0, fleet.context, np.array([True, False]), timing, calibration) == (200., 7.)
    assert catchup(fleet, counts, 1, 0, fleet.context, np.array([True, False]), timing, calibration) == (25., 0.)
    assert kv_transfer_bytes(fleet, [30, 40], calibration).tolist() == [0., 0.]
    assert fleet.context.tolist() == [100., 100.]
    fleet.metadata["replay_cached_tokens"] = [0., 0.]
    reset = np.array([True, False])
    low = np.array([20., 100.])
    baseline = catchup(fleet, counts, 0, 0, low, reset, timing, calibration)[1]
    fleet.metadata["replay_cached_tokens"] = [1e-12, 0.]
    assert catchup(fleet, counts, 0, 0, low, reset, timing, calibration)[1] == pytest.approx(baseline)
    fleet.metadata["replay_cached_tokens"] = [20., 0.]
    assert catchup(fleet, counts, 0, 0, low, reset, timing, calibration)[1] == pytest.approx(2.)


def test_reported_32k_wire_anchor_and_invalid_cache_assumptions():
    from pool_shed_calibration import replay_seconds
    from pool_shed_execution import kv_transfer_bytes

    table, _, calibration = case()
    calibration.update(kv_block_tokens=256, kv_block_bytes=12582912,
                       replay_context_tokens=[100], replay_tps=[10], replay_completion_s=2.)
    table.fleet.metadata["kv_wire_scale"] = 800_000_000 / (32768 * 49152)
    assert kv_transfer_bytes(table.fleet, [32768], calibration) == pytest.approx([800_000_000])
    assert replay_seconds([100], calibration, [100]) == pytest.approx([2.])
    for invalid in (-1., np.nan):
        with pytest.raises(ValueError, match="cached tokens"):
            replay_seconds([100], calibration, invalid)
        table.fleet.metadata["kv_wire_scale"] = invalid
        with pytest.raises(ValueError, match="KV wire"):
            kv_transfer_bytes(table.fleet, [32768], calibration)


def test_replay_catchup_charges_full_context_overhead_and_long_context_packing():
    from pool_shed_execution import catchup
    table, timing, calibration = case()
    calibration.update(replay_context_tokens=[100, 200], replay_tps=[10, 5], replay_completion_s=2.)
    fleet, counts, reset = table.fleet, np.array([8., 0.]), np.zeros(2, bool)
    timing["kappa"] = .25
    fleet.metadata["batch_context_limit"] = 150
    short = catchup(fleet, counts, 0, 0, np.array([101., 100.]), reset, timing, calibration)
    long = catchup(fleet, counts, 0, 0, np.array([191., 100.]), reset, timing, calibration, origin_context=np.array([190., 100.]))
    assert short == pytest.approx((8 * 202, (8 * .25 + .75) * (101 / 9.95 + 2)))
    assert long == pytest.approx((8 * 382, 8 * (191 / 5.45 + 2)))
    assert catchup(fleet, counts, 0, 0, fleet.context, reset, timing, calibration) == (0., 0.)
    with pytest.raises(ValueError, match="measured context support"):
        catchup(fleet, counts, 0, 0, np.array([201., 100.]), reset, timing, calibration)


def test_staggered_quiescence_counts_arrivals_before_the_batch_barrier():
    from pool_shed_execution import source_snapshot
    table, _, calibration = case()
    table.fleet.metadata.update(paced_source=True, protect_resident=False, source_session_rps=1.,
        sequence_cycle=True, source_phase_s=[.8, .1], turn_duration_s=[[.8], [.05]], turn_work_s=[[2.], [3.]],
        turn_sequences=[[{"context": 100, "prompt": 1, "output": 0}]] * 2)
    context, turns = source_snapshot(table.fleet, 0.)
    assert context.tolist() == [101, 101] and turns.tolist() == [1, 1]
    end, _, _, _ = _quiesce(table.fleet, np.ones(2), .3)
    assert end == pytest.approx(1.)
    assert _buffered(table.fleet, np.ones(2), .3, end, calibration, quiescing=True) == (1, 3)
    assert _buffered(table.fleet, np.ones(2), .3, .9, calibration, quiescing=True) == (0, 0)
    table.fleet.metadata["source_phase_s"] = [0., 1.]
    with pytest.raises(ValueError, match="source phases"):
        source_snapshot(table.fleet, 0.)


def test_smaller_destinations_reduce_service_memory_and_endpoints_without_resizing_source():
    from pool_shed_execution import PooledExecution, network_nodes
    table, timing, calibration = case(gpus=4, load=.5, demand=(1.1, 0.))
    table.fleet.gpus_per_node = 2
    table.fleet.metadata["destination_gpus"] = 2
    execution = PooledExecution(table, timing, calibration)
    assert execution.gpus == 2 and table.fleet.gpus == 4
    assert execution.free_memory == table.fleet.kv_capacity / 2
    assert network_nodes(table.fleet).tolist() == [1, 1, 2]
    with pytest.raises(ValueError, match="destination serving or memory capacity"):
        execution.admit([1., 0.])


def test_released_network_capacity_finishes_feasible_mixed_schedule():
    table, timing, calibration = case()
    result = run(table, timing, calibration)
    assert result["shed_fraction"] == 1
    assert result["last_completion_s"] == pytest.approx(10)
    assert result["transferred_bytes"] == pytest.approx([2, 98, 100])


@pytest.mark.parametrize("gpus", [1, 2])
def test_pool_spare_capacity_does_not_erase_reported_local_resident_displacement(gpus):
    table, timing, calibration = case(gpus=gpus, load=.5, demand=(0., 0.))
    timing["resident_replay_loss"] = .9
    result = run(table, timing, calibration, mass=[1., 0.])
    assert result["resident_displaced_work_s"] == pytest.approx([.5 * .9 * 8, 0.])
    assert result["resident_pool_compensation_work_s"] == pytest.approx([3.6 if gpus == 2 else 0., 0.])
    assert result["resident_debt_generated_work_s"] == pytest.approx([0. if gpus == 2 else 3.6, 0.])
    assert result["peak_migration_replicas"] == [1., 0.]
    assert not result["resident_latency_validated"]
    assert "no resident GPU affinity" in result["service_recovery_scope"]


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


def test_replay_resident_debt_accumulates_and_recovers_after_handoff():
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), deadline=30.)
    timing["resident_replay_loss"] = .9
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(8.2)
    assert result["resident_debt_generated_work_s"] == pytest.approx([1.8, 0])
    assert result["resident_debt_recovered_work_s"] == pytest.approx([1.8, 0])
    assert result["pending_resident_debt_work_s"] == pytest.approx([0, 0])
    assert result["service_ready_s"] == pytest.approx(10.6)


def test_handoff_before_deadline_does_not_imply_resident_debt_is_cleared():
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), deadline=9.)
    timing["resident_replay_loss"] = .9
    result = run(table, timing, calibration)
    assert result["shed_fraction"] == .5
    assert result["service_ready_s"] is None
    assert result["pending_resident_debt_work_s"] == pytest.approx([1.2, 0])
    assert np.array(result["resident_debt_generated_work_s"]) == pytest.approx(
        np.array(result["resident_debt_recovered_work_s"]) + result["pending_resident_debt_work_s"])


def test_kv_network_wait_does_not_consume_resident_service():
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), deadline=5.)
    timing.update(resident_replay_loss=.9, kv_completion_s=2.)
    assert run(table, timing, calibration)["resident_debt_generated_work_s"] == [0, 0]
    table.deadline = 30
    result = run(table, timing, calibration)
    assert result["resident_debt_generated_work_s"] == pytest.approx([.5, 0])
    assert result["service_ready_s"] == pytest.approx(12 + 2 / 3)


def test_nonmigrating_replica_capacity_absorbs_displaced_resident_work():
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      gpus=2, load=.25, demand=(0., 0.), deadline=30.)
    timing["resident_replay_loss"] = .9
    result = run(table, timing, calibration)
    assert result["resident_debt_generated_work_s"] == [0, 0]
    assert result["service_ready_s"] == result["last_completion_s"]


@pytest.mark.parametrize("gpus", [1, 10])
def test_affine_debt_recovers_only_on_its_replica_after_persistent_incoming_load(gpus):
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      gpus=gpus, load=.5, demand=(.25, 0.), work=(4., 1.), deadline=10.)
    table.fleet.metadata["resident_affinity"] = True
    timing["resident_replay_loss"] = 1.
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(4.2)
    assert result["resident_debt_generated_work_s"] == pytest.approx([2., 0.])
    assert result["pending_resident_debt_work_s"] == pytest.approx([.55, 0.])
    assert result["resident_pool_compensation_work_s"] == [0., 0.]
    assert result["shed_fraction"] == .5 and result["recovered_handoff_fraction"] == 0.
    assert result["resident_latency_validated"] is False


@pytest.mark.parametrize("failure", ["serving", "memory", "replicas"])
def test_affine_admission_enforces_local_and_permanent_replica_capacity(failure):
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      route=(0, 0), gpus=10, load=.5, demand=(.1, .1), work=(1., 1.))
    table.fleet.metadata["resident_affinity"] = True
    timing["resident_replay_loss"] = 1.
    if failure == "serving":
        table.fleet.demand[0] = .6
    elif failure == "memory":
        table.fleet.kv_capacity = 500.
    else:
        table.fleet.metadata["destination_gpus"] = 1
    engine = PooledExecution(table, timing, calibration)
    if failure == "replicas":
        engine.admit([1., 0.])
        engine.advance(10.)
        assert engine.state[0] == 6
    with pytest.raises(ValueError, match="replica"):
        engine.admit([0., 1.] if failure == "replicas" else [1., 0.])


def test_affine_resident_priority_keeps_buffer_recovery_on_same_gpu_and_clone_independent():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      gpus=10, load=.25, demand=(.25, 0.), deadline=5.)
    table.fleet.metadata["resident_affinity"] = True
    timing["resident_replay_loss"] = 1.
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.state[:], engine.remaining[:], engine.origin_time[:] = 6, 0., 0.
    engine.replica_debt[:], engine.resident_debt[0], engine.backlog[:] = .5, .5, 1.
    engine.loads[0] += .25 / 10
    clone = engine.nominal_continuation(table, timing, calibration)
    clone.advance(1.)
    assert clone.replica_debt[0] == pytest.approx(0.) and clone.backlog[0] == 1.
    assert engine.replica_debt[0] == .5 and engine.backlog[0] == 1.
    clone.advance(3.)
    assert clone.backlog[0] == pytest.approx(0.)


@pytest.mark.parametrize("chunks", [1, 4])
def test_affine_compute_load_is_local_and_interrupted_execution_is_identical(chunks):
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      route=(0, 0), gpus=2, load=.25, demand=(.25, .25), work=(1., 8.), deadline=20.)
    table.fleet.metadata["resident_affinity"] = True
    timing.update(resident_replay_loss=1., beta=1.)
    expected = execute_pooled(table, np.ones(2), timing, calibration, chunks)
    assert expected["last_completion_s"] == pytest.approx(.4 + 8 * np.exp(.25))
    actual = PooledExecution(table, timing, calibration, chunks)
    actual.admit(np.ones(2))
    for until in (.1, 1., 2., 8., 15., 20.):
        actual.advance(until)
    for key in ("last_completion_s", "service_ready_s", "resident_debt_generated_work_s", "pending_resident_debt_work_s",
                "recovered_handoff_fraction", "recovered_action_fractions"):
        assert actual.result()[key] == pytest.approx(expected[key])
    assert actual.result()["recovered_handoff_fraction"] == pytest.approx(1.)


def test_causal_source_snapshot_quiescence_and_buffer_counts_use_actual_long_turn():
    from pool_shed_execution import source_snapshot
    table, _, calibration = case()
    fleet = table.fleet
    fleet.metadata.update(paced_source=True, causal_source=True, source_session_rps=1., turn_sequences=[[
        {"context": 100, "prompt": 1, "output": 3}, {"context": 104, "prompt": 1, "output": 0}], []],
        turn_duration_s=[[4.5, .5], []], turn_work_s=[[.2, .1], []])
    context, completed = source_snapshot(fleet, 3.)
    assert context[0] == 100 and completed[0] == 0
    end, context, _, _ = _quiesce(fleet, np.array([1., 0.]), 3.)
    assert end == 4.5 and context[0] == 104
    assert _buffered(fleet, np.array([1., 0.]), 3., 5.5, calibration, quiescing=True) == pytest.approx((1., .1))
    context, completed = source_snapshot(fleet, 5.)
    assert context[0] == 105 and completed[0] == 2


def test_resident_debt_recovers_before_migrated_source_buffers():
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), deadline=3.3)
    table.fleet.kv[0] = 1
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 100 + i, "prompt": 1, "output": 0} for i in range(20)], []])
    timing.update(resident_replay_loss=.9, kv_completion_s=2.)
    calibration.update(F=1., G=1.)
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(3.1)
    assert result["pending_resident_debt_work_s"] == pytest.approx([.35, 0])
    assert result["pending_destination_buffered_requests"] == result["transferred_buffered_requests"]


def test_admission_roundoff_does_not_generate_fictitious_resident_debt():
    table, timing, calibration = case(replay=((0., 0.), (0., 0.)), kv=((1., 0.), (0., 1.)),
                                      gpus=66666, load=.5, demand=(33333. + 1e-8, 0.), deadline=3600.)
    table.fleet.kv[0] = 1
    timing["resident_replay_loss"] = .9
    result = run(table, timing, calibration)
    assert result["final_destination_load"] == [1., .5]
    assert result["pending_resident_debt_work_s"] == [0., 0.]
    assert result["service_ready_s"] == result["last_completion_s"]


def test_uncommitted_source_buffer_work_is_separate_from_destination_backlog():
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      demand=(0., 0.), deadline=2.5)
    table.fleet.kv[0] = 1
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 100 + i, "prompt": 1, "output": 0} for i in range(20)], []])
    timing["kv_completion_s"] = 2
    calibration.update(F=1., G=1.)
    result = run(table, timing, calibration)
    assert result["completed_sessions"] == 0
    assert result["pending_source_buffer_work_s"] == 2
    assert result["pending_backlog_reference_work_s"] == 0
    assert result["pending_buffered_work_s"] == 2


def test_pipelined_kv_dispatch_matches_fifo_without_whole_population_barrier():
    table, timing, calibration = case(context=(1., 100.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), gpus=100, demand=(0., 0.), deadline=10.)
    table.fleet.count[:] = [100, 0]
    table.fleet.gain[:] = [.01, 0]
    table.fleet.kv[0] = 1
    timing["kv_completion_s"] = .1
    result = execute_pooled(table, np.array([100.]), timing, calibration, chunks=100)
    assert result["completed_sessions"] == pytest.approx(99)
    assert result["shed_fraction"] == pytest.approx(.99)
    assert result["transferred_bytes"] == pytest.approx([100, 0, 100])
    assert len(result["completion_events"]) == 1
    assert result["completion_events"][0]["multiplicity"] == 99
    assert result["completion_events"][0]["first_completion_s"] == pytest.approx(.2)
    assert execute_pooled(table, np.array([100.]), timing, calibration, chunks=1)["completed_sessions"] == 0


@pytest.mark.parametrize("chunks", [64, 128, 256, 512])
def test_finer_dispatch_keeps_enough_endpoint_concurrency_to_fill_wan(chunks):
    table, timing, calibration = case(context=(1., 100.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), gpus=100, demand=(0., 0.), deadline=10.)
    table.fleet.count[:], table.fleet.kv[0], table.endpoint[:] = [100, 0], 1., 1.
    timing["kv_completion_s"] = .1
    result = execute_pooled(table, np.array([100.]), timing, calibration, chunks=chunks)
    assert result["transferred_bytes"][0] >= 98
    assert result["completed_sessions"] >= 90


def test_zero_byte_initial_waves_need_no_network_admission():
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      gpus=100, demand=(0., 0.), deadline=9.)
    table.fleet.count[:] = [100, 0]
    table.fleet.log[0] = 0
    result = execute_pooled(table, np.array([100.]), timing, calibration, chunks=16)
    assert result["completed_sessions"] == 100
    assert result["last_completion_s"] == 8


def test_final_delta_priority_does_not_duplicate_application_node_capacity():
    table, timing, calibration = case(context=(1., 100.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), demand=(0., 0.), deadline=2.)
    table.fleet.count[:] = [2, 0]
    table.fleet.kv[0] = 1
    table.endpoint[:], table.budgets[:] = 100, 100
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 1 + i, "prompt": 1, "output": 0} for i in range(10)], []])
    timing["regional_kv_bytes_per_s"] = [1., 1.]
    calibration.update(F=1., G=1.)
    result = execute_pooled(table, np.array([2.]), timing, calibration, chunks=2)
    assert result["completed_sessions"] == 1
    assert result["last_completion_s"] == 2
    assert result["transferred_bytes"] == pytest.approx([2, 0, 2])


@pytest.mark.parametrize("chunks", [1, 8])
def test_interrupted_execution_preserves_handoff_debt_and_recovery(chunks):
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      route=(0, 0), load=.25, demand=(.1, .1), deadline=40.)
    timing.update(resident_replay_loss=.9, beta=.4)
    expected = execute_pooled(table, np.ones(2), timing, calibration, chunks)
    execution = PooledExecution(table, timing, calibration, chunks)
    execution.admit(np.ones(2))
    for until in (.1, 1., 2., 3., 6., 10., 20., 40.):
        execution.advance(until)
    actual = execution.result()
    for key in ("shed_fraction", "last_completion_s", "service_ready_s", "resident_debt_generated_work_s",
                "resident_debt_recovered_work_s", "pending_resident_debt_work_s", "transferred_bytes",
                "batch_replica_seconds", "final_destination_load", "action_counts"):
        assert actual[key] == pytest.approx(expected[key])
    assert execution.now == 40.


def test_later_admission_preserves_debt_and_source_reservations():
    from copy import deepcopy
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      route=(0, 0), load=.5, demand=(0., 0.), work=(1., 1.), deadline=20.)
    timing.update(resident_replay_loss=1., beta=.5)
    execution = PooledExecution(table, timing, calibration)
    execution.advance(2.)
    execution.admit([1., 0.])
    execution.advance(3.5)
    assert execution.result()["completed_sessions"] == 1
    assert execution.resident_debt[0] > 0
    clear = deepcopy(execution)
    clear.resident_debt[:] = 0
    debt = execution.resident_debt.copy()
    execution.admit([0., .25])
    assert execution.resident_debt == pytest.approx(debt)
    assert execution.selected_total == pytest.approx([1, .25])
    with pytest.raises(ValueError, match="duplicated source"):
        execution.admit([1., 0.])
    clear.admit([0., .25])
    execution.advance(20.)
    clear.advance(20.)
    assert execution.result()["last_completion_s"] > clear.result()["last_completion_s"]
    assert execution.result()["pending_resident_debt_work_s"] == [0, 0]
    assert execution.result()["resident_debt_generated_work_s"] == pytest.approx(execution.result()["resident_debt_recovered_work_s"])


def test_later_admission_keeps_original_snapshot_and_current_source_context():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(context=(1., 100.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), demand=(0., 0.), deadline=10.)
    table.fleet.kv[0] = 1.
    table.fleet.metadata.update(source_session_rps=1., turn_sequences=[[
        {"context": 1 + i, "prompt": 1, "output": 0} for i in range(20)], []])
    calibration.update(F=1., G=1.)
    execution = PooledExecution(table, timing, calibration)
    execution.advance(3.)
    execution.admit([1.])
    execution.advance(10.)
    assert execution.result()["transferred_bytes"] == pytest.approx([5., 0., 5.])
    assert execution.result()["last_completion_s"] == pytest.approx(4.4)
    with pytest.raises(ValueError, match="current time"):
        execution.advance(9.)
    with pytest.raises(ValueError, match="deadline"):
        execution.advance(11.)


def test_observable_phase_progress_accumulates_across_interrupts_and_resets():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.), (0., 1.)), kv=((0., 0.), (0., 0.)),
                                      route=(0, 0), demand=(0., 0.), work=(8., 8.), deadline=20.)
    execution = PooledExecution(table, timing, calibration)
    execution.advance(2.)
    execution.admit([1., 1.])
    execution.advance(2.2)
    assert execution.phase_started == pytest.approx([2., 2.])
    assert execution.phase_transferred_bytes == pytest.approx([1., 1.])
    assert execution.phase_replica_seconds == pytest.approx([0., 0.])
    execution.advance(2.8)
    assert execution.state.tolist() == [1, 1]
    assert execution.phase_started == pytest.approx([2.4, 2.4])
    assert execution.phase_transferred_bytes == pytest.approx([0., 0.])
    assert execution.phase_replica_seconds == pytest.approx([.2, .2])
    execution.advance(3.2)
    assert execution.phase_replica_seconds == pytest.approx([.4, .4])
    execution.advance(20.)
    assert execution.state.tolist() == [6, 6]
    assert execution.phase_replica_seconds == pytest.approx([0., 0.])


def test_protected_compute_shares_one_safe_occupancy_budget():
    from pool_shed_execution import compute_allocation
    assert compute_allocation(.5, 1., 1., 1., .9, True) == pytest.approx((.25, .5))
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), deadline=30.)
    table.fleet.log[0] = 0
    table.fleet.metadata.update(protect_resident=True, timing_load_factor=.5)
    timing.update(beta=2., resident_replay_loss=.9)
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(8 * np.exp(.25) / .75)
    assert result["resident_debt_generated_work_s"] == [0, 0]
    assert result["batch_replica_seconds"][0] == pytest.approx(8 * np.exp(.25))


def test_protected_source_quiesces_immediately_between_requests():
    table, _, _ = case()
    table.fleet.metadata.update(protect_resident=True, source_session_rps=.1, sequence_cycle=True,
        turn_sequences=[[{"context": 100, "prompt": 1, "output": 0}], []], turn_duration_s=[[.1], []])
    assert _quiesce(table.fleet, np.array([1, 0]), .05)[0] == .1
    assert _quiesce(table.fleet, np.array([1, 0]), 2.)[0] == 2.
    assert _quiesce(table.fleet, np.array([1, 0]), 10.)[0] == 10.1


@pytest.mark.parametrize("load,completed,pending", [(.25, 1., 0.), (.75, 0., .5)])
def test_protected_handoff_waits_for_buffer_and_never_strands_it_at_destination(load, completed, pending):
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      load=load, demand=(.25, 0.), deadline=4.)
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 1, "prompt": 1, "output": 0}], []], turn_duration_s=[[.25], []])
    calibration["resident_service"] = {"prefill": [[1, 4], [100, 4]], "decode": [[1, 4], [100, 4]], "bound": 1.}
    execution = PooledExecution(table, timing, calibration)
    execution.admit([1.])
    execution.state[:], execution.remaining[:], execution.release[:], execution.now = 5, 0., 2., 2.
    execution.advance(4.)
    result = execution.result()
    assert result["completed_sessions"] == completed
    assert result["pending_source_buffer_work_s"] == pytest.approx(pending)
    assert result["pending_backlog_reference_work_s"] == 0
    assert result["resident_debt_generated_work_s"] == [0, 0]
    assert result["buffered_requests"] == pytest.approx(result["completed_buffered_requests"] + result["pending_buffered_requests"])
    assert result["buffered_requests"] == pytest.approx(result["transferred_buffered_requests"] + result["source_buffered_requests"])
    if completed:
        assert result["last_completion_s"] == 3.
        assert result["service_ready_s"] == 3.


def test_protected_gate_buffer_has_priority_over_other_replay():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(route=(0, 0), load=.25, demand=(.25, 0.), deadline=4.)
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1.,
        turn_sequences=[[{"context": 1, "prompt": 1, "output": 0}] * 4, []], turn_duration_s=[[.25] * 4, []])
    calibration["resident_service"] = {"prefill": [[1, 4], [100, 4]], "decode": [[1, 4], [100, 4]], "bound": 1.}
    execution = PooledExecution(table, timing, calibration)
    execution.admit([1., 1.])
    execution.state[:], execution.remaining[:], execution.release[:], execution.now = [5, 1], [0., 8.], [2., 0.], 2.
    execution.advance(2.5)
    assert execution.backlog[0] == pytest.approx(.25)
    assert execution.remaining[1] == 8.
    assert execution.serving_load() == pytest.approx([.5, .25])
    assert execution.result()["pending_source_buffer_work_s"] == pytest.approx(.25)
    assert execution.result()["pending_backlog_reference_work_s"] == 0


def test_scoped_primitive_cache_preserves_interrupted_execution_exactly():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.), (0., 1.)), kv=((1., 0.), (0., 0.)),
                                      route=(0, 0), load=.25, demand=(.2, .2), deadline=30.)
    timing.update(beta=1., resident_replay_loss=.9)
    engines = [PooledExecution(table, timing, calibration, chunks=16) for _ in range(2)]
    engines[1].primitive_cache = None
    for engine in engines:
        engine.admit([1., 1.])
    for end in (3., 10., 30.):
        for engine in engines:
            engine.advance(end)
        assert engines[0].result() == engines[1].result()
    assert engines[0].primitive_cache


@pytest.mark.parametrize("protected,cycle", [(False, False), (False, True), (True, False), (True, True)])
def test_cached_quiescence_preserves_idle_gaps_boundaries_and_resets(protected, cycle):
    table, _, _ = case()
    table.fleet.metadata.update(protect_resident=protected, sequence_cycle=cycle, source_session_rps=.1,
        turn_sequences=[[{"context": 100, "prompt": 10, "output": 2},
                         {"context": 0, "prompt": 5, "output": 1, "reset": True}], []],
        turn_duration_s=[[.2, .1], []], turn_offset=[1, 0])
    cache, counts = {}, np.array([1, 0])
    for now in (2., 0., .05, .1, .2, 9.999999, 10., 10.05, 10.2, 20., 21., 40., 40.1):
        cached, plain = _quiesce(table.fleet, counts, now, cache), _quiesce(table.fleet, counts, now)
        assert cached[0] == plain[0] and cached[3] == plain[3]
        assert np.array_equal(cached[1], plain[1]) and np.array_equal(cached[2], plain[2])


def test_near_simultaneous_gigabyte_transfers_share_one_completion_event():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.), (0., 0.)), kv=((1., 0.), (0., 1.)),
                                      demand=(0., 0.), deadline=2.)
    table.endpoint[:], table.budgets[:] = 1e9, [1e9, 1e9, 2e9]
    table.fleet.kv[:] = [1e9, 1e9 + 5e-6]
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1., 1.])
    engine.advance(2.)
    assert engine.quiesced[0] == engine.quiesced[1]
    assert engine.result()["transferred_bytes"] == [*table.fleet.kv, sum(table.fleet.kv)]
    assert engine.result()["shed_fraction"] == 1.


def test_completion_coalescing_never_crosses_requested_horizon():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      demand=(0., 0.), deadline=2.)
    table.endpoint[:], table.budgets[:] = 1e9, 1e9
    table.fleet.kv[0] = 1e9 + .01
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(1.)
    assert engine.state[0] == 0 and engine.remaining[0] > 0
    assert engine.result()["transferred_bytes"][0] == 1e9


def test_streamed_kv_completion_is_invariant_to_proportional_fleet_scale():
    from pool_shed_campaign import calibration, execute_feedback, forecast
    central, results = calibration(0), []
    for gpus in (6400, 64000):
        table = forecast("measured_pack", 0, gpus, 8, .5, 5 * gpus / 8, 30)[0]
        result = execute_feedback(table, table, "kv_only", table.timing, central, chunks=64, resolution=.5)
        assert result["resident_debt_generated_work_s"] == [0, 0]
        assert result["buffered_requests"] == pytest.approx(result["completed_buffered_requests"] + result["pending_buffered_requests"])
        results.append(result["shed_fraction"])
    assert abs(results[0] - results[1]) <= 1e-8


def test_nominal_progress_integrates_observed_load_with_central_beta():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,), load=.2, demand=(0., 0.))
    table.fleet.log[:] = 0.
    table.fleet.metadata["protect_resident"] = True
    timing["beta"], calibration["timing"] = 5., [{"beta": 2.}]
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(1.)
    engine.loads[0] = .4
    engine.advance(2.)
    expected = .8 * np.exp(-.4) + .6 * np.exp(-.8)
    assert engine.phase_nominal_work[0] == pytest.approx(expected)
    clone = engine.nominal_continuation(table, {**timing, "beta": 2.}, calibration)
    assert clone.remaining[0] == pytest.approx(8. - expected)
    assert clone.remaining[0] < engine.remaining[0]


def test_nominal_continuation_ignores_latent_draw_state_and_is_isolated():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(deadline=30., demand=(0., 0.))
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1., 1.])
    engine.advance(1.)
    first = engine.nominal_continuation(table, timing, calibration)
    engine.remaining[:] = 123456.
    engine.tail[:], engine.delta[:] = 987654., 456789.
    engine.timing = {**timing, "beta": 999., "kv_completion_s": 999.}
    second = engine.nominal_continuation(table, timing, calibration)
    for end in (3., 30.):
        left, right = first.advance(end, collect=True), second.advance(end, collect=True)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        assert first.result() == second.result()
    assert engine.now == 1. and np.all(engine.remaining == 123456.)


def test_resource_collector_conserves_work_and_counts_idle_standing():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      load=.25, demand=(0., 0.), work=(1.5, 1.), deadline=10.)
    table.fleet.log[:] = 0.
    table.fleet.metadata["protect_resident"] = True
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    busy = engine.advance(2., collect=True)
    assert engine.state[0] == 6
    assert busy["migration"] == pytest.approx([1.5, 0.])
    assert busy["serving"] == pytest.approx([.5, .5])
    assert busy["peak"] == pytest.approx([1., .25])
    assert busy["service_peak"] == pytest.approx([.25, .25])
    idle = engine.advance(10., collect=True)
    assert idle["serving"] == pytest.approx([2., 2.])
    assert idle["migration"] == pytest.approx([0., 0.])
    assert idle["peak"] == pytest.approx([.25, .25])


def test_resource_collector_tracks_only_kv_bytes_as_application_traffic():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(demand=(0., 0.))
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1., 1.])
    usage = engine.advance(10., collect=True)
    assert usage["network"] == pytest.approx(engine.network_used)
    assert usage["migration"] == pytest.approx(engine.compute_used)
    assert usage["application"] == pytest.approx([0., 98.])


@pytest.mark.parametrize("at,phase,gated", [(.1, 0, False), (.25, 1, False), (.4, 2, False),
                                          (1., 3, False), (4.51, 4, False), (5., 5, False), (6., 5, True)])
def test_central_continuation_matches_every_phase_reset_and_partial_gate(at, phase, gated):
    from copy import deepcopy
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((1., 0.),), kv=((0., 0.),), route=(0,),
                                      load=.25, demand=(.2, 0.), work=(.1, 1.), deadline=12.)
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 0, "prompt": 20, "output": 0, "reset": True}], []],
        turn_duration_s=[[.5], []], turn_work_s=[[.2], []])
    calibration.update(replay_context_tokens=[20., 100.], replay_tps=[1000., 1000.], replay_completion_s=0., switch_s=1.)
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(at)
    assert engine.state[0] == phase and engine.gated[0] == gated
    baseline = deepcopy(engine)
    first = engine.nominal_continuation(table, timing, calibration)
    engine.remaining[:], engine.tail[:], engine.delta[:] = 123456., 987654., 456789.
    engine.timing = {**timing, "beta": 999.}
    second = engine.nominal_continuation(table, timing, calibration)
    left, right = first.advance(12., collect=True), second.advance(12., collect=True)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    baseline.advance(12.)
    for key in ("shed_fraction", "last_completion_s", "transferred_bytes", "batch_replica_seconds", "completed_buffered_requests"):
        assert first.result()[key] == pytest.approx(baseline.result()[key], abs=1e-10)
    assert engine.now == at and np.all(engine.remaining == 123456.)


def test_source_snapshot_excludes_inflight_request_and_catchup_uses_capture_generation():
    from pool_shed_execution import source_snapshot, catchup, initial_work
    table, timing, calibration = case()
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 0, "prompt": 20, "output": 0, "reset": True},
                         {"context": 20, "prompt": 15, "output": 0}], []], turn_duration_s=[[.5, .5], []])
    for now, expected, turns in [(0., 100, 0), (.5, 20, 1), (1.2, 20, 1), (1.5, 35, 2), (2.2, 35, 2), (2.5, 20, 3)]:
        context, completed = source_snapshot(table.fleet, now)
        assert context[0] == expected and completed[0] == turns
    counts = np.array([1., 0.])
    for start, expected in [(1.6, 20.), (2.6, 0.)]:
        origin, turn = source_snapshot(table.fleet, start)
        _, context, reset, _ = _quiesce(table.fleet, counts, 2.7, origin_turn=turn)
        assert catchup(table.fleet, counts, 1, 0, context, reset, timing, calibration, origin_context=origin)[0] == expected
    calibration.update(replay_context_tokens=[20., 100.], replay_tps=[1000., 1000.], replay_completion_s=0.)
    assert initial_work(table.fleet, counts, 0, 0, np.array([20., 100.]), timing, calibration) == pytest.approx((40., .02))
    assert initial_work(table.fleet, counts, 0, 0, np.array([0., 100.]), timing, calibration) == (0., 0.)


def test_queued_wave_captures_at_first_dispatch_and_clone_preserves_uncaptured_origin():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      demand=(0., 0.), deadline=20.)
    table.fleet.count[0] = 2.
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 0, "prompt": 10, "output": 0, "reset": True}], []],
        turn_duration_s=[[.1], []], turn_work_s=[[0.], []])
    engine = PooledExecution(table, timing, calibration, chunks=2)
    engine.admit([2.])
    engine.advance(9.9)
    assert engine.origin_time[0] == 0. and np.isnan(engine.origin_time[1])
    clone = engine.nominal_continuation(table, timing, calibration)
    assert np.isnan(clone.origin_time[1])
    for end in (10.05, 10.06, 20.):
        engine.advance(end)
        clone.advance(end)
        assert engine.result() == clone.result()
        assert engine.origin_time[1] == pytest.approx(10.)
    assert engine.result()["transferred_bytes"] == pytest.approx([130., 0., 130.])


def test_zero_byte_source_image_captures_before_entering_quiescence():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(context=(0., 0.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), demand=(0., 0.), deadline=2.)
    table.fleet.kv[0] = 0.
    table.fleet.metadata.update(protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 0, "prompt": 5, "output": 0}], []],
        turn_duration_s=[[.25], []], turn_work_s=[[0.], []])
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(.1)
    assert engine.origin_time[0] == 0. and engine.state[0] == 2
    engine.advance(2.)
    assert engine.result()["last_completion_s"] == pytest.approx(.75)
    assert engine.result()["transferred_bytes"] == [5., 0., 5.]


def test_wave_records_keep_repeated_chunk_admissions_without_changing_execution():
    from pool_shed_execution import PooledExecution
    results = []
    for record in (False, True):
        table, timing, calibration = case(deadline=30., demand=(0., 0.))
        table.fleet.metadata["record_wave_schedules"] = record
        engine = PooledExecution(table, timing, calibration, chunks=2)
        engine.admit([.5, 0.])
        engine.advance(1.)
        engine.admit([.5, 0.])
        engine.advance(30.)
        results.append(engine.result())
    records = results[1].pop("wave_schedules")
    assert results[0] == results[1]
    assert len(results[0]["completion_events"]) == 1
    assert [r["wave_id"] for r in records] == [0, 1, 2, 3]
    assert [r["admitted_s"] for r in records] == [0., 0., 1., 1.]
    assert all(r["column"] == 0 and r["mass"] == .25 and r["state"] == 6 for r in records)
    assert all(r["phase_enter_s"][0] == r["admitted_s"] and r["phase_enter_s"][7] is None for r in records)


def test_wave_records_keep_uncaptured_origins_and_clone_arrays_independent():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
                                      demand=(0., 0.), deadline=20.)
    table.fleet.count[0] = 2.
    table.fleet.metadata.update(record_wave_schedules=True, protect_resident=True, source_session_rps=1., sequence_cycle=True,
        turn_sequences=[[{"context": 0, "prompt": 10, "output": 0, "reset": True}], []],
        turn_duration_s=[[.1], []], turn_work_s=[[0.], []])
    engine = PooledExecution(table, timing, calibration, chunks=2)
    engine.admit([2.])
    engine.advance(9.9)
    before = engine.result()["wave_schedules"]
    assert before[1]["admitted_s"] == 0. and before[1]["origin_s"] is None
    assert before[1]["phase_enter_s"] == [0., None, None, None, None, None, None, None]
    assert before[1]["pause_requested_s"] is None and before[1]["origin_context"] is None
    clone = engine.nominal_continuation(table, timing, calibration)
    assert not np.shares_memory(engine.phase_enter_s, clone.phase_enter_s)
    assert not np.shares_memory(engine.admitted_s, clone.admitted_s)
    clone.advance(20.)
    assert engine.result()["wave_schedules"] == before
    engine.advance(20.)
    assert engine.result() == clone.result()
    second = engine.result()["wave_schedules"][1]
    assert second["origin_s"] == pytest.approx(10.)
    assert second["origin_context"][0] == 10. and second["origin_turn"][0] == 10
    assert second["phase_enter_s"][1] is None  # KV has no initial compute phase.


def test_wave_records_pause_source_turns_before_future_handoff_and_stamp_zero_work():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(context=(0., 0.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), demand=(0., 0.), deadline=4.)
    table.fleet.kv[0] = 0.
    table.fleet.metadata.update(record_wave_schedules=True, paced_source=True, causal_source=True,
        source_session_rps=1., sequence_cycle=True, source_phase_s=[0., 0.],
        turn_sequences=[[{"context": 0, "prompt": 5, "output": 0}], []],
        turn_duration_s=[[.25], []], turn_work_s=[[0.], []])
    calibration["switch_s"] = 2.
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(.1)
    row = engine.result()["wave_schedules"][0]
    assert row["state"] == 2 and row["phase_enter_s"][:3] == [0., None, 0.]
    assert row["origin_turn"][0] == 0 and row["paused_turns"][0] == 1
    assert row["pause_requested_s"] == 0. and row["quiesced_s"] == .25
    assert row["quiesced_context"][0] == 5.
    engine.advance(4.)
    row = engine.result()["wave_schedules"][0]
    assert row["phase_enter_s"] == pytest.approx([0., None, 0., .25, .75, .75, 2.75, None])
    assert row["paused_turns"][0] == 1  # Not source_turns(future handoff), which would be three.


def test_wave_records_memory_blocked_entry_remains_unfinished():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(context=(40., 40.), replay=((0., 0.),), kv=((1., 0.),),
                                      route=(0,), deadline=30.)
    table.fleet.kv[0], table.fleet.kv_capacity = 1., 60.
    table.fleet.metadata.update(record_wave_schedules=True, source_session_rps=1.,
                               turn_sequences=[[{"context": 40, "prompt": 30, "output": 0}], []])
    engine = PooledExecution(table, timing, calibration)
    engine.admit([1.])
    engine.advance(30.)
    row = engine.result()["wave_schedules"][0]
    assert row["state"] == 7 and row["phase_enter_s"][7] == pytest.approx(.1)
    assert row["phase_enter_s"][2:7] == [None] * 5
    assert row["quiesced_context"][0] == 70 and row["paused_turns"][0] == 1


@pytest.mark.parametrize("action", [0, 1])
def test_fixed_host_history_shares_cap_initial_payload_without_idle_borrowing(action):
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((8. if not action else 0., 0.),),
        kv=((8. if action else 0., 0.),), route=(0,), work=(0., 0.), demand=(0., 0.), gpus=8, deadline=110.)
    table.fleet.gpus_per_node, table.fleet.count[0] = 8, 8
    table.fleet.log[0] = 100.
    table.fleet.metadata.update(resident_affinity=True, fixed_host_shares=True, host_migration_gbps=512e-9)
    timing["resident_replay_loss"] = 0.
    table.endpoint[:], table.budgets[:] = 1e6, 1e6
    engine = PooledExecution(table, timing, calibration, chunks=4)
    engine.admit([1.])
    usage = engine.advance(50., collect=True)
    assert usage["network"] == pytest.approx([400., 0., 400.])
    assert engine.result()["shed_fraction"] == 0.
    engine.advance(110.)
    assert engine.result()["last_completion_s"] == pytest.approx(100.)
    assert engine.result()["transferred_bytes"] == pytest.approx([800., 0., 800.])


def test_fixed_host_worker_pool_uses_gpu_equivalents_and_legacy_retains_nodes():
    from pool_shed_execution import PooledExecution
    transferred = []
    for enabled in (False, True):
        table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),),
            route=(0,), demand=(0., 0.), gpus=8)
        table.fleet.gpus_per_node, table.fleet.count[0] = 8, 8
        table.fleet.metadata.update(resident_affinity=True, fixed_host_shares=enabled, host_migration_gbps=512e-9)
        timing.update(resident_replay_loss=0., regional_kv_bytes_per_s=[.5, .5])
        table.endpoint[:], table.budgets[:] = 1e6, 1e6
        engine = PooledExecution(table, timing, calibration)
        engine.admit([8.])
        transferred.append(engine.advance(1., collect=True)["application"][0])
    assert transferred == pytest.approx([.5, 4.])


def test_fixed_host_delta_and_queued_snapshot_caps_keep_frozen_origins_in_clone():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(context=(10., 10.), replay=((0., 0.),), kv=((1., 1.),),
        route=(0,), demand=(0., 0.), gpus=8, deadline=120.)
    table.fleet.gpus_per_node, table.fleet.kv[:] = 8, 10.
    table.fleet.metadata.update(resident_affinity=True, fixed_host_shares=True, host_migration_gbps=512e-9,
        paced_source=True, causal_source=True, source_session_rps=1., record_wave_schedules=True,
        turn_sequences=[[{"context": 100, "prompt": 0, "output": 0}], []],
        turn_duration_s=[[5.], []], turn_work_s=[[0.], []])
    timing["resident_replay_loss"] = 0.
    table.endpoint[:], table.budgets[:] = 1e6, 1.
    engine = PooledExecution(table, timing, calibration, chunks=2)
    engine.admit([1.])
    engine.advance(9.)
    assert engine.origin_time[0] == 0. and np.isnan(engine.origin_time[1])
    engine.advance(10.5)
    assert engine.state.tolist() == [3, 0]
    assert engine.origin_time.tolist() == pytest.approx([0., 10.])
    assert [engine.network_cap(i) for i in (0, 1)] == pytest.approx([1., 1.1])
    assert engine.remaining.tolist() == pytest.approx([89.5, 109.5])
    clone = engine.nominal_continuation(table, timing, calibration)
    engine.advance(11.)
    clone.advance(11.)
    assert engine.result() == clone.result()


def test_fixed_host_execution_rejects_invalid_shares_and_zero_work_payloads():
    from pool_shed_execution import PooledExecution
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,), gpus=8)
    table.fleet.gpus_per_node = 8
    table.fleet.metadata.update(resident_affinity=True, fixed_host_shares=True, host_migration_gbps=512e-9)
    timing["resident_replay_loss"] = 0.
    for invalid in (-1., np.nan, np.inf):
        table.fleet.kv[0] = invalid
        with pytest.raises(ValueError):
            PooledExecution(table, timing, calibration).admit([1.])
    table.fleet.kv[0] = 100.
    for invalid in (-1., np.nan, np.inf):
        table.fleet.metadata["host_migration_gbps"] = invalid
        with pytest.raises(ValueError):
            PooledExecution(table, timing, calibration)


def test_fixed_host_fractional_rate_survives_integer_endpoint_inputs():
    table, timing, calibration = case(replay=((0., 0.),), kv=((1., 0.),), route=(0,),
        demand=(0., 0.), gpus=8, deadline=250.)
    table.fleet.gpus_per_node = 8
    table.fleet.metadata.update(resident_affinity=True, fixed_host_shares=True, host_migration_gbps=256e-9)
    timing["resident_replay_loss"] = 0.
    table.endpoint, table.budgets = np.array([1000, 1000]), np.array([1000, 1000, 1000])
    result = run(table, timing, calibration)
    assert result["last_completion_s"] == pytest.approx(200.)
    assert result["transferred_bytes"] == pytest.approx([100., 0., 100.])
