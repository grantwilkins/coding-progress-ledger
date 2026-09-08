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
