import pytest
from pool_replay_resident import Trajectory, arrival_trace, summarize


def test_trajectory_retains_output_inside_next_recorded_append():
    t=Trajectory('a',[{'context':0,'prompt':20,'output':3,'reset':False},
                       {'context':20,'prompt':8,'output':2,'reset':False}],0,7101)
    prompt,_,_,_,added=t.prompt()
    t.accept(prompt,{'done':True,'status':200,'recorded_output_tokens':3,
                     'planned_output_tokens':3,'token_ids':[91,92,93]})
    updated,_,_,_,added=t.prompt()
    assert len(updated)==28 and updated[:23]==prompt+[91,92,93] and added==5
    t.accept(updated,{'done':True,'status':200,'recorded_output_tokens':2,
                      'planned_output_tokens':2,'token_ids':[94,95]})
    wrapped,_,index,reset,_=t.prompt()
    assert reset and index==0 and len(wrapped)==20


def test_dependency_failure_hard_fails_and_burst_preserves_offered_count():
    t=Trajectory('a',[],0,7101)
    with pytest.raises(RuntimeError,match='dependency'):
        t.accept([],{'done':False,'status':200,'recorded_output_tokens':0,'planned_output_tokens':3})
    assert t.failed
    regular=arrival_trace(2,[2,1],7101,180,60)
    burst=arrival_trace(2,[2,1],7101,180,60,True)
    assert len(regular)==len(burst)
    for session in (0,1):
        assert [r['turn'] for r in burst if r['session']==session]==list(range(sum(r['session']==session for r in burst)))


def test_window_retains_late_work_and_counts_completions_by_completion_time():
    row={'scheduled_ns':59_000_000_000,'end_ns':61_000_000_000,'done':True,'status':200,
         'exact_token_timestamps':False,'send_lateness_s':0,'prompt_tokens':100,'output_tokens':1}
    trace=[{'offset_s':59}]
    before=summarize([row],trace,0,(0,60),[])
    after=summarize([row],trace,0,(60,90),[])
    assert before['unfinished_or_failed_requests']==1 and before['completed_rps']==0
    assert after['completions_in_window']==1 and after['completed_rps']==1/30


def test_client_queue_excludes_scheduler_wakeup_lateness():
    row={'scheduled_ns':0,'client_wakeup_ns':100_000_000,'client_dispatch_ns':250_000_000,
         'end_ns':600_000_000,'first_ns':400_000_000,'done':True,'status':200,
         'exact_token_timestamps':True,'send_lateness_s':.3,'prompt_tokens':100,'output_tokens':2,'mean_tpot_s':.1}
    summary=summarize([row],[{'offset_s':0}],0,(0,60),[])
    assert summary['p90_client_queue_s']==.15
    assert summary['p90_client_schedule_lateness_s']==.1 and summary['p90_client_send_lateness_s']==.3


def test_physical_population_has_eight_equal_weight_lanes_and_records_exclusions():
    from pool_replay_resident import physical_workload
    workload={'turn_sequences':[[{'context':100,'prompt':10,'output':3}],
                                [{'context':32000,'prompt':200,'output':100}]],
              'turn_offset':[0,0],'cohort_counts':[1,1000]}
    chosen=physical_workload(workload,7101)
    assert chosen['cohort_counts']==[1]*8
    assert chosen['sampled_trajectory_indices']==[0]*8
    assert chosen['migration_context_excluded_indices']==[1]
    assert chosen==physical_workload(workload,7101)


def test_paused_arrivals_recheck_ownership_after_waiting_for_inflight_request():
    import asyncio
    async def run():
        t=Trajectory('session',[],0,7101);t.route='source'
        entered=[]
        await t.lock.acquire()
        async def queued(index):
            async with t.owned():entered.append((index,t.route))
        pending=[asyncio.create_task(queued(i)) for i in range(3)]
        await asyncio.sleep(0)
        t.admission.clear()
        t.lock.release()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not entered
        t.route='destination';t.admission.set()
        await asyncio.gather(*pending)
        assert entered==[(i,'destination') for i in range(3)]
    asyncio.run(run())


def test_materialized_history_and_real_generated_tokens_survive_route_switch():
    t=Trajectory('session',[{'context':10,'prompt':6,'output':3,'reset':False},
                            {'context':16,'prompt':8,'output':2,'reset':False}],0,7101)
    t.initial_history=list(range(10))
    prompt,*_=t.prompt()
    assert prompt[:10]==list(range(10))
    t.accept(prompt,{'done':True,'status':200,'recorded_output_tokens':3,
                     'planned_output_tokens':3,'token_ids':[201,202,203]})
    snapshot=list(t.history);generation=t.completed
    t.route='destination'
    next_prompt,_,_,reset,added=t.prompt()
    assert next_prompt[:len(snapshot)]==snapshot and not reset and added==5 and generation==1


def test_cache_isolation_waits_for_locks_and_async_storage_not_stale_result_handles():
    import copy
    from pool_replay_resident import cache_idle
    status={'is_healthy':True,'active_prefetch_jobs':3,'storage_manager':{
        'l1_manager':{'write_locked_count':0,'read_locked_count':0,'temporary_count':0},
        'store_controller':{'pending_keys_count':0,'in_flight_task_count':0},
        'prefetch_controller':{'submission_queue_size':0,'pending_queue_size':0,'in_flight_request_count':0}}}
    assert cache_idle(status)
    for group,values in status['storage_manager'].items():
        for key in values:
            pending=copy.deepcopy(status);pending['storage_manager'][group][key]=1
            assert not cache_idle(pending)
    with pytest.raises(KeyError):cache_idle({'is_healthy':True,'storage_manager':{}})
    with pytest.raises(RuntimeError):cache_idle({'is_healthy':False})
