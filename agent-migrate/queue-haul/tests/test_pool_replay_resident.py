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
