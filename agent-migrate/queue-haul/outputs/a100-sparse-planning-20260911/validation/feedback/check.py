"""Four matched production-feedback runs; no request-latency audit or runtime claim."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import issparse
from _queue_haul_native import _queue_haul_native as native
import pool_shed_campaign as q
import pool_shed_bandwidth_sweep as sweep

OUT=q.ROOT/'outputs/a100-sparse-planning-20260911/validation/feedback'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def array(value): return value.toarray() if issparse(value) else np.asarray(value)
def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return dict(exact=bool(np.array_equal(a,b)),maximum_absolute=float(np.max(abs(a-b),initial=0.)))
def accounting(fleet,result,table):
    used=np.zeros(len(fleet.count)); completed=np.zeros((4,len(fleet.count)))
    replicas=np.zeros(2); peak_local=0.; phase_mass=np.zeros((4,8))
    for w in result['wave_schedules']:
        counts=np.asarray(w['counts']); a=2*w['route']+(w['action']=='kv_transfer')
        used+=w['mass']*counts; replicas[w['route']]+=w['mass']
        peak_local=max(peak_local,float(counts@fleet.demand))
        if w['state']==6: completed[a]+=w['mass']*counts
        phase_mass[a]+=w['mass']*np.asarray([t is not None for t in w['phase_enter_s']])
    np.testing.assert_allclose(completed.sum(1),result['action_counts'],rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(completed@fleet.gain,result['action_fractions'],rtol=1e-10,atol=1e-10)
    assert np.all(used<=fleet.count+1e-6) and np.all(replicas<=fleet.gpus+1e-6)
    return dict(max_source_relative_overload=float(np.max(used/fleet.count-1,initial=0.)),
        reserved_replicas=replicas.tolist(),maximum_incoming_local_load=peak_local,
        network_volume_budget_fraction=(np.asarray(result['transferred_bytes'])/(table.budgets*table.deadline)).tolist(),
        compute_replica_time_fraction=(np.asarray(result['batch_replica_seconds'])/(fleet.gpus*table.deadline)).tolist(),
        resident_debt_balance_max_abs=float(np.max(abs(np.asarray(result['resident_debt_generated_work_s'])-result['resident_debt_recovered_work_s']-result['pending_resident_debt_work_s']),initial=0.)),
        phase_entered_replica_mass_by_action=phase_mass.tolist(),wave_count=len(result['wave_schedules']),
        last_completion_s=result['last_completion_s'],service_ready_s=result['service_ready_s'])

if __name__=='__main__':
    assert not OUT.exists(), 'Choose a fresh output directory; do not overwrite an integration check'
    measured=q.calibration(0); fleet=sweep.fleet_for('coding',measured)
    endpoint=np.r_[np.median(q.network_samples()[:,:2],axis=0),0.]; endpoint[2]=endpoint[:2].sum()
    inputs={str(p.relative_to(q.ROOT)):sha(p) for p in q.ROOT.glob('pool_shed*.py')}
    binary=sha(Path(native.__file__))
    tables={}; candidates={}; fastest={}
    for encoding in ('dense','csr'):
        sparse=encoding=='csr'
        fastest[encoding]=q.isolated_methods(fleet,sweep.LOAD,endpoint,np.full(3,1e12/8),measured['timing'][0],compact=sparse)
        r,k=q.include_isolated(*q.library(fleet,sparse=sparse),fastest[encoding])
        candidates[encoding]=(r,k)
        tables[encoding]=q.schedule_table(fleet,r,k,sweep.LOAD,30.,endpoint,np.full(3,10e12/8),measured['timing'][0])
    np.testing.assert_array_equal(fastest['dense'],fastest['csr'])
    for a,b in zip(candidates['dense'],candidates['csr']): np.testing.assert_array_equal(a,array(b))
    table_difference={key:difference(array(getattr(tables['dense'],key)).astype(float),array(getattr(tables['csr'],key)).astype(float))
        for key in ('replay','kv','route','duration','release','kv_release','log_bytes','kv_bytes','rate','eligible','fastest','matrix','capacities','gains','debt','nominal_commit','service_time')}
    for key in ('route','eligible','fastest','capacities'): assert table_difference[key]['exact'],key
    records={}
    for encoding in ('dense','csr'):
        table=tables[encoding]
        for policy in ('queue_haul','greedy_priced'):
            result=q.execute_feedback(table,table,policy,measured['timing'][0],measured,chunks=64)
            sweep.validate_result(fleet,result,30.,table.budgets,policy)
            record=dict(encoding=encoding,policy=policy,result=result,accounting=accounting(fleet,result,table))
            path=OUT/f'{encoding}-{policy}.json.gz'; q.write_json(path,record)
            records[encoding,policy]=record
            print(encoding,policy,result['shed_fraction'],result['action_fractions'],len(result['wave_schedules']),flush=True)
    comparisons={}
    for policy in ('queue_haul','greedy_priced'):
        dense,csr=[records[e,policy]['result'] for e in ('dense','csr')]
        metrics={key:difference(dense[key],csr[key]) for key in ('shed_fraction','recovered_handoff_fraction','admitted_shed_fraction','action_counts','action_fractions','recovered_action_fractions','transferred_bytes','batch_replica_seconds','resident_debt_generated_work_s','resident_debt_recovered_work_s','pending_resident_debt_work_s','pending_buffered_work_s','final_destination_load','last_completion_s')}
        structure=lambda result:[(w['column'],w['action'],w['route'],w['counts'],w['state']) for w in result['wave_schedules']]
        same=structure(dense)==structure(csr)
        phases=dict(same_wave_structure=same,dense_count=len(dense['wave_schedules']),csr_count=len(csr['wave_schedules']))
        if same:
            phases['mass_difference']=difference([w['mass'] for w in dense['wave_schedules']],[w['mass'] for w in csr['wave_schedules']])
            left,right=[np.asarray([[np.nan if t is None else t for t in w['phase_enter_s']] for w in result['wave_schedules']]) for result in (dense,csr)]
            phases['same_phase_presence']=bool(np.array_equal(np.isnan(left),np.isnan(right)))
            mask=np.isfinite(left)&np.isfinite(right)
            phases['shared_phase_time_difference']=difference(left[mask],right[mask])
        comparisons[policy]=dict(signed_handoff_difference_pp=100*(csr['shed_fraction']-dense['shed_fraction']),metrics=metrics,phases=phases,
            planning_steps={e:records[e,policy]['result']['planning_steps'] for e in ('dense','csr')})
    assert all(sha(q.ROOT/p)==h for p,h in inputs.items()) and sha(Path(native.__file__))==binary
    report=dict(scope='Four complete feedback runs on the same production coding 30 s / 10 Tb/s 2 MW source and destination scenario, dispatch64. Dense and CSR use matching candidate construction and isolated closure, unchanged physical coefficients and acceptance tolerances. Current validate_result and independent saved-wave accounting pass for every run. Runtime is not compared; resident TTFT/TPOT is not checked or certified.',
        workload='coding',deadline_s=30,bandwidth_tbps=10,source_gpus=fleet.gpus,destination_gpus_per_site=fleet.gpus,
        fleet_contract_sha256=q.digest(dict(metadata=fleet.metadata,count=fleet.count.tolist(),context=fleet.context.tolist())),
        candidate_sha256=q.digest(dict(replay=candidates['dense'][0].tolist(),kv=candidates['dense'][1].tolist())),
        source_sha256=inputs,measurement_sha256=q.provenance(measured),native_binary_sha256=binary,
        script_sha256=sha(Path(__file__)),record_sha256={p.name:sha(p) for p in OUT.glob('*.json.gz')},
        table_difference=table_difference,comparisons=comparisons,accounting={f'{e}-{p}':r['accounting'] for (e,p),r in records.items()},resident_latency_validated=False)
    q.write_json(OUT/'review.json',report)
    print('DONE',OUT/'review.json',flush=True)
