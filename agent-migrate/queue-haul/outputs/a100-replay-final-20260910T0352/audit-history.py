"""Verify actual generated histories and causal dispatch from retained service rows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def audit(rows):
    groups=defaultdict(list)
    for row in rows:
        if row.get('phase')=='service' and row.get('cohort') in ('resident','incoming'):
            groups[(row['episode'],row['cohort'],row['session'])].append(row)
    sessions=[]
    for key,values in sorted(groups.items()):
        values.sort(key=lambda row:row['turn']);links=[]
        for prior,current in zip(values,values[1:]):
            if not prior.get('done') or not current.get('done'):continue
            retained=prior['full_prompt_token_ids']+prior['token_ids']
            links.append({'prior_turn':prior['turn'],'turn':current['turn'],'reset':current['reset'],
                'prior_prompt_tokens':prior['prompt_tokens'],'prompt_tokens':current['prompt_tokens'],
                'actual_prior_output_tokens':len(prior['token_ids']),'source_owned_prior':prior.get('serving_role')=='source',
                'source_to_destination':prior.get('serving_role')=='source' and current.get('serving_role')=='destination',
                'causal_dispatch':prior['end_ns']<=current['client_dispatch_ns'],
                'consecutive_turn':current['turn']==prior['turn']+1,
                'nonreset_exact_retained_history':None if current['reset'] else current['full_prompt_token_ids'][:len(retained)]==retained})
        sessions.append({'episode':key[0],'cohort':key[1],'physical_session':key[2],
            'session_id':values[0].get('session_id'),'observed_turns':[r['turn'] for r in values],
            'completed_turns':sum(bool(r.get('done')) for r in values),'links':links})
    links=[link for row in sessions for link in row['links']]
    return {'sessions':sessions,'completed_links':len(links),'nonreset_links':sum(not r['reset'] for r in links),
        'reset_links':sum(r['reset'] for r in links),'causal_violations':sum(not r['causal_dispatch'] for r in links),
        'source_to_destination_links':sum(r['source_to_destination'] for r in links),
        'turn_order_violations':sum(not r['consecutive_turn'] for r in links),
        'retained_history_violations':sum(r['nonreset_exact_retained_history'] is False for r in links)}


def resets(rows,workloads):
    result=[]
    for row in rows:
        if row.get('phase')!='service' or not row.get('done') or not row.get('reset'):continue
        shape=workloads[row['episode']]['turn_sequences'][row['session']][row['recorded_turn']]
        origin='initialization' if row['turn']==0 else 'assumed_cycle_wrap' if row['recorded_turn']==0 else 'recorded_shape_reset'
        if origin=='recorded_shape_reset' and not shape['reset']:raise ValueError('reset has no recorded or cycle basis')
        result.append({**{k:row[k] for k in ('episode','cohort','session','turn','recorded_turn')},'origin':origin,'shape_reset':shape['reset'],'retained_context_tokens':shape['context'],'prompt_tokens':row['prompt_tokens']})
    return {'counts':{kind:sum(r['origin']==kind for r in result) for kind in ('initialization','assumed_cycle_wrap','recorded_shape_reset')},'events':result}


def coverage(rows,traces):
    result={}
    for episode,trace in traces.items():
        key=lambda row:(row['cohort'],row['session'],row['turn'])
        expected={key(row) for row in trace};actual=Counter(key(row) for row in rows if row.get('episode')==episode and row.get('phase')=='service')
        result[episode]={'offered':len(trace),'terminal_rows':sum(actual.values()),'missing':sorted(expected-actual.keys()),'unexpected':sorted(actual.keys()-expected),'duplicates':[list(k) for k,v in actual.items() if v!=1]}
    return {'valid':all(not any(row[k] for k in ('missing','unexpected','duplicates')) and row['offered']==row['terminal_rows'] for row in result.values()),'episodes':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root;raw=(root/'requests.jsonl').read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(line) for line in complete.splitlines()]
    result={'input':'requests.jsonl','input_prefix_bytes':len(complete),'input_prefix_sha256':hashlib.sha256(complete).hexdigest(),
        'input_sha256':hashlib.sha256(raw).hexdigest(),'input_complete_at_read':len(raw)==len(complete),
        'scope':'complete service records; unfinished and unobserved links remain unvalidated',
        'physical_workload_files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*/physical-workload.json')},
        'reset_classification':resets(rows,{p.parent.name:json.loads(p.read_text()) for p in root.glob('*/physical-workload.json')}),
        'trace_coverage':coverage(rows,{p.parent.name:json.loads((p.parent/'offered-trace.json').read_text()) for p in root.glob('*/result.json')}),**audit(rows)}
    (root/'resident-history-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['trace_coverage']['valid']:raise ValueError('offered arrivals lack exactly one terminal service row; evidence retained')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sessions','physical_workload_files','reset_classification','trace_coverage')}))


if __name__=='__main__':main()
