"""Reduce bounded replay evidence without fitting simulator coefficients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pool_shed_campaign as q
from pool_shed_calibration import replay_seconds
from pool_shed_execution import initial_work
from pool_replay_measure import write


def requests(out):
    recovery = json.loads((out/'node-recovery.json').read_text())
    path = out/'requests.jsonl'
    raw = path.read_bytes()
    damaged = recovery['raw_pre_reboot'][path.name]
    assert hashlib.sha256(raw).hexdigest() == damaged['sha256']
    return [json.loads(line) for line in raw[:damaged['valid_prefix_bytes']].splitlines()]


def csv_rows(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def unloaded(out, raw, calibration):
    phases = {(path.stem, phase['phase']): phase
              for path in out.glob('unloaded*/unloaded-*.json')
              for phase in json.loads(path.read_text())['phases']}
    rows = []
    for episode in sorted({r['episode'] for r in raw if 'retained_tokens' in r}):
        trial = next(r for r in raw if r.get('episode') == episode)
        context, append, width = (trial[k] for k in ('retained_tokens','append_tokens','width'))
        for phase in ('initial','catch_up','cold_updated'):
            selected = [r for r in raw if r.get('episode') == episode and r['phase'] == phase]
            done = [r for r in selected if r['status'] == 'complete']
            valid = [r for r in done if r['messages'][-1]['content'].split('code ')[-1].rstrip('.') in r['response_text']]
            target = context if phase == 'initial' else context+append
            fleet = SimpleNamespace(context=np.array([target]), t1=replay_seconds([target],calibration), log=np.array([2*target]),
                metadata={'packing_context_tokens':calibration['packing_context_tokens'],
                          'batch_context_limit':calibration['batch_context_limit']})
            _, predicted = initial_work(fleet,np.array([width]),0,0,None,
                {**calibration['timing'][0],'regional_replay_factor':[1.,1.]},calibration)
            elapsed = (max(r['end_ns'] for r in selected)-min(r.get('start_ns',r['dispatch_ns']) for r in selected))/1e9 if selected else None
            row = {'episode':episode,'seed':trial['seed'],'context':context,'append':append,'width':width,'phase':phase,
                'requests':len(selected),'http_complete':len(done),'state_valid':len(valid),
                'phase_valid':len(valid)==width,'missing_requests':width-len(selected),
                'exact_requests':sum(r['exact_token_timestamps'] for r in done),
                'elapsed_s':elapsed,'model_full_rebuild_local_s':predicted,
                'model_minus_observed_s':predicted-elapsed if elapsed is not None else None,
                'optimistic_error_s':max(0,elapsed-predicted) if elapsed is not None else None,
                'model_predicts_30s_but_observed_exceeds':elapsed is not None and predicted<=30<elapsed,
                'reported_cached_tokens':json.dumps([r.get('cached_tokens') for r in selected]),
                'prompt_counts':json.dumps([r['prompt_tokens'] for r in selected]),
                'output_counts':json.dumps([r.get('output_tokens') for r in selected])}
            evidence = phases.get((episode,phase))
            for metric in ('prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_hits_total',
                           'request_prefill_kv_computed_tokens_sum','request_queue_time_seconds_sum',
                           'request_inference_time_seconds_sum','request_prefill_time_seconds_sum','request_decode_time_seconds_sum'):
                key = 'vllm:'+metric
                row[metric] = (evidence['engine_after'][key]-evidence['engine_before'][key]
                    if evidence and key in evidence['engine_after'] and key in evidence['engine_before'] else None)
            queries, hits = row['prefix_cache_queries_total'], row['prefix_cache_hits_total']
            row['cache_state'] = ('cold_verified' if queries == sum(r['prompt_tokens'] for r in selected) and hits == 0
                else 'native_reuse_verified' if hits is not None and hits>0 else 'unknown')
            rows.append(row)
    csv_rows(out/'unloaded-observations.csv',rows)
    heldout = []
    for row in rows:
        if row['seed'] != 7102: continue
        prior = next(r for r in rows if r['seed']==7101 and all(r[k]==row[k] for k in ('context','append','width','phase')))
        usable = prior['phase_valid'] and row['phase_valid']
        heldout.append({k:row[k] for k in ('context','append','width','phase')} | {
            'training_phase_valid':prior['phase_valid'],'validation_phase_valid':row['phase_valid'],
            'training_observed_s':prior['elapsed_s'],'validation_observed_s':row['elapsed_s'],
            'repeat_error_s':prior['elapsed_s']-row['elapsed_s'] if usable else None,
            'repeat_relative_error':prior['elapsed_s']/row['elapsed_s']-1 if usable else None,
            'baseline_model_error_s':row['model_minus_observed_s'],
            'optimistic_error_s':row['optimistic_error_s']})
    csv_rows(out/'unloaded-heldout.csv',heldout)
    write(out/'unloaded-analysis.json',{'rows':rows,'heldout':heldout,
        'fitting':'No fitted correction. First-repeat lookup versus independent second repeat is a repeatability diagnostic only.',
        'prediction_scope':'Existing full-context work and packing, one destination GPU, regional factor explicitly 1; no resident load. Warm phases compare with the existing full-rebuild catch-up assumption.',
        'timing_scope':'Elapsed client request interval includes validation/decode; server queue/prefill/decode sums are engine histogram deltas, not additive GPU work or per-request attribution.',
        'missing_phase_evidence':'Trial 08 seed 7101 aborted before its cold-updated batch; raw initial/catch-up requests survive but phase counter snapshots do not.',
        'state_validation':'HTTP completion and exact token timing do not establish a valid response; expected state code is checked independently.',
        'wire_scope':'No paired KV transfer measured. 800000000 decimal effective-wire bytes per32768 tokens remains an assumption, separate from 1610612736 native serialized bytes and resident KV footprint.'})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows=unloaded(args.out,requests(args.out),q.calibration(0))
    print('Unloaded phase observations:',len(rows),'valid:',sum(r['phase_valid'] for r in rows))


if __name__=='__main__':main()
